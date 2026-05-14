"""Cox proportional-hazards survival model.

Uses ``lifelines.CoxPHFitter`` when installed; otherwise solves the
partial-likelihood with Newton-Raphson + ridge penalty and a Breslow
baseline hazard estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CoxSurvivalModel:
    duration_col: str = "time_to_event"
    event_col: str = "event"
    penalizer: float = 0.01
    time_horizons: List[float] = field(default_factory=lambda: [1, 3, 6, 12])
    feature_cols: Optional[List[str]] = None
    _model: object | None = field(default=None, init=False)
    _backend: str = field(default="lifelines", init=False)
    _coefs: Optional[np.ndarray] = field(default=None, init=False)
    _baseline_times: Optional[np.ndarray] = field(default=None, init=False)
    _baseline_hazard: Optional[np.ndarray] = field(default=None, init=False)
    _means: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "CoxSurvivalModel":
        if feature_cols is None:
            feature_cols = [
                c for c in df.columns if c not in (self.duration_col, self.event_col)
            ]
        self.feature_cols = list(feature_cols)
        try:
            from lifelines import CoxPHFitter  # type: ignore

            cph = CoxPHFitter(penalizer=self.penalizer)
            cph.fit(
                df[self.feature_cols + [self.duration_col, self.event_col]],
                duration_col=self.duration_col,
                event_col=self.event_col,
            )
            self._model = cph
            self._backend = "lifelines"
            logger.info("Cox fit via lifelines on %d records", len(df))
        except ImportError:
            logger.warning("lifelines missing; using internal Cox solver")
            self._fit_internal(df)
        return self

    def _fit_internal(self, df: pd.DataFrame) -> None:
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        durations = df[self.duration_col].to_numpy(dtype=np.float64)
        events = df[self.event_col].to_numpy(dtype=np.int64)
        means = X.mean(axis=0)
        Xc = X - means
        beta = np.zeros(Xc.shape[1])
        order = np.argsort(-durations)
        Xc_sorted = Xc[order]
        events_sorted = events[order]
        for _ in range(50):
            risk = np.exp(Xc_sorted @ beta)
            cum_risk = np.cumsum(risk)
            cum_risk_x = np.cumsum(Xc_sorted * risk[:, None], axis=0)
            cum_risk_xx = np.cumsum(
                (Xc_sorted[:, :, None] * Xc_sorted[:, None, :]) * risk[:, None, None],
                axis=0,
            )
            mean_x = cum_risk_x / np.maximum(cum_risk, 1e-12)[:, None]
            mean_xx = cum_risk_xx / np.maximum(cum_risk, 1e-12)[:, None, None]
            grad = ((Xc_sorted - mean_x) * events_sorted[:, None]).sum(axis=0)
            hess = -((mean_xx - mean_x[:, :, None] * mean_x[:, None, :]) * events_sorted[:, None, None]).sum(axis=0)
            grad -= self.penalizer * beta
            hess -= self.penalizer * np.eye(Xc.shape[1])
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            beta -= step
            if np.linalg.norm(step) < 1e-6:
                break
        self._coefs = beta
        self._means = means
        risk_all = np.exp(Xc @ beta)
        sort_idx = np.argsort(durations)
        sorted_times = durations[sort_idx]
        sorted_events = events[sort_idx]
        sorted_risk = risk_all[sort_idx]
        cum_baseline = np.cumsum(
            sorted_events / np.maximum(np.flip(np.cumsum(np.flip(sorted_risk))), 1e-12)
        )
        self._baseline_times = sorted_times
        self._baseline_hazard = cum_baseline
        self._backend = "internal"

    def predict_survival(
        self,
        df: pd.DataFrame,
        times: Optional[Iterable[float]] = None,
    ) -> pd.DataFrame:
        if self.feature_cols is None:
            raise RuntimeError("model must be fit before predicting")
        times = list(times) if times is not None else list(self.time_horizons)
        if self._backend == "lifelines":
            survival = self._model.predict_survival_function(df[self.feature_cols], times=times)  # type: ignore[union-attr]
            survival = survival.T
            survival.columns = [f"S(t={t})" for t in times]
            survival.index = df.index
            return survival
        return self._predict_survival_internal(df, times)

    def crash_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        survival = self.predict_survival(df, self.time_horizons)
        crash = 1.0 - survival.values
        return pd.DataFrame(
            crash,
            index=survival.index,
            columns=[f"P(crash<={t}m)" for t in self.time_horizons],
        )

    def concordance_index(self, df: pd.DataFrame) -> float:
        if self._backend == "lifelines":
            try:
                return float(self._model.concordance_index_)  # type: ignore[union-attr]
            except AttributeError:
                from lifelines.utils import concordance_index  # type: ignore

                partial = self._model.predict_partial_hazard(df[self.feature_cols])  # type: ignore[union-attr]
                return float(
                    concordance_index(df[self.duration_col], -partial, df[self.event_col])
                )
        return self._concordance_internal(df)

    def _predict_survival_internal(
        self, df: pd.DataFrame, times: List[float]
    ) -> pd.DataFrame:
        X = df[self.feature_cols].to_numpy(dtype=np.float64) - self._means
        partial = np.exp(X @ self._coefs)
        cols = {}
        for t in times:
            idx = np.searchsorted(self._baseline_times, t, side="right") - 1
            idx = max(0, idx)
            h0 = float(self._baseline_hazard[idx])
            cols[f"S(t={t})"] = np.exp(-h0 * partial)
        return pd.DataFrame(cols, index=df.index)

    def _concordance_internal(self, df: pd.DataFrame) -> float:
        X = df[self.feature_cols].to_numpy(dtype=np.float64) - self._means
        scores = X @ self._coefs
        durations = df[self.duration_col].to_numpy()
        events = df[self.event_col].to_numpy()
        n = len(durations)
        if n > 5000:
            rng = np.random.default_rng(0)
            sample = rng.choice(n, size=5000, replace=False)
            durations, events, scores = durations[sample], events[sample], scores[sample]
        concordant = permissible = 0
        for i in range(len(durations)):
            for j in range(i + 1, len(durations)):
                if events[i] == 0 and events[j] == 0:
                    continue
                if durations[i] == durations[j]:
                    continue
                if durations[i] < durations[j] and events[i] == 0:
                    continue
                if durations[j] < durations[i] and events[j] == 0:
                    continue
                permissible += 1
                if durations[i] < durations[j] and scores[i] > scores[j]:
                    concordant += 1
                elif durations[j] < durations[i] and scores[j] > scores[i]:
                    concordant += 1
                elif scores[i] == scores[j]:
                    concordant += 0.5
        return concordant / permissible if permissible else 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "duration_col": self.duration_col,
                "event_col": self.event_col,
                "penalizer": self.penalizer,
                "time_horizons": self.time_horizons,
                "feature_cols": self.feature_cols,
                "backend": self._backend,
                "model": self._model,
                "coefs": self._coefs,
                "means": self._means,
                "baseline_times": self._baseline_times,
                "baseline_hazard": self._baseline_hazard,
            },
            path,
        )
        logger.info("Saved Cox model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CoxSurvivalModel":
        payload = joblib.load(path)
        model = cls(
            duration_col=payload["duration_col"],
            event_col=payload["event_col"],
            penalizer=payload["penalizer"],
            time_horizons=payload["time_horizons"],
            feature_cols=payload["feature_cols"],
        )
        model._backend = payload["backend"]
        model._model = payload["model"]
        model._coefs = payload["coefs"]
        model._means = payload["means"]
        model._baseline_times = payload["baseline_times"]
        model._baseline_hazard = payload["baseline_hazard"]
        return model
