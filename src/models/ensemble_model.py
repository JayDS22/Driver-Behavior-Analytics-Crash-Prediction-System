"""Soft-voting ensemble: Random Forest + XGBoost + MLP."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    rf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 4,
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )
    mlp_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "random_state": 42,
        }
    )
    weights: Sequence[float] = (0.3, 0.4, 0.3)


class EnsembleRiskModel:
    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(**self.config.rf_params)
        self.mlp = MLPClassifier(**self.config.mlp_params)
        self.xgb = self._build_xgb()
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False

    def _build_xgb(self):
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(**self.config.xgb_params)
        except ImportError:
            logger.warning("xgboost missing; substituting GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(random_state=42)

    def fit(self, X, y) -> "EnsembleRiskModel":
        X_df = self._to_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        X_scaled = self.scaler.fit_transform(X_df.values)
        self.rf.fit(X_scaled, y)
        self.xgb.fit(X_scaled, y)
        self.mlp.fit(X_scaled, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X_df = self._to_dataframe(X)
        X_scaled = self.scaler.transform(X_df.values)
        probs = [
            self.rf.predict_proba(X_scaled),
            self.xgb.predict_proba(X_scaled),
            self.mlp.predict_proba(X_scaled),
        ]
        weights = np.asarray(self.config.weights, dtype=np.float32)
        weights = weights / weights.sum()
        stacked = np.stack(probs, axis=0)
        return (stacked * weights.reshape(-1, 1, 1)).sum(axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def risk_score(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "scaler": self.scaler,
                "rf": self.rf,
                "xgb": self.xgb,
                "mlp": self.mlp,
                "feature_names": self.feature_names_,
                "is_fitted": self.is_fitted_,
            },
            path,
        )
        logger.info("Saved ensemble model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleRiskModel":
        payload = joblib.load(path)
        model = cls(config=payload["config"])
        model.scaler = payload["scaler"]
        model.rf = payload["rf"]
        model.xgb = payload["xgb"]
        model.mlp = payload["mlp"]
        model.feature_names_ = payload.get("feature_names")
        model.is_fitted_ = payload.get("is_fitted", True)
        return model

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("EnsembleRiskModel must be fitted before prediction")

    def _to_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                missing = [c for c in self.feature_names_ if c not in X.columns]
                if missing:
                    raise KeyError(f"missing feature columns: {missing}")
                return X[self.feature_names_]
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = self.feature_names_ or [f"f{i}" for i in range(arr.shape[1])]
        if len(cols) != arr.shape[1]:
            cols = [f"f{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)


def evaluate(model: EnsembleRiskModel, X, y) -> Dict[str, float]:
    from src.utils.metrics import compute_auc, compute_classification_report

    proba = model.risk_score(X)
    preds = (proba >= 0.5).astype(int)
    report = compute_classification_report(y, preds)
    report["auc"] = compute_auc(y, proba)
    return report


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
