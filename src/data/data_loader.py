"""Dataset loaders for video frames, tabular features, and crash records."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VehicleFrameDataset:
    source: str
    max_frames: Optional[int] = None
    stride: int = 1

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        path = Path(self.source)
        if path.is_dir():
            yield from self._iter_directory(path)
        elif path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            yield from self._iter_video(path)
        elif path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = _read_image(path)
            if img is not None:
                yield 0, img
        else:
            raise FileNotFoundError(f"Unsupported or missing source: {self.source}")

    def _iter_directory(self, path: Path) -> Iterator[Tuple[int, np.ndarray]]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
        count = 0
        for i, file in enumerate(files):
            if i % self.stride != 0:
                continue
            img = _read_image(file)
            if img is None:
                continue
            yield count, img
            count += 1
            if self.max_frames and count >= self.max_frames:
                break

    def _iter_video(self, path: Path) -> Iterator[Tuple[int, np.ndarray]]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV required for video decode") from exc

        cap = cv2.VideoCapture(str(path))
        try:
            frame_idx = 0
            kept = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.stride == 0:
                    yield kept, frame
                    kept += 1
                    if self.max_frames and kept >= self.max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()


def _read_image(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore

        return cv2.imread(str(path))
    except ImportError:
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as im:
                return np.array(im.convert("RGB"))[:, :, ::-1]
        except Exception:
            return None


def load_features_csv(
    path: str | Path,
    target_col: str = "crash_risk",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not in {df.columns.tolist()}")
    y = df[target_col]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    return X, y


@dataclass
class CrashRecordDataset:
    path: str
    duration_col: str = "time_to_event"
    event_col: str = "event"

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        for col in (self.duration_col, self.event_col):
            if col not in df.columns:
                raise KeyError(f"required column '{col}' missing from {self.path}")
        return df


def make_synthetic_features(
    n_samples: int = 1000, n_features: int = 32, seed: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    logits = X @ weights + rng.normal(scale=0.5, size=n_samples)
    y = (logits > np.median(logits)).astype(int)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="crash_risk")


def make_synthetic_crash_records(n_samples: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vehicle_age = rng.integers(0, 15, size=n_samples)
    annual_mileage = rng.normal(15_000, 4_000, size=n_samples)
    driver_age = rng.integers(18, 75, size=n_samples)
    prior_claims = rng.integers(0, 5, size=n_samples)
    hazard = (
        0.01
        + 0.002 * vehicle_age
        + 0.00001 * annual_mileage
        + 0.05 * prior_claims
    )
    time_to_event = rng.exponential(scale=1.0 / np.maximum(hazard, 1e-3))
    censor_time = rng.uniform(0, 36, size=n_samples)
    observed = (time_to_event <= censor_time).astype(int)
    duration = np.minimum(time_to_event, censor_time)
    return pd.DataFrame(
        {
            "vehicle_age": vehicle_age,
            "annual_mileage": annual_mileage,
            "driver_age": driver_age,
            "prior_claims": prior_claims,
            "time_to_event": duration,
            "event": observed,
        }
    )
