"""End-to-end: train ensemble + Cox, run frame inference."""
from __future__ import annotations

import numpy as np

from src.api.inference_api import SafetySystem
from src.data.data_loader import make_synthetic_crash_records, make_synthetic_features
from src.data.preprocessor import FramePreprocessor
from src.models.ensemble_model import EnsembleRiskModel
from src.models.feature_extractor import FeatureExtractor
from src.models.survival_analysis import CoxSurvivalModel
from src.models.yolo_detector import VehicleDetector


def test_end_to_end_pipeline():
    X, y = make_synthetic_features(n_samples=300, n_features=32)
    ensemble = EnsembleRiskModel().fit(X, y)

    crash_df = make_synthetic_crash_records(n_samples=400)
    survival = CoxSurvivalModel(time_horizons=[1, 3, 6, 12]).fit(
        crash_df,
        feature_cols=["vehicle_age", "annual_mileage", "driver_age", "prior_claims"],
    )

    system = SafetySystem(
        detector=VehicleDetector(),
        feature_extractor=FeatureExtractor(pretrained=False),
        ensemble=ensemble,
        survival=survival,
        preprocessor=FramePreprocessor(),
    )

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    metadata = {
        "vehicle_age": 5,
        "annual_mileage": 18000,
        "driver_age": 42,
        "prior_claims": 1,
    }
    result = system.process_frame(frame, metadata=metadata)
    assert 0.0 <= result["overall_risk"] <= 1.0
    for key, value in result["crash_probability"].items():
        assert key.endswith("_months")
        assert 0.0 <= value <= 1.0
    assert result["processing_time_ms"] >= 0

    metrics = system.monitor.snapshot()
    assert metrics["count"] >= 1
