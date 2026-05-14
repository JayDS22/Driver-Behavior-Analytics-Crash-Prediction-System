"""Tests for detector, feature extractor, ensemble, and Cox model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data_loader import make_synthetic_crash_records, make_synthetic_features
from src.models.ensemble_model import EnsembleRiskModel, evaluate, split_train_test
from src.models.feature_extractor import FeatureExtractor
from src.models.survival_analysis import CoxSurvivalModel
from src.models.yolo_detector import Detection, VehicleDetector


def test_detector_returns_detections(synthetic_frame):
    detector = VehicleDetector(model_path=None)
    detections = detector.predict(synthetic_frame)
    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)
    if detections:
        d = detections[0]
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert x2 > x1 and y2 > y1


def test_detector_deterministic(synthetic_frame):
    d1 = VehicleDetector().predict(synthetic_frame)
    d2 = VehicleDetector().predict(synthetic_frame)
    assert len(d1) == len(d2)


def test_feature_extractor_shape(synthetic_frame):
    fe = FeatureExtractor(pretrained=False)
    features = fe.extract(synthetic_frame)
    assert features.ndim == 1
    assert features.shape[0] > 0


def test_feature_extractor_motion_and_edges(synthetic_frames):
    fe = FeatureExtractor(pretrained=False)
    motion = fe.extract_motion(synthetic_frames[0], synthetic_frames[1])
    edges = fe.extract_edges(synthetic_frames[0])
    assert motion.shape == (3,)
    assert edges.shape == (2,)


def test_ensemble_fit_predict_save_load(tmp_path):
    X, y = make_synthetic_features(n_samples=400, n_features=16)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.25)
    model = EnsembleRiskModel()
    model.fit(X_train, y_train)
    report = evaluate(model, X_test, y_test)

    assert 0.0 <= report["auc"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["auc"] > 0.6

    path = tmp_path / "ensemble.pkl"
    model.save(path)
    loaded = EnsembleRiskModel.load(path)
    np.testing.assert_allclose(
        loaded.risk_score(X_test), model.risk_score(X_test), rtol=1e-5
    )


def test_survival_model_trains(tmp_path):
    df = make_synthetic_crash_records(n_samples=600)
    model = CoxSurvivalModel(penalizer=0.05, time_horizons=[1, 3, 6, 12])
    model.fit(df, feature_cols=["vehicle_age", "annual_mileage", "driver_age", "prior_claims"])
    c_index = model.concordance_index(df)
    assert 0.5 <= c_index <= 1.0
    survival = model.predict_survival(df.head(5))
    assert isinstance(survival, pd.DataFrame)
    assert survival.shape[0] == 5
    crash = model.crash_probability(df.head(5))
    assert (crash.values >= 0).all() and (crash.values <= 1).all()

    save_path = tmp_path / "cox.pkl"
    model.save(save_path)
    loaded = CoxSurvivalModel.load(save_path)
    assert loaded.feature_cols == model.feature_cols
