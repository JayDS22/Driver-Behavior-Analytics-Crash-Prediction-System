"""FastAPI endpoint tests."""
from __future__ import annotations

import base64

import numpy as np
import pytest

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

from src.api.inference_api import SafetySystem, build_app
from src.data.preprocessor import FramePreprocessor
from src.models.feature_extractor import FeatureExtractor
from src.models.yolo_detector import VehicleDetector

fastapi = pytest.importorskip("fastapi")
try:
    from fastapi import testclient  # type: ignore  # noqa: F401
    from fastapi.testclient import TestClient
except Exception as exc:
    pytest.skip(
        f"fastapi.testclient unavailable ({exc})",
        allow_module_level=True,
    )


@pytest.fixture
def safety_system() -> SafetySystem:
    return SafetySystem(
        detector=VehicleDetector(model_path=None),
        feature_extractor=FeatureExtractor(pretrained=False),
        preprocessor=FramePreprocessor(),
    )


@pytest.fixture
def client(safety_system):
    app = build_app(system=safety_system)
    return TestClient(app)


def _encode_frame(frame: np.ndarray) -> str:
    if cv2 is not None:
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        return base64.b64encode(buf.tobytes()).decode("ascii")
    from io import BytesIO

    from PIL import Image  # type: ignore

    buf = BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "detector_backend" in body


def test_predict_endpoint(client, synthetic_frame):
    encoded = _encode_frame(synthetic_frame)
    response = client.post(
        "/predict",
        json={"image": encoded, "metadata": {"vehicle_age": 4}},
    )
    assert response.status_code == 200
    body = response.json()
    preds = body["predictions"]
    assert 0.0 <= preds["overall_risk"] <= 1.0
    assert isinstance(preds["vehicles"], list)
    assert "crash_probability" in preds
    assert preds["processing_time_ms"] >= 0


def test_predict_rejects_bad_image(client):
    response = client.post("/predict", json={"image": "***not base64***"})
    assert response.status_code in {400, 422}


def test_metrics_after_predict(client, synthetic_frame):
    encoded = _encode_frame(synthetic_frame)
    client.post("/predict", json={"image": encoded})
    metrics = client.get("/metrics").json()
    assert metrics["count"] >= 1
    assert metrics["latency_ms_mean"] >= 0
