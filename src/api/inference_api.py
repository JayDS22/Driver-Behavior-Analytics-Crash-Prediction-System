"""FastAPI inference service: detector + ensemble risk + Cox survival."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    from pydantic import BaseModel as _PydanticBase
except ImportError:
    _PydanticBase = object  # type: ignore[assignment, misc]

from src import __version__
from src.api.monitoring import InferenceMonitor
from src.data.preprocessor import FramePreprocessor
from src.models.ensemble_model import EnsembleRiskModel
from src.models.feature_extractor import FeatureExtractor
from src.models.survival_analysis import CoxSurvivalModel
from src.models.yolo_detector import VehicleDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("config/model_config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        logger.warning("Config %s not found; falling back to defaults", path)
        return {}
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class SafetySystem:
    detector: VehicleDetector
    feature_extractor: FeatureExtractor
    ensemble: Optional[EnsembleRiskModel] = None
    survival: Optional[CoxSurvivalModel] = None
    preprocessor: FramePreprocessor = field(default_factory=FramePreprocessor)
    monitor: InferenceMonitor = field(default_factory=InferenceMonitor)
    feature_dim: int = 32

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "SafetySystem":
        cfg = config or load_config()
        yolo_cfg = cfg.get("yolo", {})
        fe_cfg = cfg.get("feature_extractor", {})
        detector = VehicleDetector(
            model_path=yolo_cfg.get("model_path"),
            hub_repo=yolo_cfg.get("hub_repo", "WongKinYiu/yolov7"),
            hub_model=yolo_cfg.get("hub_model", "yolov7"),
            confidence_threshold=yolo_cfg.get("confidence_threshold", 0.5),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
            input_size=tuple(yolo_cfg.get("input_size", [640, 640])),
            classes=yolo_cfg.get(
                "classes",
                ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"],
            ),
        )
        feature_extractor = FeatureExtractor(
            backbone=fe_cfg.get("backbone", "resnet50"),
            pretrained=fe_cfg.get("pretrained", True),
        )
        ensemble = None
        survival = None
        ensemble_path = cfg.get("ensemble", {}).get("model_path")
        if ensemble_path and Path(ensemble_path).exists():
            ensemble = EnsembleRiskModel.load(ensemble_path)
        survival_path = cfg.get("survival", {}).get("model_path")
        if survival_path and Path(survival_path).exists():
            survival = CoxSurvivalModel.load(survival_path)
        preprocessor = FramePreprocessor(
            input_size=tuple(yolo_cfg.get("input_size", [640, 640]))
        )
        return cls(
            detector=detector,
            feature_extractor=feature_extractor,
            ensemble=ensemble,
            survival=survival,
            preprocessor=preprocessor,
        )

    def process_frame(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            detections = self.detector.predict(frame)
            features = self.feature_extractor.extract(frame)
            ensemble_risk = self._ensemble_risk(features)
            risk_scores = self._per_detection_risk(detections, ensemble_risk)
            crash_probs = self._crash_probability(metadata or {}, ensemble_risk)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.monitor.record(elapsed_ms)
            return {
                "vehicles": [
                    {**det.to_dict(), "risk_score": risk_scores[i]}
                    for i, det in enumerate(detections)
                ],
                "overall_risk": float(ensemble_risk),
                "crash_probability": crash_probs,
                "processing_time_ms": round(elapsed_ms, 2),
                "model_version": __version__,
                "detector_backend": self.detector.backend,
            }
        except Exception:
            self.monitor.record((time.perf_counter() - t0) * 1000.0, error=True)
            raise

    def process_video(self, source: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        from src.data.data_loader import VehicleFrameDataset

        dataset = VehicleFrameDataset(source=source, max_frames=max_frames)
        per_frame: List[Dict[str, Any]] = []
        risks: List[float] = []
        for idx, frame in dataset:
            result = self.process_frame(frame, metadata={"frame_index": idx})
            per_frame.append(result)
            risks.append(result["overall_risk"])
        return {
            "source": source,
            "frames_processed": len(per_frame),
            "risk_score": float(np.mean(risks)) if risks else 0.0,
            "max_frame_risk": float(np.max(risks)) if risks else 0.0,
            "per_frame": per_frame,
        }

    def _ensemble_risk(self, features: np.ndarray) -> float:
        if self.ensemble is None or not self.ensemble.is_fitted_:
            return float(1.0 / (1.0 + np.exp(-float(features.mean()))))
        expected = len(self.ensemble.feature_names_ or [])
        if expected and features.shape[0] != expected:
            if features.shape[0] > expected:
                features = features[:expected]
            else:
                features = np.pad(features, (0, expected - features.shape[0]))
        score = float(self.ensemble.risk_score(features.reshape(1, -1))[0])
        return score

    def _per_detection_risk(self, detections, base_risk: float) -> List[float]:
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            modifier = min(1.0, area / (640 * 640))
            risk = float(np.clip(base_risk * (0.7 + 0.6 * modifier) * det.confidence, 0.0, 1.0))
            scores.append(risk)
        return scores

    def _crash_probability(self, metadata: Dict[str, Any], base_risk: float) -> Dict[str, float]:
        horizons = [1, 3, 6, 12]
        if self.survival is None or self.survival.feature_cols is None:
            return {f"{t}_months": float(min(0.95, base_risk * (t / 12.0))) for t in horizons}
        feat_cols = self.survival.feature_cols
        row = {col: float(metadata.get(col, 0.0)) for col in feat_cols}
        import pandas as pd

        df = pd.DataFrame([row])
        probs = self.survival.crash_probability(df).iloc[0]
        return {f"{int(t)}_months": float(probs[f"P(crash<={t}m)"]) for t in self.survival.time_horizons}


class PredictRequest(_PydanticBase):  # type: ignore[misc, valid-type]
    image: str
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(_PydanticBase):  # type: ignore[misc, valid-type]
    status: str
    version: str
    detector_backend: str


def build_app(system: Optional[SafetySystem] = None):
    from fastapi import Body, FastAPI, HTTPException

    safety = system or SafetySystem.from_config()
    app = FastAPI(
        title="Driver Behavior Analytics & Crash Prediction API",
        version=__version__,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            detector_backend=safety.detector.backend,
        )

    @app.get("/metrics")
    def metrics() -> Dict[str, float]:
        return safety.monitor.snapshot()

    @app.post("/predict")
    def predict(req: PredictRequest = Body(...)) -> Dict[str, Any]:
        try:
            raw = base64.b64decode(req.image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
        frame = _decode_image(raw)
        if frame is None:
            raise HTTPException(status_code=400, detail="could not decode image bytes")
        result = safety.process_frame(frame, metadata=req.metadata or {})
        return {"predictions": result}

    return app


def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        try:
            import io

            from PIL import Image  # type: ignore

            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))[:, :, ::-1]
        except Exception:
            return None


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    app = build_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
