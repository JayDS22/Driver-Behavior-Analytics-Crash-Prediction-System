"""YOLOv7 vehicle detector.

Load order: local checkpoint -> torch.hub -> hash-based stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CLASSES = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
)


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


@dataclass
class VehicleDetector:
    model_path: Optional[str] = None
    hub_repo: str = "WongKinYiu/yolov7"
    hub_model: str = "yolov7"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_size: Tuple[int, int] = (640, 640)
    classes: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_CLASSES))
    device: str = "auto"
    _backend: str = field(default="stub", init=False)
    _model: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._load()

    @property
    def backend(self) -> str:
        return self._backend

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            logger.warning("torch unavailable; using stub detector")
            self._backend = "stub"
            return

        device = self._resolve_device()

        if self.model_path and Path(self.model_path).is_file():
            try:
                model = torch.load(self.model_path, map_location=device)
                if hasattr(model, "eval"):
                    model.eval()
                self._model = model
                self._backend = "local"
                logger.info("Loaded local YOLO weights from %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Local weights load failed (%s); trying torch.hub", exc)

        try:
            model = torch.hub.load(self.hub_repo, self.hub_model, trust_repo=True)
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            self._model = model
            self._backend = "torch_hub"
            logger.info("Loaded YOLO from torch.hub: %s/%s", self.hub_repo, self.hub_model)
            return
        except Exception as exc:
            logger.warning("torch.hub load failed (%s); using stub detector", exc)
            self._backend = "stub"

    def predict(self, image: np.ndarray) -> List[Detection]:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        if self._backend in {"local", "torch_hub"}:
            return self._predict_torch(image)
        return self._predict_stub(image)

    def predict_batch(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        return [self.predict(img) for img in images]

    def _predict_torch(self, image: np.ndarray) -> List[Detection]:
        results = self._model(image)
        try:
            tensor = results.xyxy[0].detach().cpu().numpy()
        except AttributeError:
            try:
                tensor = np.asarray(results)
            except Exception:
                return []
        detections: List[Detection] = []
        for row in tensor:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, cls = row[:6]
            if score < self.confidence_threshold:
                continue
            cls_id = int(cls)
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                    class_id=cls_id,
                    class_name=self._class_name(cls_id),
                )
            )
        return detections

    def _predict_stub(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        digest = int(np.abs(image.astype(np.int64).sum()) % 7) + 1
        rng = np.random.default_rng(int(image.astype(np.int64).sum() & 0xFFFFFFFF))
        detections: List[Detection] = []
        for _ in range(digest):
            cx = rng.uniform(0.2, 0.8) * w
            cy = rng.uniform(0.2, 0.8) * h
            bw = rng.uniform(0.05, 0.25) * w
            bh = rng.uniform(0.05, 0.25) * h
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)
            score = float(np.clip(rng.normal(0.78, 0.07), self.confidence_threshold, 0.99))
            cls_id = int(rng.integers(0, len(self.classes)))
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=cls_id,
                    class_name=self._class_name(cls_id),
                )
            )
        return detections

    def _class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.classes):
            return self.classes[cls_id]
        return f"class_{cls_id}"
