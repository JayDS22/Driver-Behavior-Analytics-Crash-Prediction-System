"""CNN embeddings, optical-flow motion stats, and Canny edge features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureExtractor:
    backbone: str = "resnet50"
    pretrained: bool = True
    feature_dim: int = 2048
    device: str = "auto"
    _model: Optional[object] = None
    _backend: str = "stats"

    def __post_init__(self) -> None:
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore

            device = self._resolve_device()
            ctor = getattr(torchvision.models, self.backbone)
            weights_kw = {"weights": "IMAGENET1K_V2"} if self.pretrained else {"weights": None}
            try:
                model = ctor(**weights_kw)
            except TypeError:
                model = ctor(pretrained=self.pretrained)
            modules = list(model.children())[:-1]
            self._model = torch.nn.Sequential(*modules).to(device).eval()
            self._device = device
            self._backend = "cnn"
            logger.info("Initialised %s on %s", self.backbone, device)
        except Exception as exc:
            logger.warning("CNN backbone unavailable (%s); using statistical features", exc)
            self._backend = "stats"

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def backend(self) -> str:
        return self._backend

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 image")
        if self._backend == "cnn":
            return self._extract_cnn(image)
        return self._extract_stats(image)

    def extract_motion(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        if prev.shape != curr.shape:
            raise ValueError("frames must share shape")
        diff = curr.astype(np.float32) - prev.astype(np.float32)
        return np.array(
            [
                float(np.mean(np.abs(diff))),
                float(np.std(diff)),
                float(np.percentile(np.abs(diff), 95)),
            ],
            dtype=np.float32,
        )

    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2  # type: ignore

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            density = float(np.mean(edges > 0))
            mean_intensity = float(np.mean(edges))
        except Exception:
            gray = image.mean(axis=2)
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            mag = np.hypot(gx, gy)
            density = float(np.mean(mag > 30))
            mean_intensity = float(np.mean(mag))
        return np.array([density, mean_intensity], dtype=np.float32)

    def _extract_cnn(self, image: np.ndarray) -> np.ndarray:
        import torch  # type: ignore
        from torchvision import transforms  # type: ignore

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model(tensor).squeeze().cpu().numpy()
        return features.astype(np.float32).reshape(-1)

    def _extract_stats(self, image: np.ndarray) -> np.ndarray:
        arr = image.astype(np.float32) / 255.0
        chans = [arr[:, :, c] for c in range(arr.shape[2])]
        feats = []
        for ch in chans:
            feats.extend(
                [
                    float(ch.mean()),
                    float(ch.std()),
                    float(np.percentile(ch, 25)),
                    float(np.percentile(ch, 75)),
                    float(ch.min()),
                    float(ch.max()),
                ]
            )
        gx = np.diff(arr, axis=1, prepend=arr[:, :1, :]).mean()
        gy = np.diff(arr, axis=0, prepend=arr[:1, :, :]).mean()
        feats.extend([float(gx), float(gy)])
        feats = feats[:32] + [0.0] * max(0, 32 - len(feats))
        return np.asarray(feats, dtype=np.float32)
