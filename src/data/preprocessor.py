"""Letterbox + normalize for the YOLO and feature-extractor inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore


@dataclass
class FramePreprocessor:
    input_size: Tuple[int, int] = (640, 640)
    mean: Sequence[float] = (0.0, 0.0, 0.0)
    std: Sequence[float] = (1.0, 1.0, 1.0)
    to_rgb: bool = True

    def letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if cv2 is not None:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = _resize_nearest(image, new_h, new_w)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas = np.full((target_h, target_w, image.shape[2]), 114, dtype=image.dtype)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas, scale, (pad_top, pad_left)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")
        canvas, _, _ = self.letterbox(image)
        if self.to_rgb and cv2 is not None:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        arr = canvas.astype(np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        return np.transpose(arr, (2, 0, 1))


def _resize_nearest(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    return image[row_idx][:, col_idx]
