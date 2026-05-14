"""Bounding-box overlay helpers."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore


def _color_for_class(cls_idx: int) -> Tuple[int, int, int]:
    palette = [
        (0, 200, 0),
        (0, 0, 255),
        (255, 165, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]
    return palette[cls_idx % len(palette)]


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[dict],
    label_field: str = "class_name",
    score_field: str = "confidence",
    risk_field: Optional[str] = "risk_score",
) -> np.ndarray:
    if cv2 is None:
        return frame
    out = frame.copy()
    for det in detections:
        bbox: Sequence[float] = det["bbox"]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = _color_for_class(int(det.get("class_id", 0)))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get(label_field, 'obj')} {det.get(score_field, 0.0):.2f}"
        if risk_field and risk_field in det:
            label += f" r={det[risk_field]:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
