"""Simple image augmentation pipeline for training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    noise_std: float = 0.0


def _hflip(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1, :].copy()


def _brightness(image: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) + delta * 255.0, 0, 255).astype(image.dtype)


def _contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = image.astype(np.float32).mean()
    return np.clip(
        (image.astype(np.float32) - mean) * factor + mean, 0, 255
    ).astype(image.dtype)


def _noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, std * 255.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(image.dtype)


def build_augmentation_pipeline(
    config: AugmentationConfig | None = None,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    cfg = config or AugmentationConfig()
    rng = np.random.default_rng(seed)

    def pipeline(image: np.ndarray) -> np.ndarray:
        out = image
        if rng.random() < cfg.horizontal_flip_prob:
            out = _hflip(out)
        if cfg.brightness_jitter > 0:
            out = _brightness(out, float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)))
        if cfg.contrast_jitter > 0:
            factor = 1.0 + float(rng.uniform(-cfg.contrast_jitter, cfg.contrast_jitter))
            out = _contrast(out, factor)
        if cfg.noise_std > 0:
            out = _noise(out, cfg.noise_std, rng)
        return out

    return pipeline


def chain(transforms: List[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    def runner(image: np.ndarray) -> np.ndarray:
        for t in transforms:
            image = t(image)
        return image

    return runner
