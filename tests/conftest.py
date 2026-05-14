"""Shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_frames() -> list:
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8) for _ in range(4)]
