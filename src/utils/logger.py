"""Structured stdout logger."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_INITIALIZED: set[str] = set()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _INITIALIZED:
        return logger

    effective = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logger.setLevel(effective)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
    logger.addHandler(handler)
    logger.propagate = False
    _INITIALIZED.add(name)
    return logger
