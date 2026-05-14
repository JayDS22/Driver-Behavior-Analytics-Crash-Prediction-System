"""Tests for AUC, classification report, and mAP."""
from __future__ import annotations

import numpy as np

from src.utils.metrics import compute_auc, compute_classification_report, mean_average_precision


def test_compute_auc_perfect():
    auc = compute_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert auc == 1.0


def test_compute_auc_single_class():
    auc = compute_auc([1, 1, 1], [0.5, 0.6, 0.7])
    assert auc == 0.5


def test_classification_report_fields():
    report = compute_classification_report([0, 1, 1, 0], [0, 1, 0, 0])
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in report
        assert 0.0 <= report[key] <= 1.0


def test_map_perfect_detector():
    preds = [[((10, 10, 50, 50), 0.99, 0)]]
    gts = [[((10, 10, 50, 50), 0)]]
    score = mean_average_precision(preds, gts, num_classes=1)
    assert score == 1.0


def test_map_zero_when_no_overlap():
    preds = [[((0, 0, 5, 5), 0.99, 0)]]
    gts = [[((100, 100, 150, 150), 0)]]
    score = mean_average_precision(preds, gts, num_classes=1)
    assert score == 0.0
