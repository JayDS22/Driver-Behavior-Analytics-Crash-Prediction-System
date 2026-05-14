"""Evaluation metrics used across model components."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_classification_report(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean_average_precision(
    predictions: Iterable[List[Tuple[Sequence[float], float, int]]],
    ground_truths: Iterable[List[Tuple[Sequence[float], int]]],
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> float:
    """VOC-style mAP@iou_threshold.

    predictions:    list of [(bbox, score, class), ...] per image
    ground_truths:  list of [(bbox, class), ...] per image
    """
    predictions = list(predictions)
    ground_truths = list(ground_truths)
    average_precisions: List[float] = []

    for cls in range(num_classes):
        preds: List[Tuple[int, Sequence[float], float]] = []
        gts: Dict[int, List[Sequence[float]]] = {}
        for img_id, (img_preds, img_gts) in enumerate(zip(predictions, ground_truths)):
            for bbox, score, c in img_preds:
                if c == cls:
                    preds.append((img_id, bbox, score))
            gts[img_id] = [b for b, c in img_gts if c == cls]

        if not preds:
            average_precisions.append(0.0)
            continue

        preds.sort(key=lambda x: x[2], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched: Dict[int, set] = {i: set() for i in gts}
        total_gts = sum(len(v) for v in gts.values())

        for i, (img_id, bbox, _score) in enumerate(preds):
            candidates = gts.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(candidates):
                if j in matched[img_id]:
                    continue
                iou = _iou(bbox, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp[i] = 1.0
                matched[img_id].add(best_idx)
            else:
                fp[i] = 1.0

        if total_gts == 0:
            average_precisions.append(0.0)
            continue

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / total_gts
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        rec = np.concatenate(([0.0], recalls, [1.0]))
        prec = np.concatenate(([1.0], precisions, [0.0]))
        for k in range(len(prec) - 1, 0, -1):
            prec[k - 1] = max(prec[k - 1], prec[k])
        idx = np.where(rec[1:] != rec[:-1])[0]
        ap = float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0
