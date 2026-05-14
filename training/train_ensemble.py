"""Train the soft-voting RF + XGB + MLP crash-risk classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import load_features_csv, make_synthetic_features
from src.models.ensemble_model import EnsembleConfig, EnsembleRiskModel, evaluate, split_train_test
from src.utils.logger import get_logger
from src.utils.metrics import compute_auc

logger = get_logger("training.train_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ensemble crash-risk classifier")
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--target", default="crash_risk")
    parser.add_argument("--output", default="data/models/ensemble_model.pkl")
    parser.add_argument("--cross-validation", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic training data",
    )
    return parser.parse_args()


def _load(args: argparse.Namespace):
    if args.synthetic or not Path(args.data).exists():
        if not args.synthetic:
            logger.warning("dataset %s missing; generating synthetic", args.data)
        return make_synthetic_features(n_samples=2000, n_features=32)
    return load_features_csv(args.data, target_col=args.target)


def main() -> int:
    args = parse_args()
    X, y = _load(args)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    cv_aucs = []
    if args.cross_validation > 1:
        skf = StratifiedKFold(
            n_splits=args.cross_validation, shuffle=True, random_state=args.random_state
        )
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_model = EnsembleRiskModel(EnsembleConfig())
            fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            proba = fold_model.risk_score(X_train.iloc[val_idx])
            auc = compute_auc(y_train.iloc[val_idx], proba)
            cv_aucs.append(auc)
            logger.info("Fold %d AUC=%.4f", fold + 1, auc)

    model = EnsembleRiskModel(EnsembleConfig())
    model.fit(X_train, y_train)
    report = evaluate(model, X_test, y_test)

    output = Path(args.output)
    model.save(output)
    metrics = {
        "test_auc": report["auc"],
        "test_accuracy": report["accuracy"],
        "test_precision": report["precision"],
        "test_recall": report["recall"],
        "test_f1": report["f1"],
        "cv_auc_mean": float(np.mean(cv_aucs)) if cv_aucs else None,
        "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else None,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("metrics -> %s: %s", metrics_path, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
