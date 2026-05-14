"""Train the Cox proportional-hazards crash-time model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.data_loader import CrashRecordDataset, make_synthetic_crash_records
from src.models.survival_analysis import CoxSurvivalModel
from src.utils.logger import get_logger

logger = get_logger("training.train_survival")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cox crash survival model")
    parser.add_argument("--data", default="data/processed/crash_data.csv")
    parser.add_argument("--features", default="config/survival_features.yaml")
    parser.add_argument("--output", default="data/models/cox_model.pkl")
    parser.add_argument("--penalizer", type=float, default=0.01)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    feat_cfg = {}
    if Path(args.features).exists():
        feat_cfg = yaml.safe_load(Path(args.features).read_text()) or {}
    duration_col = feat_cfg.get("duration_col", "time_to_event")
    event_col = feat_cfg.get("event_col", "event")
    feature_cols = (feat_cfg.get("features", {}) or {}).get("numeric")

    if args.synthetic or not Path(args.data).exists():
        if not args.synthetic:
            logger.warning("dataset %s missing; generating synthetic", args.data)
        df = make_synthetic_crash_records(n_samples=2000)
        feature_cols = [c for c in df.columns if c not in (duration_col, event_col)]
    else:
        df = CrashRecordDataset(args.data, duration_col=duration_col, event_col=event_col).load()

    model = CoxSurvivalModel(
        duration_col=duration_col,
        event_col=event_col,
        penalizer=args.penalizer,
    )
    model.fit(df, feature_cols=feature_cols)
    c_index = model.concordance_index(df)
    model.save(args.output)

    metrics = {
        "c_index": c_index,
        "backend": model._backend,
        "n_records": int(len(df)),
        "feature_cols": model.feature_cols,
    }
    Path(args.output).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Cox fit: C-index=%.4f backend=%s", c_index, model._backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
