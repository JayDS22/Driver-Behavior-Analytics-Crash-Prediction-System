"""Generate synthetic training datasets matching the processed-input schema.

The upstream Bridgestone datasets are proprietary; this script produces
schema-compatible synthetic data so the training pipeline runs end-to-end.
Pass ``--source`` once real-data access is configured.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.data_loader import make_synthetic_crash_records, make_synthetic_features
from src.utils.logger import get_logger

logger = get_logger("scripts.download_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or synthesise project data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--n-features-samples", type=int, default=5000)
    parser.add_argument("--n-crash-samples", type=int, default=5000)
    parser.add_argument("--source", default=None, help="Remote URL or local archive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.source:
        logger.warning("--source not implemented; emitting synthetic data")

    X, y = make_synthetic_features(n_samples=args.n_features_samples, n_features=32)
    X.assign(crash_risk=y).to_csv(out / "features.csv", index=False)
    logger.info("wrote %s", out / "features.csv")

    crash = make_synthetic_crash_records(n_samples=args.n_crash_samples)
    crash.to_csv(out / "crash_data.csv", index=False)
    logger.info("wrote %s", out / "crash_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
