"""Bootstrap dependencies, model weights, and synthetic data."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"[setup_environment] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up the project environment")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--generate-synthetic", action="store_true")
    return parser.parse_args()


def generate_synthetic_data() -> None:
    from src.data.data_loader import make_synthetic_crash_records, make_synthetic_features

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    X, y = make_synthetic_features(n_samples=5_000, n_features=32)
    X.assign(crash_risk=y).to_csv(processed / "features.csv", index=False)
    print(f"[setup_environment] wrote {processed / 'features.csv'}")

    crash = make_synthetic_crash_records(n_samples=5_000)
    crash.to_csv(processed / "crash_data.csv", index=False)
    print(f"[setup_environment] wrote {processed / 'crash_data.csv'}")


def main() -> int:
    args = parse_args()
    if not args.no_install:
        run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
    if args.download_models:
        run([sys.executable, str(ROOT / "scripts" / "download_models.py")])
    if args.generate_synthetic:
        generate_synthetic_data()
    print("[setup_environment] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
