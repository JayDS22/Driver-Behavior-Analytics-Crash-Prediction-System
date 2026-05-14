"""Train the YOLOv7 vehicle detector.

Delegates to ``${YOLOV7_DIR}/train.py`` (WongKinYiu/yolov7) when available;
otherwise runs in --dry-run mode, validating configs only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger("training.train_yolo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv7 vehicle detector")
    parser.add_argument("--data", default="config/vehicle_dataset.yaml")
    parser.add_argument("--cfg", default="config/yolov7.yaml")
    parser.add_argument("--weights", default="data/models/yolov7.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="vehicle_safety")
    parser.add_argument(
        "--yolov7-dir",
        default=os.environ.get("YOLOV7_DIR", "yolov7"),
        help="Path to cloned WongKinYiu/yolov7 repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs only",
    )
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    for path in (args.data, args.cfg):
        if not Path(path).exists():
            logger.warning("Config file missing: %s", path)


def main() -> int:
    args = parse_args()
    _validate_paths(args)

    yolov7_train = Path(args.yolov7_dir) / "train.py"
    if args.dry_run or not yolov7_train.exists():
        if not yolov7_train.exists():
            logger.warning(
                "YOLOv7 train.py not found at %s -- clone "
                "https://github.com/WongKinYiu/yolov7 or pass --yolov7-dir",
                yolov7_train,
            )
        logger.info("args: %s", vars(args))
        return 0

    cmd = [
        sys.executable,
        str(yolov7_train),
        "--data", args.data,
        "--cfg", args.cfg,
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img", str(args.img_size),
        "--device", args.device,
        "--project", args.project,
        "--name", args.name,
    ]
    logger.info("exec: %s", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
