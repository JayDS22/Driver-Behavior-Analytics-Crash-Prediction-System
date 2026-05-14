"""Batch inference over an image folder or video file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.inference_api import SafetySystem  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.run_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a folder or video")
    parser.add_argument("--input", required=True, help="Folder of images, image file, or video file")
    parser.add_argument("--output", default="results/predictions.json")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = SafetySystem.from_config()
    results = system.process_video(args.input, max_frames=args.max_frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    summary = {k: v for k, v in results.items() if k != "per_frame"}
    logger.info("wrote %d predictions -> %s", results["frames_processed"], output_path)
    logger.info("summary: %s", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
