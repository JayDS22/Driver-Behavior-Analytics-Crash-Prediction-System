"""Download pretrained weights into ``data/models/``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "yolov7_vehicle.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
}


def download(url: str, dest: Path, chunk: int = 1 << 16) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as resp, dest.open("wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
        return True
    except (URLError, TimeoutError, OSError) as exc:
        print(f"[download_models] WARNING: failed to fetch {url}: {exc}", file=sys.stderr)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument("--target-dir", default=str(ROOT / "data" / "models"))
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip remote downloads",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if args.skip_network:
        print("[download_models] --skip-network set; nothing to do")
        return 0

    ok = True
    for filename, url in DEFAULTS.items():
        dest = target / filename
        if dest.exists():
            print(f"[download_models] {dest} already present, skipping")
            continue
        print(f"[download_models] fetching {url} -> {dest}")
        if not download(url, dest):
            ok = False

    if not ok:
        readme = target / "README.txt"
        readme.write_text(
            "Weight download failed. Re-run with network access to fetch\n"
            "yolov7_vehicle.pt. Until then the detector uses the stub backend.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
