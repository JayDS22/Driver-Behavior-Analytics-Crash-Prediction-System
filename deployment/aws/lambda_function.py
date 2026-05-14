"""AWS Lambda inference handler.

The SafetySystem is built on cold start and reused across warm invocations
to keep p50 below the 150ms SLO.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import numpy as np

_SYSTEM = None


def _get_system():
    global _SYSTEM
    if _SYSTEM is None:
        from src.api.inference_api import SafetySystem

        _SYSTEM = SafetySystem.from_config()
    return _SYSTEM


def _decode_image(b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    try:
        import cv2  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        try:
            import io

            from PIL import Image  # type: ignore

            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))[:, :, ::-1]
        except Exception:
            return None


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    body = event.get("body")
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _response(400, {"error": "invalid JSON body"})
    else:
        payload = body or event

    image_b64 = payload.get("image")
    if not image_b64:
        return _response(400, {"error": "missing 'image' field"})
    frame = _decode_image(image_b64)
    if frame is None:
        return _response(400, {"error": "could not decode image"})

    system = _get_system()
    result = system.process_frame(frame, metadata=payload.get("metadata") or {})
    return _response(200, {"predictions": result})


def _response(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: lambda_function.py <event.json>")
        raise SystemExit(2)
    with open(sys.argv[1]) as fh:
        event_data = json.load(fh)
    print(json.dumps(handler(event_data), indent=2))
