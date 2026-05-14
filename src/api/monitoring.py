"""In-process monitoring counters and latency histogram."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class InferenceMonitor:
    window: int = 1024
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=1024), init=False)
    _count: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._latencies_ms = deque(maxlen=self.window)

    def record(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self._latencies_ms.append(float(latency_ms))
            self._count += 1
            if error:
                self._errors += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            n = len(latencies)
            uptime = max(time.time() - self._started_at, 1e-6)
            if not latencies:
                p50 = p95 = p99 = mean = 0.0
            else:
                p50 = latencies[int(0.50 * (n - 1))]
                p95 = latencies[int(0.95 * (n - 1))]
                p99 = latencies[int(0.99 * (n - 1))]
                mean = sum(latencies) / n
            return {
                "count": self._count,
                "errors": self._errors,
                "throughput_per_sec": self._count / uptime,
                "latency_ms_mean": mean,
                "latency_ms_p50": p50,
                "latency_ms_p95": p95,
                "latency_ms_p99": p99,
                "uptime_sec": uptime,
            }
