from __future__ import annotations

import os
import time
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional local dependency during syntax checks
    psutil = None


STARTED_AT = time.time()


def system_health() -> dict[str, Any]:
    if psutil is None:
        return {
            "uptime_sec": int(time.time() - STARTED_AT),
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "pid": os.getpid(),
        }
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return {
        "uptime_sec": int(time.time() - STARTED_AT),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": mem.percent,
        "ram_used_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        "ram_total_mb": round(mem.total / (1024 * 1024), 1),
        "pid": os.getpid(),
    }

