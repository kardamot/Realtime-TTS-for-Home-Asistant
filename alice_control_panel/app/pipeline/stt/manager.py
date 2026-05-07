from __future__ import annotations

import time
from typing import Any

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus


class SttManager:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._loaded = False
        self._last_error = ""
        self._started_at: float | None = None
        self._last_capture: dict[str, Any] = {}

    async def status(self) -> dict[str, Any]:
        cfg = (await self._config_store.get(include_secrets=False)).get("stt", {})
        return {
            "provider": cfg.get("provider", "faster_whisper"),
            "model": cfg.get("model", "small"),
            "language": cfg.get("language", "tr"),
            "compute_type": cfg.get("compute_type", "int8"),
            "loaded": self._loaded,
            "last_error": self._last_error,
            "last_capture": self._last_capture,
            "uptime_sec": int(time.time() - self._started_at) if self._started_at else 0,
        }

    async def restart(self) -> dict[str, Any]:
        self._loaded = False
        self._last_error = ""
        self._started_at = time.time()
        await self._log_bus.emit("INFO", "STT", "STT manager restart requested")
        return await self.status()

    async def transcribe_pcm16(self, audio: bytes, sample_rate: int) -> dict[str, Any]:
        sample_count = len(audio) // 2
        duration_sec = sample_count / sample_rate if sample_rate > 0 else 0.0
        peak = 0
        rms = 0
        if sample_count:
            samples = memoryview(audio[: sample_count * 2]).cast("h")
            total_sq = 0
            for sample in samples:
                value = int(sample)
                abs_value = abs(value)
                if abs_value > peak:
                    peak = abs_value
                total_sq += value * value
            rms = int((total_sq / sample_count) ** 0.5)

        self._last_capture = {
            "bytes": len(audio),
            "sample_rate": sample_rate,
            "samples": sample_count,
            "duration_sec": round(duration_sec, 2),
            "rms": rms,
            "peak": peak,
            "captured_at": time.time(),
        }
        await self._log_bus.emit(
            "INFO",
            "STT",
            "PCM audio captured for STT",
            self._last_capture,
        )
        return {
            "ok": False,
            "captured": True,
            "text": "",
            "message": (
                f"Audio captured: {duration_sec:.1f}s, {len(audio)} bytes, "
                f"rms {rms}, peak {peak}. STT engine is not wired yet."
            ),
            "capture": self._last_capture,
        }
