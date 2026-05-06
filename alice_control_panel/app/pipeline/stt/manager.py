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

    async def status(self) -> dict[str, Any]:
        cfg = (await self._config_store.get(include_secrets=False)).get("stt", {})
        return {
            "provider": cfg.get("provider", "faster_whisper"),
            "model": cfg.get("model", "small"),
            "language": cfg.get("language", "tr"),
            "compute_type": cfg.get("compute_type", "int8"),
            "loaded": self._loaded,
            "last_error": self._last_error,
            "uptime_sec": int(time.time() - self._started_at) if self._started_at else 0,
        }

    async def restart(self) -> dict[str, Any]:
        self._loaded = False
        self._last_error = ""
        self._started_at = time.time()
        await self._log_bus.emit("INFO", "STT", "STT manager restart requested")
        return await self.status()

    async def transcribe_pcm16(self, _: bytes, sample_rate: int) -> dict[str, Any]:
        await self._log_bus.emit(
            "WARN",
            "STT",
            "Local STT transcription is not wired in the control panel preview yet",
            {"sample_rate": sample_rate},
        )
        return {
            "ok": False,
            "text": "",
            "message": "STT engine placeholder; Realtime/ESP audio bridge will wire this in next pass.",
        }

