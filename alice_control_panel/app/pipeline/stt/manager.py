from __future__ import annotations

import asyncio
import os
import tempfile
import time
import wave
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
        self._model: Any = None
        self._model_key = ""

    async def status(self) -> dict[str, Any]:
        cfg = (await self._config_store.get(include_secrets=False)).get("stt", {})
        return {
            "provider": cfg.get("provider", "faster_whisper"),
            "model": cfg.get("model", "small"),
            "language": cfg.get("language", "tr"),
            "compute_type": cfg.get("compute_type", "int8"),
            "beam_size": cfg.get("beam_size", 1),
            "vad_filter": bool(cfg.get("vad_filter", False)),
            "loaded": self._loaded,
            "last_error": self._last_error,
            "last_capture": self._last_capture,
            "uptime_sec": int(time.time() - self._started_at) if self._started_at else 0,
        }

    async def restart(self) -> dict[str, Any]:
        self._loaded = False
        self._model = None
        self._model_key = ""
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
        await self._log_bus.emit("INFO", "STT", "PCM audio captured for STT", self._last_capture)

        config = await self._config_store.get(include_secrets=False)
        cfg = config.get("stt", {}) if isinstance(config.get("stt"), dict) else {}
        provider = str(cfg.get("provider") or "faster_whisper").lower()
        if provider in {"none", "mock"}:
            return {
                "ok": False,
                "captured": True,
                "text": "",
                "message": (
                    f"Audio captured: {duration_sec:.1f}s, {len(audio)} bytes, "
                    f"rms {rms}, peak {peak}. STT provider is {provider}."
                ),
                "capture": self._last_capture,
            }
        if provider != "faster_whisper":
            message = f"STT provider '{provider}' is not implemented yet."
            self._last_error = message
            await self._log_bus.emit("WARN", "STT", message)
            return {
                "ok": False,
                "captured": True,
                "text": "",
                "message": message,
                "capture": self._last_capture,
            }

        try:
            text = await asyncio.to_thread(self._transcribe_faster_whisper, audio, sample_rate, cfg)
        except Exception as exc:
            self._last_error = str(exc)
            await self._log_bus.emit("ERROR", "STT", "faster-whisper transcription failed", {"error": str(exc)})
            return {
                "ok": False,
                "captured": True,
                "text": "",
                "message": f"STT failed: {exc}",
                "capture": self._last_capture,
            }

        self._last_error = ""
        self._last_capture["text"] = text
        if text:
            await self._log_bus.emit("INFO", "STT", "STT transcription completed", {"text": text, "chars": len(text)})
            return {
                "ok": True,
                "captured": True,
                "text": text,
                "message": text,
                "capture": self._last_capture,
            }

        await self._log_bus.emit("WARN", "STT", "STT completed with empty transcript", self._last_capture)
        return {
            "ok": False,
            "captured": True,
            "text": "",
            "message": (
                f"Audio captured: {duration_sec:.1f}s, {len(audio)} bytes, "
                f"rms {rms}, peak {peak}. STT returned empty text."
            ),
            "capture": self._last_capture,
        }

    def _ensure_faster_whisper_model(self, cfg: dict[str, Any]) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed in the add-on image.") from exc

        model_name = str(cfg.get("model") or "small")
        compute_type = str(cfg.get("compute_type") or "int8")
        model_key = f"{model_name}|{compute_type}"
        if self._model is not None and self._model_key == model_key:
            return self._model

        self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        self._model_key = model_key
        self._loaded = True
        self._started_at = self._started_at or time.time()
        return self._model

    def _transcribe_faster_whisper(self, audio: bytes, sample_rate: int, cfg: dict[str, Any]) -> str:
        model = self._ensure_faster_whisper_model(cfg)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio)

            segments, _info = model.transcribe(
                temp_path,
                language=str(cfg.get("language") or "tr") or None,
                beam_size=max(1, int(cfg.get("beam_size") or 1)),
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
