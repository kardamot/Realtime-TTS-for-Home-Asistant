from __future__ import annotations

import json
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Any


DEFAULT_MAX_PCM_BYTES = 16 * 1024 * 1024
LATEST_CAPTURE_NAME = "latest.wav"
LATEST_METADATA_NAME = "latest.json"


class TtsCaptureSession:
    def __init__(
        self,
        store: "TtsCaptureStore",
        sample_rate: int,
        channels: int,
        provider: str,
        transport: str,
    ) -> None:
        self._store = store
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        self.provider = str(provider or "unknown")
        self.transport = str(transport or "unknown")
        self.pcm_bytes = 0
        self._failed_reason = ""
        self._closed = False
        self._temp_path = store.directory / f"capture-{uuid.uuid4().hex}.tmp.wav"
        self._writer = wave.open(str(self._temp_path), "wb")
        self._writer.setnchannels(self.channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(self.sample_rate)

    def write(self, pcm: bytes) -> None:
        if self._closed or self._failed_reason or not pcm:
            return
        next_size = self.pcm_bytes + len(pcm)
        if next_size > self._store.max_pcm_bytes:
            self.abort("capture_too_large")
            return
        try:
            self._writer.writeframesraw(pcm)
            self.pcm_bytes = next_size
        except Exception as exc:
            self.abort(f"capture_write_failed:{type(exc).__name__}")

    def finish(self) -> dict[str, Any] | None:
        if self._closed:
            return None
        if self._failed_reason or self.pcm_bytes <= 0:
            self.abort(self._failed_reason or "capture_empty")
            return None
        try:
            self._writer.close()
            self._closed = True
            return self._store._publish(self)
        except Exception as exc:
            self._failed_reason = f"capture_finish_failed:{type(exc).__name__}"
            self._store._record_error(self._failed_reason)
            self._close_and_remove()
            return None

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self._failed_reason = str(reason or "capture_aborted")
        self._store._record_error(self._failed_reason)
        self._close_and_remove()

    def _close_and_remove(self) -> None:
        if not self._closed:
            try:
                self._writer.close()
            except Exception:
                pass
            self._closed = True
        try:
            self._temp_path.unlink(missing_ok=True)
        except OSError:
            pass


class TtsCaptureStore:
    def __init__(self, directory: Path, max_pcm_bytes: int = DEFAULT_MAX_PCM_BYTES) -> None:
        self.directory = Path(directory)
        self.max_pcm_bytes = max(1024, int(max_pcm_bytes))
        self._last_error = ""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._cleanup_temps()
        except OSError as exc:
            self._record_error(f"capture_directory_failed:{type(exc).__name__}")

    @property
    def latest_path(self) -> Path:
        return self.directory / LATEST_CAPTURE_NAME

    @property
    def metadata_path(self) -> Path:
        return self.directory / LATEST_METADATA_NAME

    def begin(
        self,
        sample_rate: int,
        channels: int,
        provider: str,
        transport: str,
    ) -> TtsCaptureSession | None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            return TtsCaptureSession(self, sample_rate, channels, provider, transport)
        except Exception as exc:
            self._record_error(f"capture_start_failed:{type(exc).__name__}")
            return None

    def status(self) -> dict[str, Any]:
        metadata = self._read_metadata()
        try:
            file_size = self.latest_path.stat().st_size
        except OSError:
            file_size = 0
        available = file_size > 44
        created_at = float(metadata.get("created_at") or 0)
        filename = str(metadata.get("filename") or "alice_tts_latest.wav")
        return {
            "available": available,
            "url": "/api/pipeline/tts/latest.wav" if available else "",
            "filename": filename if available else "",
            "provider": str(metadata.get("provider") or "") if available else "",
            "transport": str(metadata.get("transport") or "") if available else "",
            "sample_rate": int(metadata.get("sample_rate") or 0) if available else 0,
            "channels": int(metadata.get("channels") or 0) if available else 0,
            "pcm_bytes": int(metadata.get("pcm_bytes") or max(0, file_size - 44)) if available else 0,
            "file_bytes": file_size if available else 0,
            "created_at": created_at if available else 0,
            "max_pcm_bytes": self.max_pcm_bytes,
            "last_error": self._last_error,
        }

    def _publish(self, session: TtsCaptureSession) -> dict[str, Any]:
        created_at = time.time()
        filename = time.strftime("alice_tts_%Y%m%d_%H%M%S.wav", time.localtime(created_at))
        metadata = {
            "filename": filename,
            "provider": session.provider,
            "transport": session.transport,
            "sample_rate": session.sample_rate,
            "channels": session.channels,
            "pcm_bytes": session.pcm_bytes,
            "created_at": created_at,
        }
        metadata_temp = self.directory / f"metadata-{uuid.uuid4().hex}.tmp.json"
        try:
            metadata_temp.write_text(json.dumps(metadata, ensure_ascii=True), encoding="utf-8")
            os.replace(session._temp_path, self.latest_path)
            os.replace(metadata_temp, self.metadata_path)
        finally:
            try:
                metadata_temp.unlink(missing_ok=True)
            except OSError:
                pass
        self._last_error = ""
        return {**metadata, "available": True, "url": "/api/pipeline/tts/latest.wav"}

    def _read_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _cleanup_temps(self) -> None:
        for pattern in ("capture-*.tmp.wav", "metadata-*.tmp.json"):
            for path in self.directory.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass

    def _record_error(self, reason: str) -> None:
        self._last_error = str(reason or "capture_failed")


class CapturingPcmOutput:
    def __init__(
        self,
        output: Any,
        store: TtsCaptureStore,
        provider: str,
        transport: str,
    ) -> None:
        self._output = output
        self._store = store
        self._provider = provider
        self._transport = transport
        self._capture: TtsCaptureSession | None = None

    @property
    def pace_pcm(self) -> bool:
        return bool(getattr(self._output, "pace_pcm", True))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._output, name)

    def set_trace(self, trace: Any) -> None:
        self._output.set_trace(trace)

    async def start(self, sample_rate: int, channels: int = 1) -> None:
        await self._output.start(sample_rate, channels)
        self._capture = self._store.begin(sample_rate, channels, self._provider, self._transport)

    async def write(self, pcm: bytes) -> None:
        await self._output.write(pcm)
        if self._capture is not None:
            self._capture.write(pcm)

    async def done(self) -> None:
        try:
            await self._output.done()
        except Exception:
            if self._capture is not None:
                self._capture.abort("output_done_failed")
            raise
        if self._capture is not None:
            self._capture.finish()

    async def error(self, message: str, status: int = 500) -> None:
        if self._capture is not None:
            self._capture.abort(f"output_error:{status}")
        await self._output.error(message, status)
