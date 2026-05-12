from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from typing import Any

import aiohttp
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.prompt_store import PromptStore
from app.core.ws_hub import WsHub
from app.pipeline.llm.openai_compatible import active_llm_config


OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
EMOTION_TAG_RE = re.compile(r"<emotion:\s*([^>]+)>", re.IGNORECASE)
INCOMPLETE_EMOTION_TAG_RE = re.compile(r"<emotion:\s*[^>]*$", re.IGNORECASE)
STREAM_CHUNK_MIN_CHARS = 28
STREAM_CHUNK_HARD_CHARS = 90


def safe_exc_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()


def normalize_esp_pcm_packet(chunk: bytes) -> tuple[bytes, int | None]:
    if not chunk:
        return b"", None
    header: int | None = None
    if len(chunk) & 1:
        header = chunk[0]
        chunk = chunk[1:]
    if len(chunk) & 1:
        chunk = chunk[:-1]
    return bytes(chunk), header


def realtime_ws_url(cfg: dict[str, Any]) -> str:
    base = str(cfg.get("ws_url") or OPENAI_REALTIME_WS_URL).strip().rstrip("/")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}model={str(cfg.get('model') or 'gpt-realtime-mini')}"


def active_realtime_config(config: dict[str, Any]) -> dict[str, Any]:
    realtime = config.get("realtime", {}) if isinstance(config, dict) else {}
    if not isinstance(realtime, dict):
        return {}
    provider = str(realtime.get("provider") or "openai").lower()
    providers = realtime.get("providers", {}) if isinstance(realtime.get("providers"), dict) else {}
    profile = providers.get(provider, {}) if isinstance(providers.get(provider), dict) else {}
    merged = {**realtime, **profile}
    merged["provider"] = provider
    merged["providers"] = providers
    return merged


def pcm16le_resample_linear(chunk: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not chunk or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return chunk
    samples = np.frombuffer(chunk, dtype="<i2")
    if samples.size <= 1:
        return chunk
    out_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    src_x = np.arange(samples.size, dtype=np.float32)
    dst_x = np.linspace(0, samples.size - 1, out_len, dtype=np.float32)
    out = np.interp(dst_x, src_x, samples.astype(np.float32))
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


def extract_realtime_text_delta(doc: dict[str, Any]) -> str:
    for key in ("delta", "text", "transcript"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_realtime_response_text(doc: dict[str, Any]) -> str:
    response = doc.get("response")
    if not isinstance(response, dict):
        return ""
    parts: list[str] = []
    for output in response.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("transcript")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


class RealtimeTextChunker:
    def __init__(self) -> None:
        self._raw_pending = ""
        self._spoken_pending = ""
        self._all_spoken_text = ""

    @property
    def all_text(self) -> str:
        return self._all_spoken_text.strip()

    def _strip_emotions(self) -> list[str]:
        emotions: list[str] = []
        while True:
            match = EMOTION_TAG_RE.search(self._raw_pending)
            if not match:
                break
            emotion = match.group(1).strip()
            if emotion:
                emotions.append(emotion)
            self._raw_pending = self._raw_pending[: match.start()] + self._raw_pending[match.end() :]
        return emotions

    def _flush_safe_text(self) -> None:
        incomplete = INCOMPLETE_EMOTION_TAG_RE.search(self._raw_pending)
        if incomplete:
            safe_text = self._raw_pending[: incomplete.start()]
            self._raw_pending = self._raw_pending[incomplete.start() :]
        else:
            safe_text = self._raw_pending
            self._raw_pending = ""
        if not safe_text:
            return
        self._spoken_pending += safe_text
        self._all_spoken_text += safe_text

    def _find_boundary(self) -> int:
        text = self._spoken_pending
        for idx, ch in enumerate(text):
            if ch in ".?!\n" and idx + 1 >= STREAM_CHUNK_MIN_CHARS:
                next_char_ok = idx + 1 >= len(text) or text[idx + 1].isspace() or text[idx + 1] in "\"'"
                if next_char_ok:
                    return idx + 1
        if len(text) >= STREAM_CHUNK_HARD_CHARS:
            split_idx = text.rfind(" ", STREAM_CHUNK_MIN_CHARS, STREAM_CHUNK_HARD_CHARS)
            if split_idx > 0:
                return split_idx
            return STREAM_CHUNK_HARD_CHARS
        return -1

    def _drain_chunks(self, final: bool) -> list[str]:
        parts: list[str] = []
        while True:
            boundary = self._find_boundary()
            if boundary < 0:
                break
            part = self._spoken_pending[:boundary].strip()
            self._spoken_pending = self._spoken_pending[boundary:].lstrip()
            if part:
                parts.append(part)
        if final:
            tail = self._spoken_pending.strip()
            self._spoken_pending = ""
            if tail:
                parts.append(tail)
        return parts

    def push(self, delta: str) -> tuple[list[str], list[str]]:
        self._raw_pending += delta
        emotions = self._strip_emotions()
        self._flush_safe_text()
        return emotions, self._drain_chunks(final=False)

    def finish(self) -> tuple[list[str], list[str], str]:
        emotions = self._strip_emotions()
        if self._raw_pending and not INCOMPLETE_EMOTION_TAG_RE.search(self._raw_pending):
            self._flush_safe_text()
        else:
            self._raw_pending = ""
        chunks = self._drain_chunks(final=True)
        return emotions, chunks, self.all_text


class OpenAIRealtimeBridge:
    def __init__(
        self,
        config_store: ConfigStore,
        prompt_store: PromptStore,
        log_bus: LogBus,
        ws_hub: WsHub,
        tts_relay: Any,
        esp_client: Any,
        ha_bridge: Any | None = None,
    ) -> None:
        self._config_store = config_store
        self._prompt_store = prompt_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._tts_relay = tts_relay
        self._esp_client = esp_client
        self._ha_bridge = ha_bridge
        self._active = False
        self._connected = False
        self._last_event = "idle"
        self._last_error = ""
        self._session_id = ""
        self._model = ""
        self._started_at: float | None = None
        self._cancel_event = asyncio.Event()

    async def should_handle_voice_ws(self) -> bool:
        config = await self._config_store.get(include_secrets=True)
        realtime = self._realtime_cfg(config)
        return (
            bool(realtime.get("enabled", False))
            and str(realtime.get("provider") or "openai").lower() == "openai"
            and bool(self._api_key(config, realtime))
        )

    async def status(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=False)
        realtime = self._realtime_cfg(config)
        return {
            "enabled": bool(realtime.get("enabled", False)),
            "provider": str(realtime.get("provider") or "openai"),
            "model": str(realtime.get("model") or "gpt-realtime-mini"),
            "active": self._active,
            "connected": self._connected,
            "session_id": self._session_id,
            "last_event": self._last_event,
            "last_error": self._last_error,
            "uptime_sec": int(time.time() - self._started_at) if self._started_at else 0,
        }

    async def websocket_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        config = await self._config_store.get(include_secrets=True)
        realtime = self._realtime_cfg(config)
        api_key = self._api_key(config, realtime)
        source_sample_rate = 16000
        target_sample_rate = max(8000, int(realtime.get("input_sample_rate") or 24000))
        language = str(config.get("stt", {}).get("language") or "tr") if isinstance(config.get("stt"), dict) else "tr"
        session_id = f"rt-{uuid.uuid4().hex[:10]}"
        transcript = ""
        assistant_text = ""
        text_chunker = RealtimeTextChunker()
        tts_chunk_started = False
        response_requested = False
        response_done = False
        stt_result_sent = False
        llm_started_sent = False
        speech_started = False
        audio_ms = 0
        buffered_audio_ms = 0
        input_committed = False
        stripped_packet_headers = 0
        transcript_event = asyncio.Event()
        response_wait_task: asyncio.Task[None] | None = None
        realtime_ws: aiohttp.ClientWebSocketResponse | None = None
        client_session: aiohttp.ClientSession | None = None
        reader_task: asyncio.Task[None] | None = None
        self._cancel_event = asyncio.Event()
        self._active = True
        self._connected = False
        self._last_error = ""
        self._last_event = "connecting"
        self._session_id = session_id
        self._model = str(realtime.get("model") or "gpt-realtime-mini")
        self._started_at = time.time()

        async def send_event(event_type: str, **data: Any) -> None:
            payload = {"type": event_type, **data}
            try:
                await websocket.send_json(payload)
            except Exception:
                pass
            await self._ws_hub.publish("pipeline_status", {"realtime": await self.status()})

        async def send_realtime_json(payload: dict[str, Any]) -> None:
            if realtime_ws is not None and not realtime_ws.closed:
                await realtime_ws.send_str(json.dumps(payload, ensure_ascii=False))

        async def request_response(force: bool = False) -> None:
            nonlocal response_requested
            if response_done or (response_requested and not force):
                return
            response_requested = True
            await send_realtime_json(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["text"],
                        "instructions": await self._instructions(config, realtime),
                    },
                }
            )
            self._last_event = "response_requested"
            await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime response requested", {"session_id": session_id})

        async def send_stt_result_once(reason: str) -> None:
            nonlocal stt_result_sent
            if stt_result_sent:
                return
            stt_result_sent = True
            await send_event("stt_result", text=transcript.strip(), provider="openai_realtime", reason=reason)

        async def send_llm_started_once() -> None:
            nonlocal llm_started_sent
            if llm_started_sent:
                return
            llm_started_sent = True
            await send_event("llm_started", model=self._model, provider="openai_realtime")

        async def send_tts_chunk(text: str, final: bool) -> None:
            nonlocal tts_chunk_started
            if text.strip():
                tts_chunk_started = True
            await send_event("llm_chunk", text=text, final=final)

        async def request_response_after_transcript_wait(reason: str = "realtime_committed") -> None:
            if response_requested or response_done:
                return
            wait_ms = max(0, int(realtime.get("transcript_wait_ms") or 800))
            if wait_ms and not transcript_event.is_set():
                try:
                    await asyncio.wait_for(transcript_event.wait(), timeout=wait_ms / 1000)
                except asyncio.TimeoutError:
                    pass
            if response_requested or response_done:
                return
            if bool(realtime.get("suppress_empty_transcript_response", True)) and not transcript.strip():
                await send_stt_result_once("empty_transcript_suppressed")
                await finish_response(reason="empty_transcript_suppressed")
                return
            await send_stt_result_once(reason)
            if await try_home_assistant_route(reason):
                return
            await request_response()

        async def finish_response(doc: dict[str, Any] | None = None, reason: str = "openai_realtime_done") -> None:
            nonlocal response_done, assistant_text, text_chunker, tts_chunk_started
            if response_done:
                return
            response_done = True
            final_text = extract_realtime_response_text(doc or {})
            ready_chunks: list[str] = []
            ready_emotions: list[str] = []
            if final_text and not assistant_text.strip():
                assistant_text = final_text
                text_chunker = RealtimeTextChunker()
                ready_emotions, ready_chunks = text_chunker.push(final_text)
            elif assistant_text.strip() and not text_chunker.all_text:
                text_chunker = RealtimeTextChunker()
                ready_emotions, ready_chunks = text_chunker.push(assistant_text)

            emotions, final_chunks, spoken_text = text_chunker.finish()
            for emotion in [*ready_emotions, *emotions]:
                await send_event("emotion", name=emotion)
            if spoken_text:
                assistant_text = spoken_text

            chunks_to_send = [*ready_chunks, *final_chunks]
            if chunks_to_send:
                for chunk in chunks_to_send[:-1]:
                    await send_tts_chunk(chunk, final=False)
                await send_tts_chunk(chunks_to_send[-1], final=True)
            elif assistant_text.strip() and tts_chunk_started:
                await send_tts_chunk("", final=True)
            else:
                await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime completed without assistant text", {"session_id": session_id, "reason": reason})
            if assistant_text.strip():
                await send_event("llm_result", text=assistant_text)
            await send_event(
                "session_completed",
                reason=reason,
                audio_ms=audio_ms,
                assistant_text=assistant_text,
                transcript=transcript,
            )
            self._last_event = "completed"
            await self._log_bus.emit(
                "INFO",
                "PIPELINE",
                "OpenAI Realtime session completed",
                {"session_id": session_id, "transcript_chars": len(transcript), "assistant_chars": len(assistant_text)},
            )
            # The ESP voice client consumes llm_chunk/llm_result events and opens
            # the TTS relay itself. Starting a second direct ESP audio stream here
            # races the firmware player and can trigger tts_still_active.

        async def try_home_assistant_route(reason: str) -> bool:
            nonlocal assistant_text, response_requested, text_chunker
            if self._ha_bridge is None or response_done or response_requested:
                return False
            user_text = transcript.strip()
            if not user_text:
                return False
            try:
                if not await self._ha_bridge.should_route_home_control(user_text):
                    return False
                result = await self._ha_bridge.handle_text_command(user_text)
                if not result.get("handled"):
                    return False
                speech = str(result.get("speech") or "").strip()
                if not speech:
                    return False
                response_requested = True
                assistant_text = speech
                text_chunker = RealtimeTextChunker()
                await send_llm_started_once()
                self._last_event = "ha_route"
                await self._log_bus.emit(
                    "INFO",
                    "HA",
                    "Realtime HA route completed",
                    {
                        "session_id": session_id,
                        "reason": reason,
                        "ok": bool(result.get("ok")),
                        "entity_id": result.get("entity_id"),
                        "action": result.get("action"),
                    },
                )
                await finish_response(reason="ha_route")
                return True
            except PermissionError as exc:
                await self._log_bus.emit("WARN", "HA", "Realtime HA route denied", {"session_id": session_id, "error": str(exc)})
                return False
            except Exception as exc:
                await self._log_bus.emit("ERROR", "HA", "Realtime HA route failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                return False

        async def handle_realtime_event(doc: dict[str, Any]) -> None:
            nonlocal transcript, assistant_text, text_chunker, response_done, response_requested, speech_started, input_committed, response_wait_task
            event_type = str(doc.get("type") or "")
            if not event_type:
                return
            self._last_event = event_type
            if event_type == "error":
                error = doc.get("error") if isinstance(doc.get("error"), dict) else {}
                message = str(error.get("message") or doc.get("message") or "OpenAI Realtime error")
                self._last_error = message
                await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime error", {"session_id": session_id, "error": message})
                await send_event("error", message=f"OpenAI Realtime: {message}")
                return
            if event_type == "session.updated":
                await send_event("realtime_session_updated", model=self._model)
                return
            if event_type == "input_audio_buffer.speech_started":
                speech_started = True
                await self._cancel_playback("realtime_barge_in")
                await send_event("vad_start", vad_provider="openai_realtime", audio_ts=doc.get("audio_start_ms"))
                return
            if event_type == "input_audio_buffer.speech_stopped":
                await send_event("vad_end", vad_provider="openai_realtime", audio_ts=doc.get("audio_end_ms"), reason="server_vad")
                return
            if event_type == "input_audio_buffer.committed":
                input_committed = True
                if response_wait_task is None or response_wait_task.done():
                    response_wait_task = asyncio.create_task(request_response_after_transcript_wait("realtime_committed"))
                return
            if event_type == "conversation.item.input_audio_transcription.delta":
                delta = extract_realtime_text_delta(doc)
                if delta:
                    transcript += delta
                    await send_event("stt_delta", text=delta, provider="openai_realtime")
                return
            if event_type == "conversation.item.input_audio_transcription.completed":
                text = str(doc.get("transcript") or "").strip()
                if text:
                    transcript = text
                transcript_event.set()
                if stt_result_sent:
                    await send_event("stt_transcript", text=transcript.strip(), provider="openai_realtime", late=True)
                elif input_committed and (response_wait_task is None or response_wait_task.done()):
                    response_wait_task = asyncio.create_task(request_response_after_transcript_wait("transcription_completed"))
                return
            if event_type == "response.created":
                await send_llm_started_once()
                return
            if event_type in {"response.output_text.delta", "response.text.delta", "response.audio_transcript.delta"}:
                await send_llm_started_once()
                delta = extract_realtime_text_delta(doc)
                if delta:
                    assistant_text += delta
                    await send_event("llm_delta", text=delta)
                    emotions, chunks = text_chunker.push(delta)
                    for emotion in emotions:
                        await send_event("emotion", name=emotion)
                    for chunk in chunks:
                        await send_tts_chunk(chunk, final=False)
                return
            if event_type == "response.done":
                await finish_response(doc)
                return
            if event_type == "response.cancelled":
                response_done = True
                await send_event("session_cancelled", reason="response_cancelled")
                return

        async def reader_loop() -> None:
            if realtime_ws is None:
                return
            try:
                async for msg in realtime_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            await handle_realtime_event(json.loads(msg.data))
                        except json.JSONDecodeError:
                            continue
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(str(realtime_ws.exception() or "OpenAI Realtime websocket error"))
            except Exception as exc:
                if not self._cancel_event.is_set():
                    self._last_error = safe_exc_message(exc)
                    await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime reader failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                    await send_event("error", message=f"OpenAI Realtime reader failed: {safe_exc_message(exc)}")

        async def open_realtime() -> bool:
            nonlocal realtime_ws, reader_task, client_session
            if realtime_ws is not None and not realtime_ws.closed:
                return True
            if not api_key:
                await send_event("error", message="Realtime is enabled but OpenAI API key is empty.")
                return False
            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=240)
                client_session = aiohttp.ClientSession(timeout=timeout)
                realtime_ws = await client_session.ws_connect(
                    realtime_ws_url(realtime),
                    headers={"Authorization": f"Bearer {api_key}"},
                    heartbeat=20,
                )
                await send_realtime_json(await self._session_update_payload(config, realtime, target_sample_rate, language))
                reader_task = asyncio.create_task(reader_loop())
                self._connected = True
                self._last_event = "connected"
                await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime connected", {"session_id": session_id, "model": self._model})
                return True
            except Exception as exc:
                self._last_error = safe_exc_message(exc)
                await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime connect failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                await send_event("error", message=f"OpenAI Realtime connect failed: {safe_exc_message(exc)}")
                return False

        try:
            await send_event(
                "hello",
                service="alice_control_panel",
                version="0.1.60",
                session_id=session_id,
                endpointing_enabled=True,
                endpointing_provider="openai_realtime",
                realtime_enabled=True,
                realtime_provider="openai",
                realtime_model=self._model,
                llm_enabled=True,
                tts_enabled=True,
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect
                if message.get("text") is not None:
                    doc = json.loads(str(message["text"]))
                    msg_type = str(doc.get("type") or "").strip().lower()
                    if msg_type == "start":
                        transcript = ""
                        assistant_text = ""
                        text_chunker = RealtimeTextChunker()
                        tts_chunk_started = False
                        response_requested = False
                        response_done = False
                        stt_result_sent = False
                        llm_started_sent = False
                        speech_started = False
                        audio_ms = 0
                        buffered_audio_ms = 0
                        input_committed = False
                        transcript_event = asyncio.Event()
                        if response_wait_task is not None and not response_wait_task.done():
                            response_wait_task.cancel()
                        response_wait_task = None
                        source_sample_rate = int(doc.get("sample_rate") or source_sample_rate)
                        language = str(doc.get("language") or language).strip() or "tr"
                        session_id = str(doc.get("session_id") or session_id).strip() or session_id
                        self._session_id = session_id
                        if await open_realtime():
                            await send_event(
                                "session_started",
                                session_id=session_id,
                                sample_rate=source_sample_rate,
                                realtime_enabled=True,
                                realtime_model=self._model,
                                endpointing_provider="openai_realtime",
                            )
                        continue
                    if msg_type in {"end", "eos"}:
                        if not realtime_ws:
                            await send_event("error", message="Realtime session is not started.")
                            continue
                        if input_committed or response_done:
                            continue
                        try:
                            if speech_started and buffered_audio_ms >= 100:
                                await send_realtime_json({"type": "input_audio_buffer.commit"})
                                input_committed = True
                            else:
                                await send_realtime_json({"type": "input_audio_buffer.clear"})
                                input_committed = True
                                await send_event(
                                    "stt_result",
                                    text="",
                                    provider="openai_realtime",
                                    reason="audio_buffer_too_small" if buffered_audio_ms < 100 else "no_speech",
                                )
                                await finish_response(reason="audio_buffer_too_small" if buffered_audio_ms < 100 else "no_speech")
                        except Exception as exc:
                            await send_event("error", message=f"Realtime commit failed: {safe_exc_message(exc)}")
                        continue
                    if msg_type in {"cancel", "cancel_response"}:
                        await self.cancel(str(doc.get("reason") or "client_cancel"))
                        await send_event("session_cancelled", session_id=session_id, reason=str(doc.get("reason") or "client_cancel"))
                        continue
                    if msg_type == "reset":
                        await self.cancel("reset")
                        transcript = ""
                        assistant_text = ""
                        text_chunker = RealtimeTextChunker()
                        tts_chunk_started = False
                        response_requested = False
                        response_done = False
                        stt_result_sent = False
                        llm_started_sent = False
                        speech_started = False
                        audio_ms = 0
                        buffered_audio_ms = 0
                        input_committed = False
                        transcript_event = asyncio.Event()
                        if response_wait_task is not None and not response_wait_task.done():
                            response_wait_task.cancel()
                        response_wait_task = None
                        await send_event("session_reset", session_id=session_id)
                        continue
                    await send_event("error", message=f"Unknown realtime message type: {msg_type}")
                    continue
                chunk = message.get("bytes")
                if chunk is None:
                    continue
                if realtime_ws is None:
                    if not await open_realtime():
                        continue
                    await send_event("session_started", session_id=session_id, sample_rate=source_sample_rate, realtime_enabled=True)
                raw, stripped_header = normalize_esp_pcm_packet(bytes(chunk))
                if stripped_header is not None:
                    stripped_packet_headers += 1
                    if stripped_packet_headers <= 3:
                        await self._log_bus.emit(
                            "INFO",
                            "PIPELINE",
                            "OpenAI Realtime mic packet header stripped",
                            {"session_id": session_id, "packet_len": len(chunk), "header": stripped_header},
                        )
                if not raw:
                    continue
                chunk_ms = int((len(raw) / 2) / max(1, source_sample_rate) * 1000)
                audio_ms += chunk_ms
                buffered_audio_ms += chunk_ms
                target = pcm16le_resample_linear(raw, source_sample_rate, target_sample_rate)
                await send_realtime_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(target).decode("ascii")})
        except WebSocketDisconnect:
            await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime client disconnected", {"session_id": session_id})
        except Exception as exc:
            self._last_error = safe_exc_message(exc)
            await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime session failed", {"session_id": session_id, "error": safe_exc_message(exc)})
            try:
                await websocket.send_json({"type": "error", "message": safe_exc_message(exc)})
            except Exception:
                pass
        finally:
            self._active = False
            self._connected = False
            self._last_event = "closed"
            self._cancel_event.set()
            if realtime_ws is not None and not realtime_ws.closed:
                try:
                    await realtime_ws.close()
                except Exception:
                    pass
            if reader_task is not None and not reader_task.done():
                reader_task.cancel()
            if client_session is not None:
                try:
                    await client_session.close()
                except Exception:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass
            await self._ws_hub.publish("pipeline_status", {"realtime": await self.status()})

    async def cancel(self, reason: str = "manual_cancel") -> None:
        self._cancel_event.set()
        await self._cancel_playback(reason)

    async def _cancel_playback(self, reason: str) -> None:
        try:
            await self._esp_client.send_audio_error(f"cancelled: {reason}")
        except Exception:
            pass

    async def _session_update_payload(self, config: dict[str, Any], realtime: dict[str, Any], sample_rate: int, language: str) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": sample_rate},
            "turn_detection": self._turn_detection(realtime),
        }
        noise = str(realtime.get("noise_reduction") or "near_field").strip().lower()
        if noise in {"near_field", "far_field"}:
            audio_input["noise_reduction"] = {"type": noise}
        elif noise in {"none", "off", "disabled", "kapali"}:
            audio_input["noise_reduction"] = None
        transcription_model = str(realtime.get("transcription_model") or "").strip()
        if transcription_model:
            audio_input["transcription"] = {"model": transcription_model, "language": language}
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": str(realtime.get("model") or "gpt-realtime-mini"),
                "instructions": await self._instructions(config, realtime),
                "audio": {"input": audio_input},
            },
        }

    async def _instructions(self, config: dict[str, Any], realtime: dict[str, Any]) -> str:
        text = str(realtime.get("instructions") or "").strip()
        if text:
            return text
        llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
        text = str(llm.get("system_prompt") or "").strip()
        if text:
            return text
        return await self._prompt_store.active_prompt_text()

    def _turn_detection(self, realtime: dict[str, Any]) -> dict[str, Any]:
        mode = str(realtime.get("turn_detection") or "server_vad").strip().lower()
        if mode == "semantic_vad":
            eagerness = str(realtime.get("semantic_eagerness") or "high").strip().lower()
            if eagerness not in {"low", "medium", "high", "auto"}:
                eagerness = "high"
            return {"type": "semantic_vad", "eagerness": eagerness, "create_response": False, "interrupt_response": True}
        return {
            "type": "server_vad",
            "threshold": max(0.0, min(1.0, float(realtime.get("vad_threshold") or 0.5))),
            "prefix_padding_ms": max(0, int(realtime.get("prefix_padding_ms") or 300)),
            "silence_duration_ms": max(120, int(realtime.get("silence_duration_ms") or 420)),
            "create_response": False,
            "interrupt_response": True,
        }

    @staticmethod
    def _realtime_cfg(config: dict[str, Any]) -> dict[str, Any]:
        return active_realtime_config(config)

    @staticmethod
    def _api_key(config: dict[str, Any], realtime: dict[str, Any]) -> str:
        value = str(realtime.get("api_key") or "").strip()
        if value:
            return value
        llm = config.get("llm", {}) if isinstance(config, dict) and isinstance(config.get("llm"), dict) else {}
        providers = llm.get("providers", {}) if isinstance(llm.get("providers"), dict) else {}
        openai = providers.get("openai", {}) if isinstance(providers.get("openai"), dict) else {}
        value = str(openai.get("api_key") or "").strip()
        if value:
            return value
        active_llm = active_llm_config(config)
        if str(active_llm.get("provider") or "").lower() == "openai":
            return str(active_llm.get("api_key") or "").strip()
        return ""
