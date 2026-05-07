from __future__ import annotations

import asyncio
import base64
import json
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


def safe_exc_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()


def realtime_ws_url(cfg: dict[str, Any]) -> str:
    base = str(cfg.get("ws_url") or OPENAI_REALTIME_WS_URL).strip().rstrip("/")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}model={str(cfg.get('model') or 'gpt-realtime-mini')}"


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


class OpenAIRealtimeBridge:
    def __init__(
        self,
        config_store: ConfigStore,
        prompt_store: PromptStore,
        log_bus: LogBus,
        ws_hub: WsHub,
        tts_relay: Any,
        esp_client: Any,
    ) -> None:
        self._config_store = config_store
        self._prompt_store = prompt_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._tts_relay = tts_relay
        self._esp_client = esp_client
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
        llm = active_llm_config(config)
        return (
            bool(realtime.get("enabled", False))
            and str(realtime.get("provider") or "openai").lower() == "openai"
            and bool(str(llm.get("api_key") or "").strip())
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
        llm = active_llm_config(config)
        api_key = str(llm.get("api_key") or "").strip()
        source_sample_rate = 16000
        target_sample_rate = max(8000, int(realtime.get("input_sample_rate") or 24000))
        language = str(config.get("stt", {}).get("language") or "tr") if isinstance(config.get("stt"), dict) else "tr"
        session_id = f"rt-{uuid.uuid4().hex[:10]}"
        transcript = ""
        assistant_text = ""
        response_requested = False
        response_done = False
        speech_started = False
        audio_ms = 0
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

        async def finish_response(doc: dict[str, Any] | None = None, reason: str = "openai_realtime_done") -> None:
            nonlocal response_done, assistant_text
            if response_done:
                return
            response_done = True
            final_text = extract_realtime_response_text(doc or {})
            if final_text and not assistant_text.strip():
                assistant_text = final_text
                await send_event("llm_chunk", text=assistant_text, final=True)
            else:
                await send_event("llm_chunk", text="", final=True)
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
            if assistant_text.strip() and not self._cancel_event.is_set():
                self._last_event = "tts"
                await self._tts_relay.synthesize_to_esp(assistant_text, self._esp_client, self._cancel_event)

        async def handle_realtime_event(doc: dict[str, Any]) -> None:
            nonlocal transcript, assistant_text, response_done, response_requested, speech_started
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
                wait_ms = max(0, int(realtime.get("transcript_wait_ms") or 800))
                if wait_ms:
                    await asyncio.sleep(wait_ms / 1000)
                if bool(realtime.get("suppress_empty_transcript_response", True)) and not transcript.strip():
                    await send_event("stt_result", text="", provider="openai_realtime", reason="empty_transcript_suppressed")
                    await finish_response(reason="empty_transcript_suppressed")
                    return
                await send_event("stt_result", text=transcript.strip(), provider="openai_realtime", reason="realtime_committed")
                await request_response()
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
                await send_event("stt_result", text=transcript.strip(), provider="openai_realtime")
                return
            if event_type in {"response.output_text.delta", "response.text.delta", "response.audio_transcript.delta"}:
                delta = extract_realtime_text_delta(doc)
                if delta:
                    assistant_text += delta
                    await send_event("llm_chunk", text=delta, final=False)
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
                version="0.1.38",
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
                        try:
                            if speech_started:
                                await send_realtime_json({"type": "input_audio_buffer.commit"})
                                await request_response()
                            else:
                                await send_realtime_json({"type": "input_audio_buffer.commit"})
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
                        response_requested = False
                        response_done = False
                        speech_started = False
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
                raw = bytes(chunk)
                audio_ms += int((len(raw) / 2) / max(1, source_sample_rate) * 1000)
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
        realtime = config.get("realtime", {}) if isinstance(config, dict) else {}
        return realtime if isinstance(realtime, dict) else {}
