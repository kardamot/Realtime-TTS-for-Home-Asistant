from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.ws_hub import WsHub
from app.esp.esp_client import EspClient
from app.pipeline.llm.openai_compatible import OpenAICompatibleLlm
from app.pipeline.realtime.openai_realtime import OpenAIRealtimeBridge
from app.pipeline.stt.manager import SttManager
from app.pipeline.stt.vad import SileroVadRuntime
from app.pipeline.tts.relay import TtsRelay


class VoicePipeline:
    def __init__(
        self,
        config_store: ConfigStore,
        log_bus: LogBus,
        ws_hub: WsHub,
        llm: OpenAICompatibleLlm,
        stt: SttManager,
        tts_relay: TtsRelay,
        esp: EspClient,
        ha_bridge: Any | None = None,
        realtime_bridge: OpenAIRealtimeBridge | None = None,
    ) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._llm = llm
        self._stt = stt
        self._tts_relay = tts_relay
        self._esp = esp
        self._ha_bridge = ha_bridge
        self._realtime_bridge = realtime_bridge
        self._state = "IDLE"
        self._last_user_text = ""
        self._stt_result = ""
        self._llm_response = ""
        self._tts_status = "idle"
        self._stream_active = False
        self._last_audio_capture: dict[str, Any] = {}
        self._timeline: list[dict[str, Any]] = []
        self._cancel_event = asyncio.Event()
        self._session_active = False
        self._session_id = ""
        self._session_started_at: float | None = None
        self._session_mode = "manual"
        self._session_turns = 0
        self._session_last_event = "idle"
        self._live_clients = 0
        self._last_live_mic: dict[str, Any] = {}

    async def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "last_user_text": self._last_user_text,
            "stt_result": self._stt_result,
            "llm_response": self._llm_response,
            "tts_status": self._tts_status,
            "stream_active": self._stream_active,
            "last_audio_capture": self._last_audio_capture,
            "timeline": self._timeline[-20:],
            "session": {
                "active": self._session_active,
                "session_id": self._session_id,
                "mode": self._session_mode,
                "turns": self._session_turns,
                "last_event": self._session_last_event,
                "started_at": self._session_started_at,
                "uptime_sec": int(time.time() - self._session_started_at) if self._session_started_at else 0,
                "cancel_requested": self._cancel_event.is_set(),
            },
            "live_mic": {
                "clients": self._live_clients,
                "last": self._last_live_mic,
            },
            "realtime": await self._realtime_bridge.status() if self._realtime_bridge else {},
        }

    async def restart_tts(self) -> dict[str, Any]:
        self._tts_status = "restarted"
        self._mark("TTS", "restart requested")
        await self._log_bus.emit("INFO", "TTS", "TTS restart requested")
        return await self.status()

    async def reload_prompt(self) -> dict[str, Any]:
        self._mark("PROMPT", "reload requested")
        await self._log_bus.emit("INFO", "PIPELINE", "Prompt reload requested")
        return await self.status()

    async def start_session(self, mode: str = "manual") -> dict[str, Any]:
        if self._session_active:
            return await self.status()
        self._session_active = True
        self._session_id = f"session-{uuid.uuid4().hex[:10]}"
        self._session_started_at = time.time()
        self._session_mode = mode or "manual"
        self._session_turns = 0
        self._session_last_event = "started"
        self._mark("SESSION", "voice session started")
        await self._log_bus.emit("INFO", "PIPELINE", "Voice session started", {"session_id": self._session_id, "mode": self._session_mode})
        await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def stop_session(self, reason: str = "manual_stop") -> dict[str, Any]:
        if self._stream_active:
            await self.cancel_response(reason)
        self._session_active = False
        self._session_last_event = "stopped"
        self._mark("SESSION", "voice session stopped")
        await self._log_bus.emit("INFO", "PIPELINE", "Voice session stopped", {"session_id": self._session_id, "reason": reason})
        await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def cancel_response(self, reason: str = "manual_cancel") -> dict[str, Any]:
        self._cancel_event.set()
        self._tts_status = "cancelled"
        self._session_last_event = "cancel_requested"
        self._mark("SESSION", f"response cancel requested: {reason}")
        await self._log_bus.emit("WARN", "PIPELINE", "Response cancel requested", {"reason": reason})
        if self._realtime_bridge:
            await self._realtime_bridge.cancel(reason)
        if self._stream_active or self._state in {"LLM", "TTS"}:
            try:
                await self._esp.send_audio_error(f"cancelled: {reason}")
            except Exception:
                pass
        await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def live_mic_websocket(self, websocket: WebSocket) -> None:
        if self._realtime_bridge and await self._realtime_bridge.should_handle_voice_ws():
            await self._realtime_bridge.websocket_session(websocket)
            return
        await websocket.accept()
        self._live_clients += 1
        session_id = f"live-{uuid.uuid4().hex[:10]}"
        await self._log_bus.emit("INFO", "STT", "Live mic WebSocket connected", {"session_id": session_id})

        config = await self._config_store.get(include_secrets=False)
        pipeline_cfg = config.get("pipeline", {}) if isinstance(config.get("pipeline"), dict) else {}
        realtime_cfg = config.get("realtime", {}) if isinstance(config.get("realtime"), dict) else {}
        ha_bridge_cfg = config.get("ha_bridge", {}) if isinstance(config.get("ha_bridge"), dict) else {}
        tts_cfg = config.get("tts", {}) if isinstance(config.get("tts"), dict) else {}
        await websocket.send_json(
            {
                "type": "hello",
                "service": "alice_control_panel",
                "version": "0.1.37",
                "session_id": session_id,
                "endpointing_enabled": True,
                "endpointing_provider": str(pipeline_cfg.get("live_vad_provider") or "silero"),
                "realtime_enabled": bool(realtime_cfg.get("enabled", False)),
                "realtime_provider": str(realtime_cfg.get("provider") or "openai"),
                "ha_bridge_enabled": bool(ha_bridge_cfg.get("enabled", True)),
                "llm_enabled": True,
                "tts_enabled": bool(tts_cfg.get("enabled", True)),
            }
        )
        await websocket.send_json({"type": "ready", "session_id": session_id})
        if not bool(pipeline_cfg.get("live_mic_enabled", True)):
            await websocket.send_json({"type": "error", "message": "Live mic is disabled in config."})
            await websocket.close(code=1008)
            self._live_clients = max(0, self._live_clients - 1)
            return

        sample_rate = 16000
        channels = 1
        encoding = "pcm_s16le"
        vad_enabled = bool(pipeline_cfg.get("live_vad_enabled", True))
        endpoint = await self._create_live_endpoint(pipeline_cfg, sample_rate)
        await self.start_session("live_ws")
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect
                if message.get("text") is not None:
                    doc = json.loads(str(message["text"]))
                    msg_type = str(doc.get("type") or "").lower()
                    if msg_type == "start":
                        sample_rate = int(doc.get("sample_rate") or sample_rate)
                        channels = int(doc.get("channels") or channels)
                        encoding = str(doc.get("encoding") or encoding)
                        vad_enabled = bool(doc.get("vad_enabled", vad_enabled))
                        endpoint = await self._create_live_endpoint(pipeline_cfg, sample_rate)
                        self._last_live_mic = {
                            "session_id": session_id,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "encoding": encoding,
                            "vad_enabled": vad_enabled,
                            "started_at": time.time(),
                            "state": "listening",
                        }
                        await websocket.send_json({"type": "started", "session_id": session_id})
                        await self._ws_hub.publish("pipeline_status", await self.status())
                        continue
                    if msg_type in {"end", "eos"}:
                        await self._finalize_live_mic(websocket, endpoint, session_id, sample_rate, channels, encoding, "manual_end")
                        endpoint.reset()
                        continue
                    if msg_type in {"cancel", "cancel_response"}:
                        endpoint.reset()
                        await self.cancel_response(str(doc.get("reason") or "live_mic_cancel"))
                        await websocket.send_json({"type": "cancelled", "session_id": session_id})
                        continue
                    if msg_type == "reset":
                        endpoint.reset()
                        await self.cancel_response("live_mic_reset")
                        self._stt_result = ""
                        self._llm_response = ""
                        self._last_audio_capture = {}
                        await websocket.send_json({"type": "session_reset", "session_id": session_id})
                        await self._ws_hub.publish("pipeline_status", await self.status())
                        continue
                    if msg_type == "conversation_reset":
                        self._last_audio_capture.pop("ha_conversation", None)
                        await websocket.send_json({"type": "conversation_reset_done", "session_id": session_id})
                        continue
                    if await self._handle_ha_ws_message(websocket, doc):
                        continue
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                    continue
                chunk = message.get("bytes")
                if chunk is None:
                    continue
                result = endpoint.feed(bytes(chunk), use_vad=vad_enabled)
                self._last_live_mic = {
                    "session_id": session_id,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "encoding": encoding,
                    **endpoint.status(),
                }
                if result.get("speech_started"):
                    self._session_last_event = "vad_start"
                    if bool(pipeline_cfg.get("barge_in_enabled", True)):
                        await self.cancel_response("live_mic_barge_in")
                    await websocket.send_json({"type": "vad_start", "session_id": session_id, **endpoint.status()})
                if result.get("final"):
                    await websocket.send_json({"type": "vad_end", "session_id": session_id, **endpoint.status()})
                    await self._finalize_live_mic(
                        websocket,
                        endpoint,
                        session_id,
                        sample_rate,
                        channels,
                        encoding,
                        str(result.get("reason") or "vad_end"),
                    )
                    endpoint.reset(keep_pre_roll=True)
        except WebSocketDisconnect:
            await self._log_bus.emit("INFO", "STT", "Live mic WebSocket disconnected", {"session_id": session_id})
        except Exception as exc:
            await self._log_bus.emit("ERROR", "STT", "Live mic WebSocket failed", {"session_id": session_id, "error": str(exc)})
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            self._live_clients = max(0, self._live_clients - 1)
            self._last_live_mic = {**self._last_live_mic, "state": "disconnected", "ended_at": time.time()}
            try:
                await websocket.close()
            except Exception:
                pass
            await self._ws_hub.publish("pipeline_status", await self.status())

    async def run_text(self, text: str, cancel_current: bool = True) -> dict[str, Any]:
        if cancel_current and self._stream_active and not self._cancel_event.is_set():
            await self.cancel_response("new_text")
        run_cancel_event = asyncio.Event()
        self._cancel_event = run_cancel_event
        self._last_user_text = text
        self._stt_result = text
        self._llm_response = ""
        self._state = "LLM"
        self._stream_active = True
        self._mark("STT", "text input accepted")
        await self._ws_hub.publish("pipeline_status", await self.status())
        try:
            config = await self._config_store.get(include_secrets=False)
            ha_response = await self._try_home_assistant_route(text, run_cancel_event)
            if ha_response is not None:
                self._llm_response = ha_response
                self._state = "TTS"
                self._tts_status = "queued"
                self._mark("HA", "Home Assistant conversation completed")
                await self._log_bus.emit("INFO", "PIPELINE", "Home Assistant conversation completed", {"chars": len(ha_response)})
                await self._stream_tts_to_esp(ha_response, config, run_cancel_event)
                self._state = "IDLE" if not run_cancel_event.is_set() else "CANCELLED"
                return await self.status()
            async for chunk in self._llm.stream_chat(text):
                self._llm_response += chunk
                await self._ws_hub.publish("llm_delta", {"text": chunk})
            self._state = "TTS"
            self._tts_status = "queued"
            self._mark("LLM", "response completed")
            await self._log_bus.emit("INFO", "PIPELINE", "LLM response completed", {"chars": len(self._llm_response)})
            await self._stream_tts_to_esp(self._llm_response, config, run_cancel_event)
            self._state = "IDLE" if not run_cancel_event.is_set() else "CANCELLED"
        except Exception as exc:
            self._state = "ERROR"
            self._tts_status = "error"
            await self._log_bus.emit("ERROR", "PIPELINE", "Pipeline text run failed", {"error": str(exc)})
            raise
        finally:
            self._stream_active = False
            if self._session_active:
                self._session_last_event = "turn_completed" if not run_cancel_event.is_set() else "turn_cancelled"
            await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def run_tts_text(self, text: str) -> dict[str, Any]:
        if self._stream_active and not self._cancel_event.is_set():
            await self.cancel_response("new_tts_test")
        run_cancel_event = asyncio.Event()
        self._cancel_event = run_cancel_event
        self._last_user_text = text
        self._stt_result = text
        self._llm_response = text
        self._state = "TTS"
        self._stream_active = True
        self._tts_status = "queued"
        self._mark("TTS", "direct TTS test accepted")
        await self._ws_hub.publish("pipeline_status", await self.status())
        try:
            config = await self._config_store.get(include_secrets=False)
            await self._stream_tts_to_esp(text, config, run_cancel_event)
            self._state = "IDLE" if not run_cancel_event.is_set() else "CANCELLED"
        except Exception as exc:
            self._state = "ERROR"
            self._tts_status = "error"
            await self._log_bus.emit("ERROR", "TTS", "Direct TTS test failed", {"error": str(exc)})
            raise
        finally:
            self._stream_active = False
            if self._session_active:
                self._session_last_event = "tts_completed" if not run_cancel_event.is_set() else "tts_cancelled"
            await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def run_audio_capture(self, metadata: dict[str, Any], audio: bytes) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=False)
        if self._stream_active and bool(config.get("pipeline", {}).get("barge_in_enabled", True)):
            await self.cancel_response("mic_barge_in")
        sample_rate = int(metadata.get("sample_rate") or 16000)
        self._state = "STT"
        self._stream_active = True
        self._cancel_event = asyncio.Event()
        if self._session_active:
            self._session_turns += 1
            self._session_last_event = "mic_capture"
        self._last_audio_capture = {
            **metadata,
            "bytes_buffered": len(audio),
            "received_at": time.time(),
        }
        self._mark("STT", f"mic capture received: {len(audio)} bytes")
        await self._ws_hub.publish("pipeline_status", await self.status())
        try:
            result = await self._stt.transcribe_pcm16(audio, sample_rate)
            text = str(result.get("text") or "").strip()
            self._stt_result = text or str(result.get("message") or "Audio captured; no transcript yet.")
            self._last_user_text = text
            self._last_audio_capture["stt"] = result
            if result.get("ok") and text:
                mode = str(config.get("pipeline", {}).get("mic_response_mode") or "assistant").lower()
                if mode == "echo":
                    await self._echo_transcript(text, config, self._cancel_event)
                elif mode == "echo_then_assistant":
                    await self._echo_transcript(text, config, self._cancel_event)
                    self._state = "LLM"
                    await self.run_text(text, cancel_current=False)
                else:
                    self._state = "LLM"
                    await self.run_text(text, cancel_current=False)
            else:
                self._state = "IDLE"
                self._mark("STT", "capture stored; STT engine pending")
            return await self.status()
        except Exception as exc:
            self._state = "ERROR"
            self._last_audio_capture["error"] = str(exc)
            await self._log_bus.emit("ERROR", "STT", "Audio capture processing failed", {"error": str(exc)})
            raise
        finally:
            self._stream_active = False
            await self._ws_hub.publish("pipeline_status", await self.status())

    async def _echo_transcript(self, text: str, config: dict[str, Any], cancel_event: asyncio.Event) -> None:
        self._state = "TTS"
        self._llm_response = text
        self._tts_status = "queued"
        self._mark("STT", "transcript echo queued")
        await self._log_bus.emit("INFO", "PIPELINE", "STT transcript echo queued", {"text": text})
        await self._ws_hub.publish("pipeline_status", await self.status())
        await self._stream_tts_to_esp(text, config, cancel_event)
        self._state = "IDLE"

    async def _stream_tts_to_esp(self, text: str, config: dict[str, Any], cancel_event: asyncio.Event) -> None:
        if not config.get("pipeline", {}).get("stream_to_esp", True):
            self._tts_status = "stream_to_esp_disabled"
            await self._log_bus.emit("INFO", "TTS", "TTS stream-to-ESP is disabled")
            return
        result = await self._tts_relay.synthesize_to_esp(text, self._esp, cancel_event=cancel_event)
        self._tts_status = str(result.get("status") or "unknown")
        if result.get("ok"):
            self._mark("TTS", f"streamed {result.get('bytes', 0)} bytes to ESP")
            return
        self._mark("TTS", str(result.get("message") or self._tts_status))
        await self._log_bus.emit("WARN", "TTS", "TTS audio stream skipped", result)

    async def _try_home_assistant_route(self, text: str, cancel_event: asyncio.Event) -> str | None:
        if self._ha_bridge is None or cancel_event.is_set():
            return None
        try:
            if not await self._ha_bridge.should_route_home_control(text):
                return None
            self._mark("HA", "home control route selected")
            await self._log_bus.emit("INFO", "PIPELINE", "Home Assistant route selected", {"reason": "home_control_keywords"})
            result = await self._ha_bridge.conversation(text)
            speech = self._ha_bridge.extract_conversation_speech(result)
            self._last_audio_capture["ha_conversation"] = {
                "conversation_id": result.get("conversation_id"),
                "continue_conversation": result.get("continue_conversation"),
                "response_type": (result.get("response") or {}).get("response_type") if isinstance(result.get("response"), dict) else "",
            }
            return speech or "Home Assistant komutu islendi."
        except PermissionError as exc:
            await self._log_bus.emit("WARN", "PIPELINE", "Home Assistant route skipped", {"error": str(exc)})
            return None
        except Exception as exc:
            await self._log_bus.emit("ERROR", "PIPELINE", "Home Assistant route failed", {"error": str(exc)})
            return None

    async def _handle_ha_ws_message(self, websocket: WebSocket, doc: dict[str, Any]) -> bool:
        msg_type = str(doc.get("type") or "").strip().lower()
        if msg_type not in {"ha_get_state", "ha_list_states", "ha_search_entities", "ha_call_service", "ha_conversation"}:
            return False
        if self._ha_bridge is None:
            await websocket.send_json({"type": "ha_error", "ok": False, "error": "ha_bridge is not available"})
            return True
        try:
            if msg_type == "ha_get_state":
                entity_id = str(doc.get("entity_id") or "").strip()
                if not entity_id:
                    await websocket.send_json({"type": "ha_state_result", "ok": False, "error": "entity_id is required"})
                    return True
                entity = await self._ha_bridge.get_state(entity_id)
                await websocket.send_json({"type": "ha_state_result", "ok": entity is not None, "entity": entity, "entity_id": entity_id})
                return True
            if msg_type == "ha_list_states":
                entities = await self._ha_bridge.list_states(
                    domain=str(doc.get("domain") or ""),
                    limit=int(doc.get("limit") or 64),
                )
                await websocket.send_json({"type": "ha_list_states_result", "ok": True, "count": len(entities), "entities": entities})
                return True
            if msg_type == "ha_search_entities":
                entities = await self._ha_bridge.search_states(
                    query=str(doc.get("query") or doc.get("q") or ""),
                    domain=str(doc.get("domain") or ""),
                    limit=int(doc.get("limit") or 8),
                )
                await websocket.send_json({"type": "ha_search_entities_result", "ok": True, "count": len(entities), "entities": entities})
                return True
            if msg_type == "ha_call_service":
                data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
                domain = str(doc.get("domain") or "").strip().lower()
                service = str(doc.get("service") or "").strip()
                if not domain or not service:
                    await websocket.send_json({"type": "ha_service_result", "ok": False, "error": "domain and service are required"})
                    return True
                result = await self._ha_bridge.call_service(
                    domain=domain,
                    service=service,
                    data=data,
                )
                await websocket.send_json({"type": "ha_service_result", "ok": True, "result": result})
                return True
            result = await self._ha_bridge.conversation(
                str(doc.get("text") or ""),
                language=str(doc.get("language") or ""),
                conversation_id=str(doc.get("conversation_id") or ""),
            )
            await websocket.send_json(
                {
                    "type": "ha_conversation_result",
                    "ok": True,
                    "speech": self._ha_bridge.extract_conversation_speech(result),
                    "result": result,
                }
            )
            return True
        except Exception as exc:
            await websocket.send_json({"type": f"{msg_type}_result", "ok": False, "error": str(exc)})
            return True

    def _mark(self, category: str, message: str) -> None:
        self._timeline.append({"ts": time.time(), "category": category, "message": message})
        self._timeline = self._timeline[-50:]

    async def _finalize_live_mic(
        self,
        websocket: WebSocket,
        endpoint: "_EnergyEndpoint",
        session_id: str,
        sample_rate: int,
        channels: int,
        encoding: str,
        reason: str,
    ) -> None:
        audio = endpoint.audio()
        if not audio:
            await websocket.send_json({"type": "empty", "session_id": session_id, "reason": reason})
            return
        metadata = {
            "stream_id": session_id,
            "source": "live_ws",
            "sample_rate": sample_rate,
            "channels": channels,
            "encoding": encoding,
            "reason": reason,
            **endpoint.status(),
        }
        await websocket.send_json(
            {
                "type": "utterance_finalized",
                "session_id": session_id,
                "bytes": len(audio),
                "reason": reason,
            }
        )
        status = await self.run_audio_capture(metadata, audio)
        await websocket.send_json({"type": "pipeline_status", "session_id": session_id, "payload": status})

    async def _create_live_endpoint(self, pipeline_cfg: dict[str, Any], sample_rate: int) -> "_EnergyEndpoint":
        provider = str(pipeline_cfg.get("live_vad_provider") or "silero").strip().lower()
        if provider == "silero":
            try:
                endpoint = _SileroEndpoint(pipeline_cfg, sample_rate)
                await self._log_bus.emit(
                    "INFO",
                    "STT",
                    "Live mic VAD ready",
                    {
                        "provider": "silero",
                        "sample_rate": sample_rate,
                        "frame_ms": endpoint.frame_ms,
                        "start_prob": endpoint.start_probability,
                        "end_prob": endpoint.end_probability,
                    },
                )
                return endpoint
            except Exception as exc:
                await self._log_bus.emit(
                    "WARN",
                    "STT",
                    "Silero VAD unavailable; falling back to energy endpointing",
                    {"error": str(exc), "sample_rate": sample_rate},
                )
        return _EnergyEndpoint(pipeline_cfg, sample_rate)


class _EnergyEndpoint:
    def __init__(self, cfg: dict[str, Any], sample_rate: int) -> None:
        self.provider = "energy"
        self.sample_rate = max(1, int(sample_rate or 16000))
        self.start_rms = max(1, int(cfg.get("live_vad_start_rms") or 450))
        self.end_rms = max(1, int(cfg.get("live_vad_end_rms") or 260))
        self.min_speech_ms = max(0, int(cfg.get("live_vad_min_speech_ms") or 120))
        self.end_silence_ms = max(80, int(cfg.get("live_vad_end_silence_ms") or 650))
        self.pre_roll_ms = max(0, int(cfg.get("live_vad_pre_roll_ms") or 300))
        self.max_utterance_ms = max(500, int(cfg.get("live_vad_max_utterance_ms") or 12000))
        self.max_buffer_bytes = int(max(1, float(cfg.get("live_vad_max_buffer_sec") or 20)) * self.sample_rate * 2)
        self._pre_roll = bytearray()
        self._audio = bytearray()
        self._speaking = False
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._received_ms = 0.0
        self._last_rms = 0
        self._last_peak = 0
        self._chunks = 0
        self._started_at: float | None = None

    def feed(self, chunk: bytes, use_vad: bool = True) -> dict[str, Any]:
        duration_ms = self._duration_ms(chunk)
        rms, peak = self._levels(chunk)
        self._last_rms = rms
        self._last_peak = peak
        self._chunks += 1
        self._received_ms += duration_ms

        if not use_vad:
            if not self._speaking:
                self._start_speech()
            self._append_audio(chunk)
            return {"speech_started": self._chunks == 1, "final": False}

        speech_started = False
        if not self._speaking:
            self._append_pre_roll(chunk)
            if rms >= self.start_rms:
                self._voice_ms += duration_ms
            else:
                self._voice_ms = 0.0
            if self._voice_ms >= self.min_speech_ms:
                self._start_speech()
                self._audio.extend(self._pre_roll)
                self._pre_roll.clear()
                speech_started = True
            return {"speech_started": speech_started, "final": False}

        self._append_audio(chunk)
        self._speech_ms += duration_ms
        if rms <= self.end_rms:
            self._silence_ms += duration_ms
        else:
            self._silence_ms = 0.0
        if self._speech_ms >= self.max_utterance_ms:
            return {"speech_started": False, "final": True, "reason": "max_utterance"}
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.end_silence_ms:
            return {"speech_started": False, "final": True, "reason": "silence"}
        if len(self._audio) >= self.max_buffer_bytes:
            return {"speech_started": False, "final": True, "reason": "max_buffer"}
        return {"speech_started": False, "final": False}

    def audio(self) -> bytes:
        if self._audio:
            return bytes(self._audio)
        return bytes(self._pre_roll)

    def status(self) -> dict[str, Any]:
        return {
            "vad_provider": self.provider,
            "state": "speaking" if self._speaking else "listening",
            "rms": self._last_rms,
            "peak": self._last_peak,
            "chunks": self._chunks,
            "received_ms": int(self._received_ms),
            "speech_ms": int(self._speech_ms),
            "silence_ms": int(self._silence_ms),
            "bytes_buffered": len(self._audio) or len(self._pre_roll),
        }

    def reset(self, keep_pre_roll: bool = False) -> None:
        old_pre_roll = bytes(self._pre_roll[-self._pre_roll_bytes :]) if keep_pre_roll else b""
        self._pre_roll = bytearray(old_pre_roll)
        self._audio = bytearray()
        self._speaking = False
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._started_at = None

    def _start_speech(self) -> None:
        self._speaking = True
        self._started_at = time.time()
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def _append_audio(self, chunk: bytes) -> None:
        if len(self._audio) < self.max_buffer_bytes:
            remaining = self.max_buffer_bytes - len(self._audio)
            self._audio.extend(chunk[:remaining])

    def _append_pre_roll(self, chunk: bytes) -> None:
        self._pre_roll.extend(chunk)
        if len(self._pre_roll) > self._pre_roll_bytes:
            del self._pre_roll[: len(self._pre_roll) - self._pre_roll_bytes]

    @property
    def _pre_roll_bytes(self) -> int:
        return int(self.sample_rate * 2 * self.pre_roll_ms / 1000)

    def _duration_ms(self, chunk: bytes) -> float:
        return (len(chunk) // 2) / self.sample_rate * 1000.0

    @staticmethod
    def _levels(chunk: bytes) -> tuple[int, int]:
        even_len = len(chunk) & ~1
        if even_len <= 0:
            return 0, 0
        samples = memoryview(chunk[:even_len]).cast("h")
        total_sq = 0
        peak = 0
        for sample in samples:
            value = int(sample)
            abs_value = abs(value)
            if abs_value > peak:
                peak = abs_value
            total_sq += value * value
        return int((total_sq / len(samples)) ** 0.5), peak


class _SileroEndpoint(_EnergyEndpoint):
    def __init__(self, cfg: dict[str, Any], sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        self.provider = "silero"
        self.start_probability = float(cfg.get("live_vad_silero_start_prob") or 0.50)
        self.end_probability = float(cfg.get("live_vad_silero_end_prob") or 0.28)
        self._vad = SileroVadRuntime(self.sample_rate)
        self.frame_ms = self._vad.frame_ms
        self._last_probability = 0.0

    def feed(self, chunk: bytes, use_vad: bool = True) -> dict[str, Any]:
        if not use_vad:
            return super().feed(chunk, use_vad=False)

        duration_ms = self._duration_ms(chunk)
        rms, peak = self._levels(chunk)
        self._last_rms = rms
        self._last_peak = peak
        self._chunks += 1
        self._received_ms += duration_ms

        probabilities = self._vad.push_pcm16le(chunk)
        speech_started = False

        if not self._speaking:
            self._append_pre_roll(chunk)
            for probability in probabilities:
                self._last_probability = probability
                if probability >= self.start_probability:
                    self._voice_ms += self.frame_ms
                else:
                    self._voice_ms = 0.0
                if self._voice_ms >= self.min_speech_ms:
                    self._start_speech()
                    self._audio.extend(self._pre_roll)
                    self._pre_roll.clear()
                    speech_started = True
                    break
            return {"speech_started": speech_started, "final": False}

        self._append_audio(chunk)
        self._speech_ms += duration_ms
        for probability in probabilities:
            self._last_probability = probability
            if probability >= self.end_probability:
                self._silence_ms = 0.0
            else:
                self._silence_ms += self.frame_ms

        if self._speech_ms >= self.max_utterance_ms:
            return {"speech_started": False, "final": True, "reason": "max_utterance"}
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.end_silence_ms:
            return {"speech_started": False, "final": True, "reason": "silence"}
        if len(self._audio) >= self.max_buffer_bytes:
            return {"speech_started": False, "final": True, "reason": "max_buffer"}
        return {"speech_started": False, "final": False}

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "probability": round(self._last_probability, 4),
            "start_probability": self.start_probability,
            "end_probability": self.end_probability,
            "frame_ms": self.frame_ms,
        }

    def reset(self, keep_pre_roll: bool = False) -> None:
        super().reset(keep_pre_roll=keep_pre_roll)
        self._last_probability = 0.0
        if not keep_pre_roll:
            self._vad.reset_state()
