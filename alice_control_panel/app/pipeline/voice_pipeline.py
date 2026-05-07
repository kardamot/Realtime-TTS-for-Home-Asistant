from __future__ import annotations

import time
from typing import Any

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.ws_hub import WsHub
from app.esp.esp_client import EspClient
from app.pipeline.llm.openai_compatible import OpenAICompatibleLlm
from app.pipeline.stt.manager import SttManager
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
    ) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._llm = llm
        self._stt = stt
        self._tts_relay = tts_relay
        self._esp = esp
        self._state = "IDLE"
        self._last_user_text = ""
        self._stt_result = ""
        self._llm_response = ""
        self._tts_status = "idle"
        self._stream_active = False
        self._last_audio_capture: dict[str, Any] = {}
        self._timeline: list[dict[str, Any]] = []

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

    async def run_text(self, text: str) -> dict[str, Any]:
        self._last_user_text = text
        self._stt_result = text
        self._llm_response = ""
        self._state = "LLM"
        self._stream_active = True
        self._mark("STT", "text input accepted")
        await self._ws_hub.publish("pipeline_status", await self.status())
        try:
            async for chunk in self._llm.stream_chat(text):
                self._llm_response += chunk
                await self._ws_hub.publish("llm_delta", {"text": chunk})
            self._state = "TTS"
            self._tts_status = "queued"
            self._mark("LLM", "response completed")
            await self._log_bus.emit("INFO", "PIPELINE", "LLM response completed", {"chars": len(self._llm_response)})
            config = await self._config_store.get(include_secrets=False)
            await self._stream_tts_to_esp(self._llm_response, config)
            self._state = "IDLE"
        except Exception as exc:
            self._state = "ERROR"
            self._tts_status = "error"
            await self._log_bus.emit("ERROR", "PIPELINE", "Pipeline text run failed", {"error": str(exc)})
            raise
        finally:
            self._stream_active = False
            await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def run_tts_text(self, text: str) -> dict[str, Any]:
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
            await self._stream_tts_to_esp(text, config)
            self._state = "IDLE"
        except Exception as exc:
            self._state = "ERROR"
            self._tts_status = "error"
            await self._log_bus.emit("ERROR", "TTS", "Direct TTS test failed", {"error": str(exc)})
            raise
        finally:
            self._stream_active = False
            await self._ws_hub.publish("pipeline_status", await self.status())
        return await self.status()

    async def run_audio_capture(self, metadata: dict[str, Any], audio: bytes) -> dict[str, Any]:
        sample_rate = int(metadata.get("sample_rate") or 16000)
        self._state = "STT"
        self._stream_active = True
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
                config = await self._config_store.get(include_secrets=False)
                mode = str(config.get("pipeline", {}).get("mic_response_mode") or "assistant").lower()
                if mode == "echo":
                    await self._echo_transcript(text, config)
                elif mode == "echo_then_assistant":
                    await self._echo_transcript(text, config)
                    self._state = "LLM"
                    await self.run_text(text)
                else:
                    self._state = "LLM"
                    await self.run_text(text)
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

    async def _echo_transcript(self, text: str, config: dict[str, Any]) -> None:
        self._state = "TTS"
        self._llm_response = text
        self._tts_status = "queued"
        self._mark("STT", "transcript echo queued")
        await self._log_bus.emit("INFO", "PIPELINE", "STT transcript echo queued", {"text": text})
        await self._ws_hub.publish("pipeline_status", await self.status())
        await self._stream_tts_to_esp(text, config)
        self._state = "IDLE"

    async def _stream_tts_to_esp(self, text: str, config: dict[str, Any]) -> None:
        if not config.get("pipeline", {}).get("stream_to_esp", True):
            self._tts_status = "stream_to_esp_disabled"
            await self._log_bus.emit("INFO", "TTS", "TTS stream-to-ESP is disabled")
            return
        result = await self._tts_relay.synthesize_to_esp(text, self._esp)
        self._tts_status = str(result.get("status") or "unknown")
        if result.get("ok"):
            self._mark("TTS", f"streamed {result.get('bytes', 0)} bytes to ESP")
            return
        self._mark("TTS", str(result.get("message") or self._tts_status))
        await self._log_bus.emit("WARN", "TTS", "TTS audio stream skipped", result)

    def _mark(self, category: str, message: str) -> None:
        self._timeline.append({"ts": time.time(), "category": category, "message": message})
        self._timeline = self._timeline[-50:]
