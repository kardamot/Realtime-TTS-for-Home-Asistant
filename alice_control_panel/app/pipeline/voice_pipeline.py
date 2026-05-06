from __future__ import annotations

import time
from typing import Any

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.ws_hub import WsHub
from app.esp.esp_client import EspClient
from app.pipeline.llm.openai_compatible import OpenAICompatibleLlm


class VoicePipeline:
    def __init__(
        self,
        config_store: ConfigStore,
        log_bus: LogBus,
        ws_hub: WsHub,
        llm: OpenAICompatibleLlm,
        esp: EspClient,
    ) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._llm = llm
        self._esp = esp
        self._state = "IDLE"
        self._last_user_text = ""
        self._stt_result = ""
        self._llm_response = ""
        self._tts_status = "idle"
        self._stream_active = False
        self._timeline: list[dict[str, Any]] = []

    async def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "last_user_text": self._last_user_text,
            "stt_result": self._stt_result,
            "llm_response": self._llm_response,
            "tts_status": self._tts_status,
            "stream_active": self._stream_active,
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
            esp_status = await self._esp.status()
            if not esp_status.get("online"):
                self._tts_status = "mock_skipped_esp_offline"
                await self._log_bus.emit("WARN", "TTS", "ESP offline; TTS audio stream skipped")
            else:
                self._tts_status = "todo_stream_to_esp"
                await self._log_bus.emit("WARN", "TTS", "ESP TTS WebSocket streaming is a planned integration point")
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

    def _mark(self, category: str, message: str) -> None:
        self._timeline.append({"ts": time.time(), "category": category, "message": message})
        self._timeline = self._timeline[-50:]

