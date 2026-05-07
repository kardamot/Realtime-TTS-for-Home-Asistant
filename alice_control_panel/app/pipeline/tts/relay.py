from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus


OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
DEFAULT_PCM_SAMPLE_RATE = 44100
DEFAULT_PCM_CHANNELS = 1
PCM_PACE_INITIAL_BURST_MS = 700
PCM_PACE_MAX_SLEEP = 0.05
RELAY_CHUNK_BYTES = 4096
API_KEY_QUERY_RE = re.compile(r"(api_key=)[^&\s]+")


@dataclass(slots=True)
class TtsRelayConfig:
    enabled: bool = True
    provider: str = "openai"
    pcm_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE
    esp_initial_buffer_ms: int = 900
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "coral"
    openai_instructions: str = ""
    cartesia_api_key: str = ""
    cartesia_model_id: str = "sonic-3"
    cartesia_voice_id: str = ""
    cartesia_language: str = "tr"
    cartesia_version: str = "2026-03-01"


@dataclass(slots=True)
class StreamCommand:
    msg_type: str
    text: str
    final: bool
    provider: str


class PcmOutput:
    pace_pcm = True

    async def start(self, sample_rate: int, channels: int = DEFAULT_PCM_CHANNELS) -> None:
        raise NotImplementedError

    async def write(self, pcm: bytes) -> None:
        raise NotImplementedError

    async def done(self) -> None:
        raise NotImplementedError

    async def error(self, message: str, status: int = 500) -> None:
        raise NotImplementedError


class WebSocketPcmOutput(PcmOutput):
    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def start(self, sample_rate: int, channels: int = DEFAULT_PCM_CHANNELS) -> None:
        await send_pcm_start(self._ws, sample_rate, channels)

    async def write(self, pcm: bytes) -> None:
        await self._ws.send_bytes(pcm)

    async def done(self) -> None:
        await send_done(self._ws)

    async def error(self, message: str, status: int = 500) -> None:
        await send_error(self._ws, message, status)


class EspPcmOutput(PcmOutput):
    pace_pcm = True

    def __init__(self, esp_client: Any, log_bus: LogBus, initial_buffer_ms: int = 900) -> None:
        self._esp_client = esp_client
        self._log_bus = log_bus
        self._sample_rate = DEFAULT_PCM_SAMPLE_RATE
        self._channels = DEFAULT_PCM_CHANNELS
        self._started = False
        self._buffer = bytearray()
        self._initial_buffer_ms = max(0, int(initial_buffer_ms))
        self.bytes_sent = 0
        self.failed = False
        self.error_message = ""

    async def start(self, sample_rate: int, channels: int = DEFAULT_PCM_CHANNELS) -> None:
        self._sample_rate = sample_rate
        self._channels = channels

    async def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if not self._started:
            self._buffer.extend(pcm)
            if len(self._buffer) < self._initial_buffer_bytes:
                return
            await self._flush_start()
            return
        await self._send_chunk(pcm)

    async def done(self) -> None:
        if not self._started:
            await self._flush_start()
        await self._esp_client.send_audio_end(ok=True)

    async def error(self, message: str, status: int = 500) -> None:
        self.failed = True
        self.error_message = message
        await self._log_bus.emit("ERROR", "TTS", "ESP audio stream error", {"status": status, "message": message})
        try:
            await self._esp_client.send_audio_error(message)
        except Exception as exc:
            await self._log_bus.emit("WARN", "TTS", "ESP audio error notification failed", {"error": safe_exc_message(exc)})

    @property
    def _initial_buffer_bytes(self) -> int:
        bytes_per_second = max(1, self._sample_rate) * max(1, self._channels) * 2
        return int(bytes_per_second * self._initial_buffer_ms / 1000)

    async def _flush_start(self) -> None:
        if self._started:
            return
        self._started = True
        await self._esp_client.send_audio_start(sample_rate=self._sample_rate, channels=self._channels)
        buffered = bytes(self._buffer)
        self._buffer.clear()
        for offset in range(0, len(buffered), RELAY_CHUNK_BYTES):
            await self._send_chunk(buffered[offset : offset + RELAY_CHUNK_BYTES])

    async def _send_chunk(self, pcm: bytes) -> None:
        await self._esp_client.send_audio_chunk(pcm)
        self.bytes_sent += len(pcm)


class PcmPacer:
    def __init__(
        self,
        sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
        channels: int = DEFAULT_PCM_CHANNELS,
        initial_burst_ms: int = PCM_PACE_INITIAL_BURST_MS,
    ) -> None:
        self.bytes_per_second = max(1, sample_rate) * max(1, channels) * 2
        self.sent_bytes = 0
        self.started_at: float | None = None
        self.initial_burst_bytes = int(self.bytes_per_second * initial_burst_ms / 1000)

    async def after_send(self, byte_count: int) -> None:
        loop_time = asyncio.get_running_loop().time()
        if self.started_at is None:
            self.started_at = loop_time
        self.sent_bytes += max(0, byte_count)
        if self.sent_bytes <= self.initial_burst_bytes:
            return
        target_elapsed = (self.sent_bytes - self.initial_burst_bytes) / self.bytes_per_second
        actual_elapsed = loop_time - self.started_at
        sleep_for = target_elapsed - actual_elapsed
        if sleep_for > 0:
            await asyncio.sleep(min(sleep_for, PCM_PACE_MAX_SLEEP))


def safe_exc_message(exc: Exception) -> str:
    return API_KEY_QUERY_RE.sub(r"\1***", str(exc))


def parse_stream_command(doc: dict[str, Any], expect_start: bool) -> StreamCommand:
    msg_type = str(doc.get("type", "")).strip().lower()
    if expect_start and msg_type != "start":
        raise ValueError("Expected TTS message type 'start'.")
    if not expect_start and msg_type not in {"start", "append"}:
        raise ValueError("Expected TTS message type 'start' or 'append'.")
    text = str(doc.get("text", ""))
    final = bool(doc.get("final", False))
    provider = str(doc.get("provider", "")).strip().lower()
    if not text.strip() and not final:
        raise ValueError("Intermediate TTS chunk cannot be empty.")
    return StreamCommand(msg_type=msg_type, text=text, final=final, provider=provider)


async def receive_stream_command(ws: WebSocket, expect_start: bool, timeout: float = 30) -> StreamCommand:
    try:
        doc = await asyncio.wait_for(ws.receive_json(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise asyncio.TimeoutError("Timed out while waiting for TTS relay command.") from exc
    return parse_stream_command(doc, expect_start)


async def send_error(ws: WebSocket, message: str, status: int = 500) -> None:
    await ws.send_json({"type": "error", "status": status, "message": message})


async def send_pcm_start(
    ws: WebSocket,
    sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
    channels: int = DEFAULT_PCM_CHANNELS,
) -> None:
    await ws.send_json(
        {
            "type": "start",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
            "channels": channels,
        }
    )


async def send_done(ws: WebSocket) -> None:
    await ws.send_json({"type": "done"})


async def send_pcm_bytes(ws: WebSocket, pcm: bytes, pacer: PcmPacer | None = None) -> None:
    await send_pcm_bytes_to_output(WebSocketPcmOutput(ws), pcm, pacer)


async def send_pcm_bytes_to_output(output: PcmOutput, pcm: bytes, pacer: PcmPacer | None = None) -> None:
    even_len = len(pcm) & ~1
    if even_len <= 0:
        return
    for i in range(0, even_len, RELAY_CHUNK_BYTES):
        chunk = pcm[i : i + RELAY_CHUNK_BYTES]
        await output.write(chunk)
        if pacer is not None:
            await pacer.after_send(len(chunk))


def relay_config_from_panel(config: dict[str, Any], provider_override: str = "") -> TtsRelayConfig:
    tts = config.get("tts", {}) if isinstance(config, dict) else {}
    openai = tts.get("openai", {}) if isinstance(tts.get("openai"), dict) else {}
    cartesia = tts.get("cartesia", {}) if isinstance(tts.get("cartesia"), dict) else {}
    return TtsRelayConfig(
        enabled=bool(tts.get("enabled", True)),
        provider=(provider_override or str(tts.get("provider") or "openai")).lower(),
        pcm_sample_rate=int(tts.get("pcm_sample_rate") or DEFAULT_PCM_SAMPLE_RATE),
        esp_initial_buffer_ms=int(tts.get("esp_initial_buffer_ms") or 900),
        openai_api_key=str(openai.get("api_key") or tts.get("openai_api_key") or ""),
        openai_model=str(openai.get("model") or tts.get("openai_model") or tts.get("model") or "gpt-4o-mini-tts"),
        openai_voice=str(openai.get("voice") or tts.get("openai_voice") or tts.get("voice") or "coral"),
        openai_instructions=str(openai.get("instructions") or tts.get("openai_instructions") or tts.get("instructions") or ""),
        cartesia_api_key=str(cartesia.get("api_key") or tts.get("cartesia_api_key") or ""),
        cartesia_model_id=str(cartesia.get("model_id") or tts.get("cartesia_model_id") or "sonic-3"),
        cartesia_voice_id=str(cartesia.get("voice_id") or tts.get("cartesia_voice_id") or ""),
        cartesia_language=str(cartesia.get("language") or tts.get("cartesia_language") or "tr"),
        cartesia_version=str(cartesia.get("version") or tts.get("cartesia_version") or "2026-03-01"),
    )


class CartesiaContinuationRelay:
    def __init__(self, session: aiohttp.ClientSession, output: PcmOutput, cfg: TtsRelayConfig, log_bus: LogBus) -> None:
        self._session = session
        self._output = output
        self._cfg = cfg
        self._log_bus = log_bus
        self._context_id = f"alice-{uuid.uuid4()}"
        self._upstream: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._start_sent = False
        self._pacer: PcmPacer | None = None
        self._final_sent = False
        self._error: str | None = None

    async def _ensure_open(self) -> None:
        if self._upstream is not None:
            return
        query = urlencode({"api_key": self._cfg.cartesia_api_key, "cartesia_version": self._cfg.cartesia_version})
        self._upstream = await self._session.ws_connect(
            f"{CARTESIA_WS_URL}?{query}",
            timeout=15,
            receive_timeout=120,
            heartbeat=20,
        )
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        assert self._upstream is not None
        try:
            async for msg in self._upstream:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"Cartesia websocket error: {self._upstream.exception()}")
                    continue
                try:
                    doc = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                msg_type = str(doc.get("type", "")).lower()
                if msg_type == "chunk":
                    audio_b64 = str(doc.get("data") or doc.get("audio") or "").strip()
                    if not audio_b64:
                        continue
                    pcm = base64.b64decode(audio_b64)
                    if not self._start_sent:
                        await self._output.start(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._pacer = PcmPacer(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS) if self._output.pace_pcm else None
                        self._start_sent = True
                    await send_pcm_bytes_to_output(self._output, pcm, self._pacer)
                    continue
                if msg_type == "done":
                    if not self._start_sent:
                        await self._output.start(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._start_sent = True
                    await self._output.done()
                    self._done.set()
                    return
                if msg_type == "error":
                    error = str(doc.get("error") or doc.get("message") or "Cartesia returned an error.")
                    self._error = error
                    await self._output.error(f"Cartesia TTS error: {error[:300]}", int(doc.get("status_code", 502) or 502))
                    self._done.set()
                    return
        except Exception as exc:
            self._error = safe_exc_message(exc)
            await self._log_bus.emit("ERROR", "TTS", "Cartesia receiver failed", {"error": self._error})
            try:
                await self._output.error(f"Cartesia continuation error: {self._error}", 502)
            except Exception:
                pass
            self._done.set()

    async def send_input(self, text: str, final: bool) -> None:
        if self._final_sent:
            raise RuntimeError("Cartesia continuation cannot accept input after final chunk.")
        await self._ensure_open()
        assert self._upstream is not None
        payload = {
            "model_id": self._cfg.cartesia_model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": self._cfg.cartesia_voice_id},
            "language": self._cfg.cartesia_language,
            "context_id": self._context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._cfg.pcm_sample_rate,
            },
            "continue": not final,
        }
        await self._upstream.send_json(payload)
        self._final_sent = final

    async def wait_done(self) -> None:
        await self._done.wait()

    async def close(self) -> None:
        if self._upstream is not None and not self._upstream.closed:
            if not self._final_sent:
                try:
                    await self._upstream.send_json({"context_id": self._context_id, "cancel": True})
                except Exception:
                    pass
            await self._upstream.close()
        if self._receiver_task is not None:
            try:
                await self._receiver_task
            except Exception:
                pass


class TtsRelay:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._log_bus = log_bus

    async def status(self) -> dict[str, Any]:
        cfg = relay_config_from_panel(await self._config_store.get(include_secrets=False))
        return {
            "enabled": cfg.enabled,
            "provider": cfg.provider,
            "pcm_sample_rate": cfg.pcm_sample_rate,
            "openai_api_key_configured": bool(cfg.openai_api_key),
            "cartesia_api_key_configured": bool(cfg.cartesia_api_key),
            "cartesia_voice_configured": bool(cfg.cartesia_voice_id),
        }

    async def websocket_session(self, ws: WebSocket) -> None:
        await ws.accept()
        first_cmd: StreamCommand | None = None
        try:
            first_cmd = await receive_stream_command(ws, expect_start=True)
            cfg = relay_config_from_panel(await self._config_store.get(include_secrets=True), first_cmd.provider)
            output = WebSocketPcmOutput(ws)
            await self._log_bus.emit("INFO", "TTS", "TTS relay websocket started", {"provider": cfg.provider})
            async with aiohttp.ClientSession() as session:
                if cfg.provider == "cartesia":
                    await self._relay_cartesia_continuation(session, output, cfg, first_cmd, ws)
                elif cfg.provider == "openai":
                    text = first_cmd.text if first_cmd.final else await self._collect_buffered_stream_text(ws, first_cmd)
                    await self._relay_openai_stream(session, output, cfg, text)
                else:
                    await output.error(f"TTS provider '{cfg.provider}' is configured but not implemented in this preview.", 501)
        except WebSocketDisconnect:
            await self._log_bus.emit("INFO", "TTS", "TTS relay websocket disconnected")
        except Exception as exc:
            await self._log_bus.emit("ERROR", "TTS", "TTS relay websocket failed", {"error": safe_exc_message(exc)})
            try:
                await send_error(ws, safe_exc_message(exc), 500)
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    async def synthesize_to_esp(self, text: str, esp_client: Any) -> dict[str, Any]:
        cfg = relay_config_from_panel(await self._config_store.get(include_secrets=True))
        if not cfg.enabled:
            return {"ok": False, "status": "disabled", "message": "TTS is disabled."}
        if not text.strip():
            return {"ok": False, "status": "empty_text", "message": "TTS text is empty."}
        if not await esp_client.audio_stream_ready():
            return {"ok": False, "status": "esp_ws_offline", "message": "ESP WebSocket is not connected."}

        output = EspPcmOutput(esp_client, self._log_bus, cfg.esp_initial_buffer_ms)
        await self._log_bus.emit("INFO", "TTS", "ESP TTS stream starting", {"provider": cfg.provider})
        async with aiohttp.ClientSession() as session:
            if cfg.provider == "openai":
                await self._relay_openai_stream(session, output, cfg, text)
            elif cfg.provider == "cartesia":
                first_cmd = StreamCommand(msg_type="start", text=text, final=True, provider=cfg.provider)
                await self._relay_cartesia_continuation(session, output, cfg, first_cmd)
            else:
                await output.error(f"TTS provider '{cfg.provider}' is configured but not implemented in this preview.", 501)
                return {"ok": False, "status": "provider_not_implemented", "provider": cfg.provider}
        if output.failed:
            return {
                "ok": False,
                "status": "stream_failed",
                "provider": cfg.provider,
                "message": output.error_message,
                "bytes": output.bytes_sent,
            }
        await self._log_bus.emit("INFO", "TTS", "ESP TTS stream finished", {"bytes": output.bytes_sent})
        return {"ok": True, "status": "streamed_to_esp", "provider": cfg.provider, "bytes": output.bytes_sent}

    async def _collect_buffered_stream_text(self, ws: WebSocket, first_cmd: StreamCommand) -> str:
        chunks = [first_cmd.text]
        cmd = first_cmd
        while not cmd.final:
            cmd = await receive_stream_command(ws, expect_start=False, timeout=60)
            chunks.append(cmd.text)
        return "".join(chunks)

    async def _relay_openai_stream(
        self,
        session: aiohttp.ClientSession,
        output: PcmOutput,
        cfg: TtsRelayConfig,
        text: str,
    ) -> None:
        if not cfg.openai_api_key:
            await output.error("OpenAI API key is not configured.", 500)
            return
        headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": cfg.openai_model,
            "voice": cfg.openai_voice,
            "input": text,
            "response_format": "pcm",
        }
        if cfg.openai_instructions:
            payload["instructions"] = cfg.openai_instructions
        timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_read=120)
        try:
            async with session.post(OPENAI_SPEECH_URL, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    await self._log_bus.emit("ERROR", "TTS", "OpenAI TTS failed", {"status": resp.status, "body": body[:500]})
                    await output.error(f"OpenAI TTS error: {body[:300]}", resp.status)
                    return
                await output.start(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS) if output.pace_pcm else None
                pending = b""
                async for chunk in resp.content.iter_chunked(RELAY_CHUNK_BYTES):
                    if not chunk:
                        continue
                    pending += chunk
                    even_len = len(pending) & ~1
                    if even_len:
                        await send_pcm_bytes_to_output(output, pending[:even_len], pacer)
                        pending = pending[even_len:]
                await output.done()
        except Exception as exc:
            await self._log_bus.emit("ERROR", "TTS", "OpenAI TTS stream failed", {"error": safe_exc_message(exc)})
            await output.error(f"OpenAI stream error: {safe_exc_message(exc)}", 502)

    async def _relay_cartesia_continuation(
        self,
        session: aiohttp.ClientSession,
        output: PcmOutput,
        cfg: TtsRelayConfig,
        first_cmd: StreamCommand,
        input_ws: WebSocket | None = None,
    ) -> None:
        if not cfg.cartesia_api_key:
            await output.error("Cartesia API key is not configured.", 500)
            return
        if not cfg.cartesia_voice_id:
            await output.error("Cartesia voice_id is not configured.", 500)
            return
        relay = CartesiaContinuationRelay(session, output, cfg, self._log_bus)
        try:
            await relay.send_input(first_cmd.text, first_cmd.final)
            cmd = first_cmd
            while not cmd.final:
                if input_ws is None:
                    await output.error("Cartesia continuation needs an input WebSocket for non-final chunks.", 500)
                    return
                cmd = await receive_stream_command(input_ws, expect_start=False, timeout=60)
                await relay.send_input(cmd.text, cmd.final)
            await relay.wait_done()
        finally:
            await relay.close()
