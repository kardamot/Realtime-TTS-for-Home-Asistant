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
    provider: str = "openai"
    pcm_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE
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
    even_len = len(pcm) & ~1
    if even_len <= 0:
        return
    for i in range(0, even_len, RELAY_CHUNK_BYTES):
        chunk = pcm[i : i + RELAY_CHUNK_BYTES]
        await ws.send_bytes(chunk)
        if pacer is not None:
            await pacer.after_send(len(chunk))


def relay_config_from_panel(config: dict[str, Any], provider_override: str = "") -> TtsRelayConfig:
    tts = config.get("tts", {}) if isinstance(config, dict) else {}
    openai = tts.get("openai", {}) if isinstance(tts.get("openai"), dict) else {}
    cartesia = tts.get("cartesia", {}) if isinstance(tts.get("cartesia"), dict) else {}
    return TtsRelayConfig(
        provider=(provider_override or str(tts.get("provider") or "openai")).lower(),
        pcm_sample_rate=int(tts.get("pcm_sample_rate") or DEFAULT_PCM_SAMPLE_RATE),
        openai_api_key=str(openai.get("api_key") or ""),
        openai_model=str(openai.get("model") or "gpt-4o-mini-tts"),
        openai_voice=str(openai.get("voice") or "coral"),
        openai_instructions=str(openai.get("instructions") or ""),
        cartesia_api_key=str(cartesia.get("api_key") or ""),
        cartesia_model_id=str(cartesia.get("model_id") or "sonic-3"),
        cartesia_voice_id=str(cartesia.get("voice_id") or ""),
        cartesia_language=str(cartesia.get("language") or "tr"),
        cartesia_version=str(cartesia.get("version") or "2026-03-01"),
    )


class CartesiaContinuationRelay:
    def __init__(self, session: aiohttp.ClientSession, ws: WebSocket, cfg: TtsRelayConfig, log_bus: LogBus) -> None:
        self._session = session
        self._ws = ws
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
                        await send_pcm_start(self._ws, self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._pacer = PcmPacer(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._start_sent = True
                    await send_pcm_bytes(self._ws, pcm, self._pacer)
                    continue
                if msg_type == "done":
                    if not self._start_sent:
                        await send_pcm_start(self._ws, self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._start_sent = True
                    await send_done(self._ws)
                    self._done.set()
                    return
                if msg_type == "error":
                    error = str(doc.get("error") or doc.get("message") or "Cartesia returned an error.")
                    self._error = error
                    await send_error(self._ws, f"Cartesia TTS error: {error[:300]}", int(doc.get("status_code", 502) or 502))
                    self._done.set()
                    return
        except Exception as exc:
            self._error = safe_exc_message(exc)
            await self._log_bus.emit("ERROR", "TTS", "Cartesia receiver failed", {"error": self._error})
            try:
                await send_error(self._ws, f"Cartesia continuation error: {self._error}", 502)
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
            "enabled": (await self._config_store.get(include_secrets=False)).get("tts", {}).get("enabled", True),
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
            await self._log_bus.emit("INFO", "TTS", "TTS relay websocket started", {"provider": cfg.provider})
            async with aiohttp.ClientSession() as session:
                if cfg.provider == "cartesia":
                    await self._relay_cartesia_continuation(session, ws, cfg, first_cmd)
                elif cfg.provider == "openai":
                    text = first_cmd.text if first_cmd.final else await self._collect_buffered_stream_text(ws, first_cmd)
                    await self._relay_openai_stream(session, ws, cfg, text)
                else:
                    await send_error(ws, f"TTS provider '{cfg.provider}' is configured but not implemented in this preview.", 501)
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
        ws: WebSocket,
        cfg: TtsRelayConfig,
        text: str,
    ) -> None:
        if not cfg.openai_api_key:
            await send_error(ws, "OpenAI API key is not configured.", 500)
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
                    await send_error(ws, f"OpenAI TTS error: {body[:300]}", resp.status)
                    return
                await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                pending = b""
                async for chunk in resp.content.iter_chunked(RELAY_CHUNK_BYTES):
                    if not chunk:
                        continue
                    pending += chunk
                    even_len = len(pending) & ~1
                    if even_len:
                        await send_pcm_bytes(ws, pending[:even_len], pacer)
                        pending = pending[even_len:]
                await send_done(ws)
        except Exception as exc:
            await self._log_bus.emit("ERROR", "TTS", "OpenAI TTS stream failed", {"error": safe_exc_message(exc)})
            await send_error(ws, f"OpenAI stream error: {safe_exc_message(exc)}", 502)

    async def _relay_cartesia_continuation(
        self,
        session: aiohttp.ClientSession,
        ws: WebSocket,
        cfg: TtsRelayConfig,
        first_cmd: StreamCommand,
    ) -> None:
        if not cfg.cartesia_api_key:
            await send_error(ws, "Cartesia API key is not configured.", 500)
            return
        if not cfg.cartesia_voice_id:
            await send_error(ws, "Cartesia voice_id is not configured.", 500)
            return
        relay = CartesiaContinuationRelay(session, ws, cfg, self._log_bus)
        try:
            await relay.send_input(first_cmd.text, first_cmd.final)
            cmd = first_cmd
            while not cmd.final:
                cmd = await receive_stream_command(ws, expect_start=False, timeout=60)
                await relay.send_input(cmd.text, cmd.final)
            await relay.wait_done()
        finally:
            await relay.close()

