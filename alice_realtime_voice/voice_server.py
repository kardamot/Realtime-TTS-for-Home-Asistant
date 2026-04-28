import asyncio
import json
import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
from faster_whisper import WhisperModel

_LOG = logging.getLogger("alice_realtime_voice")
OPTIONS_PATH = Path("/data/options.json")


@dataclass
class SttConfig:
    provider: str = "faster_whisper"
    model: str = "small"
    language: str = "tr"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = True


@dataclass
class LlmConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = ""
    system_prompt: str = ""


@dataclass
class TtsConfig:
    enabled: bool = False
    relay_url: str = "ws://127.0.0.1:8765/ws"


@dataclass
class VoiceConfig:
    port: int = 8766
    debug_logs: bool = True
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)


def _read_group(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def _read_str(raw: dict, key: str, default: str = "") -> str:
    return str(raw.get(key, default) or default).strip()


def _read_int(raw: dict, key: str, default: int) -> int:
    try:
        return int(raw.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _read_bool(raw: dict, key: str, default: bool = False) -> bool:
    return bool(raw.get(key, default))


def load_config() -> VoiceConfig:
    if not OPTIONS_PATH.exists():
        return VoiceConfig()

    with OPTIONS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    stt_raw = _read_group(raw, "stt")
    llm_raw = _read_group(raw, "llm")
    tts_raw = _read_group(raw, "tts")

    return VoiceConfig(
        port=_read_int(raw, "port", 8766),
        debug_logs=_read_bool(raw, "debug_logs", True),
        stt=SttConfig(
            provider=_read_str(stt_raw, "provider", "faster_whisper"),
            model=_read_str(stt_raw, "model", "small"),
            language=_read_str(stt_raw, "language", "tr"),
            compute_type=_read_str(stt_raw, "compute_type", "int8"),
            beam_size=_read_int(stt_raw, "beam_size", 1),
            vad_filter=_read_bool(stt_raw, "vad_filter", True),
        ),
        llm=LlmConfig(
            provider=_read_str(llm_raw, "provider", "openai"),
            model=_read_str(llm_raw, "model", "gpt-5-mini"),
            api_key=_read_str(llm_raw, "api_key"),
            base_url=_read_str(llm_raw, "base_url"),
            system_prompt=_read_str(llm_raw, "system_prompt"),
        ),
        tts=TtsConfig(
            enabled=_read_bool(tts_raw, "enabled", False),
            relay_url=_read_str(tts_raw, "relay_url", "ws://127.0.0.1:8765/ws"),
        ),
    )


class WhisperEngine:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            _LOG.info(
                "Whisper model yukleniyor. model=%s compute_type=%s",
                self._cfg.model,
                self._cfg.compute_type,
            )
            self._model = WhisperModel(
                self._cfg.model,
                device="cpu",
                compute_type=self._cfg.compute_type,
            )
        return self._model

    def transcribe_pcm16le(self, pcm_bytes: bytes, sample_rate: int) -> str:
        model = self._ensure_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)

            segments, _info = model.transcribe(
                temp_path,
                language=self._cfg.language or None,
                beam_size=max(1, self._cfg.beam_size),
                vad_filter=self._cfg.vad_filter,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


class VoiceSession:
    def __init__(self, ws: web.WebSocketResponse, engine: WhisperEngine) -> None:
        self.ws = ws
        self.engine = engine
        self.sample_rate = 16000
        self.language = "tr"
        self.session_id = ""
        self.audio = bytearray()
        self.started = False

    async def send_event(self, event_type: str, **data) -> None:
        if self.ws.closed:
            return
        payload = {"type": event_type, **data}
        await self.ws.send_json(payload)

    async def handle_start(self, doc: dict) -> None:
        self.sample_rate = int(doc.get("sample_rate") or 16000)
        self.language = str(doc.get("language") or "tr").strip() or "tr"
        self.session_id = str(doc.get("session_id") or "").strip()
        self.audio.clear()
        self.started = True
        await self.send_event("session_started", sample_rate=self.sample_rate, language=self.language)

    async def handle_eos(self) -> None:
        if not self.started:
            await self.send_event("error", message="Oturum baslatilmadan eos alindi.")
            return

        await self.send_event("stt_started", bytes=len(self.audio))
        text = await asyncio.to_thread(
            self.engine.transcribe_pcm16le,
            bytes(self.audio),
            self.sample_rate,
        )
        await self.send_event("stt_result", text=text, session_id=self.session_id)
        await self.send_event("session_completed")
        self.audio.clear()
        self.started = False

    async def handle_message(self, msg: web.WSMessage) -> bool:
        if msg.type == web.WSMsgType.BINARY:
            if not self.started:
                await self.send_event("error", message="Binary ses verisi start oncesi geldi.")
                return True
            self.audio.extend(msg.data)
            return True

        if msg.type != web.WSMsgType.TEXT:
            return False

        try:
            doc = json.loads(msg.data)
        except json.JSONDecodeError:
            await self.send_event("error", message="Gecersiz JSON komutu.")
            return True

        msg_type = str(doc.get("type") or "").strip().lower()
        if msg_type == "start":
            await self.handle_start(doc)
            return True
        if msg_type == "eos":
            await self.handle_eos()
            return True
        if msg_type == "reset":
            self.audio.clear()
            self.started = False
            await self.send_event("session_reset")
            return True

        await self.send_event("error", message=f"Desteklenmeyen mesaj tipi: {msg_type}")
        return True


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "alice_realtime_voice"})


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    engine: WhisperEngine = request.app["stt_engine"]
    session = VoiceSession(ws, engine)
    await session.send_event("hello", service="alice_realtime_voice")

    async for msg in ws:
        if msg.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.CLOSING}:
            break
        if msg.type == web.WSMsgType.ERROR:
            break
        devam = await session.handle_message(msg)
        if not devam:
            break

    await ws.close()
    return ws


def build_app() -> web.Application:
    cfg = load_config()
    level = logging.INFO if cfg.debug_logs else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = web.Application()
    app["cfg"] = cfg
    app["stt_engine"] = WhisperEngine(cfg.stt)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    app = build_app()
    cfg: VoiceConfig = app["cfg"]
    _LOG.info(
        "Alice Realtime Voice add-on basliyor. port=%s stt_provider=%s model=%s",
        cfg.port,
        cfg.stt.provider,
        cfg.stt.model,
    )
    web.run_app(app, host="0.0.0.0", port=cfg.port)
