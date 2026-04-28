import asyncio
import json
import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
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
class EndpointingConfig:
    enabled: bool = True
    auto_finalize_on_vad_end: bool = False
    no_speech_timeout_ms: int = 5000
    speech_start_min_ms: int = 160
    speech_end_silence_ms: int = 700
    max_utterance_ms: int = 15000
    start_avg_abs_threshold: int = 220
    end_avg_abs_threshold: int = 140


@dataclass
class HaBridgeConfig:
    enabled: bool = True
    api_base_url: str = "http://supervisor/core/api"


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
    endpointing: EndpointingConfig = field(default_factory=EndpointingConfig)
    ha_bridge: HaBridgeConfig = field(default_factory=HaBridgeConfig)
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
    endpointing_raw = _read_group(raw, "endpointing")
    ha_bridge_raw = _read_group(raw, "ha_bridge")
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
        endpointing=EndpointingConfig(
            enabled=_read_bool(endpointing_raw, "enabled", True),
            auto_finalize_on_vad_end=_read_bool(endpointing_raw, "auto_finalize_on_vad_end", False),
            no_speech_timeout_ms=_read_int(endpointing_raw, "no_speech_timeout_ms", 5000),
            speech_start_min_ms=_read_int(endpointing_raw, "speech_start_min_ms", 160),
            speech_end_silence_ms=_read_int(endpointing_raw, "speech_end_silence_ms", 700),
            max_utterance_ms=_read_int(endpointing_raw, "max_utterance_ms", 15000),
            start_avg_abs_threshold=_read_int(endpointing_raw, "start_avg_abs_threshold", 220),
            end_avg_abs_threshold=_read_int(endpointing_raw, "end_avg_abs_threshold", 140),
        ),
        ha_bridge=HaBridgeConfig(
            enabled=_read_bool(ha_bridge_raw, "enabled", True),
            api_base_url=_read_str(ha_bridge_raw, "api_base_url", "http://supervisor/core/api"),
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


class HomeAssistantBridge:
    def __init__(self, cfg: HaBridgeConfig) -> None:
        self._cfg = cfg
        self._base_url = cfg.api_base_url.rstrip("/")
        self._token = os.environ.get("SUPERVISOR_TOKEN", "").strip()

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def is_ready(self) -> bool:
        return self._cfg.enabled and bool(self._token)

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def health(self, session: aiohttp.ClientSession) -> dict:
        if not self._cfg.enabled:
            return {"enabled": False, "connected": False, "reason": "disabled"}
        if not self._token:
            return {"enabled": True, "connected": False, "reason": "missing_supervisor_token"}

        try:
            async with session.get(
                self._base_url,
                headers=self._headers(),
            ) as resp:
                text = await resp.text()
                return {
                    "enabled": True,
                    "connected": resp.status < 400,
                    "status": resp.status,
                    "body": text[:96],
                }
        except Exception as exc:  # pragma: no cover - best effort probe
            return {"enabled": True, "connected": False, "reason": str(exc)}

    async def get_state(self, session: aiohttp.ClientSession, entity_id: str) -> dict | None:
        async with session.get(
            f"{self._base_url}/states/{entity_id}",
            headers=self._headers(),
        ) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json()

    async def list_states(
        self,
        session: aiohttp.ClientSession,
        domain: str = "",
        limit: int = 64,
    ) -> list[dict]:
        async with session.get(
            f"{self._base_url}/states",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            raw_states = await resp.json()

        domain = domain.strip().lower()
        slimmed: list[dict] = []
        for item in raw_states:
            entity_id = str(item.get("entity_id") or "")
            if not entity_id:
                continue
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            attributes = item.get("attributes") or {}
            slimmed.append(
                {
                    "entity_id": entity_id,
                    "state": item.get("state"),
                    "friendly_name": attributes.get("friendly_name", ""),
                }
            )
            if len(slimmed) >= max(1, min(limit, 256)):
                break
        return slimmed

    async def call_service(
        self,
        session: aiohttp.ClientSession,
        domain: str,
        service: str,
        data: dict | None = None,
    ) -> list | dict | None:
        async with session.post(
            f"{self._base_url}/services/{domain}/{service}",
            headers=self._headers(),
            json=data or {},
        ) as resp:
            resp.raise_for_status()
            if resp.content_type == "application/json":
                return await resp.json()
            return await resp.text()


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
    def __init__(
        self,
        ws: web.WebSocketResponse,
        engine: WhisperEngine,
        endpointing: EndpointingConfig,
        bridge: HomeAssistantBridge,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self.ws = ws
        self.engine = engine
        self.endpointing = endpointing
        self.bridge = bridge
        self.http_session = http_session
        self.sample_rate = 16000
        self.language = "tr"
        self.session_id = ""
        self.audio = bytearray()
        self.started = False
        self._speech_started = False
        self._speech_ended = False
        self._consecutive_voice_ms = 0
        self._silence_ms = 0
        self._received_audio_ms = 0
        self._speech_started_at_ms = 0
        self._last_avg_abs = 0
        self._no_speech_timeout_sent = False
        self._finalizing = False

    async def send_event(self, event_type: str, **data) -> None:
        if self.ws.closed:
            return
        payload = {"type": event_type, **data}
        await self.ws.send_json(payload)

    def _reset_runtime_state(self) -> None:
        self.audio.clear()
        self.started = False
        self._speech_started = False
        self._speech_ended = False
        self._consecutive_voice_ms = 0
        self._silence_ms = 0
        self._received_audio_ms = 0
        self._speech_started_at_ms = 0
        self._last_avg_abs = 0
        self._no_speech_timeout_sent = False
        self._finalizing = False

    @staticmethod
    def _chunk_duration_ms(byte_count: int, sample_rate: int) -> int:
        if sample_rate <= 0 or byte_count <= 1:
            return 0
        sample_count = byte_count // 2
        return int((sample_count * 1000) / sample_rate)

    @staticmethod
    def _avg_abs_from_pcm(data: bytes) -> int:
        if len(data) < 2:
            return 0
        pcm = np.frombuffer(data, dtype="<i2")
        if pcm.size == 0:
            return 0
        abs_mean = np.abs(pcm.astype(np.int32)).mean()
        return int(abs_mean)

    async def _process_endpointing(self, chunk: bytes) -> None:
        if not self.endpointing.enabled:
            return

        chunk_ms = self._chunk_duration_ms(len(chunk), self.sample_rate)
        if chunk_ms <= 0:
            return

        self._received_audio_ms += chunk_ms
        avg_abs = self._avg_abs_from_pcm(chunk)
        self._last_avg_abs = avg_abs

        if not self._speech_started:
            if avg_abs >= self.endpointing.start_avg_abs_threshold:
                self._consecutive_voice_ms += chunk_ms
            else:
                self._consecutive_voice_ms = 0

            if (
                not self._no_speech_timeout_sent
                and self._received_audio_ms >= self.endpointing.no_speech_timeout_ms
            ):
                self._no_speech_timeout_sent = True
                await self.send_event(
                    "no_speech_timeout",
                    audio_ts=self._received_audio_ms,
                    avg_abs=avg_abs,
                )

            if self._consecutive_voice_ms >= self.endpointing.speech_start_min_ms:
                self._speech_started = True
                self._speech_started_at_ms = max(0, self._received_audio_ms - self._consecutive_voice_ms)
                self._silence_ms = 0
                await self.send_event(
                    "vad_start",
                    audio_ts=self._speech_started_at_ms,
                    avg_abs=avg_abs,
                )
            return

        if self._speech_ended:
            return

        if avg_abs >= self.endpointing.end_avg_abs_threshold:
            self._silence_ms = 0
        else:
            self._silence_ms += chunk_ms

        utterance_ms = max(0, self._received_audio_ms - self._speech_started_at_ms)
        if utterance_ms >= self.endpointing.max_utterance_ms:
            self._speech_ended = True
            await self.send_event(
                "max_utterance_reached",
                audio_ts=self._received_audio_ms,
                utterance_ms=utterance_ms,
            )
            if self.endpointing.auto_finalize_on_vad_end:
                await self.handle_eos(reason="max_utterance")
            return

        if self._silence_ms >= self.endpointing.speech_end_silence_ms:
            self._speech_ended = True
            vad_end_ts = max(self._speech_started_at_ms, self._received_audio_ms - self._silence_ms)
            await self.send_event(
                "vad_end",
                audio_ts=vad_end_ts,
                utterance_ms=utterance_ms,
            )
            if self.endpointing.auto_finalize_on_vad_end:
                await self.handle_eos(reason="vad_end")

    async def handle_start(self, doc: dict) -> None:
        self.sample_rate = int(doc.get("sample_rate") or 16000)
        self.language = str(doc.get("language") or "tr").strip() or "tr"
        self.session_id = str(doc.get("session_id") or "").strip()
        self._reset_runtime_state()
        self.started = True
        await self.send_event(
            "session_started",
            sample_rate=self.sample_rate,
            language=self.language,
            endpointing_enabled=self.endpointing.enabled,
        )

    async def handle_ha_get_state(self, doc: dict) -> None:
        entity_id = str(doc.get("entity_id") or "").strip()
        if not entity_id:
            await self.send_event("ha_state_result", ok=False, error="entity_id gerekli")
            return
        if not self.bridge.is_ready():
            await self.send_event("ha_state_result", ok=False, error="ha_bridge hazir degil")
            return
        try:
            result = await self.bridge.get_state(self.http_session, entity_id)
            await self.send_event("ha_state_result", ok=True, entity=result)
        except Exception as exc:
            await self.send_event("ha_state_result", ok=False, error=str(exc), entity_id=entity_id)

    async def handle_ha_list_states(self, doc: dict) -> None:
        domain = str(doc.get("domain") or "").strip()
        limit = _read_int(doc, "limit", 64)
        if not self.bridge.is_ready():
            await self.send_event("ha_list_states_result", ok=False, error="ha_bridge hazir degil")
            return
        try:
            entities = await self.bridge.list_states(self.http_session, domain=domain, limit=limit)
            await self.send_event(
                "ha_list_states_result",
                ok=True,
                domain=domain,
                count=len(entities),
                entities=entities,
            )
        except Exception as exc:
            await self.send_event("ha_list_states_result", ok=False, error=str(exc), domain=domain)

    async def handle_ha_call_service(self, doc: dict) -> None:
        domain = str(doc.get("domain") or "").strip()
        service = str(doc.get("service") or "").strip()
        data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
        if not domain or not service:
            await self.send_event(
                "ha_service_result",
                ok=False,
                error="domain ve service gerekli",
            )
            return
        if not self.bridge.is_ready():
            await self.send_event("ha_service_result", ok=False, error="ha_bridge hazir degil")
            return
        try:
            result = await self.bridge.call_service(self.http_session, domain, service, data=data)
            await self.send_event(
                "ha_service_result",
                ok=True,
                domain=domain,
                service=service,
                result=result,
            )
        except Exception as exc:
            await self.send_event(
                "ha_service_result",
                ok=False,
                error=str(exc),
                domain=domain,
                service=service,
            )

    async def handle_eos(self, reason: str = "client_eos") -> None:
        if not self.started:
            await self.send_event("error", message="Oturum baslatilmadan eos alindi.")
            return
        if self._finalizing:
            return

        self._finalizing = True

        await self.send_event(
            "stt_started",
            bytes=len(self.audio),
            reason=reason,
            audio_ms=self._received_audio_ms,
            avg_abs=self._last_avg_abs,
        )
        text = await asyncio.to_thread(
            self.engine.transcribe_pcm16le,
            bytes(self.audio),
            self.sample_rate,
        )
        await self.send_event("stt_result", text=text, session_id=self.session_id)
        await self.send_event("session_completed", reason=reason, audio_ms=self._received_audio_ms)
        self._reset_runtime_state()

    async def handle_message(self, msg: web.WSMessage) -> bool:
        if msg.type == web.WSMsgType.BINARY:
            if not self.started:
                await self.send_event("error", message="Binary ses verisi start oncesi geldi.")
                return True
            self.audio.extend(msg.data)
            await self._process_endpointing(msg.data)
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
            self._reset_runtime_state()
            await self.send_event("session_reset")
            return True
        if msg_type == "ha_get_state":
            await self.handle_ha_get_state(doc)
            return True
        if msg_type == "ha_list_states":
            await self.handle_ha_list_states(doc)
            return True
        if msg_type == "ha_call_service":
            await self.handle_ha_call_service(doc)
            return True

        await self.send_event("error", message=f"Desteklenmeyen mesaj tipi: {msg_type}")
        return True


async def health_handler(_request: web.Request) -> web.Response:
    app = _request.app
    bridge: HomeAssistantBridge = app["ha_bridge"]
    http_session: aiohttp.ClientSession = app["http_session"]
    ha_status = await bridge.health(http_session)
    return web.json_response(
        {
            "ok": True,
            "service": "alice_realtime_voice",
            "version": "0.4.0",
            "ha_bridge": ha_status,
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    engine: WhisperEngine = request.app["stt_engine"]
    cfg: VoiceConfig = request.app["cfg"]
    bridge: HomeAssistantBridge = request.app["ha_bridge"]
    http_session: aiohttp.ClientSession = request.app["http_session"]
    session = VoiceSession(ws, engine, cfg.endpointing, bridge, http_session)
    await session.send_event(
        "hello",
        service="alice_realtime_voice",
        version="0.4.0",
        endpointing_enabled=cfg.endpointing.enabled,
        ha_bridge_enabled=cfg.ha_bridge.enabled,
    )

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


async def create_http_session(_app: web.Application) -> None:
    timeout = aiohttp.ClientTimeout(total=10)
    _app["http_session"] = aiohttp.ClientSession(timeout=timeout)


async def close_http_session(_app: web.Application) -> None:
    session: aiohttp.ClientSession | None = _app.get("http_session")
    if session is not None:
        await session.close()


def build_app() -> web.Application:
    cfg = load_config()
    level = logging.INFO if cfg.debug_logs else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = web.Application()
    app["cfg"] = cfg
    app["stt_engine"] = WhisperEngine(cfg.stt)
    app["ha_bridge"] = HomeAssistantBridge(cfg.ha_bridge)
    app.on_startup.append(create_http_session)
    app.on_cleanup.append(close_http_session)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    app = build_app()
    cfg: VoiceConfig = app["cfg"]
    _LOG.info(
        "Alice Realtime Voice add-on basliyor. version=%s port=%s stt_provider=%s model=%s endpointing=%s ha_bridge=%s",
        "0.4.0",
        cfg.port,
        cfg.stt.provider,
        cfg.stt.model,
        "acik" if cfg.endpointing.enabled else "kapali",
        "acik" if cfg.ha_bridge.enabled else "kapali",
    )
    web.run_app(app, host="0.0.0.0", port=cfg.port)
