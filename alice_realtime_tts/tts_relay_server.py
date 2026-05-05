import asyncio
import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
from aiohttp import ClientConnectionResetError
from aiohttp import web
from google.auth.transport.requests import Request
from google.oauth2 import service_account

_LOG = logging.getLogger("alice_realtime_tts")

OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
GOOGLE_AI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GOOGLE_CLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

ADDON_VERSION = "0.5.7"
DEFAULT_PCM_SAMPLE_RATE = 24000
DEFAULT_PCM_CHANNELS = 1
RELAY_CHUNK_BYTES = 2048
PCM_PACE_INITIAL_BURST_MS = 700
PCM_PACE_MAX_SLEEP = 0.05
OPTIONS_PATH = Path("/data/options.json")

PCM_OUTPUT_RE = re.compile(r"^pcm_(\d+)$")
API_KEY_QUERY_RE = re.compile(r"(api_key=)[^&\s]+")


@dataclass
class RelayConfig:
    provider: str = "openai"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "coral"
    openai_instructions: str = ""

    cartesia_api_key: str = ""
    cartesia_model_id: str = "sonic-3"
    cartesia_voice_id: str = ""
    cartesia_language: str = "tr"
    cartesia_version: str = "2026-03-01"

    elevenlabs_api_key: str = ""
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = ""
    elevenlabs_output_format: str = "pcm_16000"
    elevenlabs_latency_mode: int = 3

    google_ai_api_key: str = ""
    google_ai_model: str = "gemini-3.1-flash-tts-preview"
    google_ai_voice_name: str = "Kore"
    google_ai_prompt_prefix: str = ""

    google_cloud_credentials_json: str = ""
    google_cloud_voice_name: str = ""
    google_cloud_language_code: str = "tr-TR"
    google_cloud_ssml_gender: str = "FEMALE"

    pcm_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE
    port: int = 8765


@dataclass
class StreamCommand:
    msg_type: str
    text: str
    final: bool
    provider: str = ""
    content_type: str = ""


class PcmPacer:
    def __init__(
        self,
        sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
        channels: int = DEFAULT_PCM_CHANNELS,
        initial_burst_ms: int = PCM_PACE_INITIAL_BURST_MS,
    ) -> None:
        self.bytes_per_second = max(1, sample_rate) * max(1, channels) * 2
        self.initial_burst_bytes = int(self.bytes_per_second * max(0, initial_burst_ms) / 1000)
        self.sent_bytes = 0
        self.started_at: float | None = None

    async def after_send(self, sent_len: int) -> None:
        self.sent_bytes += max(0, sent_len)
        if self.sent_bytes <= self.initial_burst_bytes:
            return

        loop = asyncio.get_running_loop()
        if self.started_at is None:
            self.started_at = loop.time()

        paced_bytes = self.sent_bytes - self.initial_burst_bytes
        target_elapsed = paced_bytes / self.bytes_per_second
        sleep_for = self.started_at + target_elapsed - loop.time()
        if sleep_for > 0:
            await asyncio.sleep(min(sleep_for, PCM_PACE_MAX_SLEEP))


def _read_str(raw: dict, key: str, default: str = "") -> str:
    return str(raw.get(key, default) or default).strip()


def _read_int(raw: dict, key: str, default: int) -> int:
    try:
        return int(raw.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _read_group(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config() -> RelayConfig:
    if not OPTIONS_PATH.exists():
        return RelayConfig()

    with OPTIONS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    openai = _read_group(raw, "openai")
    cartesia = _read_group(raw, "cartesia")
    elevenlabs = _read_group(raw, "elevenlabs")
    google_ai = _read_group(raw, "google_ai")
    google_cloud = _read_group(raw, "google_cloud")

    return RelayConfig(
        provider=_read_str(raw, "provider", "openai").lower() or "openai",
        openai_api_key=_read_str(openai, "api_key", _read_str(raw, "openai_api_key")),
        openai_model=_read_str(openai, "model", _read_str(raw, "openai_model", _read_str(raw, "model", "gpt-4o-mini-tts"))) or "gpt-4o-mini-tts",
        openai_voice=_read_str(openai, "voice", _read_str(raw, "openai_voice", _read_str(raw, "voice", "coral"))) or "coral",
        openai_instructions=_read_str(openai, "instructions", _read_str(raw, "openai_instructions", _read_str(raw, "instructions"))),
        cartesia_api_key=_read_str(cartesia, "api_key", _read_str(raw, "cartesia_api_key")),
        cartesia_model_id=_read_str(cartesia, "model_id", _read_str(raw, "cartesia_model_id", "sonic-3")) or "sonic-3",
        cartesia_voice_id=_read_str(cartesia, "voice_id", _read_str(raw, "cartesia_voice_id")),
        cartesia_language=_read_str(cartesia, "language", _read_str(raw, "cartesia_language", "tr")) or "tr",
        cartesia_version=_read_str(cartesia, "version", _read_str(raw, "cartesia_version", "2026-03-01")) or "2026-03-01",
        elevenlabs_api_key=_read_str(elevenlabs, "api_key", _read_str(raw, "elevenlabs_api_key")),
        elevenlabs_model_id=_read_str(elevenlabs, "model_id", _read_str(raw, "elevenlabs_model_id", "eleven_flash_v2_5")) or "eleven_flash_v2_5",
        elevenlabs_voice_id=_read_str(elevenlabs, "voice_id", _read_str(raw, "elevenlabs_voice_id")),
        elevenlabs_output_format=_read_str(elevenlabs, "output_format", _read_str(raw, "elevenlabs_output_format", "pcm_16000")) or "pcm_16000",
        elevenlabs_latency_mode=_read_int(elevenlabs, "latency_mode", _read_int(raw, "elevenlabs_latency_mode", 3)),
        google_ai_api_key=_read_str(google_ai, "api_key", _read_str(raw, "google_ai_api_key")),
        google_ai_model=_read_str(google_ai, "model", _read_str(raw, "google_ai_model", "gemini-3.1-flash-tts-preview")) or "gemini-3.1-flash-tts-preview",
        google_ai_voice_name=_read_str(google_ai, "voice_name", _read_str(raw, "google_ai_voice_name", "Kore")) or "Kore",
        google_ai_prompt_prefix=_read_str(google_ai, "prompt_prefix", _read_str(raw, "google_ai_prompt_prefix")),
        google_cloud_credentials_json=_read_str(google_cloud, "credentials_json", _read_str(raw, "google_cloud_credentials_json")),
        google_cloud_voice_name=_read_str(google_cloud, "voice_name", _read_str(raw, "google_cloud_voice_name")),
        google_cloud_language_code=_read_str(google_cloud, "language_code", _read_str(raw, "google_cloud_language_code", "tr-TR")) or "tr-TR",
        google_cloud_ssml_gender=_read_str(google_cloud, "ssml_gender", _read_str(raw, "google_cloud_ssml_gender", "FEMALE")).upper() or "FEMALE",
        pcm_sample_rate=_read_int(raw, "pcm_sample_rate", DEFAULT_PCM_SAMPLE_RATE),
        port=_read_int(raw, "port", 8765),
    )


async def send_error(ws: web.WebSocketResponse, message: str, status: int = 500) -> None:
    if ws.closed:
        return
    try:
        await ws.send_json({"type": "error", "message": message, "status_code": status})
    except ClientConnectionResetError:
        _LOG.info("ESP baglantisi kapanirken hata mesaji gonderilemedi.")


async def send_pcm_start(
    ws: web.WebSocketResponse,
    sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
    channels: int = DEFAULT_PCM_CHANNELS,
) -> None:
    if ws.closed:
        raise ClientConnectionResetError("Cannot write to closing transport")
    await ws.send_json(
        {
            "type": "start",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
            "channels": channels,
        }
    )


async def send_done(ws: web.WebSocketResponse) -> None:
    if ws.closed:
        return
    try:
        await ws.send_json({"type": "done"})
    except ClientConnectionResetError:
        _LOG.info("ESP baglantisi kapanirken done mesaji gonderilemedi.")


async def send_pcm_bytes(
    ws: web.WebSocketResponse,
    pcm: bytes,
    pacer: PcmPacer | None = None,
) -> None:
    if not pcm:
        return
    if ws.closed:
        raise ClientConnectionResetError("Cannot write to closing transport")

    even_len = len(pcm) & ~1
    if even_len <= 0:
        return

    for i in range(0, even_len, RELAY_CHUNK_BYTES):
        if ws.closed:
            raise ClientConnectionResetError("Cannot write to closing transport")
        chunk = pcm[i:i + RELAY_CHUNK_BYTES]
        await ws.send_bytes(chunk)
        if pacer is not None:
            await pacer.after_send(len(chunk))


def parse_pcm_output_format(output_format: str) -> int | None:
    match = PCM_OUTPUT_RE.match(output_format.strip().lower())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def safe_exc_message(exc: Exception) -> str:
    mesaj = str(exc)
    return API_KEY_QUERY_RE.sub(r"\1***", mesaj)


def wav_payload_strip(audio: bytes) -> bytes:
    if len(audio) < 44:
        return audio
    if audio[0:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio
    pos = 12
    total = len(audio)
    while pos + 8 <= total:
        chunk_id = audio[pos:pos + 4]
        chunk_size = int.from_bytes(audio[pos + 4:pos + 8], "little", signed=False)
        data_start = pos + 8
        data_end = data_start + chunk_size
        if data_end > total:
            break
        if chunk_id == b"data":
            return audio[data_start:data_end]
        pos = data_end + (chunk_size & 1)
    return audio


def build_google_ai_prompt(cfg: RelayConfig, text: str) -> str:
    if cfg.google_ai_prompt_prefix:
        return f"{cfg.google_ai_prompt_prefix.strip()}\n\n{text}"
    return text


def google_cloud_token_al(cfg: RelayConfig) -> str:
    raw = cfg.google_cloud_credentials_json.strip()
    if not raw:
        raise ValueError("Google Cloud credentials json ayarlanmamis.")

    info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(info, scopes=[GOOGLE_CLOUD_SCOPE])
    creds.refresh(Request())
    token = creds.token
    if not token:
        raise RuntimeError("Google Cloud access token alinamadi.")
    return token


def parse_stream_command(doc: dict, expect_start: bool) -> StreamCommand:
    msg_type = str(doc.get("type", "")).strip().lower()
    if expect_start:
        if msg_type != "start":
            raise ValueError("Beklenen mesaj tipi 'start'.")
    elif msg_type not in {"start", "append"}:
        raise ValueError("Beklenen mesaj tipi 'start' veya 'append'.")

    text = str(doc.get("text", ""))
    final = bool(doc.get("final", False))
    provider = str(doc.get("provider", "")).strip().lower()
    content_type = str(doc.get("content_type", "")).strip().lower()

    if not text.strip() and not final:
        raise ValueError("Ara TTS parcasi bos olamaz.")

    return StreamCommand(
        msg_type=msg_type,
        text=text,
        final=final,
        provider=provider,
        content_type=content_type,
    )


async def receive_stream_command(
    ws: web.WebSocketResponse,
    timeout: float,
    expect_start: bool,
) -> StreamCommand:
    try:
        msg = await ws.receive(timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise asyncio.TimeoutError(
            "TTS relay komut beklerken zaman asimina ugradi."
        ) from exc

    if msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
        raise ClientConnectionResetError("ESP websocket baglantisini kapatti.")
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise ValueError("TTS relay komutu metin tabanli JSON olmali.")

    try:
        doc = json.loads(msg.data)
    except json.JSONDecodeError as exc:
        raise ValueError("Gecersiz JSON TTS relay komutu.") from exc

    return parse_stream_command(doc, expect_start)


class CartesiaContinuationRelay:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws: web.WebSocketResponse,
        cfg: RelayConfig,
    ) -> None:
        self._session = session
        self._ws = ws
        self._cfg = cfg
        self._context_id = f"alice-{uuid.uuid4()}"
        self._upstream: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._start_sent = False
        self._pacer: PcmPacer | None = None
        self._final_sent = False
        self._closed = False
        self._error: str | None = None

    async def _ensure_open(self) -> None:
        if self._upstream is not None:
            return

        query = urlencode(
            {
                "api_key": self._cfg.cartesia_api_key,
                "cartesia_version": self._cfg.cartesia_version,
            }
        )
        ws_url = f"{CARTESIA_WS_URL}?{query}"
        self._upstream = await self._session.ws_connect(
            ws_url,
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
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"Cartesia websocket hata verdi: {self._upstream.exception()}")
                    continue

                try:
                    doc = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                mesaj_turu = str(doc.get("type", "")).strip().lower()
                if mesaj_turu == "chunk":
                    audio_b64 = str(doc.get("data", "") or doc.get("audio", "")).strip()
                    if not audio_b64:
                        continue
                    try:
                        pcm = base64.b64decode(audio_b64)
                    except (ValueError, TypeError):
                        raise RuntimeError("Cartesia gecersiz PCM chunk gonderdi.")

                    if not self._start_sent:
                        await send_pcm_start(self._ws, self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._pacer = PcmPacer(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._start_sent = True
                    if pcm:
                        await send_pcm_bytes(self._ws, pcm, self._pacer)
                    continue

                if mesaj_turu == "done":
                    if not self._start_sent:
                        await send_pcm_start(self._ws, self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._pacer = PcmPacer(self._cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        self._start_sent = True
                    await send_done(self._ws)
                    self._done.set()
                    return

                if mesaj_turu == "error":
                    hata = str(doc.get("error", "") or doc.get("message", "") or "Cartesia hata dondu.")
                    kod = int(doc.get("status_code", 502) or 502)
                    _LOG.error("Cartesia TTS hatasi status=%s body=%s", kod, doc)
                    self._error = hata
                    await send_error(self._ws, f"Cartesia TTS hatasi: {hata[:300]}", kod)
                    self._done.set()
                    return

                # flush_done / timestamps / phoneme_timestamps kasitli olarak yok sayiliyor.

            if not self._done.is_set():
                self._error = "Cartesia websocket baglantisi beklenmedik sekilde kapandi."
                await send_error(self._ws, self._error, 502)
                self._done.set()
        except ClientConnectionResetError:
            _LOG.info("ESP istemcisi Cartesia continuation akisi sirasinda baglantiyi kapatti.")
            self._error = "ESP websocket baglantisi kapandi."
            self._done.set()
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("Cartesia continuation alici hatasi")
            self._error = safe_exc_message(exc)
            await send_error(self._ws, f"Cartesia continuation hatasi: {self._error}", 502)
            self._done.set()

    async def send_input(self, text: str, final: bool) -> None:
        if self._final_sent:
            raise RuntimeError("Cartesia continuation final sonrasinda yeni metin kabul etmez.")

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
        if final:
            self._final_sent = True

    async def finish_on_timeout(self) -> None:
        if self._final_sent:
            return
        _LOG.warning("Cartesia continuation final komutu gelmedi; bos final parcasi gonderiliyor.")
        await self.send_input("", True)

    async def wait_done(self) -> None:
        await self._done.wait()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._upstream is not None and not self._upstream.closed and not self._final_sent:
            try:
                await self._upstream.send_json({"context_id": self._context_id, "cancel": True})
            except Exception:  # noqa: BLE001
                pass

        if self._upstream is not None and not self._upstream.closed:
            try:
                await self._upstream.close()
            except Exception:  # noqa: BLE001
                pass

        if self._receiver_task is not None:
            try:
                await self._receiver_task
            except Exception:  # noqa: BLE001
                pass


async def relay_openai_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
) -> None:
    if not cfg.openai_api_key:
        await send_error(ws, "OpenAI API anahtari ayarlanmamis.", 500)
        return

    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
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
                _LOG.error("OpenAI TTS hatasi status=%s body=%s", resp.status, body)
                await send_error(ws, f"OpenAI TTS hatasi: {body[:300]}", resp.status)
                return

            await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
            pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)

            bekleyen = b""
            async for chunk in resp.content.iter_chunked(RELAY_CHUNK_BYTES):
                if not chunk:
                    continue
                bekleyen += chunk
                cift_len = len(bekleyen) & ~1
                if cift_len:
                    await send_pcm_bytes(ws, bekleyen[:cift_len], pacer)
                    bekleyen = bekleyen[cift_len:]

            if bekleyen:
                _LOG.warning("OpenAI akisi sonunda tek bayt kaldi; son PCM chunk kirpildi.")

            await send_done(ws)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi OpenAI TTS akisi sirasinda baglantiyi kapatti.")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        _LOG.exception("OpenAI stream hatasi")
        await send_error(ws, f"OpenAI stream hatasi: {safe_exc_message(exc)}", 502)


async def relay_cartesia_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
) -> None:
    if not cfg.cartesia_api_key:
        await send_error(ws, "Cartesia API anahtari ayarlanmamis.", 500)
        return
    if not cfg.cartesia_voice_id:
        await send_error(ws, "Cartesia voice_id ayarlanmamis.", 500)
        return

    query = urlencode(
        {
            "api_key": cfg.cartesia_api_key,
            "cartesia_version": cfg.cartesia_version,
        }
    )
    ws_url = f"{CARTESIA_WS_URL}?{query}"
    payload = {
        "model_id": cfg.cartesia_model_id,
        "transcript": text,
        "voice": {"mode": "id", "id": cfg.cartesia_voice_id},
        "language": cfg.cartesia_language,
        "context_id": f"alice-{uuid.uuid4()}",
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": cfg.pcm_sample_rate,
        },
        "continue": False,
    }

    start_gonderildi = False
    pacer: PcmPacer | None = None

    try:
        async with session.ws_connect(
            ws_url,
            timeout=15,
            receive_timeout=120,
            heartbeat=20,
        ) as upstream:
            await upstream.send_json(payload)

            async for msg in upstream:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"Cartesia websocket hata verdi: {upstream.exception()}")
                    continue

                try:
                    doc = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                mesaj_turu = str(doc.get("type", "")).strip().lower()

                if mesaj_turu == "chunk":
                    audio_b64 = str(doc.get("data", "") or doc.get("audio", "")).strip()
                    if not audio_b64:
                        continue
                    try:
                        pcm = base64.b64decode(audio_b64)
                    except (ValueError, TypeError):
                        await send_error(ws, "Cartesia gecersiz PCM chunk gonderdi.", 502)
                        return
                    if not start_gonderildi:
                        await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        start_gonderildi = True
                    if pcm:
                        await send_pcm_bytes(ws, pcm, pacer)
                    continue

                if mesaj_turu == "done":
                    if not start_gonderildi:
                        await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                        pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
                    await send_done(ws)
                    return

                if mesaj_turu == "error":
                    hata = str(doc.get("error", "") or doc.get("message", "") or "Cartesia hata dondu.")
                    kod = int(doc.get("status_code", 502) or 502)
                    _LOG.error("Cartesia TTS hatasi status=%s body=%s", kod, doc)
                    await send_error(ws, f"Cartesia TTS hatasi: {hata[:300]}", kod)
                    return

                # timestamps / flush_done / phoneme_timestamps bilerek yok sayiliyor

        await send_error(ws, "Cartesia websocket baglantisi beklenmedik sekilde kapandi.", 502)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi Cartesia TTS akisi sirasinda baglantiyi kapatti.")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        _LOG.exception("Cartesia stream hatasi")
        await send_error(ws, f"Cartesia stream hatasi: {safe_exc_message(exc)}", 502)
    except RuntimeError as exc:
        _LOG.exception("Cartesia websocket hatasi")
        await send_error(ws, str(exc), 502)


async def relay_elevenlabs_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
) -> None:
    if not cfg.elevenlabs_api_key:
        await send_error(ws, "ElevenLabs API anahtari ayarlanmamis.", 500)
        return
    if not cfg.elevenlabs_voice_id:
        await send_error(ws, "ElevenLabs voice_id ayarlanmamis.", 500)
        return

    sample_rate = parse_pcm_output_format(cfg.elevenlabs_output_format)
    if sample_rate is None:
        await send_error(ws, "ElevenLabs output_format pcm_16000/22050/24000/44100 gibi PCM olmali.", 400)
        return

    headers = {
        "xi-api-key": cfg.elevenlabs_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": cfg.elevenlabs_model_id,
    }
    params = {
        "output_format": cfg.elevenlabs_output_format,
    }
    if 0 <= cfg.elevenlabs_latency_mode <= 4:
        params["optimize_streaming_latency"] = str(cfg.elevenlabs_latency_mode)

    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_read=120)
    url = ELEVENLABS_STREAM_URL.format(voice_id=cfg.elevenlabs_voice_id)

    try:
        async with session.post(url, headers=headers, json=payload, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                _LOG.error("ElevenLabs TTS hatasi status=%s body=%s", resp.status, body)
                await send_error(ws, f"ElevenLabs TTS hatasi: {body[:300]}", resp.status)
                return

            await send_pcm_start(ws, sample_rate, DEFAULT_PCM_CHANNELS)
            pacer = PcmPacer(sample_rate, DEFAULT_PCM_CHANNELS)

            bekleyen = b""
            async for chunk in resp.content.iter_chunked(RELAY_CHUNK_BYTES):
                if not chunk:
                    continue
                bekleyen += chunk
                cift_len = len(bekleyen) & ~1
                if cift_len:
                    await send_pcm_bytes(ws, bekleyen[:cift_len], pacer)
                    bekleyen = bekleyen[cift_len:]

            if bekleyen:
                _LOG.warning("ElevenLabs akisi sonunda tek bayt kaldi; son PCM chunk kirpildi.")

            await send_done(ws)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi ElevenLabs TTS akisi sirasinda baglantiyi kapatti.")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        _LOG.exception("ElevenLabs stream hatasi")
        await send_error(ws, f"ElevenLabs stream hatasi: {safe_exc_message(exc)}", 502)


async def relay_google_ai_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
) -> None:
    if not cfg.google_ai_api_key:
        await send_error(ws, "Google AI API anahtari ayarlanmamis.", 500)
        return

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_google_ai_prompt(cfg, text),
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": cfg.google_ai_voice_name,
                    }
                }
            },
        },
    }
    params = {"key": cfg.google_ai_api_key}
    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_read=120)
    url = GOOGLE_AI_URL.format(model=cfg.google_ai_model)

    try:
        async with session.post(url, params=params, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                _LOG.error("Google AI TTS hatasi status=%s body=%s", resp.status, body)
                await send_error(ws, f"Google AI TTS hatasi: {body[:300]}", resp.status)
                return

            doc = await resp.json()
            data = (
                doc.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("inlineData", {})
                .get("data", "")
            )
            if not data:
                await send_error(ws, "Google AI TTS ses verisi donmedi.", 502)
                return

            pcm = base64.b64decode(data)
            await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
            pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
            await send_pcm_bytes(ws, pcm, pacer)
            await send_done(ws)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi Google AI TTS akisi sirasinda baglantiyi kapatti.")
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        _LOG.exception("Google AI stream hatasi")
        await send_error(ws, f"Google AI stream hatasi: {safe_exc_message(exc)}", 502)


async def relay_google_cloud_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
) -> None:
    if not cfg.google_cloud_credentials_json:
        await send_error(ws, "Google Cloud credentials json ayarlanmamis.", 500)
        return
    if not cfg.google_cloud_voice_name:
        await send_error(ws, "Google Cloud voice name ayarlanmamis.", 500)
        return

    try:
        access_token = await asyncio.to_thread(google_cloud_token_al, cfg)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        await send_error(ws, f"Google Cloud kimlik hatasi: {exc}", 500)
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": cfg.google_cloud_language_code,
            "name": cfg.google_cloud_voice_name,
            "ssmlGender": cfg.google_cloud_ssml_gender,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": cfg.pcm_sample_rate,
        },
    }
    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_read=120)

    try:
        async with session.post(GOOGLE_CLOUD_TTS_URL, headers=headers, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                _LOG.error("Google Cloud TTS hatasi status=%s body=%s", resp.status, body)
                await send_error(ws, f"Google Cloud TTS hatasi: {body[:300]}", resp.status)
                return

            doc = await resp.json()
            audio_b64 = str(doc.get("audioContent", "")).strip()
            if not audio_b64:
                await send_error(ws, "Google Cloud TTS ses verisi donmedi.", 502)
                return

            wav_or_pcm = base64.b64decode(audio_b64)
            pcm = wav_payload_strip(wav_or_pcm)
            await send_pcm_start(ws, cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
            pacer = PcmPacer(cfg.pcm_sample_rate, DEFAULT_PCM_CHANNELS)
            await send_pcm_bytes(ws, pcm, pacer)
            await send_done(ws)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi Google Cloud TTS akisi sirasinda baglantiyi kapatti.")
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        _LOG.exception("Google Cloud stream hatasi")
        await send_error(ws, f"Google Cloud stream hatasi: {safe_exc_message(exc)}", 502)


async def relay_stream(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    text: str,
    provider_override: str = "",
) -> None:
    provider = (provider_override or cfg.provider or "openai").strip().lower()

    if provider == "openai":
        await relay_openai_stream(session, ws, cfg, text)
        return
    if provider == "cartesia":
        await relay_cartesia_stream(session, ws, cfg, text)
        return
    if provider == "elevenlabs":
        await relay_elevenlabs_stream(session, ws, cfg, text)
        return
    if provider == "google_ai":
        await relay_google_ai_stream(session, ws, cfg, text)
        return
    if provider == "google_cloud":
        await relay_google_cloud_stream(session, ws, cfg, text)
        return

    await send_error(ws, f"Desteklenmeyen TTS provider: {provider}", 400)


async def collect_buffered_stream_text(
    ws: web.WebSocketResponse,
    first_cmd: StreamCommand,
) -> tuple[str, str, int]:
    provider_override = first_cmd.provider
    parca_sayisi = 1
    metin_parcalari: list[str] = []
    if first_cmd.text:
        metin_parcalari.append(first_cmd.text)

    final = first_cmd.final
    while not final:
        cmd = await receive_stream_command(ws, 120, expect_start=False)
        parca_sayisi += 1
        if cmd.provider:
            if provider_override and cmd.provider != provider_override:
                raise ValueError("Ayni TTS oturumunda provider degistirilemez.")
            provider_override = cmd.provider
        if cmd.text:
            metin_parcalari.append(cmd.text)
        final = cmd.final

    return "".join(metin_parcalari), provider_override, parca_sayisi


async def relay_cartesia_continuation_session(
    session: aiohttp.ClientSession,
    ws: web.WebSocketResponse,
    cfg: RelayConfig,
    first_cmd: StreamCommand,
) -> None:
    if not cfg.cartesia_api_key:
        await send_error(ws, "Cartesia API anahtari ayarlanmamis.", 500)
        return
    if not cfg.cartesia_voice_id:
        await send_error(ws, "Cartesia voice_id ayarlanmamis.", 500)
        return

    relay = CartesiaContinuationRelay(session, ws, cfg)
    parca_sayisi = 0
    try:
        cmd = first_cmd
        while True:
            parca_sayisi += 1
            await relay.send_input(cmd.text, cmd.final)
            if cmd.final:
                break
            try:
                cmd = await receive_stream_command(ws, 120, expect_start=False)
            except asyncio.TimeoutError:
                await relay.finish_on_timeout()
                break

        _LOG.info("Cartesia continuation acik. parca=%s", parca_sayisi)
        await relay.wait_done()
    finally:
        await relay.close()


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=1024 * 1024)
    await ws.prepare(request)

    cfg: RelayConfig = request.app["cfg"]
    session: aiohttp.ClientSession = request.app["session"]

    try:
        first_cmd = await receive_stream_command(ws, 30, expect_start=True)
    except asyncio.TimeoutError:
        await send_error(ws, "TTS relay baslangic istegi zaman asimina ugradi.", 408)
        await ws.close()
        return ws
    except (ClientConnectionResetError, ValueError) as exc:
        await send_error(ws, str(exc), 400)
        await ws.close()
        return ws

    provider_override = first_cmd.provider
    provider = provider_override or cfg.provider or "openai"

    try:
        if provider == "cartesia":
            _LOG.info(
                "Realtime TTS continuation istegi alindi. provider=%s ilk_karakter=%s final=%s",
                provider,
                len(first_cmd.text),
                first_cmd.final,
            )
            await relay_cartesia_continuation_session(session, ws, cfg, first_cmd)
        else:
            text, provider_override, parca_sayisi = await collect_buffered_stream_text(ws, first_cmd)
            provider = provider_override or cfg.provider or "openai"
            if not text.strip():
                await send_error(ws, "TTS icin metin bos.", 400)
            else:
                _LOG.info(
                    "Realtime TTS istegi alindi. provider=%s karakter=%s parca=%s",
                    provider,
                    len(text),
                    parca_sayisi,
                )
                await relay_stream(session, ws, cfg, text, provider_override)
    except ClientConnectionResetError:
        _LOG.info("ESP istemcisi relay oturumu sirasinda baglantiyi kapatti.")
    except asyncio.TimeoutError as exc:
        await send_error(ws, str(exc), 408)
    except ValueError as exc:
        await send_error(ws, str(exc), 400)
    await ws.close()
    return ws


async def create_session(_: web.Application):
    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_read=120)
    session = aiohttp.ClientSession(timeout=timeout)
    yield session
    await session.close()


def build_app() -> web.Application:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    app = web.Application()
    app["cfg"] = cfg

    async def session_ctx(app_: web.Application):
        async for session in create_session(app_):
            app_["session"] = session
            yield

    app.cleanup_ctx.append(session_ctx)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    app = build_app()
    cfg: RelayConfig = app["cfg"]
    _LOG.info(
        "Alice Realtime TTS add-on basliyor. version=%s port=%s provider=%s pcm_rate=%s pcm_pacing=acik burst_ms=%s",
        ADDON_VERSION,
        cfg.port,
        cfg.provider,
        cfg.pcm_sample_rate,
        PCM_PACE_INITIAL_BURST_MS,
    )
    web.run_app(app, host="0.0.0.0", port=cfg.port)
