import asyncio
import base64
import fnmatch
import json
import logging
import os
import re
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import ClientConnectionResetError, web
from faster_whisper import WhisperModel

_LOG = logging.getLogger("alice_realtime_voice")
OPTIONS_PATH = Path("/data/options.json")
EMOTION_TAG_RE = re.compile(r"<emotion:\s*([^>]+)>", re.IGNORECASE)
INCOMPLETE_EMOTION_TAG_RE = re.compile(r"<emotion:\s*[^>]*$", re.IGNORECASE)
APP_VERSION = "0.9.20"
MAX_HISTORY_MESSAGES = 12
STREAM_CHUNK_MIN_CHARS = 28
STREAM_CHUNK_HARD_CHARS = 90
OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
SILERO_VAD_FRAME_SAMPLES_16K = 512
SILERO_VAD_FRAME_SAMPLES_8K = 256
PRE_SPEECH_DEBUG_INTERVAL_MS = 1000
POST_SPEECH_DEBUG_INTERVAL_MS = 1000
PRE_SPEECH_CALIBRATION_MS = 240
START_MIN_LEVEL_ABS = 500
START_MIN_PEAK_ABS = 1800
EARLY_FORCE_START_LEVEL_ABS = 900
EARLY_FORCE_START_PEAK_ABS = 3500
RESUME_MIN_LEVEL_ABS = 450
RESUME_MIN_PEAK_ABS = 2200
DEFAULT_VOICE_SYSTEM_PROMPT = (
    "Sen Alice adli kisa konusan bir sesli asistansin. Turkce yanit ver. "
    "Yaniti 1-3 kisa cumle tut. Kullanici ozellikle istemedikce kod, madde listesi, "
    "SSML, TTS veya ses dosyasi teknik detayi anlatma."
)


@dataclass
class SttConfig:
    provider: str = "faster_whisper"
    model: str = "small"
    language: str = "tr"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = False


@dataclass
class EndpointingConfig:
    enabled: bool = True
    provider: str = "silero"
    auto_finalize_on_vad_end: bool = False
    no_speech_timeout_ms: int = 5000
    speech_start_min_ms: int = 96
    speech_end_silence_ms: int = 400
    max_utterance_ms: int = 15000
    start_avg_abs_threshold: int = 150
    end_avg_abs_threshold: int = 110
    silero_start_threshold: float = 0.50
    silero_end_threshold: float = 0.28


@dataclass
class HaBridgeConfig:
    enabled: bool = True
    api_base_url: str = "http://supervisor/core/api"
    conversation_agent_id: str = ""
    conversation_language: str = "tr"
    route_home_control: bool = True
    expose_all_entities: bool = False
    exposed_entities: str = ""
    exposed_domains: str = ""
    blocked_entities: str = ""
    allow_conversation_tool: bool = False


@dataclass
class LlmConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    system_prompt: str = ""


@dataclass
class RealtimeConfig:
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-realtime-mini"
    ws_url: str = OPENAI_REALTIME_WS_URL
    input_sample_rate: int = 24000
    turn_detection: str = "server_vad"
    vad_threshold: float = 0.50
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 420
    semantic_eagerness: str = "high"
    transcription_model: str = "gpt-4o-mini-transcribe"
    transcript_wait_ms: int = 500
    response_timeout_ms: int = 12000
    ha_tools_enabled: bool = True
    noise_reduction: str = "near_field"
    instructions: str = ""


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
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
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


def _read_float(raw: dict, key: str, default: float) -> float:
    try:
        return float(raw.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _read_bool(raw: dict, key: str, default: bool = False) -> bool:
    return bool(raw.get(key, default))


def parse_scope_items(value: str) -> list[str]:
    return [item.strip().lower() for item in re.split(r"[\s,;]+", value or "") if item.strip()]


def load_config() -> VoiceConfig:
    if not OPTIONS_PATH.exists():
        return VoiceConfig()

    with OPTIONS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    stt_raw = _read_group(raw, "stt")
    endpointing_raw = _read_group(raw, "endpointing")
    ha_bridge_raw = _read_group(raw, "ha_bridge")
    llm_raw = _read_group(raw, "llm")
    realtime_raw = _read_group(raw, "realtime")
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
            vad_filter=_read_bool(stt_raw, "vad_filter", False),
        ),
        endpointing=EndpointingConfig(
            enabled=_read_bool(endpointing_raw, "enabled", True),
            provider=_read_str(endpointing_raw, "provider", "silero"),
            auto_finalize_on_vad_end=_read_bool(endpointing_raw, "auto_finalize_on_vad_end", False),
            no_speech_timeout_ms=_read_int(endpointing_raw, "no_speech_timeout_ms", 5000),
            speech_start_min_ms=_read_int(endpointing_raw, "speech_start_min_ms", 96),
            speech_end_silence_ms=_read_int(endpointing_raw, "speech_end_silence_ms", 400),
            max_utterance_ms=_read_int(endpointing_raw, "max_utterance_ms", 15000),
            start_avg_abs_threshold=_read_int(endpointing_raw, "start_avg_abs_threshold", 150),
            end_avg_abs_threshold=_read_int(endpointing_raw, "end_avg_abs_threshold", 110),
            silero_start_threshold=_read_float(endpointing_raw, "silero_start_threshold", 0.50),
            silero_end_threshold=_read_float(endpointing_raw, "silero_end_threshold", 0.28),
        ),
        ha_bridge=HaBridgeConfig(
            enabled=_read_bool(ha_bridge_raw, "enabled", True),
            api_base_url=_read_str(ha_bridge_raw, "api_base_url", "http://supervisor/core/api"),
            conversation_agent_id=_read_str(ha_bridge_raw, "conversation_agent_id"),
            conversation_language=_read_str(ha_bridge_raw, "conversation_language", "tr"),
            route_home_control=_read_bool(ha_bridge_raw, "route_home_control", True),
            expose_all_entities=_read_bool(ha_bridge_raw, "expose_all_entities", False),
            exposed_entities=_read_str(ha_bridge_raw, "exposed_entities"),
            exposed_domains=_read_str(ha_bridge_raw, "exposed_domains"),
            blocked_entities=_read_str(ha_bridge_raw, "blocked_entities"),
            allow_conversation_tool=_read_bool(ha_bridge_raw, "allow_conversation_tool", False),
        ),
        llm=LlmConfig(
            provider=_read_str(llm_raw, "provider", "openai"),
            model=_read_str(llm_raw, "model", "gpt-5-mini"),
            api_key=_read_str(llm_raw, "api_key"),
            base_url=_read_str(llm_raw, "base_url", "https://api.openai.com/v1"),
            system_prompt=_read_str(llm_raw, "system_prompt"),
        ),
        realtime=RealtimeConfig(
            enabled=_read_bool(realtime_raw, "enabled", True),
            provider=_read_str(realtime_raw, "provider", "openai").lower() or "openai",
            model=_read_str(realtime_raw, "model", "gpt-realtime-mini"),
            ws_url=_read_str(realtime_raw, "ws_url", OPENAI_REALTIME_WS_URL),
            input_sample_rate=_read_int(realtime_raw, "input_sample_rate", 24000),
            turn_detection=_read_str(realtime_raw, "turn_detection", "server_vad"),
            vad_threshold=_read_float(realtime_raw, "vad_threshold", 0.50),
            prefix_padding_ms=_read_int(realtime_raw, "prefix_padding_ms", 300),
            silence_duration_ms=_read_int(realtime_raw, "silence_duration_ms", 420),
            semantic_eagerness=_read_str(realtime_raw, "semantic_eagerness", "high"),
            transcription_model=_read_str(realtime_raw, "transcription_model", "gpt-4o-mini-transcribe"),
            transcript_wait_ms=_read_int(realtime_raw, "transcript_wait_ms", 500),
            response_timeout_ms=_read_int(realtime_raw, "response_timeout_ms", 12000),
            ha_tools_enabled=_read_bool(realtime_raw, "ha_tools_enabled", True),
            noise_reduction=_read_str(realtime_raw, "noise_reduction", "near_field"),
            instructions=_read_str(realtime_raw, "instructions"),
        ),
        tts=TtsConfig(
            enabled=_read_bool(tts_raw, "enabled", False),
            relay_url=_read_str(tts_raw, "relay_url", "ws://127.0.0.1:8765/ws"),
        ),
    )


def safe_exc_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()


def openai_compatible_base_url(cfg: LlmConfig) -> str:
    if cfg.base_url:
        return cfg.base_url.rstrip("/")
    if cfg.provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    return "https://api.openai.com/v1"


def realtime_ws_url(cfg: RealtimeConfig) -> str:
    base = (cfg.ws_url or OPENAI_REALTIME_WS_URL).strip().rstrip("/")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}model={cfg.model}"


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


def extract_realtime_text_delta(doc: dict) -> str:
    for key in ("delta", "text", "transcript"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_realtime_response_text(doc: dict) -> str:
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


def extract_text_delta(doc: dict) -> str:
    choices = doc.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def turkce_basit_normalize(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized


class HomeAssistantBridge:
    def __init__(self, cfg: HaBridgeConfig) -> None:
        self._cfg = cfg
        self._base_url = cfg.api_base_url.rstrip("/")
        self._token = os.environ.get("SUPERVISOR_TOKEN", "").strip()
        self._exposed_entities = parse_scope_items(cfg.exposed_entities)
        self._exposed_domains = parse_scope_items(cfg.exposed_domains)
        self._blocked_entities = parse_scope_items(cfg.blocked_entities)

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

    def has_entity_scope(self) -> bool:
        return self._cfg.expose_all_entities or bool(self._exposed_entities) or bool(self._exposed_domains)

    @property
    def allow_conversation_tool(self) -> bool:
        return self._cfg.allow_conversation_tool

    def is_entity_allowed(self, entity_id: str) -> bool:
        entity_id = (entity_id or "").strip().lower()
        if not entity_id:
            return False
        if any(fnmatch.fnmatch(entity_id, pattern) for pattern in self._blocked_entities):
            return False
        domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
        if self._cfg.expose_all_entities:
            return True
        if domain and domain in self._exposed_domains:
            return True
        return any(fnmatch.fnmatch(entity_id, pattern) for pattern in self._exposed_entities)

    def assert_entity_allowed(self, entity_id: str) -> None:
        if not self.is_entity_allowed(entity_id):
            raise PermissionError(f"Entity bu add-on icin expose edilmemis: {entity_id}")

    def _service_entity_ids(self, data: dict | None) -> list[str]:
        if not isinstance(data, dict):
            return []
        raw = data.get("entity_id")
        if isinstance(raw, str):
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return []

    async def health(self, session: aiohttp.ClientSession) -> dict:
        if not self._cfg.enabled:
            return {"enabled": False, "connected": False, "reason": "disabled"}
        if not self._token:
            return {"enabled": True, "connected": False, "reason": "missing_supervisor_token"}

        try:
            async with session.get(self._base_url, headers=self._headers()) as resp:
                text = await resp.text()
                return {
                    "enabled": True,
                    "connected": resp.status < 400,
                    "status": resp.status,
                    "body": text[:96],
                }
        except Exception as exc:  # pragma: no cover
            return {"enabled": True, "connected": False, "reason": str(exc)}

    async def get_state(self, session: aiohttp.ClientSession, entity_id: str) -> dict | None:
        self.assert_entity_allowed(entity_id)
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
        async with session.get(f"{self._base_url}/states", headers=self._headers()) as resp:
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
            if not self.is_entity_allowed(entity_id):
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

    async def search_states(
        self,
        session: aiohttp.ClientSession,
        query: str,
        domain: str = "",
        limit: int = 8,
    ) -> list[dict]:
        query_norm = turkce_basit_normalize(query)
        query_terms = [term for term in re.split(r"\s+", query_norm) if term]
        if not query_terms:
            return []

        states = await self.list_states(session, domain=domain, limit=256)
        scored: list[tuple[int, dict]] = []
        for item in states:
            entity_id = str(item.get("entity_id") or "")
            friendly_name = str(item.get("friendly_name") or "")
            haystack = turkce_basit_normalize(f"{entity_id} {friendly_name}")
            score = 0
            for term in query_terms:
                if term in haystack:
                    score += 3
                if haystack.startswith(term) or f".{term}" in haystack:
                    score += 2
            if score <= 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("entity_id") or "")))
        return [item for _score, item in scored[: max(1, min(limit, 20))]]

    async def call_service(
        self,
        session: aiohttp.ClientSession,
        domain: str,
        service: str,
        data: dict | None = None,
    ) -> list | dict | str | None:
        entity_ids = self._service_entity_ids(data)
        if not entity_ids and not self._cfg.expose_all_entities:
            raise PermissionError("Servis cagrisi icin allowlist kapsaminda entity_id gerekli.")
        for entity_id in entity_ids:
            self.assert_entity_allowed(entity_id)
        async with session.post(
            f"{self._base_url}/services/{domain}/{service}",
            headers=self._headers(),
            json=data or {},
        ) as resp:
            resp.raise_for_status()
            if resp.content_type == "application/json":
                return await resp.json()
            return await resp.text()

    async def process_conversation(
        self,
        session: aiohttp.ClientSession,
        text: str,
        language: str = "",
        conversation_id: str = "",
    ) -> dict:
        if not self._cfg.allow_conversation_tool:
            raise PermissionError("ha_conversation tool bu add-on ayarlarinda kapali.")
        payload: dict = {
            "text": text,
            "language": language or self._cfg.conversation_language or "tr",
        }
        if self._cfg.conversation_agent_id:
            payload["agent_id"] = self._cfg.conversation_agent_id
        if conversation_id:
            payload["conversation_id"] = conversation_id

        async with session.post(
            f"{self._base_url}/conversation/process",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def extract_conversation_speech(result: dict) -> str:
        response = result.get("response") or {}
        speech = response.get("speech") or {}
        plain = speech.get("plain") or {}
        if isinstance(plain, dict):
            text = plain.get("speech")
            if isinstance(text, str):
                return text.strip()
        ssml = speech.get("ssml") or {}
        if isinstance(ssml, dict):
            text = ssml.get("speech")
            if isinstance(text, str):
                return text.strip()
        return ""


class WhisperEngine:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self._model: WhisperModel | None = None
        self._vad_override_logged = False

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

    def warm_up(self) -> None:
        self._ensure_model()

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

            if self._cfg.vad_filter and not self._vad_override_logged:
                _LOG.warning(
                    "Whisper dahili vad_filter zorla kapatildi. Dis endpointing kullaniyoruz; sesin tamami korunacak."
                )
                self._vad_override_logged = True

            segments, _info = model.transcribe(
                temp_path,
                language=self._cfg.language or None,
                beam_size=max(1, self._cfg.beam_size),
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


class OpenAICompatibleLlmEngine:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg

    def is_enabled(self) -> bool:
        return self._cfg.provider.strip().lower() != "none" and bool(self._cfg.api_key.strip())

    def _system_prompt(self) -> str:
        custom_prompt = self._cfg.system_prompt.strip()
        if custom_prompt:
            return custom_prompt
        return DEFAULT_VOICE_SYSTEM_PROMPT

    async def stream_reply(
        self,
        session: aiohttp.ClientSession,
        history: list[dict[str, str]],
        user_text: str,
    ):
        if not self.is_enabled():
            return

        messages: list[dict[str, str]] = []
        system_prompt = self._system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        endpoint = f"{openai_compatible_base_url(self._cfg)}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self._cfg.provider.strip().lower() == "openrouter":
            headers["HTTP-Referer"] = "https://local.alice/addons"
            headers["X-Title"] = "Alice Realtime Voice"

        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": True,
        }

        timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=240)
        async with session.post(endpoint, headers=headers, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"LLM hatasi status={resp.status} body={body[:300]}")

            event_lines: list[str] = []
            async for raw_chunk in resp.content:
                if not raw_chunk:
                    continue
                text_chunk = raw_chunk.decode("utf-8", errors="ignore")
                for line in text_chunk.splitlines():
                    if not line.strip():
                        if not event_lines:
                            continue
                        for item in event_lines:
                            if not item.startswith("data:"):
                                continue
                            data = item[5:].strip()
                            if not data:
                                continue
                            if data == "[DONE]":
                                return
                            try:
                                doc = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            delta = extract_text_delta(doc)
                            if delta:
                                yield delta
                        event_lines.clear()
                        continue
                    event_lines.append(line)


class StreamTextProcessor:
    def __init__(self) -> None:
        self._raw_pending = ""
        self._spoken_pending = ""
        self._all_spoken_text = ""

    @property
    def all_text(self) -> str:
        return self._all_spoken_text

    def _strip_emotions(self) -> list[str]:
        emotions: list[str] = []
        while True:
            match = EMOTION_TAG_RE.search(self._raw_pending)
            if not match:
                break
            emotion = match.group(1).strip()
            if emotion:
                emotions.append(emotion)
            self._raw_pending = self._raw_pending[:match.start()] + self._raw_pending[match.end():]
        return emotions

    def _flush_safe_text(self) -> None:
        incomplete = INCOMPLETE_EMOTION_TAG_RE.search(self._raw_pending)
        if incomplete:
            safe_text = self._raw_pending[:incomplete.start()]
            self._raw_pending = self._raw_pending[incomplete.start():]
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
                next_char_ok = idx + 1 >= len(text) or text[idx + 1].isspace() or text[idx + 1] in "\"”'"
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
        return emotions, chunks, self._all_spoken_text.strip()


class RelayTtsStreamer:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        client_ws: web.WebSocketResponse,
        relay_url: str,
    ) -> None:
        self._session = session
        self._client_ws = client_ws
        self._relay_url = relay_url
        self._relay_ws: aiohttp.ClientWebSocketResponse | None = None
        self._forward_task: asyncio.Task[None] | None = None
        self._first_cmd = True
        self._done = asyncio.Event()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    async def _ensure_connected(self) -> None:
        if self._relay_ws is not None:
            return
        self._relay_ws = await self._session.ws_connect(
            self._relay_url,
            timeout=20,
            receive_timeout=240,
            heartbeat=20,
        )
        self._forward_task = asyncio.create_task(self._forward_loop())

    async def _forward_loop(self) -> None:
        assert self._relay_ws is not None
        try:
            async for msg in self._relay_ws:
                if self._client_ws.closed:
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._client_ws.send_bytes(msg.data)
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        self._error = str(self._relay_ws.exception() or "TTS relay websocket hatasi")
                        break
                    continue

                await self._client_ws.send_str(msg.data)
                try:
                    doc = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                msg_type = str(doc.get("type") or "").strip().lower()
                if msg_type == "done":
                    self._done.set()
                    return
                if msg_type == "error":
                    self._error = str(doc.get("message") or "TTS relay hata dondu.")
                    self._done.set()
                    return

            if not self._done.is_set():
                self._done.set()
        except Exception as exc:  # pragma: no cover
            self._error = safe_exc_message(exc)
            self._done.set()

    async def send_text(self, text: str, final: bool) -> None:
        await self._ensure_connected()
        assert self._relay_ws is not None
        msg_type = "start" if self._first_cmd else "append"
        payload = {"type": msg_type, "text": text, "final": final}
        await self._relay_ws.send_json(payload)
        self._first_cmd = False

    async def wait_done(self, timeout: float = 240.0) -> None:
        await asyncio.wait_for(self._done.wait(), timeout=timeout)

    async def close(self) -> None:
        if self._relay_ws is not None and not self._relay_ws.closed:
            try:
                await self._relay_ws.close()
            except Exception:  # pragma: no cover
                pass
        if self._forward_task is not None:
            try:
                await self._forward_task
            except Exception:  # pragma: no cover
                pass


class SileroVadRuntime:
    def __init__(self, sample_rate: int) -> None:
        if sample_rate == 16000:
            self.frame_samples = SILERO_VAD_FRAME_SAMPLES_16K
        elif sample_rate == 8000:
            self.frame_samples = SILERO_VAD_FRAME_SAMPLES_8K
        else:
            raise ValueError(f"Silero VAD sadece 8000/16000 Hz destekler. sample_rate={sample_rate}")

        import onnxruntime  # type: ignore
        from faster_whisper.utils import get_assets_path

        model_path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD modeli bulunamadi: {model_path}")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self.sample_rate = sample_rate
        self.frame_bytes = self.frame_samples * 2
        self.frame_ms = int((self.frame_samples * 1000) / sample_rate)
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context_samples = 64 if sample_rate == 16000 else 32
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)
        self._pcm_buffer = bytearray()

    def _process_frame(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        framed = samples.reshape(1, self.frame_samples)
        model_input = np.concatenate([self._context, framed], axis=1).astype(np.float32, copy=False)
        output, self._h, self._c = self._session.run(
            None,
            {"input": model_input, "h": self._h, "c": self._c},
        )
        self._context = framed[:, -self._context_samples:]
        return float(np.asarray(output).reshape(-1)[0])

    def push_pcm16le(self, chunk: bytes) -> list[float]:
        self._pcm_buffer.extend(chunk)
        probs: list[float] = []
        while len(self._pcm_buffer) >= self.frame_bytes:
            frame = bytes(self._pcm_buffer[:self.frame_bytes])
            del self._pcm_buffer[:self.frame_bytes]
            probs.append(self._process_frame(frame))
        return probs


class VoiceSession:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        engine: WhisperEngine,
        llm_engine: OpenAICompatibleLlmEngine,
        endpointing: EndpointingConfig,
        realtime: RealtimeConfig,
        tts_cfg: TtsConfig,
        bridge: HomeAssistantBridge,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self.ws = ws
        self.engine = engine
        self.llm_engine = llm_engine
        self.endpointing = endpointing
        self.realtime = realtime
        self.tts_cfg = tts_cfg
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
        self._last_raw_avg_abs = 0
        self._last_centered_avg_abs = 0
        self._last_peak_abs = 0
        self._noise_floor_abs = 0
        self._endpoint_debug_last_ms = 0
        self._active_debug_last_ms = 0
        self._vad_processed_audio_ms = 0
        self._silero_vad: SileroVadRuntime | None = None
        self._silero_last_probability = 0.0
        self._active_endpointing_provider = "energy"
        self._stripped_packet_header_count = 0
        self._no_speech_timeout_sent = False
        self._finalizing = False
        self._runtime_generation = 0
        self._history: list[dict[str, str]] = []
        self._ha_conversation_id = ""
        self._realtime_ws: aiohttp.ClientWebSocketResponse | None = None
        self._realtime_reader_task: asyncio.Task[None] | None = None
        self._realtime_processor: StreamTextProcessor | None = None
        self._realtime_transcript_event = asyncio.Event()
        self._realtime_transcript = ""
        self._realtime_transcript_item_id = ""
        self._realtime_stt_result_sent = False
        self._realtime_llm_started = False
        self._realtime_response_requested = False
        self._realtime_response_done = False
        self._realtime_closed_by_us = False
        self._realtime_function_names: dict[str, str] = {}
        self._realtime_function_args: dict[str, str] = {}
        self._realtime_tool_response_ids: set[str] = set()
        self._realtime_tool_calls_inflight = 0

    async def send_event(self, event_type: str, **data) -> None:
        if self.ws.closed:
            return
        payload = {"type": event_type, **data}
        try:
            await self.ws.send_json(payload)
        except (ClientConnectionResetError, ConnectionResetError, OSError):
            _LOG.warning("Voice websocket kapaniyor; '%s' eventi istemciye yazilamadi.", event_type)

    def _reset_runtime_state(self) -> None:
        self._runtime_generation += 1
        self.audio.clear()
        self.started = False
        self._speech_started = False
        self._speech_ended = False
        self._consecutive_voice_ms = 0
        self._silence_ms = 0
        self._received_audio_ms = 0
        self._speech_started_at_ms = 0
        self._last_avg_abs = 0
        self._last_raw_avg_abs = 0
        self._last_centered_avg_abs = 0
        self._last_peak_abs = 0
        self._noise_floor_abs = 0
        self._endpoint_debug_last_ms = 0
        self._active_debug_last_ms = 0
        self._vad_processed_audio_ms = 0
        self._silero_vad = None
        self._silero_last_probability = 0.0
        self._active_endpointing_provider = "energy"
        self._stripped_packet_header_count = 0
        self._no_speech_timeout_sent = False
        self._finalizing = False
        self._realtime_processor = None
        self._realtime_transcript_event = asyncio.Event()
        self._realtime_transcript = ""
        self._realtime_transcript_item_id = ""
        self._realtime_stt_result_sent = False
        self._realtime_llm_started = False
        self._realtime_response_requested = False
        self._realtime_response_done = False
        self._realtime_closed_by_us = False
        self._realtime_function_names = {}
        self._realtime_function_args = {}
        self._realtime_tool_response_ids = set()
        self._realtime_tool_calls_inflight = 0

    def _reset_conversation(self) -> None:
        self._history.clear()
        self._ha_conversation_id = ""

    def _append_history(self, role: str, content: str) -> None:
        clean = content.strip()
        if not clean:
            return
        self._history.append({"role": role, "content": clean})
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]

    @staticmethod
    def _chunk_duration_ms(byte_count: int, sample_rate: int) -> int:
        if sample_rate <= 0 or byte_count <= 1:
            return 0
        sample_count = byte_count // 2
        return int((sample_count * 1000) / sample_rate)

    def _normalize_audio_chunk(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        if len(chunk) & 1:
            header = chunk[0]
            self._stripped_packet_header_count += 1
            if self._stripped_packet_header_count <= 3:
                _LOG.info(
                    "Voice binary paketinde 1 byte handler/header ayiklandi. session_id=%s packet_len=%s header=%s",
                    self.session_id or "-",
                    len(chunk),
                    header,
                )
            chunk = chunk[1:]
        return bytes(chunk)

    def _setup_endpointing_provider(self) -> None:
        provider = self.endpointing.provider.strip().lower()
        if provider != "silero":
            self._active_endpointing_provider = "energy"
            return
        try:
            self._silero_vad = SileroVadRuntime(self.sample_rate)
            self._active_endpointing_provider = "silero"
            _LOG.info(
                "Voice endpointing provider hazir. session_id=%s provider=silero frame_ms=%s start_prob=%.2f end_prob=%.2f",
                self.session_id or "-",
                self._silero_vad.frame_ms,
                self.endpointing.silero_start_threshold,
                self.endpointing.silero_end_threshold,
            )
        except Exception as exc:
            self._silero_vad = None
            self._active_endpointing_provider = "energy"
            _LOG.warning(
                "Silero VAD baslatilamadi; enerji tabanli endpointing fallback kullanilacak. session_id=%s error=%s",
                self.session_id or "-",
                safe_exc_message(exc),
            )

    def _realtime_instructions(self) -> str:
        base_prompt = self.realtime.instructions.strip() or self.llm_engine._system_prompt()
        if self._realtime_ha_tools_available():
            base_prompt = (
                f"{base_prompt}\n\n"
                "Home Assistant cihazlari, hava durumu, sensorler, isik/klima/priz/perde gibi ev islemleri "
                "hakkinda asla tahmin uretme. Bu konularda cevap vermeden once uygun Home Assistant tool'unu "
                "cagir. Cihaz kontrolu veya dogal ev komutlari icin oncelikle ha_conversation kullan. "
                "Belirli entity durumunu okumak icin ha_get_state, entity bulmak icin ha_search_entities kullan. "
                "Bir cihazi acmak, kapatmak veya durumunu degistirmek icin once ilgili entity'yi bul, sonra "
                "ha_call_service tool'unu kullan. Tool basarili olmadan kullaniciya islem tamamlandi deme. "
                "Tool sonucunu kullanarak Turkce, kisa ve dogal cevap ver."
            )
        if not self._history:
            return base_prompt

        lines = [base_prompt, "", "Son konusma baglami:"]
        for item in self._history[-6:]:
            role = "Kullanici" if item["role"] == "user" else "Alice"
            lines.append(f"{role}: {item['content']}")
        return "\n".join(line for line in lines if line is not None)

    def _realtime_ha_tools_available(self) -> bool:
        return (
            self.realtime.ha_tools_enabled
            and self.bridge.is_ready()
            and (self.bridge.has_entity_scope() or self.bridge.allow_conversation_tool)
        )

    def _realtime_ha_tools(self) -> list[dict]:
        if not self._realtime_ha_tools_available():
            return []
        tools: list[dict] = []
        if self.bridge.allow_conversation_tool:
            tools.append(
                {
                "type": "function",
                "name": "ha_conversation",
                "description": (
                    "Home Assistant Assist'e dogal dil komutu veya sorgusu gonderir. "
                    "Ev cihazlarini kontrol etmek, hava durumunu sormak veya HA tarafinda cevaplanacak "
                    "akilli ev istekleri icin bunu kullan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Kullanicinin Home Assistant'a iletilecek Turkce komutu veya sorusu.",
                        }
                    },
                    "required": ["text"],
                },
                }
            )
        if self.bridge.has_entity_scope():
            tools.extend([
            {
                "type": "function",
                "name": "ha_search_entities",
                "description": (
                    "Home Assistant entity listesinden ilgili cihaz/sensor adaylarini arar. "
                    "Tum cihaz listesini isteme; sadece kullanicinin bahsettigi alan veya cihaz icin ara."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Aranacak cihaz, oda veya sensor adi. Ornek: salon isik, mutfak, sicaklik.",
                        },
                        "domain": {
                            "type": "string",
                            "description": "Istege bagli HA domain filtresi. Ornek: light, switch, sensor, climate, weather.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Dondurulecek maksimum aday sayisi. Varsayilan 8.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "ha_get_state",
                "description": "Bilinen bir Home Assistant entity_id icin anlik state ve sade attribute bilgisini getirir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Okunacak entity_id. Ornek: light.salon, weather.home, sensor.salon_temperature.",
                        }
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "type": "function",
                "name": "ha_call_service",
                "description": (
                    "Allowlist icindeki bir Home Assistant entity icin servis cagirir. "
                    "Isik ac/kapat icin domain=light service=turn_on/turn_off kullan. "
                    "Sadece kullanicinin istedigi entity_id uzerinde islem yap."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Servis domain'i. Ornek: light, switch, climate, cover, fan.",
                        },
                        "service": {
                            "type": "string",
                            "description": "Cagrilacak servis. Ornek: turn_on, turn_off, toggle.",
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Islem yapilacak allowlist kapsamindaki entity_id.",
                        },
                        "data": {
                            "type": "object",
                            "description": "Istege bagli ek servis verisi. entity_id disindaki alanlar icin kullanilir.",
                        },
                    },
                    "required": ["domain", "service", "entity_id"],
                },
            },
            ])
        return tools

    def _realtime_turn_detection_config(self) -> dict:
        mode = self.realtime.turn_detection.strip().lower()
        if mode == "semantic_vad":
            eagerness = self.realtime.semantic_eagerness.strip().lower() or "high"
            if eagerness not in {"low", "medium", "high", "auto"}:
                eagerness = "high"
            return {
                "type": "semantic_vad",
                "eagerness": eagerness,
                "create_response": False,
                "interrupt_response": True,
            }
        return {
            "type": "server_vad",
            "threshold": max(0.0, min(1.0, self.realtime.vad_threshold)),
            "prefix_padding_ms": max(0, self.realtime.prefix_padding_ms),
            "silence_duration_ms": max(120, self.realtime.silence_duration_ms),
            "create_response": False,
            "interrupt_response": True,
        }

    def _realtime_session_update_payload(self) -> dict:
        audio_input: dict = {
            "format": {
                "type": "audio/pcm",
                "rate": max(8000, self.realtime.input_sample_rate),
            },
            "turn_detection": self._realtime_turn_detection_config(),
        }

        noise_reduction = self.realtime.noise_reduction.strip().lower()
        if noise_reduction in {"near_field", "far_field"}:
            audio_input["noise_reduction"] = {"type": noise_reduction}
        elif noise_reduction in {"none", "off", "disabled", "kapali"}:
            audio_input["noise_reduction"] = None

        transcription_model = self.realtime.transcription_model.strip()
        if transcription_model:
            audio_input["transcription"] = {
                "model": transcription_model,
                "language": self.language,
            }

        session_payload = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.realtime.model,
                "instructions": self._realtime_instructions(),
                "audio": {"input": audio_input},
            },
        }
        tools = self._realtime_ha_tools()
        if tools:
            session_payload["session"]["tools"] = tools
            session_payload["session"]["tool_choice"] = "auto"
        return session_payload

    async def _send_realtime_json(self, payload: dict) -> None:
        if self._realtime_ws is None or self._realtime_ws.closed:
            return
        await self._realtime_ws.send_str(json.dumps(payload, ensure_ascii=False))

    async def _start_realtime(self) -> bool:
        if not self.realtime.enabled:
            return False
        provider = self.realtime.provider.strip().lower()
        if provider != "openai":
            await self.send_event("error", message=f"Desteklenmeyen realtime provider: {provider}")
            return False
        api_key = self.llm_engine._cfg.api_key.strip()
        if not api_key:
            await self.send_event("error", message="Realtime acik ama OpenAI API key bos.")
            return False

        try:
            ws_timeout = aiohttp.ClientWSTimeout(ws_close=20, ws_receive=240)
            self._realtime_ws = await self.http_session.ws_connect(
                realtime_ws_url(self.realtime),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=ws_timeout,
                heartbeat=20,
            )
            await self._send_realtime_json(self._realtime_session_update_payload())
            self._realtime_processor = StreamTextProcessor()
            self._realtime_reader_task = asyncio.create_task(self._realtime_reader_loop())
            _LOG.info(
                "OpenAI Realtime baglandi. session_id=%s model=%s input_rate=%s turn_detection=%s text_output=evet",
                self.session_id or "-",
                self.realtime.model,
                self.realtime.input_sample_rate,
                self.realtime.turn_detection,
            )
            return True
        except Exception as exc:
            _LOG.exception("OpenAI Realtime baglantisi baslatilamadi")
            await self.send_event("error", message=f"OpenAI Realtime baglantisi baslatilamadi: {safe_exc_message(exc)}")
            await self._close_realtime()
            return False

    async def _close_realtime(self) -> None:
        self._realtime_closed_by_us = True
        ws = self._realtime_ws
        self._realtime_ws = None
        if ws is not None and not ws.closed:
            try:
                await ws.close()
            except Exception:  # pragma: no cover
                pass

        task = self._realtime_reader_task
        self._realtime_reader_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except Exception:  # pragma: no cover
                task.cancel()

    async def close(self) -> None:
        await self._close_realtime()

    def _track_audio_chunk(self, chunk: bytes) -> None:
        chunk_ms = self._chunk_duration_ms(len(chunk), self.sample_rate)
        if chunk_ms <= 0:
            return
        self._received_audio_ms += chunk_ms
        raw_avg_abs, centered_avg_abs, peak_abs = self._pcm_stats_from_chunk(chunk)
        self._last_avg_abs = centered_avg_abs
        self._last_raw_avg_abs = raw_avg_abs
        self._last_centered_avg_abs = centered_avg_abs
        self._last_peak_abs = peak_abs

    async def _send_realtime_audio(self, chunk: bytes) -> None:
        self._track_audio_chunk(chunk)
        if self._realtime_ws is None or self._realtime_ws.closed:
            return
        target_rate = max(8000, self.realtime.input_sample_rate)
        audio = pcm16le_resample_linear(chunk, self.sample_rate, target_rate)
        await self._send_realtime_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )

    async def _emit_realtime_stt_result_once(self, reason: str) -> None:
        if self._realtime_stt_result_sent:
            return
        self._realtime_stt_result_sent = True
        text = self._realtime_transcript.strip()
        await self.send_event(
            "stt_result",
            text=text,
            session_id=self.session_id,
            provider="openai_realtime",
            model=self.realtime.transcription_model,
            reason=reason,
            audio_ms=self._received_audio_ms,
        )

    async def _request_realtime_response_after_transcript_wait(self, generation: int) -> None:
        if generation != self._runtime_generation:
            return
        if self._realtime_response_requested or self._realtime_response_done:
            return

        wait_ms = max(0, self.realtime.transcript_wait_ms)
        if wait_ms and not self._realtime_transcript_event.is_set():
            try:
                await asyncio.wait_for(self._realtime_transcript_event.wait(), timeout=wait_ms / 1000)
            except asyncio.TimeoutError:
                pass

        if generation != self._runtime_generation:
            return
        if self._realtime_response_requested or self._realtime_response_done:
            return
        await self._emit_realtime_stt_result_once("realtime_committed")
        await self._request_realtime_response()

    async def _request_realtime_response(self, force: bool = False) -> None:
        if self._realtime_response_done:
            return
        if not force and self._realtime_response_requested:
            return
        self._realtime_response_requested = True
        await self._send_realtime_json(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["text"],
                    "instructions": self._realtime_instructions(),
                },
            }
        )
        _LOG.info(
            "OpenAI Realtime response.create gonderildi. session_id=%s text_output=evet transcript_len=%s",
            self.session_id or "-",
            len(self._realtime_transcript.strip()),
        )
        asyncio.create_task(self._realtime_response_timeout_watchdog(self._runtime_generation))

    async def _realtime_response_timeout_watchdog(self, generation: int) -> None:
        timeout_ms = max(1500, self.realtime.response_timeout_ms)
        await asyncio.sleep(timeout_ms / 1000)
        if generation != self._runtime_generation:
            return
        if not self._realtime_response_requested or self._realtime_response_done:
            return

        _LOG.warning(
            "OpenAI Realtime response timeout. session_id=%s timeout_ms=%s transcript_len=%s",
            self.session_id or "-",
            timeout_ms,
            len(self._realtime_transcript.strip()),
        )
        await self.send_event(
            "error",
            message=f"OpenAI Realtime response timeout: {timeout_ms} ms",
            recoverable=True,
        )
        await self._finish_realtime_response(reason="openai_realtime_response_timeout")

    async def _realtime_response_fallback_after_speech_stop(self, generation: int) -> None:
        await asyncio.sleep(0.8)
        if generation != self._runtime_generation:
            return
        if self._realtime_response_requested or self._realtime_response_done or not self._speech_ended:
            return
        _LOG.info(
            "OpenAI Realtime committed eventi beklenmeden response fallback calisiyor. session_id=%s",
            self.session_id or "-",
        )
        await self._request_realtime_response_after_transcript_wait(generation)

    async def _finish_realtime_response(
        self,
        doc: dict | None = None,
        reason: str = "openai_realtime_done",
    ) -> None:
        if self._realtime_response_done:
            return
        self._realtime_response_done = True

        await self._emit_realtime_stt_result_once("realtime_response_done")
        processor = self._realtime_processor or StreamTextProcessor()
        final_text = extract_realtime_response_text(doc or {})
        pre_final_chunks: list[str] = []
        if final_text and not processor.all_text.strip():
            emotions, pre_final_chunks = processor.push(final_text)
            for emotion in emotions:
                await self.send_event("emotion", name=emotion)

        emotions, final_chunks, assistant_text = processor.finish()
        for emotion in emotions:
            await self.send_event("emotion", name=emotion)

        chunks_to_send = pre_final_chunks + final_chunks
        if chunks_to_send:
            for chunk in chunks_to_send[:-1]:
                await self.send_event("llm_chunk", text=chunk, final=False)
            await self.send_event("llm_chunk", text=chunks_to_send[-1], final=True)
        else:
            await self.send_event("llm_chunk", text="", final=True)

        if not assistant_text and final_text:
            assistant_text = final_text

        await self.send_event("llm_result", text=assistant_text)
        if self._realtime_transcript.strip():
            self._append_history("user", self._realtime_transcript)
        if assistant_text.strip():
            self._append_history("assistant", assistant_text)

        await self.send_event(
            "session_completed",
            reason=reason,
            audio_ms=self._received_audio_ms,
            assistant_text=assistant_text,
            history_messages=len(self._history),
        )
        _LOG.info(
            "OpenAI Realtime oturum tamamlandi. session_id=%s reason=%s audio_ms=%s transcript_len=%s assistant_len=%s",
            self.session_id or "-",
            reason,
            self._received_audio_ms,
            len(self._realtime_transcript.strip()),
            len(assistant_text.strip()),
        )
        await self._close_realtime()
        self._reset_runtime_state()

    def _realtime_response_contains_tool_call(self, doc: dict) -> bool:
        response = doc.get("response")
        if not isinstance(response, dict):
            return False
        response_id = str(response.get("id") or "")
        if response_id and response_id in self._realtime_tool_response_ids:
            return True
        for item in response.get("output") or []:
            if isinstance(item, dict) and str(item.get("type") or "") == "function_call":
                return True
        return False

    async def _send_realtime_tool_output(self, call_id: str, output: dict) -> None:
        await self._send_realtime_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                },
            }
        )

    async def _execute_realtime_tool_call(
        self,
        name: str,
        arguments: str,
        call_id: str,
        response_id: str,
        generation: int,
    ) -> None:
        if generation != self._runtime_generation:
            return
        self._realtime_tool_calls_inflight += 1
        started_ms = self._received_audio_ms
        try:
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            _LOG.info(
                "OpenAI Realtime tool cagrisi. session_id=%s name=%s call_id=%s response_id=%s args=%s",
                self.session_id or "-",
                name,
                call_id,
                response_id,
                json.dumps(args, ensure_ascii=False)[:240],
            )
            await self.send_event("ha_tool_started", name=name, call_id=call_id)

            result: dict
            if not self.bridge.is_ready():
                result = {"ok": False, "error": "Home Assistant bridge hazir degil."}
            elif name == "ha_conversation":
                text = str(args.get("text") or "").strip()
                if not text:
                    result = {"ok": False, "error": "text gerekli"}
                else:
                    ha_result = await self.bridge.process_conversation(
                        self.http_session,
                        text,
                        language=self.language,
                        conversation_id=self._ha_conversation_id,
                    )
                    self._ha_conversation_id = str(
                        ha_result.get("conversation_id") or self._ha_conversation_id or ""
                    ).strip()
                    response = ha_result.get("response") if isinstance(ha_result.get("response"), dict) else {}
                    result = {
                        "ok": True,
                        "speech": self.bridge.extract_conversation_speech(ha_result),
                        "response_type": response.get("response_type", ""),
                        "continue_conversation": bool(ha_result.get("continue_conversation", False)),
                        "conversation_id": self._ha_conversation_id,
                    }
            elif name == "ha_search_entities":
                query = str(args.get("query") or "").strip()
                domain = str(args.get("domain") or "").strip()
                limit = int(args.get("limit") or 8)
                entities = await self.bridge.search_states(
                    self.http_session,
                    query=query,
                    domain=domain,
                    limit=limit,
                )
                result = {"ok": True, "count": len(entities), "entities": entities}
            elif name == "ha_get_state":
                entity_id = str(args.get("entity_id") or "").strip()
                if not entity_id:
                    result = {"ok": False, "error": "entity_id gerekli"}
                else:
                    state = await self.bridge.get_state(self.http_session, entity_id)
                    if state is None:
                        result = {"ok": False, "error": "entity bulunamadi", "entity_id": entity_id}
                    else:
                        attributes = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}
                        result = {
                            "ok": True,
                            "entity": {
                                "entity_id": state.get("entity_id"),
                                "state": state.get("state"),
                                "friendly_name": attributes.get("friendly_name", ""),
                                "attributes": {
                                    key: value
                                    for key, value in attributes.items()
                                    if key
                                    in {
                                        "friendly_name",
                                        "unit_of_measurement",
                                        "device_class",
                                        "temperature",
                                        "current_temperature",
                                        "humidity",
                                        "forecast",
                                    }
                                },
                            },
                        }
            elif name == "ha_call_service":
                domain = str(args.get("domain") or "").strip().lower()
                service = str(args.get("service") or "").strip()
                entity_id = str(args.get("entity_id") or "").strip()
                data = args.get("data") if isinstance(args.get("data"), dict) else {}
                if not domain or not service or not entity_id:
                    result = {"ok": False, "error": "domain, service ve entity_id gerekli"}
                else:
                    entity_domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
                    if entity_domain and domain != entity_domain:
                        result = {
                            "ok": False,
                            "error": "Servis domain'i entity domain'i ile ayni olmali.",
                            "domain": domain,
                            "entity_domain": entity_domain,
                            "entity_id": entity_id,
                        }
                    else:
                        payload = dict(data)
                        payload["entity_id"] = entity_id
                        service_result = await self.bridge.call_service(
                            self.http_session,
                            domain,
                            service,
                            data=payload,
                        )
                        result = {
                            "ok": True,
                            "domain": domain,
                            "service": service,
                            "entity_id": entity_id,
                            "result": service_result,
                        }
            else:
                result = {"ok": False, "error": f"Desteklenmeyen tool: {name}"}

            await self._send_realtime_tool_output(call_id, result)
            await self.send_event(
                "ha_tool_result",
                name=name,
                call_id=call_id,
                ok=bool(result.get("ok", False)),
            )
        except Exception as exc:
            _LOG.exception("Realtime HA tool hatasi")
            await self._send_realtime_tool_output(
                call_id,
                {"ok": False, "error": safe_exc_message(exc), "tool": name},
            )
            await self.send_event("ha_tool_result", name=name, call_id=call_id, ok=False, error=safe_exc_message(exc))
        finally:
            self._realtime_tool_calls_inflight = max(0, self._realtime_tool_calls_inflight - 1)

        if generation != self._runtime_generation or self._realtime_response_done:
            return
        self._realtime_response_requested = False
        _LOG.info(
            "OpenAI Realtime tool sonucu modele verildi. session_id=%s name=%s call_id=%s audio_ms=%s elapsed_audio_ms=%s",
            self.session_id or "-",
            name,
            call_id,
            self._received_audio_ms,
            max(0, self._received_audio_ms - started_ms),
        )
        await self._request_realtime_response(force=True)

    async def _handle_realtime_event(self, doc: dict) -> None:
        event_type = str(doc.get("type") or "").strip()
        if not event_type:
            return

        if event_type == "error":
            error = doc.get("error") if isinstance(doc.get("error"), dict) else {}
            message = str(error.get("message") or doc.get("message") or "OpenAI Realtime hata dondu.")
            _LOG.warning("OpenAI Realtime hata. session_id=%s message=%s", self.session_id or "-", message)
            await self.send_event("error", message=f"OpenAI Realtime: {message}")
            return

        if event_type == "session.updated":
            _LOG.info("OpenAI Realtime session.updated alindi. session_id=%s", self.session_id or "-")
            return

        if event_type == "input_audio_buffer.speech_started":
            self._speech_started = True
            audio_start_ms = doc.get("audio_start_ms")
            self._speech_started_at_ms = int(audio_start_ms) if audio_start_ms is not None else self._received_audio_ms
            _LOG.info(
                "OpenAI Realtime vad_start. session_id=%s audio_ts=%s",
                self.session_id or "-",
                self._speech_started_at_ms,
            )
            await self.send_event(
                "vad_start",
                audio_ts=self._speech_started_at_ms,
                vad_provider="openai_realtime",
            )
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self._speech_ended = True
            audio_end = doc.get("audio_end_ms")
            audio_end_ms = int(audio_end) if audio_end is not None else self._received_audio_ms
            utterance_ms = max(0, audio_end_ms - self._speech_started_at_ms)
            _LOG.info(
                "OpenAI Realtime vad_end. session_id=%s audio_ts=%s utterance_ms=%s",
                self.session_id or "-",
                audio_end_ms,
                utterance_ms,
            )
            await self.send_event(
                "vad_end",
                audio_ts=audio_end_ms,
                utterance_ms=utterance_ms,
                vad_provider="openai_realtime",
                reason="server_vad",
            )
            asyncio.create_task(self._realtime_response_fallback_after_speech_stop(self._runtime_generation))
            return

        if event_type == "input_audio_buffer.committed":
            asyncio.create_task(self._request_realtime_response_after_transcript_wait(self._runtime_generation))
            return

        if event_type == "conversation.item.input_audio_transcription.delta":
            item_id = str(doc.get("item_id") or "")
            if item_id and item_id != self._realtime_transcript_item_id:
                self._realtime_transcript_item_id = item_id
                self._realtime_transcript = ""
                self._realtime_transcript_event = asyncio.Event()
            delta = extract_realtime_text_delta(doc)
            if delta:
                self._realtime_transcript += delta
                await self.send_event("stt_delta", text=delta, provider="openai_realtime")
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(doc.get("transcript") or "").strip()
            if transcript:
                self._realtime_transcript = transcript
            self._realtime_transcript_event.set()
            if self._realtime_stt_result_sent:
                await self.send_event(
                    "stt_transcript",
                    text=self._realtime_transcript.strip(),
                    provider="openai_realtime",
                    late=True,
                )
            return

        if event_type == "response.created":
            if not self._realtime_llm_started:
                self._realtime_llm_started = True
                await self.send_event("llm_started", model=self.realtime.model, provider="openai_realtime")
            return

        if event_type in {"response.output_item.added", "response.output_item.done", "conversation.item.added", "conversation.item.done"}:
            item = doc.get("item") if isinstance(doc.get("item"), dict) else {}
            if str(item.get("type") or "") == "function_call":
                call_id = str(item.get("call_id") or "")
                name = str(item.get("name") or "")
                response_id = str(doc.get("response_id") or "")
                if call_id and name:
                    self._realtime_function_names[call_id] = name
                if response_id:
                    self._realtime_tool_response_ids.add(response_id)
            return

        if event_type == "response.function_call_arguments.delta":
            call_id = str(doc.get("call_id") or "")
            delta = str(doc.get("delta") or "")
            if call_id and delta:
                self._realtime_function_args[call_id] = self._realtime_function_args.get(call_id, "") + delta
            return

        if event_type == "response.function_call_arguments.done":
            call_id = str(doc.get("call_id") or "")
            response_id = str(doc.get("response_id") or "")
            name = str(doc.get("name") or "") or self._realtime_function_names.get(call_id, "")
            arguments = str(doc.get("arguments") or "") or self._realtime_function_args.get(call_id, "")
            if response_id:
                self._realtime_tool_response_ids.add(response_id)
            if not call_id or not name:
                _LOG.warning(
                    "OpenAI Realtime tool cagrisi eksik bilgiyle geldi. session_id=%s call_id=%s name=%s",
                    self.session_id or "-",
                    call_id,
                    name,
                )
                return
            asyncio.create_task(
                self._execute_realtime_tool_call(
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                    response_id=response_id,
                    generation=self._runtime_generation,
                )
            )
            return

        if event_type in {"response.output_text.delta", "response.text.delta"}:
            if not self._realtime_llm_started:
                self._realtime_llm_started = True
                await self.send_event("llm_started", model=self.realtime.model, provider="openai_realtime")
            delta = extract_realtime_text_delta(doc)
            if not delta:
                return
            await self.send_event("llm_delta", text=delta)
            processor = self._realtime_processor or StreamTextProcessor()
            self._realtime_processor = processor
            emotions, chunks = processor.push(delta)
            for emotion in emotions:
                await self.send_event("emotion", name=emotion)
            for chunk in chunks:
                await self.send_event("llm_chunk", text=chunk, final=False)
            return

        if event_type in {"response.output_text.done", "response.text.done"}:
            await self._finish_realtime_response(doc)
            return

        if event_type == "response.done":
            if self._realtime_tool_calls_inflight > 0 or self._realtime_response_contains_tool_call(doc):
                _LOG.info(
                    "OpenAI Realtime tool response.done final cevap olarak kapatilmadi. session_id=%s inflight=%s",
                    self.session_id or "-",
                    self._realtime_tool_calls_inflight,
                )
                return
            await self._finish_realtime_response(doc)
            return

        if event_type in {"input_audio_buffer.cleared", "rate_limits.updated"}:
            return

        if event_type.startswith("response.") or event_type.startswith("conversation."):
            _LOG.debug("OpenAI Realtime event. session_id=%s type=%s", self.session_id or "-", event_type)

    async def _realtime_reader_loop(self) -> None:
        ws = self._realtime_ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                        break
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(str(ws.exception() or "OpenAI Realtime websocket hatasi"))
                    continue
                try:
                    doc = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await self._handle_realtime_event(doc)
        except Exception as exc:
            if not self._realtime_closed_by_us and not self.ws.closed:
                _LOG.exception("OpenAI Realtime okuyucu hatasi")
                await self.send_event("error", message=f"OpenAI Realtime okuyucu hatasi: {safe_exc_message(exc)}")

    async def _finalize_realtime_from_client_eos(self, reason: str) -> None:
        if self._finalizing:
            return
        self._finalizing = True
        _LOG.info(
            "OpenAI Realtime client eos. session_id=%s reason=%s audio_ms=%s speech_started=%s speech_ended=%s response_requested=%s",
            self.session_id or "-",
            reason,
            self._received_audio_ms,
            self._speech_started,
            self._speech_ended,
            self._realtime_response_requested,
        )

        if not self._speech_started:
            await self.send_event(
                "stt_result",
                text="",
                session_id=self.session_id,
                provider="openai_realtime",
                reason="no_speech_eos",
                audio_ms=self._received_audio_ms,
            )
            await self.send_event(
                "session_completed",
                reason="no_speech_eos",
                audio_ms=self._received_audio_ms,
                assistant_text="",
                history_messages=len(self._history),
            )
            await self._close_realtime()
            self._reset_runtime_state()
            return

        if self._realtime_response_requested:
            await self._emit_realtime_stt_result_once("realtime_response_inflight")
            return
        if self._realtime_response_done:
            return

        if not self._speech_ended:
            try:
                await self._send_realtime_json({"type": "input_audio_buffer.commit"})
            except Exception as exc:
                _LOG.warning(
                    "OpenAI Realtime input commit basarisiz. session_id=%s error=%s",
                    self.session_id or "-",
                    safe_exc_message(exc),
                )
            await self._request_realtime_response_after_transcript_wait(self._runtime_generation)
            return

        await self._request_realtime_response_after_transcript_wait(self._runtime_generation)

    @staticmethod
    def _pcm_stats_from_chunk(data: bytes) -> tuple[int, int, int]:
        if len(data) < 2:
            return 0, 0, 0
        pcm = np.frombuffer(data, dtype="<i2")
        if pcm.size == 0:
            return 0, 0, 0
        samples = pcm.astype(np.int32)
        raw_avg_abs = int(np.abs(samples).mean())
        centered = samples - int(samples.mean())
        centered_avg_abs = int(np.abs(centered).mean())
        centered_peak_abs = int(np.abs(centered).max())
        return raw_avg_abs, centered_avg_abs, centered_peak_abs

    @staticmethod
    def _ema_update(current: int, sample: int, alpha_num: int = 1, alpha_den: int = 10) -> int:
        if current <= 0:
            return sample
        return int(((current * (alpha_den - alpha_num)) + (sample * alpha_num)) / alpha_den)

    @staticmethod
    def _noise_floor_update(current: int, sample: int) -> int:
        if current <= 0:
            return sample
        if sample < current:
            return VoiceSession._ema_update(current, sample, alpha_num=1, alpha_den=4)
        return VoiceSession._ema_update(current, sample, alpha_num=1, alpha_den=40)

    def _dynamic_start_threshold(self) -> int:
        floor = self._noise_floor_abs
        if floor <= 0:
            return self.endpointing.start_avg_abs_threshold
        delta = max(220, min(700, int(floor * 0.60)))
        return max(self.endpointing.start_avg_abs_threshold, floor + delta)

    def _dynamic_end_threshold(self) -> int:
        floor = self._noise_floor_abs
        if floor <= 0:
            return self.endpointing.end_avg_abs_threshold
        delta = max(70, min(320, int(floor * 0.20)))
        return max(self.endpointing.end_avg_abs_threshold, floor + delta)

    def _dynamic_resume_threshold(self) -> int:
        floor = self._noise_floor_abs
        if floor <= 0:
            return RESUME_MIN_LEVEL_ABS
        delta = max(260, min(900, int(floor * 1.15)))
        return max(RESUME_MIN_LEVEL_ABS, floor + delta)

    async def _send_silero_no_speech_timeout(
        self,
        level_abs: int,
        raw_avg_abs: int,
        centered_avg_abs: int,
        peak_abs: int,
    ) -> None:
        if self._no_speech_timeout_sent or self._received_audio_ms < self.endpointing.no_speech_timeout_ms:
            return
        self._no_speech_timeout_sent = True
        _LOG.info(
            "Voice session no_speech_timeout. session_id=%s provider=silero audio_ms=%s probability=%.3f level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s consecutive_ms=%s stripped_headers=%s",
            self.session_id or "-",
            self._received_audio_ms,
            self._silero_last_probability,
            level_abs,
            raw_avg_abs,
            centered_avg_abs,
            peak_abs,
            self._consecutive_voice_ms,
            self._stripped_packet_header_count,
        )
        await self.send_event(
            "no_speech_timeout",
            audio_ts=self._received_audio_ms,
            vad_provider="silero",
            probability=self._silero_last_probability,
            avg_abs=level_abs,
            raw_avg_abs=raw_avg_abs,
            centered_avg_abs=centered_avg_abs,
            peak_abs=peak_abs,
            stripped_headers=self._stripped_packet_header_count,
        )

    async def _process_silero_endpointing(
        self,
        chunk: bytes,
        level_abs: int,
        raw_avg_abs: int,
        centered_avg_abs: int,
        peak_abs: int,
    ) -> None:
        if self._silero_vad is None:
            return

        probabilities = self._silero_vad.push_pcm16le(chunk)
        if not probabilities:
            if not self._speech_started:
                await self._send_silero_no_speech_timeout(level_abs, raw_avg_abs, centered_avg_abs, peak_abs)
            return

        frame_ms = self._silero_vad.frame_ms
        for probability in probabilities:
            self._silero_last_probability = probability
            self._vad_processed_audio_ms += frame_ms

            if not self._speech_started:
                if probability >= self.endpointing.silero_start_threshold:
                    self._consecutive_voice_ms += frame_ms
                else:
                    self._consecutive_voice_ms = 0

                if self._vad_processed_audio_ms - self._endpoint_debug_last_ms >= PRE_SPEECH_DEBUG_INTERVAL_MS:
                    self._endpoint_debug_last_ms = self._vad_processed_audio_ms
                    _LOG.info(
                        "Voice endpointing pre_speech. session_id=%s provider=silero audio_ms=%s probability=%.3f start_prob=%.2f consecutive_ms=%s level_abs=%s peak_abs=%s stripped_headers=%s",
                        self.session_id or "-",
                        self._vad_processed_audio_ms,
                        probability,
                        self.endpointing.silero_start_threshold,
                        self._consecutive_voice_ms,
                        level_abs,
                        peak_abs,
                        self._stripped_packet_header_count,
                    )

                if self._consecutive_voice_ms >= self.endpointing.speech_start_min_ms:
                    self._speech_started = True
                    self._speech_started_at_ms = max(0, self._vad_processed_audio_ms - self._consecutive_voice_ms)
                    self._silence_ms = 0
                    _LOG.info(
                        "Voice session vad_start. session_id=%s provider=silero audio_ts=%s probability=%.3f level_abs=%s peak_abs=%s",
                        self.session_id or "-",
                        self._speech_started_at_ms,
                        probability,
                        level_abs,
                        peak_abs,
                    )
                    await self.send_event(
                        "vad_start",
                        audio_ts=self._speech_started_at_ms,
                        vad_provider="silero",
                        probability=probability,
                        avg_abs=level_abs,
                        raw_avg_abs=raw_avg_abs,
                        centered_avg_abs=centered_avg_abs,
                        peak_abs=peak_abs,
                        stripped_headers=self._stripped_packet_header_count,
                    )
                    continue

                await self._send_silero_no_speech_timeout(level_abs, raw_avg_abs, centered_avg_abs, peak_abs)
                continue

            if self._speech_ended:
                continue

            if probability >= self.endpointing.silero_end_threshold:
                self._silence_ms = 0
            else:
                self._silence_ms += frame_ms

            if self._vad_processed_audio_ms - self._active_debug_last_ms >= POST_SPEECH_DEBUG_INTERVAL_MS:
                self._active_debug_last_ms = self._vad_processed_audio_ms
                _LOG.info(
                    "Voice endpointing active. session_id=%s provider=silero audio_ms=%s probability=%.3f end_prob=%.2f silence_ms=%s level_abs=%s peak_abs=%s",
                    self.session_id or "-",
                    self._vad_processed_audio_ms,
                    probability,
                    self.endpointing.silero_end_threshold,
                    self._silence_ms,
                    level_abs,
                    peak_abs,
                )

            utterance_ms = max(0, self._vad_processed_audio_ms - self._speech_started_at_ms)
            if utterance_ms >= self.endpointing.max_utterance_ms:
                self._speech_ended = True
                _LOG.info(
                    "Voice session max_utterance_reached. session_id=%s provider=silero audio_ts=%s utterance_ms=%s probability=%.3f",
                    self.session_id or "-",
                    self._vad_processed_audio_ms,
                    utterance_ms,
                    probability,
                )
                await self.send_event(
                    "max_utterance_reached",
                    audio_ts=self._vad_processed_audio_ms,
                    utterance_ms=utterance_ms,
                    vad_provider="silero",
                    probability=probability,
                )
                await self.send_event(
                    "vad_end",
                    audio_ts=self._vad_processed_audio_ms,
                    utterance_ms=utterance_ms,
                    vad_provider="silero",
                    probability=probability,
                    reason="max_utterance",
                    stripped_headers=self._stripped_packet_header_count,
                )
                if self.endpointing.auto_finalize_on_vad_end:
                    await self.handle_eos(reason="max_utterance")
                continue

            if self._silence_ms >= self.endpointing.speech_end_silence_ms:
                self._speech_ended = True
                vad_end_ts = max(self._speech_started_at_ms, self._vad_processed_audio_ms - self._silence_ms)
                _LOG.info(
                    "Voice session vad_end. session_id=%s provider=silero audio_ts=%s utterance_ms=%s probability=%.3f silence_ms=%s",
                    self.session_id or "-",
                    vad_end_ts,
                    utterance_ms,
                    probability,
                    self._silence_ms,
                )
                await self.send_event(
                    "vad_end",
                    audio_ts=vad_end_ts,
                    utterance_ms=utterance_ms,
                    vad_provider="silero",
                    probability=probability,
                    reason="silence",
                    stripped_headers=self._stripped_packet_header_count,
                )
                if self.endpointing.auto_finalize_on_vad_end:
                    await self.handle_eos(reason="vad_end")

    async def _process_endpointing(self, chunk: bytes) -> None:
        if not self.endpointing.enabled:
            return

        chunk_ms = self._chunk_duration_ms(len(chunk), self.sample_rate)
        if chunk_ms <= 0:
            return

        self._received_audio_ms += chunk_ms
        raw_avg_abs, centered_avg_abs, peak_abs = self._pcm_stats_from_chunk(chunk)
        level_abs = centered_avg_abs
        self._last_avg_abs = level_abs
        self._last_raw_avg_abs = raw_avg_abs
        self._last_centered_avg_abs = centered_avg_abs
        self._last_peak_abs = peak_abs

        if self._active_endpointing_provider == "silero" and self._silero_vad is not None:
            await self._process_silero_endpointing(
                chunk,
                level_abs,
                raw_avg_abs,
                centered_avg_abs,
                peak_abs,
            )
            return

        if not self._speech_started:
            if self._noise_floor_abs <= 0:
                self._noise_floor_abs = level_abs
            start_threshold = self._dynamic_start_threshold()

            calibrating = self._received_audio_ms < PRE_SPEECH_CALIBRATION_MS
            forced_early_start = (
                level_abs >= max(EARLY_FORCE_START_LEVEL_ABS, start_threshold * 2)
                or peak_abs >= EARLY_FORCE_START_PEAK_ABS
            )
            threshold_start = (
                level_abs >= max(START_MIN_LEVEL_ABS, start_threshold)
                and peak_abs >= START_MIN_PEAK_ABS
            )
            voice_like = (not calibrating and threshold_start) or forced_early_start
            if voice_like:
                self._consecutive_voice_ms += chunk_ms
            else:
                self._consecutive_voice_ms = 0

            if not voice_like or level_abs < self._noise_floor_abs:
                self._noise_floor_abs = self._noise_floor_update(self._noise_floor_abs, level_abs)

            if self._received_audio_ms - self._endpoint_debug_last_ms >= PRE_SPEECH_DEBUG_INTERVAL_MS:
                self._endpoint_debug_last_ms = self._received_audio_ms
                _LOG.info(
                    "Voice endpointing pre_speech. session_id=%s audio_ms=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s start_threshold=%s consecutive_ms=%s calibrating=%s threshold_start=%s forced=%s stripped_headers=%s",
                    self.session_id or "-",
                    self._received_audio_ms,
                    level_abs,
                    raw_avg_abs,
                    centered_avg_abs,
                    peak_abs,
                    self._noise_floor_abs,
                    start_threshold,
                    self._consecutive_voice_ms,
                    calibrating,
                    threshold_start,
                    forced_early_start,
                    self._stripped_packet_header_count,
                )

            if self._consecutive_voice_ms >= self.endpointing.speech_start_min_ms:
                self._speech_started = True
                self._speech_started_at_ms = max(0, self._received_audio_ms - self._consecutive_voice_ms)
                self._silence_ms = 0
                _LOG.info(
                    "Voice session vad_start. session_id=%s audio_ts=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s threshold=%s forced=%s",
                    self.session_id or "-",
                    self._speech_started_at_ms,
                    level_abs,
                    raw_avg_abs,
                    centered_avg_abs,
                    peak_abs,
                    self._noise_floor_abs,
                    start_threshold,
                    forced_early_start,
                )
                await self.send_event(
                    "vad_start",
                    audio_ts=self._speech_started_at_ms,
                    avg_abs=level_abs,
                    raw_avg_abs=raw_avg_abs,
                    centered_avg_abs=centered_avg_abs,
                    peak_abs=peak_abs,
                    noise_floor=self._noise_floor_abs,
                    threshold=start_threshold,
                    forced=forced_early_start,
                    stripped_headers=self._stripped_packet_header_count,
                )
                return

            if (
                not self._no_speech_timeout_sent
                and self._received_audio_ms >= self.endpointing.no_speech_timeout_ms
            ):
                self._no_speech_timeout_sent = True
                _LOG.info(
                    "Voice session no_speech_timeout. session_id=%s audio_ms=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s start_threshold=%s consecutive_ms=%s stripped_headers=%s",
                    self.session_id or "-",
                    self._received_audio_ms,
                    level_abs,
                    raw_avg_abs,
                    centered_avg_abs,
                    peak_abs,
                    self._noise_floor_abs,
                    start_threshold,
                    self._consecutive_voice_ms,
                    self._stripped_packet_header_count,
                )
                await self.send_event(
                    "no_speech_timeout",
                    audio_ts=self._received_audio_ms,
                    avg_abs=level_abs,
                    raw_avg_abs=raw_avg_abs,
                    centered_avg_abs=centered_avg_abs,
                    peak_abs=peak_abs,
                    noise_floor=self._noise_floor_abs,
                    threshold=start_threshold,
                    stripped_headers=self._stripped_packet_header_count,
                )
            return

        if self._speech_ended:
            return

        end_threshold = self._dynamic_end_threshold()
        resume_threshold = self._dynamic_resume_threshold()
        strong_voice = level_abs >= resume_threshold or peak_abs >= RESUME_MIN_PEAK_ABS
        if strong_voice:
            self._silence_ms = 0
        elif level_abs < end_threshold:
            self._silence_ms += chunk_ms
            self._noise_floor_abs = self._noise_floor_update(self._noise_floor_abs, level_abs)
        else:
            self._silence_ms += max(1, chunk_ms // 2)

        if self._received_audio_ms - self._active_debug_last_ms >= POST_SPEECH_DEBUG_INTERVAL_MS:
            self._active_debug_last_ms = self._received_audio_ms
            _LOG.info(
                "Voice endpointing active. session_id=%s audio_ms=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s end_threshold=%s resume_threshold=%s strong_voice=%s silence_ms=%s",
                self.session_id or "-",
                self._received_audio_ms,
                level_abs,
                raw_avg_abs,
                centered_avg_abs,
                peak_abs,
                self._noise_floor_abs,
                end_threshold,
                resume_threshold,
                strong_voice,
                self._silence_ms,
            )

        utterance_ms = max(0, self._received_audio_ms - self._speech_started_at_ms)
        if utterance_ms >= self.endpointing.max_utterance_ms:
            self._speech_ended = True
            _LOG.info(
                "Voice session max_utterance_reached. session_id=%s audio_ts=%s utterance_ms=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s threshold=%s",
                self.session_id or "-",
                self._received_audio_ms,
                utterance_ms,
                level_abs,
                raw_avg_abs,
                centered_avg_abs,
                peak_abs,
                self._noise_floor_abs,
                end_threshold,
            )
            await self.send_event(
                "max_utterance_reached",
                audio_ts=self._received_audio_ms,
                utterance_ms=utterance_ms,
            )
            await self.send_event(
                "vad_end",
                audio_ts=self._received_audio_ms,
                utterance_ms=utterance_ms,
                avg_abs=level_abs,
                raw_avg_abs=raw_avg_abs,
                centered_avg_abs=centered_avg_abs,
                peak_abs=peak_abs,
                noise_floor=self._noise_floor_abs,
                threshold=end_threshold,
                reason="max_utterance",
                stripped_headers=self._stripped_packet_header_count,
            )
            if self.endpointing.auto_finalize_on_vad_end:
                await self.handle_eos(reason="max_utterance")
            return

        if self._silence_ms >= self.endpointing.speech_end_silence_ms:
            self._speech_ended = True
            vad_end_ts = max(self._speech_started_at_ms, self._received_audio_ms - self._silence_ms)
            _LOG.info(
                "Voice session vad_end. session_id=%s audio_ts=%s utterance_ms=%s level_abs=%s raw_avg_abs=%s centered_avg_abs=%s peak_abs=%s noise_floor=%s threshold=%s",
                self.session_id or "-",
                vad_end_ts,
                utterance_ms,
                level_abs,
                raw_avg_abs,
                centered_avg_abs,
                peak_abs,
                self._noise_floor_abs,
                end_threshold,
            )
            await self.send_event(
                "vad_end",
                audio_ts=vad_end_ts,
                utterance_ms=utterance_ms,
                avg_abs=level_abs,
                raw_avg_abs=raw_avg_abs,
                centered_avg_abs=centered_avg_abs,
                peak_abs=peak_abs,
                noise_floor=self._noise_floor_abs,
                threshold=end_threshold,
                reason="silence",
                stripped_headers=self._stripped_packet_header_count,
            )
            if self.endpointing.auto_finalize_on_vad_end:
                await self.handle_eos(reason="vad_end")

    async def handle_start(self, doc: dict) -> None:
        self.sample_rate = int(doc.get("sample_rate") or 16000)
        self.language = str(doc.get("language") or "tr").strip() or "tr"
        self.session_id = str(doc.get("session_id") or "").strip()
        if self._realtime_ws is not None:
            await self._close_realtime()
        self._reset_runtime_state()
        if self.realtime.enabled:
            self._active_endpointing_provider = "openai_realtime"
            realtime_started = await self._start_realtime()
            if not realtime_started:
                return
        else:
            self._setup_endpointing_provider()
        self.started = True
        _LOG.info(
            "Voice session basladi. session_id=%s sample_rate=%s language=%s endpointing=%s endpointing_provider=%s realtime=%s llm=%s tts=%s",
            self.session_id or "-",
            self.sample_rate,
            self.language,
            self.endpointing.enabled,
            self._active_endpointing_provider,
            self.realtime.enabled,
            self.llm_engine.is_enabled(),
            self.tts_cfg.enabled,
        )
        await self.send_event(
            "session_started",
            sample_rate=self.sample_rate,
            language=self.language,
            endpointing_enabled=self.endpointing.enabled,
            endpointing_provider=self._active_endpointing_provider,
            realtime_enabled=self.realtime.enabled,
            realtime_model=self.realtime.model if self.realtime.enabled else "",
            llm_enabled=self.llm_engine.is_enabled(),
            tts_enabled=self.tts_cfg.enabled,
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
            await self.send_event("ha_service_result", ok=False, error="domain ve service gerekli")
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

    async def _run_llm_and_tts(self, user_text: str) -> str:
        if not self.llm_engine.is_enabled() or self.ws.closed:
            return ""

        processor = StreamTextProcessor()
        tts_streamer: RelayTtsStreamer | None = None
        tts_started = False
        assistant_text = ""

        if self.tts_cfg.enabled and self.tts_cfg.relay_url.strip():
            tts_streamer = RelayTtsStreamer(self.http_session, self.ws, self.tts_cfg.relay_url.strip())

        await self.send_event("llm_started", model=self.llm_engine._cfg.model, provider=self.llm_engine._cfg.provider)
        async for delta in self.llm_engine.stream_reply(self.http_session, list(self._history), user_text):
            if self.ws.closed:
                _LOG.warning(
                    "Voice websocket kapandi; LLM akis devam ettirilmiyor. session_id=%s",
                    self.session_id or "-",
                )
                break
            await self.send_event("llm_delta", text=delta)
            emotions, chunks = processor.push(delta)
            for emotion in emotions:
                await self.send_event("emotion", name=emotion)
            for chunk in chunks:
                await self.send_event("llm_chunk", text=chunk, final=False)
                if tts_streamer is not None:
                    await tts_streamer.send_text(chunk, False)
                    tts_started = True

        emotions, final_chunks, assistant_text = processor.finish()
        for emotion in emotions:
            await self.send_event("emotion", name=emotion)

        if final_chunks:
            for chunk in final_chunks[:-1]:
                await self.send_event("llm_chunk", text=chunk, final=False)
                if tts_streamer is not None:
                    await tts_streamer.send_text(chunk, False)
                    tts_started = True

            last_chunk = final_chunks[-1]
            await self.send_event("llm_chunk", text=last_chunk, final=True)
            if tts_streamer is not None:
                await tts_streamer.send_text(last_chunk, True)
                tts_started = True
        elif tts_streamer is not None and tts_started:
            await tts_streamer.send_text("", True)

        await self.send_event("llm_result", text=assistant_text)

        if tts_streamer is not None and tts_started:
            try:
                await tts_streamer.wait_done()
            finally:
                await tts_streamer.close()
            if tts_streamer.error:
                await self.send_event("tts_result", ok=False, error=tts_streamer.error)
            else:
                await self.send_event("tts_result", ok=True)

        return assistant_text

    def _ha_konusmaya_yonlendirilsin_mi(self, user_text: str) -> bool:
        if not self.bridge.is_ready() or not self.bridge._cfg.route_home_control:
            return False

        text = turkce_basit_normalize(user_text)
        weather_terms = ["hava", "derece", "sicaklik", "nem", "ruzgar", "yagmur"]
        device_terms = [
            "isik",
            "lamba",
            "priz",
            "klima",
            "perde",
            "panjur",
            "isitici",
            "fan",
            "garaj",
            "kapi",
            "kilit",
            "tv",
            "televizyon",
            "muzik",
            "alarm",
        ]
        action_terms = [
            "ac",
            "kapat",
            "yak",
            "sondur",
            "ayarla",
            "artir",
            "azalt",
            "baslat",
            "durdur",
            "durumu",
            "acik",
            "kapali",
            "kac",
            "ne durumda",
        ]

        if any(term in text for term in weather_terms):
            return True
        has_device = any(term in text for term in device_terms)
        has_action = any(term in text for term in action_terms)
        return has_device and has_action

    async def _run_ha_conversation_and_tts(self, user_text: str) -> str:
        if self.ws.closed:
            return ""
        await self.send_event("llm_started", model="home_assistant", provider="ha_conversation")
        result = await self.bridge.process_conversation(
            self.http_session,
            user_text,
            language=self.language,
            conversation_id=self._ha_conversation_id,
        )
        self._ha_conversation_id = str(result.get("conversation_id") or self._ha_conversation_id or "").strip()
        response = result.get("response") or {}
        continue_conversation = bool(result.get("continue_conversation", False))
        speech = self.bridge.extract_conversation_speech(result)
        response_type = str(response.get("response_type") or "")

        await self.send_event(
            "ha_conversation_result",
            response_type=response_type,
            continue_conversation=continue_conversation,
            conversation_id=self._ha_conversation_id,
        )

        tts_streamer: RelayTtsStreamer | None = None
        if self.tts_cfg.enabled and self.tts_cfg.relay_url.strip() and speech:
            tts_streamer = RelayTtsStreamer(self.http_session, self.ws, self.tts_cfg.relay_url.strip())

        if speech:
            await self.send_event("llm_chunk", text=speech, final=True)
            if tts_streamer is not None:
                await tts_streamer.send_text(speech, True)

        await self.send_event("llm_result", text=speech)

        if tts_streamer is not None:
            try:
                await tts_streamer.wait_done()
            finally:
                await tts_streamer.close()
            if tts_streamer.error:
                await self.send_event("tts_result", ok=False, error=tts_streamer.error)
            else:
                await self.send_event("tts_result", ok=True)

        return speech

    async def handle_eos(self, reason: str = "client_eos") -> None:
        if not self.started:
            await self.send_event("error", message="Oturum baslatilmadan eos alindi.")
            return
        if self._realtime_ws is not None:
            await self._finalize_realtime_from_client_eos(reason)
            return
        if self._finalizing:
            return

        self._finalizing = True
        _LOG.info(
            "Voice session eos. session_id=%s reason=%s audio_ms=%s bytes=%s",
            self.session_id or "-",
            reason,
            self._received_audio_ms,
            len(self.audio),
        )

        if self.endpointing.enabled and not self._speech_started:
            _LOG.info(
                "Voice session eos no_speech. session_id=%s reason=%s audio_ms=%s provider=%s last_probability=%.3f",
                self.session_id or "-",
                reason,
                self._received_audio_ms,
                self._active_endpointing_provider,
                self._silero_last_probability,
            )
            await self.send_event(
                "stt_result",
                text="",
                session_id=self.session_id,
                reason="no_speech_eos",
                audio_ms=self._received_audio_ms,
                vad_provider=self._active_endpointing_provider,
                probability=self._silero_last_probability,
            )
            await self.send_event(
                "session_completed",
                reason="no_speech_eos",
                audio_ms=self._received_audio_ms,
                assistant_text="",
                history_messages=len(self._history),
            )
            self._reset_runtime_state()
            return

        await self.send_event(
            "stt_started",
            bytes=len(self.audio),
            reason=reason,
            audio_ms=self._received_audio_ms,
            avg_abs=self._last_avg_abs,
            raw_avg_abs=self._last_raw_avg_abs,
            centered_avg_abs=self._last_centered_avg_abs,
            peak_abs=self._last_peak_abs,
            stripped_headers=self._stripped_packet_header_count,
        )
        text = await asyncio.to_thread(self.engine.transcribe_pcm16le, bytes(self.audio), self.sample_rate)
        if self.ws.closed:
            _LOG.warning(
                "Voice websocket kapandi; STT sonrasi akis iptal edildi. session_id=%s",
                self.session_id or "-",
            )
            self._reset_runtime_state()
            return
        await self.send_event("stt_result", text=text, session_id=self.session_id)

        assistant_text = ""
        if text.strip() and (self.llm_engine.is_enabled() or self._ha_konusmaya_yonlendirilsin_mi(text)):
            self._append_history("user", text)
            try:
                if self._ha_konusmaya_yonlendirilsin_mi(text):
                    await self.send_event("ha_route_selected", reason="home_control_keywords")
                    assistant_text = await self._run_ha_conversation_and_tts(text)
                else:
                    assistant_text = await self._run_llm_and_tts(text)
            except Exception as exc:
                _LOG.exception("LLM/TTS akis hatasi")
                await self.send_event("error", message=f"LLM/TTS akis hatasi: {safe_exc_message(exc)}")
            if assistant_text.strip():
                self._append_history("assistant", assistant_text)

        await self.send_event(
            "session_completed",
            reason=reason,
            audio_ms=self._received_audio_ms,
            assistant_text=assistant_text,
            history_messages=len(self._history),
        )
        self._reset_runtime_state()

    async def handle_message(self, msg: aiohttp.WSMessage) -> bool:
        if msg.type == aiohttp.WSMsgType.BINARY:
            if not self.started:
                await self.send_event("error", message="Binary ses verisi start oncesi geldi.")
                return True
            raw_chunk = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
            chunk = self._normalize_audio_chunk(raw_chunk)
            if not chunk:
                return True
            self.audio.extend(chunk)
            if self._realtime_ws is not None:
                await self._send_realtime_audio(chunk)
                return True
            await self._process_endpointing(chunk)
            return True

        if msg.type in {aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG, aiohttp.WSMsgType.CONTINUATION}:
            return True

        if msg.type != aiohttp.WSMsgType.TEXT:
            return True

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
            await self._close_realtime()
            self._reset_runtime_state()
            self._reset_conversation()
            await self.send_event("session_reset")
            return True
        if msg_type == "conversation_reset":
            self._reset_conversation()
            await self.send_event("conversation_reset_done")
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


async def health_handler(request: web.Request) -> web.Response:
    app = request.app
    cfg: VoiceConfig = app["cfg"]
    bridge: HomeAssistantBridge = app["ha_bridge"]
    http_session: aiohttp.ClientSession = app["http_session"]
    ha_status = await bridge.health(http_session)
    return web.json_response(
        {
            "ok": True,
            "service": "alice_realtime_voice",
            "version": APP_VERSION,
            "llm": {
                "provider": cfg.llm.provider,
                "model": cfg.llm.model,
                "configured": bool(cfg.llm.api_key.strip()),
            },
            "realtime": {
                "enabled": cfg.realtime.enabled,
                "provider": cfg.realtime.provider,
                "model": cfg.realtime.model,
                "input_sample_rate": cfg.realtime.input_sample_rate,
                "turn_detection": cfg.realtime.turn_detection,
                "response_timeout_ms": cfg.realtime.response_timeout_ms,
                "ha_tools_enabled": cfg.realtime.ha_tools_enabled,
                "ha_tools_available": cfg.realtime.ha_tools_enabled and bridge.is_ready(),
                "text_output": True,
                "external_tts_expected": not cfg.tts.enabled,
            },
            "tts": {
                "enabled": cfg.tts.enabled,
                "relay_url": cfg.tts.relay_url,
            },
            "endpointing": {
                "enabled": cfg.endpointing.enabled,
                "provider": cfg.endpointing.provider,
                "silero_start_threshold": cfg.endpointing.silero_start_threshold,
                "silero_end_threshold": cfg.endpointing.silero_end_threshold,
            },
            "ha_bridge": ha_status,
            "ha_route_home_control": cfg.ha_bridge.route_home_control,
            "test_mode": {
                "first_live_test": True,
                "esp_uses_voice_ws_for_stt_llm": True,
                "tts_relay_passthrough_expected": not cfg.tts.enabled,
            },
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, protocols=("arduino",))
    await ws.prepare(request)

    engine: WhisperEngine = request.app["stt_engine"]
    llm_engine: OpenAICompatibleLlmEngine = request.app["llm_engine"]
    cfg: VoiceConfig = request.app["cfg"]
    bridge: HomeAssistantBridge = request.app["ha_bridge"]
    http_session: aiohttp.ClientSession = request.app["http_session"]
    session = VoiceSession(ws, engine, llm_engine, cfg.endpointing, cfg.realtime, cfg.tts, bridge, http_session)
    await session.send_event(
        "hello",
        service="alice_realtime_voice",
        version=APP_VERSION,
        endpointing_enabled=cfg.endpointing.enabled,
        endpointing_provider=cfg.endpointing.provider,
        realtime_enabled=cfg.realtime.enabled,
        realtime_provider=cfg.realtime.provider,
        realtime_model=cfg.realtime.model,
        ha_bridge_enabled=cfg.ha_bridge.enabled,
        llm_enabled=llm_engine.is_enabled(),
        tts_enabled=cfg.tts.enabled,
    )

    async for msg in ws:
        if msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
            break
        if msg.type == aiohttp.WSMsgType.ERROR:
            break
        try:
            devam = await session.handle_message(msg)
        except Exception as exc:  # pragma: no cover
            _LOG.exception("Voice websocket mesaji islenirken hata olustu")
            await session.send_event("error", message=f"Voice websocket mesaj isleme hatasi: {safe_exc_message(exc)}")
            break
        if not devam:
            break

    await session.close()
    await ws.close()
    return ws


async def create_http_session(app: web.Application) -> None:
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=240)
    app["http_session"] = aiohttp.ClientSession(timeout=timeout)


async def warmup_stt_engine(app: web.Application) -> None:
    engine: WhisperEngine = app["stt_engine"]
    try:
        await asyncio.to_thread(engine._ensure_model)
        _LOG.info("Whisper model warmup tamamlandi.")
    except Exception:  # pragma: no cover
        _LOG.exception("Whisper model warmup basarisiz")


async def close_http_session(app: web.Application) -> None:
    session: aiohttp.ClientSession | None = app.get("http_session")
    if session is not None:
        await session.close()


def build_app() -> web.Application:
    cfg = load_config()
    level = logging.INFO if cfg.debug_logs else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = web.Application()
    app["cfg"] = cfg
    app["stt_engine"] = WhisperEngine(cfg.stt)
    app["llm_engine"] = OpenAICompatibleLlmEngine(cfg.llm)
    app["ha_bridge"] = HomeAssistantBridge(cfg.ha_bridge)
    app.on_startup.append(create_http_session)
    app.on_cleanup.append(close_http_session)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)

    try:
        if cfg.realtime.enabled:
            _LOG.info("Realtime modu acik; Whisper startup warmup atlandi.")
        elif cfg.stt.provider.strip().lower() == "faster_whisper":
            app["stt_engine"].warm_up()
            _LOG.info("Whisper model startup warmup tamamlandi.")
    except Exception:  # pragma: no cover
        _LOG.exception("Whisper model startup warmup basarisiz")
    return app


if __name__ == "__main__":
    app = build_app()
    cfg: VoiceConfig = app["cfg"]
    _LOG.info(
        "Alice Realtime Voice add-on basliyor. version=%s port=%s realtime=%s realtime_provider=%s realtime_model=%s realtime_turn_detection=%s realtime_ha_tools=%s stt_provider=%s model=%s endpointing=%s endpointing_provider=%s ha_bridge=%s llm=%s tts=%s first_live_test=%s",
        APP_VERSION,
        cfg.port,
        "acik" if cfg.realtime.enabled else "kapali",
        cfg.realtime.provider,
        cfg.realtime.model,
        cfg.realtime.turn_detection,
        "acik" if cfg.realtime.ha_tools_enabled else "kapali",
        cfg.stt.provider,
        cfg.stt.model,
        "acik" if cfg.endpointing.enabled else "kapali",
        cfg.endpointing.provider,
        "acik" if cfg.ha_bridge.enabled else "kapali",
        cfg.llm.provider,
        "acik" if cfg.tts.enabled else "kapali",
        "evet",
    )
    web.run_app(app, host="0.0.0.0", port=cfg.port)
