import asyncio
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
APP_VERSION = "0.9.0"
MAX_HISTORY_MESSAGES = 12
STREAM_CHUNK_MIN_CHARS = 48
STREAM_CHUNK_HARD_CHARS = 110


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
    conversation_agent_id: str = ""
    conversation_language: str = "tr"
    route_home_control: bool = True


@dataclass
class LlmConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
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
            conversation_agent_id=_read_str(ha_bridge_raw, "conversation_agent_id"),
            conversation_language=_read_str(ha_bridge_raw, "conversation_language", "tr"),
            route_home_control=_read_bool(ha_bridge_raw, "route_home_control", True),
        ),
        llm=LlmConfig(
            provider=_read_str(llm_raw, "provider", "openai"),
            model=_read_str(llm_raw, "model", "gpt-5-mini"),
            api_key=_read_str(llm_raw, "api_key"),
            base_url=_read_str(llm_raw, "base_url", "https://api.openai.com/v1"),
            system_prompt=_read_str(llm_raw, "system_prompt"),
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
    ) -> list | dict | str | None:
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


class OpenAICompatibleLlmEngine:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg

    def is_enabled(self) -> bool:
        return self._cfg.provider.strip().lower() != "none" and bool(self._cfg.api_key.strip())

    async def stream_reply(
        self,
        session: aiohttp.ClientSession,
        history: list[dict[str, str]],
        user_text: str,
    ):
        if not self.is_enabled():
            return

        messages: list[dict[str, str]] = []
        if self._cfg.system_prompt.strip():
            messages.append({"role": "system", "content": self._cfg.system_prompt.strip()})
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


class VoiceSession:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        engine: WhisperEngine,
        llm_engine: OpenAICompatibleLlmEngine,
        endpointing: EndpointingConfig,
        tts_cfg: TtsConfig,
        bridge: HomeAssistantBridge,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self.ws = ws
        self.engine = engine
        self.llm_engine = llm_engine
        self.endpointing = endpointing
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
        self._no_speech_timeout_sent = False
        self._finalizing = False
        self._history: list[dict[str, str]] = []
        self._ha_conversation_id = ""

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
                await self.send_event("no_speech_timeout", audio_ts=self._received_audio_ms, avg_abs=avg_abs)

            if self._consecutive_voice_ms >= self.endpointing.speech_start_min_ms:
                self._speech_started = True
                self._speech_started_at_ms = max(0, self._received_audio_ms - self._consecutive_voice_ms)
                self._silence_ms = 0
                await self.send_event("vad_start", audio_ts=self._speech_started_at_ms, avg_abs=avg_abs)
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
            await self.send_event("vad_end", audio_ts=vad_end_ts, utterance_ms=utterance_ms)
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
        if not self.llm_engine.is_enabled():
            return ""

        processor = StreamTextProcessor()
        tts_streamer: RelayTtsStreamer | None = None
        tts_started = False
        assistant_text = ""

        if self.tts_cfg.enabled and self.tts_cfg.relay_url.strip():
            tts_streamer = RelayTtsStreamer(self.http_session, self.ws, self.tts_cfg.relay_url.strip())

        await self.send_event("llm_started", model=self.llm_engine._cfg.model, provider=self.llm_engine._cfg.provider)
        async for delta in self.llm_engine.stream_reply(self.http_session, list(self._history), user_text):
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
        text = await asyncio.to_thread(self.engine.transcribe_pcm16le, bytes(self.audio), self.sample_rate)
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
            "tts": {
                "enabled": cfg.tts.enabled,
                "relay_url": cfg.tts.relay_url,
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
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    engine: WhisperEngine = request.app["stt_engine"]
    llm_engine: OpenAICompatibleLlmEngine = request.app["llm_engine"]
    cfg: VoiceConfig = request.app["cfg"]
    bridge: HomeAssistantBridge = request.app["ha_bridge"]
    http_session: aiohttp.ClientSession = request.app["http_session"]
    session = VoiceSession(ws, engine, llm_engine, cfg.endpointing, cfg.tts, bridge, http_session)
    await session.send_event(
        "hello",
        service="alice_realtime_voice",
        version=APP_VERSION,
        endpointing_enabled=cfg.endpointing.enabled,
        ha_bridge_enabled=cfg.ha_bridge.enabled,
        llm_enabled=llm_engine.is_enabled(),
        tts_enabled=cfg.tts.enabled,
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


async def create_http_session(app: web.Application) -> None:
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=240)
    app["http_session"] = aiohttp.ClientSession(timeout=timeout)


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
    return app


if __name__ == "__main__":
    app = build_app()
    cfg: VoiceConfig = app["cfg"]
    _LOG.info(
        "Alice Realtime Voice add-on basliyor. version=%s port=%s stt_provider=%s model=%s endpointing=%s ha_bridge=%s llm=%s tts=%s first_live_test=%s",
        APP_VERSION,
        cfg.port,
        cfg.stt.provider,
        cfg.stt.model,
        "acik" if cfg.endpointing.enabled else "kapali",
        "acik" if cfg.ha_bridge.enabled else "kapali",
        cfg.llm.provider,
        "acik" if cfg.tts.enabled else "kapali",
        "evet",
    )
    web.run_app(app, host="0.0.0.0", port=cfg.port)
