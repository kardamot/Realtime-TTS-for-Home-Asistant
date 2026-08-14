from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from typing import Any

import aiohttp
from fastapi import WebSocket, WebSocketDisconnect

from app.core.barge_in import realtime_turn_detection
from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.prompt_store import PromptStore
from app.core.ws_hub import WsHub
from app.pipeline.audio_processing import Pcm16LevelMeter, StreamingPcm16Resampler
from app.pipeline.llm.openai_compatible import active_llm_config
from app.system.ha_narrator import HaNarrator
from app.system.ha_safety import sanitize_assistant_output


OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
EMOTION_TAG_RE = re.compile(r"<emotion:\s*([^>]+)>", re.IGNORECASE)
INCOMPLETE_EMOTION_TAG_RE = re.compile(r"<emotion:\s*[^>]*$", re.IGNORECASE)
STREAM_CHUNK_MIN_CHARS = 28
STREAM_CHUNK_HARD_CHARS = 90
REALTIME_TRANSCRIPTION_PROMPT_MAX_CHARS = 1024
HOME_ASSISTANT_RUNTIME_GUARDRAILS = (
    "Runtime guardrails for Home Assistant control:\n"
    "- Alice Control Panel handles Home Assistant state reads and service calls outside the model.\n"
    "- Assistant messages produced by Alice Control Panel from Home Assistant are trusted live snapshots. Treat them as authoritative even though the model did not call Home Assistant itself.\n"
    "- Never retract a recent Home Assistant result or claim that live data is unavailable after Alice Control Panel has supplied that result.\n"
    "- Never invent, print, or narrate tool calls, JSON, service names, entity_id values, ha-* calls, light.turn_on, switch.turn_off, or 'Calling Home Assistant'.\n"
    "- For home-control requests, speak only a short natural Turkish result or ask a brief clarification.\n"
    "- Do not expose internal command syntax to the user."
)
ROBOT_BEHAVIOR_RUNTIME_HINT = (
    "Robot behavior cue:\n"
    "- When a visible emotion would help, include at most one short tag such as <emotion: happy>, <emotion: curious>, "
    "<emotion: thinking>, <emotion: surprised>, <emotion: fear>, <emotion: focused>, <emotion: angry>, or <emotion: neutral>.\n"
    "- The tag is consumed by Alice's eyes/motion controller and is not spoken."
)
CURRENT_TURN_RUNTIME_GUARDRAIL = (
    "Conversation focus:\n"
    "- Answer the newest user utterance directly.\n"
    "- Do not reopen a previous topic unless the newest utterance clearly refers to it.\n"
    "- If the newest transcript is an isolated word, unclear, or likely mistranscribed, briefly ask the user to repeat it. Do not guess its meaning from the previous topic.\n"
    "- For an unclear transcript, do not retract earlier information and do not make capability disclaimers about the previous topic.\n"
    "- A Home Assistant result already present as an assistant message is complete; do not answer it again."
)
HOME_CONTROL_FRAGMENT_TERMS = {
    "hava",
    "derece",
    "sicaklik",
    "nem",
    "ruzgar",
    "yagmur",
    "isik",
    "isig",
    "lamba",
    "led",
    "renk",
    "priz",
    "klima",
    "perde",
    "panjur",
    "fan",
    "sensor",
    "kamera",
    "ses",
    "muzik",
    "tv",
    "nemlendirici",
    "nemlendir",
}
_TR_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u00c7": "C",
        "\u011e": "G",
        "\u0130": "I",
        "\u00d6": "O",
        "\u015e": "S",
        "\u00dc": "U",
    }
)
REALTIME_LATENCY_DELTAS = (
    ("wake_to_openai_ms", "start_received", "openai_connected"),
    ("wake_to_first_audio_ms", "start_received", "first_audio_chunk"),
    ("wake_to_speech_start_ms", "start_received", "speech_started"),
    ("speech_duration_ms", "speech_started", "speech_stopped"),
    ("speech_to_commit_ms", "speech_started", "input_committed"),
    ("speech_to_transcript_ms", "speech_started", "transcription_completed"),
    ("speech_stop_to_commit_ms", "speech_stopped", "input_committed"),
    ("speech_stop_to_transcript_ms", "speech_stopped", "transcription_completed"),
    ("commit_to_transcript_ms", "input_committed", "transcription_completed"),
    ("transcript_to_response_request_ms", "transcription_completed", "response_requested"),
    ("transcript_to_first_delta_ms", "transcription_completed", "first_llm_delta"),
    ("transcript_to_first_tts_ms", "transcription_completed", "first_tts_chunk"),
    ("speech_to_response_request_ms", "speech_started", "response_requested"),
    ("speech_to_first_delta_ms", "speech_started", "first_llm_delta"),
    ("speech_stop_to_first_delta_ms", "speech_stopped", "first_llm_delta"),
    ("commit_to_first_delta_ms", "input_committed", "first_llm_delta"),
    ("response_to_first_delta_ms", "response_requested", "first_llm_delta"),
    ("first_delta_to_first_chunk_ms", "first_llm_delta", "first_tts_chunk"),
    ("tts_text_to_worker_ms", "first_tts_chunk", "tts_worker_started"),
    ("tts_worker_to_request_ms", "tts_worker_started", "google_tts_request_send_start"),
    ("tts_request_to_headers_ms", "google_tts_request_send_start", "google_tts_response_headers_received"),
    ("tts_request_to_first_byte_ms", "google_tts_request_send_start", "google_tts_first_byte_received"),
    ("tts_request_to_first_audio_ms", "google_tts_request_send_start", "google_tts_first_audio_chunk_received"),
    ("tts_first_audio_to_decoded_ms", "google_tts_first_audio_chunk_received", "google_tts_first_audio_chunk_decoded"),
    ("tts_decode_to_esp_chunk_ms", "google_tts_first_audio_chunk_decoded", "first_chunk_sent_to_esp"),
    ("esp_chunk_to_speaker_ms", "first_chunk_sent_to_esp", "speaker_started"),
    ("esp_chunk_to_speaker_finished_ms", "first_chunk_sent_to_esp", "speaker_finished"),
    ("tts_text_to_relay_connect_ms", "first_tts_chunk", "tts_relay_connected"),
    ("tts_text_to_relay_request_ms", "first_tts_chunk", "tts_relay_request_sent"),
    ("tts_text_to_relay_start_ms", "first_tts_chunk", "tts_relay_started"),
    ("tts_text_to_speaker_ms", "first_tts_chunk", "speaker_started"),
    ("tts_text_to_speaker_finished_ms", "first_tts_chunk", "speaker_finished"),
    ("wake_to_speaker_ms", "start_received", "speaker_started"),
    ("wake_to_speaker_finished_ms", "start_received", "speaker_finished"),
    ("llm_to_speaker_ms", "first_llm_delta", "speaker_started"),
    ("speaker_playback_ms", "speaker_started", "speaker_finished"),
    ("wake_to_first_tts_ms", "start_received", "first_tts_chunk"),
    ("wake_to_response_done_ms", "start_received", "response_done"),
    ("wake_to_session_completed_ms", "start_received", "session_completed"),
    ("wake_to_complete_ms", "start_received", "speaker_finished"),
    ("total_ms", "start_received", "speaker_finished"),
)


def looks_like_home_control_fragment(text: str) -> bool:
    normalized = str(text or "").translate(_TR_TRANSLATION_TABLE).lower()
    return any(term in normalized for term in HOME_CONTROL_FRAGMENT_TERMS)
REALTIME_STAGE_DEFINITIONS = (
    ("client_connected", "Client linked", "ESP voice WebSocket reached the add-on"),
    ("start_received", "Turn start", "Wake/manual session reached backend"),
    ("openai_connected", "OpenAI connected", "Realtime socket ready"),
    ("session_update_sent", "Session configured", "Realtime model/VAD/STT settings sent"),
    ("hello_sent", "Hello sent", "Protocol handshake sent to ESP"),
    ("first_audio_chunk", "First mic packet", "First microphone PCM reached OpenAI path"),
    ("speech_started", "Speech started", "Realtime VAD detected speech"),
    ("speech_stopped", "Speech stopped", "Realtime VAD detected end of speech"),
    ("manual_commit_sent", "Manual commit", "ESP/add-on forced audio commit"),
    ("input_committed", "Audio committed", "Input audio buffer accepted for STT"),
    ("first_stt_delta", "First STT text", "Partial transcript started"),
    ("transcription_completed", "STT completed", "Final transcript is available"),
    ("stt_result_sent", "Transcript sent", "Transcript forwarded to the agent step"),
    ("ha_route_completed", "HA route done", "Home Assistant handled the request"),
    ("response_requested", "LLM requested", "Assistant response requested"),
    ("response_created", "LLM started", "OpenAI started the response"),
    ("first_llm_delta", "First LLM text", "First assistant text token arrived"),
    ("first_tts_chunk", "TTS text queued", "First text chunk sent toward the firmware TTS player"),
    ("tts_text_queued", "TTS backend queued", "TTS relay accepted the text for provider synthesis"),
    ("tts_worker_started", "TTS worker started", "Provider-specific TTS worker started"),
    ("tts_relay_ws_connect_start", "TTS relay WS start", "ESP relay WebSocket connection was initiated before provider work"),
    ("tts_relay_ws_connected", "TTS relay WS connected", "ESP relay WebSocket is connected to the add-on"),
    ("google_tts_request_build_start", "Google build start", "Google TTS request payload build started"),
    ("google_tts_request_built", "Google request built", "Google TTS request payload is ready"),
    ("google_tts_request_send_start", "Google request start", "Google TTS HTTP request is being sent"),
    ("google_tts_request_sent", "Google request sent", "HTTP request has been sent; response headers may already be available"),
    ("google_tts_response_headers_received", "Google headers", "Google TTS response headers arrived"),
    ("google_tts_first_byte_received", "Google first byte", "First byte from Google TTS response arrived"),
    ("google_tts_response_body_buffered", "Google body buffered", "Google JSON/base64 response body finished buffering"),
    ("google_tts_first_audio_chunk_received", "Google audio found", "First audio payload extracted from Google response"),
    ("google_tts_first_audio_chunk_decoded", "Google audio decoded", "Base64 audio payload decoded"),
    ("audio_resample_start", "Audio convert start", "Audio format parse/resample stage started"),
    ("audio_resample_done", "Audio convert done", "Audio format parse/resample stage completed"),
    ("google_tts_stream_completed", "Google stream done", "Google streaming TTS response completed"),
    ("first_chunk_sent_to_esp", "First ESP chunk", "First real audio chunk was sent toward ESP"),
    ("esp_first_pcm_reported", "ESP first PCM", "ESP reported the first PCM write"),
    ("speaker_started", "Speaker started", "ESP speaker output actually started"),
    ("tts_relay_connected", "TTS relay connected", "ESP connected to the TTS relay WebSocket"),
    ("tts_relay_request_sent", "TTS request sent", "ESP sent the text request to the TTS relay"),
    ("tts_relay_started", "TTS stream started", "TTS relay returned audio format/start metadata"),
    ("speaker_finished", "Speaker finished", "ESP finished draining the speaker PCM stream"),
    ("google_tts_error", "Google TTS error", "Google TTS returned an error or failed locally"),
    ("response_done", "LLM done", "Assistant response completed"),
    ("session_completed", "Turn completed", "Add-on finished the voice turn"),
    ("response_cancelled", "Cancelled", "Assistant response was cancelled"),
    ("client_cancelled", "Client cancel", "ESP/client cancelled the turn"),
    ("transcript_wait_timeout", "Transcript wait timeout", "LLM path continued after STT wait limit"),
)


def safe_exc_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()


def normalize_esp_pcm_packet(chunk: bytes) -> tuple[bytes, int | None]:
    if not chunk:
        return b"", None
    header: int | None = None
    if len(chunk) & 1:
        header = chunk[0]
        chunk = chunk[1:]
    if len(chunk) & 1:
        chunk = chunk[:-1]
    return bytes(chunk), header


def realtime_ws_url(cfg: dict[str, Any]) -> str:
    base = str(cfg.get("ws_url") or OPENAI_REALTIME_WS_URL).strip().rstrip("/")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}model={str(cfg.get('model') or 'gpt-realtime-mini')}"


def active_realtime_config(config: dict[str, Any]) -> dict[str, Any]:
    realtime = config.get("realtime", {}) if isinstance(config, dict) else {}
    if not isinstance(realtime, dict):
        return {}
    provider = str(realtime.get("provider") or "openai").lower()
    providers = realtime.get("providers", {}) if isinstance(realtime.get("providers"), dict) else {}
    profile = providers.get(provider, {}) if isinstance(providers.get(provider), dict) else {}
    merged = {**realtime, **profile}
    merged["provider"] = provider
    merged["providers"] = providers
    return merged


def extract_realtime_text_delta(doc: dict[str, Any]) -> str:
    for key in ("delta", "text", "transcript"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_realtime_response_text(doc: dict[str, Any]) -> str:
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


def strip_emotion_tags(text: str) -> str:
    return EMOTION_TAG_RE.sub("", str(text or "")).strip()


def append_display_text(current: str, next_text: str) -> str:
    current_clean = strip_emotion_tags(current).rstrip()
    next_clean = strip_emotion_tags(next_text)
    if not next_clean:
        return current_clean
    if not current_clean:
        return next_clean
    needs_space = next_clean[0] not in ".,;:!?)]}%"
    return f"{current_clean}{' ' if needs_space else ''}{next_clean}".strip()


class RealtimeTextChunker:
    def __init__(self) -> None:
        self._raw_pending = ""
        self._spoken_pending = ""
        self._all_spoken_text = ""
        self._display_delta = ""

    @property
    def all_text(self) -> str:
        return self._all_spoken_text.strip()

    @property
    def display_delta(self) -> str:
        return self._display_delta

    def _strip_emotions(self) -> list[str]:
        emotions: list[str] = []
        while True:
            match = EMOTION_TAG_RE.search(self._raw_pending)
            if not match:
                break
            emotion = match.group(1).strip()
            if emotion:
                emotions.append(emotion)
            self._raw_pending = self._raw_pending[: match.start()] + self._raw_pending[match.end() :]
        return emotions

    def _flush_safe_text(self) -> None:
        incomplete = INCOMPLETE_EMOTION_TAG_RE.search(self._raw_pending)
        if incomplete:
            safe_text = self._raw_pending[: incomplete.start()]
            self._raw_pending = self._raw_pending[incomplete.start() :]
        else:
            safe_text = self._raw_pending
            self._raw_pending = ""
        if not safe_text:
            return
        self._spoken_pending += safe_text
        self._all_spoken_text += safe_text
        self._display_delta += safe_text

    def _find_boundary(self) -> int:
        text = self._spoken_pending
        for idx, ch in enumerate(text):
            if ch in ".?!\n" and idx + 1 >= STREAM_CHUNK_MIN_CHARS:
                next_char_ok = idx + 1 >= len(text) or text[idx + 1].isspace() or text[idx + 1] in "\"'"
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
        self._display_delta = ""
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
        return emotions, chunks, self.all_text


class OpenAIRealtimeBridge:
    def __init__(
        self,
        config_store: ConfigStore,
        prompt_store: PromptStore,
        log_bus: LogBus,
        ws_hub: WsHub,
        tts_relay: Any,
        esp_client: Any,
        ha_bridge: Any | None = None,
        ha_narrator: HaNarrator | None = None,
    ) -> None:
        self._config_store = config_store
        self._prompt_store = prompt_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._tts_relay = tts_relay
        self._esp_client = esp_client
        self._ha_bridge = ha_bridge
        self._ha_narrator = ha_narrator
        self._active = False
        self._connected = False
        self._last_event = "idle"
        self._last_error = ""
        self._session_id = ""
        self._model = ""
        self._last_transcript = ""
        self._last_assistant_text = ""
        self._last_tts_text = ""
        self._last_input_audio: dict[str, Any] = {}
        self._message_history: list[dict[str, Any]] = []
        self._message_seq = 0
        self._started_at: float | None = None
        self._latency_session_id = ""
        self._latency_started_monotonic: float | None = None
        self._latency_events: list[dict[str, Any]] = []
        self._latency_summary: dict[str, int] = {}
        self._latency_updated_at: float | None = None
        self._latency_history: list[dict[str, Any]] = []
        self._cancel_event = asyncio.Event()

    async def should_handle_voice_ws(self) -> bool:
        config = await self._config_store.get(include_secrets=True)
        realtime = self._realtime_cfg(config)
        return (
            bool(realtime.get("enabled", False))
            and str(realtime.get("provider") or "openai").lower() == "openai"
            and bool(self._api_key(config, realtime))
        )

    async def status(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=False)
        realtime = self._realtime_cfg(config)
        return {
            "enabled": bool(realtime.get("enabled", False)),
            "provider": str(realtime.get("provider") or "openai"),
            "model": str(realtime.get("model") or "gpt-realtime-mini"),
            "transcription_model": str(realtime.get("transcription_model") or ""),
            "active": self._active,
            "connected": self._connected,
            "session_id": self._session_id,
            "last_event": self._last_event,
            "last_error": self._last_error,
            "last_transcript": self._last_transcript,
            "last_assistant_text": self._last_assistant_text,
            "last_tts_text": self._last_tts_text,
            "last_input_audio": dict(self._last_input_audio),
            "messages": [dict(item) for item in self._message_history[-120:]],
            "uptime_sec": int(time.time() - self._started_at) if self._started_at else 0,
            "latency": self._latency_snapshot(),
        }

    def _remember_message(self, role: str, source: str, text: str, meta: dict[str, Any] | None = None) -> None:
        clean = strip_emotion_tags(text)
        if not clean:
            return
        last = self._message_history[-1] if self._message_history else {}
        if last.get("role") == role and last.get("text") == clean and last.get("source") == source:
            return
        self._message_seq += 1
        self._message_history.append(
            {
                "id": f"rt-{self._message_seq}",
                "ts": time.time(),
                "role": role,
                "source": source,
                "text": clean,
                "meta": meta or {},
            }
        )
        self._message_history = self._message_history[-120:]

    def clear_message_history(self) -> None:
        self._message_history.clear()

    def start_tts_trace_turn(self, text: str, reason: str = "manual_tts") -> str:
        session_id = f"tts-{uuid.uuid4().hex[:10]}"
        self._session_id = session_id
        self._last_event = reason
        self._last_transcript = str(text or "").strip()
        self._last_assistant_text = ""
        self._last_tts_text = str(text or "").strip()
        self._reset_latency(session_id, model=self._model or "tts", source=reason, text_chars=len(self._last_tts_text))
        self._mark_latency("start_received", reason=reason)
        return session_id

    def finish_tts_trace_turn(self, reason: str = "manual_tts_done", text: str = "") -> None:
        if not self._latency_session_id:
            return
        final_text = str(text or self._last_tts_text or "").strip()
        self._last_event = reason
        self._mark_latency("session_completed", reason=reason)
        self._remember_latency_turn(reason, final_text, final_text, 0)

    def _reset_latency(self, session_id: str, **data: Any) -> None:
        self._latency_session_id = session_id
        self._latency_started_monotonic = time.monotonic()
        self._latency_events = []
        self._latency_summary = {}
        self._latency_updated_at = time.time()
        self._mark_latency("client_connected", **data)

    def _mark_latency(self, name: str, **data: Any) -> None:
        started = self._latency_started_monotonic
        elapsed_ms = 0 if started is None else max(0, int((time.monotonic() - started) * 1000))
        clean = {
            key: value
            for key, value in data.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        event = {"name": name, "ms": elapsed_ms, "at": time.time(), **clean}
        self._latency_events.append(event)
        self._latency_events = self._latency_events[-64:]
        self._latency_summary = self._build_latency_summary()
        self._latency_updated_at = event["at"]

    def _build_latency_summary(self) -> dict[str, int]:
        first_by_name: dict[str, int] = {}
        for event in self._latency_events:
            name = str(event.get("name") or "")
            if name and name not in first_by_name:
                first_by_name[name] = int(event.get("ms") or 0)
        summary: dict[str, int] = {}
        for field, start, end in REALTIME_LATENCY_DELTAS:
            if start in first_by_name and end in first_by_name:
                summary[field] = max(0, first_by_name[end] - first_by_name[start])
        return summary

    def _latency_event_detail(self, event: dict[str, Any], fallback: str = "") -> str:
        parts: list[str] = []
        if event.get("reason"):
            parts.append(str(event["reason"]).replace("_", " "))
        if event.get("connect_ms") is not None:
            parts.append(f"connect {int(event['connect_ms'])}ms")
        if event.get("audio_ms") is not None:
            parts.append(f"audio {int(event['audio_ms'])}ms")
        if event.get("audio_ts") is not None:
            parts.append(f"audioTs {int(event['audio_ts'])}ms")
        if event.get("buffered_audio_ms") is not None:
            parts.append(f"buffer {int(event['buffered_audio_ms'])}ms")
        if event.get("chars") is not None:
            parts.append(f"{int(event['chars'])} chars")
        if event.get("wait_ms") is not None:
            parts.append(f"wait {int(event['wait_ms'])}ms")
        if event.get("trace_id"):
            parts.append(str(event["trace_id"]))
        if event.get("turn_id"):
            parts.append(f"turn {event['turn_id']}")
        if event.get("provider"):
            parts.append(str(event["provider"]))
        if event.get("transport"):
            parts.append(str(event["transport"]))
        if event.get("text_chars") is not None:
            parts.append(f"text {int(event['text_chars'])} chars")
        if event.get("text_bytes") is not None:
            parts.append(f"{int(event['text_bytes'])} text bytes")
        if event.get("provider_ms") is not None:
            parts.append(f"provider +{int(event['provider_ms'])}ms")
        if event.get("payload_build_ms") is not None:
            parts.append(f"payload {int(event['payload_build_ms'])}ms")
        if event.get("request_payload_bytes") is not None:
            parts.append(f"payload {int(event['request_payload_bytes'])} bytes")
        if event.get("http_status") is not None:
            parts.append(f"HTTP {int(event['http_status'])}")
        if event.get("retry_after"):
            parts.append(f"retry-after {event['retry_after']}")
        if event.get("response_content_type"):
            parts.append(str(event["response_content_type"]))
        if event.get("response_content_length"):
            parts.append(f"content-length {event['response_content_length']}")
        if event.get("response_bytes") is not None:
            parts.append(f"response {int(event['response_bytes'])} bytes")
        if event.get("response_chunk_count") is not None:
            parts.append(f"{int(event['response_chunk_count'])} response chunks")
        if event.get("first_chunk_bytes") is not None:
            parts.append(f"first byte chunk {int(event['first_chunk_bytes'])} bytes")
        if event.get("audio_bytes") is not None:
            parts.append(f"audio {int(event['audio_bytes'])} bytes")
        if event.get("decoded_audio_bytes") is not None:
            parts.append(f"decoded {int(event['decoded_audio_bytes'])} bytes")
        if event.get("audio_chunk_count") is not None:
            parts.append(f"{int(event['audio_chunk_count'])} audio chunks")
        if event.get("response_buffered") is not None:
            parts.append(f"buffered={bool(event['response_buffered'])}")
        if event.get("streaming_response") is not None:
            parts.append(f"streaming={bool(event['streaming_response'])}")
        if event.get("operation"):
            parts.append(str(event["operation"]))
        if event.get("audio_format"):
            parts.append(str(event["audio_format"]))
        if event.get("resample") is not None:
            parts.append(f"resample={bool(event['resample'])}")
        if event.get("pcm_bytes") is not None:
            parts.append(f"pcm {int(event['pcm_bytes'])} bytes")
        if event.get("total_audio_bytes") is not None:
            parts.append(f"total audio {int(event['total_audio_bytes'])} bytes")
        if event.get("chunk_bytes") is not None:
            parts.append(f"chunk {int(event['chunk_bytes'])} bytes")
        if event.get("initial_buffer_ms") is not None:
            parts.append(f"initial buffer {int(event['initial_buffer_ms'])}ms")
        if event.get("silence_prefix_ms") is not None:
            parts.append(f"silence prefix {int(event['silence_prefix_ms'])}ms")
        if event.get("esp_offset_ms") is not None:
            parts.append(f"ESP stream +{int(event['esp_offset_ms'])}ms")
        if event.get("relay_ms") is not None:
            parts.append(f"stage {int(event['relay_ms'])}ms")
        if event.get("prebuffer_bytes") is not None:
            parts.append(f"prebuffer {int(event['prebuffer_bytes'])} bytes")
        if event.get("source_rate") is not None and event.get("target_rate") is not None:
            parts.append(f"{int(event['source_rate'])}Hz -> {int(event['target_rate'])}Hz")
        if event.get("sample_rate") is not None:
            channels = int(event.get("channels") or 1)
            parts.append(f"{int(event['sample_rate'])}Hz x{channels}")
        if event.get("model"):
            parts.append(str(event["model"]))
        if event.get("note"):
            parts.append(str(event["note"]))
        return "; ".join(parts) or fallback

    def _first_latency_event(
        self,
        name: str,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        selected = events if events is not None else self._latency_events
        for event in selected:
            if str(event.get("name") or "") == name:
                return event
        return None

    def _latency_stages(self, events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        selected = events if events is not None else self._latency_events
        first_by_name: dict[str, dict[str, Any]] = {}
        for event in selected:
            name = str(event.get("name") or "")
            if name and name not in first_by_name:
                first_by_name[name] = event
        stages: list[dict[str, Any]] = []
        for key, label, fallback in REALTIME_STAGE_DEFINITIONS:
            event = first_by_name.get(key)
            if not event:
                continue
            stages.append(
                {
                    "key": key,
                    "label": label,
                    "ms": int(event.get("ms") or 0),
                    "detail": self._latency_event_detail(event, fallback),
                    "at": event.get("at"),
                }
            )
        return stages

    def _remember_latency_turn(self, reason: str, transcript: str, assistant_text: str, audio_ms: int) -> None:
        if not self._latency_events:
            return
        events = [dict(event) for event in self._latency_events]
        first_event = events[0] if events else {}
        turn = {
            "session_id": self._latency_session_id,
            "started_at": first_event.get("at"),
            "ended_at": time.time(),
            "reason": reason,
            "transcript": str(transcript or "").strip(),
            "assistant_text": str(assistant_text or "").strip(),
            "audio_ms": int(audio_ms or 0),
            "summary": dict(self._latency_summary),
            "stages": self._latency_stages(events),
            "events": events[-64:],
        }
        if self._latency_history and self._latency_history[-1].get("session_id") == self._latency_session_id:
            self._latency_history[-1] = turn
        else:
            self._latency_history.append(turn)
        self._latency_history = self._latency_history[-12:]

    def _refresh_latest_latency_turn(self) -> None:
        if not self._latency_history:
            return
        if self._latency_history[-1].get("session_id") != self._latency_session_id:
            return
        events = [dict(event) for event in self._latency_events]
        self._latency_history[-1]["summary"] = dict(self._latency_summary)
        self._latency_history[-1]["stages"] = self._latency_stages(events)
        self._latency_history[-1]["events"] = events[-64:]
        self._latency_history[-1]["ended_at"] = self._latency_updated_at or time.time()

    async def record_esp_tts_timing(self, payload: dict[str, Any]) -> None:
        if not self._latency_session_id or not isinstance(payload, dict):
            return
        event_name = str(payload.get("event") or "").strip()
        if event_name not in {
            "tts_relay_connected",
            "tts_relay_request_sent",
            "tts_relay_started",
            "speaker_first_audio",
            "speaker_audio_finished",
        }:
            return
        data: dict[str, Any] = {}
        for key in ("esp_offset_ms", "relay_ms", "prebuffer_bytes", "sample_rate", "channels", "esp_millis"):
            value = payload.get(key)
            if value is not None:
                data[key] = value
        if event_name == "speaker_first_audio":
            self._last_event = "speaker_started"
            self._mark_latency("esp_first_pcm_reported", **data)
            self._mark_latency("speaker_started", **data)
        elif event_name == "speaker_audio_finished":
            self._last_event = "speaker_finished"
            self._mark_latency("speaker_finished", **data)
        else:
            self._last_event = event_name
            self._mark_latency(event_name, **data)
        self._refresh_latest_latency_turn()

    async def record_tts_trace(self, payload: dict[str, Any]) -> None:
        if not self._latency_session_id or not isinstance(payload, dict):
            return
        event_name = str(payload.get("event") or payload.get("name") or "").strip()
        if event_name not in {
            "tts_text_queued",
            "tts_worker_started",
            "tts_relay_ws_connect_start",
            "tts_relay_ws_connected",
            "google_tts_request_build_start",
            "google_tts_request_built",
            "google_tts_request_send_start",
            "google_tts_request_sent",
            "google_tts_response_headers_received",
            "google_tts_first_byte_received",
            "google_tts_response_body_buffered",
            "google_tts_first_audio_chunk_received",
            "google_tts_first_audio_chunk_decoded",
            "audio_resample_start",
            "audio_resample_done",
            "google_tts_stream_completed",
            "first_chunk_sent_to_esp",
            "google_tts_error",
        }:
            return
        data = {
            key: value
            for key, value in payload.items()
            if key not in {"event", "name", "ms", "at"} and (value is None or isinstance(value, (str, int, float, bool)))
        }
        if payload.get("ms") is not None:
            data["provider_ms"] = payload.get("ms")
        data["turn_id"] = self._latency_session_id
        self._last_event = event_name
        self._mark_latency(event_name, **data)
        self._refresh_latest_latency_turn()

    def _speaker_first_audio_snapshot(self) -> dict[str, Any]:
        event = self._first_latency_event("speaker_started") or self._first_latency_event("speaker_first_audio")
        if not event:
            return {
                "available": False,
                "note": "ESP does not report first speaker PCM yet; Wake -> TTS text is the current proxy.",
            }
        return {
            "available": True,
            "ms": int(event.get("ms") or 0),
            "at": event.get("at"),
            "esp_offset_ms": event.get("esp_offset_ms"),
            "prebuffer_bytes": event.get("prebuffer_bytes"),
            "sample_rate": event.get("sample_rate"),
            "channels": event.get("channels"),
        }

    def _latency_snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self._latency_session_id,
            "events": [dict(event) for event in self._latency_events],
            "summary": dict(self._latency_summary),
            "stages": self._latency_stages(),
            "history": [dict(item) for item in self._latency_history[-12:]],
            "updated_at": self._latency_updated_at,
            "speaker_first_audio": self._speaker_first_audio_snapshot(),
        }

    async def websocket_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        config = await self._config_store.get(include_secrets=True)
        realtime = self._realtime_cfg(config)
        pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
        barge_in_enabled = bool(pipeline.get("barge_in_enabled", True))
        api_key = self._api_key(config, realtime)
        source_sample_rate = 16000
        target_sample_rate = max(8000, int(realtime.get("input_sample_rate") or 24000))
        resampler = StreamingPcm16Resampler(source_sample_rate, target_sample_rate)
        input_level_meter = Pcm16LevelMeter()
        input_audio_metadata: dict[str, Any] = {}
        language = str(config.get("stt", {}).get("language") or "tr") if isinstance(config.get("stt"), dict) else "tr"
        session_id = f"rt-{uuid.uuid4().hex[:10]}"
        transcript = ""
        assistant_text = ""
        text_chunker = RealtimeTextChunker()
        tts_chunk_started = False
        emotion_sent = False
        response_requested = False
        response_done = False
        stt_result_sent = False
        stt_message_sent = False
        assistant_message_sent = False
        llm_started_sent = False
        technical_output_blocked = False
        speech_started = False
        audio_ms = 0
        buffered_audio_ms = 0
        input_committed = False
        first_audio_marked = False
        first_transcript_delta_marked = False
        first_llm_delta_marked = False
        first_tts_chunk_marked = False
        stripped_packet_headers = 0
        transcript_event = asyncio.Event()
        response_wait_task: asyncio.Task[None] | None = None
        realtime_ws: aiohttp.ClientWebSocketResponse | None = None
        client_session: aiohttp.ClientSession | None = None
        reader_task: asyncio.Task[None] | None = None
        self._cancel_event = asyncio.Event()
        self._active = True
        self._connected = False
        self._last_error = ""
        self._last_event = "connecting"
        self._session_id = session_id
        self._model = str(realtime.get("model") or "gpt-realtime-mini")
        self._started_at = time.time()
        self._reset_latency(session_id, model=self._model)

        async def send_event(event_type: str, **data: Any) -> None:
            payload = {"type": event_type, **data}
            try:
                await websocket.send_json(payload)
            except Exception:
                pass
            await self._ws_hub.publish("pipeline_status", {"realtime": await self.status()})

        async def send_realtime_json(payload: dict[str, Any]) -> None:
            if realtime_ws is not None and not realtime_ws.closed:
                await realtime_ws.send_str(json.dumps(payload, ensure_ascii=False))

        def input_audio_snapshot() -> dict[str, Any]:
            return {
                **input_level_meter.summary(),
                "source_rate": source_sample_rate,
                "target_rate": target_sample_rate,
                "resampler": resampler.method,
                "resampler_delay_ms": round(resampler.delay_ms, 3),
                **input_audio_metadata,
            }

        async def remember_local_assistant_reply(text: str, source: str) -> None:
            clean = str(text or "").strip()
            if not clean:
                return
            await send_realtime_json(
                {
                    "type": "conversation.item.create",
                    "event_id": f"alice-{source}-{uuid.uuid4().hex[:12]}",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": clean}],
                    },
                }
            )
            self._mark_latency("local_assistant_context_synced", source=source, chars=len(clean))

        async def request_response(force: bool = False) -> None:
            nonlocal response_requested
            if response_done or (response_requested and not force):
                return
            response_requested = True
            await send_realtime_json(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["text"],
                        "instructions": await self._instructions(config, realtime),
                    },
                }
            )
            self._last_event = "response_requested"
            self._mark_latency("response_requested")
            await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime response requested", {"session_id": session_id})

        async def send_stt_result_once(reason: str) -> None:
            nonlocal stt_result_sent, stt_message_sent
            if stt_result_sent:
                return
            stt_result_sent = True
            self._last_transcript = transcript.strip()
            if self._last_transcript and not stt_message_sent:
                self._remember_message("user", "openai_realtime_stt", self._last_transcript, {"reason": reason, "model": self._model})
                stt_message_sent = True
            self._mark_latency("stt_result_sent", reason=reason)
            await send_event("stt_result", text=transcript.strip(), provider="openai_realtime", reason=reason)

        async def send_llm_started_once() -> None:
            nonlocal llm_started_sent
            if llm_started_sent:
                return
            llm_started_sent = True
            self._mark_latency("llm_started")
            await send_event("llm_started", model=self._model, provider="openai_realtime")

        async def send_tts_chunk(text: str, final: bool) -> None:
            nonlocal first_tts_chunk_marked, technical_output_blocked, tts_chunk_started
            safe_text = sanitize_assistant_output(text)
            if safe_text != str(text or "").strip():
                text = "" if technical_output_blocked else safe_text
                technical_output_blocked = True
                await self._log_bus.emit(
                    "WARN",
                    "HA",
                    "Technical Home Assistant-shaped assistant output blocked",
                    {"session_id": session_id},
                )
            if text.strip():
                tts_chunk_started = True
                self._last_tts_text = append_display_text(self._last_tts_text, text)
                if not first_tts_chunk_marked:
                    first_tts_chunk_marked = True
                    self._mark_latency("first_tts_chunk", chars=len(text))
            await send_event("llm_chunk", text=text, final=final)

        async def send_emotion_once(emotions: list[str]) -> None:
            nonlocal emotion_sent
            if emotion_sent:
                return
            clean_emotions = [str(emotion or "").strip() for emotion in emotions if str(emotion or "").strip()]
            if not clean_emotions:
                return
            emotion_sent = True
            await send_event("emotion", name=clean_emotions[0])

        async def request_response_after_transcript_wait(reason: str = "realtime_committed") -> None:
            nonlocal assistant_text, response_requested, text_chunker
            if response_requested or response_done:
                return
            wait_ms = max(0, int(realtime.get("transcript_wait_ms") or 800))
            if wait_ms and not transcript_event.is_set():
                try:
                    await asyncio.wait_for(transcript_event.wait(), timeout=wait_ms / 1000)
                except asyncio.TimeoutError:
                    self._mark_latency("transcript_wait_timeout", wait_ms=wait_ms)
                    pass

            home_control_candidate = False
            if self._ha_bridge is not None and transcript.strip():
                try:
                    home_control_candidate = await self._ha_bridge.is_home_control_candidate(
                        transcript,
                        partial=not transcript_event.is_set(),
                    )
                except Exception as exc:
                    await self._log_bus.emit(
                        "WARN",
                        "HA",
                        "Realtime HA candidate check failed",
                        {"session_id": session_id, "error": safe_exc_message(exc)},
                    )
                    home_control_candidate = looks_like_home_control_fragment(transcript)

            if home_control_candidate and not transcript_event.is_set():
                self._mark_latency("home_control_candidate", transcript_chars=len(transcript))
                extra_wait_ms = max(0, int(realtime.get("home_control_transcript_wait_ms") or 1600))
                if extra_wait_ms:
                    try:
                        await asyncio.wait_for(transcript_event.wait(), timeout=extra_wait_ms / 1000)
                    except asyncio.TimeoutError:
                        self._mark_latency("home_control_transcript_wait_timeout", wait_ms=extra_wait_ms)
                        pass
            if response_requested or response_done:
                return
            if bool(realtime.get("suppress_empty_transcript_response", True)) and not transcript.strip():
                await send_stt_result_once("empty_transcript_suppressed")
                await finish_response(reason="empty_transcript_suppressed")
                return
            await send_stt_result_once(reason)
            if await try_home_assistant_route(reason):
                return
            if home_control_candidate and not transcript_event.is_set():
                speech = self._ha_bridge.route_error_speech("transcript")
                await finish_local_response(speech, "home_control_transcript_timeout", sync_context=True)
                return
            await request_response()

        async def finish_response(doc: dict[str, Any] | None = None, reason: str = "openai_realtime_done") -> None:
            nonlocal response_done, assistant_text, text_chunker, tts_chunk_started, assistant_message_sent, technical_output_blocked
            if response_done:
                return
            response_done = True
            final_text = extract_realtime_response_text(doc or {})
            ready_chunks: list[str] = []
            ready_emotions: list[str] = []
            if final_text and not assistant_text.strip():
                assistant_text = final_text
                text_chunker = RealtimeTextChunker()
                ready_emotions, ready_chunks = text_chunker.push(final_text)
            elif assistant_text.strip() and not text_chunker.all_text:
                text_chunker = RealtimeTextChunker()
                ready_emotions, ready_chunks = text_chunker.push(assistant_text)

            safe_assistant_text = sanitize_assistant_output(assistant_text)
            if safe_assistant_text != assistant_text.strip():
                technical_output_blocked = True
                assistant_text = safe_assistant_text
                text_chunker = RealtimeTextChunker()
                ready_emotions, ready_chunks = text_chunker.push(assistant_text)

            emotions, final_chunks, spoken_text = text_chunker.finish()
            await send_emotion_once([*ready_emotions, *emotions])
            if spoken_text:
                assistant_text = spoken_text

            self._last_transcript = transcript.strip()
            self._last_assistant_text = assistant_text.strip()
            chunks_to_send = [*ready_chunks, *final_chunks]
            if chunks_to_send:
                for chunk in chunks_to_send[:-1]:
                    await send_tts_chunk(chunk, final=False)
                await send_tts_chunk(chunks_to_send[-1], final=True)
            elif assistant_text.strip() and tts_chunk_started:
                await send_tts_chunk("", final=True)
            else:
                await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime completed without assistant text", {"session_id": session_id, "reason": reason})
            if assistant_text.strip() and not self._last_tts_text.strip():
                self._last_tts_text = assistant_text.strip()
            if assistant_text.strip():
                if not assistant_message_sent:
                    self._remember_message("assistant", "openai_realtime", assistant_text.strip(), {"reason": reason, "model": self._model})
                    assistant_message_sent = True
                await send_event("llm_result", text=assistant_text)
            self._mark_latency("session_completed", reason=reason, audio_ms=audio_ms)
            self._remember_latency_turn(reason, transcript, assistant_text, audio_ms)
            self._last_input_audio = input_audio_snapshot()
            await send_event(
                "session_completed",
                reason=reason,
                audio_ms=audio_ms,
                assistant_text=assistant_text,
                transcript=transcript,
                input_audio=self._last_input_audio,
            )
            self._last_event = "completed"
            latency = self._latency_snapshot()
            await self._log_bus.emit(
                "INFO",
                "PIPELINE",
                "OpenAI Realtime session completed",
                {
                    "session_id": session_id,
                    "transcript_chars": len(transcript),
                    "assistant_chars": len(assistant_text),
                    "input_audio": self._last_input_audio,
                    "latency": latency.get("summary", {}),
                },
            )
            # The ESP voice client consumes llm_chunk/llm_result events and opens
            # the TTS relay itself. Starting a second direct ESP audio stream here
            # races the firmware player and can trigger tts_still_active.

        async def finish_local_response(speech: str, reason: str, *, sync_context: bool) -> bool:
            nonlocal assistant_text, response_requested, text_chunker
            clean = str(speech or "").strip()
            if not clean or response_done:
                return False
            response_requested = True
            assistant_text = clean
            text_chunker = RealtimeTextChunker()
            await send_llm_started_once()
            if sync_context:
                try:
                    await remember_local_assistant_reply(clean, reason)
                except Exception as exc:
                    await self._log_bus.emit(
                        "WARN",
                        "PIPELINE",
                        "Local reply could not be synced to Realtime context",
                        {"session_id": session_id, "reason": reason, "error": safe_exc_message(exc)},
                    )
            await finish_response(reason=reason)
            return True

        async def try_home_assistant_route(reason: str) -> bool:
            if self._ha_bridge is None or response_done or response_requested:
                return False
            user_text = transcript.strip()
            if not user_text:
                return False
            try:
                if not await self._ha_bridge.should_route_home_control(user_text):
                    return False
                result = await self._ha_bridge.handle_text_command(user_text)
                if not result.get("handled"):
                    return False
                speech = str(result.get("speech") or "").strip()
                if self._ha_narrator is not None and result.get("ok"):
                    speech = (await self._ha_narrator.narrate(user_text, result, speech)).strip()
                if not speech:
                    return False
                self._last_event = "ha_route"
                self._mark_latency("ha_route_completed")
                await self._log_bus.emit(
                    "INFO",
                    "HA",
                    "Realtime HA route completed",
                    {
                        "session_id": session_id,
                        "reason": reason,
                        "ok": bool(result.get("ok")),
                        "entity_id": result.get("entity_id"),
                        "action": result.get("action"),
                    },
                )
                return await finish_local_response(speech, "ha_route", sync_context=True)
            except PermissionError as exc:
                await self._log_bus.emit("WARN", "HA", "Realtime HA route denied", {"session_id": session_id, "error": str(exc)})
                return await finish_local_response(
                    self._ha_bridge.route_error_speech("permission"),
                    "ha_route_denied",
                    sync_context=True,
                )
            except Exception as exc:
                await self._log_bus.emit("ERROR", "HA", "Realtime HA route failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                return await finish_local_response(
                    self._ha_bridge.route_error_speech("connection"),
                    "ha_route_failed",
                    sync_context=True,
                )

        async def handle_realtime_event(doc: dict[str, Any]) -> None:
            nonlocal first_llm_delta_marked, first_transcript_delta_marked, input_committed, response_requested, response_done, response_wait_task, speech_started, technical_output_blocked, text_chunker, transcript, assistant_text, stt_message_sent
            event_type = str(doc.get("type") or "")
            if not event_type:
                return
            self._last_event = event_type
            if event_type == "error":
                error = doc.get("error") if isinstance(doc.get("error"), dict) else {}
                message = str(error.get("message") or doc.get("message") or "OpenAI Realtime error")
                self._last_error = message
                await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime error", {"session_id": session_id, "error": message})
                await send_event("error", message=f"OpenAI Realtime: {message}")
                return
            if event_type == "session.updated":
                self._mark_latency("session_updated")
                await send_event("realtime_session_updated", model=self._model)
                return
            if event_type == "input_audio_buffer.speech_started":
                speech_started = True
                self._mark_latency("speech_started", audio_ts=doc.get("audio_start_ms"))
                if barge_in_enabled:
                    await self._cancel_playback("realtime_barge_in")
                    self._mark_latency("barge_in_cancel_sent")
                await send_event("vad_start", vad_provider="openai_realtime", audio_ts=doc.get("audio_start_ms"))
                return
            if event_type == "input_audio_buffer.speech_stopped":
                self._mark_latency("speech_stopped", audio_ts=doc.get("audio_end_ms"))
                await send_event("vad_end", vad_provider="openai_realtime", audio_ts=doc.get("audio_end_ms"), reason="server_vad")
                return
            if event_type == "input_audio_buffer.committed":
                input_committed = True
                self._mark_latency("input_committed")
                if response_wait_task is None or response_wait_task.done():
                    response_wait_task = asyncio.create_task(request_response_after_transcript_wait("realtime_committed"))
                return
            if event_type == "conversation.item.input_audio_transcription.delta":
                delta = extract_realtime_text_delta(doc)
                if delta:
                    if not first_transcript_delta_marked:
                        first_transcript_delta_marked = True
                        self._mark_latency("first_stt_delta", chars=len(delta))
                    transcript += delta
                    await send_event("stt_delta", text=delta, provider="openai_realtime")
                return
            if event_type == "conversation.item.input_audio_transcription.completed":
                text = str(doc.get("transcript") or "").strip()
                if text:
                    transcript = text
                self._last_transcript = transcript.strip()
                if self._last_transcript and not stt_message_sent:
                    self._remember_message("user", "openai_realtime_stt", self._last_transcript, {"reason": "transcription_completed", "model": self._model})
                    stt_message_sent = True
                self._mark_latency("transcription_completed", chars=len(transcript))
                transcript_event.set()
                if stt_result_sent:
                    await send_event("stt_transcript", text=transcript.strip(), provider="openai_realtime", late=True)
                elif input_committed and (response_wait_task is None or response_wait_task.done()):
                    response_wait_task = asyncio.create_task(request_response_after_transcript_wait("transcription_completed"))
                return
            if event_type == "response.created":
                self._mark_latency("response_created")
                await send_llm_started_once()
                return
            if event_type in {"response.output_text.delta", "response.text.delta", "response.audio_transcript.delta"}:
                await send_llm_started_once()
                delta = extract_realtime_text_delta(doc)
                if delta:
                    if technical_output_blocked:
                        return
                    if not first_llm_delta_marked:
                        first_llm_delta_marked = True
                        self._mark_latency("first_llm_delta", chars=len(delta))
                    assistant_text += delta
                    safe_assistant_text = sanitize_assistant_output(assistant_text)
                    if safe_assistant_text != assistant_text.strip():
                        technical_output_blocked = True
                        assistant_text = safe_assistant_text
                        text_chunker = RealtimeTextChunker()
                        await self._log_bus.emit(
                            "WARN",
                            "HA",
                            "Technical Home Assistant-shaped Realtime output blocked",
                            {"session_id": session_id},
                        )
                        await send_event("llm_delta", text=assistant_text)
                        return
                    emotions, chunks = text_chunker.push(delta)
                    display_delta = text_chunker.display_delta
                    if display_delta:
                        self._last_assistant_text = text_chunker.all_text
                        await send_event("llm_delta", text=display_delta)
                    await send_emotion_once(emotions)
                    for chunk in chunks:
                        await send_tts_chunk(chunk, final=False)
                return
            if event_type == "response.done":
                self._mark_latency("response_done")
                await finish_response(doc)
                return
            if event_type == "response.cancelled":
                response_done = True
                self._mark_latency("response_cancelled")
                await send_event("session_cancelled", reason="response_cancelled")
                return

        async def reader_loop() -> None:
            if realtime_ws is None:
                return
            try:
                async for msg in realtime_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            await handle_realtime_event(json.loads(msg.data))
                        except json.JSONDecodeError:
                            continue
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(str(realtime_ws.exception() or "OpenAI Realtime websocket error"))
            except Exception as exc:
                if not self._cancel_event.is_set():
                    self._last_error = safe_exc_message(exc)
                    await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime reader failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                    await send_event("error", message=f"OpenAI Realtime reader failed: {safe_exc_message(exc)}")

        async def open_realtime() -> bool:
            nonlocal realtime_ws, reader_task, client_session
            if realtime_ws is not None and not realtime_ws.closed:
                return True
            if not api_key:
                await send_event("error", message="Realtime is enabled but OpenAI API key is empty.")
                return False
            try:
                connect_started = time.monotonic()
                timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=240)
                client_session = aiohttp.ClientSession(timeout=timeout)
                realtime_ws = await client_session.ws_connect(
                    realtime_ws_url(realtime),
                    headers={"Authorization": f"Bearer {api_key}"},
                    heartbeat=20,
                )
                self._mark_latency("openai_connected", connect_ms=int((time.monotonic() - connect_started) * 1000))
                await send_realtime_json(await self._session_update_payload(config, realtime, target_sample_rate, language))
                self._mark_latency("session_update_sent")
                reader_task = asyncio.create_task(reader_loop())
                self._connected = True
                self._last_event = "connected"
                await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime connected", {"session_id": session_id, "model": self._model})
                return True
            except Exception as exc:
                self._last_error = safe_exc_message(exc)
                await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime connect failed", {"session_id": session_id, "error": safe_exc_message(exc)})
                await send_event("error", message=f"OpenAI Realtime connect failed: {safe_exc_message(exc)}")
                return False

        try:
            await send_event(
                "hello",
                service="alice_control_panel",
                version="0.1.198",
                session_id=session_id,
                endpointing_enabled=True,
                endpointing_provider="openai_realtime",
                realtime_enabled=True,
                realtime_provider="openai",
                realtime_model=self._model,
                llm_enabled=True,
                tts_enabled=True,
            )
            self._mark_latency("hello_sent")
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect
                if message.get("text") is not None:
                    doc = json.loads(str(message["text"]))
                    msg_type = str(doc.get("type") or "").strip().lower()
                    if msg_type == "start":
                        config = await self._config_store.get(include_secrets=True)
                        realtime = self._realtime_cfg(config)
                        pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
                        barge_in_enabled = bool(pipeline.get("barge_in_enabled", True))
                        target_sample_rate = max(8000, int(realtime.get("input_sample_rate") or 24000))
                        transcript = ""
                        assistant_text = ""
                        self._last_tts_text = ""
                        text_chunker = RealtimeTextChunker()
                        tts_chunk_started = False
                        response_requested = False
                        response_done = False
                        stt_result_sent = False
                        stt_message_sent = False
                        assistant_message_sent = False
                        llm_started_sent = False
                        technical_output_blocked = False
                        speech_started = False
                        audio_ms = 0
                        buffered_audio_ms = 0
                        input_committed = False
                        first_audio_marked = False
                        first_transcript_delta_marked = False
                        first_llm_delta_marked = False
                        first_tts_chunk_marked = False
                        transcript_event = asyncio.Event()
                        if response_wait_task is not None and not response_wait_task.done():
                            response_wait_task.cancel()
                        response_wait_task = None
                        source_sample_rate = int(doc.get("sample_rate") or source_sample_rate)
                        resampler = StreamingPcm16Resampler(source_sample_rate, target_sample_rate)
                        input_level_meter.reset()
                        input_audio_metadata = {
                            "mic_shift_bits": doc.get("mic_shift_bits"),
                            "device_local_dsp": bool(doc.get("mic_local_dsp", False)),
                            "device_aec_enabled": bool(doc.get("mic_aec_enabled", False)),
                            "mic_channel": str(doc.get("mic_channel") or ""),
                            "device_pre_roll_ms": max(0, int(doc.get("mic_pre_roll_ms") or 0)),
                            "mic_shift_source": str(doc.get("mic_shift_source") or ""),
                        }
                        language = str(doc.get("language") or language).strip() or "tr"
                        session_id = str(doc.get("session_id") or session_id).strip() or session_id
                        self._session_id = session_id
                        self._reset_latency(
                            session_id,
                            model=self._model,
                            source_rate=source_sample_rate,
                            target_rate=target_sample_rate,
                        )
                        self._mark_latency("start_received")
                        realtime_was_open = realtime_ws is not None and not realtime_ws.closed
                        if await open_realtime():
                            if realtime_was_open:
                                await send_realtime_json(
                                    await self._session_update_payload(
                                        config,
                                        realtime,
                                        target_sample_rate,
                                        language,
                                    )
                                )
                            await send_event(
                                "session_started",
                                session_id=session_id,
                                sample_rate=source_sample_rate,
                                target_sample_rate=target_sample_rate,
                                resampler=resampler.method,
                                resampler_delay_ms=round(resampler.delay_ms, 3),
                                realtime_enabled=True,
                                realtime_model=self._model,
                                endpointing_provider="openai_realtime",
                            )
                            server_noise_reduction = str(realtime.get("noise_reduction") or "near_field").strip().lower()
                            await self._log_bus.emit(
                                "WARN" if input_audio_metadata["device_local_dsp"] and server_noise_reduction not in {"none", "off", "disabled", "kapali"} else "INFO",
                                "STT",
                                "Realtime input audio configured",
                                {
                                    "session_id": session_id,
                                    **input_audio_snapshot(),
                                    "server_noise_reduction": server_noise_reduction,
                                },
                            )
                            self._last_input_audio = input_audio_snapshot()
                        continue
                    if msg_type in {"end", "eos"}:
                        if not realtime_ws:
                            await send_event("error", message="Realtime session is not started.")
                            continue
                        if input_committed or response_done:
                            continue
                        try:
                            if speech_started and buffered_audio_ms >= 100:
                                self._mark_latency("manual_commit_sent", buffered_audio_ms=buffered_audio_ms)
                                await send_realtime_json({"type": "input_audio_buffer.commit"})
                                input_committed = True
                            else:
                                await send_realtime_json({"type": "input_audio_buffer.clear"})
                                input_committed = True
                                await send_event(
                                    "stt_result",
                                    text="",
                                    provider="openai_realtime",
                                    reason="audio_buffer_too_small" if buffered_audio_ms < 100 else "no_speech",
                                )
                                await finish_response(reason="audio_buffer_too_small" if buffered_audio_ms < 100 else "no_speech")
                        except Exception as exc:
                            await send_event("error", message=f"Realtime commit failed: {safe_exc_message(exc)}")
                        continue
                    if msg_type in {"cancel", "cancel_response"}:
                        self._mark_latency("client_cancelled", reason=str(doc.get("reason") or "client_cancel"))
                        await self.cancel(str(doc.get("reason") or "client_cancel"))
                        await send_event("session_cancelled", session_id=session_id, reason=str(doc.get("reason") or "client_cancel"))
                        continue
                    if msg_type == "reset":
                        self._mark_latency("session_reset")
                        await self.cancel("reset")
                        transcript = ""
                        assistant_text = ""
                        self._last_tts_text = ""
                        text_chunker = RealtimeTextChunker()
                        tts_chunk_started = False
                        response_requested = False
                        response_done = False
                        stt_result_sent = False
                        stt_message_sent = False
                        assistant_message_sent = False
                        llm_started_sent = False
                        technical_output_blocked = False
                        speech_started = False
                        audio_ms = 0
                        buffered_audio_ms = 0
                        input_committed = False
                        first_audio_marked = False
                        first_transcript_delta_marked = False
                        first_llm_delta_marked = False
                        first_tts_chunk_marked = False
                        resampler = StreamingPcm16Resampler(source_sample_rate, target_sample_rate)
                        input_level_meter.reset()
                        input_audio_metadata = {}
                        transcript_event = asyncio.Event()
                        if response_wait_task is not None and not response_wait_task.done():
                            response_wait_task.cancel()
                        response_wait_task = None
                        await send_event("session_reset", session_id=session_id)
                        continue
                    await send_event("error", message=f"Unknown realtime message type: {msg_type}")
                    continue
                chunk = message.get("bytes")
                if chunk is None:
                    continue
                if realtime_ws is None:
                    if not await open_realtime():
                        continue
                    await send_event("session_started", session_id=session_id, sample_rate=source_sample_rate, realtime_enabled=True)
                raw, stripped_header = normalize_esp_pcm_packet(bytes(chunk))
                if stripped_header is not None:
                    stripped_packet_headers += 1
                    if stripped_packet_headers <= 3:
                        await self._log_bus.emit(
                            "INFO",
                            "PIPELINE",
                            "OpenAI Realtime mic packet header stripped",
                            {"session_id": session_id, "packet_len": len(chunk), "header": stripped_header},
                        )
                if not raw:
                    continue
                input_level_meter.add(raw)
                if not first_audio_marked:
                    first_audio_marked = True
                    self._mark_latency(
                        "first_audio_chunk",
                        source_rate=source_sample_rate,
                        target_rate=target_sample_rate,
                        resampler=resampler.method,
                    )
                chunk_ms = int((len(raw) / 2) / max(1, source_sample_rate) * 1000)
                audio_ms += chunk_ms
                buffered_audio_ms += chunk_ms
                target = resampler.process(raw)
                if not target:
                    continue
                await send_realtime_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(target).decode("ascii")})
        except WebSocketDisconnect:
            await self._log_bus.emit("INFO", "PIPELINE", "OpenAI Realtime client disconnected", {"session_id": session_id})
        except Exception as exc:
            self._last_error = safe_exc_message(exc)
            await self._log_bus.emit("ERROR", "PIPELINE", "OpenAI Realtime session failed", {"session_id": session_id, "error": safe_exc_message(exc)})
            try:
                await websocket.send_json({"type": "error", "message": safe_exc_message(exc)})
            except Exception:
                pass
        finally:
            self._active = False
            self._connected = False
            self._last_event = "closed"
            self._cancel_event.set()
            if realtime_ws is not None and not realtime_ws.closed:
                try:
                    await realtime_ws.close()
                except Exception:
                    pass
            if reader_task is not None and not reader_task.done():
                reader_task.cancel()
            if client_session is not None:
                try:
                    await client_session.close()
                except Exception:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass
            await self._ws_hub.publish("pipeline_status", {"realtime": await self.status()})

    async def cancel(self, reason: str = "manual_cancel") -> None:
        self._cancel_event.set()
        await self._cancel_playback(reason)

    async def _cancel_playback(self, reason: str) -> None:
        try:
            await self._esp_client.send_audio_error(f"cancelled: {reason}")
        except Exception:
            pass

    async def _session_update_payload(self, config: dict[str, Any], realtime: dict[str, Any], sample_rate: int, language: str) -> dict[str, Any]:
        pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
        barge_in_enabled = bool(pipeline.get("barge_in_enabled", True))
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": sample_rate},
            "turn_detection": self._turn_detection(realtime, barge_in_enabled=barge_in_enabled),
        }
        noise = str(realtime.get("noise_reduction") or "near_field").strip().lower()
        if noise in {"near_field", "far_field"}:
            audio_input["noise_reduction"] = {"type": noise}
        elif noise in {"none", "off", "disabled", "kapali"}:
            audio_input["noise_reduction"] = None
        transcription_model = str(realtime.get("transcription_model") or "").strip()
        if transcription_model:
            transcription = {"model": transcription_model, "language": language}
            transcription_prompt = str(realtime.get("transcription_prompt") or "").strip()
            if transcription_prompt:
                if len(transcription_prompt) > REALTIME_TRANSCRIPTION_PROMPT_MAX_CHARS:
                    await self._log_bus.emit(
                        "WARN",
                        "PIPELINE",
                        "OpenAI Realtime STT prompt truncated",
                        {
                            "chars": len(transcription_prompt),
                            "max_chars": REALTIME_TRANSCRIPTION_PROMPT_MAX_CHARS,
                        },
                    )
                    transcription_prompt = transcription_prompt[:REALTIME_TRANSCRIPTION_PROMPT_MAX_CHARS]
                transcription["prompt"] = transcription_prompt
            audio_input["transcription"] = transcription
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": str(realtime.get("model") or "gpt-realtime-mini"),
                "instructions": await self._instructions(config, realtime),
                "audio": {"input": audio_input},
            },
        }

    async def _instructions(self, config: dict[str, Any], realtime: dict[str, Any]) -> str:
        text = str(realtime.get("instructions") or "").strip()
        if text:
            return self._with_runtime_guardrails(text)
        llm = config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}
        text = str(llm.get("system_prompt") or "").strip()
        if text:
            return self._with_runtime_guardrails(text)
        return self._with_runtime_guardrails(await self._prompt_store.active_prompt_text())

    def _with_runtime_guardrails(self, text: str) -> str:
        clean = str(text or "").strip()
        blocks = [HOME_ASSISTANT_RUNTIME_GUARDRAILS, ROBOT_BEHAVIOR_RUNTIME_HINT, CURRENT_TURN_RUNTIME_GUARDRAIL]
        for block in blocks:
            if block in clean:
                continue
            clean = block if not clean else f"{clean}\n\n{block}"
        return clean

    def _turn_detection(self, realtime: dict[str, Any], barge_in_enabled: bool = True) -> dict[str, Any]:
        return realtime_turn_detection(realtime, barge_in_enabled=barge_in_enabled)

    @staticmethod
    def _realtime_cfg(config: dict[str, Any]) -> dict[str, Any]:
        return active_realtime_config(config)

    @staticmethod
    def _api_key(config: dict[str, Any], realtime: dict[str, Any]) -> str:
        value = str(realtime.get("api_key") or "").strip()
        if value:
            return value
        llm = config.get("llm", {}) if isinstance(config, dict) and isinstance(config.get("llm"), dict) else {}
        providers = llm.get("providers", {}) if isinstance(llm.get("providers"), dict) else {}
        openai = providers.get("openai", {}) if isinstance(providers.get("openai"), dict) else {}
        value = str(openai.get("api_key") or "").strip()
        if value:
            return value
        active_llm = active_llm_config(config)
        if str(active_llm.get("provider") or "").lower() == "openai":
            return str(active_llm.get("api_key") or "").strip()
        return ""
