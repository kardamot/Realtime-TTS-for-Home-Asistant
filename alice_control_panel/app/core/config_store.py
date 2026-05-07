from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

from .paths import CONFIG_PATH, OPTIONS_PATH


SECRET_KEY_PARTS = ("api_key", "token", "password", "secret", "credentials")


DEFAULT_CONFIG: dict[str, Any] = {
    "panel": {
        "port": 8099,
        "token": "",
        "password": "",
        "title": "Alice Control Panel",
    },
    "esp": {
        "base_url": "",
        "ws_url": "",
        "poll_interval_sec": 3,
        "reconnect_sec": 5,
        "max_auto_reconnects": 40,
        "command_timeout_sec": 4,
        "audio_ack_timeout_sec": 3,
        "mock_mode": True,
    },
    "stt": {
        "provider": "faster_whisper",
        "model": "small",
        "model_cache_dir": "/data/models/faster_whisper",
        "language": "tr",
        "compute_type": "int8",
        "beam_size": 1,
        "vad_filter": False,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "system_prompt": "",
        "temperature": 0.6,
        "stream": True,
        "providers": {
            "openai": {
                "api_key": "",
                "model": "gpt-5-mini",
                "base_url": "https://api.openai.com/v1",
            },
            "openrouter": {
                "api_key": "",
                "model": "openai/gpt-5-mini",
                "base_url": "https://openrouter.ai/api/v1",
            },
            "openai_compatible": {
                "api_key": "",
                "model": "",
                "base_url": "",
            },
        },
    },
    "realtime": {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-realtime-mini",
        "ws_url": "wss://api.openai.com/v1/realtime",
        "input_sample_rate": 24000,
        "turn_detection": "server_vad",
        "vad_threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 420,
        "transcription_model": "gpt-4o-mini-transcribe",
        "response_timeout_ms": 12000,
        "noise_reduction": "near_field",
        "instructions": "",
    },
    "tts": {
        "enabled": True,
        "provider": "openai",
        "pcm_sample_rate": 44100,
        "esp_initial_buffer_ms": 1500,
        "esp_silence_prefix_ms": 450,
        "openai": {
            "api_key": "",
            "model": "gpt-4o-mini-tts",
            "voice": "coral",
            "instructions": (
                "Turkce konus. Kadinsi, alimli, muzip, hafif seytansi ve "
                "tatli tatli igneleyici bir tonda oku."
            ),
        },
        "cartesia": {
            "api_key": "",
            "model_id": "sonic-3",
            "voice_id": "",
            "language": "tr",
            "version": "2026-03-01",
        },
        "elevenlabs": {
            "api_key": "",
            "model_id": "eleven_flash_v2_5",
            "voice_id": "",
            "output_format": "pcm_16000",
            "latency_mode": 3,
        },
        "google_ai": {
            "api_key": "",
            "model": "gemini-2.5-flash-preview-tts",
            "voice_name": "Kore",
            "prompt_prefix": "",
        },
        "google_cloud": {
            "credentials_json": "",
            "voice_name": "tr-TR-Chirp3-HD-Kore",
            "language_code": "tr-TR",
            "ssml_gender": "FEMALE",
        },
    },
    "prompts": {
        "active_profile": "alice",
    },
    "pipeline": {
        "stream_to_esp": True,
        "max_log_events_per_sec": 10,
        "mic_response_mode": "assistant",
        "barge_in_enabled": True,
        "live_mic_enabled": True,
        "live_vad_enabled": True,
        "live_vad_provider": "silero",
        "live_vad_silero_start_prob": 0.50,
        "live_vad_silero_end_prob": 0.28,
        "live_vad_start_rms": 450,
        "live_vad_end_rms": 260,
        "live_vad_min_speech_ms": 120,
        "live_vad_end_silence_ms": 650,
        "live_vad_pre_roll_ms": 300,
        "live_vad_max_utterance_ms": 12000,
        "live_vad_max_buffer_sec": 20,
    },
    "ui": {
        "dark_mode": True,
        "compact": False,
    },
    "debug_logs": True,
    "safe_mode": False,
    "updated_at": None,
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _addon_options_to_config(raw: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    panel: dict[str, Any] = {}
    esp: dict[str, Any] = {}
    for source, target in (
        ("port", "port"),
        ("panel_token", "token"),
        ("panel_password", "password"),
    ):
        if source in raw:
            panel[target] = raw[source]
    if panel:
        mapped["panel"] = panel
    if "esp_base_url" in raw:
        esp["base_url"] = raw["esp_base_url"]
    if "esp_max_auto_reconnects" in raw:
        esp["max_auto_reconnects"] = raw["esp_max_auto_reconnects"]
    if "esp_audio_ack_timeout_sec" in raw:
        esp["audio_ack_timeout_sec"] = raw["esp_audio_ack_timeout_sec"]
    if esp:
        mapped["esp"] = esp
    for key in ("debug_logs", "safe_mode", "stt", "llm", "realtime", "tts", "pipeline", "ui"):
        if key in raw:
            mapped[key] = raw[key]
    return mapped


def mask_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            key_l = key.lower()
            if any(part in key_l for part in SECRET_KEY_PARTS):
                masked[key] = "" if not item else "********"
            else:
                masked[key] = mask_secrets(item)
        return masked
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return value


def hydrate_provider_profiles(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm")
    if isinstance(llm, dict):
        providers = llm.setdefault("providers", {})
        if isinstance(providers, dict):
            active_provider = str(llm.get("provider") or "openai").lower()
            active_profile = providers.setdefault(active_provider, {})
            if isinstance(active_profile, dict):
                default_profile = (
                    DEFAULT_CONFIG.get("llm", {})
                    .get("providers", {})
                    .get(active_provider, {})
                )
                for key in ("api_key", "model", "base_url"):
                    if llm.get(key) and (
                        not active_profile.get(key)
                        or active_profile.get(key) == default_profile.get(key)
                    ):
                        active_profile[key] = llm[key]

    tts = config.get("tts")
    if isinstance(tts, dict):
        for provider in ("openai", "cartesia", "elevenlabs", "google_ai", "google_cloud"):
            group = tts.setdefault(provider, {})
            if not isinstance(group, dict):
                continue
            flat_api_key = tts.get(f"{provider}_api_key")
            if flat_api_key and not group.get("api_key"):
                group["api_key"] = flat_api_key
            flat_voice = tts.get(f"{provider}_voice") or tts.get(f"{provider}_voice_id")
            if flat_voice and not group.get("voice_id") and provider in {"cartesia", "elevenlabs"}:
                group["voice_id"] = flat_voice
            if flat_voice and not group.get("voice") and provider == "openai":
                group["voice"] = flat_voice
        google_ai = tts.get("google_ai")
        if isinstance(google_ai, dict) and google_ai.get("model") == "gemini-3.1-flash-tts-preview":
            google_ai["model"] = "gemini-2.5-flash-preview-tts"
        google_cloud = tts.get("google_cloud")
        if isinstance(google_cloud, dict) and not google_cloud.get("voice_name"):
            google_cloud["voice_name"] = "tr-TR-Chirp3-HD-Kore"
    return config


class ConfigStore:
    def __init__(self, config_path: Path = CONFIG_PATH, options_path: Path = OPTIONS_PATH) -> None:
        self._config_path = config_path
        self._options_path = options_path
        self._lock = asyncio.Lock()
        self._config = copy.deepcopy(DEFAULT_CONFIG)

    async def load(self) -> dict[str, Any]:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            file_config = _read_json(self._config_path)
            addon_options = _addon_options_to_config(_read_json(self._options_path))
            self._config = deep_merge(DEFAULT_CONFIG, addon_options)
            self._config = deep_merge(self._config, file_config)
            self._config = hydrate_provider_profiles(self._config)
            if not self._config_path.exists():
                self._config["updated_at"] = time.time()
                self._write_locked()
            return copy.deepcopy(self._config)

    async def get(self, include_secrets: bool = True) -> dict[str, Any]:
        async with self._lock:
            data = copy.deepcopy(self._config)
        return data if include_secrets else mask_secrets(data)

    async def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._config = deep_merge(self._config, patch)
            self._config = hydrate_provider_profiles(self._config)
            self._config["updated_at"] = time.time()
            self._write_locked()
            return copy.deepcopy(self._config)

    async def replace(self, config: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._config = deep_merge(DEFAULT_CONFIG, config)
            self._config = hydrate_provider_profiles(self._config)
            self._config["updated_at"] = time.time()
            self._write_locked()
            return copy.deepcopy(self._config)

    async def export(self, include_secrets: bool = False) -> dict[str, Any]:
        data = await self.get(include_secrets=True)
        return data if include_secrets else mask_secrets(data)

    async def set_active_prompt(self, profile: str) -> None:
        await self.update({"prompts": {"active_profile": profile}})

    def _write_locked(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._config_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._config, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        tmp_path.replace(self._config_path)
