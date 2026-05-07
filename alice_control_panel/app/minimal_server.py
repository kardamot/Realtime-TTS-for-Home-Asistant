from __future__ import annotations

import json
import os
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DATA_DIR = Path(os.getenv("ALICE_DATA_DIR", "/data"))
OPTIONS_PATH = DATA_DIR / "options.json"
CONFIG_PATH = DATA_DIR / "alice_config.json"
PROMPTS_DIR = DATA_DIR / "prompts"
STATIC_DIR = Path(os.getenv("ALICE_STATIC_DIR", "/app/static"))
STARTED_AT = time.time()
LOGS: list[dict] = []


DEFAULT_CONFIG = {
    "panel": {"port": 8099, "token": "", "password": "", "title": "Alice Control Panel"},
    "esp": {
        "base_url": "",
        "ws_url": "",
        "poll_interval_sec": 3,
        "reconnect_sec": 5,
        "max_auto_reconnects": 40,
        "mock_mode": True,
    },
    "stt": {"provider": "faster_whisper", "model": "small", "language": "tr", "compute_type": "int8"},
    "llm": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "system_prompt": "",
        "temperature": 0.6,
        "stream": True,
        "providers": {
            "openai": {"api_key": "", "model": "gpt-5-mini", "base_url": "https://api.openai.com/v1"},
            "openrouter": {"api_key": "", "model": "openai/gpt-5-mini", "base_url": "https://openrouter.ai/api/v1"},
            "openai_compatible": {"api_key": "", "model": "", "base_url": ""},
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
        "esp_initial_buffer_ms": 900,
        "openai": {"api_key": "", "model": "gpt-4o-mini-tts", "voice": "coral", "instructions": ""},
        "cartesia": {"api_key": "", "model_id": "sonic-3", "voice_id": "", "language": "tr", "version": "2026-03-01"},
        "elevenlabs": {"api_key": "", "model_id": "eleven_flash_v2_5", "voice_id": "", "output_format": "pcm_16000", "latency_mode": 3},
        "google_ai": {"api_key": "", "model": "gemini-3.1-flash-tts-preview", "voice_name": "Kore", "prompt_prefix": ""},
        "google_cloud": {"credentials_json": "", "voice_name": "", "language_code": "tr-TR", "ssml_gender": "FEMALE"},
    },
    "prompts": {"active_profile": "alice"},
    "pipeline": {"stream_to_esp": True, "max_log_events_per_sec": 10},
    "debug_logs": True,
    "safe_mode": False,
}

DEFAULT_PROMPTS = {
    "alice": {
        "slug": "alice",
        "name": "Alice",
        "description": "Main assistant personality and home-control prompt.",
        "prompt": "Sen Alice'sin. Turkce konusan, zeki, pratik ve ev otomasyonu ile robot kontrolunu yoneten bir asistansin.",
    },
    "debug": {
        "slug": "debug",
        "name": "Debug",
        "description": "Verbose diagnostic prompt for pipeline testing.",
        "prompt": "Debug modundasin. Sistem durumunu net sekilde acikla.",
    },
    "minimal": {
        "slug": "minimal",
        "name": "Minimal",
        "description": "Short responses for fast command testing.",
        "prompt": "Cok kisa ve net Turkce cevap ver.",
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_json(path: Path) -> dict:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
                return value if isinstance(value, dict) else {}
    except Exception:
        return {}
    return {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def options_to_config(raw: dict) -> dict:
    mapped: dict = {}
    panel: dict = {}
    esp: dict = {}
    if "port" in raw:
        panel["port"] = raw["port"]
    if "panel_token" in raw:
        panel["token"] = raw["panel_token"]
    if "panel_password" in raw:
        panel["password"] = raw["panel_password"]
    if panel:
        mapped["panel"] = panel
    if "esp_base_url" in raw:
        esp["base_url"] = raw["esp_base_url"]
    if "esp_max_auto_reconnects" in raw:
        esp["max_auto_reconnects"] = raw["esp_max_auto_reconnects"]
    if esp:
        mapped["esp"] = esp
    for key in ("debug_logs", "safe_mode", "stt", "llm", "realtime", "tts"):
        if key in raw:
            mapped[key] = raw[key]
    return mapped


def hydrate_provider_profiles(config: dict) -> dict:
    llm = config.get("llm")
    if isinstance(llm, dict):
        providers = llm.setdefault("providers", {})
        if isinstance(providers, dict):
            active_provider = str(llm.get("provider") or "openai").lower()
            active_profile = providers.setdefault(active_provider, {})
            if isinstance(active_profile, dict):
                default_profile = DEFAULT_CONFIG.get("llm", {}).get("providers", {}).get(active_provider, {})
                for key in ("api_key", "model", "base_url"):
                    if llm.get(key) and (
                        not active_profile.get(key)
                        or active_profile.get(key) == default_profile.get(key)
                    ):
                        active_profile[key] = llm[key]
    return config


def active_llm_config(config: dict) -> dict:
    llm = config.get("llm", {}) if isinstance(config, dict) else {}
    if not isinstance(llm, dict):
        return {}
    provider = str(llm.get("provider") or "openai").lower()
    providers = llm.get("providers", {}) if isinstance(llm.get("providers"), dict) else {}
    profile = providers.get(provider, {}) if isinstance(providers.get(provider), dict) else {}
    return {
        "provider": provider,
        "model": profile.get("model") or llm.get("model") or "gpt-5-mini",
        "api_key": profile.get("api_key") or llm.get("api_key") or "",
        "base_url": profile.get("base_url") or llm.get("base_url") or "",
    }


def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = deep_merge(DEFAULT_CONFIG, options_to_config(read_json(OPTIONS_PATH)))
    config = deep_merge(config, read_json(CONFIG_PATH))
    config = hydrate_provider_profiles(config)
    if not CONFIG_PATH.exists():
        write_json(CONFIG_PATH, config)
    return config


def mask_secrets(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            low = key.lower()
            if any(part in low for part in ("api_key", "token", "password", "secret", "credentials")):
                out[key] = "********" if item else ""
            else:
                out[key] = mask_secrets(item)
        return out
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return value


def log(level: str, category: str, message: str, details: dict | None = None) -> None:
    LOGS.append({
        "id": f"{int(time.time() * 1000)}-{len(LOGS)}",
        "ts": time.time(),
        "level": level,
        "category": category,
        "message": message,
        "details": details or {},
    })
    del LOGS[:-1000]


def ensure_prompts() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for slug, doc in DEFAULT_PROMPTS.items():
        path = PROMPTS_DIR / f"{slug}.json"
        if not path.exists():
            write_json(path, doc)


def prompt_doc(slug: str) -> dict:
    ensure_prompts()
    return read_json(PROMPTS_DIR / f"{slug}.json") or DEFAULT_PROMPTS.get(slug, DEFAULT_PROMPTS["alice"])


def list_prompts(config: dict) -> dict:
    ensure_prompts()
    active = config.get("prompts", {}).get("active_profile", "alice")
    profiles = []
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        doc = read_json(path)
        slug = doc.get("slug") or path.stem
        profiles.append({
            "slug": slug,
            "name": doc.get("name") or slug.title(),
            "description": doc.get("description") or "",
            "active": slug == active,
        })
    return {"active_profile": active, "profiles": profiles}


class Handler(SimpleHTTPRequestHandler):
    server_version = "AliceControlPanel/0.1.18"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        log("INFO", "HTTP", fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/auth/check":
                cfg = load_config()
                self.json({"auth_required": bool(cfg["panel"].get("token") or cfg["panel"].get("password"))})
            elif path == "/api/health":
                self.json(health())
            elif path == "/api/status":
                if not self.authorized():
                    return
                self.json(status())
            elif path == "/api/config":
                if not self.authorized():
                    return
                self.json(mask_secrets(load_config()))
            elif path == "/api/config/export":
                if not self.authorized():
                    return
                query = parse_qs(parsed.query)
                include = query.get("include_secrets", ["false"])[0].lower() == "true"
                self.json(load_config() if include else mask_secrets(load_config()))
            elif path == "/api/logs":
                if not self.authorized():
                    return
                self.json({"entries": LOGS[-250:]})
            elif path == "/api/logs/download":
                if not self.authorized():
                    return
                body = "\n".join(f"{item['ts']:.3f} [{item['level']}] {item['category']}: {item['message']}" for item in LOGS) + "\n"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            elif path == "/api/prompts":
                if not self.authorized():
                    return
                self.json(list_prompts(load_config()))
            elif path.startswith("/api/prompts/"):
                if not self.authorized():
                    return
                slug = path.rsplit("/", 1)[-1]
                self.json(prompt_doc(slug))
            elif path.startswith("/api/esp/status"):
                if not self.authorized():
                    return
                self.json(esp_status())
            else:
                if path == "/" or not (STATIC_DIR / path.lstrip("/")).exists():
                    self.path = "/index.html"
                return super().do_GET()
        except Exception as exc:
            log("ERROR", "SYSTEM", "GET failed", {"path": path, "error": str(exc)})
            self.json({"ok": False, "message": str(exc)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if not self.authorized():
            return
        try:
            payload = self.read_body()
            if path == "/api/config":
                cfg = hydrate_provider_profiles(deep_merge(load_config(), payload if isinstance(payload, dict) else {}))
                write_json(CONFIG_PATH, cfg)
                log("INFO", "UI/API", "Configuration updated")
                self.json({"ok": True, "config": cfg})
            elif path == "/api/config/import":
                cfg = payload.get("config", payload) if isinstance(payload, dict) else {}
                write_json(CONFIG_PATH, hydrate_provider_profiles(deep_merge(DEFAULT_CONFIG, cfg)))
                log("INFO", "UI/API", "Configuration imported")
                self.json({"ok": True})
            elif path == "/api/command":
                command = str(payload.get("command", ""))
                log("WARN", "COMMAND", "Command accepted in stdlib fallback mode", {"command": command})
                if command == "clear_logs":
                    LOGS.clear()
                self.json({"ok": command != "", "implemented": False, "message": "Stdlib fallback mode: command logged only.", "command": command})
            elif path == "/api/pipeline/text":
                text_value = str(payload.get("text", ""))
                log("INFO", "PIPELINE", "Text pipeline test accepted", {"text": text_value})
                self.json({"ok": True, "pipeline": pipeline_status(text_value)})
            elif path.startswith("/api/prompts/") and path.endswith("/activate"):
                slug = path.split("/")[-2]
                cfg = load_config()
                cfg.setdefault("prompts", {})["active_profile"] = slug
                write_json(CONFIG_PATH, cfg)
                log("INFO", "PROMPT", "Active prompt changed", {"slug": slug})
                self.json({"ok": True, "active_profile": slug})
            elif path.startswith("/api/prompts/"):
                slug = path.rsplit("/", 1)[-1]
                doc = payload if isinstance(payload, dict) else {}
                doc["slug"] = slug
                write_json(PROMPTS_DIR / f"{slug}.json", doc)
                log("INFO", "PROMPT", "Prompt saved", {"slug": slug})
                self.json({"ok": True, "prompt": doc})
            else:
                self.json({"ok": False, "message": "Unknown endpoint"}, status=404)
        except Exception as exc:
            log("ERROR", "SYSTEM", "POST failed", {"path": path, "error": str(exc)})
            self.json({"ok": False, "message": str(exc)}, status=500)

    def do_DELETE(self):
        if not self.authorized():
            return
        if urlparse(self.path).path == "/api/logs":
            LOGS.clear()
            self.json({"ok": True})
        else:
            self.json({"ok": False, "message": "Unknown endpoint"}, status=404)

    def read_body(self):
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def authorized(self) -> bool:
        cfg = load_config()
        token = str(cfg["panel"].get("token") or "")
        password = str(cfg["panel"].get("password") or "")
        if not token and not password:
            return True
        provided = self.headers.get("x-alice-token") or parse_qs(urlparse(self.path).query).get("token", [""])[0]
        if provided and provided in {token, password}:
            return True
        self.json({"detail": "Alice panel token required"}, status=401)
        return False

    def json(self, value, status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def health() -> dict:
    cfg = load_config()
    return {
        "ok": True,
        "service": "alice_control_panel",
            "version": "0.1.18",
        "safe_mode": bool(cfg.get("safe_mode")),
        "debug_logs": bool(cfg.get("debug_logs")),
        "system": {
            "uptime_sec": int(time.time() - STARTED_AT),
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "pid": os.getpid(),
        },
    }


def esp_status() -> dict:
    cfg = load_config()
    base_url = cfg.get("esp", {}).get("base_url") or ""
    return {
        "online": False,
        "mock_mode": True,
        "ip": base_url,
        "wifi": {"ssid": "", "rssi": None, "connected": False},
        "uptime_sec": 0,
        "state": "OFFLINE",
        "heap_free": None,
        "heap_min": None,
        "hardware": {
            "mic": "unknown",
            "speaker": "unknown",
            "servo_position": "center",
            "amp_muted": None,
            "wake_enabled": None,
            "errors": [],
        },
        "last_seen": None,
        "last_error": "Stdlib fallback mode: ESP polling disabled until FastAPI runtime is enabled.",
        "ws_connected": False,
        "ws_url": "",
        "last_ws_error": "",
        "reconnects": 0,
        "max_auto_reconnects": int(cfg.get("esp", {}).get("max_auto_reconnects") or 40),
        "auto_reconnect_paused": False,
    }


def pipeline_status(text_value: str = "") -> dict:
    return {
        "state": "IDLE",
        "last_user_text": text_value,
        "stt_result": text_value,
        "llm_response": "Stdlib fallback mode: pipeline backend is not active yet.",
        "tts_status": "disabled_minimal_mode",
        "stream_active": False,
        "timeline": [{"ts": time.time(), "category": "SYSTEM", "message": "Stdlib fallback mode"}],
    }


def status() -> dict:
    cfg = load_config()
    llm = active_llm_config(cfg)
    return {
        "health": health(),
        "esp": esp_status(),
        "pipeline": pipeline_status(),
        "stt": {"provider": cfg.get("stt", {}).get("provider", "faster_whisper"), "model": cfg.get("stt", {}).get("model", "small"), "loaded": False},
        "llm": {"provider": llm.get("provider", "openai"), "model": llm.get("model", "gpt-5-mini"), "api_key_configured": bool(llm.get("api_key"))},
        "tts": {"enabled": cfg.get("tts", {}).get("enabled", True), "provider": cfg.get("tts", {}).get("provider", "openai"), "pcm_sample_rate": cfg.get("tts", {}).get("pcm_sample_rate", 44100)},
        "config": mask_secrets(cfg),
    }


def main() -> None:
    cfg = load_config()
    ensure_prompts()
    log("INFO", "SYSTEM", "Alice Control Panel minimal server starting")
    port = int(os.getenv("PORT") or cfg.get("panel", {}).get("port") or 8099)
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Alice Control Panel minimal server listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
