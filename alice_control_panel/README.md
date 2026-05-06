# Alice Control Panel

Direct web panel and backend server for the Alice ESP32 robot and voice pipeline.

Open the panel at:

```text
http://HOME_ASSISTANT_IP:8099
```

This add-on does not use Home Assistant ingress. It exposes its own port and serves a single modern dashboard for robot status, ESP commands, STT/LLM/TTS state, provider config, prompt editing, and live logs.

## First Version Scope

- FastAPI backend with modular config, prompt, log, ESP, and pipeline services.
- Installer-safe static dashboard UI. The React/Vite source is kept under `frontend/` for the next richer panel pass.
- `/data/alice_config.json` central persistent config.
- `/data/prompts/*.yaml` prompt profiles.
- Unified in-memory log ring buffer with WebSocket streaming.
- ESP offline/mock mode when `esp_base_url` is empty or unavailable.
- ESP command stubs for the future lightweight ESP HTTP/WebSocket API.
- OpenAI PCM TTS stream and Cartesia continuation relay moved into the new structure.
- No Node build or heavy ML dependency is required during add-on installation.
- Version `0.1.7` moves live logs into the upper dashboard area and pushes config to the bottom while keeping the lightweight FastAPI runtime.

The old add-ons remain untouched:

- `alice_realtime_tts`
- `alice_realtime_voice`

## ESP Interface Target

The panel expects the ESP to eventually expose:

```text
GET  /api/status
GET  /api/config
POST /api/config
POST /api/command
WS   /ws
```

Until firmware support exists, commands are logged and the UI shows mock/offline state.
