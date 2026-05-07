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
- Panel-saved config in `/data/alice_config.json` wins over Home Assistant bootstrap options after the first save.
- `/data/prompts/*.yaml` prompt profiles.
- Unified in-memory log ring buffer with WebSocket streaming.
- ESP offline/mock mode when `esp_base_url` is empty or unavailable.
- ESP auto reconnect pauses after `esp_max_auto_reconnects` failures; `0` keeps unlimited retries.
- TTS relay can stream generated PCM audio to the connected ESP WebSocket using the lightweight `audio_start` / binary PCM / `audio_end` protocol.
- ESP command stubs for the future lightweight ESP HTTP/WebSocket API.
- OpenAI PCM TTS stream and Cartesia continuation relay moved into the new structure.
- Google AI Studio Gemini TTS and Google Cloud Text-to-Speech provider paths are available from the TTS config panel.
- No Node build or heavy ML dependency is required during add-on installation.
- Version `0.1.38` adds the first safe Home Assistant command resolver, using only allowlisted entities instead of HA Assist/conversation.
- Version `0.1.37` makes Home Assistant access allowlist-only: only entity IDs in the panel list can be read or controlled, and legacy broad access fields are ignored.
- Version `0.1.36` adds the first integrated OpenAI Realtime `/voice/ws` bridge path so the old realtime voice add-on can be bypassed when an OpenAI key is configured.
- Version `0.1.35` fixes the integrated ElevenLabs relay config shape after the `0.1.34` legacy migration.
- Version `0.1.34` folds in the missing legacy pieces: ElevenLabs TTS, direct `/tts/ws` and `/voice/ws` compatibility endpoints, and Home Assistant bridge APIs.
- Version `0.1.33` makes Silero VAD the default live mic endpointing provider, with energy endpointing kept as fallback.
- Version `0.1.32` adds `/api/pipeline/mic/ws`, a live PCM WebSocket for future continuous voice sessions.
- Version `0.1.31` adds voice session controls and a cancellable TTS response path for barge-in groundwork.
- Version `0.1.30` stores faster-whisper models under `/data/models` and adds mic response modes for assistant, transcript echo, or both.
- Version `0.1.29` wires captured ESP PCM into faster-whisper STT.
- Version `0.1.28` enlarges the header logo, tightens connection errors, and logs reconnect pause once.
- Version `0.1.27` moves the Alice logo into the main header and trims the sidebar.
- Version `0.1.26` waits for wake/mic ownership to release before panel mic capture.
- Version `0.1.25` adds the first ESP mic-capture bridge into the panel STT path.
- Version `0.1.24` chunks the ESP silence prefix and clears ESP playback state on failed audio frames.

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

## Home Assistant Entity Scope

Home Assistant access is allowlist-only. Put only the entity IDs Alice may
read/control in `ha_bridge.exposed_entities`, one per line or separated by
spaces/commas. The backend reads those exact entity IDs one by one instead of
fetching the full Home Assistant entity list. Legacy broad-access settings such as exposing all
entities, allowing whole domains, or blacklist-style filtering are ignored by the
runtime even if those old option keys remain for Supervisor compatibility.

`ha_bridge.api_base_url` defaults to `http://supervisor/core/api` inside the add-on and normally does not need editing. Home Assistant Assist/conversation agents are not used for Alice home control because that endpoint cannot be constrained to the panel allowlist.
