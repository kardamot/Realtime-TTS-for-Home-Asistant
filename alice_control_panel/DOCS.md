# Alice Control Panel Add-on Docs

## Installation

1. Add this repository folder to Home Assistant as a local add-on repository.
2. Install `Alice Control Panel`.
3. Set the add-on options:
   - `port`: default `8099`
   - `panel_token` or `panel_password`: optional local auth
   - `esp_base_url`: empty until ESP firmware exposes the target API
   - `esp_max_auto_reconnects`: default `40`; set `0` for unlimited automatic reconnects
   - provider API keys under `llm` and `tts`
4. Start the add-on.
5. Open `http://HOME_ASSISTANT_IP:8099`.

## Persistent Files

Runtime state is stored in the add-on `/data` directory:

```text
/data/alice_config.json
/data/prompts/alice.yaml
/data/prompts/debug.yaml
/data/prompts/minimal.yaml
```

Secrets are never committed to the repository. The UI masks secrets on export unless `include secrets` is explicitly selected.
After the first panel save, `/data/alice_config.json` is the source of truth. Home Assistant add-on options are used as bootstrap defaults and no longer overwrite panel-saved values during updates/restarts.

## API

Status:

```text
GET /api/health
GET /api/status
GET /health
```

Config:

```text
GET  /api/config
POST /api/config
POST /api/config/import
GET  /api/config/export?include_secrets=false
```

Prompts:

```text
GET    /api/prompts
GET    /api/prompts/{slug}
POST   /api/prompts/{slug}
POST   /api/prompts/{slug}/activate
POST   /api/prompts/{slug}/copy
DELETE /api/prompts/{slug}
```

Logs:

```text
GET    /api/logs
DELETE /api/logs
GET    /api/logs/download
WS     /api/ws/logs
```

Home Assistant bridge:

```text
GET  /api/ha/health
GET  /api/ha/states
GET  /api/ha/states/{entity_id}
GET  /api/ha/search?q=...
GET  /api/ha/allowed
POST /api/ha/service
POST /api/ha/command
```

Home Assistant access is allowlist-only. Add entity IDs in
`ha_bridge.exposed_entities`, one per line or separated by spaces/commas. The
backend reads those exact entity IDs one by one instead of fetching the full Home
Assistant entity list. The runtime ignores legacy `expose_all_entities`, domain
allow, and blacklist fields even if they remain in Home Assistant options for
Supervisor compatibility. Service calls without an allowlisted `entity_id` are rejected.
`ha_bridge.api_base_url` defaults to `http://supervisor/core/api` inside the
add-on and normally does not need editing. Home Assistant Assist/conversation
agents are not used for Alice home control because that endpoint cannot be
constrained to the panel allowlist.

Commands:

```text
POST /api/command
POST /api/esp/command
```

Voice and TTS endpoints:

```text
WS /api/pipeline/tts/ws
WS /api/pipeline/mic/ws
WS /tts/ws
WS /voice/ws
WS /ws?mode=voice
WS /ws?mode=tts
POST /api/pipeline/tts/text
```

The TTS WebSocket accepts JSON commands:

```json
{"type":"start","provider":"openai","text":"Merhaba","final":true}
```

`POST /api/pipeline/tts/text` accepts `{"text":"Merhaba"}` and sends generated TTS audio directly to the connected ESP WebSocket when `pipeline.stream_to_esp` is enabled.

Compatibility endpoints for firmware/client migration:

- `/voice/ws` and `/ws` accept PCM/binary audio plus `start`, `eos`, `cancel_response`, `reset`, and safe Home Assistant helper messages.
- `/tts/ws` and `/ws?mode=tts` accept the TTS relay `start`/`append` JSON flow and stream PCM frames back to the client.

For streaming text providers:

```json
{"type":"start","provider":"cartesia","text":"Merhaba ","final":false}
{"type":"append","text":"Alice burada.","final":true}
```

Google provider notes:

- `google_ai` uses a Google AI Studio / Gemini API key. Set `tts.provider` to `google_ai`, `tts.google_ai.api_key`, model `gemini-2.5-flash-preview-tts`, and a voice such as `Kore`, `Zephyr`, or `Aoede`. Gemini TTS returns 24 kHz PCM.
- `google_cloud` uses a Google Cloud service-account JSON. Enable Cloud Text-to-Speech API in that project, paste the full JSON into `tts.google_cloud.credentials_json`, and use a voice such as `tr-TR-Chirp3-HD-Kore`.
- For ESP playback stability, `tts.esp_initial_buffer_ms` defaults to `1500` and `tts.esp_silence_prefix_ms` defaults to `450`. Increase these if the first second of playback still underruns or crackles.

The server sends:

```json
{"type":"start","encoding":"pcm_s16le","sample_rate":44100,"channels":1}
```

then binary PCM chunks, then:

```json
{"type":"done"}
```

The live mic WebSocket accepts:

```json
{"type":"start","sample_rate":16000,"channels":1,"encoding":"pcm_s16le","vad_enabled":true}
```

Then send binary `pcm_s16le` chunks. The server uses Silero VAD by default, falls back to energy endpointing if Silero cannot be initialized, and emits `vad_start`, `vad_end`, `utterance_finalized`, and `pipeline_status`. Send `{"type":"end"}` for manual finalize or `{"type":"cancel"}` to discard the current utterance.

## ESP Contract

ESP should stay light. It reports state, receives commands, streams events/logs, and plays audio. Heavy STT/LLM/TTS remains inside the add-on/server.

Expected `GET /api/status` shape:

```json
{
  "state": "IDLE",
  "ip": "192.168.1.50",
  "wifi": {"connected": true, "ssid": "home", "rssi": -55},
  "uptime_sec": 1234,
  "heap_free": 180000,
  "heap_min": 140000,
  "hardware": {
    "mic": "ok",
    "speaker": "ok",
    "servo_position": "center",
    "amp_muted": false,
    "wake_enabled": true,
    "errors": []
  }
}
```

Expected `POST /api/command` body:

```json
{"command":"test_speaker","payload":{}}
```

Expected `WS /ws` text messages:

```json
{"type":"status","payload":{"state":"IDLE","heap_free":180000}}
{"type":"log","payload":{"level":"INFO","category":"ESP","message":"speaker test started","details":{}}}
{"type":"event","payload":{"name":"wake_word","source":"mic"}}
```

The server reconnects to this socket automatically. If `esp.ws_url` is empty, it is derived from `esp.base_url` as `/ws`.

When TTS stream-to-ESP is enabled, the server sends audio over the same ESP WebSocket:

```text
TEXT   {"type":"audio_start","stream_id":"tts-...","payload":{"stream_id":"tts-...","encoding":"pcm_s16le","sample_rate":44100,"channels":1}}
TEXT   ESP replies {"type":"audio_ready","stream_id":"tts-...","payload":{"stream_id":"tts-...","message":"ready"}}
BINARY raw little-endian signed 16-bit PCM chunks
TEXT   {"type":"audio_end","stream_id":"tts-...","payload":{"stream_id":"tts-...","ok":true,"message":""}}
TEXT   {"type":"audio_error","stream_id":"tts-...","payload":{"stream_id":"tts-...","message":"..."}}
```

If ESP cannot prepare playback, it should reply with `audio_rejected` and a short reason. The backend waits for this ACK before sending PCM chunks.

If ESP audio playback support is not implemented yet, the backend logs the failure and the rest of the panel remains usable.

Supported first-pass commands:

```text
test_speaker, test_mic, capture_mic, wake_on, wake_off, servo_left, servo_right,
servo_center, amp_mute_on, amp_mute_off, reconnect, reboot
```

Server commands:

```text
restart_stt, restart_tts, reload_prompt, clear_logs, safe_mode_on, safe_mode_off
```

## Notes

- This is the first integrated control-panel version.
- Faster-whisper is wired for one-shot ESP mic captures; OpenAI Realtime now has a first integrated `/voice/ws` bridge path for live-duplex migration.
- The React/Vite frontend source is kept in the repository, but the add-on image serves the bundled `static/` panel to avoid HA install-time npm builds.
- `0.1.40` removes the remaining internal HA conversation helper from the Home Assistant control path.
- `0.1.39` removes the public HA conversation endpoint from the control path and marks the control panel as the primary add-on path.
- `0.1.38` adds the first safe Home Assistant command resolver, using only allowlisted entities instead of HA Assist/conversation.
- `0.1.37` makes Home Assistant access allowlist-only: only entity IDs in the panel list can be read or controlled, and legacy broad access fields are ignored.
- `0.1.36` adds the first integrated OpenAI Realtime `/voice/ws` bridge path for live-duplex voice.
- `0.1.35` fixes the integrated ElevenLabs relay config shape.
- `0.1.34` folds in ElevenLabs TTS, direct `/tts/ws` and `/voice/ws` compatibility endpoints, and Home Assistant bridge APIs.
- `0.1.33` makes Silero VAD the default live mic endpointing provider, with energy endpointing kept as fallback.
- `0.1.32` adds `/api/pipeline/mic/ws`, a live PCM WebSocket for future continuous voice sessions.
- `0.1.31` adds voice session start/stop/cancel controls and a cancellable ESP TTS stream path for barge-in groundwork.
- `0.1.30` persists faster-whisper models under `/data/models` and adds selectable mic response modes.
- `0.1.29` wires captured ESP PCM into faster-whisper STT for one-shot mic pipeline tests.
- `0.1.28` enlarges the header logo, tightens connection errors, and logs reconnect pause once.
- `0.1.27` moves the Alice logo into the main header and trims desktop/mobile sidebar space.
- `0.1.26` waits for wake/mic ownership to release before `capture_mic`.
- `0.1.25` adds `capture_mic`, a short ESP-to-panel PCM bridge for the next STT integration step.
- `0.1.24` chunks the ESP silence prefix and clears ESP playback state on failed audio frames.
- ESP-side audio playback for this protocol can be implemented independently after this backend path is installed.
