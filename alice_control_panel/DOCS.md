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

Commands:

```text
POST /api/command
POST /api/esp/command
```

TTS relay:

```text
WS /api/pipeline/tts/ws
POST /api/pipeline/tts/text
```

The TTS WebSocket accepts JSON commands:

```json
{"type":"start","provider":"openai","text":"Merhaba","final":true}
```

`POST /api/pipeline/tts/text` accepts `{"text":"Merhaba"}` and sends generated TTS audio directly to the connected ESP WebSocket when `pipeline.stream_to_esp` is enabled.

For streaming text providers:

```json
{"type":"start","provider":"cartesia","text":"Merhaba ","final":false}
{"type":"append","text":"Alice burada.","final":true}
```

Google provider notes:

- `google_ai` uses a Google AI Studio / Gemini API key. Set `tts.provider` to `google_ai`, `tts.google_ai.api_key`, model `gemini-2.5-flash-preview-tts`, and a voice such as `Kore`, `Zephyr`, or `Aoede`. Gemini TTS returns 24 kHz PCM.
- `google_cloud` uses a Google Cloud service-account JSON. Enable Cloud Text-to-Speech API in that project, paste the full JSON into `tts.google_cloud.credentials_json`, and use a voice such as `tr-TR-Chirp3-HD-Kore`.

The server sends:

```json
{"type":"start","encoding":"pcm_s16le","sample_rate":44100,"channels":1}
```

then binary PCM chunks, then:

```json
{"type":"done"}
```

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
test_speaker, test_mic, wake_on, wake_off, servo_left, servo_right,
servo_center, amp_mute_on, amp_mute_off, reconnect, reboot
```

Server commands:

```text
restart_stt, restart_tts, reload_prompt, clear_logs, safe_mode_on, safe_mode_off
```

## Notes

- This is the first integrated control-panel version.
- Faster-whisper and OpenAI Realtime code paths are scaffolded for migration; heavy ML dependencies are intentionally not installed in this first installer-safe image.
- The React/Vite frontend source is kept in the repository, but the add-on image serves the bundled `static/` panel to avoid HA install-time npm builds.
- `0.1.22` adds Google AI Studio Gemini TTS and Google Cloud Text-to-Speech provider paths.
- ESP-side audio playback for this protocol can be implemented independently after this backend path is installed.
