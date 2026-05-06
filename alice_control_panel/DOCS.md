# Alice Control Panel Add-on Docs

## Installation

1. Add this repository folder to Home Assistant as a local add-on repository.
2. Install `Alice Control Panel`.
3. Set the add-on options:
   - `port`: default `8099`
   - `panel_token` or `panel_password`: optional local auth
   - `esp_base_url`: empty until ESP firmware exposes the target API
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
```

The TTS WebSocket accepts JSON commands:

```json
{"type":"start","provider":"openai","text":"Merhaba","final":true}
```

For streaming text providers:

```json
{"type":"start","provider":"cartesia","text":"Merhaba ","final":false}
{"type":"append","text":"Alice burada.","final":true}
```

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
- `0.1.11` adds the Alice generated logo as the Home Assistant add-on icon/logo and dashboard brand mark.
- ESP WebSocket audio playback integration is intentionally left as a clear next step because the ESP API does not exist yet.
