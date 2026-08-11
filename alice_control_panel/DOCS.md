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
`ha_bridge.aliases` can map allowlisted entity IDs to natural Turkish names such
as `light.masa_lambasi: masa lambasi, masalambasi`; aliases do not grant access
to entities outside the allowlist.
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

- `google_ai` uses a Google AI Studio / Gemini API key. Set `tts.provider` to `google_ai`, `tts.google_ai.api_key`, model `gemini-3.1-flash-tts-preview`, and a voice such as `Kore`, `Zephyr`, or `Aoede`. Gemini TTS returns 24 kHz PCM.
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
  "sleep_mode": false,
  "heap_free": 180000,
  "heap_min": 140000,
  "hardware": {
    "mic": "ok",
    "speaker": "ok",
    "servo_position": "center",
    "amp_muted": false,
    "wake_enabled": true,
    "motion_sensor": "ready",
    "motion_sensor_present": true,
    "motion_sensor_ready": true,
    "touch_sensor": "ready",
    "touch_sensor_ready": true,
    "touch_sensor_active": false,
    "sleep_mode": false,
    "eyes_expression": "NORMAL",
    "eyes_sleeping": false,
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
test_speaker, test_mic, capture_mic, listen_start, listen_stop,
follow_up_on, follow_up_off, touch_reactions_on, touch_reactions_off,
lift_reactions_on, lift_reactions_off, motor_forward, motor_backward,
motor_left, motor_right, motor_stop, wake_on, wake_off, barge_in_on, barge_in_off, amp_mute_on,
amp_mute_off, radar_calibrate_empty, radar_clear_empty, reconnect, reboot
```

Server commands:

```text
restart_stt, restart_tts, reload_prompt, start_voice_session, stop_voice_session,
cancel_response, safe_mode_on, safe_mode_off
```

## Notes

- This is the first integrated control-panel version.
- Faster-whisper is wired for one-shot ESP mic captures; OpenAI Realtime now has a first integrated `/voice/ws` bridge path for live-duplex migration.
- The React/Vite frontend source is kept in the repository, but the add-on image serves the bundled `static/` panel to avoid HA install-time npm builds.
- Panel logs are kept in the backend memory buffer, not persisted to a log file; an add-on restart/update or explicit Clear resets the visible buffer.
- `0.1.192` replaces packet-local linear mic resampling with a stateful polyphase FIR path and records per-turn RMS/peak/clipping diagnostics.
- `0.1.191` makes the conversation-interrupt switch authoritative for OpenAI Realtime and synchronizes it to compatible ESP firmware.
- `0.1.190` adds a bounded last-successful TTS WAV capture and download action to Voice Pipeline.
- `0.1.189` prevents unknown room names from falling back to other allowlisted room lights during bulk commands.
- `0.1.188` fixes ESP WebSocket PING/PONG handling, summarizes reconnect storms, and retains WARN/ERROR logs for 31 days within a 32 MB archive cap.
- `0.1.187` keeps Google AI TTS character guidance in the supported Interactions input prompt instead of the unsupported developer-instruction field.
- `0.1.159` waits a little longer for final Realtime transcripts when a partial utterance looks like a Home Assistant command, preventing the model from answering before the allowlist router can act.
- `0.1.158` keeps weather read intents inside the weather allowlist and avoids "multiple candidates" wording when only one candidate is shown.
- `0.1.157` switches Gemini 3.1 Google AI TTS to Interactions streaming and handles closed relay WebSockets without empty error logs.
- `0.1.156` restores inline log details and gives the detail block a fixed readable height so it cannot collapse between rows.
- `0.1.155` moves expanded log details into a fixed inspector above the log list so full panels cannot clip the details.
- `0.1.154` makes expanded log details a real scroll-list row so details are not clipped when the log panel is full.
- `0.1.153` makes latency totals wait for ESP speaker-finished timing when available and separates speaker-start from speaker-finish metrics.
- `0.1.152` moves Timing and Messages out of the Voice Pipeline tabs into dedicated Latency Timeline and Conversation panels.
- `0.1.151` adds detailed Google TTS timing traces for request, first byte, audio decode, ESP transfer, and speaker start latency.
- `0.1.150` reports ESP speaker-first-PCM timing in the Voice Pipeline so TTS text delay and real speaker delay can be separated.
- `0.1.149` shows the loaded frontend version beside backend status so stale panel JavaScript is obvious after updates.
- `0.1.148` makes every log row expandable through delegated list clicks and shows a compact summary when a log has no extra details.
- `0.1.147` makes log detail clicks trigger their own render so expanded details open immediately instead of waiting for the next log update.
- `0.1.146` refreshes add-on metadata so Home Assistant can surface the latest log panel update as a new package.
- `0.1.144` keeps expanded log details visible when the Logs panel is full or scrolled to the bottom.
- `0.1.143` refreshes add-on metadata so Home Assistant can surface the latest HA control reply changes as an update.
- `0.1.142` keeps HA control deterministic while adding safer, more varied natural replies from alias names, local fallbacks, and optional LLM narration.
- `0.1.141` lets strong aliases outside a misleading domain hint still match safely, so `switch.masa_lambasi` can answer "masa lambasini ac" while color/brightness remains light-only.
- `0.1.140` adds a safer Home Assistant intent parser for Turkish light color/brightness commands, alias matching, room groups, and clarification instead of guessing between multiple allowlisted entities.
- `0.1.115` adds firmware soft sleep mode, deep-sleep eyes, and Sleep/Wake panel control.
- `0.1.98` ends empty-room radar calibration even when no radar frames arrive and tightens the mobile radar header controls.
- `0.1.97` moves ESP panel API JSON buffers off the HTTP task stack to prevent command-triggered stack overflows.
- `0.1.96` queues radar empty-room calibration safely so the HTTP command path does not touch radar state directly.
- `0.1.95` preserves Gemini 3.1 Flash TTS selections and routes 3.1 Google AI TTS through the Interactions endpoint.
- `0.1.94` adds radar jump rejection, continuity confidence, and empty-room calibration commands.
- `0.1.93` fixes the RD-03D radar X orientation and makes radar tracking updates more responsive.
- `0.1.92` compacts the Voice Pipeline transcript and realtime timeline so short RT rows do not create unnecessary scrollbars.
- `0.1.91` keeps the Radar view switch fixed while technical calibration buttons appear to its left.
- `0.1.82` widens the left dashboard column and restores Radar summary values to a single compact row.
- `0.1.81` compacts the top status strip, folds hardware state into it, and moves Radar under Connections.
- `0.1.80` adds radar decision smoothing and hysteresis, drawing raw and filtered target positions separately.
- `0.1.79` shows computed radar distance and per-target raw rows to separate real position from RD-03D resolution data.
- `0.1.78` improves the radar map scale, angle readout, and center dead-zone visualization for close-range tests.
- `0.1.77` adds the RD-03D radar status/event path and a live radar target view in the panel.
- `0.1.76` caps OpenAI Realtime STT prompts at the API limit so long hint text cannot break live sessions.
- `0.1.75` routes Home Assistant weather data through the active LLM so Alice can give natural, advice-aware weather replies.
- `0.1.74` equalizes Command Panel button sizing across ESP, server, and Mic Debug sections.
- `0.1.73` adds OpenAI Realtime latency markers for speech, transcript, response, and first chunk timing without changing audio rate or VAD tuning.
- `0.1.72` keeps OpenAI Realtime and fallback live VAD settings available as separate voice paths while surfacing mic shift/clip metrics.
- `0.1.71` maps the former servo buttons to short N20 motor test controls.
- `0.1.63` makes help buttons more subtle and adds detailed `??` field guides for complex config sections.
- `0.1.62` adds compact contextual help bubbles to dashboard and config panel headings.
- `0.1.61` adds OpenAI Live semantic eagerness/STT prompt fields and fixes OpenAI TTS PCM rate metadata.
- `0.1.60` strengthens active provider card borders and compacts Home Assistant bridge controls.
- `0.1.59` gives main panel headings a clearer, slightly larger violet accent.
- `0.1.58` gives main panel headings a larger violet accent.
- `0.1.57` improves panel/config heading hierarchy and gives config separators more breathing room.
- `0.1.56` makes Live Voice activation rely on the provider selector and moves `None` to the right.
- `0.1.55` improves Config section spacing, moves Safe mode next to Debug logs, and adds a `None` Live Voice selector.
- `0.1.54` makes Home Assistant weather entity replies more natural and advice-oriented.
- `0.1.46` suppresses empty Realtime commits/responses and ignores empty TTS relay requests.
- `0.1.45` applies ESP mic packet header stripping to the OpenAI Realtime live voice bridge.
- `0.1.44` strips ESP mic packet handler bytes and suppresses no-speech/hallucinated live transcripts.
- `0.1.43` makes `/voice/ws` emit the legacy voice session events expected by the ESP firmware.
- `0.1.42` separates Live Voice config into prominent OpenAI Live and Gemini Live profiles.
- `0.1.41` adds separate Groq and Gemini LLM provider profiles while preserving OpenAI, OpenRouter, and generic OpenAI-compatible settings.
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
