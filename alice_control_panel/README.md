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
- Version `0.1.188` fixes ESP WebSocket PING/PONG handling, summarizes reconnect storms, and retains WARN/ERROR logs for 31 days within a 32 MB archive cap.
- Version `0.1.187` preserves Google AI TTS character guidance by placing it in the supported Gemini 3.1 speech prompt instead of a rejected developer instruction.
- Version `0.1.115` adds firmware soft sleep mode, deep-sleep eyes, and Sleep/Wake panel control.
- Version `0.1.98` ends empty-room radar calibration even when no radar frames arrive and tightens the mobile radar header controls.
- Version `0.1.97` moves ESP panel API JSON buffers off the HTTP task stack to prevent command-triggered stack overflows.
- Version `0.1.96` queues radar empty-room calibration safely so the HTTP command path does not touch radar state directly.
- Version `0.1.95` preserves Gemini 3.1 Flash TTS selections and routes 3.1 Google AI TTS through the Interactions endpoint.
- Version `0.1.94` adds radar jump rejection, continuity confidence, and empty-room calibration commands.
- Version `0.1.93` fixes the RD-03D radar X orientation and makes radar tracking updates more responsive.
- Version `0.1.92` compacts the Voice Pipeline transcript and realtime timeline so short RT rows do not create unnecessary scrollbars.
- Version `0.1.159` waits a little longer for final Realtime transcripts when a partial utterance looks like a Home Assistant command, preventing the model from answering before the allowlist router can act.
- Version `0.1.158` keeps weather read intents inside the weather allowlist and avoids "multiple candidates" wording when only one candidate is shown.
- Version `0.1.157` switches Gemini 3.1 Google AI TTS to Interactions streaming and handles closed relay WebSockets without empty error logs.
- Version `0.1.156` restores inline log details and gives the detail block a fixed readable height so it cannot collapse between rows.
- Version `0.1.155` moves expanded log details into a fixed inspector above the log list so full panels cannot clip the details.
- Version `0.1.154` makes expanded log details a real scroll-list row so details are not clipped when the log panel is full.
- Version `0.1.153` makes latency totals wait for ESP speaker-finished timing when available and separates speaker-start from speaker-finish metrics.
- Version `0.1.152` moves Timing and Messages out of the Voice Pipeline tabs into dedicated Latency Timeline and Conversation panels.
- Version `0.1.151` adds detailed Google TTS timing traces for request, first byte, audio decode, ESP transfer, and speaker start latency.
- Version `0.1.150` reports ESP speaker-first-PCM timing in the Voice Pipeline so TTS text delay and real speaker delay can be separated.
- Version `0.1.149` shows the loaded frontend version beside backend status so stale panel JavaScript is obvious after updates.
- Version `0.1.148` makes every log row expandable through delegated list clicks and shows a compact summary when a log has no extra details.
- Version `0.1.147` makes log detail clicks trigger their own render so expanded details open immediately instead of waiting for the next log update.
- Version `0.1.146` refreshes add-on metadata so Home Assistant can surface the latest log panel update as a new package.
- Version `0.1.144` keeps expanded log details visible when the Logs panel is full or scrolled to the bottom.
- Version `0.1.143` refreshes add-on metadata so Home Assistant can surface the latest HA control reply changes as an update.
- Version `0.1.142` keeps HA control deterministic while adding safer, more varied natural replies from alias names, local fallbacks, and optional LLM narration.
- Version `0.1.141` lets strong aliases outside a misleading domain hint still match safely, so `switch.masa_lambasi` can answer "masa lambasini ac" while color/brightness remains light-only.
- Version `0.1.140` adds a safer Home Assistant intent parser for Turkish light color/brightness commands, alias matching, room groups, and clarification instead of guessing between multiple allowlisted entities.
- Version `0.1.91` keeps the Radar view switch fixed while technical calibration buttons appear to its left.
- Version `0.1.82` widens the left dashboard column and restores Radar summary values to a single compact row.
- Version `0.1.81` compacts the top status strip, folds hardware state into it, and moves Radar under Connections.
- Version `0.1.80` adds radar decision smoothing and hysteresis, drawing raw and filtered target positions separately.
- Version `0.1.79` shows computed radar distance and per-target raw rows to separate real position from RD-03D resolution data.
- Version `0.1.78` improves the radar map scale, angle readout, and center dead-zone visualization for close-range tests.
- Version `0.1.77` adds the RD-03D radar status/event path and a live radar target view in the panel.
- Version `0.1.76` caps OpenAI Realtime STT prompts at the API limit so long hint text cannot break live sessions.
- Version `0.1.75` routes Home Assistant weather data through the active LLM so Alice can give natural, advice-aware weather replies.
- Version `0.1.74` equalizes Command Panel button sizing across ESP, server, and Mic Debug sections.
- Version `0.1.73` adds OpenAI Realtime latency markers for speech, transcript, response, and first chunk timing without changing audio rate or VAD tuning.
- Version `0.1.72` keeps OpenAI Realtime and fallback live VAD settings available as separate voice paths while surfacing mic shift/clip metrics.
- Version `0.1.71` maps the former servo buttons to short N20 motor test controls.
- Version `0.1.63` makes help buttons more subtle and adds detailed `??` field guides for complex config sections.
- Version `0.1.62` adds compact contextual help bubbles to dashboard and config panel headings.
- Version `0.1.61` adds OpenAI Live semantic eagerness/STT prompt fields and fixes OpenAI TTS PCM rate metadata.
- Version `0.1.60` strengthens active provider card borders and compacts Home Assistant bridge controls.
- Version `0.1.59` gives main panel headings a clearer, slightly larger violet accent.
- Version `0.1.58` gives main panel headings a larger violet accent.
- Version `0.1.57` improves panel/config heading hierarchy and gives config separators more breathing room.
- Version `0.1.56` makes Live Voice activation rely on the provider selector and moves `None` to the right.
- Version `0.1.55` improves Config section spacing, moves Safe mode next to Debug logs, and adds a `None` Live Voice selector.
- Version `0.1.54` makes Home Assistant weather entity replies more natural and advice-oriented.
- Version `0.1.46` suppresses empty Realtime commits/responses and ignores empty TTS relay requests.
- Version `0.1.45` applies ESP mic packet header stripping to the OpenAI Realtime live voice bridge.
- Version `0.1.44` strips ESP mic packet handler bytes and suppresses no-speech/hallucinated live transcripts.
- Version `0.1.43` makes `/voice/ws` emit the legacy voice session events expected by the ESP firmware.
- Version `0.1.42` separates Live Voice config into prominent OpenAI Live and Gemini Live profiles.
- Version `0.1.41` adds separate Groq and Gemini LLM provider profiles while preserving OpenAI, OpenRouter, and generic OpenAI-compatible settings.
- Version `0.1.40` removes the remaining internal HA conversation helper from the Home Assistant control path.
- Version `0.1.39` removes the public HA conversation endpoint from the control path and marks the control panel as the primary add-on path.
- Version `0.1.38` adds the first safe Home Assistant command resolver, using only allowlisted entities instead of HA Assist/conversation.
- Version `0.1.37` makes Home Assistant access allowlist-only: only entity IDs in the panel list can be read or controlled, and legacy broad access fields are ignored.
- Version `0.1.36` adds the first integrated OpenAI Realtime `/voice/ws` bridge path for live-duplex voice.
- Version `0.1.35` fixes the integrated ElevenLabs relay config shape.
- Version `0.1.34` folds in ElevenLabs TTS, direct `/tts/ws` and `/voice/ws` compatibility endpoints, and Home Assistant bridge APIs.
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

The legacy add-on folders are reference archives only. `Alice Control Panel` is the primary runtime for panel, ESP, STT, LLM, TTS, and Home Assistant control.

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
Use `ha_bridge.aliases` to add natural names for allowlisted entities, for example
`light.masa_lambasi: masa lambasi, masalambasi, calisma lambasi`. Aliases never
grant access by themselves; they only improve matching for entity IDs already in
the allowlist.

`ha_bridge.api_base_url` defaults to `http://supervisor/core/api` inside the add-on and normally does not need editing. Home Assistant Assist/conversation agents are not used for Alice home control because that endpoint cannot be constrained to the panel allowlist.
