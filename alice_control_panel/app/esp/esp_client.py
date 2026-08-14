from __future__ import annotations

import asyncio
import copy
import json
import struct
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.ws_hub import WsHub


DEFAULT_STATUS: dict[str, Any] = {
    "online": False,
    "mock_mode": True,
    "ip": "",
    "wifi": {"ssid": "", "rssi": None, "connected": False},
    "uptime_sec": 0,
    "state": "OFFLINE",
    "sleep_mode": False,
    "sleep_level": "active",
    "power_mode": "active",
    "heap_free": None,
    "heap_min": None,
    "system": {
        "monitor_ready": False,
        "cpu_percent": None,
        "cpu_mhz": None,
        "cpu_cores": [],
        "temperature_ready": False,
        "temperature_c": None,
        "ram": {"total": None, "free": None, "min_free": None, "largest_free": None},
        "psram": {"total": None, "free": None, "largest_free": None},
        "reset_reason": "",
        "reset_reason_code": None,
        "reset_risk": "info",
    },
    "hardware": {
        "mic": "unknown",
        "mic_sample_rate": None,
        "mic_shift_bits": None,
        "mic_shift_source": "",
        "mic_local_dsp_enabled": None,
        "mic_pre_roll_ms": None,
        "speaker": "unknown",
        "speaker_volume_percent": None,
        "speaker_gain_q12": None,
        "radar": "unknown",
        "servo_position": "center",
        "amp_muted": None,
        "wake_enabled": None,
        "barge_in_enabled": None,
        "follow_up_enabled": None,
        "touch_reactions_enabled": None,
        "lift_reactions_enabled": None,
        "motion_sensor": "unknown",
        "motion_sensor_present": None,
        "motion_sensor_ready": None,
        "touch_sensor": "unknown",
        "touch_sensor_ready": None,
        "touch_sensor_active": None,
        "eyes_expression": "unknown",
        "idle_eye_tracking": "disabled",
        "idle_eye_tracking_active": False,
        "eyes_sleeping": None,
        "sleep_mode": None,
        "errors": [],
    },
    "idle_tracking": {
        "enabled": False,
        "active": False,
        "state": "disabled",
        "direction": "center",
        "x_mm": 0,
        "y_mm": 0,
        "distance_mm": 0,
        "confidence": 0,
        "stable_frames": 0,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "pupil_attention": 0.0,
        "last_seen_ms": 0,
    },
    "radar": {
        "enabled": False,
        "ready": False,
        "fresh": False,
        "state": "unknown",
        "direction": "BELIRSIZ",
        "target_count": 0,
        "selected_target": -1,
        "confidence": 0,
        "stable_frames": 0,
        "jump_rejects": 0,
        "last_jump_rejected": False,
        "background_active": False,
        "background_learning": False,
        "background_points": 0,
        "background_samples": 0,
        "background_suppressed": 0,
        "targets": [],
    },
    "last_seen": None,
    "last_error": "",
    "ws_connected": False,
    "ws_url": "",
    "last_ws_error": "",
    "last_ws_connected_at": None,
    "last_ws_disconnected_at": None,
    "last_ws_message_at": None,
    "ws_connection_started_at": None,
    "last_ws_connection_duration_sec": 0,
    "ws_disconnect_streak": 0,
    "reconnects": 0,
    "max_auto_reconnects": 40,
    "auto_reconnect_paused": False,
}


ESP_COMMANDS = {
    "test_speaker",
    "test_mic",
    "capture_mic",
    "capture_mic_left",
    "capture_mic_right",
    "barge_lab_capture_arm",
    "barge_lab_capture_clear",
    "speaker_volume_set",
    "listen_start",
    "listen_stop",
    "follow_up_on",
    "follow_up_off",
    "touch_reactions_on",
    "touch_reactions_off",
    "lift_reactions_on",
    "lift_reactions_off",
    "behavior_recover",
    "behavior_normal",
    "behavior_happy",
    "behavior_curious",
    "behavior_thinking",
    "behavior_surprised",
    "behavior_fear",
    "behavior_focused",
    "behavior_angry",
    "behavior_love",
    "behavior_worried",
    "motors_on",
    "motors_off",
    "motor_forward",
    "motor_backward",
    "motor_left",
    "motor_right",
    "motor_stop",
    "wake_on",
    "wake_off",
    "barge_in_on",
    "barge_in_off",
    "soft_sleep_on",
    "night_sleep_on",
    "sleep_mode_on",
    "sleep_mode_off",
    "eyes_sleep_on",
    "eyes_sleep_off",
    "servo_left",
    "servo_right",
    "servo_center",
    "amp_mute_on",
    "amp_mute_off",
    "radar_calibrate_empty",
    "radar_clear_empty",
    "reconnect",
    "reboot",
}


def validate_barge_capture_wav(data: bytes) -> dict[str, int]:
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("ESP diagnostic capture is not a RIFF/WAVE file")
    if data[12:16] != b"fmt " or data[36:40] != b"data":
        raise ValueError("ESP diagnostic capture has an unsupported WAV layout")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", data, 20)
    declared_data_bytes = struct.unpack_from("<I", data, 40)[0]
    if audio_format != 1 or channels != 4 or sample_rate != 16000 or bits != 16:
        raise ValueError(
            f"Unexpected ESP diagnostic WAV format: format={audio_format} channels={channels} "
            f"rate={sample_rate} bits={bits}"
        )
    expected_block_align = channels * (bits // 8)
    if block_align != expected_block_align or byte_rate != sample_rate * expected_block_align:
        raise ValueError("ESP diagnostic WAV byte alignment is invalid")
    if declared_data_bytes <= 0 or declared_data_bytes != len(data) - 44:
        raise ValueError("ESP diagnostic WAV payload is incomplete")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "bits": bits,
        "frames": declared_data_bytes // block_align,
        "data_bytes": declared_data_bytes,
    }


SAFE_MODE_ALLOWED_COMMANDS = {
    "reconnect",
    "speaker_volume_set",
    "follow_up_on",
    "follow_up_off",
    "touch_reactions_on",
    "touch_reactions_off",
    "lift_reactions_on",
    "lift_reactions_off",
    "motor_stop",
    "motors_off",
    "radar_calibrate_empty",
    "radar_clear_empty",
    "soft_sleep_on",
    "night_sleep_on",
    "sleep_mode_on",
    "sleep_mode_off",
    "eyes_sleep_on",
    "eyes_sleep_off",
}
MIC_CAPTURE_MAX_BYTES = 768 * 1024
WS_HEARTBEAT_SECONDS = 20.0
WS_STABLE_CONNECTION_SECONDS = 60.0
WS_FLAP_LOG_INTERVAL_SECONDS = 5 * 60.0
WS_RUNTIME_STATUS_FIELDS = (
    "ws_connected",
    "ws_url",
    "last_ws_error",
    "last_ws_connected_at",
    "last_ws_disconnected_at",
    "last_ws_message_at",
    "ws_connection_started_at",
    "last_ws_connection_duration_sec",
    "ws_disconnect_streak",
    "reconnects",
    "max_auto_reconnects",
    "auto_reconnect_paused",
)


class EspClient:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus, ws_hub: WsHub) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._status: dict[str, Any] = copy.deepcopy(DEFAULT_STATUS)
        self._poll_task: asyncio.Task[None] | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_send_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._stop = asyncio.Event()
        self._last_poll_log_at = 0.0
        self._last_ws_log_at = 0.0
        self._poll_failure_streak = 0
        self._unreported_poll_failures = 0
        self._ws_connected_monotonic = 0.0
        self._ws_connection_stable = False
        self._ws_ever_connected = False
        self._ws_disconnect_streak = 0
        self._unreported_ws_disconnects = 0
        self._pause_log_emitted = False
        self._audio_ack_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._active_audio_stream_id = ""
        self._active_mic_stream: dict[str, Any] | None = None
        self._active_mic_buffer = bytearray()
        self._mic_stream_handler: Callable[[dict[str, Any], bytes], Awaitable[dict[str, Any]]] | None = None
        self._tts_timing_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._last_reset_reason_logged = ""
        self._temperature_alert_level = "ok"
        self._last_diag_log_at = 0.0
        self._last_runtime_config_signature = ""
        self._last_runtime_config_attempt_at = 0.0
        self._last_esp_uptime_sec: float | None = None

    def set_mic_stream_handler(
        self,
        handler: Callable[[dict[str, Any], bytes], Awaitable[dict[str, Any]]] | None,
    ) -> None:
        self._mic_stream_handler = handler

    def set_tts_timing_handler(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        self._tts_timing_handler = handler

    async def start(self) -> None:
        if self._poll_task and not self._poll_task.done():
            return
        self._stop.clear()
        self._session = aiohttp.ClientSession()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="alice-esp-poll")
        self._ws_task = asyncio.create_task(self._ws_loop(), name="alice-esp-ws")
        await self._log_bus.emit("INFO", "ESP", "ESP manager started")

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._poll_task, self._ws_task):
            if task:
                task.cancel()
        for task in (self._poll_task, self._ws_task):
            if not task:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None
        self._ws = None

    async def status(self) -> dict[str, Any]:
        return dict(self._status)

    async def audio_stream_ready(self) -> bool:
        return bool(self._status.get("ws_connected")) and self._ws is not None and not self._ws.closed

    async def send_audio_start(
        self,
        sample_rate: int,
        channels: int = 1,
        encoding: str = "pcm_s16le",
        stream_id: str = "",
    ) -> str:
        config = await self._config_store.get(include_secrets=True)
        timeout_sec = max(0.5, float(config.get("esp", {}).get("audio_ack_timeout_sec") or 3))
        stream_id = stream_id or f"tts-{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        ack_waiter: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._audio_ack_waiters[stream_id] = ack_waiter
        try:
            await self._send_ws_json(
                {
                    "type": "audio_start",
                    "stream_id": stream_id,
                    "payload": {
                        "stream_id": stream_id,
                        "encoding": encoding,
                        "sample_rate": sample_rate,
                        "channels": channels,
                    },
                }
            )
        except Exception:
            self._audio_ack_waiters.pop(stream_id, None)
            raise
        await self._log_bus.emit(
            "INFO",
            "ESP",
            "ESP audio start sent",
            {"stream_id": stream_id, "sample_rate": sample_rate, "channels": channels},
        )
        try:
            ack = await asyncio.wait_for(ack_waiter, timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            self._audio_ack_waiters.pop(stream_id, None)
            raise RuntimeError(f"ESP audio start ACK timed out for {stream_id}.") from exc
        if not ack.get("ok"):
            message = str(ack.get("message") or "ESP rejected audio start.")
            raise RuntimeError(message)
        self._active_audio_stream_id = stream_id
        return stream_id

    async def send_audio_chunk(self, chunk: bytes, stream_id: str = "") -> None:
        if not chunk:
            return
        async with self._ws_send_lock:
            ws = self._ws
            if ws is None or ws.closed:
                raise RuntimeError("ESP WebSocket is not connected.")
            await ws.send_bytes(chunk)

    async def send_audio_end(self, ok: bool = True, message: str = "", stream_id: str = "") -> None:
        stream_id = stream_id or self._active_audio_stream_id
        await self._send_ws_json(
            {
                "type": "audio_end",
                "stream_id": stream_id,
                "payload": {"stream_id": stream_id, "ok": ok, "message": message},
            }
        )
        if stream_id and stream_id == self._active_audio_stream_id:
            self._active_audio_stream_id = ""

    async def send_audio_error(self, message: str, stream_id: str = "") -> None:
        stream_id = stream_id or self._active_audio_stream_id
        await self._send_ws_json(
            {
                "type": "audio_error",
                "stream_id": stream_id,
                "payload": {"stream_id": stream_id, "message": message},
            }
        )
        if stream_id and stream_id == self._active_audio_stream_id:
            self._active_audio_stream_id = ""

    async def poll_once(self, force: bool = False) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        esp_cfg = config.get("esp", {})
        base_url = str(esp_cfg.get("base_url") or "").strip().rstrip("/")
        timeout_sec = float(esp_cfg.get("command_timeout_sec") or 4)
        self._status["max_auto_reconnects"] = self._max_auto_reconnects(esp_cfg)
        if not base_url:
            self._status = self._offline_status("ESP base URL is not configured")
            return await self.status()
        if not force and await self._pause_if_reconnect_limit_reached(esp_cfg):
            return await self.status()
        session = self._session or aiohttp.ClientSession()
        self._session = session
        try:
            previous_uptime = self._last_esp_uptime_sec
            async with session.get(f"{base_url}/api/status", timeout=timeout_sec) as resp:
                doc = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"ESP status HTTP {resp.status}: {doc}")
            self._status = self._normalize_status(doc, base_url)
            current_uptime = self._float_or_none(self._status.get("uptime_sec"))
            if previous_uptime is not None and current_uptime is not None and current_uptime + 5 < previous_uptime:
                self._last_runtime_config_signature = ""
            self._last_esp_uptime_sec = current_uptime
            await self._sync_runtime_config_if_needed(config)
            if self._poll_failure_streak:
                await self._log_bus.emit(
                    "INFO",
                    "ESP",
                    "ESP status polling recovered",
                    {"failed_polls": self._poll_failure_streak},
                )
                self._poll_failure_streak = 0
                self._unreported_poll_failures = 0
            await self._observe_system_diagnostics(self._status)
            await self._ws_hub.publish("esp_status", self._status)
        except Exception as exc:
            reconnects = self._record_reconnect_failure(esp_cfg)
            self._status = self._offline_status(str(exc), reconnects=reconnects)
            self._poll_failure_streak += 1
            self._unreported_poll_failures += 1
            now = time.time()
            if self._poll_failure_streak == 1 or now - self._last_poll_log_at >= WS_FLAP_LOG_INTERVAL_SECONDS:
                self._last_poll_log_at = now
                await self._log_bus.emit(
                    "WARN",
                    "ESP",
                    "ESP status poll failed" if self._poll_failure_streak == 1 else "ESP status polling repeatedly failing",
                    {
                        "error": str(exc),
                        "failures_since_last_log": self._unreported_poll_failures,
                        "failure_streak": self._poll_failure_streak,
                    },
                )
                self._unreported_poll_failures = 0
        return await self.status()

    async def send_command(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        esp_cfg = config.get("esp", {})
        base_url = str(esp_cfg.get("base_url") or "").strip().rstrip("/")
        timeout_sec = float(esp_cfg.get("command_timeout_sec") or 4)
        payload = payload or {}
        if command not in ESP_COMMANDS:
            await self._log_bus.emit("WARN", "ESP", "Unknown ESP command rejected", {"command": command})
            return {"ok": False, "implemented": False, "message": "Unknown ESP command.", "command": command}
        if bool(config.get("safe_mode")) and command not in SAFE_MODE_ALLOWED_COMMANDS:
            await self._log_bus.emit("WARN", "ESP", "ESP command blocked by safe mode", {"command": command})
            return {
                "ok": False,
                "implemented": False,
                "blocked_by_safe_mode": True,
                "message": "Safe mode is enabled; ESP hardware command was not sent.",
                "command": command,
            }
        if command == "reconnect":
            self._reset_reconnect_budget()
            await self._log_bus.emit("INFO", "ESP", "Manual ESP reconnect requested")
            await self.poll_once(force=True)
        if not base_url:
            await self._log_bus.emit(
                "WARN",
                "ESP",
                "ESP command accepted in mock mode",
                {"command": command, "payload": payload},
            )
            return {
                "ok": False,
                "implemented": False,
                "mock_mode": True,
                "message": "ESP base_url is empty; command logged only.",
                "command": command,
            }
        session = self._session or aiohttp.ClientSession()
        self._session = session
        body = {"command": command, "payload": payload}
        try:
            async with session.post(f"{base_url}/api/command", json=body, timeout=timeout_sec) as resp:
                doc = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"ESP command HTTP {resp.status}: {doc}")
            await self._log_bus.emit("INFO", "ESP", "ESP command sent", {"command": command})
            return {"ok": True, "implemented": True, "response": doc, "command": command}
        except Exception as exc:
            await self._log_bus.emit("ERROR", "ESP", "ESP command failed", {"command": command, "error": str(exc)})
            return {"ok": False, "implemented": False, "message": str(exc), "command": command}

    async def fetch_barge_lab_capture(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        esp_cfg = config.get("esp", {})
        base_url = str(esp_cfg.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("ESP base_url is empty")
        timeout_sec = max(30.0, float(esp_cfg.get("command_timeout_sec") or 4))
        session = self._session or aiohttp.ClientSession()
        self._session = session
        try:
            async with session.get(f"{base_url}/api/barge-lab-capture.wav", timeout=timeout_sec) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    message = data.decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(f"ESP diagnostic capture HTTP {resp.status}: {message}")
            metadata = validate_barge_capture_wav(data)
            await self._log_bus.emit(
                "INFO",
                "BARGE_IN",
                "ESP four-channel diagnostic WAV received",
                metadata,
            )
            return {"data": data, "metadata": metadata}
        except Exception as exc:
            await self._log_bus.emit(
                "ERROR",
                "BARGE_IN",
                "ESP diagnostic WAV download failed",
                {"error": str(exc)},
            )
            raise

    async def get_config(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        base_url = str(config.get("esp", {}).get("base_url") or "").strip().rstrip("/")
        if not base_url:
            return {"ok": False, "mock_mode": True, "config": {}}
        session = self._session or aiohttp.ClientSession()
        self._session = session
        async with session.get(f"{base_url}/api/config", timeout=4) as resp:
            doc = await resp.json(content_type=None)
        return {"ok": resp.status < 400, "status": resp.status, "config": doc}

    async def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        base_url = str(config.get("esp", {}).get("base_url") or "").strip().rstrip("/")
        if not base_url:
            return {"ok": False, "mock_mode": True, "message": "ESP base_url is empty"}
        session = self._session or aiohttp.ClientSession()
        self._session = session
        async with session.post(f"{base_url}/api/config", json=patch, timeout=4) as resp:
            doc = await resp.json(content_type=None)
        return {"ok": resp.status < 400, "status": resp.status, "response": doc}

    async def _sync_runtime_config_if_needed(self, config: dict[str, Any]) -> None:
        pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
        barge_lab = config.get("barge_lab") if isinstance(config.get("barge_lab"), dict) else {}
        patch = {
            "barge_in_enabled": bool(pipeline.get("barge_in_enabled", True)),
            "barge_lab": barge_lab,
        }
        signature = json.dumps(patch, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if signature == self._last_runtime_config_signature:
            return
        now = time.time()
        if now - self._last_runtime_config_attempt_at < 10:
            return
        self._last_runtime_config_attempt_at = now
        try:
            result = await self.update_config(patch)
        except Exception:
            return
        if result.get("ok"):
            self._last_runtime_config_signature = signature

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            config = await self._config_store.get(include_secrets=True)
            interval = max(1.0, float(config.get("esp", {}).get("poll_interval_sec") or 3))
            await self.poll_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _ws_loop(self) -> None:
        while not self._stop.is_set():
            config = await self._config_store.get(include_secrets=True)
            esp_cfg = config.get("esp", {})
            reconnect_sec = max(1.0, float(esp_cfg.get("reconnect_sec") or 5))
            self._status["max_auto_reconnects"] = self._max_auto_reconnects(esp_cfg)
            ws_url = self._resolve_ws_url(esp_cfg)
            if not ws_url:
                self._set_ws_state(False, "", "")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=reconnect_sec)
                except asyncio.TimeoutError:
                    pass
                continue
            if await self._pause_if_reconnect_limit_reached(esp_cfg):
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=max(reconnect_sec, 15.0))
                except asyncio.TimeoutError:
                    pass
                continue
            session = self._session or aiohttp.ClientSession()
            self._session = session
            active_ws: aiohttp.ClientWebSocketResponse | None = None
            try:
                async with session.ws_connect(
                    ws_url,
                    heartbeat=WS_HEARTBEAT_SECONDS,
                    receive_timeout=120,
                ) as ws:
                    active_ws = ws
                    self._ws = ws
                    first_connection = not self._ws_ever_connected
                    self._begin_ws_connection(ws_url)
                    if first_connection:
                        await self._log_bus.emit("INFO", "ESP", "ESP WebSocket connected", {"url": ws_url})
                    await self._ws_hub.publish("esp_status", await self.status())
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._note_ws_activity(ws_url)
                            await self._handle_ws_text(msg.data, ws_url)
                            continue
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            await self._note_ws_activity(ws_url)
                            await self._handle_ws_binary(msg.data)
                            continue
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"ESP websocket error: {ws.exception()}")
                    if not self._stop.is_set():
                        raise RuntimeError("ESP websocket closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                connected_seconds = self._finish_ws_connection()
                self._status["reconnects"] = self._record_reconnect_failure(esp_cfg)
                self._set_ws_state(False, ws_url, str(exc))
                await self._record_ws_disconnect(ws_url, str(exc), connected_seconds)
            finally:
                if self._ws is active_ws:
                    self._ws = None
                self._active_audio_stream_id = ""
                self._clear_mic_stream("ESP websocket disconnected")
                self._fail_audio_ack_waiters("ESP websocket disconnected")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=reconnect_sec)
            except asyncio.TimeoutError:
                pass

    async def _handle_ws_text(self, raw: str, ws_url: str) -> None:
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            await self._log_bus.emit("INFO", "ESP", raw[:300])
            return
        if not isinstance(doc, dict):
            await self._ws_hub.publish("esp_event", {"value": doc})
            return
        msg_type = str(doc.get("type") or doc.get("event") or "").lower()
        payload = doc.get("payload") if isinstance(doc.get("payload"), dict) else doc
        if msg_type == "status":
            status_payload = payload.get("status") if isinstance(payload.get("status"), dict) else payload
            self._status = self._normalize_status(status_payload, ws_url)
            self._set_ws_state(True, ws_url, "")
            await self._observe_system_diagnostics(self._status)
            await self._ws_hub.publish("esp_status", await self.status())
            return
        if msg_type == "log":
            level = str(payload.get("level") or "INFO")
            category = str(payload.get("category") or "ESP")
            message = str(payload.get("message") or payload.get("msg") or "")
            details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
            await self._log_bus.emit(level, category, message or "ESP log event", details)
            return
        if msg_type in {"audio_ready", "audio_rejected"}:
            stream_id = str(doc.get("stream_id") or payload.get("stream_id") or "")
            ok = msg_type == "audio_ready"
            message = str(payload.get("message") or ("ESP audio stream ready" if ok else "ESP audio stream rejected"))
            event = {"ok": ok, "stream_id": stream_id, "message": message, "payload": payload}
            waiter = self._audio_ack_waiters.pop(stream_id, None) if stream_id else None
            if waiter is None and len(self._audio_ack_waiters) == 1:
                _, waiter = self._audio_ack_waiters.popitem()
            if waiter is not None and not waiter.done():
                waiter.set_result(event)
            await self._log_bus.emit(
                "INFO" if ok else "WARN",
                "ESP",
                "ESP audio start accepted" if ok else "ESP audio start rejected",
                {"stream_id": stream_id, "message": message},
            )
            await self._ws_hub.publish("esp_event", {"type": msg_type, "payload": payload})
            return
        if msg_type == "tts_timing":
            event_payload = {"type": "tts_timing", "payload": payload}
            await self._ws_hub.publish("esp_event", event_payload)
            handler = self._tts_timing_handler
            if handler is not None:
                await handler(dict(payload))
            return
        if msg_type in {"barge_sample", "barge_samples"}:
            await self._ws_hub.publish("esp_event", {"type": msg_type, "payload": payload})
            return
        if msg_type == "mic_start":
            await self._handle_mic_start(doc, payload)
            return
        if msg_type == "mic_end":
            await self._handle_mic_end(doc, payload)
            return
        if msg_type == "mic_error":
            await self._handle_mic_error(doc, payload)
            return
        if msg_type == "radar_targets":
            self._status["radar"] = payload
            hardware = self._status.setdefault("hardware", {})
            if isinstance(hardware, dict):
                hardware["radar"] = str(payload.get("state") or "unknown")
            await self._ws_hub.publish("esp_event", {"type": "radar_targets", "payload": payload})
            return
        event_payload = {"type": msg_type or "event", "payload": payload}
        await self._ws_hub.publish("esp_event", event_payload)
        await self._log_bus.emit("INFO", "ESP", f"ESP event: {event_payload['type']}", {"payload": payload})

    async def _handle_ws_binary(self, data: bytes) -> None:
        if not self._active_mic_stream:
            await self._ws_hub.publish("esp_binary", {"bytes": len(data)})
            return
        stream = self._active_mic_stream
        stream["chunks"] = int(stream.get("chunks") or 0) + 1
        stream["bytes_received"] = int(stream.get("bytes_received") or 0) + len(data)
        remaining = MIC_CAPTURE_MAX_BYTES - len(self._active_mic_buffer)
        if remaining <= 0:
            stream["truncated"] = True
            return
        if len(data) > remaining:
            self._active_mic_buffer.extend(data[:remaining])
            stream["truncated"] = True
            return
        self._active_mic_buffer.extend(data)

    async def _handle_mic_start(self, doc: dict[str, Any], payload: dict[str, Any]) -> None:
        if self._active_mic_stream:
            await self._handle_mic_end({"type": "mic_end"}, {"message": "replaced by a new mic stream"})
        stream_id = str(doc.get("stream_id") or payload.get("stream_id") or f"mic-{uuid.uuid4().hex}")
        sample_rate = int(payload.get("sample_rate") or doc.get("sample_rate") or 16000)
        channels = int(payload.get("channels") or doc.get("channels") or 1)
        encoding = str(payload.get("encoding") or doc.get("encoding") or "pcm_s16le")
        channel = str(payload.get("channel") or doc.get("channel") or "current")
        purpose = str(payload.get("purpose") or doc.get("purpose") or "pipeline")
        try:
            shift_bits = int(payload.get("shift_bits") or doc.get("shift_bits"))
        except (TypeError, ValueError):
            shift_bits = None
        self._active_mic_stream = {
            "stream_id": stream_id,
            "sample_rate": sample_rate,
            "channels": channels,
            "encoding": encoding,
            "channel": channel,
            "purpose": purpose,
            "shift_bits": shift_bits,
            "source": str(payload.get("source") or "esp"),
            "started_at": time.time(),
            "bytes_received": 0,
            "chunks": 0,
            "truncated": False,
        }
        self._active_mic_buffer = bytearray()
        await self._log_bus.emit(
            "INFO",
            "STT",
            "ESP mic stream started",
            {
                "stream_id": stream_id,
                "sample_rate": sample_rate,
                "channels": channels,
                "channel": channel,
                "purpose": purpose,
                "shift_bits": shift_bits,
            },
        )
        await self._ws_hub.publish("esp_event", {"type": "mic_start", "payload": dict(self._active_mic_stream)})

    async def _handle_mic_end(self, doc: dict[str, Any], payload: dict[str, Any]) -> None:
        if not self._active_mic_stream:
            await self._log_bus.emit("WARN", "STT", "ESP mic end received without active stream")
            return
        stream = dict(self._active_mic_stream)
        stream.update({
            "message": str(payload.get("message") or doc.get("message") or ""),
            "ended_at": time.time(),
            "bytes_buffered": len(self._active_mic_buffer),
        })
        for key in (
            "samples",
            "avg_abs",
            "rms",
            "peak",
            "clip_pct",
            "silent_pct",
            "shift_bits",
            "duration_ms",
            "bytes",
            "chunks",
        ):
            value = payload.get(key, doc.get(key))
            if value is not None:
                stream[key] = value
        stream["duration_sec"] = round(float(stream["ended_at"] - stream.get("started_at", stream["ended_at"])), 2)
        audio = bytes(self._active_mic_buffer)
        self._active_mic_stream = None
        self._active_mic_buffer = bytearray()
        await self._log_bus.emit(
            "INFO",
            "STT",
            "ESP mic stream completed",
            {
                "stream_id": stream.get("stream_id"),
                "bytes": stream.get("bytes_buffered"),
                "chunks": stream.get("chunks"),
                "truncated": stream.get("truncated"),
                "rms": stream.get("rms"),
                "peak": stream.get("peak"),
                "clip_pct": stream.get("clip_pct"),
                "shift_bits": stream.get("shift_bits"),
            },
        )
        await self._ws_hub.publish("esp_event", {"type": "mic_end", "payload": stream})
        self._dispatch_mic_capture(stream, audio)

    async def _handle_mic_error(self, doc: dict[str, Any], payload: dict[str, Any]) -> None:
        message = str(payload.get("message") or doc.get("message") or "ESP mic stream error")
        stream = dict(payload or {})
        if self._active_mic_stream:
            stream.update(self._active_mic_stream)
        stream["message"] = message
        self._active_mic_stream = None
        self._active_mic_buffer = bytearray()
        await self._log_bus.emit(
            "WARN",
            "STT",
            message,
            {
                "stream_id": stream.get("stream_id"),
                "channel": stream.get("channel"),
                "purpose": stream.get("purpose"),
            },
        )
        await self._ws_hub.publish("esp_event", {"type": "mic_error", "payload": stream})

    def _dispatch_mic_capture(self, metadata: dict[str, Any], audio: bytes) -> None:
        handler = self._mic_stream_handler
        if handler is None:
            return

        async def run_handler() -> None:
            try:
                await handler(metadata, audio)
            except Exception as exc:
                await self._log_bus.emit("ERROR", "STT", "Mic capture handler failed", {"error": str(exc)})

        asyncio.create_task(run_handler(), name="alice-mic-capture-handler")

    def _clear_mic_stream(self, message: str) -> None:
        if self._active_mic_stream:
            self._active_mic_stream["error"] = message
        self._active_mic_stream = None
        self._active_mic_buffer = bytearray()

    def _set_ws_state(self, connected: bool, ws_url: str, error: str) -> None:
        self._status["ws_connected"] = connected
        self._status["ws_url"] = ws_url
        self._status["last_ws_error"] = error
        if connected:
            self._status["online"] = True
            self._status["mock_mode"] = False
            self._status["last_seen"] = time.time()
            self._status["ip"] = self._status.get("ip") or self._host_from_url(ws_url)
            if str(self._status.get("state") or "").upper() == "OFFLINE":
                self._status["state"] = "IDLE"

    def _offline_status(self, error: str, reconnects: int | None = None) -> dict[str, Any]:
        status = copy.deepcopy(DEFAULT_STATUS)
        status["mock_mode"] = True
        status["last_error"] = error
        self._copy_ws_runtime_status(status)
        status["reconnects"] = int(reconnects if reconnects is not None else self._status.get("reconnects") or 0)
        return status

    def _normalize_status(self, doc: dict[str, Any], base_url: str) -> dict[str, Any]:
        status = copy.deepcopy(DEFAULT_STATUS)
        status.update(doc if isinstance(doc, dict) else {})
        status["online"] = True
        status["mock_mode"] = False
        status["ip"] = status.get("ip") or self._host_from_url(base_url)
        status["state"] = str(status.get("state") or "IDLE").upper()
        status["last_seen"] = time.time()
        status["last_error"] = ""
        self._copy_ws_runtime_status(status)
        return status

    def _copy_ws_runtime_status(self, target: dict[str, Any]) -> None:
        for key in WS_RUNTIME_STATUS_FIELDS:
            target[key] = copy.deepcopy(self._status.get(key, DEFAULT_STATUS.get(key)))

    def _begin_ws_connection(self, ws_url: str) -> None:
        now = time.time()
        self._ws_ever_connected = True
        self._ws_connected_monotonic = time.monotonic()
        self._ws_connection_stable = False
        self._status["last_ws_connected_at"] = now
        self._status["ws_connection_started_at"] = now
        self._set_ws_state(True, ws_url, "")

    async def _note_ws_activity(self, ws_url: str) -> None:
        self._status["last_ws_message_at"] = time.time()
        if self._ws_connection_stable or not self._ws_connected_monotonic:
            return
        connected_seconds = time.monotonic() - self._ws_connected_monotonic
        if connected_seconds < WS_STABLE_CONNECTION_SECONDS:
            return
        self._ws_connection_stable = True
        interruptions = self._ws_disconnect_streak
        unreported = self._unreported_ws_disconnects
        self._reset_reconnect_budget()
        self._ws_disconnect_streak = 0
        self._unreported_ws_disconnects = 0
        self._status["ws_disconnect_streak"] = 0
        if interruptions:
            await self._log_bus.emit(
                "INFO",
                "ESP",
                "ESP WebSocket connection recovered",
                {
                    "url": ws_url,
                    "interruptions": interruptions,
                    "unreported_interruptions": unreported,
                    "stable_seconds": int(connected_seconds),
                },
            )

    def _finish_ws_connection(self) -> float:
        if not self._ws_connected_monotonic:
            return 0.0
        connected_seconds = max(0.0, time.monotonic() - self._ws_connected_monotonic)
        self._ws_connected_monotonic = 0.0
        self._status["last_ws_connection_duration_sec"] = round(connected_seconds, 1)
        self._status["last_ws_disconnected_at"] = time.time()
        self._status["ws_connection_started_at"] = None
        return connected_seconds

    async def _record_ws_disconnect(self, ws_url: str, error: str, connected_seconds: float) -> None:
        self._ws_disconnect_streak += 1
        self._unreported_ws_disconnects += 1
        self._status["ws_disconnect_streak"] = self._ws_disconnect_streak
        now = time.time()
        max_attempts = int(self._status.get("max_auto_reconnects") or 0)
        reconnects = int(self._status.get("reconnects") or 0)
        should_log = (
            self._ws_disconnect_streak == 1
            or now - self._last_ws_log_at >= WS_FLAP_LOG_INTERVAL_SECONDS
            or bool(max_attempts and reconnects >= max_attempts)
        )
        if not should_log:
            return
        self._last_ws_log_at = now
        pending = self._unreported_ws_disconnects
        self._unreported_ws_disconnects = 0
        await self._log_bus.emit(
            "WARN",
            "ESP",
            "ESP WebSocket disconnected" if self._ws_disconnect_streak == 1 else "ESP WebSocket repeatedly disconnecting",
            {
                "url": ws_url,
                "error": error,
                "connected_seconds": round(connected_seconds, 1),
                "disconnects_since_last_log": pending,
                "disconnect_streak": self._ws_disconnect_streak,
                "reconnects": reconnects,
                "max_auto_reconnects": max_attempts,
            },
        )

    async def _observe_system_diagnostics(self, status: dict[str, Any]) -> None:
        system = status.get("system") if isinstance(status.get("system"), dict) else {}
        reason = str(system.get("reset_reason") or "")
        risk = str(system.get("reset_risk") or "info").lower()
        if reason and reason != self._last_reset_reason_logged:
            self._last_reset_reason_logged = reason
            await self._log_bus.emit(
                "WARN" if risk == "warn" else "INFO",
                "ESP",
                f"ESP reset reason: {reason}",
                {"system": system},
            )

        temperature = self._float_or_none(system.get("temperature_c"))
        if temperature is None:
            return
        now = time.time()
        next_level = "error" if temperature >= 78.0 else "warn" if temperature >= 70.0 else "ok"
        if next_level in {"warn", "error"} and next_level != self._temperature_alert_level:
            self._temperature_alert_level = next_level
            self._last_diag_log_at = now
            await self._log_bus.emit(
                "ERROR" if next_level == "error" else "WARN",
                "ESP",
                "ESP temperature high",
                {"temperature_c": temperature, "system": system},
            )
        elif next_level == "ok" and self._temperature_alert_level != "ok" and temperature <= 65.0:
            if now - self._last_diag_log_at > 60:
                self._temperature_alert_level = "ok"
                self._last_diag_log_at = now
                await self._log_bus.emit("INFO", "ESP", "ESP temperature recovered", {"temperature_c": temperature})

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number else None

    async def _send_ws_json(self, payload: dict[str, Any]) -> None:
        async with self._ws_send_lock:
            ws = self._ws
            if ws is None or ws.closed:
                raise RuntimeError("ESP WebSocket is not connected.")
            await ws.send_json(payload)

    def _fail_audio_ack_waiters(self, message: str) -> None:
        waiters = self._audio_ack_waiters
        self._audio_ack_waiters = {}
        for waiter in waiters.values():
            if not waiter.done():
                waiter.set_result({"ok": False, "message": message})

    def _max_auto_reconnects(self, esp_cfg: dict[str, Any]) -> int:
        try:
            return max(0, int(esp_cfg.get("max_auto_reconnects", 40)))
        except (TypeError, ValueError):
            return 40

    def _record_reconnect_failure(self, esp_cfg: dict[str, Any]) -> int:
        max_attempts = self._max_auto_reconnects(esp_cfg)
        self._status["max_auto_reconnects"] = max_attempts
        reconnects = int(self._status.get("reconnects") or 0) + 1
        if max_attempts and reconnects >= max_attempts:
            reconnects = max_attempts
            self._status["auto_reconnect_paused"] = True
        return reconnects

    def _reset_reconnect_budget(self) -> None:
        self._status["reconnects"] = 0
        self._status["auto_reconnect_paused"] = False
        self._status["last_error"] = ""
        self._status["last_ws_error"] = ""
        self._pause_log_emitted = False

    async def _pause_if_reconnect_limit_reached(self, esp_cfg: dict[str, Any]) -> bool:
        max_attempts = self._max_auto_reconnects(esp_cfg)
        if not max_attempts:
            self._status["auto_reconnect_paused"] = False
            self._status["max_auto_reconnects"] = 0
            self._pause_log_emitted = False
            return False
        reconnects = int(self._status.get("reconnects") or 0)
        if reconnects < max_attempts:
            self._status["auto_reconnect_paused"] = False
            self._status["max_auto_reconnects"] = max_attempts
            self._pause_log_emitted = False
            return False
        self._status["reconnects"] = max_attempts
        self._status["max_auto_reconnects"] = max_attempts
        self._status["auto_reconnect_paused"] = True
        self._status["online"] = False
        self._status["mock_mode"] = True
        self._status["ws_connected"] = False
        self._status["last_error"] = (
            f"Auto reconnect paused after {max_attempts} failed attempts. "
            "Press reconnect to try again."
        )
        if not self._pause_log_emitted:
            self._pause_log_emitted = True
            await self._log_bus.emit(
                "WARN",
                "ESP",
                "ESP auto reconnect paused",
                {"max_auto_reconnects": max_attempts},
            )
        return True

    @staticmethod
    def _resolve_ws_url(esp_cfg: dict[str, Any]) -> str:
        explicit = str(esp_cfg.get("ws_url") or "").strip()
        if explicit:
            return explicit
        base_url = str(esp_cfg.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            return ""
        if base_url.startswith("https://"):
            return f"wss://{base_url[8:]}/ws"
        if base_url.startswith("http://"):
            return f"ws://{base_url[7:]}/ws"
        return f"ws://{base_url}/ws"

    @staticmethod
    def _host_from_url(url: str) -> str:
        for prefix in ("https://", "http://", "wss://", "ws://"):
            if url.startswith(prefix):
                return url[len(prefix) :].split("/")[0]
        return url.split("/")[0]
