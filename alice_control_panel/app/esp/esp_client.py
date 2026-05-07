from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
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
    "last_error": "",
    "ws_connected": False,
    "ws_url": "",
    "last_ws_error": "",
    "reconnects": 0,
    "max_auto_reconnects": 40,
    "auto_reconnect_paused": False,
}


ESP_COMMANDS = {
    "test_speaker",
    "test_mic",
    "wake_on",
    "wake_off",
    "servo_left",
    "servo_right",
    "servo_center",
    "amp_mute_on",
    "amp_mute_off",
    "reconnect",
    "reboot",
}
SAFE_MODE_ALLOWED_COMMANDS = {"reconnect"}


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
        self._last_pause_log_at = 0.0
        self._audio_ack_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._active_audio_stream_id = ""

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
            async with session.get(f"{base_url}/api/status", timeout=timeout_sec) as resp:
                doc = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise RuntimeError(f"ESP status HTTP {resp.status}: {doc}")
            self._status = self._normalize_status(doc, base_url)
            await self._ws_hub.publish("esp_status", self._status)
        except Exception as exc:
            reconnects = self._record_reconnect_failure(esp_cfg)
            self._status = self._offline_status(str(exc), reconnects=reconnects)
            now = time.time()
            if now - self._last_poll_log_at > 15:
                self._last_poll_log_at = now
                await self._log_bus.emit("WARN", "ESP", "ESP status poll failed", {"error": str(exc)})
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
                async with session.ws_connect(ws_url, heartbeat=20, receive_timeout=120) as ws:
                    active_ws = ws
                    self._ws = ws
                    self._set_ws_state(True, ws_url, "")
                    await self._log_bus.emit("INFO", "ESP", "ESP WebSocket connected", {"url": ws_url})
                    await self._ws_hub.publish("esp_status", await self.status())
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_text(msg.data, ws_url)
                            continue
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            await self._ws_hub.publish("esp_binary", {"bytes": len(msg.data)})
                            continue
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"ESP websocket error: {ws.exception()}")
                    self._set_ws_state(False, ws_url, "ESP websocket closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._status["reconnects"] = self._record_reconnect_failure(esp_cfg)
                self._set_ws_state(False, ws_url, str(exc))
                now = time.time()
                if now - self._last_ws_log_at > 15:
                    self._last_ws_log_at = now
                    await self._log_bus.emit("WARN", "ESP", "ESP WebSocket disconnected", {"url": ws_url, "error": str(exc)})
            finally:
                if self._ws is active_ws:
                    self._ws = None
                self._active_audio_stream_id = ""
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
        event_payload = {"type": msg_type or "event", "payload": payload}
        await self._ws_hub.publish("esp_event", event_payload)
        await self._log_bus.emit("INFO", "ESP", f"ESP event: {event_payload['type']}", {"payload": payload})

    def _set_ws_state(self, connected: bool, ws_url: str, error: str) -> None:
        self._status["ws_connected"] = connected
        self._status["ws_url"] = ws_url
        self._status["last_ws_error"] = error
        if connected:
            self._status["online"] = True
            self._status["mock_mode"] = False
            self._status["last_seen"] = time.time()
            self._status["reconnects"] = 0
            self._status["auto_reconnect_paused"] = False
            self._status["ip"] = self._status.get("ip") or self._host_from_url(ws_url)
            if str(self._status.get("state") or "").upper() == "OFFLINE":
                self._status["state"] = "IDLE"

    def _offline_status(self, error: str, reconnects: int | None = None) -> dict[str, Any]:
        status = copy.deepcopy(DEFAULT_STATUS)
        status["mock_mode"] = True
        status["last_error"] = error
        status["ws_connected"] = bool(self._status.get("ws_connected"))
        status["ws_url"] = str(self._status.get("ws_url") or "")
        status["last_ws_error"] = str(self._status.get("last_ws_error") or "")
        status["reconnects"] = int(reconnects if reconnects is not None else self._status.get("reconnects") or 0)
        status["max_auto_reconnects"] = int(self._status.get("max_auto_reconnects") or 0)
        status["auto_reconnect_paused"] = bool(self._status.get("auto_reconnect_paused"))
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
        status["ws_connected"] = bool(self._status.get("ws_connected"))
        status["ws_url"] = str(self._status.get("ws_url") or "")
        status["last_ws_error"] = str(self._status.get("last_ws_error") or "")
        status["reconnects"] = 0
        status["max_auto_reconnects"] = int(self._status.get("max_auto_reconnects") or 40)
        status["auto_reconnect_paused"] = False
        return status

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

    async def _pause_if_reconnect_limit_reached(self, esp_cfg: dict[str, Any]) -> bool:
        max_attempts = self._max_auto_reconnects(esp_cfg)
        if not max_attempts:
            self._status["auto_reconnect_paused"] = False
            self._status["max_auto_reconnects"] = 0
            return False
        reconnects = int(self._status.get("reconnects") or 0)
        if reconnects < max_attempts:
            self._status["auto_reconnect_paused"] = False
            self._status["max_auto_reconnects"] = max_attempts
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
        now = time.time()
        if now - self._last_pause_log_at > 60:
            self._last_pause_log_at = now
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
