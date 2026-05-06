from __future__ import annotations

import asyncio
import copy
import json
import time
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
        self._session: aiohttp.ClientSession | None = None
        self._stop = asyncio.Event()
        self._last_ws_log_at = 0.0

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

    async def status(self) -> dict[str, Any]:
        return dict(self._status)

    async def poll_once(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        esp_cfg = config.get("esp", {})
        base_url = str(esp_cfg.get("base_url") or "").strip().rstrip("/")
        timeout_sec = float(esp_cfg.get("command_timeout_sec") or 4)
        if not base_url:
            self._status = self._offline_status("ESP base URL is not configured")
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
            old_reconnects = int(self._status.get("reconnects") or 0)
            self._status = self._offline_status(str(exc), reconnects=old_reconnects + 1)
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
            await self.poll_once()
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
            ws_url = self._resolve_ws_url(esp_cfg)
            if not ws_url:
                self._set_ws_state(False, "", "")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=reconnect_sec)
                except asyncio.TimeoutError:
                    pass
                continue
            session = self._session or aiohttp.ClientSession()
            self._session = session
            try:
                async with session.ws_connect(ws_url, heartbeat=20, receive_timeout=120) as ws:
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                old_reconnects = int(self._status.get("reconnects") or 0)
                self._status["reconnects"] = old_reconnects + 1
                self._set_ws_state(False, ws_url, str(exc))
                now = time.time()
                if now - self._last_ws_log_at > 15:
                    self._last_ws_log_at = now
                    await self._log_bus.emit("WARN", "ESP", "ESP WebSocket disconnected", {"url": ws_url, "error": str(exc)})
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
        if "reconnects" not in status or status.get("reconnects") is None:
            status["reconnects"] = int(self._status.get("reconnects") or 0)
        return status

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
