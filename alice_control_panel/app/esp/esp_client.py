from __future__ import annotations

import asyncio
import copy
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
    "reconnects": 0,
}


class EspClient:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus, ws_hub: WsHub) -> None:
        self._config_store = config_store
        self._log_bus = log_bus
        self._ws_hub = ws_hub
        self._status: dict[str, Any] = copy.deepcopy(DEFAULT_STATUS)
        self._task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop(), name="alice-esp-poll")
        await self._log_bus.emit("INFO", "ESP", "ESP manager started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
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

    def _offline_status(self, error: str, reconnects: int | None = None) -> dict[str, Any]:
        status = copy.deepcopy(DEFAULT_STATUS)
        status["mock_mode"] = True
        status["last_error"] = error
        status["reconnects"] = int(reconnects if reconnects is not None else self._status.get("reconnects") or 0)
        return status

    @staticmethod
    def _normalize_status(doc: dict[str, Any], base_url: str) -> dict[str, Any]:
        status = copy.deepcopy(DEFAULT_STATUS)
        status.update(doc if isinstance(doc, dict) else {})
        status["online"] = True
        status["mock_mode"] = False
        status["ip"] = status.get("ip") or base_url.replace("http://", "").replace("https://", "").split("/")[0]
        status["state"] = str(status.get("state") or "IDLE").upper()
        status["last_seen"] = time.time()
        status["last_error"] = ""
        return status
