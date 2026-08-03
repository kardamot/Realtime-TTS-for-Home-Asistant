from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.esp.esp_client import EspClient


class PowerManager:
    WAKE_HOLD_SECONDS = 120.0

    def __init__(self, config_store: ConfigStore, esp_client: EspClient, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._esp_client = esp_client
        self._log_bus = log_bus
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._last_activity_mono = time.monotonic()
        self._last_activity_reason = "startup"
        self._last_auto_mode = "active"
        self._last_desired_mode = "active"
        self._last_error = ""
        self._last_radar_presence = False
        self._last_observed_mode: str | None = None
        self._wake_hold_until_mono = 0.0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="alice-power-manager")
        await self._log_bus.emit("INFO", "SYSTEM", "Power manager started")

    async def stop(self) -> None:
        self._stop.set()
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def notify_activity(self, reason: str = "activity") -> None:
        now = time.monotonic()
        self._last_activity_mono = now
        self._last_activity_reason = reason
        if reason in {"command:sleep_mode_off", "command:eyes_sleep_off"}:
            self._wake_hold_until_mono = now + self.WAKE_HOLD_SECONDS
        elif reason in {
            "command:sleep_mode_on",
            "command:eyes_sleep_on",
            "command:soft_sleep_on",
            "command:night_sleep_on",
        }:
            self._wake_hold_until_mono = 0.0

    async def status(self) -> dict[str, Any]:
        cfg = await self._config_store.get(include_secrets=False)
        power_cfg = cfg.get("power", {}) if isinstance(cfg.get("power"), dict) else {}
        return {
            "enabled": bool(power_cfg.get("enabled")),
            "soft_sleep_enabled": bool(power_cfg.get("soft_sleep_enabled")),
            "night_sleep_enabled": bool(power_cfg.get("night_sleep_enabled")),
            "desired_mode": self._last_desired_mode,
            "last_auto_mode": self._last_auto_mode,
            "last_activity_age_sec": max(0, int(time.monotonic() - self._last_activity_mono)),
            "last_activity_reason": self._last_activity_reason,
            "wake_hold_remaining_sec": max(0, int(self._wake_hold_until_mono - time.monotonic())),
            "radar_presence": self._last_radar_presence,
            "last_error": self._last_error,
        }

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                await self._log_bus.emit("WARN", "SYSTEM", "Power manager tick failed", {"error": str(exc)})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=20)
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        config = await self._config_store.get(include_secrets=True)
        power_cfg = config.get("power", {}) if isinstance(config.get("power"), dict) else {}
        if not bool(power_cfg.get("enabled")):
            self._last_desired_mode = "active"
            return

        esp_status = await self._esp_client.status()
        current = self._current_mode(esp_status)
        self._observe_esp_activity(esp_status, current)
        if not esp_status.get("online"):
            self._last_error = "esp offline"
            return

        desired = self._desired_mode(power_cfg, esp_status)
        self._last_desired_mode = desired

        if desired == current:
            self._last_auto_mode = desired if desired != "active" else "active"
            return

        if desired == "active" and current == "active" and self._last_auto_mode == "active":
            return

        command = {
            "active": "sleep_mode_off",
            "soft_sleep": "soft_sleep_on",
            "night_sleep": "night_sleep_on",
        }.get(desired)
        if not command:
            return

        result = await self._esp_client.send_command(command, {"source": "power_manager"})
        if result.get("ok"):
            self._last_auto_mode = desired
            self._last_error = ""
            await self._log_bus.emit(
                "INFO",
                "SYSTEM",
                "Power mode changed",
                {"mode": desired, "command": command, "reason": self._last_activity_reason},
            )
        else:
            self._last_error = str(result.get("message") or "power command failed")
            await self._log_bus.emit(
                "WARN",
                "SYSTEM",
                "Power mode command failed",
                {"mode": desired, "command": command, "result": result},
            )

    def _observe_esp_activity(self, esp_status: dict[str, Any], current_mode: str) -> None:
        if not esp_status.get("online"):
            return
        previous_mode = self._last_observed_mode
        self._last_observed_mode = current_mode
        if previous_mode in {"soft_sleep", "night_sleep"} and current_mode == "active":
            now = time.monotonic()
            self._wake_hold_until_mono = now + self.WAKE_HOLD_SECONDS
            self._last_activity_mono = now
            self._last_activity_reason = "esp_woke"

        state = str(esp_status.get("state") or "").upper()
        hardware = esp_status.get("hardware") if isinstance(esp_status.get("hardware"), dict) else {}
        mic = str(hardware.get("mic") or "").lower()
        speaker = str(hardware.get("speaker") or "").lower()
        if state and state not in {"IDLE", "OFFLINE"}:
            self.notify_activity(f"esp_state:{state.lower()}")
        elif "streaming" in mic:
            self.notify_activity("esp_mic_stream")
        elif "playing" in speaker:
            self.notify_activity("esp_speaker")
        elif self._radar_sees_presence(esp_status):
            self.notify_activity("radar_presence")

    def _desired_mode(self, power_cfg: dict[str, Any], esp_status: dict[str, Any]) -> str:
        if time.monotonic() < self._wake_hold_until_mono:
            return "active"
        if bool(power_cfg.get("night_sleep_enabled")) and self._in_night_window(power_cfg):
            return "night_sleep"
        if self._radar_sees_presence(esp_status):
            return "active"
        if bool(power_cfg.get("soft_sleep_enabled")):
            idle_minutes = self._positive_float(power_cfg.get("soft_sleep_idle_minutes"), 30.0)
            if time.monotonic() - self._last_activity_mono >= idle_minutes * 60.0:
                return "soft_sleep"
        return "active"

    def _radar_sees_presence(self, esp_status: dict[str, Any]) -> bool:
        radar = esp_status.get("radar") if isinstance(esp_status.get("radar"), dict) else {}
        idle_tracking = esp_status.get("idle_tracking") if isinstance(esp_status.get("idle_tracking"), dict) else {}

        present = self._radar_payload_has_presence(radar)
        if not present:
            present = bool(idle_tracking.get("active")) and self._positive_int(idle_tracking.get("last_seen_ms"), 0) <= 5000

        self._last_radar_presence = present
        return present

    @classmethod
    def _radar_payload_has_presence(cls, radar: dict[str, Any]) -> bool:
        if not radar:
            return False

        state = str(radar.get("state") or "").lower()
        if state in {"offline", "disabled", "unknown", "idle", "waiting", "empty", "sleep"}:
            state_has_presence = False
        else:
            state_has_presence = state in {"tracking", "presence", "present", "target", "targets", "active"}

        fresh = cls._truthy(radar.get("fresh"))
        ready = cls._truthy(radar.get("ready"))
        target_count = cls._positive_int(radar.get("target_count"), 0)
        selected_target = cls._positive_int(radar.get("selected_target"), -1)
        confidence = cls._positive_int(radar.get("confidence"), 0)
        stable_frames = cls._positive_int(radar.get("stable_frames"), 0)
        targets = radar.get("targets")
        target_list_present = isinstance(targets, list) and len(targets) > 0

        has_target = target_count > 0 or selected_target >= 0 or target_list_present or state_has_presence
        if not has_target:
            return False

        # If firmware reports freshness, trust it so stale targets do not keep Alice awake.
        if "fresh" in radar:
            return fresh

        # Older payloads may not have `fresh`; require a stronger signal.
        return ready or confidence > 0 or stable_frames > 0 or target_list_present or state_has_presence

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "active", "ready", "tracking"}
        return bool(value)

    @staticmethod
    def _positive_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _current_mode(esp_status: dict[str, Any]) -> str:
        raw = str(esp_status.get("power_mode") or esp_status.get("sleep_level") or "").lower()
        if raw in {"soft_sleep", "night_sleep", "active"}:
            return raw
        if esp_status.get("sleep_mode"):
            return "night_sleep"
        return "active"

    @staticmethod
    def _positive_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback

    def _in_night_window(self, power_cfg: dict[str, Any]) -> bool:
        start = self._time_to_minutes(power_cfg.get("night_sleep_start"), 23 * 60)
        end = self._time_to_minutes(power_cfg.get("night_sleep_end"), 7 * 60)
        now = datetime.now()
        current = now.hour * 60 + now.minute
        if start == end:
            return False
        if start < end:
            return start <= current < end
        return current >= start or current < end

    @staticmethod
    def _time_to_minutes(value: Any, fallback: int) -> int:
        text = str(value or "").strip()
        try:
            hour_text, minute_text = text.split(":", 1)
            hour = max(0, min(23, int(hour_text)))
            minute = max(0, min(59, int(minute_text)))
            return hour * 60 + minute
        except (TypeError, ValueError):
            return fallback
