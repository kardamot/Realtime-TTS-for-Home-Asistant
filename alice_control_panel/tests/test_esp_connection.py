from __future__ import annotations

import io
import pathlib
import sys
import time
import types
import unittest
import wave


ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

previous_aiohttp = sys.modules.get("aiohttp")
sys.modules["aiohttp"] = types.ModuleType("aiohttp")
from app.esp.esp_client import (  # noqa: E402
    EspClient,
    WS_FLAP_LOG_INTERVAL_SECONDS,
    WS_STABLE_CONNECTION_SECONDS,
    validate_barge_capture_wav,
)
if previous_aiohttp is None:
    sys.modules.pop("aiohttp", None)
else:
    sys.modules["aiohttp"] = previous_aiohttp

from app.core.log_bus import LogBus  # noqa: E402
from app.core.ws_hub import WsHub  # noqa: E402


class EspConnectionPolicyTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self) -> EspClient:
        return EspClient(object(), LogBus(maxlen=50), WsHub())  # type: ignore[arg-type]

    def test_short_successful_connections_do_not_reset_retry_budget(self) -> None:
        client = self.make_client()
        config = {"max_auto_reconnects": 5}

        for expected in range(1, 4):
            client._begin_ws_connection("ws://alice/ws")
            client._finish_ws_connection()
            client._status["reconnects"] = client._record_reconnect_failure(config)
            client._set_ws_state(False, "ws://alice/ws", "closed")
            self.assertEqual(client._status["reconnects"], expected)

    async def test_activity_after_stability_window_resets_retry_budget(self) -> None:
        client = self.make_client()
        client._status["reconnects"] = 4
        client._ws_disconnect_streak = 4
        client._unreported_ws_disconnects = 2
        client._begin_ws_connection("ws://alice/ws")
        client._ws_connected_monotonic = time.monotonic() - WS_STABLE_CONNECTION_SECONDS - 1

        await client._note_ws_activity("ws://alice/ws")

        self.assertEqual(client._status["reconnects"], 0)
        self.assertEqual(client._status["ws_disconnect_streak"], 0)
        entries = await client._log_bus.list(limit=20)
        self.assertEqual(entries[-1]["message"], "ESP WebSocket connection recovered")

    def test_http_status_normalization_preserves_websocket_runtime_state(self) -> None:
        client = self.make_client()
        client._status.update(
            {
                "ws_connected": True,
                "ws_url": "ws://alice/ws",
                "reconnects": 3,
                "last_ws_message_at": 123.0,
                "ws_disconnect_streak": 2,
            }
        )

        normalized = client._normalize_status({"state": "idle"}, "http://alice")

        self.assertTrue(normalized["ws_connected"])
        self.assertEqual(normalized["reconnects"], 3)
        self.assertEqual(normalized["last_ws_message_at"], 123.0)
        self.assertEqual(normalized["ws_disconnect_streak"], 2)

    async def test_repeated_disconnects_are_summarized(self) -> None:
        client = self.make_client()
        client._status["max_auto_reconnects"] = 40

        await client._record_ws_disconnect("ws://alice/ws", "no pong", 30.0)
        await client._record_ws_disconnect("ws://alice/ws", "no pong", 30.0)
        warnings = await client._log_bus.list(level="WARN", limit=20)
        self.assertEqual(len(warnings), 1)

        client._last_ws_log_at = time.time() - WS_FLAP_LOG_INTERVAL_SECONDS - 1
        await client._record_ws_disconnect("ws://alice/ws", "no pong", 30.0)
        warnings = await client._log_bus.list(level="WARN", limit=20)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(warnings[-1]["details"]["disconnects_since_last_log"], 2)
        self.assertEqual(warnings[-1]["details"]["disconnect_streak"], 3)

    def test_four_channel_barge_capture_wav_is_validated(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(4)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00" * (20 * 4 * 2))

        metadata = validate_barge_capture_wav(buffer.getvalue())

        self.assertEqual(metadata["channels"], 4)
        self.assertEqual(metadata["sample_rate"], 16000)
        self.assertEqual(metadata["frames"], 20)

    def test_incomplete_barge_capture_wav_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_barge_capture_wav(
                b"RIFF" + (76).to_bytes(4, "little") + b"WAVEfmt " +
                (16).to_bytes(4, "little") + (1).to_bytes(2, "little") +
                (4).to_bytes(2, "little") + (16000).to_bytes(4, "little") +
                (128000).to_bytes(4, "little") + (8).to_bytes(2, "little") +
                (16).to_bytes(2, "little") + b"data" + (40).to_bytes(4, "little")
            )


if __name__ == "__main__":
    unittest.main()
