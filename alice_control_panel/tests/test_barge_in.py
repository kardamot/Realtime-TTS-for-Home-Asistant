from __future__ import annotations

import unittest

from app.core.barge_in import realtime_turn_detection, sync_barge_in_to_esp


class _FakeEspClient:
    def __init__(self, result: dict | None = None, error: Exception | None = None) -> None:
        self.result = result or {"ok": True}
        self.error = error
        self.patches: list[dict] = []

    async def update_config(self, patch: dict) -> dict:
        self.patches.append(patch)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeLogBus:
    def __init__(self) -> None:
        self.entries: list[tuple] = []

    async def emit(self, *entry) -> None:
        self.entries.append(entry)


class BargeInTurnDetectionTests(unittest.TestCase):
    def test_server_vad_interrupt_follows_switch(self) -> None:
        enabled = realtime_turn_detection({"turn_detection": "server_vad"}, barge_in_enabled=True)
        disabled = realtime_turn_detection({"turn_detection": "server_vad"}, barge_in_enabled=False)

        self.assertTrue(enabled["interrupt_response"])
        self.assertFalse(disabled["interrupt_response"])

    def test_semantic_vad_interrupt_follows_switch(self) -> None:
        disabled = realtime_turn_detection(
            {"turn_detection": "semantic_vad", "semantic_eagerness": "medium"},
            barge_in_enabled=False,
        )

        self.assertEqual(disabled["type"], "semantic_vad")
        self.assertEqual(disabled["eagerness"], "medium")
        self.assertFalse(disabled["interrupt_response"])


class BargeInEspSyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_setting_is_sent_to_esp(self) -> None:
        esp_client = _FakeEspClient()
        log_bus = _FakeLogBus()

        result = await sync_barge_in_to_esp(esp_client, log_bus, {"pipeline": {"barge_in_enabled": False}})

        self.assertTrue(result["ok"])
        self.assertEqual(esp_client.patches, [{"barge_in_enabled": False}])
        self.assertEqual(log_bus.entries[0][0], "INFO")

    async def test_sync_failure_does_not_fail_config_save(self) -> None:
        esp_client = _FakeEspClient(error=RuntimeError("offline"))
        log_bus = _FakeLogBus()

        result = await sync_barge_in_to_esp(esp_client, log_bus, {"pipeline": {"barge_in_enabled": True}})

        self.assertFalse(result["ok"])
        self.assertEqual(result["message"], "offline")
        self.assertEqual(log_bus.entries[0][0], "WARN")


if __name__ == "__main__":
    unittest.main()
