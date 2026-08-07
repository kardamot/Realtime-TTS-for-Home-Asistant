from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time
import unittest


ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

from app.core.log_bus import LogBus  # noqa: E402


class LogBusPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_warn_and_error_survive_restart_but_info_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp)
            bus = LogBus(maxlen=20, critical_dir=archive)
            await bus.emit("INFO", "SYSTEM", "ordinary event")
            await bus.emit("WARN", "ESP", "connection unstable", {"count": 3})
            await bus.emit("ERROR", "TTS", "provider failed")

            restarted = LogBus(maxlen=20, critical_dir=archive)
            entries = await restarted.list(limit=20)
            messages = [entry["message"] for entry in entries]

            self.assertEqual(messages, ["connection unstable", "provider failed"])
            self.assertEqual(restarted.archive_status()["file_count"], 1)
            downloaded = await restarted.download_text()
            self.assertEqual(downloaded.count("connection unstable"), 1)
            self.assertEqual(downloaded.count("provider failed"), 1)

    async def test_archive_is_pruned_after_retention_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp)
            bus = LogBus(maxlen=20, critical_dir=archive, critical_retention_days=31)
            await bus.emit("ERROR", "SYSTEM", "old failure")
            old = time.time() - 32 * 24 * 60 * 60
            for path in archive.glob("critical-*.jsonl"):
                os.utime(path, (old, old))

            restarted = LogBus(maxlen=20, critical_dir=archive, critical_retention_days=31)

            self.assertEqual(await restarted.list(limit=20), [])
            self.assertEqual(restarted.archive_status()["file_count"], 0)

    async def test_archive_total_size_has_a_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp)
            bus = LogBus(
                maxlen=20,
                critical_dir=archive,
                critical_max_file_bytes=1024,
                critical_max_total_bytes=1500,
            )
            for index in range(6):
                await bus.emit("ERROR", "SYSTEM", f"failure {index}", {"blob": "x" * 700})

            self.assertLessEqual(bus.archive_status()["total_bytes"], 1500)

    async def test_clear_removes_memory_and_persistent_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp)
            bus = LogBus(maxlen=20, critical_dir=archive)
            await bus.emit("WARN", "ESP", "temporary failure")

            await bus.clear()

            self.assertEqual(await bus.list(limit=20), [])
            self.assertEqual(list(archive.glob("critical-*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
