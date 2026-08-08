from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import wave


ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

from app.pipeline.tts.capture import CapturingPcmOutput, TtsCaptureStore  # noqa: E402


class FakePcmOutput:
    pace_pcm = True

    def __init__(self) -> None:
        self.started: tuple[int, int] | None = None
        self.written = bytearray()
        self.completed = False
        self.errors: list[tuple[str, int]] = []
        self.trace = None

    def set_trace(self, trace: object) -> None:
        self.trace = trace

    async def start(self, sample_rate: int, channels: int = 1) -> None:
        self.started = (sample_rate, channels)

    async def write(self, pcm: bytes) -> None:
        self.written.extend(pcm)

    async def done(self) -> None:
        self.completed = True

    async def error(self, message: str, status: int = 500) -> None:
        self.errors.append((message, status))


class TtsCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_pcm_is_saved_as_downloadable_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TtsCaptureStore(pathlib.Path(temp_dir))
            target = FakePcmOutput()
            output = CapturingPcmOutput(target, store, "google_ai", "direct_esp")
            pcm = b"\x01\x00" * 240

            await output.start(24000, 1)
            await output.write(pcm)
            await output.done()

            self.assertEqual(pcm, bytes(target.written))
            self.assertTrue(target.completed)
            capture = store.status()
            self.assertTrue(capture["available"])
            self.assertEqual("google_ai", capture["provider"])
            self.assertEqual("direct_esp", capture["transport"])
            self.assertEqual(len(pcm), capture["pcm_bytes"])
            with wave.open(str(store.latest_path), "rb") as wav_file:
                self.assertEqual(1, wav_file.getnchannels())
                self.assertEqual(2, wav_file.getsampwidth())
                self.assertEqual(24000, wav_file.getframerate())
                self.assertEqual(pcm, wav_file.readframes(wav_file.getnframes()))

    async def test_failed_capture_preserves_previous_successful_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TtsCaptureStore(pathlib.Path(temp_dir))
            first = CapturingPcmOutput(FakePcmOutput(), store, "openai", "relay_ws")
            await first.start(24000, 1)
            await first.write(b"\x02\x00" * 160)
            await first.done()
            previous = store.latest_path.read_bytes()

            failed_target = FakePcmOutput()
            failed = CapturingPcmOutput(failed_target, store, "openai", "relay_ws")
            await failed.start(24000, 1)
            await failed.write(b"\x03\x00" * 160)
            await failed.error("cancelled", 499)

            self.assertEqual(previous, store.latest_path.read_bytes())
            self.assertEqual([("cancelled", 499)], failed_target.errors)

    async def test_size_limit_does_not_interrupt_audio_or_publish_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TtsCaptureStore(pathlib.Path(temp_dir), max_pcm_bytes=1024)
            target = FakePcmOutput()
            output = CapturingPcmOutput(target, store, "cartesia", "direct_esp")
            pcm = b"\x04\x00" * 700

            await output.start(44100, 1)
            await output.write(pcm)
            await output.done()

            self.assertEqual(pcm, bytes(target.written))
            self.assertTrue(target.completed)
            self.assertFalse(store.status()["available"])
            self.assertEqual("capture_too_large", store.status()["last_error"])
            self.assertEqual([], list(pathlib.Path(temp_dir).glob("*.tmp.*")))
