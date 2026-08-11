from __future__ import annotations

import unittest

import numpy as np

from app.pipeline.audio_processing import Pcm16LevelMeter, StreamingPcm16Resampler


class StreamingPcm16ResamplerTests(unittest.TestCase):
    @staticmethod
    def _tone(sample_rate: int, frequency: float, duration_sec: float, amplitude: float = 12000.0) -> bytes:
        count = int(sample_rate * duration_sec)
        times = np.arange(count, dtype=np.float64) / sample_rate
        return np.rint(amplitude * np.sin(2.0 * np.pi * frequency * times)).astype("<i2").tobytes()

    def test_chunk_boundaries_do_not_change_output(self) -> None:
        audio = self._tone(16000, 1000.0, 0.25)
        whole = StreamingPcm16Resampler(16000, 24000).process(audio)

        streaming = StreamingPcm16Resampler(16000, 24000)
        parts: list[bytes] = []
        cursor = 0
        for size in (514, 1024, 258, 2048, 4096, 8192):
            if cursor >= len(audio):
                break
            parts.append(streaming.process(audio[cursor : cursor + size]))
            cursor += size
        parts.append(streaming.process(audio[cursor:]))
        self.assertEqual(whole, b"".join(parts))

    def test_16k_to_24k_preserves_duration_and_tone_level(self) -> None:
        audio = self._tone(16000, 1000.0, 0.5)
        output = StreamingPcm16Resampler(16000, 24000).process(audio)
        samples = np.frombuffer(output, dtype="<i2").astype(np.float64)
        self.assertEqual(12000, samples.size)
        # Ignore the short causal filter warm-up when comparing RMS.
        steady = samples[256:]
        self.assertGreater(float(np.sqrt(np.mean(steady * steady))), 7800.0)
        self.assertLess(float(np.sqrt(np.mean(steady * steady))), 9000.0)

    def test_passthrough_is_exact(self) -> None:
        audio = self._tone(16000, 600.0, 0.1)
        resampler = StreamingPcm16Resampler(16000, 16000)
        self.assertEqual(audio, resampler.process(audio))
        self.assertEqual("passthrough", resampler.method)


class Pcm16LevelMeterTests(unittest.TestCase):
    def test_reports_rms_peak_and_clipping(self) -> None:
        samples = np.array([0, 1000, -1000, 32767, -32768], dtype="<i2")
        meter = Pcm16LevelMeter()
        meter.add(samples.tobytes())
        summary = meter.summary()
        self.assertEqual(5, summary["samples"])
        self.assertEqual(32768, summary["peak"])
        self.assertEqual(40.0, summary["clip_pct"])
        self.assertIsNotNone(summary["rms_dbfs"])


if __name__ == "__main__":
    unittest.main()
