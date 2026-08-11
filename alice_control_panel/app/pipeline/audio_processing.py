from __future__ import annotations

import math
from typing import Any

import numpy as np


class StreamingPcm16Resampler:
    """Stateful mono PCM16 resampler using a causal polyphase FIR filter."""

    def __init__(self, source_rate: int, target_rate: int, taps_per_phase: int = 16) -> None:
        self.source_rate = max(1, int(source_rate))
        self.target_rate = max(1, int(target_rate))
        divisor = math.gcd(self.source_rate, self.target_rate)
        self.up = self.target_rate // divisor
        self.down = self.source_rate // divisor
        self._passthrough = self.up == self.down

        if self._passthrough:
            self._filter = np.ones(1, dtype=np.float64)
        else:
            half_len = max(8, int(taps_per_phase)) * max(self.up, self.down)
            offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
            cutoff = 1.0 / max(self.up, self.down)
            # Kaiser-windowed sinc. Multiplication by ``up`` preserves unity gain
            # after zero insertion and downsampling.
            window = np.kaiser(offsets.size, 8.6)
            fir = cutoff * np.sinc(cutoff * offsets) * window
            fir /= np.sum(fir)
            self._filter = fir * self.up

        self._state = np.zeros(max(0, self._filter.size - 1), dtype=np.float64)
        self._upsampled_offset = 0
        self.input_samples = 0
        self.output_samples = 0

    @property
    def method(self) -> str:
        return "passthrough" if self._passthrough else "polyphase_fir_kaiser"

    @property
    def delay_ms(self) -> float:
        if self._passthrough:
            return 0.0
        group_delay_upsampled = (self._filter.size - 1) / 2.0
        return (group_delay_upsampled / (self.source_rate * self.up)) * 1000.0

    def reset(self) -> None:
        self._state.fill(0.0)
        self._upsampled_offset = 0
        self.input_samples = 0
        self.output_samples = 0

    def process(self, pcm16le: bytes) -> bytes:
        if not pcm16le:
            return b""
        if len(pcm16le) & 1:
            pcm16le = pcm16le[:-1]
        if not pcm16le:
            return b""

        samples = np.frombuffer(pcm16le, dtype="<i2")
        self.input_samples += int(samples.size)
        if self._passthrough:
            self.output_samples += int(samples.size)
            return bytes(pcm16le)

        upsampled = np.zeros(samples.size * self.up, dtype=np.float64)
        upsampled[:: self.up] = samples.astype(np.float64)
        extended = np.concatenate((self._state, upsampled))
        filtered = np.convolve(extended, self._filter, mode="valid")
        self._state = extended[-(self._filter.size - 1) :]

        first = (-self._upsampled_offset) % self.down
        output = filtered[first:: self.down]
        self._upsampled_offset += int(upsampled.size)
        self.output_samples += int(output.size)
        return np.clip(np.rint(output), -32768, 32767).astype("<i2").tobytes()


class Pcm16LevelMeter:
    """Collect calibration-friendly level statistics without changing audio."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.sum_squares = 0.0
        self.peak = 0
        self.clipped = 0

    def add(self, pcm16le: bytes) -> None:
        if len(pcm16le) < 2:
            return
        usable = pcm16le[: len(pcm16le) & ~1]
        samples = np.frombuffer(usable, dtype="<i2").astype(np.int32)
        if samples.size == 0:
            return
        absolute = np.abs(samples)
        self.samples += int(samples.size)
        self.sum_squares += float(np.dot(samples.astype(np.float64), samples.astype(np.float64)))
        self.peak = max(self.peak, int(np.max(absolute)))
        self.clipped += int(np.count_nonzero(absolute >= 32760))

    @staticmethod
    def _dbfs(value: float) -> float | None:
        if value <= 0.0:
            return None
        return round(20.0 * math.log10(value / 32768.0), 2)

    def summary(self) -> dict[str, Any]:
        rms = math.sqrt(self.sum_squares / self.samples) if self.samples else 0.0
        return {
            "samples": self.samples,
            "rms": round(rms, 2),
            "rms_dbfs": self._dbfs(rms),
            "peak": self.peak,
            "peak_dbfs": self._dbfs(float(self.peak)),
            "clip_pct": round((self.clipped * 100.0 / self.samples), 4) if self.samples else 0.0,
        }
