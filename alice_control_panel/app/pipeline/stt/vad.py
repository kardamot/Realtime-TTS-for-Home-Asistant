from __future__ import annotations

from pathlib import Path


SILERO_VAD_FRAME_SAMPLES_16K = 512
SILERO_VAD_FRAME_SAMPLES_8K = 256


class SileroVadRuntime:
    """Small wrapper around faster-whisper's bundled Silero VAD ONNX model."""

    def __init__(self, sample_rate: int) -> None:
        if sample_rate == 16000:
            self.frame_samples = SILERO_VAD_FRAME_SAMPLES_16K
        elif sample_rate == 8000:
            self.frame_samples = SILERO_VAD_FRAME_SAMPLES_8K
        else:
            raise ValueError(f"Silero VAD supports only 8000/16000 Hz. sample_rate={sample_rate}")

        import numpy as np  # type: ignore
        import onnxruntime  # type: ignore
        from faster_whisper.utils import get_assets_path

        model_path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self._np = np
        self.sample_rate = sample_rate
        self.frame_bytes = self.frame_samples * 2
        self.frame_ms = int((self.frame_samples * 1000) / sample_rate)
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._context_samples = 64 if sample_rate == 16000 else 32
        self._pcm_buffer = bytearray()
        self.reset_state()

    def reset_state(self) -> None:
        np = self._np
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)
        self._pcm_buffer.clear()

    def _process_frame(self, frame: bytes) -> float:
        np = self._np
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        framed = samples.reshape(1, self.frame_samples)
        model_input = np.concatenate([self._context, framed], axis=1).astype(np.float32, copy=False)
        output, self._h, self._c = self._session.run(
            None,
            {"input": model_input, "h": self._h, "c": self._c},
        )
        self._context = framed[:, -self._context_samples :]
        return float(np.asarray(output).reshape(-1)[0])

    def push_pcm16le(self, chunk: bytes) -> list[float]:
        self._pcm_buffer.extend(chunk)
        probabilities: list[float] = []
        while len(self._pcm_buffer) >= self.frame_bytes:
            frame = bytes(self._pcm_buffer[: self.frame_bytes])
            del self._pcm_buffer[: self.frame_bytes]
            probabilities.append(self._process_frame(frame))
        return probabilities
