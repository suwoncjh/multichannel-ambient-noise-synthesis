from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def as_samples_channels(audio: np.ndarray) -> np.ndarray:
    """Return floating-point audio as ``[samples, channels]``."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("audio must have shape [samples] or [samples, channels]")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("audio must contain at least one sample and one channel")
    if not np.all(np.isfinite(x)):
        raise ValueError("audio must contain only finite values")
    return x


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV/audio file as float64 ``[samples, channels]``."""
    audio, sample_rate = sf.read(Path(path), always_2d=True, dtype="float64")
    return as_samples_channels(audio), int(sample_rate)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save finite ``[samples, channels]`` audio using floating-point WAV."""
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    x = as_samples_channels(audio)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x, int(sample_rate), subtype="FLOAT")
