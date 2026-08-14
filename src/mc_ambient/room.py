from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .io import as_samples_channels


def _as_fir_bank(firs: np.ndarray, name: str) -> np.ndarray:
    h = np.asarray(firs, dtype=np.float64)
    if h.ndim == 1:
        h = h[:, None]
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] == 0:
        raise ValueError(f"{name} must have shape [taps, channels]")
    if not np.all(np.isfinite(h)):
        raise ValueError(f"{name} must contain only finite values")
    return h


def apply_multichannel_rir(source: np.ndarray, rirs: np.ndarray) -> np.ndarray:
    """Convolve one mono source with a bank of channel-specific RIRs."""
    x = np.asarray(source, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1 or x.size == 0:
        raise ValueError("source must be mono with shape [samples] or [samples, 1]")
    if not np.all(np.isfinite(x)):
        raise ValueError("source must contain only finite values")
    h = _as_fir_bank(rirs, "rirs")
    return np.stack([fftconvolve(x, h[:, ch], mode="full") for ch in range(h.shape[1])], axis=-1)


def apply_channel_firs(audio: np.ndarray, firs: np.ndarray) -> np.ndarray:
    """Apply one FIR per microphone channel, e.g. measured port/device responses."""
    x = as_samples_channels(audio)
    h = _as_fir_bank(firs, "firs")
    if h.shape[1] != x.shape[1]:
        raise ValueError("firs channel count must match audio channel count")
    return np.stack([fftconvolve(x[:, ch], h[:, ch], mode="full") for ch in range(x.shape[1])], axis=-1)


def mix_components(*components: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Zero-pad, sum compatible multichannel components, and peak-limit by scaling."""
    if not components:
        raise ValueError("at least one component is required")
    if peak <= 0:
        raise ValueError("peak must be positive")
    arrays = [as_samples_channels(component) for component in components]
    channels = arrays[0].shape[1]
    if any(x.shape[1] != channels for x in arrays[1:]):
        raise ValueError("all components must have the same channel count")
    length = max(x.shape[0] for x in arrays)
    mixed = np.zeros((length, channels), dtype=np.float64)
    for x in arrays:
        mixed[: x.shape[0]] += x
    maximum = float(np.max(np.abs(mixed)))
    if maximum > peak:
        mixed *= peak / maximum
    return mixed
