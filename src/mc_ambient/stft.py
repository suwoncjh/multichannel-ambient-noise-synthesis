from __future__ import annotations

import numpy as np
from scipy import signal

from .io import as_samples_channels


def _validate_params(n_fft: int, hop_length: int) -> None:
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if hop_length <= 0 or hop_length > n_fft:
        raise ValueError("hop_length must satisfy 0 < hop_length <= n_fft")


def stft_mc(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Compute a Hann-window STFT with layout ``[frequency, frame, channel]``."""
    _validate_params(n_fft, hop_length)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = as_samples_channels(audio)
    if x.shape[0] < n_fft:
        x = np.pad(x, ((0, n_fft - x.shape[0]), (0, 0)))
    noverlap = n_fft - hop_length
    channels = []
    for ch in range(x.shape[1]):
        _, _, z = signal.stft(
            x[:, ch], fs=sample_rate, window="hann", nperseg=n_fft,
            noverlap=noverlap, nfft=n_fft, boundary="zeros", padded=True,
            return_onesided=True,
        )
        channels.append(z)
    return np.stack(channels, axis=-1)


def istft_mc(
    spec: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    length: int | None = None,
) -> np.ndarray:
    """Invert an STFT with layout ``[frequency, frame, channel]``."""
    _validate_params(n_fft, hop_length)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    z = np.asarray(spec)
    if z.ndim != 3:
        raise ValueError("spec must have shape [frequency, frame, channel]")
    if not np.all(np.isfinite(z)):
        raise ValueError("spec must contain only finite values")
    expected_bins = n_fft // 2 + 1
    if z.shape[0] != expected_bins:
        raise ValueError(f"expected {expected_bins} frequency bins, got {z.shape[0]}")
    noverlap = n_fft - hop_length
    channels = []
    for ch in range(z.shape[2]):
        _, x = signal.istft(
            z[:, :, ch], fs=sample_rate, window="hann", nperseg=n_fft,
            noverlap=noverlap, nfft=n_fft, input_onesided=True, boundary=True,
        )
        channels.append(x)
    audio = np.stack(channels, axis=-1)
    if length is not None:
        if length < 0:
            raise ValueError("length must be non-negative")
        if audio.shape[0] < length:
            audio = np.pad(audio, ((0, length - audio.shape[0]), (0, 0)))
        else:
            audio = audio[:length]
    return audio
