from __future__ import annotations

import numpy as np

from .scm import estimate_scm, factor_scm
from .stft import stft_mc


def _validate_rng(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")
    return rng


def _resolve_stft_geometry(n_freq: int, n_fft: int | None, hop_length: int | None) -> tuple[int, int]:
    if n_fft is None:
        if n_freq < 2:
            raise ValueError("cannot infer n_fft from fewer than two one-sided frequency bins")
        n_fft = 2 * (n_freq - 1)
    if n_fft <= 0 or n_fft // 2 + 1 != n_freq:
        raise ValueError("n_fft is incompatible with the SCM frequency-bin count")
    if hop_length is None:
        hop_length = max(1, n_fft // 4)
    if hop_length <= 0 or hop_length > n_fft:
        raise ValueError("hop_length must satisfy 0 < hop_length <= n_fft")
    return int(n_fft), int(hop_length)


def _inverse_sqrt_scm(scm: np.ndarray, relative_floor: float = 1e-8) -> np.ndarray:
    x = np.asarray(scm, dtype=np.complex128)
    values, vectors = np.linalg.eigh(0.5 * (x + np.swapaxes(x.conj(), -1, -2)))
    max_values = np.maximum(np.max(values, axis=-1, keepdims=True), np.finfo(np.float64).tiny)
    floor = relative_floor * max_values
    values = np.maximum(values, floor)
    return np.einsum("...ik,...k,...jk->...ij", vectors, 1.0 / np.sqrt(values), vectors.conj())


def _consistent_whitened_seed(
    n_freq: int,
    n_frames: int,
    n_channels: int,
    rng: np.random.Generator,
    n_fft: int | None,
    hop_length: int | None,
) -> np.ndarray:
    """Create a time-domain-consistent STFT seed with empirical identity SCM."""
    n_fft, hop_length = _resolve_stft_geometry(n_freq, n_fft, hop_length)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    seed_length = max(n_fft, (n_frames - 1) * hop_length)
    white_time = rng.normal(size=(seed_length, n_channels))
    seed = stft_mc(white_time, sample_rate=1, n_fft=n_fft, hop_length=hop_length)
    if seed.shape[1] < n_frames:
        extra = (n_frames - seed.shape[1]) * hop_length
        white_time = np.pad(white_time, ((0, extra + n_fft), (0, 0)))
        seed = stft_mc(white_time, sample_rate=1, n_fft=n_fft, hop_length=hop_length)
    seed = seed[:, :n_frames, :]
    seed_scm = estimate_scm(seed, eps=0.0)
    whitening = _inverse_sqrt_scm(seed_scm)
    return np.einsum("fij,ftj->fti", whitening, seed)


def synthesize_static(
    scm: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    match_power: bool = True,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> np.ndarray:
    """Generate a time-domain-consistent STFT realization matching ``scm``."""
    _validate_rng(rng)
    target = np.asarray(scm, dtype=np.complex128)
    if target.ndim != 3 or target.shape[-1] != target.shape[-2]:
        raise ValueError("scm must have shape [frequency, channel, channel]")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not np.all(np.isfinite(target)):
        raise ValueError("scm must contain only finite values")
    if not match_power:
        power = np.real(np.trace(target, axis1=-2, axis2=-1)) / target.shape[-1]
        safe = np.maximum(power, np.finfo(np.float64).tiny)
        target = target / safe[:, None, None]
    factors = factor_scm(target)
    seed = _consistent_whitened_seed(
        target.shape[0], n_frames, target.shape[-1], rng,
        n_fft=n_fft, hop_length=hop_length,
    )
    return np.einsum("fij,ftj->fti", factors, seed)


def synthesize_time_varying(
    scm_windows: np.ndarray,
    centers: np.ndarray,
    n_frames: int,
    rng: np.random.Generator,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> np.ndarray:
    """Synthesize STFT noise while smoothly cross-fading adjacent SCM factors."""
    _validate_rng(rng)
    target = np.asarray(scm_windows, dtype=np.complex128)
    c = np.asarray(centers, dtype=np.float64)
    if target.ndim != 4 or target.shape[-1] != target.shape[-2]:
        raise ValueError("scm_windows must have shape [window, frequency, channel, channel]")
    if c.ndim != 1 or c.shape[0] != target.shape[0]:
        raise ValueError("centers must contain one frame center per SCM window")
    if c.size == 0 or np.any(np.diff(c) <= 0):
        raise ValueError("centers must be non-empty and strictly increasing")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(c)):
        raise ValueError("SCM windows and centers must be finite")
    factors = factor_scm(target)
    n_windows, n_freq, n_channels, _ = factors.shape
    seed = _consistent_whitened_seed(
        n_freq, n_frames, n_channels, rng,
        n_fft=n_fft, hop_length=hop_length,
    )
    output = np.empty_like(seed)
    for frame in range(n_frames):
        position = float(frame)
        if n_windows == 1 or position <= c[0]:
            factor = factors[0]
        elif position >= c[-1]:
            factor = factors[-1]
        else:
            right = int(np.searchsorted(c, position, side="right"))
            left = right - 1
            alpha = (position - c[left]) / (c[right] - c[left])
            factor = (1.0 - alpha) * factors[left] + alpha * factors[right]
        output[:, frame, :] = np.einsum("fij,fj->fi", factor, seed[:, frame, :])
    return output
