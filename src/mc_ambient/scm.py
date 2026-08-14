from __future__ import annotations

import numpy as np


def _validate_square_matrices(matrices: np.ndarray) -> np.ndarray:
    x = np.asarray(matrices, dtype=np.complex128)
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("SCM must end with square [channel, channel] dimensions")
    if x.shape[-1] == 0:
        raise ValueError("SCM must contain at least one channel")
    if not np.all(np.isfinite(x)):
        raise ValueError("SCM must contain only finite values")
    return x


def project_psd(scm: np.ndarray, eig_floor: float = 1e-8) -> np.ndarray:
    """Hermitian-symmetrize and floor eigenvalues of one or more SCMs."""
    if eig_floor < 0:
        raise ValueError("eig_floor must be non-negative")
    x = _validate_square_matrices(scm)
    hermitian = 0.5 * (x + np.swapaxes(x.conj(), -1, -2))
    values, vectors = np.linalg.eigh(hermitian)
    values = np.maximum(values, eig_floor)
    projected = np.einsum("...ik,...k,...jk->...ij", vectors, values, vectors.conj())
    return 0.5 * (projected + np.swapaxes(projected.conj(), -1, -2))


def estimate_scm(spec: np.ndarray, eps: float = 1e-8, normalize: bool = False) -> np.ndarray:
    """Estimate frequency-wise SCMs from STFT data ``[frequency, frame, channel]``."""
    x = np.asarray(spec, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError("spec must have shape [frequency, frame, channel]")
    if x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError("spec must contain at least one frame and channel")
    if not np.all(np.isfinite(x)):
        raise ValueError("spec must contain only finite values")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    scm = np.einsum("ftm,ftn->fmn", x, x.conj()) / x.shape[1]
    scm = 0.5 * (scm + np.swapaxes(scm.conj(), -1, -2))
    if normalize:
        power = np.real(np.trace(scm, axis1=-2, axis2=-1)) / x.shape[2]
        safe = np.maximum(power, np.finfo(np.float64).tiny)
        scm = scm / safe[:, None, None]
    return project_psd(scm, eig_floor=eps)


def factor_scm(scm: np.ndarray, eig_floor: float = 1e-8) -> np.ndarray:
    """Return factors ``L`` such that ``L @ L.H`` matches the PSD-projected SCM."""
    projected = project_psd(scm, eig_floor=eig_floor)
    m = projected.shape[-1]
    flat = projected.reshape((-1, m, m))
    factors = np.empty_like(flat)
    for i, matrix in enumerate(flat):
        try:
            factors[i] = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(matrix)
            values = np.maximum(values, eig_floor)
            factors[i] = vectors @ np.diag(np.sqrt(values))
    return factors.reshape(projected.shape)


def estimate_time_varying_scm(
    spec: np.ndarray,
    frames_per_window: int,
    hop_frames: int,
    eig_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate overlapping SCM windows from ``[frequency, frame, channel]`` STFT data."""
    x = np.asarray(spec, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError("spec must have shape [frequency, frame, channel]")
    if frames_per_window <= 0 or hop_frames <= 0:
        raise ValueError("frames_per_window and hop_frames must be positive")
    if frames_per_window > x.shape[1]:
        raise ValueError("frames_per_window exceeds available STFT frames")
    starts = np.arange(0, x.shape[1] - frames_per_window + 1, hop_frames, dtype=int)
    if starts.size == 0:
        raise ValueError("no valid SCM windows")
    windows = np.stack(
        [estimate_scm(x[:, start : start + frames_per_window], eps=eig_floor) for start in starts],
        axis=0,
    )
    centers = starts.astype(np.float64) + (frames_per_window - 1) / 2.0
    return windows, centers
