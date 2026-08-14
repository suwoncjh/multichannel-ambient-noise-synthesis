from __future__ import annotations

import numpy as np

from .scm import estimate_scm


def normalized_scm_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return normalized Frobenius error ``||R_ref-R_est||_F / ||R_ref||_F``."""
    ref = np.asarray(reference, dtype=np.complex128)
    est = np.asarray(estimate, dtype=np.complex128)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate SCMs must have identical shapes")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(est)):
        raise ValueError("SCMs must contain only finite values")
    denom = float(np.linalg.norm(ref))
    error = float(np.linalg.norm(ref - est))
    if denom == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / denom


def channel_psd(spec: np.ndarray) -> np.ndarray:
    """Mean STFT power per frequency/channel, shape ``[frequency, channel]``."""
    x = np.asarray(spec)
    if x.ndim != 3:
        raise ValueError("spec must have shape [frequency, frame, channel]")
    return np.mean(np.abs(x) ** 2, axis=1)


def complex_coherence(spec: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Complex coherence matrix per frequency from STFT data."""
    if eps < 0:
        raise ValueError("eps must be non-negative")
    scm = estimate_scm(spec, eps=0.0)
    diag = np.maximum(np.real(np.diagonal(scm, axis1=-2, axis2=-1)), 0.0)
    denom = np.sqrt(diag[:, :, None] * diag[:, None, :])
    out = np.zeros_like(scm)
    np.divide(scm, denom, out=out, where=denom > eps)
    return out


def ipd(spec: np.ndarray, i: int, j: int) -> np.ndarray:
    """Inter-channel phase difference ``angle(X_i * conj(X_j))`` as ``[F,T]``."""
    x = np.asarray(spec)
    if x.ndim != 3:
        raise ValueError("spec must have shape [frequency, frame, channel]")
    channels = x.shape[-1]
    if not (0 <= i < channels and 0 <= j < channels):
        raise IndexError("channel index out of range")
    return np.angle(x[:, :, i] * x[:, :, j].conj())


def scm_eigenspectrum(scm: np.ndarray) -> np.ndarray:
    """Return ascending Hermitian eigenvalues for each frequency SCM."""
    x = np.asarray(scm, dtype=np.complex128)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2]:
        raise ValueError("scm must have shape [frequency, channel, channel]")
    hermitian = 0.5 * (x + np.swapaxes(x.conj(), -1, -2))
    return np.linalg.eigvalsh(hermitian)
