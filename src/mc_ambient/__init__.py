"""Multichannel ambient-noise synthesis from measured spatial statistics."""

from .io import load_audio, save_audio
from .metrics import complex_coherence, ipd, normalized_scm_error
from .room import apply_channel_firs, apply_multichannel_rir, mix_components
from .scm import estimate_scm, estimate_time_varying_scm, factor_scm, project_psd
from .stft import istft_mc, stft_mc
from .synthesis import synthesize_static, synthesize_time_varying

__version__ = "0.1.0"

__all__ = [
    "apply_channel_firs",
    "apply_multichannel_rir",
    "complex_coherence",
    "estimate_scm",
    "estimate_time_varying_scm",
    "factor_scm",
    "ipd",
    "istft_mc",
    "load_audio",
    "mix_components",
    "normalized_scm_error",
    "project_psd",
    "save_audio",
    "stft_mc",
    "synthesize_static",
    "synthesize_time_varying",
]
