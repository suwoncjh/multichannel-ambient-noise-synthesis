from pathlib import Path

import numpy as np
import soundfile as sf

from mc_ambient.io import load_audio, save_audio
from mc_ambient.stft import istft_mc, stft_mc


def test_stft_round_trip_multichannel():
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=(8192, 3)).astype(np.float64)
    X = stft_mc(x, 16000, n_fft=512, hop_length=128)
    assert X.ndim == 3
    assert X.shape[0] == 257
    assert X.shape[2] == 3
    y = istft_mc(X, 16000, n_fft=512, hop_length=128, length=len(x))
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-8, rtol=1e-6)


def test_load_audio_normalizes_mono_to_samples_channels(tmp_path: Path):
    x = np.linspace(-0.25, 0.25, 800, dtype=np.float64)
    p = tmp_path / "mono.wav"
    sf.write(p, x, 16000, subtype="FLOAT")
    y, sr = load_audio(p)
    assert sr == 16000
    assert y.shape == (800, 1)
    np.testing.assert_allclose(y[:, 0], x, atol=1e-7)


def test_save_audio_rejects_non_finite(tmp_path: Path):
    x = np.zeros((128, 2), dtype=np.float64)
    x[3, 0] = np.nan
    try:
        save_audio(tmp_path / "bad.wav", x, 16000)
    except ValueError as exc:
        assert "finite" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_stft_accepts_clip_shorter_than_fft_size():
    x = np.arange(100, dtype=np.float64)[:, None] / 1000.0
    X = stft_mc(x, 16000, n_fft=512, hop_length=128)
    assert X.shape[0] == 257
    y = istft_mc(X, 16000, n_fft=512, hop_length=128, length=100)
    np.testing.assert_allclose(y, x, atol=1e-8, rtol=1e-6)
