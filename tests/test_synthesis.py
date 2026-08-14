import numpy as np

from mc_ambient.scm import estimate_scm, estimate_time_varying_scm
from mc_ambient.synthesis import synthesize_static, synthesize_time_varying


def _relative_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_static_synthesis_recovers_target_scm():
    target = np.array(
        [
            [1.0, 0.55 + 0.2j, 0.15 - 0.08j],
            [0.55 - 0.2j, 0.9, 0.3 + 0.1j],
            [0.15 + 0.08j, 0.3 - 0.1j, 0.7],
        ],
        dtype=np.complex128,
    )
    scm = np.repeat(target[None, :, :], 5, axis=0)
    y = synthesize_static(scm, n_frames=12000, rng=np.random.default_rng(11))
    estimated = estimate_scm(y, eps=0.0)
    assert y.shape == (5, 12000, 3)
    assert _relative_error(scm, estimated) < 0.04


def test_time_varying_scm_estimation_returns_expected_windows():
    rng = np.random.default_rng(2)
    spec = rng.normal(size=(4, 20, 2)) + 1j * rng.normal(size=(4, 20, 2))
    windows, centers = estimate_time_varying_scm(spec, frames_per_window=8, hop_frames=4)
    assert windows.shape == (4, 4, 2, 2)
    np.testing.assert_allclose(centers, [3.5, 7.5, 11.5, 15.5])


def test_time_varying_synthesis_crossfades_factor_state_without_hard_step():
    n_freq = 257
    low = np.ones((n_freq, 1, 1), dtype=np.complex128)
    high = np.full((n_freq, 1, 1), 9.0, dtype=np.complex128)
    windows = np.stack([low, high], axis=0)
    baseline_windows = np.stack([low, low], axis=0)
    centers = np.array([0.0, 20.0])

    y = synthesize_time_varying(
        windows,
        centers=centers,
        n_frames=21,
        rng=np.random.default_rng(13),
        n_fft=512,
        hop_length=128,
    )
    baseline = synthesize_time_varying(
        baseline_windows,
        centers=centers,
        n_frames=21,
        rng=np.random.default_rng(13),
        n_fft=512,
        hop_length=128,
    )

    ratio = (np.mean(np.abs(y[:, :, 0]) ** 2, axis=0) /
             np.mean(np.abs(baseline[:, :, 0]) ** 2, axis=0))
    np.testing.assert_allclose(ratio[[0, 10, 20]], [1.0, 4.0, 9.0], atol=1e-10)
    assert np.max(np.diff(ratio)) < 0.6


def test_static_synthesis_preserves_scm_after_istft_stft_round_trip():
    from mc_ambient.stft import istft_mc, stft_mc

    sr, n_fft, hop = 16000, 512, 128
    n = 32768
    rng = np.random.default_rng(21)
    u = rng.normal(size=(n, 3))
    mix = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.65, 0.55, 0.1],
            [0.25, -0.15, 0.7],
        ]
    )
    reference = u @ mix.T
    X = stft_mc(reference, sr, n_fft=n_fft, hop_length=hop)
    target = estimate_scm(X)

    Y = synthesize_static(
        target,
        n_frames=X.shape[1],
        rng=np.random.default_rng(22),
        n_fft=n_fft,
        hop_length=hop,
    )
    y = istft_mc(Y, sr, n_fft=n_fft, hop_length=hop, length=n)
    round_trip = estimate_scm(stft_mc(y, sr, n_fft=n_fft, hop_length=hop), eps=0.0)

    assert _relative_error(target, round_trip) < 0.10
