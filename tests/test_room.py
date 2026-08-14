import numpy as np

from mc_ambient.room import apply_channel_firs, apply_multichannel_rir, mix_components


def test_apply_multichannel_rir_preserves_channel_specific_delays_and_gains():
    source = np.zeros(4, dtype=np.float64)
    source[0] = 1.0
    rirs = np.zeros((6, 3), dtype=np.float64)
    rirs[0, 0] = 1.0
    rirs[2, 1] = 0.5
    rirs[5, 2] = -0.25
    y = apply_multichannel_rir(source, rirs)
    assert y.shape == (9, 3)
    np.testing.assert_allclose(y[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(y[2, 1], 0.5, atol=1e-12)
    np.testing.assert_allclose(y[5, 2], -0.25, atol=1e-12)
    assert np.count_nonzero(np.abs(y) > 1e-12) == 3


def test_apply_channel_firs_colors_each_channel_independently():
    audio = np.zeros((4, 2), dtype=np.float64)
    audio[0, :] = 1.0
    firs = np.array([[1.0, 0.5], [0.25, -0.25]], dtype=np.float64)
    y = apply_channel_firs(audio, firs)
    assert y.shape == (5, 2)
    np.testing.assert_allclose(y[:2, 0], [1.0, 0.25], atol=1e-12)
    np.testing.assert_allclose(y[:2, 1], [0.5, -0.25], atol=1e-12)


def test_mix_components_zero_pads_shorter_inputs_and_peak_limits_only_when_needed():
    a = np.ones((4, 2), dtype=np.float64) * 0.2
    b = np.ones((2, 2), dtype=np.float64) * 0.3
    y = mix_components(a, b, peak=0.95)
    np.testing.assert_allclose(y[:2], 0.5)
    np.testing.assert_allclose(y[2:], 0.2)
    loud = mix_components(np.ones((3, 2)), np.ones((3, 2)), peak=0.5)
    assert np.max(np.abs(loud)) <= 0.5 + 1e-12
