import numpy as np

from mc_ambient.scm import estimate_scm, factor_scm, project_psd


def _relative_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_estimate_scm_recovers_known_complex_covariance():
    target = np.array(
        [
            [1.0, 0.45 + 0.15j, 0.2 - 0.1j],
            [0.45 - 0.15j, 0.8, 0.25 + 0.05j],
            [0.2 + 0.1j, 0.25 - 0.05j, 0.6],
        ],
        dtype=np.complex128,
    )
    L = np.linalg.cholesky(target)
    rng = np.random.default_rng(3)
    u = (rng.normal(size=(24000, 3)) + 1j * rng.normal(size=(24000, 3))) / np.sqrt(2.0)
    x = u @ L.T
    spec = x[None, :, :]
    estimated = estimate_scm(spec)
    assert estimated.shape == (1, 3, 3)
    assert _relative_error(target, estimated[0]) < 0.025


def test_project_psd_makes_indefinite_matrix_hermitian_positive_semidefinite():
    bad = np.array([[1.0, 1.1 + 0.3j], [1.1 - 0.3j, 0.2]], dtype=np.complex128)
    projected = project_psd(bad, eig_floor=1e-5)
    np.testing.assert_allclose(projected, projected.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(projected)) >= 0.999e-5


def test_factor_scm_reconstructs_rank_deficient_target_after_flooring():
    target = np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.complex128)
    projected = project_psd(target, eig_floor=1e-7)
    factor = factor_scm(target, eig_floor=1e-7)
    reconstructed = factor @ np.swapaxes(factor.conj(), -1, -2)
    np.testing.assert_allclose(reconstructed, projected, atol=1e-9, rtol=1e-7)
