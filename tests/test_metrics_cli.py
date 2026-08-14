import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mc_ambient.cli import main
from mc_ambient.metrics import complex_coherence, ipd, normalized_scm_error


def test_normalized_scm_error_is_zero_for_identical_inputs():
    scm = np.array([[[1.0, 0.4j], [-0.4j, 0.8]]], dtype=np.complex128)
    assert normalized_scm_error(scm, scm.copy()) == 0.0


def test_complex_coherence_is_one_for_duplicated_nonzero_channels():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(6, 80)) + 1j * rng.normal(size=(6, 80))
    spec = np.stack([base, base], axis=-1)
    gamma = complex_coherence(spec)
    np.testing.assert_allclose(np.abs(gamma[:, 0, 1]), 1.0, atol=1e-10)


def test_ipd_uses_x_i_times_conjugate_x_j_convention():
    spec = np.zeros((1, 3, 2), dtype=np.complex128)
    spec[0, :, 0] = np.exp(1j * np.array([0.0, 0.5, -0.3]))
    spec[0, :, 1] = 1.0
    np.testing.assert_allclose(ipd(spec, 0, 1)[0], [0.0, 0.5, -0.3], atol=1e-12)


def _write_reference(path: Path, sample_rate: int = 16000) -> None:
    rng = np.random.default_rng(8)
    n = 4096
    a = rng.normal(scale=0.08, size=n)
    b = 0.7 * a + rng.normal(scale=0.04, size=n)
    c = -0.25 * a + 0.4 * b + rng.normal(scale=0.03, size=n)
    sf.write(path, np.stack([a, b, c], axis=-1), sample_rate, subtype="FLOAT")


def test_cli_estimate_synthesize_validate_end_to_end(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    template = tmp_path / "scm.npz"
    synthetic = tmp_path / "synthetic.wav"
    report = tmp_path / "validation"
    _write_reference(reference)

    assert main(["estimate", str(reference), "--out", str(template), "--n-fft", "512", "--hop-length", "128"]) == 0
    assert template.exists()

    assert main(["synthesize", str(template), "--duration", "0.2", "--out", str(synthetic), "--seed", "17"]) == 0
    audio, sr = sf.read(synthetic, always_2d=True)
    assert sr == 16000
    assert audio.shape == (3200, 3)
    assert np.all(np.isfinite(audio))

    assert main(["validate", str(reference), str(synthetic), "--out", str(report), "--n-fft", "512", "--hop-length", "128"]) == 0
    summary_path = report / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["channels"] == 3
    assert np.isfinite(summary["normalized_scm_error"])
    assert (report / "psd.png").exists()
    assert (report / "coherence_magnitude.png").exists()
    assert (report / "scm_eigenspectrum.png").exists()


def test_cli_synthesis_reuses_template_hop_for_stft_consistency(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    template = tmp_path / "scm_hop160.npz"
    synthetic = tmp_path / "synthetic.wav"
    report = tmp_path / "validation"
    _write_reference(reference)

    assert main([
        "estimate", str(reference), "--out", str(template),
        "--n-fft", "512", "--hop-length", "160",
    ]) == 0
    assert main([
        "synthesize", str(template), "--duration", str(4096 / 16000),
        "--out", str(synthetic), "--seed", "23",
    ]) == 0
    assert main([
        "validate", str(reference), str(synthetic), "--out", str(report),
        "--n-fft", "512", "--hop-length", "160",
    ]) == 0

    summary = json.loads((report / "summary.json").read_text())
    assert summary["normalized_scm_error"] < 0.20
