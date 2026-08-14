from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from mc_ambient import estimate_scm, istft_mc, normalized_scm_error, save_audio, stft_mc, synthesize_static


def make_reference(sample_rate: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sample_rate * seconds)
    s1 = rng.normal(size=n)
    s2 = rng.normal(size=n)

    filters = [
        (np.array([1.0, 0.25, -0.08]), np.array([0.3, 0.0, 0.1])),
        (np.r_[np.zeros(2), [0.8, 0.2]], np.array([0.15, -0.1, 0.05])),
        (np.r_[np.zeros(4), [0.55, -0.18]], np.array([0.5, 0.15])),
    ]
    channels = []
    for h1, h2 in filters:
        y = fftconvolve(s1, h1, mode="full")[:n] + fftconvolve(s2, h2, mode="full")[:n]
        channels.append(y)
    x = np.stack(channels, axis=-1)
    x *= 0.15 / np.max(np.abs(x))
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/static")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr, n_fft, hop = 16000, 512, 128
    rng = np.random.default_rng(0)

    reference = make_reference(sr, 4.0, rng)
    X = stft_mc(reference, sr, n_fft=n_fft, hop_length=hop)
    R = estimate_scm(X)
    Y = synthesize_static(R, n_frames=X.shape[1], rng=np.random.default_rng(1), n_fft=n_fft, hop_length=hop)
    synthetic = istft_mc(Y, sr, n_fft=n_fft, hop_length=hop, length=len(reference))

    save_audio(out / "reference.wav", reference, sr)
    save_audio(out / "synthetic.wav", synthetic, sr)

    R_syn = estimate_scm(stft_mc(synthetic, sr, n_fft=n_fft, hop_length=hop), eps=0.0)
    print(f"normalized SCM error: {normalized_scm_error(R, R_syn):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
