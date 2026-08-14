from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mc_ambient import estimate_time_varying_scm, istft_mc, save_audio, stft_mc, synthesize_time_varying


def make_time_varying_reference(sample_rate: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sample_rate * seconds)
    split = n // 2
    u = rng.normal(size=(n, 3))
    A = np.array([[1.0, 0.15, 0.0], [0.7, 0.45, 0.1], [0.2, -0.25, 0.65]])
    B = np.array([[0.5, 0.55, 0.1], [-0.1, 0.45, 0.8], [0.75, 0.1, 0.2]])
    x = np.empty_like(u)
    x[:split] = u[:split] @ A.T
    x[split:] = u[split:] @ B.T
    x *= 0.12 / np.max(np.abs(x))
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/time_varying")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr, n_fft, hop = 16000, 512, 128
    reference = make_time_varying_reference(sr, 6.0, np.random.default_rng(2))
    X = stft_mc(reference, sr, n_fft=n_fft, hop_length=hop)

    frames_per_window = round(1.0 * sr / hop)
    hop_frames = round(0.5 * sr / hop)
    Rk, centers = estimate_time_varying_scm(X, frames_per_window, hop_frames)
    Y = synthesize_time_varying(
        Rk, centers, n_frames=X.shape[1], rng=np.random.default_rng(3), n_fft=n_fft, hop_length=hop
    )
    synthetic = istft_mc(Y, sr, n_fft=n_fft, hop_length=hop, length=len(reference))

    save_audio(out / "reference_time_varying.wav", reference, sr)
    save_audio(out / "synthetic_time_varying.wav", synthetic, sr)
    print(f"estimated {len(centers)} SCM windows")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
