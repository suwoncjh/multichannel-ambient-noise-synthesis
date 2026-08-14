from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from mc_ambient import (
    apply_channel_firs,
    apply_multichannel_rir,
    estimate_scm,
    istft_mc,
    mix_components,
    save_audio,
    stft_mc,
    synthesize_static,
)


def synthetic_rirs(sample_rate: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    length = int(0.35 * sample_rate)
    rirs = np.zeros((length, channels), dtype=np.float64)
    direct_delays = [80, 86, 95]
    direct_gains = [1.0, 0.86, 0.72]
    tau = 0.10 * sample_rate

    for ch in range(channels):
        delay = direct_delays[ch]
        rirs[delay, ch] = direct_gains[ch]
        idx = np.arange(length - delay)
        envelope = np.exp(-idx / tau)
        tail = rng.normal(size=length - delay) * envelope * 0.015
        tail[0] = 0.0
        rirs[delay:, ch] += tail
        for reflection, gain in [(320 + 17 * ch, 0.22), (690 + 31 * ch, -0.13), (1180 + 29 * ch, 0.08)]:
            if reflection < length:
                rirs[reflection, ch] += gain
    return rirs


def make_diffuse_reference(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.normal(size=(n, 4))
    mix = np.array(
        [
            [0.75, 0.35, 0.12, 0.05],
            [0.62, -0.20, 0.42, 0.08],
            [0.52, 0.08, -0.18, 0.48],
        ]
    )
    x = u @ mix.T
    for ch in range(3):
        x[:, ch] = lfilter([0.15, 0.25, 0.35, 0.25], [1.0, -0.45], x[:, ch])
    x *= 0.08 / np.max(np.abs(x))
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/advanced")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr, n_fft, hop = 16000, 512, 128
    rng = np.random.default_rng(10)
    n = int(4.0 * sr)

    dry = lfilter([1.0, -0.6], [1.0, -0.92], rng.normal(size=n))
    dry *= 0.15 / np.max(np.abs(dry))

    rirs = synthetic_rirs(sr, 3, np.random.default_rng(11))
    directional = apply_multichannel_rir(dry, rirs)
    directional *= 0.22 / np.max(np.abs(directional))

    diffuse_reference = make_diffuse_reference(len(directional), np.random.default_rng(12))
    Xd = stft_mc(diffuse_reference, sr, n_fft=n_fft, hop_length=hop)
    Rd = estimate_scm(Xd)
    Yd = synthesize_static(
        Rd, Xd.shape[1], np.random.default_rng(13), n_fft=n_fft, hop_length=hop
    )
    diffuse = istft_mc(Yd, sr, n_fft=n_fft, hop_length=hop, length=len(directional))
    diffuse *= 0.16 / max(np.max(np.abs(diffuse)), 1e-12)

    scene = mix_components(directional, diffuse, peak=0.75)

    device_firs = np.array(
        [
            [0.72, 0.66, 0.78],
            [0.22, 0.26, 0.17],
            [0.06, 0.08, 0.05],
        ],
        dtype=np.float64,
    )
    scene_device = apply_channel_firs(scene, device_firs)
    scene_device *= 0.75 / max(np.max(np.abs(scene_device)), 1e-12)

    save_audio(out / "dry_directional.wav", dry, sr)
    save_audio(out / "rirs_3ch.wav", rirs, sr)
    save_audio(out / "directional_reverberant_3ch.wav", directional, sr)
    save_audio(out / "diffuse_synthetic_3ch.wav", diffuse, sr)
    save_audio(out / "advanced_scene_3ch.wav", scene_device, sr)
    print(f"wrote {out}")
    print("For real-room consistency, use RIRs and diffuse/SCM statistics from the same room/device configuration.")


if __name__ == "__main__":
    main()
