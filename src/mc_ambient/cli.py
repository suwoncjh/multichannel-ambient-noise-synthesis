from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import load_audio, save_audio
from .metrics import channel_psd, complex_coherence, normalized_scm_error, scm_eigenspectrum
from .scm import estimate_scm, estimate_time_varying_scm
from .stft import istft_mc, stft_mc
from .synthesis import synthesize_static, synthesize_time_varying


def _save_template(
    path: Path,
    *,
    scm: np.ndarray,
    centers: np.ndarray,
    kind: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    reference_frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scm=scm,
        centers=np.asarray(centers, dtype=np.float64),
        kind=np.asarray(kind),
        sample_rate=np.asarray(sample_rate, dtype=np.int64),
        n_fft=np.asarray(n_fft, dtype=np.int64),
        hop_length=np.asarray(hop_length, dtype=np.int64),
        reference_frames=np.asarray(reference_frames, dtype=np.int64),
    )


def _estimate(args: argparse.Namespace) -> int:
    audio, sample_rate = load_audio(args.input)
    spec = stft_mc(audio, sample_rate, n_fft=args.n_fft, hop_length=args.hop_length)

    if args.window_seconds is None:
        scm = estimate_scm(spec)
        centers = np.empty(0, dtype=np.float64)
        kind = "static"
    else:
        if args.window_seconds <= 0:
            raise ValueError("--window-seconds must be positive")
        hop_seconds = args.window_hop_seconds
        if hop_seconds is None:
            hop_seconds = args.window_seconds / 2.0
        if hop_seconds <= 0:
            raise ValueError("--window-hop-seconds must be positive")
        frames_per_window = max(1, int(round(args.window_seconds * sample_rate / args.hop_length)))
        hop_frames = max(1, int(round(hop_seconds * sample_rate / args.hop_length)))
        scm, centers = estimate_time_varying_scm(
            spec,
            frames_per_window=frames_per_window,
            hop_frames=hop_frames,
        )
        kind = "time_varying"

    _save_template(
        Path(args.out),
        scm=scm,
        centers=centers,
        kind=kind,
        sample_rate=sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        reference_frames=spec.shape[1],
    )
    return 0


def _load_template(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"scm", "centers", "kind", "sample_rate", "n_fft", "hop_length", "reference_frames"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"template is missing fields: {sorted(missing)}")
        return {
            "scm": np.asarray(data["scm"], dtype=np.complex128),
            "centers": np.asarray(data["centers"], dtype=np.float64),
            "kind": str(data["kind"].item()),
            "sample_rate": int(data["sample_rate"].item()),
            "n_fft": int(data["n_fft"].item()),
            "hop_length": int(data["hop_length"].item()),
            "reference_frames": int(data["reference_frames"].item()),
        }


def _synthesize(args: argparse.Namespace) -> int:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    template = _load_template(args.template)
    sample_rate = int(template["sample_rate"])
    if args.sample_rate is not None and args.sample_rate != sample_rate:
        raise ValueError(
            f"template was estimated at {sample_rate} Hz; refusing to reinterpret it at {args.sample_rate} Hz"
        )
    n_fft = int(template["n_fft"])
    hop_length = int(template["hop_length"])
    scm = np.asarray(template["scm"])
    channels = int(scm.shape[-1])
    n_samples = max(1, int(round(args.duration * sample_rate)))
    n_frames = max(1, int(np.ceil(n_samples / hop_length)) + 1)
    rng = np.random.default_rng(args.seed)

    if template["kind"] == "static":
        spec = synthesize_static(
            scm,
            n_frames=n_frames,
            rng=rng,
            match_power=not args.spatial_only,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    elif template["kind"] == "time_varying":
        centers = np.asarray(template["centers"], dtype=np.float64)
        reference_frames = int(template["reference_frames"])
        if reference_frames > 1 and n_frames > 1:
            centers = centers * (n_frames - 1) / (reference_frames - 1)
        spec = synthesize_time_varying(
            scm,
            centers=centers,
            n_frames=n_frames,
            rng=rng,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    else:
        raise ValueError(f"unknown template kind: {template['kind']}")

    if spec.shape[-1] != channels:
        raise RuntimeError("synthesis produced an unexpected channel count")
    audio = istft_mc(spec, sample_rate, n_fft=n_fft, hop_length=hop_length, length=n_samples)
    save_audio(args.out, audio, sample_rate)
    return 0


def _plot_psd(reference_spec: np.ndarray, synthetic_spec: np.ndarray, sample_rate: int, out: Path) -> None:
    ref = channel_psd(reference_spec)
    syn = channel_psd(synthetic_spec)
    freqs = np.linspace(0.0, sample_rate / 2.0, ref.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    floor = np.finfo(np.float64).tiny
    for ch in range(ref.shape[1]):
        ax.plot(freqs, 10 * np.log10(np.maximum(ref[:, ch], floor)), label=f"ref ch{ch}")
        ax.plot(freqs, 10 * np.log10(np.maximum(syn[:, ch], floor)), linestyle="--", label=f"syn ch{ch}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("STFT power (dB)")
    ax.set_title("Per-channel PSD proxy")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_coherence(reference_spec: np.ndarray, synthetic_spec: np.ndarray, sample_rate: int, out: Path) -> None:
    ref = complex_coherence(reference_spec)
    syn = complex_coherence(synthetic_spec)
    freqs = np.linspace(0.0, sample_rate / 2.0, ref.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    channels = ref.shape[-1]
    pairs = [(i, j) for i in range(channels) for j in range(i + 1, channels)] or [(0, 0)]
    for i, j in pairs:
        ax.plot(freqs, np.abs(ref[:, i, j]), label=f"ref {i}-{j}")
        ax.plot(freqs, np.abs(syn[:, i, j]), linestyle="--", label=f"syn {i}-{j}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|complex coherence|")
    ax.set_title("Spatial coherence")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_eigenspectrum(reference_scm: np.ndarray, synthetic_scm: np.ndarray, sample_rate: int, out: Path) -> None:
    ref = np.maximum(scm_eigenspectrum(reference_scm), np.finfo(np.float64).tiny)
    syn = np.maximum(scm_eigenspectrum(synthetic_scm), np.finfo(np.float64).tiny)
    freqs = np.linspace(0.0, sample_rate / 2.0, ref.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(ref.shape[1]):
        ax.plot(freqs, 10 * np.log10(ref[:, k]), label=f"ref eig{k}")
        ax.plot(freqs, 10 * np.log10(syn[:, k]), linestyle="--", label=f"syn eig{k}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Eigenvalue (dB)")
    ax.set_title("SCM eigenspectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _validate(args: argparse.Namespace) -> int:
    reference, sample_rate_ref = load_audio(args.reference)
    synthetic, sample_rate_syn = load_audio(args.synthetic)
    if sample_rate_ref != sample_rate_syn:
        raise ValueError("reference and synthetic audio must have the same sample rate")
    if reference.shape[1] != synthetic.shape[1]:
        raise ValueError("reference and synthetic audio must have the same channel count")

    ref_spec = stft_mc(reference, sample_rate_ref, n_fft=args.n_fft, hop_length=args.hop_length)
    syn_spec = stft_mc(synthetic, sample_rate_syn, n_fft=args.n_fft, hop_length=args.hop_length)
    ref_scm = estimate_scm(ref_spec, eps=0.0)
    syn_scm = estimate_scm(syn_spec, eps=0.0)
    ref_coh = complex_coherence(ref_spec)
    syn_coh = complex_coherence(syn_spec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot_psd(ref_spec, syn_spec, sample_rate_ref, out / "psd.png")
    _plot_coherence(ref_spec, syn_spec, sample_rate_ref, out / "coherence_magnitude.png")
    _plot_eigenspectrum(ref_scm, syn_scm, sample_rate_ref, out / "scm_eigenspectrum.png")

    summary = {
        "sample_rate": sample_rate_ref,
        "channels": int(reference.shape[1]),
        "reference_samples": int(reference.shape[0]),
        "synthetic_samples": int(synthetic.shape[0]),
        "normalized_scm_error": normalized_scm_error(ref_scm, syn_scm),
        "mean_complex_coherence_abs_error": float(np.mean(np.abs(ref_coh - syn_coh))),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mc-ambient",
        description="Measured-SCM multichannel ambient noise synthesis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="estimate static or time-varying SCMs from a real multichannel recording")
    estimate.add_argument("input")
    estimate.add_argument("--out", required=True)
    estimate.add_argument("--n-fft", type=int, default=1024)
    estimate.add_argument("--hop-length", type=int, default=256)
    estimate.add_argument("--window-seconds", type=float)
    estimate.add_argument("--window-hop-seconds", type=float)
    estimate.set_defaults(func=_estimate)

    synth = sub.add_parser("synthesize", help="generate multichannel noise from an SCM template")
    synth.add_argument("template")
    synth.add_argument("--duration", type=float, required=True)
    synth.add_argument("--out", required=True)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--sample-rate", type=int)
    synth.add_argument("--spatial-only", action="store_true", help="normalize away per-frequency absolute power")
    synth.set_defaults(func=_synthesize)

    validate = sub.add_parser("validate", help="compare real and synthetic multichannel spatial statistics")
    validate.add_argument("reference")
    validate.add_argument("synthetic")
    validate.add_argument("--out", required=True)
    validate.add_argument("--n-fft", type=int, default=1024)
    validate.add_argument("--hop-length", type=int, default=256)
    validate.set_defaults(func=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
