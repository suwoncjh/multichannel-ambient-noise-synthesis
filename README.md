# Multichannel Ambient Noise Synthesis

Research-oriented Python utilities for synthesizing **N-channel ambient noise whose frequency-dependent spatial covariance/coherence matches a real multichannel recording**.

The core idea is simple: estimate a measured spatial covariance matrix (SCM) from a real microphone-array recording, factor it, and use the factor to spatially color independent complex noise in the STFT domain. The repository also includes a time-varying SCM path and an explicit multichannel-RIR path for controlled room reverberation.

## Why this exists

A common augmentation shortcut is to copy one mono noise signal to multiple microphones and add delays or gains. That usually misses the real array statistics:

- frequency-dependent inter-channel magnitude and phase
- partial spatial coherence
- microphone geometry and port/device coloration
- reflections and reverberant spatial statistics
- time variation of the noise field

This project targets those **second-order multichannel statistics** directly.

> **Reverberation note:** if the SCM is estimated from a real multichannel recording, the measured room's reverberation is already embedded *implicitly* in the cross-spectral statistics. The SCM does **not** uniquely recover the room impulse responses. To change the room, source position, RT60/DRR, or array geometry, use the explicit multichannel-RIR workflow shown in `examples/03_directional_diffuse_mic_response.py`.

## Installation

```bash
git clone https://github.com/suwoncjh/multichannel-ambient-noise-synthesis.git
cd multichannel-ambient-noise-synthesis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.10+ and uses NumPy, SciPy, SoundFile, and Matplotlib. No PyTorch is required.

## Quick start: clone the spatial statistics of a real recording

Assume `ambient_3ch.wav` is shaped `[samples, 3 microphones]`.

### 1. Estimate a static SCM template

```bash
mc-ambient estimate ambient_3ch.wav \
  --out ambient_scm.npz \
  --n-fft 1024 \
  --hop-length 256
```

### 2. Synthesize a new multichannel realization

```bash
mc-ambient synthesize ambient_scm.npz \
  --duration 10 \
  --seed 123 \
  --out synthetic_3ch.wav
```

### 3. Compare real vs synthetic spatial statistics

```bash
mc-ambient validate ambient_3ch.wav synthetic_3ch.wav \
  --out validation \
  --n-fft 1024 \
  --hop-length 256
```

The validation directory contains:

- `summary.json` — normalized SCM error and mean complex-coherence error
- `psd.png` — per-channel STFT power
- `coherence_magnitude.png` — microphone-pair coherence magnitude
- `scm_eigenspectrum.png` — frequency-wise spatial eigenvalue spectrum

## Time-varying ambient fields

Real ambient fields are often not stationary. Estimate overlapping SCM windows instead of one global SCM:

```bash
mc-ambient estimate ambient_3ch.wav \
  --out ambient_tv_scm.npz \
  --window-seconds 1.0 \
  --window-hop-seconds 0.5

mc-ambient synthesize ambient_tv_scm.npz \
  --duration 10 \
  --seed 123 \
  --out synthetic_tv_3ch.wav
```

During synthesis, neighboring SCM **factors** are linearly cross-faded frame by frame rather than hard-switched. This avoids an artificial step in spatial coloration at window boundaries.

For a 48 kHz smartphone recording, a practical starting point is `n_fft=1024`, `hop=256`, and a 0.5–2 s SCM window. The best window is environment-dependent: shorter windows track moving/directional interferers better but give noisier covariance estimates.

## Python API

```python
import numpy as np
from mc_ambient import (
    estimate_scm,
    istft_mc,
    load_audio,
    save_audio,
    stft_mc,
    synthesize_static,
)

x, sr = load_audio("ambient_3ch.wav")
X = stft_mc(x, sr, n_fft=1024, hop_length=256)
R = estimate_scm(X)

Y = synthesize_static(
    R,
    n_frames=X.shape[1],
    rng=np.random.default_rng(0),
    n_fft=1024,
    hop_length=256,
)
y = istft_mc(Y, sr, n_fft=1024, hop_length=256, length=len(x))
save_audio("synthetic_3ch.wav", y, sr)
```

The static model is

$$
\mathbf R_x(f)=\mathbb E_t[\mathbf x(f,t)\mathbf x(f,t)^H],
$$

followed by

$$
\mathbf R_x(f)\approx\mathbf L(f)\mathbf L(f)^H,
\qquad
\mathbf y(f,t)=\mathbf L(f)\mathbf u(f,t),
$$

where `u` is a **time-domain-consistent seed STFT** whose empirical per-frequency covariance has been whitened to approximately identity. The implementation deliberately does not draw arbitrary i.i.d. complex STFT bins, because an inconsistent spectrogram can change substantially after ISTFT→STFT projection.

## Reverberation: what is and is not captured

For a set of sources observed through a room,

$$
\mathbf x(f,t)=\mathbf H(f)\mathbf s(f,t)+\mathbf v(f,t),
$$

so approximately

$$
\mathbf R_x(f)=\mathbf H(f)\mathbf R_s(f)\mathbf H(f)^H+\mathbf R_v(f).
$$

Therefore a **measured SCM inherits the combined effect** of source distribution, early/late reflections, microphone spacing, and device transfer functions at the level of second-order STFT statistics.

But many different rooms/source configurations can yield similar SCMs. SCM matching alone does not identify a unique RIR and does not reproduce all temporal or non-Gaussian texture of a real environment.

For a different room, explicitly render directional sources with multichannel RIRs:

```python
from mc_ambient import apply_multichannel_rir

# rirs: [rir_samples, microphones]
reverberant_source = apply_multichannel_rir(dry_source, rirs)
```

Then combine this directional component with a diffuse/background SCM realization and optional per-microphone FIR response. See the advanced example.

## Run the included examples

```bash
python examples/01_static_scm.py --out-dir outputs/static
python examples/02_time_varying_scm.py --out-dir outputs/time_varying
python examples/03_directional_diffuse_mic_response.py --out-dir outputs/advanced
```

All examples generate their own synthetic source/reference data; no proprietary recordings are required.

### `01_static_scm.py`

Creates a synthetic 3-microphone reference field, estimates its global SCM, generates a new realization, and reports normalized SCM error.

### `02_time_varying_scm.py`

Creates a field whose spatial mixing changes halfway through the clip, estimates overlapping SCMs, and synthesizes a smoothly changing realization.

### `03_directional_diffuse_mic_response.py`

Demonstrates

```text
dry directional source
        |
        v
multichannel RIR convolution  ---> directional reverberant component
                                      |
SCM-conditioned diffuse background --+--> mix --> per-channel device FIR --> output
```

In real experiments, use measured/simulated multichannel RIRs and, when physical consistency matters, estimate the diffuse/background SCM in the same target room/device configuration.

## Validation metrics

The repository exposes or uses:

- frequency-wise SCM
- normalized Frobenius SCM error
- complex coherence
- inter-channel phase difference (IPD)
- per-channel STFT power
- SCM eigenvalue spectrum

For microphone pair `(i, j)`, complex coherence is

$$
\Gamma_{ij}(f)=
\frac{R_{ij}(f)}{\sqrt{R_{ii}(f)R_{jj}(f)}}.
$$

A synthetic signal can match channel PSDs while still having incorrect spatial coherence, so evaluate both.

## Important limitations

1. **Second-order model.** The current generator matches covariance/coherence, not higher-order amplitude modulation, event structure, or semantic texture.
2. **Gaussian time-domain seed.** The seed STFT is generated from independent Gaussian time signals and then whitened. This preserves STFT consistency much better than arbitrary complex bins, but non-Gaussian real noise such as applause, traffic transients, or intermittent speech may still sound less realistic even with a good SCM match.
3. **No unique RIR recovery.** Measured SCMs embed reverberant statistics but do not identify the room impulse responses that produced them.
4. **Geometry-specific templates.** A measured SCM is tied to the source field, microphone geometry, sample rate, and device responses used during measurement.
5. **Hybrid-room consistency.** Mixing a diffuse SCM measured in room A with directional RIRs from room B is useful augmentation but is not a physically exact rendering of either room.

See [`docs/THEORY.md`](docs/THEORY.md) for more detail and suggested extensions.

## Repository structure

```text
src/mc_ambient/
  io.py          WAV I/O and shape checks
  stft.py        multichannel STFT / ISTFT
  scm.py         SCM estimation, PSD projection, factorization
  synthesis.py   static and time-varying SCM synthesis
  room.py        multichannel RIR and device FIR processing
  metrics.py     coherence, IPD, SCM error, eigenspectrum
  cli.py         estimate / synthesize / validate commands
examples/
docs/
tests/
```

## Related work

This implementation is intended as a compact empirical-SCM baseline rather than a replacement for the established coherence-constrained noise-field literature.

- E. A. P. Habets, I. Cohen, and S. Gannot, **“Generating nonstationary multisensor signals under a spatial coherence constraint,”** JASA, 2008. https://pubs.aip.org/asa/jasa/article/124/5/2911/910731/Generating-nonstationary-multisensor-signals-under
- D. Mirabilii, S. J. Schlecht, and E. A. P. Habets, **“Generating coherence-constrained multisensor signals using balanced mixing and spectrally smooth filters,”** JASA, 2021. Project/code resources: https://www.audiolabs-erlangen.de/resources/2020-JASA-CCR and https://github.com/audiolabs/anf-generator
- RealMAN, real-recorded multichannel speech/noise corpus and analysis: https://arxiv.org/abs/2406.19959
- SonicSim, acoustic simulation for realistic spatial audio scenes: https://arxiv.org/abs/2410.01481

## License

MIT. See [`LICENSE`](LICENSE).
