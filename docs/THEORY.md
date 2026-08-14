# Theory and Practical Guide

## 1. Problem definition

Given a real `M`-microphone ambient recording, the objective is to generate a *different waveform realization* whose second-order multichannel spatial statistics resemble the recording.

For each STFT bin, define

\[
\mathbf{x}(f,t) = [X_1(f,t),\ldots,X_M(f,t)]^T.
\]

The frequency-dependent spatial covariance matrix (SCM) is

\[
\mathbf{R}_x(f)
= \mathbb{E}_t\left[\mathbf{x}(f,t)\mathbf{x}(f,t)^H\right].
\]

This single matrix contains channel powers on the diagonal and complex cross-spectra off the diagonal.

## 2. Empirical SCM estimation

For `T` STFT frames,

\[
\hat{\mathbf R}_x(f)
= \frac{1}{T}\sum_{t=1}^{T}\mathbf{x}(f,t)\mathbf{x}(f,t)^H.
\]

Finite-sample and floating-point effects can make the numerical matrix slightly non-Hermitian or indefinite. The implementation therefore

1. symmetrizes it as `(R + R^H)/2`,
2. eigen-decomposes it,
3. floors negative/small eigenvalues,
4. reconstructs a positive-semidefinite matrix.

This is especially important for highly coherent arrays or short SCM windows, where the empirical matrix may be nearly rank deficient.

## 3. Spatial coloring

Let

\[
\mathbf R_x(f) \approx \mathbf L(f)\mathbf L(f)^H.
\]

`L` is obtained by Cholesky when numerically possible, with an eigenvalue-decomposition fallback.

A naive implementation could generate independent circular complex STFT bins, but those bins are generally **not a consistent STFT** of any time-domain signal. ISTFT followed by STFT then acts as a projection and can substantially alter the covariance.

This repository instead generates independent Gaussian channels in the **time domain**, computes their STFT using the same `n_fft`/hop, estimates the seed SCM `R_u(f)`, and whitens it:

\[
\tilde{\mathbf u}(f,t)=\mathbf R_u(f)^{-1/2}\mathbf u(f,t),
\qquad
\mathbb E[\tilde{\mathbf u}\tilde{\mathbf u}^H]\approx\mathbf I.
\]

The time-domain origin preserves STFT consistency much better while the whitening step restores the desired identity covariance. The target coloring is then

\[
\mathbf y(f,t)=\mathbf L(f)\tilde{\mathbf u}(f,t).
\]

Then

\[
\mathbb E[\mathbf y\mathbf y^H]
= \mathbf L\mathbb E[\tilde{\mathbf u}\tilde{\mathbf u}^H]\mathbf L^H
\approx \mathbf R_x.
\]

This is the core static baseline.

## 4. Why measured SCMs contain reverberation

Consider `Q` sound sources observed by an `M`-microphone array. In the narrow-band approximation,

\[
\mathbf x(f,t)=\mathbf H(f)\mathbf s(f,t)+\mathbf v(f,t),
\]

where `H` contains the room/device transfer function from each source to each microphone. Then

\[
\mathbf R_x(f)
= \mathbf H(f)\mathbf R_s(f)\mathbf H(f)^H
+ \mathbf R_v(f)
\]

when the source and residual terms are uncorrelated.

`H(f)` is shaped by

- direct-path delay and attenuation,
- early reflections,
- late reverberation,
- source/microphone positions,
- microphone spacing and orientation,
- ports, ducts, cases, and analog/digital device responses.

Therefore an SCM estimated from a real recording contains the **combined reverberant spatial statistics** of that configuration.

### What it does not mean

`R_x(f)` does not uniquely determine `H(f)`. Multiple source covariance / transfer-function combinations can produce the same covariance. Consequently:

- SCM matching is not RIR estimation,
- an SCM does not give a unique RT60 or DRR,
- the generator does not reproduce a specific reflection sequence,
- late-reverberation temporal envelopes and higher-order statistics are only approximated indirectly.

This distinction is critical when describing the method as “realistic.” It is realistic with respect to the measured second-order spatial statistics, not a complete physical reconstruction of the room.

## 5. Explicit RIR path for a new room

To control the room explicitly, render each directional source before adding diffuse/background noise:

\[
y_m[n]
= \sum_q h_{q,m}[n] * s_q[n]
+ d_m[n]
+ v_m[n].
\]

Here

- `h_{q,m}` is a measured or simulated RIR from source `q` to mic `m`,
- `s_q` is a dry directional source,
- `d_m` is a diffuse/background multichannel realization,
- `v_m` can represent device/self noise.

The repository provides `apply_multichannel_rir()` for one source at a time. Multiple directional sources can be rendered independently and summed.

### Same-room consistency

For physically consistent simulation, the directional RIRs and diffuse-field statistics should correspond to the same room, array, and device response. A hybrid of room-A SCM + room-B RIR can still be useful augmentation, but should be described as augmentation rather than a physically exact room renderer.

## 6. Time-varying SCM

A single global SCM is often inadequate for real ambient scenes. Examples include moving vehicles, intermittent talkers, opening doors, HVAC state changes, or a handheld device whose orientation changes.

Estimate

\[
\mathbf R_x(f,k)
\]

over overlapping time windows `k`.

Hard-switching between factor matrices can introduce artificial changes in spatial coloration. This implementation cross-fades adjacent factors:

\[
\mathbf L(f,t)
= (1-\alpha_t)\mathbf L_k(f)
+ \alpha_t\mathbf L_{k+1}(f),
\]

and uses

\[
\mathbf y(f,t)=\mathbf L(f,t)\mathbf u(f,t).
\]

This is deliberately simple and auditable. Interpolating factors does not equal a mathematically exact geodesic interpolation of covariance matrices; it is a pragmatic v0.1 transition rule.

## 7. Window selection

There is a bias/variance tradeoff.

**Long windows**

- more stable SCM estimates,
- better for stationary diffuse/HVAC-like fields,
- smear rapid directional changes.

**Short windows**

- track moving/intermittent sources,
- have noisier covariance/eigenvalue estimates,
- are more likely to become rank deficient.

For a 3-mic phone array at 48 kHz, a useful starting point is:

- `n_fft = 1024` (~21.3 ms),
- `hop = 256` (~5.3 ms),
- SCM windows of 0.5–2 s,
- 50% SCM-window overlap.

Treat these as starting values, not universal settings.

## 8. Validation

### 8.1 Per-channel power

\[
P_m(f)=\mathbb E_t[|X_m(f,t)|^2].
\]

Matching PSD alone is insufficient for multichannel realism.

### 8.2 Complex coherence

\[
\Gamma_{ij}(f)
= \frac{R_{ij}(f)}{\sqrt{R_{ii}(f)R_{jj}(f)}}.
\]

Inspect both magnitude and phase when debugging spatial mismatch.

### 8.3 IPD

\[
\phi_{ij}(f,t)
= \angle\left(X_i(f,t)X_j(f,t)^*\right).
\]

The distribution of IPD is useful when the downstream model exploits inter-channel phase cues.

### 8.4 SCM eigenspectrum

Eigenvalues describe how spatial energy is distributed across array modes. A diffuse/high-rank field and a single dominant directional source can have very different eigenvalue spectra despite similar per-channel PSDs.

### 8.5 Normalized SCM error

\[
e_R
= \frac{\|R_{\text{ref}}-R_{\text{syn}}\|_F}
       {\|R_{\text{ref}}\|_F}.
\]

Use this as a compact engineering metric, not a perceptual score.

## 9. What to use for seed noise

v0.1 uses independent Gaussian **time-domain** seeds, analyzes them with the target STFT geometry, and whitens their empirical per-frequency SCM before coloring. This retains a clean covariance construction without the severe ISTFT→STFT mismatch of arbitrary i.i.d. complex spectrograms.

For more realistic texture, useful extensions are:

1. use decorrelated segments from a real mono ambient recording instead of Gaussian seeds,
2. preserve subband amplitude envelopes and impose only the target spatial covariance,
3. cluster time-varying SCM states and jointly sample a texture/state sequence,
4. combine directional event tracks with SCM-conditioned diffuse residuals,
5. train a conditional generative model while explicitly regularizing SCM/coherence.

The first extension is probably the best next step if the goal is perceptual realism without introducing a large learned model.

## 10. Smartphone / compact-array caveats

For small phone arrays, several effects become important:

- closely spaced microphones have strong low-frequency coherence,
- ports/ducts/case geometry can dominate high-frequency transfer functions,
- handling noise and wind may be nonstationary and non-Gaussian,
- device rotation changes directional statistics even in the same physical room,
- a template measured on one hardware revision may not transfer to another.

If the intended training data must match a specific product microphone geometry, estimate the SCM using that device or a sufficiently faithful acoustic mock-up rather than transferring an SCM between arrays.

## 11. Recommended research extensions

A natural progression beyond this baseline is:

1. **Measured static SCM** — current baseline.
2. **Measured time-varying SCM** — current implementation.
3. **Directional + diffuse decomposition** — separate low-rank directional events from residual field statistics.
4. **RIR-conditioned scene composition** — current minimal advanced example, extended to multiple source positions and same-room diffuse models.
5. **Texture-preserving spatialization** — real mono/non-Gaussian seeds rather than Gaussian seeds.
6. **Learned conditional synthesis** — condition on room/acoustic descriptors while retaining explicit coherence/SCM losses.

## 12. Related work

- Habets, Cohen, Gannot, “Generating nonstationary multisensor signals under a spatial coherence constraint,” JASA (2008).
- Mirabilii, Schlecht, Habets, “Generating coherence-constrained multisensor signals using balanced mixing and spectrally smooth filters,” JASA (2021).
- `audiolabs/anf-generator` for coherence-constrained artificial noise fields.
- RealMAN for evidence that real-recorded spatial noise can differ materially from simple ideal diffuse-field assumptions.
- SonicSim for a broader explicit acoustic-scene simulation direction.

Links are collected in the repository README.
