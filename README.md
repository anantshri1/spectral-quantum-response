# Derivative-Informed Operator Learning for Quantum Response Functions
A systematic study to test whether a Fourier Neural Operator (FNO) trained only on forward evolution `(ψ₀, V) → ψ_T` learns the correct response `∂ψ_T/∂V `as a byproduct.

---
## Background and Setup
The forward map only constrains the endpoint `ψ(T)`. The response is the propagator sandwiched around a perturbation and integrated over the whole trajectory:

```
δψ_T = -i ∫₀ᵀ U(T,t) · δV · U(t,0) · ψ₀ dt
```

Two potentials can produce nearly identical `ψ(T)` while having quite different response operators — the derivative sees the path, the endpoint does not. Therefore, assessing whether a model only sees the endpoints during training or also learns the correct path-sensitivity is a real empirical question.

### System setup

**1D free-rotor / quantum-pendulum problem**: a particle on a periodic ring (`circumference L=2π, ℏ=m=R=1`), Hamiltonian `H = -½∂²/∂φ² + V(φ)`, untwisted boundary conditions (`θ=0`, standard FFT applies directly — twisted BC would introduce Aharonov–Bohm/θ-vacuum physics we deliberately avoid). `ψ₀` is a random Gaussian wavepacket (center, width, integer carrier momentum `k₀∈[-4,4])`; V is a smooth Gaussian random field with a tunable spectral length-scale.

**Solver and Strang splitting**: K (kinetic) is diagonal in Fourier space, V (potential) is diagonal in real space, and `exp(-i(K+V)dt) ≠ exp(-iKdt)exp(-iVdt)` since `[K,V]≠0`. Symmetrizing as `exp(-iVdt/2)·exp(-iKdt)·exp(-iVdt/2)` cancels the leading commutator term, buying 2nd-order global accuracy for one extra FFT. Implemented as `jax.lax.scan` over timesteps rather than a Python for loop (not a Python loop — scan is what makes autodifferentiating through the ~10,000-step evolution tractable; a Python loop would make the response phase computationally impossible). Kinetic phase is a pointwise multiply in Fourier space (`k` from `fftfreq`, exact integers because `L=2π`); potential phase is a pointwise multiply in real space. Validated against four independent physics gates: exact plane-wave phase, norm conservation to float64 machine precision, analytic Gaussian-wavepacket spreading rate, and an autodiff-vs-finite-difference gradient check (clean U-shaped error valley bottoming at `~3×10⁻¹⁰`, confirming `jax.jvp` through the scan is correct).

> **Gate 1 — plane-wave eigenstate phase**: With `V=0`, `ψ=e^{ikφ}` is an exact eigenstate of the kinetic operator, so one Strang step should multiply it by the analytically-known phase `exp(-i·½k²·dt)` with no shape change. Checked to `atol=1e-5`. This validates the FFT round-trip and the k-grid convention (`jnp.fft.fftfreq(nx, d=1/nx)`, not arange — the top half of the array holds negative frequencies, e.g. bin 64 of a 128-point grid is k=-64 not k=+64, a silent-bug risk since the solver would run without error but with wrong kinetic energies).

> **Gate 2 — norm conservation**: Unitary evolution must conserve `∫|ψ|²dφ` exactly. This tests the composed solver (all three Strang factors, many steps, real V) rather than individual factors. First run in float32 showed drift growing with `nt` at fixed total time (`3.3e-5 → 3.4e-4` across `nt=500→4000`) — diagnosed via an nt-scaling probe as accumulated per-step FFT round-off, confirmed by re-running in float64 where drift collapsed to `~5e-14` and stayed flat regardless of `nt`. This result is what locked the precision split (`solver/data-gen/responses` in float64, FNO training in float32) — decided from evidence, not assumed upfront, since unitary evolution has no dissipation to forgive float32 errors the way Burgers' viscosity did.

> **Gate 3 — Gaussian wavepacket spreading**: Tests genuine non-eigenstate dynamics (gate 1 only tested an eigenstate that does not change shape): a free packet's width should grow as `σ(t)²=σ₀²+(t/2σ₀)²`. First attempt failed at `t=0` (29% error before any evolution occurred), correctly diagnosed as a measurement bug rather than a solver bug, since nothing had run yet — traced to a `√2` mismatch between the amplitude-width of the constructed ψ and the density-width (|ψ|²) the analytic formula assumes. A second attempt still failed at `t=0` by the same `√2` in the opposite direction, from over-correcting; resolved by collapsing to one single source of truth (pick amplitude-width a, feed the formula `a/√2` and nothing else). Once fixed, `t=0` agreed to `~1e-12` and dynamics tracked the analytic curve to `~1e-9`, with error growing smoothly with `t` (expected accumulated round-off) and no sign of the ring's periodicity corrupting the result within the tested window.

> **Gate 4 — gradient check (autodiff vs. finite differences)**: The gate that specifically future phases: since `J_true` will be computed by `jax.jvp` through the full scan, this validates that differentiation, not just evaluation, is correct. Central FD along a random direction `δV`, swept across `ε∈[10⁻⁸,10⁻¹]`. First single-point check (`ε=1e-6`) gave `1.9e-7` relative error — plausible but not sufficient evidence alone. A one-sided sweep (only small ε) showed monotonically rising error toward small ε, initially read as concerning, until recognized as only the round-off wall (∝1/ε, from amplifying fixed floating-point noise by dividing by shrinking ε) — the truncation wall (∝ε²) hadn't been sampled yet because this particular map is smooth enough that its valley floor sits at unusually large ε (~10⁻³) rather than the generic estimate of `~10⁻⁵–10⁻⁶`. Extending the sweep to large ε revealed the missing wall, giving a complete U-shaped valley bottoming at `2.8×10⁻¹⁰` with the correct power-law slope on each side (ε² falling, 1/ε rising) — the strongest form of confirmation, since a genuinely broken gradient would show a floor (FD converging to the true derivative while autodiff sits a fixed, non-vanishing distance away) rather than a valley reaching machine-precision-level agreement.

**Precision**: solver, data generation, and the "ground truth" response Jacobian all run in float64; FNO training runs in float32. This is a deliberate split, not an oversight — long unitary evolutions are precision-sensitive (no dissipation to forgive float32 round-off) and get differentiated through, while the FNO is a shallow, dissipation-tolerant learned function. Precision floor for `train.py` (f64-trained, canonical M=16 checkpoint) is forward rel-L2 = 0.0192, response error = 0.5787.

**FNO**: hand-implemented spectral convolution, `n_modes` as the swept architecture dial (2–16 in various experiments), complex `ψ` handled as two real channels at the model boundary (`w_real/w_imag` stored separately — complex-typed Adam second moments are ill-defined and destroy training).

**Response machinery** (`src/response.py`): `J_true = jacfwd` of `V ↦ ψ_T` (V held real, ψ₀ fixed — this differentiation is Wirtinger-clean because V is real, unlike differentiating w.r.t. complex ψ₀, which we deliberately avoid). `J_fno` is the identical machinery applied to the trained surrogate. Both are rotated from position space to mode space via FFT/IFFT to get `J(k,q) = ∂ψ̂_k/∂V̂_q`, interpreted as a physical mode-coupling operator (for a local potential, natively off-diagonal via `⟨k|V|q⟩ = V̂(k-q)`). `response_error = ‖J_fno - J_true‖_F / ‖J_true‖_F`.

### Data Generation

* Design principle carried from Burgers, inverted in character: solve only at the finest grid (headroom pushed to `nx=1024`, up from an initial 512, specifically to leave room to observe unexpected high-k behavior) and subsample the solution down the ladder — never solve coarse and interpolate up. In Burgers this was safe by default (viscosity damps high-k); here it required an explicit safeguard, since unitary evolution can push energy to high-k during evolution via V's mode-coupling, and subsampling would then alias that content back down as spurious low-k signal in the coarse ground truth. This motivated a mandatory spectral-decay gate on the generated `ψ_T`, checking that energy above the coarsest grid's Nyquist (k=64) is negligible before trusting the subsampled data.

* V sampler: defined by Fourier coefficients (not grid values) drawn from a spectral envelope, so the same V is realizable at any resolution — this resolution-independence is what makes super-resolution honest. First implementation used a Gaussian-in-k envelope and an irfft-based synthesis; both were replaced after bugs (below). Final: exponential envelope `exp(-k·length_scale)`, confirmed via a densely-sampled scan (not just the 2–3 spot-checks that initially looked fine) to give a genuinely gradual complexity dial rather than a cliff — high-`k(>16)` energy fraction glides smoothly from `55%` (`ls=0.02`) to `0%` (`ls=0.5`) with no discontinuous jump. Default `length_scale=0.2` chosen by reading two competing constraints off that scan together: low high-k fraction (0.3%, learnable at 16 FNO modes) and modest `V_max` (~3.3, CFL-benign), while preserving headroom to go wiggly in both directions for later OOD sweeps.

* `ψ₀` sampler: a Gaussian wavepacket, deliberately built as a periodic (wrapped) image sum rather than a naive truncated Gaussian, chosen explicitly over the cheaper truncated version because a boundary discontinuity has a 1/k spectral tail that would land directly in the high-mode band the response Jacobian is most sensitive to: an artifact that contaminates exactly the quantity the whole project measures is worse than one that contaminates something incidental. `n_images=2` is exact to float64 for the sampled `σ` range (further images suppressed by `exp(-(2π)²/2σ²)`, `~10⁻⁴²`). Verified periodic via a jump-vs-typical-jump comparison at a worst-case seam-centered packet (seam jump 5.0e-2, comfortably below the max ordinary jump 9.7e-2 — a first attempt at this check, naively comparing `ψ[0]` to `ψ[-1]`, falsely flagged a legitimate steep-but-continuous peak as discontinuous).

* Generation driver: samples grid-free parameters → realizes `ψ₀`, `V` at `nx=1024` only → solves once at that resolution → subsamples every 2nd/4th/8th point down to 512/256/128 → renormalizes `∫|ψ|²dφ=1` independently on every grid (including `ψ_T`, decided deliberately rather than left as an inherited-from-evolution approximation, on the principle that TDSE work should treat norm as an enforced invariant everywhere, not an assumption). Storage mirrors the Burgers `.npz` convention but splits `ψ` into real/imaginary keys (`u0_re_128`, `u0_im_128`, etc.) rather than storing complex-native, after explicitly tracing where complex arithmetic is actually needed downstream (the FNO wants real channels; only the forward-metric and J_true genuinely need complex, and both reconstruct trivially with one `re+1j*im` line) — concluding split storage minimizes total conversion points rather than defaulting to either extreme.

* Before the full 2000-sample run, `nt_super` (fine-grid timestep count) was validated rather than assumed: a self-convergence check (nt vs. 2×nt on 5 worst-case draws) gave agreement of `~7×10⁻⁹`, confirming 10,000 steps is sufficient headroom above what CFL strictly requires for these band-limited inputs. A 200-sample test split was generated and inspected before committing to the full 2000-sample train split, in case of an unnoticed pipeline issue.

* **Final result**: the spectral-decay gate came back at `1.02×10⁻¹⁶` worst-case high-k(>64) energy fraction across all 2000 training samples — literally float64 machine epsilon, meaning the resolution ladder is exact: subsampling `nx=1024` down to `nx=128` loses zero information. Post-save verification confirmed the split-complex storage round-tripped correctly and the super-resolution invariant (`u0_128 == u0_1024[::8]`) survived being written to and read back from disk.

### Infrastructural Decisions

|Decision|	Value|	Why|
|--------|------|------|
|Domain	|Ring (periodic), circumference `L=2π, ℏ=m=R=1`	| Integer wavenumbers `k=n`; free-rotor spectrum `E_n=n²/2`; clean `J(k,q)` integer axes|
|Boundary conditions	|Untwisted (`θ=0`) |	Standard FFT is the untwisted transform; twisted → silent phase bugs + off-thesis θ-vacuum physics|
|Base grid|	`nx=128`|	Full J is 128×128, trivially materializable/visualizable|
|Resolution ladder	|128 → 256 → 512 → 1024 (solve at 1024, subsample down)	|Super-res headroom. Solve fine, subsample solution — never solve coarse + interpolate up (fabricates high-k)|
|Solver|	Split-step Strang (`½V, full-K, ½V`), `lax.scan` over timesteps	|2nd-order; scan is mandatory for tractable autodiff through `nt` steps |
|Precision split|	`Solver/data-gen/J_true` in float64; FNO training in float32	|Long unitary evolution is precision-sensitive & differentiated-through; FNO is shallow, tolerant, speed-critical. Boundary sits at the data/model interface.|
|`ψ₀` family|	Wrapped (periodic) Gaussian — image sum | Truncated Gaussian injects `1/k` boundary tail → high-k contamination of exactly the band the response lives in. Exact to float64 with `n_images=2`. |
|`ψ₀` params |	`x0∈[0,2π)`, `σ∈[0.3,0.6]`, `k0∈{-4..4}` integer |	`x0` full ring; `σ/k0` band-limit forward; `k0` integer for single-valuedness on ring|
|V family	|Smooth GRF via exponential spectral envelope `exp(-k·ls)`, realized by explicit Fourier sum	| Exponential = usable gradual complexity dial. Explicit sum = no irfft 1/n normalization to get wrong.|
|V defaults	|`length_scale=0.2, amplitude=1.0, n_coeffs=64`	|`ls=0.2`: high-k(>16) frac ≈0.3% (learnable at 16 modes), `V_max≈3.3` (CFL-benign), OOD headroom both ways|
|`nt_super`|	10000	|CFL-validated: 10k vs 20k agree to ~7e-9 |
|`n_modes` (FNO)	|2,4,8,12,16|	|
|FNO arch|	`d_v=32, n_blocks=4, in=4ch, out=2ch`|	~139,938 params |

### Bugs Encountered and Fixes
* `irfft`'s implicit `1/n` normalization broke the super-resolution property — the same Fourier coefficients realized at `nx=128` vs. `nx=1024` differed by an exact constant factor (confirmed via a ratio probe: flat 0.125 = 128/1024) rather than being the same function at different densities. Fixed by replacing irfft-based synthesis with an explicit Fourier sum (`V(φ)=Σ Re[c_k e^{ikφ}]`), which has no per-resolution normalization convention to get wrong — chosen deliberately after getting burned by the first convention mismatch.
* Gaussian-in-k spectral envelope for V was unusable as a complexity dial — decayed so steeply that high-k energy fraction was indistinguishable from zero at any `length_scale ≥ 0.3`, discovered only by scanning densely. Replaced with an exponential envelope, verified to give a smooth, sweepable dial across the full range.
* `√2` amplitude-width vs. density-width confusion in the Gaussian spreading test (see Gate 3 above) — a genuine physics-bookkeeping slip, not a code bug, caught by the `t=0` sanity principle (nothing evolves at `t=0`, so any `t=0` mismatch is definitionally a measurement error, not a solver error).
* Two mis-specified periodicity/seam-continuity test checks - both initially used naive endpoint-value comparisons that flagged legitimate steep local behavior as a bug; both were fixed by comparing against the typical local variation instead of an absolute threshold — a reusable lesson about what makes a numerical check well-posed, not just a one-off fix.

---
## Forward FNO: Harness, Learnability Gate, and the Norm-Emergence Result
This is the project's go/no-go gate. Before investing in the response machinery, we must establish that a 16-mode FNO can learn the forward map `(ψ₀, V) → ψ_T` at all on this data. Unlike the Burgers precedent, this is not a foregone conclusion: Burgers is dissipative (viscosity destroys high-k content, so a low-mode truncation discards energy that was decaying anyway), whereas the Schrödinger evolution is unitary — there is no viscosity to forgive the truncation, so mode-cutoff can in principle corrupt the forward map, not only the response. A forward failure here would be physics (T too long, energy fled above mode 16), not a bug — and would force a data retune before proceeding. The model itself was already adapted to complex I/O; this is about the training harness and the gate decision.

### Key Results
* **Learnability probe**: 300 training samples, 200 held-out test, 150 epochs, flat LR 3e-4 (no schedule). Test rel-L2 fell 0.998 → 0.104, monotone, still descending at the final epoch (no plateau). The train/test gap (0.059 vs 0.104) is ordinary overfitting on a 300-sample set — the kind more data closes — not a wall. This cleared the top row of the pre-agreed interpretation fork: forward is learnable, GO. Critically, T=0.5 cleared the gate with headroom rather than scraping the edge, which matters downstream: it means energy has not all fled above the mode-16 cutoff, leaving room for the off-diagonal mode-coupling the response operator lives on.

* **Full training run**: All 2000 samples, 500 epochs, exponential-decay LR (init `3e-4`, `×0.5` every 100 epochs, staircase). `Final test rel-L2 = 0.0169 (1.7%)` — a genuinely good operator surrogate. The probe extrapolation held: 300 samples/flat-LR floored at 0.104; full data + schedule reached 0.017. The train/test gap is small (0.0129 vs 0.0169), so the model is neither overfit nor obviously data-starved at N=2000. The curve was flat over the final ~50 epochs, so 500 epochs sufficed.

* **Norm-emergence**: Norm-violation — the mean departure of `∫|ψ_pred|²dφ` from 1 — was logged every epoch but never entered the loss. It nonetheless fell from `~1.0` (untrained) to `0.0057`, a ~6× improvement over the probe's 0.037, purely as a byproduct of forward accuracy. This is the project's central question previewed in miniature: *does physical structure emerge from forward-only training?* Here, partially — forward accuracy alone dragged `ψ_T` to within ~0.6% of the unit sphere. But two features matter for the honest read:
1. It did not reach machine-zero. The surrogate is approximately-but-not-exactly physical.
2. It plateaued in lockstep with rel-L2 around epoch 400, rather than continuing to fall independently.

Norm-violation tracks forward error rather than vanishing on its own. The residual unphysicality is coupled to the residual forward error — they are not independent quantities the network drives to zero separately. 

### Architecture 
* **No normalization (ψ kept raw)**: Burgers normalized inputs by train mean/std; here we deliberately do not. ψ is O(0.4) (unit-norm forces it), V is O(1), the grid is O(1) — all already network-friendly — and the relative-L2 loss is scale-robust regardless. The decisive reason is the norm diagnostic: normalizing ψ would corrupt the unit-norm structure and force a denormalize step before computing `∫|ψ|²dφ`, introducing one more place for a silent constant error. Raw ψ ⇒ the sum is the probability integral, testable directly against 1.0.

* **Complex ψ₀ recombined at the data boundary, split to channels inside the model**: On disk ψ₀ lives as two real arrays (`u0_re, u0_im`); `FNO.__call__` consumes a complex psi0 and splits it via `jnp.real/jnp.imag` on its first line, stacking `[Re ψ₀, Im ψ₀, V, grid]` into 4 input channels. So `load_split` must recombine `u0_re + 1j*u0_im` before the model sees it. (This is the load-bearing line — see Bugs.) 

* **Metric**: Frobenius norm over stacked-real channels = complex L2, with no complex array formed. The model emits `(nx, 2) = [Re ψ_T, Im ψ_T]`; the target stacks identically. `relative_l2` takes `jnp.linalg.norm(..., axis=(-2,-1))`, collapsing both the spatial and Re/Im axes into one per-sample scalar. This handles a single `(nx,2)` sample and a `(B,nx,2)` batch with the same code. The loss stays real-valued, so gradients stay clean.

* **Norm-violation as a passive observable**: This is a thesis commitment, not an oversight. Penalizing norm-violation would answer a different question — "can an FNO be trained to conserve norm?" (trivially yes; that is engineering) — and would destroy the emergence result by making norm-violation a tuned input rather than a measured output. Forward-only training answers the scientific question: "does norm conservation emerge from accuracy alone?" The diagnostic rides in `eval_step` and is logged every epoch as a first-class metric because its trajectory is a result.


## Infrastructure 

All new code lives in `scripts/train.py` (functions reused by every later phase) and `scripts/probe.py` (disposable). Run from repo root as `python -m scripts.train` / `python -m scripts.probe`.

Functions added this phase:

* `load_split(path, nx=128) -> (psi0_complex (N,nx), V (N,nx), uT (N,nx,2))` — recombines split-complex ψ₀; casts data to float32/complex64 at the boundary.
* `make_batches(psi0, V, uT, batch_size, key)` — three-way yield (was two-way in Burgers), JAX-PRNG shuffle for reproducibility.
* `relative_l2(pred, true)` — the stacked-real complex-L2 metric above.
* `norm_violation(pred, nx)` — mean `|∫|ψ|²dφ − 1|`, `dφ=2π/nx`; `nx` passed explicitly (not read inside the jitted closure) so it stays a concrete Python int at trace time.
* `train_step / eval_step` — three-input signature; `eval_step` returns a pair (`rel_l2, norm_violation`).
* `train(cfg, n_train=None, seed=None)` — `n_train` subsetting is baked in now so the (modes × N) grid reuses this driver with no refactor. CLI: `--n_train, --seed`.
* `get_git_commit()` — logs exact code state per run (provenance discipline).

MLflow logging: per-run params include `git_commit, lr_decay_rate, lr_decay_every`, and `t_end` (logged because it is still an open decision). Per-epoch metrics: `train_rel_l2, test_rel_l2, norm_violation`.

Checkpointing: every 50 epochs plus a final. Naming `fno_N{n}_seed{seed}_*` — preserving the artifact-path discipline that has bitten this project before. `fno_N2000_seed0_final.eqx` is a future input, not disposable.

### Bugs, traps, and fixes

* Silent `re/im` recombination: If real-only ψ₀ reaches the model, `jnp.imag(real_array)` returns all zeros with no error — the operation is legal on a real array, just semantically wrong. The model then trains on `[Re ψ₀, 0, V, grid]`, learns some plausible-looking map, and floors out mysteriously with no crash. Same genus as the float32-FFT drift: no exception, just quietly wrong output. Fix: `load_split` recombines to complex; the `∫|ψ₀|²dφ = 1.0` check catches any botch physically (a mangled recombination will not integrate to 1).

* `relative_l2` is global-phase-sensitive, not phase-invariant: The `Frobenius = complex-L2` identity is pure algebra (`|z|² = Re² + Im²`), not a U(1) symmetry. Multiplying ψ by a global phase `e^{iθ}` does change `‖ψ_pred − ψ_true‖`. This is acceptable here only because we match the solver's fixed, deterministic phase convention — no gauge freedom is introduced. 

* x64 leaks into model-internal arrays. `jax_enable_x64` is a global flag. The `.astype(float32)` casts in `load_split` pin the data, but the model builds `grid = jnp.linspace(...)` internally in `FNO.__call__`; under x64 that becomes float64 and silently promotes the entire forward pass (≈2× slower, and it stops matching the validated float32 numbers). No error. Fix/discipline: run training as a fresh `python -m scripts.train process` (x64 off by default), never in an x64-enabled notebook kernel. This exact collision becomes deliberate and central where `J_true` (float64, from the solver) and `J_FNO `(float32, from this model) must be reconciled in one process.

### Math background 

* Why Frobenius-over-channels equals complex L2: For a complex vector `ψ` with components `ψ_i`, `‖ψ‖₂² = Σ_i |ψ_i|² = Σ_i (Re ψ_i² + Im ψ_i²)`. The right-hand side is exactly the sum of squares of every entry of the real `(nx, 2)` array — i.e. its squared Frobenius norm. So taking the Frobenius norm of the stacked-real representation reproduces the complex L2 norm identically, by the definition of complex modulus. No symmetry argument is needed, and none is available: the norm of a difference is phase-sensitive (see Bugs).

* Why `∫|ψ|²dφ = 1` is the right check: On the ring of circumference `L=2π` with `nx` grid points, the measure element is `dφ = 2π/nx`. Unitary evolution conserves the L² norm of ψ exactly, and the stored data enforces `∫|ψ|²dφ = 1` as an invariant. A learned surrogate has no built-in reason to respect this — so the deviation of `∫|ψ_pred|²dφ` from 1 is a direct, interpretable measure of how unphysical the fit is, and (because we kept ψ raw) is computable straight from the network output with no denormalization.

* Forward vs response: The forward map returns only the endpoint `ψ(T)`; the response `∂ψ_T/∂V` is the propagator sandwiched around the perturbation and integrated over the whole path, `−i∫₀ᵀ U(T,t) δV U(t,0) ψ₀ dt`. Two potentials can produce near-identical `ψ(T)` yet quite different responses — the derivative sees the path, the state sees only the endpoint. Here, we validated the forward (1.7%); whether its Jacobian is equally good is the next question, and the reason "forward-good ≠ response-good" is a real possibility rather than a technicality.

---
## Response Machinery
The last stage gave us a trained FNO surrogate for the forward map `(ψ₀, V) → ψ_T` (`rel-L2 = 0.0169, norm-violation = 0.0057`). We now attempt to answer the central question of the project: *does that forward-only-trained FNO also get the response right — the functional derivative `∂ψ_T/∂V` — as an unsupervised byproduct?* 
This stage builds the machinery to compute and compare two Jacobians of the map `V ↦ ψ_T` (with `ψ₀` held fixed): `J_true` (autodiff through the reference solver) and `J_FNO` (identical machinery on the trained surrogate), then reads the difference in Fourier/mode space.

### Precision strategy

* The tension: `J_true` needs float64 (unitary evolution over `nt=1000` steps is precision-sensitive). `J_FNO` comes from a checkpoint trained in float32. Since the two Jacobians get subtracted at the end, they need to share a precision regime.

* The decision, and why: Rather than retraining the FNO in float64 (which would produce a different model — a different optimizer trajectory, different weights, and would silently unlock a project decision that was deliberately fixed), we run both Jacobians in float64 by casting the float32-trained weights up. This works because `float32→float64` is a lossless, bitwise-exact widening — every float32 value has an exact float64 representation (fewer exponent bits, fewer mantissa bits, both strict subsets of float64's range). We verified this empirically before trusting it: round-tripping a batch of float32 values through float64 and back returned bit-identical results, and the widened values differed from the originals by exactly 0.0. So casting the checkpoint's weights up is not an approximation — it's the same function, evaluated with cleaner arithmetic (no float32 round-off accumulating over the forward pass).

* The gate: `load_fno_x64(path, cfg)` in `scripts/compute_response.py`: build an FNO template, pin its leaves to float32 (so dtypes match the checkpoint file before reading — see Bug 2 below for why this pinning is necessary), deserialise, then cast the whole pytree up to float64. The gate: run the cast-up model on the test set and assert forward rel-L2 reproduces 0.0169 and norm-violation reproduces 0.0057. It does, in both precision regimes (float32-native and float64-cast), which is the empirical proof that "same function, cleaner arithmetic" actually holds and not just a comforting assumption.

#### Bugs found and fixed this block

All three are variants of the same root cause: a hardcoded 32-bit dtype colliding with the global jax_enable_x64 flag. Worth internalizing as a pattern — anywhere a dtype is pinned explicitly, it silently overrides the flag.

* Bug 1 — `SpectralConv1d`'s scatter buffer hardcoded `dtype=jnp.complex64`. In `fno.py`, the spectral layer builds `out_ft_full = jnp.zeros((n_ft, ...)`, `dtype=jnp.complex64`) before scattering the truncated-mode output into it. Under x64, the FFT of a float64 input produces complex128, so the scatter was silently downcasting complex128 → complex64 (JAX warned that this will become a hard error in a future release). Fix: derive the buffer's dtype from the actual computation instead of hardcoding it — `dtype=out_ft.dtype`. One-line fix; verified the x64-off path still produces complex64 and reproduces rel-L2 = 0.0169 exactly.

* Bug 2 — `load_split`'s hardcoded `.astype(np.complex64)` / `.astype(np.float32)` collide with the x64 solver. These casts are correct and necessary for the FNO harness (the model trains in float32), but under x64-on they hand the solver's `lax.scan` a complex64 carry (ψ₀) while the scan step's internal arithmetic (kinetic/potential phases, built fresh under x64) produces complex128 — and `lax.scan` requires the carry's dtype to be identical going in and coming out across iterations, because it compiles one step and reuses it. Complex64-in vs complex128-out is a hard type error. The fix is deliberately driver-local, not inside `load_split` or `evolve`. `load_split` is locked and has other callers who need the float32 pins; editing it for one new caller widens the blast radius unnecessarily. Casting inside evolve was also rejected — it's the physics-gated reference solver (norm conservation `5e-14`, autodiff-vs-FD valley bottomed at `2.8e-10`), and inserting a cast into the differentiated path risks interfering with the tangent. Instead, the driver widens the inputs immediately after loading, before they touch the solver: `psi0 = psi0.astype(jnp.complex128), V = V.astype(jnp.float64)`. Lossless, touches nothing upstream, and the direction is principled: always widen the low-precision side to match the high-precision engine, never narrow the engine to match the data. Narrowing the solver would degrade the one quantity (`J_true`) whose entire purpose is to be the trustworthy reference.

> General lesson banked: `lax.scan` is stricter about dtypes than ordinary JAX code. Plain forward passes silently promote (complex64 × complex128 → complex128, no complaint) — which is exactly why the FNO's forward pass didn't trip this bug even though it received the same complex64 ψ₀: it has no scan, so promotion happens invisibly on the first op (the lifting layer) and the model self-heals. The solver's scan carry has no such escape hatch. So a scan carry mismatch is a useful tripwire — it will catch dtype problems that a non-scan forward pass would hide.

### `J_true` (autodiff through the solver)

* What's being differentiated and why it's clean: The map is `V ↦ ψ_T` with ψ₀ fixed. V is real (ℝⁿ), `ψ_T` is complex (ℂⁿ), so this is an ℝⁿ → ℂⁿ map. Differentiating a complex-valued function with respect to a real input has no ambiguity: perturbing V is a one-dimensional choice of direction, so "the derivative" is a single well-defined complex number per output. Wirtinger calculus (the holomorphic/anti-holomorphic split needed for complex inputs) never enters, because it's the input's realness — not the output's — that removes the ambiguity. This is exactly why the response study differentiates w.r.t. V and deliberately avoids differentiating w.r.t. ψ₀ (which is complex, and would require the split).

* `jacfwd` over `jacrev`: `evolve` is a `lax.scan` over `nt=1000` timesteps. Reverse-mode autodiff (`jacrev`, built from VJPs) needs to store the entire forward trajectory to replay backward — memory grows with `nt`. Forward-mode (`jacfwd`, built from JVPs) carries a tangent vector alongside the primal in a single sweep — memory stays flat regardless of `nt`. For a long time-integration, forward-mode is the natural fit.

* Implementation: `jacobian_true(psi0, V, nx, dt, nt)` in `src/responses.py`: closes over `psi0`, calls `jax.jacfwd` on `V ↦ evolve(psi0, V, nx, dt, nt)`.

* Gate — the finite-difference cross-check: `fd_check` picks a random unit direction u in V-space and compares two independent routes to the same directional derivative: the autodiff JVP (`jax.jvp`, one forward sweep) versus a central finite difference (`[f(V+εu) − f(V−εu)] / 2ε`). These share no code path, so agreement is strong evidence of correctness. Result: `rel-err = 2.20e-6`. This is the expected floor for central FD (2nd-order accurate, but subtraction of near-equal complex vectors loses conditioning) — not a red flag, and consistent with the already-established gate that autodiff-through-scan matches FD to 2.8e-10 in gradient-valley testing.

### Full `J_true` and the mode-coupling matrix

* Materializing the full matrix: `jacobian_true` run to completion gives the full 128×128 position-space Jacobian, `J_pos[i,q] = ∂ψ_T(x_i)/∂V(x_q)`. Gated against a single validated JVP (column q=40 compared to a direct `jax.jvp` in direction e₄₀): `rel-err = 8.13e-15`, i.e. machine-epsilon agreement — this is autodiff compared against autodiff computing the exact same quantity two ways, so near-exact agreement is expected and confirms jacfwd isn't doing anything unexpected.

* Rotating to Fourier/mode space: The physically meaningful object is `J(k,q) = ∂ψ̂_k/∂V̂_q` — how perturbing Fourier mode `q` of the potential affects Fourier mode `k` of the final wavefunction. `mode_coupling_matrix(J_pos)` gets this by FFTing the output index (i→k) and inverse-FFTing the input index (q→mode) of the already-materialized position-space matrix — pure linear algebra, no re-differentiation, using the solver's own FFT convention (integer wavenumbers, `fftfreq(d=1/nx)`).

* Physics finding — the response band is centered on `k−q = k₀`, not `k−q = 0`. For a local potential operator, the bare quantum-mechanical matrix element `⟨k|V|q⟩ = V̂(k−q)` is off-diagonal but centered on `k−q=0` (multiplication by V(x) in position space is convolution by V̂ in mode space). Naively, we expected `J(k,q)` to inherit that centering. Instead, the diagnostic — which bins `|J(k,q)|` by the offset `d=k−q` and finds where the profile peaks — found the peak sitting at `k−q = 4` for sample 0, not 0. Rather than assume a bug, we tested the hypothesis that this offset equals the sample's carrier momentum k₀ (a parameter of the initial Gaussian wavepacket family, k₀∈{−4,…,4}). Checked across four samples: peak offset = k₀ exactly, every time (`k₀=4→peak 4, k₀=2→peak 2, k₀=−3→peak −3, k₀=−2→peak −2`). This is real physics, and here's the mechanism: the response is the propagator sandwich:
 ```
δψ_T = -i ∫₀ᵀ U(T,t) · δV · U(t,0) · ψ₀ dt
```
  Reading right to left: `U(t,0)ψ₀` propagates the initial state forward to the moment of perturbation — and since `ψ₀` carries momentum `k₀`, the state arriving there is already centered around `k₀` in mode space. The perturbation `δV` then couples that k₀-centered state to neighboring modes via `V̂(k−q)`; the coupling band is centered on k₀, not 0, because it's coupling a state that's already offset, not an unbiased basis state. The bare-potential matrix element `⟨k|V|q⟩` has no such offset because it doesn't know about ψ₀ — it's a property of V alone. The response does know about ψ₀, and inherits its momentum. This is the concrete, measurable expression of the project's founding intuition: "the derivative sees the path; the state sees only the endpoint." 

Two more numbers from this matrix, both later comparison baselines: `mass within |k−q|≤16 (the FNO's mode budget) = 0.995` — almost all the true response coupling is, in principle, representable within the truncation window; and `max/mean peakedness ratio = 273` — the matrix is sharply structured, not diffuse.

### `J_FNO` 

* The output-shape wrinkle: `evolve` returns a complex `(nx,)` array directly, so differentiating it gives a complex Jacobian for free. `FNO.__call__(psi0, V)` instead returns `(nx, 2)` — real-valued [Re ψ_T, Im ψ_T] channels. To keep the machinery identical, `jacobian_fno` wraps the model in a thin adapter that recombines the two channels inside the differentiated function — `out[:,0] + 1j*out[:,1]` — before applying `jax.jacfwd`. This recombination is safe under autodiff precisely because the input (V) is real: it's the same ℝⁿ→ℂⁿ regime, and the complex repackaging is a linear operation on real derivatives, not a new source of Wirtinger ambiguity.

* Results (`sample 0, k₀=4`):
  1. `J_FNO` band peak at `k−q = 3` (`J_true`'s was `4` — the FNO's band is shifted one mode inward, toward zero, relative to the true carrier-momentum center).
  2. `J_FNO` mass within `|k−q|≤16 = 0.950` (`J_true'`s was `0.995` — less of the FNO's coupling sits within its own truncation window than the true physics would predict).

### Response error and the error map

* The metric: `response_error(J_fno, J_true) = ‖J_fno − J_true‖_F / ‖J_true‖_F` — relative Frobenius error, computed without any phase-alignment step. This omission is deliberate and justified: `relative_l2`/`Frobenius` comparisons are generally global-phase sensitive (a U(1) rotation of a wavefunction changes the metric), but here both Jacobians differentiate the same phase-convention forward — the FNO was trained to match the solver's `ψ_T` exactly, phase included (that's what the 0.0169 forward match means). A derivative inherits the phase frame of the function it differentiates; no new phase freedom is introduced by differentiating. So the comparison is valid as-is. (This assumption would need revisiting only if ever comparing Jacobians from two differently-phased forwards, which doesn't apply here.)

* **Headline result**: `response error = 0.386` (identical in both position-space and mode-space bases — 0.386 vs 0.386 — which is itself a free correctness check: relative Frobenius error is invariant under a unitary rotation, and the fact that both bases agree confirms mode_coupling_matrix's rotation really is unitary). Compared against the forward error of 0.017, this is a ~20× gap between forward fidelity and response fidelity on a single sample — the core "forward-good ≠ response-good" phenomenon, now a measured number rather than a hypothesis.

* The `k−q` resolved error map: Binning `|J_FNO − J_true|` by offset `d=k−q` shows where the error concentrates: peaks at `k−q = 2`, and the error mass above the band center (k₀=4) vs below has ratio 0.42 — i.e. the error is more than twice as concentrated on the low-k side of the band, not the high-k edge.

---
## Response Fidelity vs. Spectral Truncation

We previously established the machinery to compute an FNO's learned response Jacobian `J_FNO = ∂ψ_T/∂V` and compare it against the true response `J_true` (obtained by autodiff through the split-step solver). On a single sample, the canonical M=16 model showed a large gap: `forward relative-L2 error of 0.017` but `response relative-Frobenius error of 0.386`. We now aim to turn that one-sample observation into a properly measured, causally-grounded result — and ultimately to answer the project's founding question: *does an FNO trained only on forward evolution (ψ₀, V) → ψ_T recover the correct response operator as an unsupervised byproduct?*

### The $k_0$ sweep

**Setup**: Response error was computed for all 200 test samples using the canonical checkpoint (`fno_N2000_seed0_final.eqx`, `M=16`), and binned against the wavepacket's carrier momentum `k₀`. The physical logic: we previously showed the true response's mode-coupling band `J(k,q)` is centered on `k−q=k₀`, not `k−q=0` (a real physics finding — the propagator sandwiches the perturbation between two copies of time evolution, and the state arriving at the perturbation carries momentum k₀, so the response inherits that offset). The hypothesis was that larger |k₀| slides this band toward the FNO's fixed |k|≤16 mode cutoff, degrading response fidelity as |k₀| grows.

**Result**: Response error was non-monotonic in `|k₀|` — a symmetric "M" shape, dipping at k₀=0 (0.489), peaking at |k₀|=1 (0.592), then declining monotonically to a minimum at |k₀|=4 (0.428). This falsified both pre-registered hypotheses: not the predicted monotonic rise, and not flat (which would have indicated generic underfitting rather than a truncation effect).

| Abs(k0) |  n |  num   |  den   | rel-err |
|------|----|--------|--------|---------|
|  0   | 25 | 0.2608 | 0.5337 | 0.4887  |
|  1   | 46 | 0.3159 | 0.5336 | 0.5918  |  
|  2   | 33 | 0.2874 | 0.5320 | 0.5402  |
|  3   | 51 | 0.2634 | 0.5218 | 0.5047  |
|  4   | 45 | 0.2131 | 0.4979 | 0.4280  | 

The corresponding graphs are shown below:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/40633995-4a4a-470d-8675-37dd3947990f" width="450">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f8c52263-c370-451b-b0b5-14377a8d4990" width="450">
    </td>
  </tr>
</table>


**Confounds ruled out, so the shape is trusted**:
 - Denominator artifact: response error is a ratio (`‖J_FNO−J_true‖/‖J_true‖`); if `‖J_true‖` grew with `|k₀|` the ratio could fall for reasons unrelated to FNO fidelity. Checked directly: the denominator is flat (`0.534→0.498`, ~7% drift) while the numerator (absolute error) traces the same M-shape. The signal is real, not a normalization artifact.
 - σ (packet width) leakage: wavepacket σ and k₀ are drawn independently but could accidentally correlate in a finite sample. Checked: `corr(|k₀|, σ) = 0.018` — negligible.

> **Why we stopped chasing k₀ mechanistically**: Even at the largest tested |k₀|=4, the true coupling band still keeps 95% of its mass inside the |k|≤16 mode budget. The band never actually reaches the cutoff across the tested range, so k₀ was always a weak proxy for truncation stress — sliding a band that never reaches the wall can't cleanly test what happens when the wall is hit. The M-shape is a real, banked finding, but its mechanism is unexplained and was deliberately not investigated further, in favor of a more direct causal test.

### Truncate-at-inference — the direct causal test

Rather than proxy truncation through k₀, this block manipulates the mode cutoff directly. Method: take the trained M=16 model and zero the spectral weights (`w_real, w_imag`) for modes `M..15` in every FNO block, for `M ∈ {2,4,8,12,16}`, leaving the pointwise-linear branch of each block untouched (architecturally correct — that branch doesn't depend on `n_modes`). This is implemented via `eqx.tree_at` selecting leaves by attribute path (not by value).

Gate before trusting the sweep: `M=16` truncation must be the identity (empty slice, no-op) and exactly reproduce `0.3862` on sample 0. It did, bit-exact. `M=4` moved the same sample to 1.02 — confirming the transform actually bites.

**Result**: response collapses monotonically as modes removed (200 samples)
| M  |  mean  |  sem   |
|----|--------|--------|
|  2 | 1.0373 | 0.0044 |  <- worse than zero-prediction (>1.0)
|  4 | 0.9240 | 0.0071 |
|  8 | 0.6433 | 0.0043 |
| 12 | 0.5562 | 0.0052 |
| 16 | 0.5113 | 0.0056 |  <- == full model; matches k0-sweep grand mean (consistency gate)

These are graphed below:

<img width="720" height="494" alt="image" src="https://github.com/user-attachments/assets/bcd86dd6-ed53-44e7-804e-b4e2f1d53a0f" />

### Retrain arm, and the response floor
* **The missing floor baseline**: Before this block, "`response error = 0.511`" had no interpretable anchor — was that mostly-right or mostly-wrong? Floor baseline: an FNO with the exact training architecture (`M=16`, same channel width, same block count) but never trained — random initialization only, run through the identical response machinery.
 Result: `response error = 1.0000 ± 0.0004`, flat to four decimal places across every |k₀| bin. This is the clean "predicts nothing correlated with the true response" zero-information floor. Its flatness across |k₀| is itself a useful check: it confirms the variation seen in the k₀ sweep is a property of trained models, not an artifact of how the error metric scales with k₀.

* **The founding question**: With the floor established, the canonical M=16 model's response error of 0.511 becomes interpretable: it sits almost exactly halfway between "knows nothing" (1.0) and "matches exactly" (0.0). Forward-only training recovers roughly 49% of the true response operator's structure, unsupervised, as a byproduct of learning `ψ₀,V → ψ_T` alone. Forward accuracy is near-perfect (rel-L2 = 0.017) while response is roughly half-right — a ~30× gap between the two error scales on the identical model. This is genuine, substantial partial emergence, not full emergence and not absence.

* **Retrain arm**: does the model adapt to smaller mode budgets? Four fresh FNOs were trained from scratch at `M ∈ {2,4,8,12}` (seed 0, otherwise identical: same `n_train, epochs`, LR schedule) — the only controlled variable is `n_modes`, so any response difference at matched M is attributable to training with vs clipping to that budget. `M=16` reuses the canonical checkpoint (verified: gap between truncate-arm and retrain-arm at M=16 is exactly 0.0000, since it's the same model). The results of the training are shown below:

| M  | fwd rel-L2 | response err | recovered (1 - err/1.0) |
|----|-----------|--------------|--------------------------|
|  2 |   0.2269  |    0.9873    |   1.3%                   |
|  4 |   0.1018  |    0.5573    |   44.3%                  |
|  8 |   0.0319  |    0.2442    |   75.6%                  |
| 12 |   0.0191  |    0.2252    |   77.5%                  |
| 16 |   0.0170  |    0.5113    |   48.9%                  |

 Forward accuracy improves monotonically with M — expected, more capacity strictly helps the forward task. Response accuracy is non-monotonic, peaking in the interior (`M~8–12`) rather than at the forward-optimal `M=16`. `M=2` and `M=4` are bad on both axes (forward-starved — there isn't yet enough spectral bandwidth to learn the map at all, so response has nothing to build on); from `M=12 to M=16`, forward barely moves (`0.019→0.017`) while response gets substantially worse (`0.225→0.511`).

 Interpretation: `n_modes` tuned purely for forward performance is not the same `n_modes` that's optimal for response fidelity, and this is invisible unless you measure response directly.

<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/ddd8641a-b639-40c5-8df1-eab7098c7753" />


### Clip vs Retrain

|M|	truncate (clipped)	|retrain (trained fresh)|	gap|
|-|-|-|-|
|2|	1.037|	0.987|	0.050|
|4	|0.924|	0.557|	0.367|
|8|	0.643|	0.244|	0.399|
|12|	0.556|	0.225|	0.331|
|16|	0.511|	0.511|	0.000|

At M=8 and M=12, retrained models recover roughly 75–78% of the response versus only ~44–51% for the corresponding clipped model — training within a smaller budget lets the network re-route response-relevant structure into the modes and pointwise branch it actually has, something clipping (which removes weights the model already committed to) cannot do. At M=2 the gap nearly vanishes — both strategies collapse to the floor, consistent with a genuine information-theoretic wall rather than an optimization failure (confirmed independently: forward rel-L2 itself is 0.227 at M=2, so the forward task is starved too, not just the response).

Conclusion: truncate-at-inference substantially overstates how architecturally damaging truncation is. Response information is not rigidly bound to specific modes; a model given the chance to train within a smaller budget recovers most of it.

### Confirming the `M=12` vs `M=16` inversion across seeds
The single most striking number here: `M=12` has better response fidelity than `M=16` at matched forward accuracy, in the single-seed data. This was tested with 3 additional seeds (1,2,3) at both `M=12` and `M=16` (6 more training runs total):

| seed | M=12 response | M=16 response |
|------|---------------|---------------|
|  0   |    0.2252     |    0.5113     |
|  1   |    0.2101     |    0.5257     |
|  2   |    0.2308     |    0.5339     |
|  3   |    0.2024     |    0.4636     |
| mean |    0.2171     |    0.5086     |

 No overlap across 4 seeds each — the worst `M=12` run (`0.231`) still beats the best `M=16` run (`0.464`) by a wide margin, and forward accuracy stays tightly matched across all 8 runs (`M=12: 0.019–0.021; M=16: 0.017–0.019`). This is a robust, decisive effect: **more spectral capacity bought forward headroom the task didn't need, at direct, measurable cost to response fidelity**. This is not *"fewer modes is always better"* — `M=2` and `M=4` are worse on both axes simultaneously (undertrained, not response-optimal). The real, precise claim is that response fidelity is peaked in `n_modes`, worst at both extremes, best at an interior sweet spot (~8–12), and that sweet spot does not coincide with the `n_modes` that's optimal for forward accuracy alone.

> Candidate mechanism: forward-only training constrains only the endpoint `ψ_T`, not the trajectory that reaches it. More spectral modes give the optimizer more ways to hit `ψ_T` accurately without tracing the physically correct propagator-sandwich dynamics that the response derivative (which integrates over the whole path) is sensitive to. Fewer modes tighten that constraint, so training is more often forced onto solutions whose linearization happens to be closer to correct. Untested; would need something like mid-integration trajectory-divergence diagnostics to actually confirm.

### Error Map Localization
An attempt was made to test the project's namesake spatial claim directly — does response error concentrate specifically near the mode cutoff in `(k,q)` space? `J_true`, `J_FNO(M=12)`, and `J_FNO(M=16)` were computed and rotated into mode-coupling space for several samples spanning different k₀, and error maps `|J_FNO − J_true|` were plotted with a candidate reference line at `k−q=±M`.

<img width="903" height="1000" alt="image" src="https://github.com/user-attachments/assets/58374637-331a-43b6-a197-9360d0950af1" />


This was explicitly flagged as inconclusive rather than banked as a result. The reference line's geometry is unresolved: the FNO's internal truncation acts on Fourier mode indices inside the spectral-conv layers, which is not obviously the same axis as k−q in the end-to-end input-output Jacobian's mode-coupling basis — so a diagonal line at k−q=M may simply be testing the wrong thing. What the plot does show unambiguously, without relying on the reference line at all: the M=16 error band is visibly brighter and sharper than the M=12 error band at the same locations, which is a spatial re-confirmation of the aggregate finding but not new information, and not a localization claim. 

---
## References
* **Derivative-informed neural operator acceleration of geometric MCMC for infinite-dimensional Bayesian inverse problems**, Lianghao Cao, Thomas O'Leary-Roseberry, Omar Ghattas. (2024). [arXiv: 2403.08220](https://arxiv.org/abs/2403.08220).
* **Derivative-Informed Operator Learning for Finance: On-the-Fly Greeks, Surfaces, Hedging, and Control**, Miquel Noguer I Alonso. (2026). [arXiv: 2606.05900](https://arxiv.org/abs/2606.05900).
* **Fourier Neural Operators for Learning Dynamics in Quantum Spin Systems**, Freya Shah, Taylor L. Patti, Julius Berner, Bahareh Tolooshams, Jean Kossaifi, Anima Anandkumar. (2024). [arXiv: 2409.03302](https://arxiv.org/abs/2409.03302).
* **The Propagator and the Path Integral**, Robert G. Littlejohn. (2021). [Notes](https://bohr.physics.berkeley.edu/classes/221/notes/pathint.pdf).
* **Introduction to Quantum Mechanics**, David Griffiths. (1995).
* **A particle on a ring or: how I learned to stop worrying and love θ-vacua**, Mohammad Aghaie, Ryosuke Sato. (2026). [arXiv: 2601.18248v1)](https://arxiv.org/abs/2601.18248v1).
* **Physical Chemistry for the Life Sciences**, Julio de Paula, Peter Atkins. (2006).
* **Time Evolution Split Operator Method**, Christina C. Lee. (2021). [Here](https://albi3ro.github.io/M4/Time-Evolution.html).
* **Exploring the Propagator of a Particle in a Box**, S. A. Fulling, K. S. Güntürk. [Here](https://people.tamu.edu/~fulling/box/box.pdf#:~:text=Here%20we%20take%20a%20close,wave%20function.).
* **Path Integral for a Particle in a Box**, A. Hershman. (1994). [Here](https://www.cds.caltech.edu/~marsden/wiki/uploads/projects/geomech/Hershman1994.pdf).
* **Path Integrals and Their Application to Dissipative Quantum Systems**, Gert-Ludwig Ingold. (2002). [arXiv: 0208026](https://arxiv.org/abs/quant-ph/0208026).
* **Spectral Neural Operators**, V. Fanaskov, I. Oseledets. (2022). [arXiv: 2205.10573](https://arxiv.org/abs/2205.10573).
* **Toward a Better Understanding of Fourier Neural Operators: Analysis and Improvement from a Spectral Perspective**, Shaoxiang Qin, Fuyuan Lyu, Wenhui Peng, Dingyang Geng, Ju Wang, Naiping Gao, Xue Liu, Liangzhu Leon Wang. (2024). [arXiv: 2404.07200](https://arxiv.org/abs/2404.07200v1).
* **Fourier Neural Operators Explained: A Practical Perspective**, Valentin Duruisseaux, Jean Kossaifi, Anima Anandkumar. (2024). [arXiv: 2512.01421](https://arxiv.org/abs/2512.01421).
* **Spectral Methods for Partial Differential Equations**, David Chopp. (2008). [Notes](https://people.esam.northwestern.edu/~chopp/course_notes/446-2.pdf).
* **LNO: Laplace Neural Operator for Solving Differential Equations**, Qianying Cao, Somdatta Goswami, George Em Karniadakis. (2023). [arXiv: 2303.10528](https://arxiv.org/abs/2303.10528).
* **A Library for Learning Neural Operators**, Jean Kossaifi, Nikola Kovachki, Zongyi Li, David Pitt, Miguel Liu-Schiaffini, Robert Joseph George, Boris Bonev, Kamyar Azizzadenesheli, Julius Berner, Valentin Duruisseaux, Anima Anandkumar. (2024). [arXiv: 2412.10354](https://arxiv.org/abs/2412.10354).
* **Neural Operator: Learning Maps Between Function Spaces**, Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. (2021). [arXiv: 2108.08481](https://arxiv.org/abs/2108.08481).
* **Redefining Neural Operators in d+1 Dimensions**, Haoze Song, Zhihao Li, Xiaobo Zhang, Zecheng Gan, Zhilu Lai, Wei Wang. (2025). [arXiv: 2505.11766v1](https://arxiv.org/abs/2505.11766v1).
* **Equinox: neural networks in JAX via callable PyTrees and filtered transformations**, Patrick Kidger, Cristian Garcia. (2021). [arXiv: 2111.00254](https://arxiv.org/abs/2111.00254).
* **Neural Operators: FNO and DeepONet**. [Watch here](https://www.youtube.com/watch?v=COEItKEZ-is).
* **Fourier Neural Operator for Parametric Partial Differential Equations**, Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. (2020). [arXiv: 2010.08895](https://arxiv.org/abs/2010.08895).
* **How a Fourier Neural Operator Learns to Solve PDEs — and Where It Falls Short**, Gurpreet Singh Hora, Prakhar Kapoor, Albert Matveev. (2026). [Here](https://www.physicsx.ai/newsroom/how-a-fourier-neural-operator-learns-to-solve-pdes----and-where-it-falls-short).



