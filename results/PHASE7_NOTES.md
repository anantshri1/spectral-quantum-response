# Phase 7 Notes — spectral-quantum-response (paper push)

## Status categories: [CONFIRMED] / [SUGGESTIVE] / [OPEN]

## 7-Born — first-order perturbation-theory anchor
### Conventions (locked from code)
- basis: J(k,q) = IFFT_q[FFT_i[J_pos]], integer k on L=2π, numpy FFT norm
- H0 = k²/2, free prop e^{-ik²T/2}, Strang, T=0.5, dt=5e-4, nt=1000
- Born def: J0 = ∂ψ_T/∂V |_{V=0}  (V-independent; ψ0-dependent)

### [DECIDED] Fork: discrete Born (Option B). Reason: gate to float64, not apples/aubergines.
### Solver facts (from src/solver.py)
- U = P½ · K · (P_full·K)^{nt-1} · P½   (half-kicks merge → full interior kicks)
- kick weights = trapezoidal: w_0 = w_nt = dt/2, interior w_m = dt  (Σ w_m = T)
- kin = exp(-i·½k²·dt), pot_half = exp(-i·V·dt/2), V real-space multiply
- => discrete Born integral is the trapezoid quadrature of the continuum ∫₀ᵀ

### [CONFIRMED] Block 1 — continuum Born
Setup. H = H₀ + V, with H₀ diagonal in the momentum basis, E_k = k²/2. Schrödinger gives ψ(T) = e^{-iH₀T}·ψ_I(T) where the interaction-picture state to first order is the Dyson term:
```
ψ_I(T) ≈ ψ₀ − i ∫₀ᵀ V_I(t) ψ₀ dt,      V_I(t) = e^{iH₀t} V e^{-iH₀t}
```
So ψ_T ≈ e^{-iH₀T} ψ₀  −  i e^{-iH₀T} ∫₀ᵀ V_I(t) dt · ψ₀. The first term is free evolution (V-independent → drops under ∂/∂V); the second is where the response lives.

- J0_cont(k,q) = -i e^{-iE_k T} ψ̂₀(k-q) · ∫₀ᵀ e^{iΔt}dt,  Δ = E_k - E_{k-q}
- band: |J0| peaks at k-q = k0 because ψ̂₀ peaks at carrier k0  (predicts 7A band)
- Δ→0: integral → T (removable 0/0; golden-rule resonant buildup)
- [CODE] Block 3: special-case Δ=0 → S=T, else geometric series NaNs at Δ·dt=2πn

### [CONFIRMED] Block 2 — discrete Born (closed form)
Block 2's job is to turn ∫₀ᵀ e^{iΔt}dt into the solver's actual trapezoidal kick-sum and then reconcile the overall constant against mode_coupling_matrix, so the closed form matches jacfwd(evolve) to float64. I'll own the normalization reconciliation (the 1/nx — the error-prone part I flagged). But the first step is yours, and it's the discrete analog of the Dyson term you just did.

Linearize the discrete propagator to first order in V, at V=0. Start from
```
U = P½ · K · (P_full · K)^{nt−1} · P½,   with  P = exp(-iV·τ)  (τ = dt or dt/2),  K = free step
```
U depends on V only through the P factors. Differentiate w.r.t. V(x_q) and set V=0 (so every P → I, every K-run becomes free evolution U₀). You should get a sum over kick positions:
```
∂ψ_T/∂V(x_q)|₀ = −i Σ_{m=0}^{nt} w_m · U₀(T − t_m) · |x_q⟩⟨x_q| · U₀(t_m) · ψ₀,   t_m = m·dt
```

- J0(k,q) = -(i/nx) e^{-iE_k T} ψ̂₀(k-q) S(Δ),  Δ = E_k - E_{k-q} = qk - q²/2
- S(Δ): Δ=0 → T;  Δ≠0 → dt[½ + ½ r^{nt} + (r - r^{nt})/(1-r)], r = e^{iΔ dt}
- 1/nx forced by mode_coupling IFFT_q normalization (NOT chosen) — this closes the gate
- continuum limit S→∫₀ᵀe^{iΔt}dt recovers Block 1
- ψ̂₀ = fft(psi0); k grid = fftfreq(nx, d=1/nx); E_k = k²/2

### [BUG→FIXED] Block 3 gate: Δ used raw (k-q), must use aliased (Brillouin fold)
- symptom: gate 1.7e-1→1.2e-3 (shrinks with nt), max-err at high-|k|,|q| corner, Δ·dt~O(1)
- 3a band (|J0|) passed — magnitude gather already aliased; only Δ PHASE was raw
- fix: kmq_alias = ((k-q + nx//2) % nx) - nx//2; Δ = ½(k² - kmq_alias²)
- lesson: any k² kinetic phase on the grid must fold to first Brillouin zone

### [CONFIRMED] Block 3 — Born closed form is EXACT
- gate rel-err 7.7e-14 (float64 floor); grows with nt (round-off), not shrinks → correct
- born_jacobian(psi0, nx, dt, nt) in scripts/born.py, (k,q) basis, matches mode_coupling

### [CONFIRMED] Block 4 — Born vs J_true, N=200
- Born mean 0.651, std 0.299, median 0.602, range [0.108, 1.534]
- FNO forward-only floor 0.587 → STATISTICAL TIE (Born beats FNO on 48%)
- headline: forward-only FNO response is PERTURBATION-THEORY-GRADE despite 2000 traj + 139k params
- DINO 0.057 reframed: derivative supervision is what buys the super-perturbative response
- big std ⇒ perturbative-ness is a PER-SAMPLE axis (min 0.11 perturbative, max 1.53 past random floor)
- saved results/born_errs_200.npy
### [OPEN→next] Born_err vs FNO_err per-sample scatter — decides mechanism owner:
-   correlated → FNO inherits Born's non-perturbative blind spots
-   uncorrelated → FNO fails for spectral-truncation reasons → 7A is headliner
- NEEDS: fno_errs[200] aligned to born_errs. results/ has ONLY J_true_*_sample0.npy (no 200-cache)

### [RESOLVED — correctly this time] 0.5113 vs 0.587 = TRAINING PRECISION, not seed, not eval-precision
- 0.587 = M16 λ=0 f64-TRAINED, 3-seed mean (Phase 6). f64-trained weights.
- 0.5113 = M16 seed0, fno_N2000_seed0_final.eqx = f32-TRAINED (Phase 3/5), eval'd under x64 via load_fno_x64 (cast up).
- casting f32-trained weights → f64 ≠ f64-trained model. Different networks. 0.076 gap = the "0.511→0.587 rebase" = the retrain.
- SUPERSEDES earlier notes: NOT seed granularity, NOT renorm. Both prior entries wrong.
- ⚠️ today's fno_errs_200.npy = f32-TRAINED seed0. Mislabeled for the DINO-narrative scatter.
### [DECISION NEEDED] scatter must use f64-TRAINED seed0 (the 0.587 family) to match DINO story
- need: f64-trained M16 λ=0 seed0 checkpoint path + load_fno_f64_ckpt (no cast). Does it exist on disk?

### [CONFIRMED] Block 5 RESULT (ignore all the provenance-debug churn above)
Two clean findings on the CORRECT net (f64-trained dino λ0 M16, seed0, N=200):
1. FNO response ≈ Born: FNO 0.5787 vs Born 0.6506 mean. Forward-only training ~ matches
   first-order PT (0 data, closed form). → DINO 0.057 is what actually clears this floor.
2. Born_err ⊥ FNO_err: Pearson 0.107 (p=.13), Spearman 0.097 (p=.17) → UNCORRELATED.
   FNO fails on DIFFERENT samples than PT does → failure axis is NOT perturbativeness
   → it's spectral truncation → 7A (off-band mode mass) is the mechanism headliner.
   Born = the control that RULES OUT the physics ("non-perturbative") explanation.
- claim discipline: "uncorrelated," NOT "negatively correlated" (r~0.1, n.s.)
- figure: Born_err vs FNO_err scatter (expect blob, add r on plot)
- canonical arrays: born_errs_200.npy, fno_errs_200_f64trained.npy, Jtrue_kq_200.npy

### [AUDIT CLOSED] Born baseline fully verified (survived 5 provenance rounds)
- 0.5787 (our f64 run) == 0.5787 (Phase6 logged λ0 seed0) BIT-EXACT → FNO axis correct, floor provenance nailed
- 3-seed 0.587 spread 0.017; seed0 +0.008 consistent
- machinery unchanged from Phase5: reproduced s0=0.3862 AND M16 grand-mean 0.5113 bit-exact (f32 net)
- Born gated 7.7e-14; alignment by construction; checkpoint semantics = forward-only ✓
- RESULT (manuscript): FNO marginally beats PT on mean (0.579 vs 0.651) BUT per-sample uncorrelated
  (Pearson 0.107, Spearman 0.097, n.s.) → FNO ≠ PT, orthogonal error geometry → failure axis is
  truncation, not perturbativeness → Born is the control ruling out the physics explanation → 7A headliner
- canonical: born_errs_200.npy, fno_errs_200_f64trained.npy, Jtrue_kq_200.npy, born_vs_fno_scatter.png

### [CONFIRMED] 7A Block 0 — FNO error decoupled from perturbativeness (flatness figure)
- median-Born split (med=0.602, 100/100): Born swing +0.463, FNO swing +0.016 (29× less sensitive)
- KILLS "FNO = PT + corrections": FNO 0.571 > Born 0.419 on PERTURBATIVE samples (worse where PT is ~exact)
- FNO beats Born on non-pert samples ONLY because Born diverges (0.88→random wall), not because FNO models them
- correct claim: FNO has a perturbativeness-INDEPENDENT error floor (~0.58) set by representation, not physics
- do NOT claim "captures non-pert effects" (floor-limited, not physics-capturing); θ-vacua = off-thesis (untwisted BC), no evidence here
- figure: results/flatness_bins.png

### [CONFIRMED] 7A Block 1 — per-sample band metric (Option B, symmetric around k0)
- band = smallest w s.t. |d-k0|<=w captures 95% of |J_true| mass; d = k-q folded, OFFSET grid
- gate: J_true mass within |d|<=16 = 0.995 (reproduces Phase-4, cache+convention verified)
- self-consistency: J_true own off-band mass = 0.046 ≈ 0.05 ✓
- sample-0: k0=4, w=8 → band well inside M=16 budget → FNO off-band leak = pathology, not capacity wall
- symmetric window chosen: Born predicts near-symmetric band (ψ̂₀(k-q))
- fns in scripts/offband.py: band_width_for_sample, offband_mass_fraction, OFFSET

### [INCONCLUSIVE — pre-registered prediction NOT met] 7A Block 2 off-band MASS
- predicted: off-band mass liability at λ=0 (M16>M12), asset at λ>0. NOT observed.
- observed: λ=0 M16<M12 but OVERLAP (n.s.); λ=0.1 M16>M12 (only liability row); λ=1,10 M16<M12 asset
- KEY TENSION: λ=0 M16 response ERROR ≫ M12 (0.587 vs 0.262) but off-band MASS ≈ equal (0.140 vs 0.160)
  ⇒ off-band MASS FRACTION is not the variable carrying the sign flip
- λ-collapse DOES hold cleanly: M16 mass 0.140→0.094 as λ 0→10 (DINO suppresses leak) — that part survives
- NEXT: recompute with off-band ERROR mass ‖(Jfno-Jtrue)_offband‖/‖Jfno-Jtrue‖ (phase-aware).
  - if it flips → mechanism right, measured wrong quantity (magnitude vs error)
  - if it also doesn't → off-band story FALSIFIED, liability is IN-BAND → different (still publishable) result
- raw grid saved results/offband_grid.npz (26 ckpts, don't recompute)

### [FALSIFIED] 7A off-band hallucination mechanism — liability is IN-BAND, not off-band
- Table B, λ=0, 3-seed: M16 ‖e_in‖ 0.305 vs M12 0.133 (2.3×); ‖e_off‖ 0.041 vs 0.032 (≈equal)
- ⇒ M16's λ=0 response penalty (0.262→0.587) lives INSIDE the response band, not hallucinated outside
- Phase-4 sample-0 "off-band hallucination" does NOT generalize (26 ckpts, 3 seeds — well powered)
- off-band MASS (2b) and off-band ERROR FRACTION (2c-A) both fail to carry the sign flip
- Table A λ>0 "flip" = artifact of fraction denominator (in-band error changing), NOT mechanistic — do not use
- SURVIVES: λ-collapse of off-band mass (0.140→0.094) as minor sub-result; DINO tidies off-band leak
- REFRAMED (candidate, UNTESTED across λ): liability = unsupervised capacity corrupts IN-BAND coupling
- [OPEN] need absolute in/off ERROR split across ALL λ (only ran λ=0) to test reframed mechanism +
  whether in-band penalty flips to asset at λ>0 (mirroring the response-error sign flip)
- artifacts: results/offband_errmass_grid.npz (has off_abs/in_abs for full grid ALREADY — see below)

### [CONFIRMED] 7A mechanism — sign flip is IN-BAND (3-seed, non-overlapping)
- Q1 DINO kills in-band error: M16 ‖e_in‖ 0.305→0.048→0.034→0.027 (λ 0→0.1→1→10), 6.4× collapse
- Q2 in-band sign flip: (M16−M12) ‖e_in‖ = +0.172 (λ0, LIABILITY) → −0.019/−0.009/−0.005 (λ>0, ASSET)
  → mirrors the Phase-6 response-error flip (0.262↔0.587), now LOCATED in-band
- off-band is NOT it: (M16−M12) ‖e_off‖ = +0.009 at λ0, ~20× smaller than in-band → liability ~95% in-band
- MECHANISM (measured): unsupervised extra capacity corrupts IN-BAND coupling; DINO repairs it in-band;
  the flip lives inside the response band, not in hallucinated off-band coupling
- CAVEAT (open): M12 vs M16 = separately trained nets. "M16 trains to worse in-band solution" is solid.
  "modes 12–15 specifically cause it" needs truncate-at-inference (zero M16's modes 12-15, hold weights) — TODO
- prior "off-band hallucination" mechanism FALSIFIED and removed; this replaces it
- source: results/offband_errmass_grid.npz (in_abs/off_abs, full grid, cached — no recompute)

### [CONFIRMED — final mechanism] 7A: in-band, low-mode, capacity/optimization — NOT off-band, NOT high-modes
- 2e truncate-at-inference: zeroing M16 modes 12-15 does NOT reduce in-band err (0.300→0.324, slightly worse)
  → liability is NOT modes 12-15; it's in the SHARED low modes (0-11)
- gates: truncate-to-16 no-op 0.300==0.300; e_full 0.300 ≈ 2d's 0.305 ✓
- CORRECTED mechanism: unsupervised excess capacity → optimizer finds WORSE low-mode in-band coupling
  than the smaller model; DINO recovers it (in-band err 0.305→0.048 at λ0.1); flip = in-band, low-mode
- "extra modes corrupt/hallucinate coupling" FULLY RETRACTED (off-band 2c + high-mode 2e both falsify)
- open Q (→ manuscript limitations): WHY does capacity find worse in-band soln unsupervised?
  (implicit-bias / under-constrained low modes when endpoint-only supervised) — beyond this testbed
### [LESSON banked] mass vs error: |J| mass and J-error point opposite ways; normalized fractions hide
  absolute damage (off-band FRACTION fell for the WORSE model). Always: absolute error, split by region, before ratios.
### [PAPER] off-band = ruled-out alternative, not mechanism: "one high-k0 sample suggested off-band
  hallucination; 200 samples/3 seeds → 95% in-band, falsified." Strengthens the real claim.
  
  Forward-only training leaves the response operator under-determined: the endpoint loss admits many forward-equivalent solutions with differing Jacobians, and the optimizer's seed selects among them. The signature is a decoupling — at λ=0, M=12 forward error is seed-pinned (spread 9×10⁻⁴) while its response error scatters ~25× more (spread 0.023). A derivative term breaks the degeneracy: any λ≥0.1 collapses the response spread to ≤0.004 at both M=12 and M=16. This is the variance axis, and it is fully owned by supervision.

Separately, capacity displaces the mean: M=16 forward-only reaches a worse response (0.587 vs 0.262) than M=12. Notably this is not variance inflation — M=16's seed-spread (0.017) is tighter than M=12's (0.023). More modes land more consistently on a worse linearization. Why the larger model's implicit bias favors a worse in-band coupling is not resolved by this testbed and remains a limitation.
