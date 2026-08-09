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

### [CONFIRMED] Block 5a — FNO load pristine, s2=0.41 is REAL not corruption
- forward f64, J_fno complex128, rel-L2 0.0169 (locked), pos==kq → machinery sound
- s2: Born 1.04 (non-perturbative) but FNO 0.41 (fine) → Born-hard ≠ FNO-hard
- KILLS the "FNO inherits Born blind spots" prior; points to DECOUPLED branch → 7A headliner
- (n=1; 200-sample Pearson/Spearman in 5b decides)


### [CACHE DEBT] recompute fno_errs[200] + save ALL 200 J_true_kq (7A/7D reuse) — pay JVP once

### Derivation
- (fill block-by-block)
### Gate: closed-form J0 vs jacfwd(evolve) at V=0  → target rel-err ~1e-12
- result: ______
### Science: response_error(J0, J_true) over 200 samples
- Born < 0.587 forward-only? → headline branch: ______
- Born >> 0.587? → response is non-perturbative → learning-it-is-nontrivial branch

## Artifact paths (Phase 7)
- ckpt: checkpoints/fno_N2000_seed0_final.eqx
- data: data/test.npz  (keys: psi0, V, uT, p_k0)
- born out: results/born_J0_<...>.npy   (TBD naming)