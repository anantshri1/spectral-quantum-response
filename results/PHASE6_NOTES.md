# Phase 6 — DINO response-aware training arm

## Question
Phase 5: forward-only training recovers ~49% of the true response at M=16
(response err 0.511 vs 1.000 floor). Can an explicit derivative-matching loss
recover the rest, and what does it cost forward accuracy? Deliverable: a Pareto
curve (response fidelity vs forward accuracy) as DINO weight λ sweeps.

Headline comparison: plot the M=12 forward-only point (fwd 0.019, resp 0.217)
ON the M=16+DINO frontier. Does explicit supervision at full capacity beat
Phase-5's implicit capacity-restriction, tie it, or break the tradeoff outright?

## Design (LOCKED)
- Loss: L = rel_L2(psi_T) + lambda * ||J_fno.v - J_true.v||^2 / C_norm
- C_norm: global mean of ||J_true.v||^2 over all precomputed (sample, dir) pairs.
- Directions v: isotropic Gaussian, unit-normalized (unbiased for Frobenius err).
- Precompute K directions/sample offline; step picks one precomputed dir/sample.
- JVP: jax.jvp forward-mode (V real -> no Wirtinger); J_true.v via solver JVP,
  J_fno.v via FNO JVP. Nested reverse-over-forward for grad_theta.
- lambda schedule: lambda_at(epoch), fields dino_lambda + dino_warmup_epochs
  (default 0 = constant). Warmup held FIXED across sweep when enabled.
- Dedicated dino_key, split separately; forward-training key stream untouched.
- Metric: 0.511-style full-J response err on 200 TEST samples (dir-independent).
- Norm-violation: passive log only, never in DINO loss.
- Primary arm M=16 (most headroom). M=12 secondary (does DINO saturate?).

## Gates
- Block 1: precomputed J_true.v == jacobian_true(...) @ v to ~1e-10.
- Block 2: J_fno.v == jacobian_fno(...) @ v to ~1e-6; nested-autodiff test passes.
- Block 3 (HARD): lambda=0 reproduces fwd rel-L2=0.0169, response=0.511 bit-exact.

## Confound watch
- High-lambda: monitor fwd rel-L2 AND ||J_fno||_F — distinguish real J-matching
  from FNO collapsing to a degenerate low-norm Jacobian that games rel error.
- Overfitting to training directions: test metric uses full J, independent v's.

## Block log
- [ ] Block 0: branch + notes skeleton
- [ ] Block 1: offline precompute (V, {v_k}, {J_true.v_k}) + C_norm
- [ ] Block 2: jvp_fno primitive + response residual, standalone
- [ ] Block 3: DINO loss + train step + lambda schedule; lambda=0 gate
- [ ] Block 4: lambda sweep, Pareto curve, MLflow, per-lambda checkpoints
- [ ] Block 5: M=12 arm (optional)
- [ ] Block 6: norm-penalty side arm (optional, separate knob)

## Results
TBD

## Artifacts
- checkpoints/fno_N2000_seed0_M16_dino{lambda}_final.eqx (naming TBD Block 4)
- results/dino_precompute.npz, results/dino_pareto.csv, results/dino_pareto.png

## Block 1 — precompute (DONE)
- results/dino_precompute.npz: V (2000,128), v (2000,8,128) f64,
  Jv_true (2000,8,128) c128, k0 (2000,), C_norm, K=8, dir_seed=0.
- Gate: precomputed J·v vs jacobian_true @ v = 7e-15 (f64 floor). Same map, exact.
- C_norm = 2.1495e-3. SMALL and EXPECTED: isotropic unit v is mostly high-k;
  response band sits near k-q=k0 (low order, Phase 4), so random dirs mostly
  miss response support => small ||J·v||. This is WHY normalization is a global
  constant not per-direction (per-dir would divide by ~0 on high-k dirs).
- Wall: 19.3s, 0.010s/sample. Precompute cost negligible.

## Block 2 — FNO JVP primitive (DONE)
- jvp_fno(model, psi0, V, v) -> (nx,) complex. Single forward-mode pass,
  same map as jacobian_fno (f(V)=model recombined to complex, jvp not jacfwd).
- Gate 1 (value): jvp_fno vs jacobian_fno @ v = 1.5e-15 (f64 floor). Widened
  weights are exact f64 => Jacobian is clean f64, no float32-origin penalty.
- Gate 2 (nested autodiff): grad_theta of ||J_fno·v - target||^2 via
  eqx.filter_grad. Finite, nonzero, AD vs FD on w_real[0,0,0] = 3.4e-9
  (FD floor, not AD). Reverse-over-forward through complex-recombined SPECTRAL
  weights is CORRECT. This was the top technical risk; now dead.
- scripts/check_jvp_fno.py (scratch gate, not in src). tree_at in its FD helper
  selects leaf by PATH not value (safe; contrast bug #3).

## PRECISION RE-BASELINE (Block 3 gate resolution)
Phase 3/5 trained the FORWARD pass in f32 (load_split pins f32, train.py never
widens). Phase 6 trains in f64 (response JVP requires it). This is a genuine
precision difference, NOT a bug — confirmed by running train.py's frozen f32
path, which reproduces 0.0169/0.0057 EXACTLY.

Consequence: Phase 6 is a self-consistent f64 study with its OWN floor. Direct
numerical chaining to Phase 5's f32 numbers is broken by design; the SCIENCE
(does DINO recover response, at what forward cost) is unaffected.

Driver validation: measure_floor.py on f32-canonical gives 0.5113 (== Phase-5
0.511, SEM 0.0056 == Phase-5). Driver trustworthy.

f64 forward-only floor (lam=0, M=16, seed 0):
  forward rel-L2 : 0.0192  (f32 was 0.0169)
  response err   : 0.5787  (f32 was 0.5113) — SEM 0.0053
  => recovers 42.1% of response vs 1.000 floor (f32 recovered 48.9%)
Both metrics degrade under f64: response is MORE precision-sensitive than
forward (Δfwd 0.002 -> Δresp 0.067). Consistent with the Phase-5 mechanism
framing (less implicit regularization -> worse response) via a precision axis.

ALL Phase-6 comparisons use the f64 floor 0.5787, NOT 0.5113.

TODO before money plot:
- [ ] re-confirm untrained floor in f64 (denominator for % recovered; expect ~1.0)
- [ ] re-baseline M=12 forward-only in f64 (for the M=12-vs-frontier comparison)

## Checkpoint loader convention (LOCKED)
- f32-origin checkpoints (Phase 3/4/5): load_fno_x64 (pins f32, casts up)
- f64-origin checkpoints (Phase 6 DINO): load_fno_f64_ckpt (direct f64 deser)
  in tests/measure_floor.py — do NOT use load_fno_x64 on f64 checkpoints.


## Block 4 probe — lambda=1 (warmup 50), M=16 seed 0 — DECISIVE
f64 forward-only floor: fwd 0.0192, response 0.5787.
lambda=1 result:        fwd 0.0148, response 0.0737 (SEM 0.0021, min/max 0.037/0.182)

=> response error cut 8x (0.579 -> 0.074), recovering 92.6% of the true response
   operator (vs 42% forward-only), at NO forward cost — forward IMPROVED
   (0.0192 -> 0.0148, better than even the f32 canonical 0.0169).

Confounds ruled out:
- Degenerate-J gaming: impossible. Rel-Frobenius penalizes small ||J_fno|| ->
  error 1.0, not 0.07. 0.074 means genuine 93% operator agreement.
- Direction overfitting: NO. Training saw 8 single-dir JVPs/sample; measure_floor
  scores the FULL 128x128 Jacobian, direction-independent. Generalizes from 8
  sampled dirs to full operator.
- Whole distribution moved (worst sample 0.182 << 0.58 floor), not a few outliers.

REFRAMES the sweep: expected a Pareto frontier (response down, forward up).
Instead lambda=1 improves BOTH. Sweep now characterizes the effect (min effective
lambda? does forward ever pay? where's the knee?), not hopes for it.

## Block 4 — lambda sweep (M=16, seed 0, warmup 50) — SINGLE SEED
| lambda | forward | response | recovered |
|--------|---------|----------|-----------|
| 0      | 0.0192  | 0.5787   | 42.1%     |
| 0.1    | 0.0173  | 0.1032   | 89.7%     |
| 0.3    | 0.0161  | 0.0869   | 91.3%     |
| 1      | 0.0148  | 0.0737   | 92.6%     |
| 3      | 0.0140  | 0.0645   | 93.6%     |
| 10     | 0.0137  | 0.0568   | 94.3%     |

SHAPE (single seed, needs replication):
- NO Pareto tradeoff. Forward improves monotonically WITH lambda (0.019->0.014),
  response improves monotonically too. Derivative-matching acts as a pure
  regularizer helping BOTH objectives across 100x lambda span.
- Response knee is BELOW 0.1: first whiff of lambda cuts response 5.6x
  (0.579->0.103). lambda=0.1 already recovers 90%. lambda 0.1->10 (100x) buys
  only 4 more points (0.103->0.057). Cheap and saturating.
- Headline: response info almost fully recoverable, nearly for free, at NO
  forward cost. Sharper than the expected frontier.

CAVEAT: single seed per point. High-lambda forward (0.0148/0.0140/0.0137) may be
plateau not monotone — needs SEM. lambda=0.1 forward (0.0173) is the one point
breaking smoothness. Phase-5 discipline: nothing "confirmed" until multi-seed.

NEXT: seeds 1,2 at lambda in {0, 0.1, 1, 10} to confirm shape.

## Block 4 — lambda sweep, 3 seeds (M=16, warmup 50) — CONFIRMED
Response err (mean of seeds 0,1,2; across-seed spread in parens):
| lambda | forward | response | recovered | resp spread |
|--------|---------|----------|-----------|-------------|
| 0      | 0.0187  | 0.587    | 41.3%     | 0.017       |
| 0.1    | 0.0168  | 0.103    | 89.7%     | 0.004       |
| 1      | 0.0144  | 0.0725   | 92.8%     | 0.003       |
| 10     | 0.0136  | 0.0567   | 94.3%     | 0.004       |
(lambda 0.3, 3 available at seed 0 only: resp 0.087, 0.065)

CONFIRMED (3 seeds, no overlap — gaps between lambda levels >> seed spread):
- NO Pareto tradeoff. Forward improves 0->1 then PLATEAUS (0.0187->0.0144->
  0.0136; the 1->10 move 0.0008 is within seed noise ~0.0005). Never degrades.
  Derivative-matching is a pure regularizer aiding BOTH objectives.
- Response knee BELOW 0.1: first whiff cuts response 5.7x (0.587->0.103),
  identical across seeds. lambda=0.1 recovers ~90%. Saturates above: 0.1->10
  (100x lambda) buys only 0.103->0.057.
- HEADLINE (confirmed): forward-only recovers ~41% of true response; lambda=0.1
  recovers ~90% at IMPROVED forward accuracy. Response info almost fully
  recoverable, nearly free, NO forward cost. Sharper than a frontier.

Correction to single-seed read: high-lambda forward is PLATEAU not monotone.
lambda=0.1 single-seed wobble (0.0173) was mild seed-0 noise; 3-seed mean 0.0168.

## f64 baselines — COMPLETE (denominator + M=12 comparison point)
- Untrained f64 floor: 1.0002 +/- 0.0000 (min/max 0.998/1.002). Denominator
  confirmed ~1.0 in f64 as expected (no structure for precision to perturb).
- M=12 forward-only f64 (lam=0, seed 0): forward 0.0212, response 0.2455
  (SEM 0.0027). f32 was 0.217 -> +13% under f64, SAME shift fraction as M=16
  (0.511->0.579). Precision effect uniform across M, not confounding the compare.
  CAVEAT: single seed. Phase-5 M=12 was 4-seed (0.217 f32). Seed 1,2 TODO
  before publication for rigor parity with the 3-seed DINO points.

## MONEY-PLOT PUNCHLINE (locked data)
Explicit DINO at M=16 DOMINATES Phase-5's implicit M=12 capacity-restriction
on BOTH axes:
  M=12 forward-only : forward 0.0212, response 0.2455 (75% recovered)
  M=16 + DINO l=0.1 : forward 0.0168, response 0.103  (90%) -- better on both
  M=16 + DINO l=10  : forward 0.0136, response 0.0567 (94%) -- ~4.3x better resp
=> Don't shrink the model to buy response (Phase 5); keep full capacity + a cheap
   derivative term -> MORE response than shrinking bought, at BETTER forward.
   Two-phase arc: implicit reg helps (P5), explicit supervision helps more & free (P6).

## M=12 forward-only f64 — 3 seeds (comparison point complete)
seed 0: 0.2455 | seed 1: 0.2951 | seed 2: 0.2458 -> mean 0.262, spread 0.023
forward: 0.0212/0.0209/0.0203 -> mean 0.0208

FINDING (minor but real): M=12 forward-only response is ~6x NOISIER across seeds
(spread 0.023) than any DINO point (spread ~0.004). Explicit derivative
supervision STABILIZES response across initializations, not just improves the
mean. Consistent with the mechanism: forward-only response depends on which
forward-equivalent solution the optimizer finds (P5 framing); DINO pins it.

Domination claim STRENGTHENED: M=12's BEST seed (0.2455) is still >2x worse
than DINO l=0.1 (0.103). No M=12 seed approaches the DINO curve. Holds with
3-seed uncertainty on both sides.

## M=12 DINO arm (seed 0) — closes the Phase 5->6 arc
| lambda | M=12 resp | M=16 resp | M=12 fwd | M=16 fwd |
|--------|-----------|-----------|----------|----------|
| 0      | 0.246     | 0.579     | 0.0212   | 0.0192   |
| 0.1    | 0.137     | 0.103     | 0.0198   | 0.0173   |
| 1      | 0.093     | 0.074     | 0.0168   | 0.0148   |
| 10     | 0.068     | 0.057     | 0.0154   | 0.0137   |

- DINO works at M=12 too: response collapses 0.246->0.068, same shape as M=16.
  Recoverable response structure is largely expressible within 12 modes.
- Consistent ~0.01 gap: M=12+DINO tracks M=16+DINO in PARALLEL, offset above.
  Modes 12-15 carry a little response structure, exploitable only when supervised.

KEY (arc-closing): modes 12-15 FLIP from liability to asset with supervision.
  forward-only (l=0): M12->M16 makes response WORSE (0.246->0.579) -- extra
    modes = room for wrong-response solutions (Phase 5).
  supervised (l=10):  M12->M16 makes response BETTER (0.068->0.057) -- DINO
    uses the same modes for finer response structure.
  Same modes, opposite sign, sign set by the DINO term. Reconciles P5 and P6:
  two regimes of the same capacity, separated by supervision.

SINGLE SEED — needs seeds 1,2 at M=12 lambda in {0.1,1,10} to confirm the
parallel-offset shape (the ~0.01 gap is small enough that seed noise could
blur it; M=12 forward-only was the noisiest baseline at spread 0.023).

## M=12 DINO arm — 3 SEEDS, sign-flip CONFIRMED
M=12 DINO response (mean±spread, seeds 0,1,2):
  l=0.1: 0.137±0.001 | l=1: 0.091±0.003 | l=10: 0.067±0.001
Seed variance COLLAPSES under DINO at M=12 too (forward-only spread 0.023 ->
DINO spread 0.001-0.003). Stabilization is cross-capacity, not an M=16 accident.

SIGN FLIP CONFIRMED (3-seed, no overlap) — M12->M16 response change:
  l=0 : +0.325 (0.262->0.587) WORSE — extra modes = liability (Phase 5)
  l=0.1: -0.034 (0.137->0.103) BETTER — asset
  l=1 : -0.019 (0.091->0.073) BETTER
  l=10: -0.010 (0.067->0.057) BETTER (tightest; 0.061 vs 0.066, still no overlap)
Same modes 12-15, opposite sign, sign set by supervision. Reconciles P5+P6:
capacity's effect on response fidelity is sign-flipped by the DINO term.