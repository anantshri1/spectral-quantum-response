# Phase 5 — k0 sweep (response error vs carrier momentum)

## Setup
- 200 test samples, response error = rel-Frobenius ||J_fno - J_true|| / ||J_true||
- Checkpoint: fno_N2000_seed0_final.eqx (fwd rel-L2 0.0169, norm-viol 0.0057)
- Both Jacobians in x64; sample-0 err = 0.3862 reproduces Phase-4 value exactly.
- Sweep: results/response_sweep.csv (cols: idx, k0, response_err, num, den, seconds)

## Result: response error is NON-MONOTONIC in |k0|
Folded to |k0| (mean +/- SEM):
| |k0| |  n |  num   |  den   | rel-err |
|------|----|--------|--------|---------|
|  0   | 25 | 0.2608 | 0.5337 | 0.4887  |
|  1   | 46 | 0.3159 | 0.5336 | 0.5918  |  <- peak
|  2   | 33 | 0.2874 | 0.5320 | 0.5402  |
|  3   | 51 | 0.2634 | 0.5218 | 0.5047  |
|  4   | 45 | 0.2131 | 0.4979 | 0.4280  |  <- lowest
Shape: dip at 0, peak at |k0|=1, monotonic decline 1->4. Signed plot is
symmetric about 0 (M-shape).

## What this falsifies
- Truncation-via-k0 sub-prediction (monotonic RISE with |k0|): DEAD.
  Predicted error grows as band slides toward |k|<=16 cutoff. Got rise-then-fall.
- Generic-underfitting (FLAT error): also inconsistent with the structure.
- Neither pre-registered hypothesis predicts a peak at |k0|=1.

## Confounds ruled out
- Denominator artifact: NO. ||J_true|| flat across |k0| (0.534->0.498, ~7%).
  Signal is in the numerator (absolute error), which traces the same M.
- sigma (packet width) leakage: NO. corr(|k0|, sigma) = 0.018.

## Interpretation (UNCONFIRMED - no mechanism yet)
k0 was a PROXY for truncation stress, and a poor one: even at k0=4 the coupling
band keeps 0.95 mass inside |k|<=16 (Phase 4), so the band never reaches the
cutoff across the +/-4 range. The M-shape is real but likely reports something
other than truncation. Mechanism deliberately not claimed.

## Consequence
Pivot to n_modes sweep as the DIRECT causal test of truncation (moves the
cutoff itself rather than sliding the band toward a fixed cutoff).

## Artifacts
- results/response_sweep.csv
- results/response_vs_k0.png       (signed - headline)
- results/response_vs_absk0.png    (folded |k0|)

## Block 4 — truncate-at-inference arm

Zero spectral weights for modes M..15 in every FNO block (pointwise branch
intact). Helper: scripts/sweep_truncate.py truncate_model(). Identity gate:
M=16 reproduces full-model 0.3862 bit-exact. J_true computed once per sample,
reused across all M.

### Result: response collapses monotonically as modes removed (200 samples)
| M  |  mean  |  sem   |
|----|--------|--------|
|  2 | 1.0373 | 0.0044 |  <- worse than zero-prediction (>1.0)
|  4 | 0.9240 | 0.0071 |
|  8 | 0.6433 | 0.0043 |
| 12 | 0.5562 | 0.0052 |
| 16 | 0.5113 | 0.0056 |  <- == full model; matches k0-sweep grand mean (consistency gate)

Bulk of response info in modes ~4-12. 16->12 near-saturated (+0.045);
4->2 falls past the zero baseline. DIRECT causal truncation evidence:
clipping modes moves sample 0 by ~0.7 vs ~0.1 for the full k0 range.

### SCOPE (do not overclaim)
This is truncate-AT-INFERENCE: a model trained on 16 modes, clipped. Shows
modes 4-15 carry response info the trained model DEPENDS on. Does NOT show a
model TRAINED at low M is equally bad — that's the retrain arm. The clip-vs-
retrain gap is the actual payload.

### Suggestive interaction (UNCONFIRMED - no per-bin error bars yet)
Per-|k0| curves fan + REORDER: |k0|=4 has best full-model response but
STEEPEST truncation decay (best at M=16 -> among worst at M=2); |k0|=0
shallowest. Reading: high-k0 response info sits in higher modes -> more
truncation-vulnerable. This is the Phase-4 k0-truncation intuition reappearing
in the SENSITIVITY rather than the level. Needs per-|k0| SEMs at M=2,4 before
it's more than suggestive. Deferred.

### Artifacts
- results/truncate_sweep.csv  (idx, k0, M, response_err)
- results/response_vs_modes_truncate.png

## Block 5 — retrain arm + the founding question

### Setup
4 fresh FNOs trained from scratch, n_modes in {2,4,8,12}, all else identical
to Phase-3 (seed=0, n_train=2000, n_epochs=500, same lr schedule). M=16 point
reuses the canonical Phase-3/4 checkpoint (fno_N2000_seed0_final.eqx) rather
than retraining, by construction — verified gap(M=16)=0.0000 between the
truncate and retrain sweeps (same model, same numbers).

Response floor baseline: untrained FNO, same architecture (n_modes=16),
random init, never trained. response_err = 1.0000 +/- 0.0004, FLAT across
all |k0| bins (no k0-dependence in the floor itself). This is the "predicts
nothing correlated with J_true" zero-information anchor.

### Forward + response vs M (retrain arm, seed 0)
| M  | fwd rel-L2 | response err | recovered (1 - err/1.0) |
|----|-----------|--------------|--------------------------|
|  2 |   0.2269  |    0.9873    |   1.3%                   |
|  4 |   0.1018  |    0.5573    |   44.3%                  |
|  8 |   0.0319  |    0.2442    |   75.6%                  |
| 12 |   0.0191  |    0.2252    |   77.5%                  |
| 16 |   0.0170  |    0.5113    |   48.9%                  |

## ANSWER TO THE FOUNDING QUESTION
Does forward-only training recover the correct response as a byproduct?
PARTIALLY. The canonical (M=16) forward-only-trained FNO has response error
0.5113 against a 1.0000 zero-information floor -> recovers ~49% of the true
response operator's structure, unsupervised, purely as a byproduct of
learning ψ0,V -> ψ_T. Forward is near-perfect (rel-L2 0.017); response is
roughly half-right. This is the ~30x forward-vs-response gap in absolute
terms (0.017 vs 0.511), now anchored against a real floor rather than a bare
number.

## Clip vs retrain (Block 4 vs Block 5) — the adaptation finding
Retrained models substantially OUTPERFORM clipped models at matched M for
M in {4,8,12} (e.g. M=12: retrain 0.225 vs truncate 0.556 -- 78% vs 44%
recovered). At M=2 both collapse to the floor (forward itself fails,
rel-L2=0.227) -- not enough spectral bandwidth for ANY strategy.
=> Response information is NOT rigidly tied to specific modes; a model
   allowed to train within a smaller budget re-routes/re-learns most of it.
   Truncate-at-inference OVERSTATES how damaging truncation is architecturally.

## SURPRISE, UNCONFIRMED (single seed) — non-monotonicity in the retrain curve
Retrain response error is NOT monotonic in M: 0.99 -> 0.56 -> 0.24 -> 0.23 ->
0.51 (M=2,4,8,12,16). M=12 (77.5% recovered) beats M=16 (48.9% recovered) at
nearly IDENTICAL forward accuracy (0.019 vs 0.017). If real: more modes gave
the 16-model forward headroom but WORSE response -- the extra modes 12-15
may be used in a way that corrupts response fidelity rather than helping it.
This is the sharpest possible form of the project's thesis, but it rests on
ONE seed per M. NEEDS 2-3 seeds per M before claiming the inversion is real
and not training-noise. Flagged, not claimed.

### Artifacts
- results/retrain_sweep.csv, results/untrained_sweep.csv
- results/response_clip_vs_retrain.png
- checkpoints/fno_N2000_seed0_M{2,4,8,12}_final.eqx

## CONFIRMED — M=12 beats M=16 on response, across 4 seeds each
Seed-robustness check: response error, seed in {0,1,2,3}, forward accuracy
matched across seeds (M=12: 0.0191-0.0213; M=16: 0.0170-0.0190).

| seed | M=12 response | M=16 response |
|------|---------------|---------------|
|  0   |    0.2252     |    0.5113     |
|  1   |    0.2101     |    0.5257     |
|  2   |    0.2308     |    0.5339     |
|  3   |    0.2024     |    0.4636     |
| mean |    0.2171     |    0.5086     |

NO OVERLAP across 4 seeds each: worst M=12 (0.231) still beats best M=16
(0.464) by a wide margin. This is not seed noise -- robust, decisive effect.

At near-identical forward accuracy, M=12 recovers ~2x the response structure
of M=16 (78% vs 49% recovered, floor=1.0). More spectral capacity bought
forward headroom the task didn't need, at direct cost to response fidelity.

### Candidate mechanism (UNTESTED, do not overclaim)
Forward-only training constrains only psi_T (the endpoint), not the path.
More modes = more forward-equivalent internal solutions = more ways to hit
psi_T without tracing the correct propagator-sandwich dynamics that the
response derivative is sensitive to. Fewer modes = tighter constraint = fewer
escape routes = training more often lands on solutions whose linearization
happens to be closer to correct. Consistent with roadmap's own framing
("two Hamiltonians can give near-identical psi(T) and different response...
the derivative sees the path; the state doesn't") applied to MODEL CAPACITY
rather than physical systems. Plausible, NOT tested. Would need e.g. path-
divergence diagnostics mid-integration to actually test.

### Artifacts
- results/seed_robustness_sweep.csv
- checkpoints/fno_N2000_seed{1,2,3}_M{12,16}_final.eqx

### Correction to headline framing
NOT "fewer modes = better response" (M=2,4 are worse on BOTH axes -- undertrained,
not response-optimal). Response error is NON-MONOTONIC / peaked in M: worst at the
extremes (2: forward-starved; presumably 16+ eventually: capacity-slack), best at
an INTERIOR sweet spot (~8-12) where forward capacity is sufficient without being
excessive. This sweet spot does NOT coincide with the n_modes that's optimal for
forward accuracy alone (forward keeps improving all the way to 16 and beyond).
Practical implication: n_modes tuned for forward performance (as Phase 3 did,
correctly, given only forward was measured) is not the same n_modes that's
best for response fidelity -- selecting architecture hyperparameters against a
single metric silently mis-tunes a different one you weren't measuring.