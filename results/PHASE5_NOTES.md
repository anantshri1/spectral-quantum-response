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