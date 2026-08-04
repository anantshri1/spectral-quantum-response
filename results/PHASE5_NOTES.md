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