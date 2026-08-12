# scripts/ood_gate.py  — 7D-B1a: OOD V-pipeline validation (no solver yet)
import jax; jax.config.update("jax_enable_x64", True)   # re-baseline in the exact regime
import jax.numpy as jnp
import numpy as np

from src.data_gen import realize_potential
from configs.default import Config          # the QUANTUM config (nx=128), NOT the Burgers default.py

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt   # 128, 1000, 5e-4
LS0, LS = 0.20, [0.15, 0.20, 0.25]
N = 8

# ---- load frozen test set: we need ψ₀, the generating spectra, and in-dist V ----
d = np.load("data/test.npz")
psi0 = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])  # (N,128) c128
coeffs0 = jnp.asarray(d["p_coeffs"][:N])                          # (N,64)  c128  (ls=0.20 spectra)
V128 = jnp.asarray(d["V_128"][:N])                               # (N,128) f64  (in-dist realized V)
k0 = jnp.asarray(d["p_k0"][:N])                                  # (N,)    carrier momenta

# ---- G0: provenance — does our realize pipeline reproduce the stored V? ----
V_re = jnp.stack([realize_potential(coeffs0[i], nx) for i in range(N)])
g0_err = float(jnp.max(jnp.abs(V_re - V128)))
print(f"[G0] realize(p_coeffs) vs V_128  max|Δ| = {g0_err:.2e}   {'PASS' if g0_err < 1e-10 else 'FAIL'}")

# ---- reweight the SAME spectrum from ls=0.20 to a new ls (only envelope moves) ----
kk = jnp.arange(coeffs0.shape[1])                      # k = 0..63, matches realize_potential's basis
def reweight(c, ls):
    return c * jnp.exp(-kk * (ls - LS0))               # exp(-k·(ls-ls0));  ls=0.20 → factor 1

# ---- realize + renorm (convention B: each sample → its own in-dist ‖V‖₂) ----
target = jnp.linalg.norm(V128, axis=1)                 # (N,)  per-sample in-dist norm
Vren, hikfrac, vmax = {}, {}, {}
for ls in LS:
    c = jnp.stack([reweight(coeffs0[i], ls) for i in range(N)])          # (N,64)
    V = jnp.stack([realize_potential(c[i], nx) for i in range(N)])       # (N,128)
    scale = target / jnp.linalg.norm(V, axis=1)                          # (N,)
    Vren[ls] = V * scale[:, None]
    # high-k fraction is renorm-invariant (global scale cancels) → compute on coeffs
    p = jnp.abs(c)**2
    hikfrac[ls] = float(jnp.mean(jnp.sum(p[:, 16:], axis=1) / jnp.sum(p, axis=1)))
    vmax[ls] = float(jnp.mean(jnp.max(jnp.abs(Vren[ls]), axis=1)))

# ---- G1: renorm sanity ----
g1a = max(float(jnp.max(jnp.abs(jnp.linalg.norm(Vren[ls], axis=1) - target))) for ls in LS)
g1b = float(jnp.max(jnp.abs(Vren[LS0] - V128)))        # (B)-specific: anchor must reproduce V_128
print(f"[G1a] renorm ‖Vren‖ vs target   max|Δ| = {g1a:.2e}   {'PASS' if g1a < 1e-10 else 'FAIL'}")
print(f"[G1b] anchor ls=0.20 vs V_128    max|Δ| = {g1b:.2e}   {'PASS' if g1b < 1e-10 else 'FAIL'}  (B-only)")

# ---- G2: high-k monotonicity + Strang monitor ----
print(f"\n{'ls':>6} {'high-k(>16) frac':>18} {'mean V_max':>12} {'V_max·dt':>10}")
for ls in LS:
    print(f"{ls:6.2f} {hikfrac[ls]:18.6f} {vmax[ls]:12.4f} {vmax[ls]*dt:10.2e}")
mono = hikfrac[0.15] > hikfrac[0.20] > hikfrac[0.25]
print(f"[G2] high-k monotone (0.15>0.20>0.25):  {'PASS' if mono else 'FAIL'}")

# ---- B1b: append to scripts/ood_gate.py ----
from src.solver import evolve
from src.responses import jacobian_true, mode_coupling_matrix

LS_TEST = 0.15                      # gate the roughest of the three (worst case for both checks)
V = Vren[LS_TEST]                   # (N,128), renormed, from B1a

# ---- G3: solver finite + unitary on OOD V ----
# ψ₀ from the test set is unnormalized (theta-function realization); normalize per-sample
# so "norm conserved" is a clean ≈1 statement. Unitarity is V-independent, but OOD V is
# the thing we're validating didn't break the split-step.
psi0n = psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2, axis=1, keepdims=True))
psiT = jnp.stack([evolve(psi0n[i], V[i], nx, dt, nt) for i in range(N)])   # (N,128) c128
finite = bool(jnp.all(jnp.isfinite(psiT)))
drift = float(jnp.max(jnp.abs(jnp.sum(jnp.abs(psiT)**2, axis=1) - 1.0)))
print(f"\n[G3] evolve finite = {finite} | max norm drift = {drift:.2e}   "
      f"{'PASS' if finite and drift < 1e-10 else 'FAIL'}")

# ---- G4: J_true finite + band-centered at k0 ----
# One sample is enough to validate band structure; do sample 0.
i = 0
J = jacobian_true(psi0n[i], V[i], nx, dt, nt)      # (128,128) c128, position basis
Jkq = mode_coupling_matrix(J)                      # (k,q) mode basis
jfin = bool(jnp.all(jnp.isfinite(Jkq)))

# Born predicts |J(k,q)| peaks along k−q = k0. Collapse mass onto d = (k−q) folded to [-64,63],
# then check the peak sits at k0[i].
kgrid = jnp.fft.fftfreq(nx, d=1.0/nx).astype(int)  # signed integers, fft order
dmat = ((kgrid[:, None] - kgrid[None, :] + nx//2) % nx) - nx//2   # folded k−q, (128,128)
mag = jnp.abs(Jkq)
# mass per d-bin
dvals = jnp.arange(-nx//2, nx//2)
massd = jnp.array([jnp.sum(mag[dmat == dd]) for dd in dvals])
peak_d = int(dvals[jnp.argmax(massd)])
print(f"[G4] J_true finite = {jfin} | k0 = {int(k0[i])} | peak d(k−q) = {peak_d}   "
      f"{'PASS' if jfin and peak_d == int(k0[i]) else 'FAIL'}")