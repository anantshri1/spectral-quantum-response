# scripts/ood_smoke.py — validate loader + model signature + gate on ONE cell before the 54-loop
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import ood_potential
from src.responses import jacobian_fno, mode_coupling_matrix, response_error
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, N = cfg.nx, 200

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

# the gate cell: seed0 M16 λ0 at ls=0.20 (in-dist anchor)
ckpt = "checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx"
model = load_fno_f64_ckpt(ckpt, cfg, n_modes=16)      # n_modes explicit even for M16
print(f"[load] {ckpt}  OK")

# signature probe on ONE sample
out0 = model(psi0[0], V128[0])
print(f"[sig]  model(psi0,V) → shape {out0.shape} dtype {out0.dtype}  (expect (128,2) real)")

# J_true at ls=0.20 = frozen cache
Jtrue = jnp.asarray(np.load("results/ood/Jtrue_kq_ls0.20.npy"))   # (200,128,128)

# per-sample response error, kq basis, first 200
def cell_err(i):
    Jf = mode_coupling_matrix(jacobian_fno(model, psi0[i], V128[i]))
    return float(response_error(Jf, Jtrue[i]))

errs = jnp.array([cell_err(i) for i in range(N)])
mean = float(errs.mean())

# GATE: must reproduce the frozen forward-only floor
ref = np.load("results/fno_errs_200_f64trained.npy")
gate = float(jnp.max(jnp.abs(errs - jnp.asarray(ref))))
print(f"[GATE] seed0 M16 λ0 ls0.20 mean = {mean:.4f}  (expect 0.5787)")
print(f"[GATE] per-sample vs fno_errs_200_f64trained max|Δ| = {gate:.2e}  "
      f"{'PASS' if gate < 1e-6 else 'FAIL — loader/basis/ψ0 path mismatch, STOP'}")