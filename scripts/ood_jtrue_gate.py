# scripts/ood_jtrue_gate.py — 7D-B2a: J_true pipeline must reproduce the frozen kq cache at ls=0.20
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import realize_potential
from src.responses import jacobian_true, mode_coupling_matrix
from configs.default import Config          # QUANTUM config (nx=128)

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
LS0, N = 0.20, 4                             # limit run — gate before the full 200

d = np.load("data/test.npz")
psi0    = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])   # AS STORED (FNO's convention)
coeffs0 = jnp.asarray(d["p_coeffs"][:N])
V128    = jnp.asarray(d["V_128"][:N])
Jcache  = jnp.asarray(np.load("results/Jtrue_kq_200.npy")[:N])        # (N,128,128) c128, kq basis

# reweight+renorm (convention B); at ls=0.20 the reweight factor is exp(0)=1 → must equal V_128
kk = jnp.arange(coeffs0.shape[1])
def ood_V(coeffs, Vind, ls):
    c = coeffs * jnp.exp(-kk * (ls - LS0))
    V = realize_potential(c, nx)
    return V * (jnp.linalg.norm(Vind) / jnp.linalg.norm(V))
V020 = jnp.stack([ood_V(coeffs0[i], V128[i], LS0) for i in range(N)])
print(f"[pre]  V(ls=0.20) vs V_128     max|Δ| = {float(jnp.max(jnp.abs(V020 - V128))):.2e}")

# J_true in kq basis via our pipeline, ψ₀ exactly as stored
Jkq = jnp.stack([mode_coupling_matrix(jacobian_true(psi0[i], V020[i], nx, dt, nt)) for i in range(N)])

err = float(jnp.max(jnp.abs(Jkq - Jcache)))
rel = float(jnp.linalg.norm(Jkq - Jcache) / jnp.linalg.norm(Jcache))
print(f"[GATE] J_true kq vs frozen cache  max|Δ| = {err:.2e} | rel = {rel:.2e}   "
      f"{'PASS' if err < 1e-10 else 'FAIL — ψ0 convention/basis mismatch, STOP'}")