# scripts/ood_demo_epsilon.py — B3-3b-i: find the linear-regime ε valley (gate before the demo figure)
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import ood_potential
from src.solver import evolve
from src.responses import jacobian_true, jacobian_fno
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
N = 200

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

# ---- D13: pick the sample whose λ0 response error is NEAREST the median (representative) ----
LS_PICK = 0.05
resp_l0 = np.load(f"results/ood/resperr_seed0_M16_lam0.0_ls{LS_PICK:.2f}.npy")  # (200,) seed0
med = np.median(resp_l0)
i_star = int(np.argmin(np.abs(resp_l0 - med)))
print(f"[D13] ls={LS_PICK}: median λ0 resp={med:.4f}, picked sample {i_star} "
      f"(resp={resp_l0[i_star]:.4f}, |Δmed|={abs(resp_l0[i_star]-med):.4f})")

# ---- build V at this sample/ls, and J_true·δV in POSITION space (the demo lives in ψ-space) ----
V = ood_potential(coeffs[i_star], V128[i_star], nx, LS_PICK)     # (128,) f64
ps = psi0[i_star]

# ---- D14: random unit δV, fixed seed ----
key = jax.random.PRNGKey(1234)
dV = jax.random.normal(key, (nx,), dtype=V.dtype)
dV = dV / jnp.linalg.norm(dV)

# J_true (position basis, (nx,nx) complex) and its action on δV
Jt = jacobian_true(ps, V, nx, dt, nt)          # ∂ψ_T/∂V, (128,128) c128
Jt_dV = Jt @ dV                                 # (128,) c128 — predicted dψ_T per unit ε

psiT_V = evolve(ps, V, nx, dt, nt)              # unperturbed truth

# ---- ε-valley: prediction error of J_true's linearization, normalized by ε ----
# residual(ε) = ‖ψ_T(V+εδV) − [ψ_T(V) + ε·Jt·δV]‖ / ε
#   linear regime: residual ∝ ε  (the O(ε²) term over ε) → straight line, slope 1 in log-log
#   too small ε: float64 round-off floor dominates → residual rises as ε→0
#   valley/knee between them = where to sample the demo
print(f"\n{'eps':>10} {'resid/eps':>14}  regime")
eps_grid = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
for eps in eps_grid:
    truth = evolve(ps, V + eps * dV, nx, dt, nt)
    pred  = psiT_V + eps * Jt_dV
    resid = float(jnp.linalg.norm(truth - pred) / eps)
    print(f"{eps:10.1e} {resid:14.3e}")
print("\npick ε in the flat/linear valley (resid/eps small & scaling ~linearly, above the round-off upturn)")