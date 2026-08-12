# scripts/ood_sweep.py — 7D-2b-ii: 9 models × 6 ls response-error grid
import os
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import ood_potential
from src.responses import jacobian_fno, mode_coupling_matrix, response_error
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, N = cfg.nx, 200
SEEDS = [0, 1, 2]
LAMS  = [0.0, 0.1, 10.0]
LS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
OUT = "results/ood"

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

# preload the 6 J_true caches (model-independent → load once, reuse across 9 models)
Jtrue = {ls: jnp.asarray(np.load(f"{OUT}/Jtrue_kq_ls{ls:.2f}.npy")) for ls in LS_GRID}
print(f"[load] 6 J_true caches OK")

# precompute OOD V per ls (model-independent → build once, reuse across 9 models)
Vood = {ls: jnp.stack([ood_potential(coeffs[i], V128[i], nx, ls) for i in range(N)]) for ls in LS_GRID}
print(f"[load] 6 OOD V batches OK")

def eval_cell(model, ls):
    V = Vood[ls]; Jt = Jtrue[ls]
    def err(i):
        Jf = mode_coupling_matrix(jacobian_fno(model, psi0[i], V[i]))
        return response_error(Jf, Jt[i])
    return jnp.array([err(i) for i in range(N)])

for s in SEEDS:
    for lam in LAMS:
        ckpt = f"checkpoints/dino_N2000_seed{s}_M16_lam{float(lam)}_final.eqx"
        model = None
        for ls in LS_GRID:
            path = f"{OUT}/resperr_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy"
            if os.path.exists(path):
                print(f"[skip] {path}")
                continue
            if model is None:                      # lazy-load: only if this (s,λ) has work left
                model = load_fno_f64_ckpt(ckpt, cfg, n_modes=16)
            errs = eval_cell(model, ls)
            np.save(path, np.asarray(errs))
            print(f"[save] s{s} λ{float(lam)} ls{ls:.2f}  mean={float(errs.mean()):.4f}")

# ---- assemble a summary table (means) once all 54 exist ----
print(f"\n{'ls':>6} | " + " | ".join(f"s{s}λ{float(l)}" for s in SEEDS for l in LAMS))
missing = 0
for ls in LS_GRID:
    row = []
    for s in SEEDS:
        for lam in LAMS:
            p = f"{OUT}/resperr_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy"
            if os.path.exists(p): row.append(f"{float(np.load(p).mean()):.4f}")
            else: row.append("  --  "); missing += 1
    print(f"{ls:6.2f} | " + " | ".join(row))
print(f"\n[{'done' if missing==0 else f'INCOMPLETE — {missing} missing, rerun'}]")