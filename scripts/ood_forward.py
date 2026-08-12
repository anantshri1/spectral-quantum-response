# scripts/ood_forward.py — 7D-2c: forward rel-L2 vs ls (the P3 "monitor" curve)
import os
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import ood_potential
from src.solver import evolve
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
N = 200
SEEDS = [0, 1, 2]
LAMS  = [0.0, 10.0]                       # forward-only vs heavy-DINO; the monitor contrast
LS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
OUT = "results/ood"

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])   # as-is (D9=a)
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

# reuse the SAME OOD V as 2b-ii (ood_potential is canonical → identical V per ls)
Vood = {ls: jnp.stack([ood_potential(coeffs[i], V128[i], nx, ls) for i in range(N)]) for ls in LS_GRID}
print("[load] 6 OOD V batches OK")

# ground truth: solver ψ_T on OOD V (model-independent → once per ls, reuse across models)
# NOTE: uT_* in test.npz is the IN-DIST ψ_T (ls=0.20 V). For OOD we must re-evolve.
def true_psiT(ls):
    V = Vood[ls]
    return jnp.stack([evolve(psi0[i], V[i], nx, dt, nt) for i in range(N)])   # (200,128) c128
psiT_true = {}
for ls in LS_GRID:
    cache = f"{OUT}/psiT_true_ls{ls:.2f}.npy"
    if os.path.exists(cache):
        psiT_true[ls] = jnp.asarray(np.load(cache)); print(f"[skip] {cache}")
    else:
        pt = true_psiT(ls); np.save(cache, np.asarray(pt)); psiT_true[ls] = pt
        print(f"[save] {cache}")

def fno_psiT(model, ls):
    V = Vood[ls]
    def one(i):
        out = model(psi0[i], V[i])              # (128,2) real
        return out[:, 0] + 1j * out[:, 1]       # (128,) complex
    return jnp.stack([one(i) for i in range(N)])

def rel_l2(a, b):                                # per-sample rel-L2, then mean
    num = jnp.linalg.norm(a - b, axis=1)
    den = jnp.linalg.norm(b, axis=1)
    return jnp.mean(num / den)

for s in SEEDS:
    for lam in LAMS:
        ckpt = f"checkpoints/dino_N2000_seed{s}_M16_lam{float(lam)}_final.eqx"
        model = None
        for ls in LS_GRID:
            path = f"{OUT}/fwderr_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy"
            if os.path.exists(path):
                print(f"[skip] {path}"); continue
            if model is None:
                model = load_fno_f64_ckpt(ckpt, cfg, n_modes=16)
            err = float(rel_l2(fno_psiT(model, ls), psiT_true[ls]))
            np.save(path, np.array(err))
            print(f"[save] s{s} λ{float(lam)} ls{ls:.2f}  fwd_relL2={err:.4f}")

# ---- summary: forward rel-L2, seed-averaged, per (λ, ls) ----
print(f"\n{'ls':>6} | {'λ0 fwd':>8} | {'λ10 fwd':>8}   (seed-avg)")
for ls in LS_GRID:
    row = {}
    for lam in LAMS:
        vals = [np.load(f"{OUT}/fwderr_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy")
                for s in SEEDS if os.path.exists(f"{OUT}/fwderr_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy")]
        row[lam] = float(np.mean(vals)) if vals else float("nan")
    print(f"{ls:6.2f} | {row[0.0]:8.4f} | {row[10.0]:8.4f}")