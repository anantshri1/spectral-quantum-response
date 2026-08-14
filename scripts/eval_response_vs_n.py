# scripts/eval_response_vs_n.py — Task A Block 3: response error vs N. DISPOSABLE.
# Machinery (setup + response_errs_for + gate) is copied VERBATIM from
# scripts/eval_seed3_response.py so it inherits that file's proven (k,q)
# convention and provenance — without importing its import-time side effects
# or editing the committed anchor. Only the N-loop at the bottom is new.
import jax
jax.config.update("jax_enable_x64", True)          # x64 first, as everywhere

import os
import numpy as np
import jax.numpy as jnp

from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt          # frozen f64 loader
from src.responses import mode_coupling_matrix, jacobian_fno, response_error

cfg = Config()
nx, nt = cfg.nx, cfg.nt
dt = cfg.t_end / cfg.nt

# --- test set, positionally identical to how born.py built the J_true cache ---
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128)
V    = V.astype(jnp.float64)
N = psi0.shape[0]

Jtrue_kq = np.load("results/Jtrue_kq_200.npy")             # (200,128,128) complex128
assert Jtrue_kq.shape == (N, nx, nx), "J_true cache shape mismatch — STOP"

def response_errs_for(ckpt_path, n_modes=None):
    """Per-sample response rel-Frobenius vs the frozen J_true cache. VERBATIM copy."""
    model = load_fno_f64_ckpt(ckpt_path, cfg, n_modes=n_modes) if n_modes \
            else load_fno_f64_ckpt(ckpt_path, cfg)
    errs = np.empty(N)
    for s in range(N):
        Jf_kq = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))
        errs[s] = float(response_error(Jf_kq, Jtrue_kq[s]))
    return errs

# --- GATE: reproduce seed0 N2000 M16 lam0 and match the frozen cache (verbatim) ---
gate = response_errs_for("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx")
cached = np.load("results/fno_errs_200_f64trained.npy")
print(f"[gate] mean {gate.mean():.4f}  (locked 0.587)")
print(f"[gate] max per-sample |diff| vs cache: {np.max(np.abs(gate - cached)):.2e}")
assert abs(gate.mean() - 0.587) < 0.02, "mean drifted — loader/machinery changed, STOP"
assert np.max(np.abs(gate - cached)) < 1e-6, "per-sample mismatch — STOP"
print("[gate] PASSED — loop reproduces canonical machinery. Safe to eval the N-sweep.\n")

# --- N-loop (NEW): M16, lam0, seeds 0-2. Skip-if-exists; skip-if-ckpt-missing. ---
os.makedirs("results", exist_ok=True)
NS, SEEDS = [250, 500, 1000, 2000], [0, 1, 2]

for n in NS:
    row = []
    for s in SEEDS:
        if n == 2000:
            # same checkpoint as the 4-seed grid -> reuse its per-sample cache
            # (identical object; ties the N-anchor to the canonical 0.5787/0.587)
            src = f"results/resp_errs_200_seed{s}_M16_lam0.0.npy"
            if not os.path.exists(src):
                print(f"[vsN] N=2000 seed={s}: grid cache missing, skip"); continue
            errs = np.load(src)
        else:
            save = f"results/resp_errs_vsN_N{n}_seed{s}_M16_lam0.0.npy"
            ckpt = f"checkpoints/dino_N{n}_seed{s}_M16_lam0.0_final.eqx"
            if os.path.exists(save):
                errs = np.load(save)
            elif not os.path.exists(ckpt):
                print(f"[vsN] N={n:<4} seed={s}: ckpt not trained yet, skip"); continue
            else:
                errs = response_errs_for(ckpt)                # M16 default
                np.save(save, errs)                           # save AFTER each
                assert np.all(np.isfinite(errs)) and errs.mean() < 2.0, \
                    f"N={n} seed={s} implausible — STOP"
        print(f"[vsN] N={n:<4} seed={s}  response mean {errs.mean():.4f}")
        row.append(errs.mean())
    if row:
        row = np.array(row)
        print(f"      -> N={n:<4} across {len(row)} seed(s): "
              f"mean {row.mean():.4f}  std {row.std(ddof=1) if len(row)>1 else 0:.4f}\n")