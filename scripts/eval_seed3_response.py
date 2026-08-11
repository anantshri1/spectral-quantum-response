# scripts/eval_seed3_response.py — Phase 7B: seed-3 response eval
# x64 FIRST, before any jnp work (loader + Jacobians are all f64)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt            # frozen f64 loader
from src.responses import mode_coupling_matrix, jacobian_fno, response_error

cfg = Config()
nx, nt = cfg.nx, cfg.nt
dt = cfg.t_end / cfg.nt

# --- test set, positionally identical to how born.py built the J_true cache ---
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128)
V    = V.astype(jnp.float64)
N = psi0.shape[0]

# --- frozen J_true cache (FNO-independent; valid for every seed/M/lambda) ---
Jtrue_kq = np.load("results/Jtrue_kq_200.npy")               # (200,128,128) complex128
assert Jtrue_kq.shape == (N, nx, nx), "J_true cache shape mismatch — STOP"

def response_errs_for(ckpt_path, n_modes=None):
    """Per-sample response rel-Frobenius vs the frozen J_true cache.
    n_modes=None -> M16 (loader default); pass 12 for M12 checkpoints."""
    model = load_fno_f64_ckpt(ckpt_path, cfg, n_modes=n_modes) if n_modes \
            else load_fno_f64_ckpt(ckpt_path, cfg)
    errs = np.empty(N)
    for s in range(N):
        Jf_kq = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))
        errs[s] = float(response_error(Jf_kq, Jtrue_kq[s]))   # (estimate, true)
    return errs

# --- GATE: reproduce seed-0 M16 lam0 and match BOTH frozen caches ---
gate = response_errs_for("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx")
cached = np.load("results/fno_errs_200_f64trained.npy")       # seed0 M16 lam0, per-sample
print(f"[gate] mean {gate.mean():.4f}  (locked 0.587)")
print(f"[gate] max per-sample |diff| vs cache: {np.max(np.abs(gate - cached)):.2e}")
assert abs(gate.mean() - 0.587) < 0.02, "mean drifted — loader/machinery changed, STOP"
assert np.max(np.abs(gate - cached)) < 1e-6, "per-sample mismatch — STOP, do not trust seed3"
print("[gate] PASSED — loop reproduces canonical machinery. Safe to eval seed 3.")


# --- Block 2: seed-3 checkpoints (append) ---
import os
os.makedirs("results", exist_ok=True)

runs = [
    # (label,      ckpt,                                                    n_modes, save_path)
    ("M12 lam10", "checkpoints/dino_N2000_seed3_M12_lam10.0_final.eqx", 12,   "results/resp_errs_200_seed3_M12_lam10.0.npy"),
    ("M16 lam10", "checkpoints/dino_N2000_seed3_M16_lam10.0_final.eqx", None, "results/resp_errs_200_seed3_M16_lam10.0.npy"),
    ("M12 lam0",  "checkpoints/dino_N2000_seed3_M12_lam0.0_final.eqx",  12,   "results/resp_errs_200_seed3_M12_lam0.0.npy"),
    ("M16 lam0",  "checkpoints/dino_N2000_seed3_M16_lam0.0_final.eqx",  None, "results/resp_errs_200_seed3_M16_lam0.0.npy"),
]

means = {}
for label, ckpt, nm, save in runs:
    errs = response_errs_for(ckpt, n_modes=nm)
    np.save(save, errs)                       # save AFTER each — a disconnect loses at most one
    means[label] = errs.mean()
    assert np.all(np.isfinite(errs)) and errs.mean() < 2.0, f"{label} implausible — STOP"
    print(f"[seed3] {label:9s} mean {errs.mean():.4f}   saved -> {save}")

print("\n--- pre-registered directional checks (seed 3) ---")
flip = means["M16 lam10"] < means["M12 lam10"]
bias = means["M16 lam0"]  > means["M12 lam0"]
print(f"P_flip (lam10): M16 {means['M16 lam10']:.4f} vs M12 {means['M12 lam10']:.4f}  "
      f"-> {'flip HOLDS (M16<M12)' if flip else 'FLIP BROKEN'}")
print(f"P_bias (lam0):  M16 {means['M16 lam0']:.4f}  vs M12 {means['M12 lam0']:.4f}   "
      f"-> {'bias HOLDS (M16>M12)' if bias else 'BIAS ABSENT'}")

# --- Block 3: seeds 0-2 grid, through the SAME proven loop (append) ---
import os
os.makedirs("results", exist_ok=True)

runs = []
for seed in (0, 1, 2):
    runs += [
        (f"s{seed} M12 lam10", f"checkpoints/dino_N2000_seed{seed}_M12_lam10.0_final.eqx", 12,   f"results/resp_errs_200_seed{seed}_M12_lam10.0.npy"),
        (f"s{seed} M16 lam10", f"checkpoints/dino_N2000_seed{seed}_M16_lam10.0_final.eqx", None, f"results/resp_errs_200_seed{seed}_M16_lam10.0.npy"),
        (f"s{seed} M12 lam0",  f"checkpoints/dino_N2000_seed{seed}_M12_lam0.0_final.eqx",  12,   f"results/resp_errs_200_seed{seed}_M12_lam0.0.npy"),
        (f"s{seed} M16 lam0",  f"checkpoints/dino_N2000_seed{seed}_M16_lam0.0_final.eqx",  None, f"results/resp_errs_200_seed{seed}_M16_lam0.0.npy"),
    ]

for label, ckpt, nm, save in runs:
    if os.path.exists(save):                       # Colab-safety: don't recompute survivors
        errs = np.load(save)
        print(f"[grid] {label:13s} mean {errs.mean():.4f}   (loaded from disk)")
        continue
    errs = response_errs_for(ckpt, n_modes=nm)
    np.save(save, errs)                            # save AFTER each
    assert np.all(np.isfinite(errs)) and errs.mean() < 2.0, f"{label} implausible — STOP"
    print(f"[grid] {label:13s} mean {errs.mean():.4f}   saved -> {save}")

print(f"\n[grid] done — {len(runs)} configs on disk under results/resp_errs_200_seed*.npy")

# --- Block 4: assemble 4-seed grid + disjointness on disk ---
import numpy as np
seeds = (0, 1, 2, 3)
configs = ["M12_lam10.0", "M16_lam10.0", "M12_lam0.0", "M16_lam0.0"]
grid = {c: np.array([np.load(f"results/resp_errs_200_seed{s}_{c}.npy").mean()
                     for s in seeds]) for c in configs}

for c in configs:
    a = grid[c]
    print(f"{c:12s} mean {a.mean():.4f}  std {a.std(ddof=1):.4f}  range [{a.min():.4f}, {a.max():.4f}]")

def disjoint(a, b):                      # direction-agnostic no-overlap
    A, B = grid[a], grid[b]
    if A.max() < B.min(): return True, B.min()-A.max(), f"{a} entirely below {b}"
    if B.max() < A.min(): return True, A.min()-B.max(), f"{b} entirely below {a}"
    return False, 0.0, "OVERLAP"

for lam in ("lam10.0", "lam0.0"):
    ok, gap, msg = disjoint(f"M16_{lam}", f"M12_{lam}")
    print(f"[{lam}] {'NO OVERLAP' if ok else 'OVERLAP'} (gap {gap:.4f}) — {msg}")

np.savez("results/seed_grid_summary_4seed.npz", **grid)
print("saved -> results/seed_grid_summary_4seed.npz")