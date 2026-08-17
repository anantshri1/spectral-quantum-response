# scripts/backfill_seed3_lam01.py — item (4): fold seed3 into N2000/M16/λ=0.1
# gives §4 response (raw) + §10 r,cos from ONE gated pass. x64 FIRST.
import jax; jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp
from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt
from src.responses import (mode_coupling_matrix, jacobian_fno, response_error,
                           scale_ratio, cosine_alignment, optimal_rescale)

cfg = Config(); nx = cfg.nx
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
N = psi0.shape[0]
Jtrue_kq = np.load("results/Jtrue_kq_200.npy")
assert Jtrue_kq.shape == (N, nx, nx), "J_true cache shape mismatch — STOP"

# --- decompose_for: VERBATIM from eval_align_taskb.py (do NOT import that module) ---
def decompose_for(ckpt_path, n_modes=None):
    model = load_fno_f64_ckpt(ckpt_path, cfg, n_modes=n_modes) if n_modes \
            else load_fno_f64_ckpt(ckpt_path, cfg)
    r=np.empty(N); cos=np.empty(N); al=np.empty(N); res=np.empty(N); raw=np.empty(N)
    for s in range(N):
        Jf = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))
        Jt = Jtrue_kq[s]
        r[s]   = float(scale_ratio(Jf, Jt))
        cos[s] = float(cosine_alignment(Jf, Jt))
        a, rr  = optimal_rescale(Jf, Jt); al[s]=float(a); res[s]=float(rr)
        raw[s] = float(response_error(Jf, Jt))
    return dict(r=r, cos=cos, alpha=al, resid=res, raw=raw)

# --- GATE (verbatim logic): seed0 M16 lam0 -> identities + frozen-cache match ---
d0 = decompose_for("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx")
g1 = np.max(np.abs(d0["resid"] - np.sqrt(1 - d0["cos"]**2)))
g2 = np.max(np.abs(np.sqrt(d0["r"]**2 + 1 - 2*d0["r"]*d0["cos"]) - d0["raw"]))
cached = np.load("results/fno_errs_200_f64trained.npy")
g3 = np.max(np.abs(d0["raw"] - cached))
print(f"[gate1] {g1:.2e}  [gate2] {g2:.2e}  [gate3] {g3:.2e}  seed0 mean {d0['raw'].mean():.4f} (~0.58 basis)")
assert g1 < 1e-10 and g2 < 1e-10, "closed-form identity broke — STOP"
assert g3 < 1e-6, "provenance mismatch vs frozen cache — STOP"
print("[gates] PASSED\n")

# --- the one backfill checkpoint: seed3, N2000, M16, λ=0.1 ---
save3 = "results/align_seed3_M16_lam0.1.npz"
if os.path.exists(save3):
    z = np.load(save3); d3 = {k: z[k] for k in z.files}; print(f"[seed3] loaded {save3}")
else:
    d3 = decompose_for("checkpoints/dino_N2000_seed3_M16_lam0.1_final.eqx")
    np.savez(save3, **d3); print(f"[seed3] saved -> {save3}")
assert np.all(np.isfinite(d3["raw"])) and d3["raw"].mean() < 2.0, "seed3 implausible — STOP"
np.save("results/resp_errs_vsN_N2000_seed3_M16_lam0.1.npy", np.asarray(d3["raw"]))  # §9-name, for the record
print(f"[seed3]  raw {d3['raw'].mean():.4f} | r {d3['r'].mean():.4f} | cos {d3['cos'].mean():.4f}")

# --- 4-seed re-mean of N2000/M16/λ=0.1 (§4 response + §10 first-table row) ---
seeds = (0, 1, 2, 3)
paths = [f"results/align_seed{s}_M16_lam0.1.npz" for s in seeds]
missing = [p for p in paths if not os.path.exists(p)]
assert not missing, f"missing seed files: {missing} — check seed0-2 λ=0.1 align filenames"
raw = np.array([np.load(p)["raw"].mean() for p in paths])
r   = np.array([np.load(p)["r"].mean()   for p in paths])
cos = np.array([np.load(p)["cos"].mean() for p in paths])
print("\n=== N2000 / M16 / λ=0.1 — 4-seed (0-3) ===")
print(f"response (§4):  {raw.mean():.4f} ± {raw.std(ddof=1):.4f}   [was 3-seed 0.1034 ± 0.0017]")
print(f"r       (§10):  {r.mean():.4f} ± {r.std(ddof=1):.4f}       [was 3-seed 0.9946 ± 0.0008]")
print(f"cos     (§10):  {cos.mean():.4f} ± {cos.std(ddof=1):.4f}     [was 3-seed 0.9941 ± 0.0002]")