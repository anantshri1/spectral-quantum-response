# scripts/eval_align_taskb.py — Task B block 2a: (r, cos, alpha, resid) decomposition
# x64 FIRST — loader + Jacobians are all f64 (dtype leak discipline)
import jax
jax.config.update("jax_enable_x64", True)
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

def decompose_for(ckpt_path, n_modes=None):
    """Per-sample (r, cos, alpha, resid) + raw response_error, in (k,q) space.
    cos/resid are rotation-invariant (proven: gate c), so (k,q) == position space."""
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

# ---------------- GATES: seed0 M16 lam0, against frozen provenance ----------------
d0 = decompose_for("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx")
g1 = np.max(np.abs(d0["resid"] - np.sqrt(1 - d0["cos"]**2)))          # closed-form identity
recon = np.sqrt(d0["r"]**2 + 1 - 2*d0["r"]*d0["cos"])
g2 = np.max(np.abs(recon - d0["raw"]))                                # err^2 = r^2+1-2rcos
cached = np.load("results/fno_errs_200_f64trained.npy")
g3 = np.max(np.abs(d0["raw"] - cached))                               # matches frozen cache
print(f"[gate1] resid == sqrt(1-cos^2)     max|diff| {g1:.2e}")
print(f"[gate2] raw   == sqrt(r^2+1-2rcos) max|diff| {g2:.2e}")
print(f"[gate3] raw   == frozen cache      max|diff| {g3:.2e}   mean {d0['raw'].mean():.4f} (locked 0.587)")
assert g1 < 1e-10 and g2 < 1e-10, "closed-form identity broke — STOP"
assert g3 < 1e-6, "provenance mismatch vs frozen cache — STOP, machinery drifted"
print("[gates] PASSED — decomposition is consistent with canonical machinery.\n")

# ---------------- decompose the four N2000 configs (seed 0) ----------------
runs = [
    ("M16 lam0",   "checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx",  None, "results/align_seed0_M16_lam0.0.npz"),
    ("M12 lam0",   "checkpoints/dino_N2000_seed0_M12_lam0.0_final.eqx",  12,   "results/align_seed0_M12_lam0.0.npz"),
    ("M16 lam0.1", "checkpoints/dino_N2000_seed0_M16_lam0.1_final.eqx",  None, "results/align_seed0_M16_lam0.1.npz"),
    ("M16 lam10",  "checkpoints/dino_N2000_seed0_M16_lam10.0_final.eqx", None, "results/align_seed0_M16_lam10.0.npz"),
]
print(f"{'config':11s} {'r':>7s} {'cos':>8s} {'resid':>8s} {'raw':>8s}")
for label, ckpt, nm, save in runs:
    if os.path.exists(save):
        z = np.load(save); d = {k: z[k] for k in z.files}
        tag = "(disk)"
    else:
        d = decompose_for(ckpt, n_modes=nm)
        np.savez(save, **d); tag = "-> saved"
    print(f"{label:11s} {d['r'].mean():7.4f} {d['cos'].mean():8.4f} "
          f"{d['resid'].mean():8.4f} {d['raw'].mean():8.4f}  {tag}")


# --- Block 2c: 4-seed decomposition + no-overlap on r and cos (append) ---
seeds = (0, 1, 2)
cfgs = [  # (config_tag, n_modes)
    ("M16_lam0.0",  None), ("M12_lam0.0", 12),
    ("M16_lam0.1",  None), ("M16_lam10.0", None),
]
def ck(seed, tag): return f"checkpoints/dino_N2000_seed{seed}_{tag}_final.eqx"

for seed in seeds:
    for tag, nm in cfgs:
        save = f"results/align_seed{seed}_{tag}.npz"
        if os.path.exists(save):
            continue                                   # survivor — skip
        d = decompose_for(ck(seed, tag), n_modes=nm)
        np.savez(save, **d)
        print(f"[2c] seed{seed} {tag:12s} r={d['r'].mean():.4f} cos={d['cos'].mean():.4f} -> saved")

# per-seed means -> 4-vectors
def grid(field, tag):
    return np.array([np.load(f"results/align_seed{s}_{tag}.npz")[field].mean() for s in seeds])

print(f"\n{'config':12s} {'r mean±std':>16s} {'cos mean±std':>16s}")
for tag, _ in cfgs:
    r, c = grid("r", tag), grid("cos", tag)
    print(f"{tag:12s} {r.mean():.4f}±{r.std(ddof=1):.4f}   {c.mean():.4f}±{c.std(ddof=1):.4f}")

def no_overlap(a, b):                     # direction-agnostic
    if a.max() < b.min(): return True, b.min()-a.max()
    if b.max() < a.min(): return True, a.min()-b.max()
    return False, 0.0

print("\n--- no-overlap checks (the headline gates) ---")
# P-B4: does the r>1 over-energetic gap M16>M12 (lam0) survive 4 seeds?
ok, gap = no_overlap(grid("r","M16_lam0.0"), grid("r","M12_lam0.0"))
print(f"P-B4 scale gap  r[M16 lam0] vs r[M12 lam0]: {'NO OVERLAP' if ok else 'OVERLAP'} (gap {gap:.4f})")
# P-B4: directional gap in cos
ok, gap = no_overlap(grid("cos","M16_lam0.0"), grid("cos","M12_lam0.0"))
print(f"P-B4 dir gap    cos[M16 lam0] vs cos[M12 lam0]: {'NO OVERLAP' if ok else 'OVERLAP'} (gap {gap:.4f})")
# P-B1: is M16 lam0 genuinely over-energetic (r>1) across all seeds?
r16 = grid("r","M16_lam0.0")
print(f"P-B1 over-energetic: r[M16 lam0] all-seed range [{r16.min():.4f}, {r16.max():.4f}] — {'r>1 all seeds' if r16.min()>1 else 'CROSSES 1'}")

# --- Block 2b: across-N decomposition, lam0 M16 (append; run after 2c) ---
Ns = (250, 500, 1000, 2000)
for seed in seeds:
    for n in Ns:
        save = f"results/align_vsN_N{n}_seed{seed}_M16_lam0.0.npz"
        if os.path.exists(save):
            continue
        ckpt = f"checkpoints/dino_N{n}_seed{seed}_M16_lam0.0_final.eqx"
        d = decompose_for(ckpt, n_modes=None)
        np.savez(save, **d)
        print(f"[2b] N{n} seed{seed} r={d['r'].mean():.4f} cos={d['cos'].mean():.4f} -> saved")

def gridN(field, n):
    return np.array([np.load(f"results/align_vsN_N{n}_seed{seed}_M16_lam0.0.npz")[field].mean()
                     for seed in seeds])

print(f"\n{'N':>5s} {'r mean±std':>15s} {'cos mean±std':>15s} {'raw(canon)':>11s}")
canon = {250:1.140, 500:1.000, 1000:0.824, 2000:0.587}
for n in Ns:
    r, c = gridN("r", n), gridN("cos", n)
    print(f"{n:5d} {r.mean():.4f}±{r.std(ddof=1):.4f}  {c.mean():.4f}±{c.std(ddof=1):.4f}   {canon[n]:.3f}")