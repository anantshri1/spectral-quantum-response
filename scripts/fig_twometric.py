# scripts/fig_twometric.py — MAIN FIGURE (left subpanel): operator- vs data-space error vs M.
# Uniformly 3-SEED (seeds 0,1,2). iso: recomputed from Jfno_kq matrices + Jtrue_kq_200.
# phys: mean of jvp_prior mean-field across seeds 0,1,2.  (seed3 exists for M16 but is DROPPED
# here so every M is the same 3-seed estimator -- 4-seed M16 lives in the fixed-M tables.)
import numpy as np, glob, os

Ms = [2, 4, 8, 12, 14, 16, 20, 24]
SEEDS = [0, 1, 2]
Jtrue = np.load("results/Jtrue_kq_200.npy")                 # (200,128,128)

def iso_for(M):                                             # 3-seed mean over matrices
    vals = []
    for s in SEEDS:
        p = f"results/Jfno_kq_seed{s}_M{M}.npy"
        if not os.path.exists(p): return (np.nan, np.nan)
        Jf = np.load(p)
        vals.append(np.mean([np.linalg.norm(Jf[i]-Jtrue[i])/np.linalg.norm(Jtrue[i])
                             for i in range(Jf.shape[0])]))
    return float(np.mean(vals)), float(np.std(vals))

def phys_for(M):                                            # 3-seed mean over jvp_prior scalars
    vals = []
    for s in SEEDS:
        p = f"results/jvp_prior_seed{s}_M{M}_lam0.0.npz"
        if not os.path.exists(p): return (np.nan, np.nan)
        vals.append(float(np.load(p)["mean"]))              # field is literally 'mean'
    return float(np.mean(vals)), float(np.std(vals))

iso  = np.array([iso_for(M)[0]  for M in Ms]); iso_se  = np.array([iso_for(M)[1]  for M in Ms])
phys = np.array([phys_for(M)[0] for M in Ms]); phys_se = np.array([phys_for(M)[1] for M in Ms])
print("M      iso     iso_sd    phys    phys_sd")
for i, M in enumerate(Ms):
    print(f"{M:3d}  {iso[i]:.4f}  {iso_se[i]:.4f}   {phys[i]:.4f}  {phys_se[i]:.4f}")

import matplotlib.pyplot as plt
M = np.array(Ms); SUPPORT = 12
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.axvspan(M.min(), SUPPORT, color="0.92", zorder=0)
ax.errorbar(M, iso,  yerr=iso_se,  marker="o", ms=5, lw=1.6, capsize=2.5, color="#c0392b",
            label=r"operator-space $err$ (isotropic)")
ax.errorbar(M, phys, yerr=phys_se, marker="s", ms=5, lw=1.6, capsize=2.5, color="#2c7fb8",
            label=r"data-space $err_{\mathrm{phys}}$ (prior-JVP)")
ax.axvline(SUPPORT, color="0.5", ls=":", lw=1.0)
ax.set_xlabel(r"retained spectral modes $M$"); ax.set_ylabel("relative response error")
ax.set_title("Isotropic and physical response error diverge above the data's spectral support",
             fontsize=9.5)
ax.set_xticks(M); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig("results/fig_twometric.png", dpi=200, bbox_inches="tight")
print("saved results/fig_twometric.png")