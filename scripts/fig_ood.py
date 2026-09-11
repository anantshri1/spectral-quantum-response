# scripts/fig_ood.py — OOD figure: physical response error vs spectral-roughness shift.
# Reads results/ood/resperr_seed{s}_{label}_lam{lam}_ls{ls:.2f}.npz  (scalars iso/pj/pj0).
# pj  = prior-JVP at THAT ell's prior (perturb the way that regime's physics does) -- the honest OOD number.
# 3-seed (0,1,2). Three curves: M16 lam0, M12 lam0, M16 lam10.
import numpy as np, matplotlib.pyplot as plt

LS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS = [0, 1, 2]
curves = [("M16", "0.0",  "#c0392b", "s", r"$M{=}16$, $\lambda{=}0$ (forward-only)"),
          ("M12", "0.0",  "#e08a1e", "^", r"$M{=}12$, $\lambda{=}0$ (forward-only)"),
          ("M16", "10.0", "#2c7fb8", "o", r"$M{=}16$, $\lambda{=}10$ (supervised)")]

def series(label, lam, field):
    mean, sd = [], []
    for ls in LS:
        vals = []
        for s in SEEDS:
            p = f"results/ood/resperr_seed{s}_{label}_lam{lam}_ls{ls:.2f}.npz"
            vals.append(float(np.load(p)[field]))
        mean.append(np.mean(vals)); sd.append(np.std(vals))
    return np.array(mean), np.array(sd)

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.axvline(0.20, color="0.6", ls=":", lw=1.0)
ax.text(0.202, 0.02, "training\n$\\ell=0.20$", fontsize=7.5, color="0.4")
for label, lam, c, mk, leg in curves:
    m, sd = series(label, lam, "pj")
    ax.errorbar(LS, m, yerr=sd, marker=mk, ms=5, lw=1.6, capsize=2.5, color=c, label=leg)
ax.set_xlabel(r"potential roughness  $\ell$  (smaller $=$ rougher, more high-$k$ power)")
ax.set_ylabel(r"physical response error  $err_{\mathrm{phys}}(\ell)$")
ax.invert_xaxis()                      # rough -> smooth left-to-right; shift severity grows leftward
ax.set_ylim(0, None)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("results/fig_ood.png", dpi=200, bbox_inches="tight")
print("saved results/fig_ood.png")

# provenance table for the caption + text
print(f"\n{'ls':>5} {'M16l0':>8} {'M12l0':>8} {'M16l10':>8} {'M16/M12':>8}")
for i, ls in enumerate(LS):
    a = series("M16","0.0","pj")[0][i]; b = series("M12","0.0","pj")[0][i]; c = series("M16","10.0","pj")[0][i]
    print(f"{ls:5.2f} {a:8.4f} {b:8.4f} {c:8.4f} {a/b:8.2f}")