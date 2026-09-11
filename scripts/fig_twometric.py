# scripts/fig_twometric.py — MAIN FIGURE: operator-space vs data-space response error vs M.
# Provenance: 3-seed means. iso = E3 retrain isotropic column; phys = E2/E3 prior-JVP.
# Values transcribed from E3 cross-seed output (2026-09-xx run); SEs in comments.
# If mode-sweep arrays are on disk, replace the two dicts with a loader (see bottom).
import numpy as np, matplotlib.pyplot as plt

M      = np.array([2, 4, 8, 12, 14, 16, 20, 24])
iso    = np.array([1.0115, 0.5512, 0.2636, 0.2622, 0.4018, 0.5873, 0.8343, 0.9176])
iso_se = np.array([0.0127, 0.0037, 0.0053, 0.0165, 0.0151, 0.0049, 0.0158, 0.0225])
phys   = np.array([0.5540, 0.2835, 0.1160, 0.0861, 0.0862, 0.0909, 0.0945, 0.0878])
# phys SE ~0.001-0.003 (cross-seed, CRN); direction-MC common offset ~0.0025 (see E1 note)
phys_se= np.array([0.004, 0.003, 0.003, 0.002, 0.002, 0.001, 0.002, 0.002])

SUPPORT = 12   # prior spectral support / plateau onset

fig, ax = plt.subplots(figsize=(6.2, 4.3))
ax.axvspan(M.min(), SUPPORT, color="0.92", zorder=0)
ax.text(3.0, 0.96, "capacity resolves\nthe physics", fontsize=8, color="0.35", va="top")
ax.text(15.5, 0.96, "excess enters the\nunprobed sector", fontsize=8, color="0.35", va="top")

ax.errorbar(M, iso,  yerr=iso_se,  marker="o", ms=5, lw=1.6, capsize=2.5,
            color="#c0392b", label=r"operator-space  $err$  (isotropic)")
ax.errorbar(M, phys, yerr=phys_se, marker="s", ms=5, lw=1.6, capsize=2.5,
            color="#2c7fb8", label=r"data-space  $err_{\mathrm{phys}}$  (prior-JVP)")
ax.axvline(SUPPORT, color="0.5", ls=":", lw=1.0)

ax.set_xlabel(r"retained spectral modes  $M$")
ax.set_ylabel(r"relative response error")
ax.set_xticks(M)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.set_title("Isotropic and physical response error diverge above the data's spectral support",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("Results/fig_twometric.png", dpi=200, bbox_inches="tight")
print("saved Results/fig_twometric.png")
print(f"iso  min at M={M[iso.argmin()]} ({iso.min():.3f}); iso M24/M12 = {iso[-1]/iso[3]:.2f}x")
print(f"phys plateau M>=12: {phys[3:].mean():.4f} +/- {phys[3:].std():.4f}")

# --- IF arrays are on disk, use this instead of the hardcoded dicts above: ---
# d = np.load("results/mode_sweep_3seed.npz")   # <-- confirm filename/keys
# M, iso, phys = d["M"], d["iso_mean"], d["phys_mean"]; iso_se, phys_se = d["iso_se"], d["phys_se"]