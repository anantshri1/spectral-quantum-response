# scripts/ood_fig_degradation.py — B3-3a: the thesis degradation figure
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "results/ood"
LS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS = [0, 1, 2]

def load_seedavg(kind, lam):
    """kind in {'resperr','fwderr'}; returns (mean[6], std[6]) over seeds, per ls."""
    means, stds = [], []
    for ls in LS_GRID:
        vals = []
        for s in SEEDS:
            p = f"{OUT}/{kind}_seed{s}_M16_lam{float(lam)}_ls{ls:.2f}.npy"
            if not os.path.exists(p):
                vals = None; break
            a = np.load(p)
            vals.append(float(a.mean()) if a.ndim else float(a))  # resperr=(200,), fwderr=scalar
        if vals is None:
            means.append(np.nan); stds.append(np.nan)
        else:
            means.append(np.mean(vals)); stds.append(np.std(vals))
    return np.array(means), np.array(stds)

# gather the four curves we need
resp_l0,  resp_l0_s  = load_seedavg("resperr", 0.0)
resp_l10, resp_l10_s = load_seedavg("resperr", 10.0)
fwd_l0,   fwd_l0_s   = load_seedavg("fwderr",  0.0)
fwd_l10,  fwd_l10_s  = load_seedavg("fwderr",  10.0)

x = np.array(LS_GRID)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

def panel(ax, fwd, fwd_s, resp, resp_s, title):
    ax.errorbar(x, fwd,  yerr=fwd_s,  marker="o", lw=2, capsize=3, label="forward rel-L2",  color="tab:green")
    ax.errorbar(x, resp, yerr=resp_s, marker="s", lw=2, capsize=3, label="response error", color="tab:red")
    ax.axhspan(0, 0.10, color="tab:green", alpha=0.06)   # "acceptable" band ≤10%
    ax.set_xlabel("v_length_scale  (← rougher / OOD    smoother / in-dist →)")
    ax.set_title(title)
    ax.axvline(0.20, ls=":", color="gray", lw=1)          # in-dist anchor
    ax.set_xlim(0.045, 0.305)
    ax.grid(alpha=0.25)

panel(axL, fwd_l0,  fwd_l0_s,  resp_l0,  resp_l0_s,  "Forward-only (λ=0): monitor lies")
panel(axR, fwd_l10, fwd_l10_s, resp_l10, resp_l10_s, "DINO (λ=10): monitor honest")
axL.set_ylabel("error")
axL.annotate("forward stays 'acceptable'\nresponse catastrophic everywhere",
             xy=(0.05, resp_l0[0]), xytext=(0.09, 0.42),
             fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
axR.legend(loc="center right", fontsize=9)
axL.set_ylim(-0.02, 0.66)

fig.suptitle("Forward accuracy is a misleading OOD monitor for response trustworthiness (3-seed)", y=1.02)
fig.tight_layout()
path = f"{OUT}/fig_degradation.png"
fig.savefig(path, dpi=160, bbox_inches="tight")
print(f"[save] {path}")

# print the exact numbers going into the figure (for the caption / sanity)
print(f"\n{'ls':>6} | {'λ0 fwd':>7} {'λ0 resp':>8} | {'λ10 fwd':>8} {'λ10 resp':>9}")
for i, ls in enumerate(LS_GRID):
    print(f"{ls:6.2f} | {fwd_l0[i]:7.4f} {resp_l0[i]:8.4f} | {fwd_l10[i]:8.4f} {resp_l10[i]:9.4f}")