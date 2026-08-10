# scripts/plot_born_scatter.py — Fact 2 figure: Born_err ⊥ FNO_err
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

born = np.load("results/born_errs_200.npy")
fno  = np.load("results/fno_errs_200_f64trained.npy")
r_p, _ = pearsonr(born, fno)
r_s, _ = spearmanr(born, fno)

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(born, fno, s=18, alpha=0.55, edgecolor="none", color="#2b6cb0")

# reference lines: the two floors + the random-response marker
ax.axhline(0.587, ls="--", lw=1, color="#718096", label="FNO 3-seed floor 0.587")
ax.axvline(1.0,   ls=":",  lw=1, color="#a0aec0", label="Born = random (1.0)")

ax.set_xlabel("Born (1st-order PT) response error")
ax.set_ylabel("FNO response error (f64-trained, λ=0)")
ax.set_title(f"Per-sample: no relationship\nPearson r={r_p:.2f}, Spearman ρ={r_s:.2f}  (N={len(born)}, n.s.)")
ax.set_xlim(0, max(born.max(), 1.05) * 1.02)
ax.set_ylim(0, fno.max() * 1.1)
ax.legend(fontsize=8, loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("results/born_vs_fno_scatter.png", dpi=150)
print(f"saved results/born_vs_fno_scatter.png  (r={r_p:.3f}, ρ={r_s:.3f})")