# scripts/plot_response_vs_n.py — Task A Block 4. DISPOSABLE.
import numpy as np, matplotlib.pyplot as plt

NS, SEEDS = [250, 500, 1000, 2000], [0, 1, 2]

def path(n, s, lam):
    if lam == 0.0 and n == 2000:                       # lam0 N2000 = canonical grid cache
        return f"results/resp_errs_200_seed{s}_M16_lam0.0.npy"
    return f"results/resp_errs_vsN_N{n}_seed{s}_M16_lam{lam}.npy"

def curve(lam):
    means, stds, meds = [], [], []
    for n in NS:
        per_seed = np.array([np.load(path(n, s, lam)).mean() for s in SEEDS])
        means.append(per_seed.mean()); stds.append(per_seed.std(ddof=1)); meds.append(np.median(per_seed))
    return np.array(means), np.array(stds), np.array(meds)

m0, s0, md0 = curve(0.0)
m1, s1, md1 = curve(0.1)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axhline(1.0, ls="--", lw=1, color="0.6", label="random-init floor (err=1)")
ax.errorbar(NS, m0, yerr=s0, marker="o", capsize=3, color="C3", label=r"forward-only ($\lambda=0$)")
ax.errorbar(NS, m1, yerr=s1, marker="s", capsize=3, color="C0", label=r"DINO ($\lambda=0.1$)")
ax.plot(NS, md0, ":", color="C3", alpha=0.5); ax.plot(NS, md1, ":", color="C0", alpha=0.5)  # medians
# annotate the headline crossing
ax.annotate("DINO @ N=250 beats\nforward-only @ N=2000",
            xy=(250, m1[0]), xytext=(330, 0.62), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
ax.set_xlabel("training samples N"); ax.set_ylabel("response error (rel. Frobenius)")
ax.set_title("Response fidelity vs data: supervision vs forward-only (M16)")
ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/response_vs_n.png", dpi=150)
print("saved -> results/response_vs_n.png")
print(f"gap ratios (lam0/lam0.1): {np.round(m0/m1, 2)}")