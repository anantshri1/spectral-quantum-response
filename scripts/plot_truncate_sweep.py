# scripts/plot_truncate_sweep.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    d = np.genfromtxt("results/truncate_sweep.csv", delimiter=",",
                      names=True, dtype=None, encoding="utf-8")
    idx = d["idx"].astype(int)
    ak0 = np.abs(d["k0"].astype(int))
    M   = d["M"].astype(int)
    err = d["response_err"].astype(float)

    Ms = np.sort(np.unique(M))

    # --- aggregate: mean over all samples, per M ---
    agg_mean = np.array([err[M == m].mean() for m in Ms])
    agg_sem  = np.array([err[M == m].std() / np.sqrt((M == m).sum()) for m in Ms])

    print(f"{'M':>4} {'mean':>8} {'sem':>8}")
    for m, mu, se in zip(Ms, agg_mean, agg_sem):
        print(f"{m:>4d} {mu:>8.4f} {se:>8.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.8))

    # per-|k0| curves (thin, background)
    for a in np.sort(np.unique(ak0)):
        ma = ak0 == a
        ys = np.array([err[ma & (M == m)].mean() for m in Ms])
        ax.plot(Ms, ys, "o-", lw=1, alpha=0.5, label=f"|k0|={a}")

    # aggregate (thick, foreground)
    ax.errorbar(Ms, agg_mean, yerr=agg_sem, fmt="s-", color="k", lw=2.2,
                capsize=4, label="all samples", zorder=10)

    ax.axhline(1.0, ls="--", color="grey", lw=1, alpha=0.7)   # "worse than zero" line
    ax.axvline(16, ls=":", color="C3", lw=1, alpha=0.7)        # training n_modes
    ax.set_xlabel("modes retained $M$ (truncate-at-inference)")
    ax.set_ylabel("response error (mean $\\pm$ SEM)")
    ax.set_title("Response fidelity collapses as modes are removed")
    ax.set_xticks(Ms)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("results/response_vs_modes_truncate.png", dpi=150)
    print("saved results/response_vs_modes_truncate.png")

if __name__ == "__main__":
    main()