# scripts/plot_clip_vs_retrain_f64.py — f64, 3-seed, across-seed spread.
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

def load(path):
    """Return {arm: (Ms, seed_means[M], across_seed_mean[M], across_seed_std[M])}.
    Per seed we first average response_err over the 200 samples, THEN take
    mean/std across the 3 seeds — so the error bar is across-seed spread, not
    within-seed sample SEM."""
    # accumulate per (arm, M, seed): list of sample errors
    acc = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["arm"], int(row["M"]), int(row["seed"]))
            acc[key].append(float(row["response_err"]))
    # per-seed mean over samples
    seed_mean = {k: np.mean(v) for k, v in acc.items()}
    arms = sorted({k[0] for k in seed_mean})
    out = {}
    for arm in arms:
        Ms = sorted({k[1] for k in seed_mean if k[0] == arm})
        means, stds = [], []
        for M in Ms:
            per_seed = [seed_mean[(arm, M, s)]
                        for s in sorted({k[2] for k in seed_mean
                                         if k[0] == arm and k[1] == M})]
            means.append(np.mean(per_seed))
            stds.append(np.std(per_seed))   # across-seed spread
        out[arm] = (np.array(Ms), np.array(means), np.array(stds))
    return out

def main():
    data = load("results/response_clip_vs_retrain_f64.csv")
    Mt, mt, st = data["clip"]
    Mr, mr, sr = data["retrain"]

    print(f"{'M':>4} {'clip':>18} {'retrain':>18} {'gap':>8}")
    for m, a, sa, b, sb in zip(Mt, mt, st, mr, sr):
        print(f"{m:>4d}  {a:>7.4f} ± {sa:.4f}   {b:>7.4f} ± {sb:.4f}   {a-b:>7.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(Mt, mt, yerr=st, fmt="o-", color="C3", lw=2, capsize=4,
                label="truncate-at-inference (clip trained-16 model)")
    ax.errorbar(Mr, mr, yerr=sr, fmt="s-", color="C0", lw=2, capsize=4,
                label="retrain-at-M (train fresh f64 at each M)")
    ax.axhline(1.0, ls="--", color="grey", lw=1, alpha=0.6,
               label="untrained floor")
    ax.set_xlabel("modes $M$")
    ax.set_ylabel("response error (mean $\\pm$ across-seed std)")
    ax.set_title("Clip vs retrain (f64, 3 seeds): the model adapts to fewer modes")
    ax.set_xticks(Mt)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/response_clip_vs_retrain_f64.png", dpi=150)
    print("saved results/response_clip_vs_retrain_f64.png")

if __name__ == "__main__":
    main()