# scripts/plot_retrain_vs_truncate.py
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    M   = d["M"].astype(int)
    err = d["response_err"].astype(float)
    Ms = np.sort(np.unique(M))
    mean = np.array([err[M == m].mean() for m in Ms])
    sem  = np.array([err[M == m].std() / np.sqrt((M == m).sum()) for m in Ms])
    return Ms, mean, sem

def main():
    Mt, mt, st = load("results/truncate_sweep.csv")
    Mr, mr, sr = load("results/retrain_sweep.csv")

    print(f"{'M':>4} {'truncate':>10} {'retrain':>10} {'gap':>8}")
    for m, a, b in zip(Mt, mt, mr):
        print(f"{m:>4d} {a:>10.4f} {b:>10.4f} {a-b:>8.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(Mt, mt, yerr=st, fmt="o-", color="C3", lw=2, capsize=4,
                label="truncate-at-inference (clip trained-16 model)")
    ax.errorbar(Mr, mr, yerr=sr, fmt="s-", color="C0", lw=2, capsize=4,
                label="retrain-at-M (train fresh at each M)")
    ax.axhline(1.0, ls="--", color="grey", lw=1, alpha=0.6)
    ax.set_xlabel("modes $M$")
    ax.set_ylabel("response error (mean $\\pm$ SEM)")
    ax.set_title("Clip vs retrain: does the model adapt to fewer modes?")
    ax.set_xticks(Mt)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/response_clip_vs_retrain.png", dpi=150)
    print("saved results/response_clip_vs_retrain.png")

if __name__ == "__main__":
    main()