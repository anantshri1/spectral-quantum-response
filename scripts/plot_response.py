# scripts/plot_response_sweep.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    d = np.genfromtxt("results/response_sweep.csv", delimiter=",",
                      names=True, dtype=None, encoding="utf-8")
    k0  = d["k0"].astype(int)
    err = d["response_err"].astype(float)

    ks = np.sort(np.unique(k0))                 # signed, {-4,...,4}
    means = np.array([err[k0 == k].mean() for k in ks])
    stds  = np.array([err[k0 == k].std()  for k in ks])
    ns    = np.array([(k0 == k).sum()     for k in ks])
    sems  = stds / np.sqrt(ns)                  # std error of the mean per bin

    # --- table (read this before trusting the plot) ---
    print(f"{'k0':>4} {'n':>4} {'mean':>8} {'std':>8} {'sem':>8}")
    for k, n, m, s, e in zip(ks, ns, means, stds, sems):
        print(f"{k:>+4d} {n:>4d} {m:>8.4f} {s:>8.4f} {e:>8.4f}")

    # --- signed-symmetry check: does +k match -k? ---
    print("\nsign symmetry (|k0| paired):")
    for a in [1, 2, 3, 4]:
        mp = err[k0 ==  a].mean()
        mm = err[k0 == -a].mean()
        print(f"  |k0|={a}:  +{a}={mp:.4f}  -{a}={mm:.4f}  diff={mp-mm:+.4f}")

    # --- plot: signed k0, error bars = SEM ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ks, means, yerr=sems, fmt="o-", capsize=4, lw=1.5,
                label="response rel-Frobenius")
    ax.set_xlabel(r"carrier momentum $k_0$")
    ax.set_ylabel("response error (mean $\\pm$ SEM)")
    ax.set_title("Response fidelity vs wavepacket momentum")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/response_vs_k0.png", dpi=150)
    print("\nsaved results/response_vs_k0.png")

def decompose():
    d = np.genfromtxt("results/response_sweep.csv", delimiter=",",
                      names=True, dtype=None, encoding="utf-8")
    ak0 = np.abs(d["k0"].astype(int))            # fold to |k0| (symmetry earned)
    num = d["num"].astype(float)                 # ||J_fno - J_true||
    den = d["den"].astype(float)                 # ||J_true||
    err = d["response_err"].astype(float)        # num/den

    print(f"{'|k0|':>4} {'n':>4} {'num':>9} {'den':>9} {'ratio':>8}")
    for a in np.sort(np.unique(ak0)):
        m = ak0 == a
        print(f"{a:>4d} {m.sum():>4d} {num[m].mean():>9.4f} "
              f"{den[m].mean():>9.4f} {err[m].mean():>8.4f}")

def plot_folded():
    d = np.genfromtxt("results/response_sweep.csv", delimiter=",",
                      names=True, dtype=None, encoding="utf-8")
    ak0 = np.abs(d["k0"].astype(int))
    err = d["response_err"].astype(float)

    ks = np.sort(np.unique(ak0))
    means = np.array([err[ak0 == k].mean() for k in ks])
    sems  = np.array([err[ak0 == k].std() / np.sqrt((ak0 == k).sum()) for k in ks])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(ks, means, yerr=sems, fmt="o-", capsize=4, lw=1.6, color="C3")
    ax.set_xlabel(r"$|k_0|$ (carrier momentum magnitude)")
    ax.set_ylabel("response error (mean $\\pm$ SEM)")
    ax.set_title("Response fidelity is non-monotonic in $|k_0|$")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/response_vs_absk0.png", dpi=150)
    print("saved results/response_vs_absk0.png")

if __name__ == "__main__":
    main()
    print()
    decompose()
    plot_folded()