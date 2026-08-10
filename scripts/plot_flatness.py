# scripts/plot_flatness.py — 7A Block 0: FNO error is decoupled from perturbativeness
import numpy as np
import matplotlib.pyplot as plt

born = np.load("results/born_errs_200.npy")
fno  = np.load("results/fno_errs_200_f64trained.npy")

# split samples by how perturbative they are (Born error = non-perturbativeness proxy)
med = np.median(born)
pert    = born < med      # perturbative: PT works well
nonpert = born >= med     # non-perturbative: PT struggles

def stat(a, m):           # mean ± SEM within a bin
    x = a[m]; return x.mean(), x.std()/np.sqrt(len(x))

b_lo, b_lo_e = stat(born, pert);    b_hi, b_hi_e = stat(born, nonpert)
f_lo, f_lo_e = stat(fno,  pert);    f_hi, f_hi_e = stat(fno,  nonpert)

print(f"n per bin: {pert.sum()} / {nonpert.sum()}  (median Born = {med:.3f})")
print(f"  Born:  perturbative {b_lo:.3f}±{b_lo_e:.3f}   non-pert {b_hi:.3f}±{b_hi_e:.3f}   "
      f"swing {b_hi-b_lo:+.3f}")
print(f"  FNO :  perturbative {f_lo:.3f}±{f_lo_e:.3f}   non-pert {f_hi:.3f}±{f_hi_e:.3f}   "
      f"swing {f_hi-f_lo:+.3f}")

fig, ax = plt.subplots(figsize=(5.4, 4.6))
x = np.arange(2); w = 0.38
ax.bar(x - w/2, [b_lo, b_hi], w, yerr=[b_lo_e, b_hi_e], capsize=4,
       label="Born (1st-order PT)", color="#dd6b20")
ax.bar(x + w/2, [f_lo, f_hi], w, yerr=[f_lo_e, f_hi_e], capsize=4,
       label="FNO (f64, λ=0)", color="#2b6cb0")
ax.axhline(1.0, ls=":", lw=1, color="#a0aec0")   # random-response wall
ax.set_xticks(x); ax.set_xticklabels(["perturbative\n(Born < median)",
                                       "non-perturbative\n(Born ≥ median)"])
ax.set_ylabel("response error")
ax.set_title("FNO error is flat across perturbativeness;\nBorn is not — they track different things")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("results/flatness_bins.png", dpi=150)
print("saved results/flatness_bins.png")