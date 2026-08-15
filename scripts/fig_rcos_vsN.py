# scripts/fig_rcos_vsN.py — Block-4 figure: decompose response-vs-N into (r, cos)
import numpy as np
import os
from scripts.eval_align_taskb import decompose_for
import matplotlib.pyplot as plt
seeds = (0, 1, 2)                 # lam0 vsN leg is 3-seed
Ns    = (250, 500, 1000, 2000)

# --- append after Block A's mean_std definitions ---
def mean_std_lam(field, lam):
    M = np.array([[np.load(f"results/align_vsN_N{n}_seed{s}_M16_lam{lam}.npz")[field].mean()
                   for s in seeds] for n in Ns])
    return M.mean(1), M.std(1, ddof=1)

r0,  r0s   = mean_std_lam("r",   "0.0");   c0,  c0s  = mean_std_lam("cos", "0.0")
r1,  r1s   = mean_std_lam("r",   "0.1");   c1,  c1s  = mean_std_lam("cos", "0.1")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
C0, C1 = 'tab:red', 'tab:blue'   # lam0 = red (worse), lam0.1 = blue (DINO)

# (a) squeeze — 4 curves, shared y=1
axL.errorbar(Ns, r0, r0s, marker='o', color=C0, capsize=3, label=r'$r$, $\lambda{=}0$')
axL.errorbar(Ns, c0, c0s, marker='s', color=C0, ls='--', capsize=3, label=r'$\cos$, $\lambda{=}0$')
axL.errorbar(Ns, r1, r1s, marker='o', color=C1, capsize=3, label=r'$r$, $\lambda{=}0.1$')
axL.errorbar(Ns, c1, c1s, marker='s', color=C1, ls='--', capsize=3, label=r'$\cos$, $\lambda{=}0.1$')
axL.axhline(1.0, ls=':', color='k', lw=1, alpha=0.6)
axL.text(Ns[-1], 1.005, 'perfect', ha='right', va='bottom', fontsize=8, alpha=0.7)
axL.set_xscale('log', base=2); axL.set_xticks(Ns); axL.set_xticklabels(Ns)
axL.set_xlabel('training samples $N$'); axL.set_ylabel('$r$   and   $\\cos$')
axL.set_title('(a) DINO ($\\lambda{=}0.1$) at $N{=}250$ beats\nforward-only ($\\lambda{=}0$) at $N{=}2000$ on both axes')
axL.legend(fontsize=7.5, ncol=2, loc='lower right')

# (b) gap-to-perfect, log-y — |r-1| so both regimes plot positive
axR.errorbar(Ns, np.abs(r0-1), r0s, marker='o', color=C0, capsize=3, label=r'$|r{-}1|$, $\lambda{=}0$')
axR.errorbar(Ns, 1-c0,        c0s, marker='s', color=C0, ls='--', capsize=3, label=r'$1{-}\cos$, $\lambda{=}0$')
axR.errorbar(Ns, np.abs(r1-1), r1s, marker='o', color=C1, capsize=3, label=r'$|r{-}1|$, $\lambda{=}0.1$')
axR.errorbar(Ns, 1-c1,        c1s, marker='s', color=C1, ls='--', capsize=3, label=r'$1{-}\cos$, $\lambda{=}0.1$')
axR.set_xscale('log', base=2); axR.set_yscale('log')
axR.set_xticks(Ns); axR.set_xticklabels(Ns)
axR.set_xlabel('training samples $N$'); axR.set_ylabel('gap to perfect (log)')
axR.set_title('(b) $\\lambda{=}0$: scale gap dominant (solid above dashed)\n$\\lambda{=}0.1$: axes balanced, ${\\sim}10\\times$ smaller')
axR.legend(fontsize=7.5, ncol=2)

fig.suptitle(r'Response-vs-$N$ decomposed by supervision: derivative training buys a '
             r'better-conditioned Jacobian at every $N$ (M16, 3-seed)', fontsize=10)
fig.tight_layout()
fig.savefig('fig_rcos_vsN_dino.png', dpi=150, bbox_inches='tight')
print("saved -> fig_rcos_vsN_dino.png")

# --- Block C: across-N decomposition, lam0.1 M16 (DINO) ---
Ns = (250, 500, 1000, 2000)
seeds = (0, 1, 2)
for seed in seeds:
    for n in Ns:
        save = f"results/align_vsN_N{n}_seed{seed}_M16_lam0.1.npz"
        if os.path.exists(save):
            continue
        ckpt = f"checkpoints/dino_N{n}_seed{seed}_M16_lam0.1_final.eqx"
        d = decompose_for(ckpt, n_modes=None)
        np.savez(save, **d)
        print(f"[C] N{n} seed{seed} r={d['r'].mean():.4f} cos={d['cos'].mean():.4f} -> saved")

def gridN_lam(field, n, lam):
    return np.array([np.load(f"results/align_vsN_N{n}_seed{s}_M16_lam{lam}.npz")[field].mean()
                     for s in seeds])

print(f"\n{'N':>5} {'r (lam0.1)':>15} {'cos (lam0.1)':>15}")
for n in Ns:
    r, c = gridN_lam("r", n, "0.1"), gridN_lam("cos", n, "0.1")
    print(f"{n:5d} {r.mean():.4f}±{r.std(ddof=1):.4f}  {c.mean():.4f}±{c.std(ddof=1):.4f}")