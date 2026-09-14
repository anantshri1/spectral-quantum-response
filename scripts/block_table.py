# scripts/block_table.py — §6 replacement: (S,X,B)x(S,X,B) error decomposition.
import os

import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from configs.default import Config

cfg = Config(); nx = cfg.nx
kabs  = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)
bands = {"S(<12)": kabs < 12, "X[12,16)": (kabs >= 12) & (kabs < 16), "B(>=16)": kabs >= 16}
order = list(bands)

Jtrue = np.load("results/Jtrue_kq_200.npy")            # (N,nx,nx)

def block_fracs(J):                                    # 3x3: % of sum|J|^2 by (output-k band, input-q band)
    tot = (np.abs(J)**2).sum()
    F = np.zeros((3, 3))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            blk = J[:, bands[a], :][:, :, bands[b]]
            F[i, j] = (np.abs(blk)**2).sum() / tot
    return F

def show(name, F):
    print(f"\n{name}\n  rows = output |k| band, cols = input |q| band  (% of total power)")
    print("            " + "".join(f"{b:>11}" for b in order))
    for i, a in enumerate(order):
        print(f"  {a:>9} " + "".join(f"{100*F[i,j]:11.3f}" for j in range(3)))
    print(f"    -> |k|>=16 marginal (row B, any q): {100*F[2].sum():.2f}%   in-band |k|<12: {100*F[0].sum():.2f}%")

show("J_true power", block_fracs(Jtrue))
for M in [12, 16, 24]:
    for seed in [0, 1, 2]:
        ckpt = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam0.0_final.eqx"
        if not os.path.exists(ckpt):
            print(f"[missing] {ckpt} — skipping"); continue
        show(f"M{M} seed{seed} Jfno-Jtrue error", block_fracs(np.load(f"results/Jfno_kq_seed{seed}_M{M}.npy") - Jtrue))
    #show(f"M{M} error  ||Jfno-Jtrue||^2 by block", block_fracs(np.load(f"results/Jfno_kq_seed{seed}_M{M}.npy") - Jtrue))