import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, mode_coupling_matrix
from scripts.compute_response import load_fno_x64

def pick_samples(k0_all):
    """canonical sample 0, plus first hit at k0=0, |k0|=1, |k0|=4."""
    idx = {"sample0 (k0=+4)": 0}
    for target, label in [(0, "k0=0"), (1, "|k0|=1"), (-4, "|k0|=-4")]:
        hits = np.where(k0_all == target)[0]
        if len(hits):
            idx[f"{label} (idx{hits[0]})"] = int(hits[0])
    return idx

if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = d["u0_re_128"], d["u0_im_128"], d["V_128"], d["p_k0"]

    samples = pick_samples(k0_all)
    print("samples:", samples)

    cfg12 = Config(); cfg12.n_modes = 12
    cfg16 = Config(); cfg16.n_modes = 16
    model12 = load_fno_x64("checkpoints/fno_N2000_seed0_M12_final.eqx", cfg12)
    model16 = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg16)

    fig, axes = plt.subplots(len(samples), 3, figsize=(13, 3.6 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]

    for row, (label, i) in enumerate(samples.items()):
        psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
        V    = jnp.asarray(V_all[i], dtype=jnp.float64)

        J_true_pos = jacobian_true(psi0, V, nx, dt, nt)
        J12_pos    = jacobian_fno(model12, psi0, V)
        J16_pos    = jacobian_fno(model16, psi0, V)

        J_true_kq = mode_coupling_matrix(J_true_pos)
        J12_kq    = mode_coupling_matrix(J12_pos)
        J16_kq    = mode_coupling_matrix(J16_pos)

        err12 = np.abs(np.asarray(J12_kq - J_true_kq))
        err16 = np.abs(np.asarray(J16_kq - J_true_kq))
        vmax  = max(err12.max(), err16.max())   # SHARED scale — direct visual comparison

        for col, (mat, title, cutoff) in enumerate([
            (np.abs(np.asarray(J_true_kq)), f"{label}\n|J_true|", None),
            (err12, "|J_fno(M=12) - J_true|", 12),
            (err16, "|J_fno(M=16) - J_true|", 16),
        ]):
            ax = axes[row, col]
            n = mat.shape[0]
            im = ax.imshow(mat, cmap = "viridis", vmax=(None if col == 0 else vmax),
                           origin="lower")
            ax.set_xlim(0, n - 1)
            ax.set_ylim(0, n - 1)              # HARD clamp — line can't push these out
            ax.set_title(title, fontsize=9)
            if cutoff is not None:
                # k - q = cutoff  ->  row = col + cutoff  (and row = col - cutoff)
                # in RAW index space (0..n-1), no centering assumption
                xs = np.arange(n)
                for off in (cutoff, -cutoff):
                    ys = xs + off
                    valid = (ys >= 0) & (ys < n)
                    ax.plot(xs[valid], ys[valid], color="cyan", lw=0.8,
                           alpha=0.7, ls="--")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig("results/error_maps_M12_vs_M16.png", dpi=140)
    print("saved results/error_maps_M12_vs_M16.png")