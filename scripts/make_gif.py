# scripts/make_gif.py — Phase 6 closing deliverable: error-map collapse over lambda.
# |J_fno(lambda) - J_true| for sample 0, swept lambda in {0,0.1,0.3,1,3,10} (M=16 seed 0).
# fftshift-centered, shared colorscale keyed to lambda=0 so the collapse reads.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, mode_coupling_matrix
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt

SAMPLE = 0
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
CKPTS = [f"checkpoints/dino_N2000_seed0_M16_lam{l}_final.eqx" for l in LAMBDAS]

if __name__ == "__main__":
    cfg = Config()
    psi0, V, _ = load_split("data/test.npz", nx=cfg.nx)
    psi0, V = psi0.astype(jnp.complex128), V.astype(jnp.float64)
    dt, nx, nt = cfg.t_end / cfg.nt, cfg.nx, cfg.nt
    p0, V0 = psi0[SAMPLE], V[SAMPLE]

    # J_true once
    Jt = np.fft.fftshift(np.asarray(mode_coupling_matrix(
        jacobian_true(p0, V0, nx, dt, nt))))

    # error map + scalar rel-Frobenius error per lambda
    frames, errs = [], []
    for ck in CKPTS:
        m = load_fno_f64_ckpt(ck, cfg)
        Jf = np.fft.fftshift(np.asarray(mode_coupling_matrix(
            jacobian_fno(m, p0, V0))))
        e = np.abs(Jf - Jt)
        frames.append(e)
        errs.append(float(np.linalg.norm(Jf - Jt) / np.linalg.norm(Jt)))
        print(f"  lambda={ck.split('lam')[1].split('_')[0]:>4} "
              f"sample-0 err={errs[-1]:.4f}")

    kmax = nx // 2
    ext = [-kmax, kmax, -kmax, kmax]
    vmax = frames[0].max()   # key to lambda=0 so collapse -> black reads

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(frames[0], origin="lower", extent=ext, cmap="magma",
                   vmin=0, vmax=vmax, aspect="equal")
    ax.set_xlabel("q (mode)"); ax.set_ylabel("k (mode)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("|J_fno − J_true|")
    title = ax.set_title("")

    # frame order: hold first, sweep, hold last (via repeated indices)
    order = [0, 0, 1, 2, 3, 4, 5, 5, 5]   # slow start + end hold

    def update(fi):
        i = order[fi]
        im.set_data(frames[i])
        title.set_text(f"sample 0 (k₀=+4)   λ = {LAMBDAS[i]}   "
                       f"response err = {errs[i]:.3f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(order), blit=False)
    anim.save("results/dino_error_collapse.gif",
              writer=PillowWriter(fps=1.2), dpi=110)
    print("\nsaved results/dino_error_collapse.gif")