# scripts/plot_triptych.py — Phase 6 hero figure: J(k,q) recovery under DINO.
# Row 1: |J_true|, |J_fno(l=0)|, |J_fno(l=10)|  (shared scale, magnitude story)
# Row 2: |J_fno(l=0) - J_true|, |J_fno(l=10) - J_true|  (shared scale, error story)
# Sample 0 (k0=+4), continuity with all prior figures.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, mode_coupling_matrix
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt

SAMPLE = 0
CKPT_L0  = "checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx"
CKPT_L10 = "checkpoints/dino_N2000_seed0_M16_lam10.0_final.eqx"

if __name__ == "__main__":
    cfg = Config()
    psi0, V, _ = load_split("data/test.npz", nx=cfg.nx)
    psi0, V = psi0.astype(jnp.complex128), V.astype(jnp.float64)
    dt, nx, nt = cfg.t_end / cfg.nt, cfg.nx, cfg.nt
    p0, V0 = psi0[SAMPLE], V[SAMPLE]

    # --- J_true (k,q) ---
    Jt_pos = jacobian_true(p0, V0, nx, dt, nt)
    Jt_kq  = np.asarray(mode_coupling_matrix(Jt_pos))

    # --- J_fno at lambda=0 and lambda=10 (k,q) ---
    m0  = load_fno_f64_ckpt(CKPT_L0,  cfg)
    m10 = load_fno_f64_ckpt(CKPT_L10, cfg)
    J0_kq  = np.asarray(mode_coupling_matrix(jacobian_fno(m0,  p0, V0)))
    J10_kq = np.asarray(mode_coupling_matrix(jacobian_fno(m10, p0, V0)))

    # fftshift so k=0 is centered (physicist-standard k-space view)
    def shift(M): return np.fft.fftshift(M)
    aJt, aJ0, aJ10 = map(lambda M: np.abs(shift(M)), [Jt_kq, J0_kq, J10_kq])
    e0  = np.abs(shift(J0_kq)  - shift(Jt_kq))
    e10 = np.abs(shift(J10_kq) - shift(Jt_kq))

    kmax = nx // 2
    ext = [-kmax, kmax, -kmax, kmax]   # q on x, k on y, both centered

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # --- row 1: magnitude, shared scale keyed to J_true max ---
    vmax = aJt.max()
    for ax, M, t in zip(axes[0],
                        [aJt, aJ0, aJ10],
                        ["|J_true|", "|J_fno| (λ=0)", "|J_fno| (λ=10)"]):
        im = ax.imshow(M, origin="lower", extent=ext, cmap="viridis",
                       vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(t); ax.set_xlabel("q (mode)"); ax.set_ylabel("k (mode)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- row 2: error maps, shared scale ---
    emax = e0.max()   # key both to the WORSE (l=0) so l=10 shows as dark
    for ax, M, t in zip(axes[1][:2],
                        [e0, e10],
                        [f"|J_fno−J_true| (λ=0)  err=0.58",
                         f"|J_fno−J_true| (λ=10) err=0.057"]):
        im = ax.imshow(M, origin="lower", extent=ext, cmap="magma",
                       vmin=0, vmax=emax, aspect="equal")
        ax.set_title(t); ax.set_xlabel("q (mode)"); ax.set_ylabel("k (mode)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[1][2].axis("off")   # third slot empty in error row

    plt.suptitle(f"DINO recovers the mode-coupling operator J(k,q) — sample {SAMPLE} (k₀=+4)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("results/dino_triptych.png", dpi=150, bbox_inches="tight")
    print("saved results/dino_triptych.png")