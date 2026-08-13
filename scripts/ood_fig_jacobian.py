# scripts/ood_fig_jacobian.py — B3-3c: Jacobian hue maps (|J| + masked phase), the mechanism figure
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.data_gen import ood_potential
from src.responses import jacobian_fno, mode_coupling_matrix
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, N = cfg.nx, 200
LS_ROWS = [0.05, 0.20]

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])
k0_all = d["p_k0"]

m_l0  = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx",  cfg, n_modes=16)
m_l10 = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam10.0_final.eqx", cfg, n_modes=16)

sh = np.fft.fftshift                         # center k=0
kax = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0/nx)).astype(int)   # signed k axis for ticks
ext = [kax[0], kax[-1], kax[-1], kax[0]]     # q on x, k on y (imshow origin='upper')

def row_J(ls):
    # same sample as 3b (per-ls median λ0 resp) for consistency across figures
    resp_l0 = np.load(f"results/ood/resperr_seed0_M16_lam0.0_ls{ls:.2f}.npy")
    i = int(np.argmin(np.abs(resp_l0 - np.median(resp_l0))))
    V = ood_potential(coeffs[i], V128[i], nx, ls)
    Jt  = jnp.asarray(np.load(f"results/ood/Jtrue_kq_ls{ls:.2f}.npy")[i])          # cached kq
    Jl0  = mode_coupling_matrix(jacobian_fno(m_l0,  psi0[i], V))                   # same transform
    Jl10 = mode_coupling_matrix(jacobian_fno(m_l10, psi0[i], V))
    return i, int(k0_all[i]), np.asarray(Jt), np.asarray(Jl0), np.asarray(Jl10)

rows = {ls: row_J(ls) for ls in LS_ROWS}

# ================= FIGURE 1: MAGNITUDE =================
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for r, ls in enumerate(LS_ROWS):
    i, k0, Jt, Jl0, Jl10 = rows[ls]
    vmax = np.abs(Jt).max()                  # shared scale per row = true's scale (honest)
    panels = [("|J_true|", np.abs(Jt), vmax, "viridis"),
              ("|J_FNO λ=0|", np.abs(Jl0), vmax, "viridis"),
              ("|J_FNO λ=10|", np.abs(Jl10), vmax, "viridis"),
              ("|λ0 − true|", np.abs(Jl0 - Jt), vmax, "magma"),
              ("|λ10 − true|", np.abs(Jl10 - Jt), vmax, "magma")]
    for c, (title, M, vm, cmap) in enumerate(panels):
        ax = axes[r, c]
        im = ax.imshow(sh(M), extent=ext, cmap=cmap, vmin=0, vmax=vm, aspect="equal")
        ax.set_title(f"ls={ls}: {title}", fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("k")
        # mark the Born band k−q=k0 (a diagonal line)
        ax.plot([kax[0], kax[-1]], [kax[0]+k0, kax[-1]+k0], color="cyan", lw=0.7, ls=":", alpha=0.7)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Response Jacobian |J(k,q)|: forward-only corrupts in-band coupling; DINO restores it "
             "(cyan = Born band k−q=k0)", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig("results/ood/fig_jacobian_mag.png", dpi=150, bbox_inches="tight")
print("[save] results/ood/fig_jacobian_mag.png")

# ================= FIGURE 2: PHASE (magnitude-masked) =================
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for r, ls in enumerate(LS_ROWS):
    i, k0, Jt, Jl0, Jl10 = rows[ls]
    thr = 0.05 * np.abs(Jt).max()            # mask: show phase only where |J_true| has real support
    mask = np.abs(Jt) < thr
    for c, (title, J) in enumerate([("arg J_true", Jt), ("arg λ=0", Jl0), ("arg λ=10", Jl10)]):
        ax = axes[r, c]
        ph = np.angle(J)
        ph_masked = np.ma.masked_where(mask, ph)      # gray out the empty region
        im = ax.imshow(sh(ph_masked), extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
        ax.set_title(f"ls={ls}: {title}", fontsize=10)
        ax.set_xlabel("q"); ax.set_ylabel("k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Response Jacobian phase arg J(k,q), masked to |J_true|>5% (gray = no support)", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig("results/ood/fig_jacobian_phase.png", dpi=150, bbox_inches="tight")
print("[save] results/ood/fig_jacobian_phase.png")