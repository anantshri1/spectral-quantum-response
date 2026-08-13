# scripts/ood_demo_figure.py — B3-3b-ii: linearized-prediction demo (2 ls × {error-vs-ε, spatial})
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.data_gen import ood_potential
from src.solver import evolve
from src.responses import jacobian_true, jacobian_fno
from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
N = 200
LS_ROWS = [0.05, 0.20]
EPS_DEMO = 1e-3                                  # locked from 3b-i valley
EPS_GRID = np.array([1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5])
phi = np.linspace(0, 2*np.pi, nx, endpoint=False)

d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

# models: forward-only (λ0) and heavy-DINO (λ10), seed0
m_l0  = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx",  cfg, n_modes=16)
m_l10 = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam10.0_final.eqx", cfg, n_modes=16)

key = jax.random.PRNGKey(1234)                  # D14: fixed random unit δV
dV0 = jax.random.normal(key, (nx,), dtype=V128.dtype); dV0 = dV0 / jnp.linalg.norm(dV0)

def row_data(ls):
    # D13=A: per-ls median-λ0-error representative sample
    resp_l0 = np.load(f"results/ood/resperr_seed0_M16_lam0.0_ls{ls:.2f}.npy")
    i_star = int(np.argmin(np.abs(resp_l0 - np.median(resp_l0))))
    V  = ood_potential(coeffs[i_star], V128[i_star], nx, ls)
    ps = psi0[i_star]
    psiT_V = evolve(ps, V, nx, dt, nt)          # unperturbed truth

    # three Jacobian actions on δV (position space): J·δV = predicted dψ_T per unit ε
    Jt_dV   = jacobian_true(ps, V, nx, dt, nt) @ dV0
    Jl0_dV  = jacobian_fno(m_l0,  ps, V) @ dV0
    Jl10_dV = jacobian_fno(m_l10, ps, V) @ dV0

    # error-vs-ε: ‖truth(ε) − [ψ_T(V)+ε·J·δV]‖ for each J
    def err_curve(J_dV):
        out = []
        for eps in EPS_GRID:
            truth = evolve(ps, V + float(eps)*dV0, nx, dt, nt)
            out.append(float(jnp.linalg.norm(truth - (psiT_V + float(eps)*J_dV))))
        return np.array(out)
    e_true = err_curve(Jt_dV); e_l0 = err_curve(Jl0_dV); e_l10 = err_curve(Jl10_dV)

    # spatial overlay at ε=EPS_DEMO: Re[ψ_T]
    truth_demo = evolve(ps, V + EPS_DEMO*dV0, nx, dt, nt)
    pred_true  = psiT_V + EPS_DEMO*Jt_dV
    pred_l0    = psiT_V + EPS_DEMO*Jl0_dV
    pred_l10   = psiT_V + EPS_DEMO*Jl10_dV

    # NEW: the perturbation RESPONSE (change in ψ_T) — the object the Jacobian predicts
    dpsi_truth = np.asarray(truth_demo - psiT_V)          # NEW: true change
    dpsi_true  = np.asarray(EPS_DEMO * Jt_dV)             # NEW: J_true·δV predicted change
    dpsi_l0    = np.asarray(EPS_DEMO * Jl0_dV)            # NEW
    dpsi_l10   = np.asarray(EPS_DEMO * Jl10_dV)           # NEW

    return dict(i=i_star, resp=resp_l0[i_star],
                e_true=e_true, e_l0=e_l0, e_l10=e_l10,
                truth=np.asarray(truth_demo), pt=np.asarray(pred_true),
                pl0=np.asarray(pred_l0), pl10=np.asarray(pred_l10),
                dtruth=dpsi_truth, dt_=dpsi_true, dl0=dpsi_l0, dl10=dpsi_l10)  # NEW

R = {ls: row_data(ls) for ls in LS_ROWS}

fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for r, ls in enumerate(LS_ROWS):
    D = R[ls]

    # --- col 0: error vs ε (log-log) ---
    ax = axes[r, 0]
    ax.loglog(EPS_GRID, D["e_true"], "o-", color="black",     lw=2, label="J_true")
    ax.loglog(EPS_GRID, D["e_l10"],  "s-", color="tab:green", lw=2, label="λ=10 (DINO)")
    ax.loglog(EPS_GRID, D["e_l0"],   "^-", color="tab:red",   lw=2, label="λ=0 (fwd-only)")
    ax.axvline(EPS_DEMO, ls=":", color="gray", lw=1)
    ax.set_xlabel("ε"); ax.set_ylabel("‖truth − linear pred‖")
    ax.set_title(f"ls={ls}: linearization error\n(sample {D['i']}, λ0 resp={D['resp']:.3f})", fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)

    # --- col 1: Re[ψ_T] overlay (honest: they overlap) ---
    ax = axes[r, 1]
    ax.plot(phi, D["truth"].real, color="black",     lw=3, alpha=0.5, label="truth")
    ax.plot(phi, D["pt"].real,    color="black",     lw=1, ls="--",   label="J_true")
    ax.plot(phi, D["pl10"].real,  color="tab:green", lw=1.5,          label="λ=10")
    ax.plot(phi, D["pl0"].real,   color="tab:red",   lw=1.5,          label="λ=0")
    ax.set_xlabel("φ"); ax.set_ylabel("Re ψ_T")
    ax.set_title(f"ls={ls}: predicted ψ_T (ε={EPS_DEMO})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # --- col 2: spatial residual (amplifies the peel) ---
    ax = axes[r, 2]
    tr = D["truth"].real
    ax.axhline(0, color="black", lw=2, alpha=0.4, label="truth (0)")
    ax.plot(phi, D["pt"].real  - tr, color="black",     lw=1, ls="--", label="J_true")
    ax.plot(phi, D["pl10"].real- tr, color="tab:green", lw=1.5,        label="λ=10")
    ax.plot(phi, D["pl0"].real - tr, color="tab:red",   lw=1.5,        label="λ=0")
    ax.set_xlabel("φ"); ax.set_ylabel("Re[pred − truth]")
    ax.set_title(f"ls={ls}: prediction residual (ε={EPS_DEMO})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # --- col 3: Argand plane of the RESPONSE dψ_T (both channels at once) ---
    ax = axes[r, 3]
    ax.plot(D["dtruth"].real, D["dtruth"].imag, color="black",     lw=3, alpha=0.45, label="true dψ_T")
    ax.plot(D["dt_"].real,    D["dt_"].imag,    color="black",     lw=1, ls="--",    label="J_true")
    ax.plot(D["dl10"].real,   D["dl10"].imag,   color="tab:green", lw=1.3,           label="λ=10")
    ax.plot(D["dl0"].real,    D["dl0"].imag,    color="tab:red",   lw=1.3,           label="λ=0")
    ax.set_xlabel("Re dψ_T"); ax.set_ylabel("Im dψ_T")
    ax.set_title(f"ls={ls}: response in complex plane (ε={EPS_DEMO})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")

fig.suptitle("Forward-only Jacobian mispredicts the perturbation response; DINO tracks it", y=1.01, fontsize=13)
fig.subplots_adjust(wspace=0.34, hspace=0.32)   # title-overlap fix: more horizontal room
fig.tight_layout(rect=[0, 0, 1, 0.99])
path = "results/ood/fig_linearized_demo.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"[save] {path}")

# numbers for the caption / sanity
for ls in LS_ROWS:
    D = R[ls]
    print(f"\nls={ls} sample {D['i']}  linearization error @ ε={EPS_DEMO}:")
    j = int(np.argmin(np.abs(EPS_GRID - EPS_DEMO)))
    print(f"  J_true={D['e_true'][j]:.3e}  λ10={D['e_l10'][j]:.3e}  λ0={D['e_l0'][j]:.3e}"
          f"   (λ0/true ratio = {D['e_l0'][j]/D['e_true'][j]:.1f}×)")