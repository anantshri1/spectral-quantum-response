# scripts/eps_sweep.py — item (6): linearized validity radius. x64 FIRST.
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt
from src.solver import evolve

cfg = Config(); nx, nt = cfg.nx, cfg.nt; dt = cfg.t_end / cfg.nt
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
m0  = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", cfg)
m01 = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.1_final.eqx", cfg)

# --- directional derivatives via jvp (fd_check / jvp_fno_batched house primitive) ---
def jvp_true(p, v, dV):
    f = lambda V_: evolve(p, V_, nx, dt, nt)                       # complex (nx,)
    return jax.jvp(f, (v,), (dV,))[1]
def jvp_fno(model, p, v, dV):
    f = lambda V_: (lambda o: o[:, 0] + 1j*o[:, 1])(model(p, V_))  # recombine to complex
    return jax.jvp(f, (v,), (dV,))[1]

def unit_dir(key):
    u = jax.random.normal(key, (nx,), dtype=jnp.float64)
    return u / jnp.linalg.norm(u)                                 # ||dV||=1 (fd_check convention)

def lin_err(dpsi_true, Jdv, eps):
    return float(jnp.linalg.norm(dpsi_true - eps*Jdv) / jnp.linalg.norm(dpsi_true))

# --- Block 1 gate: one sample, one direction, small eps ---
s = 0; p, v = psi0[s], V[s]
dV   = unit_dir(jax.random.PRNGKey(0))
psiV = evolve(p, v, nx, dt, nt)                                   # reused across eps
Jt_dv, Jf0_dv = jvp_true(p, v, dV), jvp_fno(m0, p, v, dV)
eps  = 1e-4
dpsi = evolve(p, v + eps*dV, nx, dt, nt) - psiV
e_true, e_fno0 = lin_err(dpsi, Jt_dv, eps), lin_err(dpsi, Jf0_dv, eps)
print(f"[gate] eps={eps:.0e}  err_Jtrue={e_true:.2e}  (linear regime -> ~1e-4)")
print(f"[gate] eps={eps:.0e}  err_Jfno0={e_fno0:.3f}  (plateau ~ directional response err)")
assert e_true < 1e-2, "J_true NOT in linear regime at small eps — roundoff/machinery issue"
assert e_fno0 > 0.1 and e_fno0 > 20*e_true, "J_fno0 should plateau FAR above J_true's linear err"
print("[gate] PASSED — J_true linearizes; J_fno(lam0) does not. That contrast IS the figure.")

# --- Block 2: full sweep + validity-radius figure ---
import matplotlib.pyplot as plt

solve = jax.jit(lambda p, v: evolve(p, v, nx, dt, nt))
jt    = jax.jit(lambda p, v, dV: jax.jvp(lambda V_: evolve(p, V_, nx, dt, nt), (v,), (dV,))[1])
def _mk(model):
    def g(p, v, dV):
        f = lambda V_: (lambda o: o[:, 0] + 1j*o[:, 1])(model(p, V_))
        return jax.jvp(f, (v,), (dV,))[1]
    return jax.jit(g)
jf0, jf01 = _mk(m0), _mk(m01)

eps_grid = np.logspace(-4, 0, 17)
n_samp, n_dir = 20, 4
labels = ["J_true", r"J_fno $\lambda$=0", r"J_fno $\lambda$=0.1"]
errs = np.full((n_samp*n_dir, len(eps_grid), 3), np.nan)

vnorms, row = [], 0
for s in range(n_samp):
    p, v = psi0[s], V[s]; vnorms.append(float(jnp.linalg.norm(v)))
    psiV = solve(p, v)
    for d in range(n_dir):
        dV   = unit_dir(jax.random.PRNGKey(1000*s + d))
        Jdvs = [jt(p, v, dV), jf0(p, v, dV), jf01(p, v, dV)]
        for j, eps in enumerate(eps_grid):
            dpsi = solve(p, v + eps*dV) - psiV
            den  = float(jnp.linalg.norm(dpsi))
            for k, Jdv in enumerate(Jdvs):
                errs[row, j, k] = float(jnp.linalg.norm(dpsi - eps*Jdv)) / den
        row += 1
    print(f"  sample {s:2d} done", end="\r")

np.savez("results/eps_sweep_arrays.npz", eps=eps_grid, errs=errs, labels=labels)
mean, std = np.nanmean(errs, axis=0), np.nanstd(errs, axis=0)
print(f"\nmean ||V|| = {float(np.mean(vnorms)):.3f}  (eps beyond this exceeds the potential itself)")

def eps_star(curve, tol):
    below = curve < tol
    if below.all():  return f">{eps_grid[-1]:.3g}"
    if not below[0]: return f"<{eps_grid[0]:.3g}"          # plateau already >= tol
    idx = int(np.argmax(~below))
    x0, x1 = np.log10(eps_grid[idx-1]), np.log10(eps_grid[idx])
    y0, y1 = np.log10(curve[idx-1]),   np.log10(curve[idx])
    return f"{10**(x0 + (np.log10(tol)-y0)*(x1-x0)/(y1-y0)):.3g}"

print("\nvalidity radius eps* (largest eps with mean err < tol):")
print(f"{'operator':16s}" + "".join(f"tol={t:<7.2g}" for t in [0.05, 0.1, 0.15, 0.2]))
for k, lab in enumerate(["J_true", "J_fno lam0", "J_fno lam0.1"]):
    print(f"{lab:16s}" + "".join(f"{eps_star(mean[:,k], t):<11s}" for t in [0.05, 0.1, 0.15, 0.2]))

colors = ["#333333", "#1f77b4", "#2ca02c"]
fig, ax = plt.subplots(figsize=(6.6, 5.0))
for k, (lab, c) in enumerate(zip(labels, colors)):
    ax.plot(eps_grid, mean[:, k], "-o", ms=3.5, color=c, label=lab, zorder=3)
    ax.fill_between(eps_grid, mean[:, k]-std[:, k], mean[:, k]+std[:, k], color=c, alpha=0.15)
ax.axhline(0.15, color="crimson", ls="--", lw=1.2, label="tolerance 0.15")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"perturbation size  $\varepsilon$   ($\|\delta V\|=1$)")
ax.set_ylabel(r"lin. error  $\|\Delta\psi_{\rm true}-\varepsilon J\delta V\|/\|\Delta\psi_{\rm true}\|$")
ax.set_title("Linearized validity radius: exact vs FNO response operators")
ax.legend(loc="center left", fontsize=8, framealpha=0.9)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout(); fig.savefig("results/fig_validity_radius.png", dpi=200)
print("\nsaved results/fig_validity_radius.png")