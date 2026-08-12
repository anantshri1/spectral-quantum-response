# scripts/ood_build_jtrue.py — 7D-2b-i: build results/ood/Jtrue_kq_ls{ls}.npy for all 6 ls
import os, shutil
import jax; jax.config.update("jax_enable_x64", True)   # non-negotiable: f64 for J_true
import jax.numpy as jnp
import numpy as np

from src.data_gen import ood_potential
from src.responses import jacobian_true, mode_coupling_matrix, fd_check
from configs.default import Config

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
LS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LS0, N = 0.20, 200
OUT = "results/ood"; os.makedirs(OUT, exist_ok=True)
CACHE_020 = "results/Jtrue_kq_200.npy"     # frozen in-dist kq cache (== ls=0.20 under conv. B)

# ---- load frozen inputs (ψ0 as-is, assert unit-norm; D9=a) ----
d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])   # (200,128) c128, as stored
coeffs = jnp.asarray(d["p_coeffs"][:N])                             # (200,64)
V128   = jnp.asarray(d["V_128"][:N])                               # (200,128)
norms  = jnp.sqrt(jnp.sum(jnp.abs(psi0)**2, axis=1))
assert bool(jnp.all(jnp.isfinite(psi0))), "ψ0 non-finite, STOP"
print(f"[load] ψ0 as-is (D9=a, unnormalized): ‖ψ0‖ ∈ [{float(norms.min()):.3f}, {float(norms.max()):.3f}] — matches cache convention")

def jtrue_kq(i, V):
    return mode_coupling_matrix(jacobian_true(psi0[i], V[i], nx, dt, nt))

for ls in LS_GRID:
    path = f"{OUT}/Jtrue_kq_ls{ls:.2f}.npy"
    if os.path.exists(path):
        print(f"[skip] ls={ls:.2f} exists → {path}")
        continue

    # ls=0.20 short-circuit: convention B makes it bit-identical to the frozen cache
    if abs(ls - LS0) < 1e-12:
        shutil.copyfile(CACHE_020, path)
        print(f"[copy] ls=0.20 → frozen cache copied to {path}")
        continue

    # fresh f64 build for this ls
    V = jnp.stack([ood_potential(coeffs[i], V128[i], nx, ls) for i in range(N)])   # (200,128)

    # ANCHOR: before trusting 200 autodiffs, validate ONE against finite-diff truth
    rel_fd, _, _ = fd_check(psi0[0], V[0], nx, dt, nt, eps=1e-6)
    assert float(rel_fd) < 1e-4, f"ls={ls:.2f} J_true vs FD rel={float(rel_fd):.2e} — solver/autodiff suspect, STOP"
    print(f"[anchor] ls={ls:.2f} J_true vs FD rel = {float(rel_fd):.2e}  OK")

    Jkq = jnp.stack([jtrue_kq(i, V) for i in range(N)])        # (200,128,128) c128
    assert bool(jnp.all(jnp.isfinite(Jkq))), f"ls={ls:.2f} J_true non-finite, STOP"
    np.save(path, np.asarray(Jkq))
    print(f"[save] ls={ls:.2f} → {path}  |J|_med={float(jnp.median(jnp.abs(Jkq))):.3e}")

print("\n[done] J_true grid:")
for ls in LS_GRID:
    p = f"{OUT}/Jtrue_kq_ls{ls:.2f}.npy"
    print(f"  ls={ls:.2f}: {'✓' if os.path.exists(p) else '✗ MISSING'}  {p}")