# scripts/ood_jtrue_diag.py — diagnose the 1.7e-8 J_true vs cache gap
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.data_gen import realize_potential, realize_wavepacket
from src.responses import jacobian_true, mode_coupling_matrix
from configs.default import Config

cfg = Config()
nx, nt, dt = cfg.nx, cfg.nt, cfg.t_end / cfg.nt
LS0, N = 0.20, 4
d = np.load("data/test.npz")

# --- H1: what precision is the frozen cache actually stored at? ---
cache_raw = np.load("results/Jtrue_kq_200.npy")
psi0_stored = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
print(f"[H1] cache dtype = {cache_raw.dtype} | our psi0 dtype = {psi0_stored.dtype}")

# --- H3: is the STORED psi0 f32-tainted vs a fresh x64 realization from the stored params? ---
x0, sig, k0p = d["p_x0"][:N], d["p_sigma"][:N], d["p_k0"][:N]
psi0_fresh = jnp.stack([realize_wavepacket(float(x0[i]), float(sig[i]), int(k0p[i]), nx) for i in range(N)])
def nrm(a): return a / jnp.sqrt(jnp.sum(jnp.abs(a)**2, axis=-1, keepdims=True))
print(f"[H3] fresh-vs-stored psi0: raw max|Δ|={float(jnp.max(jnp.abs(psi0_fresh-psi0_stored))):.2e} | "
      f"normalized max|Δ|={float(jnp.max(jnp.abs(nrm(psi0_fresh)-nrm(psi0_stored)))):.2e}")

# --- H2: is the J error UNIFORM (precision) or STRUCTURED (aliasing/convention = real bug)? ---
coeffs0 = jnp.asarray(d["p_coeffs"][:N]); V128 = jnp.asarray(d["V_128"][:N])
kk = jnp.arange(coeffs0.shape[1])
def ood_V(c, Vind, ls):
    cc = c*jnp.exp(-kk*(ls-LS0)); V = realize_potential(cc, nx)
    return V*(jnp.linalg.norm(Vind)/jnp.linalg.norm(V))
V020 = jnp.stack([ood_V(coeffs0[i], V128[i], LS0) for i in range(N)])
Jkq = jnp.stack([mode_coupling_matrix(jacobian_true(psi0_stored[i], V020[i], nx, dt, nt)) for i in range(N)])
Jcache = jnp.asarray(cache_raw[:N])
absdiff = jnp.abs(Jkq - Jcache)
big = jnp.abs(Jcache) > 1e-3
print(f"[H2] abs err max={float(absdiff.max()):.2e} median={float(jnp.median(absdiff)):.2e} | "
      f"rel(|J|>1e-3) median={float(jnp.median((absdiff/(jnp.abs(Jcache)+1e-30))[big])):.2e}")
idx = jnp.unravel_index(int(jnp.argmax(absdiff[0])), absdiff[0].shape)
print(f"[H2] sample0 worst entry at (k_bin,q_bin)={tuple(int(x) for x in idx)}  (high-|k| corner ⇒ aliasing; scattered ⇒ precision)")

# --- our own float64 self-consistency: fd_check gives an independent truth anchor ---
from src.responses import fd_check
rel_fd, _, _ = fd_check(psi0_stored[0], V020[0], nx, dt, nt, eps=1e-6)
print(f"[anchor] our J_true vs finite-diff rel = {float(rel_fd):.2e}  (if ~1e-6, OUR J is correct → cache is the stale one)")