import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.fno import FNO
from src.responses import jacobian_true, fd_check, mode_coupling_matrix
from src.solver import evolve

# --- temporary probe: does peak offset track carrier momentum k0? ---
# load_split doesn't return params; read them straight from the npz.
d = np.load("data/test.npz")
k0_all = d["p_k0"]                        # (N,) carrier momenta
print("sample 0 k0:", int(k0_all[0]))    # <-- is it 4?

# compute peak offset for a few samples, compare to their k0
def peak_offset(psi0_s, V_s):
    Jp = jacobian_true(psi0_s, V_s, nx, dt, nt)
    Jk = mode_coupling_matrix(Jp)
    aJ = jnp.abs(Jk)
    kk, qq = jnp.meshgrid(k_idx, k_idx, indexing="ij")
    off = ((kk - qq + nx//2) % nx) - nx//2
    dv = jnp.unique(off.ravel())
    prof = jnp.array([jnp.mean(aJ.ravel()[off.ravel() == dd]) for dd in dv])
    return int(dv[jnp.argmax(prof)])

for i in range(4):
    pk = peak_offset(psi0[i], V[i])
    print(f"sample {i}: k0={int(k0_all[i])}, peak k-q={pk}")