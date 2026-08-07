# tests/measure_untrained_floor.py — the 1.000 denominator, in f64.
# Untrained random FNO has no response structure; expected ~1.0 regardless of
# precision (nothing for precision to perturb). Confirms the denominator.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from configs.default import Config
from src.fno import FNO
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.train import load_split

if __name__ == "__main__":
    cfg = Config()
    # fresh untrained model, f64 under x64-on. Same seed convention as training
    # init (PRNGKey(seed) -> split -> model key), so this is the actual t=0 model.
    key = jax.random.PRNGKey(cfg.seed)
    _, mk = jax.random.split(key)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                n_blocks=cfg.n_fno_blocks, key=mk)

    psi0, V, _ = load_split("data/test.npz", nx=cfg.nx)
    psi0, V = psi0.astype(jnp.complex128), V.astype(jnp.float64)
    dt, nx, nt = cfg.t_end / cfg.nt, cfg.nx, cfg.nt

    errs = []
    for i in range(200):
        J_true = jacobian_true(psi0[i], V[i], nx, dt, nt)
        J_fno  = jacobian_fno(model, psi0[i], V[i])
        errs.append(float(response_error(J_fno, J_true)))
    errs = np.array(errs)
    print(f"untrained f64 floor: {errs.mean():.4f} +/- {errs.std()/np.sqrt(200):.4f}")
    print(f"  min/max: {errs.min():.4f} / {errs.max():.4f}")