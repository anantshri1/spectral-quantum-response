import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.fno import FNO
from src.responses import jacobian_true, jacobian_fno, response_error

if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    # random init, SAME architecture as the trained M=16 model, no training at all
    key = jax.random.PRNGKey(cfg.seed)
    model_f32 = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                    n_blocks=cfg.n_fno_blocks, key=key)
    # widen to float64 for consistency with the x64 response machinery
    # (mirrors load_fno_x64's cast-up step, just skipping the deserialise)
    model = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_array(x) and jnp.iscomplexobj(x) is False and x.dtype == jnp.float32 else x,
        model_f32
    )

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = d["u0_re_128"], d["u0_im_128"], d["V_128"], d["p_k0"]
    n = u0_re.shape[0]

    import csv
    f = open("results/untrained_sweep.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["idx", "k0", "response_err"]); f.flush()

    for i in range(n):
        psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
        V    = jnp.asarray(V_all[i], dtype=jnp.float64)
        J_true = jacobian_true(psi0, V, nx, dt, nt)
        err = float(response_error(jacobian_fno(model, psi0, V), J_true))
        w.writerow([i, int(k0_all[i]), err]); f.flush()

    f.close()
    mean = np.mean([float(r) for r in np.genfromtxt("results/untrained_sweep.csv",
                    delimiter=",", names=True)["response_err"]])
    print(f"done. untrained response error, mean over {n} samples: {mean:.4f}")