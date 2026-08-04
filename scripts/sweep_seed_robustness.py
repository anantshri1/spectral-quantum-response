# scripts/sweep_seed_robustness.py
import jax
jax.config.update("jax_enable_x64", True)

import csv, time
import numpy as np
import jax.numpy as jnp

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.compute_response import load_fno_x64

# (M, seed) pairs to check. seed=0 reuses the canonical/already-swept checkpoints.
runs = [
    (12, 0), (12, 1), (12, 2), (12, 3),
    (16, 0), (16, 1), (16, 2), (16, 3),
]

def ckpt_path(M, seed):
    if seed == 0:
        return "checkpoints/fno_N2000_seed0_final.eqx" if M == 16 \
               else f"checkpoints/fno_N2000_seed0_M{M}_final.eqx"
    return f"checkpoints/fno_N2000_seed{seed}_M{M}_final.eqx"

if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = d["u0_re_128"], d["u0_im_128"], d["V_128"], d["p_k0"]
    n = u0_re.shape[0]

    f = open("results/seed_robustness_sweep.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["M", "seed", "idx", "k0", "response_err"]); f.flush()

    t0 = time.perf_counter()
    for M, seed in runs:
        cfg_m = Config(); cfg_m.n_modes = M
        model = load_fno_x64(ckpt_path(M, seed), cfg_m)
        for i in range(n):
            psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
            V    = jnp.asarray(V_all[i], dtype=jnp.float64)
            J_true = jacobian_true(psi0, V, nx, dt, nt)
            err = float(response_error(jacobian_fno(model, psi0, V), J_true))
            w.writerow([M, seed, i, int(k0_all[i]), err]); f.flush()
        print(f"M={M} seed={seed} done at {time.perf_counter()-t0:.1f}s")

    f.close()
    print("done. results/seed_robustness_sweep.csv")