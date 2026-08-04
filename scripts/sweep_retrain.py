import jax
jax.config.update("jax_enable_x64", True)

import os, csv, time
import numpy as np
import jax.numpy as jnp

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.compute_response import load_fno_x64

Ms = [2, 4, 8, 12, 16]  # 16 = reuse Phase-3/4 checkpoint, per Block-4 agreement

def ckpt_path(M):
    if M == 16:
        return "checkpoints/fno_N2000_seed0_final.eqx"   # canonical, untouched
    return f"checkpoints/fno_N2000_seed0_M{M}_final.eqx"

if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = d["u0_re_128"], d["u0_im_128"], d["V_128"], d["p_k0"]
    n = u0_re.shape[0]

    f = open("results/retrain_sweep.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["idx", "k0", "M", "response_err"]); f.flush()

    t0 = time.perf_counter()
    for M in Ms:
        cfg_m = Config(); cfg_m.n_modes = M    # model built with THIS n_modes (arch must match ckpt)
        model = load_fno_x64(ckpt_path(M), cfg_m)
        print(f"M={M}: loaded {ckpt_path(M)}")

        for i in range(n):
            psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
            V    = jnp.asarray(V_all[i], dtype=jnp.float64)
            J_true = jacobian_true(psi0, V, nx, dt, nt)
            err = float(response_error(jacobian_fno(model, psi0, V), J_true))
            w.writerow([i, int(k0_all[i]), M, err]); f.flush()
        print(f"  M={M} done at {time.perf_counter()-t0:.1f}s")

    f.close()
    print(f"done. results/retrain_sweep.csv")