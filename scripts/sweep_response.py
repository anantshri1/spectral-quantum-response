# scripts/sweep_response.py
import jax
jax.config.update("jax_enable_x64", True)   # MUST precede array work — Phase-4 precision discipline

import os, time, csv, argparse
import numpy as np
import jax.numpy as jnp

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.compute_response import load_fno_x64   # reuse the proven x64 loader

def run_sweep(limit=None):
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt                  # 0.5/1000 = 5e-4  ← CONFIRM this matches Phase 4

    d = np.load(os.path.join(cfg.data_dir, "test.npz"))
    u0_re, u0_im = d["u0_re_128"], d["u0_im_128"]
    V_all, k0_all = d["V_128"], d["p_k0"]
    n = u0_re.shape[0] if limit is None else min(limit, u0_re.shape[0])

    model = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg)

    os.makedirs("results", exist_ok=True)
    out_path = "results/response_sweep.csv"
    f = open(out_path, "w", newline="")     # overwrite; flush-per-row = Ctrl-C safe (readable partial)
    w = csv.writer(f)
    w.writerow(["idx", "k0", "response_err", "num", "den", "seconds"])
    f.flush()

    t_all = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()

        # widen AFTER load (Block-0 lesson: scan carries must be dtype-invariant)
        psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
        V    = jnp.asarray(V_all[i], dtype=jnp.float64)

        J_true = jacobian_true(psi0, V, nx, dt, nt)   # (nx,nx) complex128
        J_fno  = jacobian_fno(model, psi0, V)         # (nx,nx) complex128
        err = float(response_error(J_fno, J_true))    # rel-Frobenius, basis-independent by unitarity
        num = float(jnp.linalg.norm(J_fno - J_true))   # absolute error
        den = float(jnp.linalg.norm(J_true))            # denominator

        secs = time.perf_counter() - t0
        w.writerow([i, int(k0_all[i]), err, num, den, f"{secs:.3f}"]); f.flush()
        print(f"  sample {i:3d} | k0={int(k0_all[i]):+d} | resp={err:.4f} | {secs:.2f}s")

    f.close()
    print(f"\ndone. {n} samples in {time.perf_counter()-t_all:.1f}s -> {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    run_sweep(limit=p.parse_args().limit)