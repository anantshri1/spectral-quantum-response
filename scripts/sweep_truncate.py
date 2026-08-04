# scripts/sweep_truncate.py
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.compute_response import load_fno_x64


def truncate_model(model, M):
    """Zero spectral weights for modes M..n_modes-1 in every FNO block.
    Selection is by attribute path (value-independent) -> safe for tree_at,
    avoids the Phase-4 where-lambda-on-values gotcha."""
    def where(m):
        leaves = []
        for b in m.blocks:
            leaves.append(b.spectral_conv.w_real)
            leaves.append(b.spectral_conv.w_imag)
        return leaves
    old = where(model)
    new = [w.at[M:].set(0.0) for w in old]   # M>=n_modes -> empty slice -> identity
    return eqx.tree_at(where, model, new)

def run_sweep(Ms=(2, 4, 8, 12, 16), limit=None):
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = (d["u0_re_128"], d["u0_im_128"],
                                   d["V_128"], d["p_k0"])
    n = u0_re.shape[0] if limit is None else min(limit, u0_re.shape[0])

    base = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg)
    models = {M: truncate_model(base, M) for M in Ms}   # build once, reuse

    import csv, time
    f = open("results/truncate_sweep.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["idx", "k0", "M", "response_err"]); f.flush()

    t0 = time.perf_counter()
    for i in range(n):
        psi0 = jnp.asarray(u0_re[i] + 1j * u0_im[i], dtype=jnp.complex128)
        V    = jnp.asarray(V_all[i], dtype=jnp.float64)
        J_true = jacobian_true(psi0, V, nx, dt, nt)      # compute ONCE per sample
        for M in Ms:
            err = float(response_error(jacobian_fno(models[M], psi0, V), J_true))
            w.writerow([i, int(k0_all[i]), M, err]); f.flush()
        if i % 20 == 0:
            print(f"  sample {i:3d} | {time.perf_counter()-t0:.1f}s")
    f.close()
    print(f"done. {n} samples x {len(Ms)} modes in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    psi0 = jnp.asarray(d["u0_re_128"][0] + 1j * d["u0_im_128"][0], dtype=jnp.complex128)
    V    = jnp.asarray(d["V_128"][0], dtype=jnp.float64)

    model = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg)

    J_true = jacobian_true(psi0, V, nx, dt, nt)

    # --- GATE: M=16 must be the identity (reproduce 0.3862) ---
    err_full = float(response_error(jacobian_fno(model, psi0, V), J_true))
    m16      = truncate_model(model, 16)
    err_16   = float(response_error(jacobian_fno(m16, psi0, V), J_true))
    m4       = truncate_model(model, 4)
    err_4    = float(response_error(jacobian_fno(m4, psi0, V), J_true))

    print(f"full model      : {err_full:.4f}   (expect 0.3862)")
    print(f"truncate(M=16)  : {err_16:.4f}   (expect == full, identity)")
    print(f"truncate(M=4)   : {err_4:.4f}   (expect != full — modes removed)")
    print(f"identity check  : {'PASS' if abs(err_16 - err_full) < 1e-12 else 'FAIL'}")

    run_sweep()