# scripts/measure_floor.py — Phase 6: measure 200-sample mean response error
# for a given checkpoint. Reuses compute_response.py's verified machinery.
# Runs the SAME checkpoint-agnostic loop on both f32-canonical and f64-lam0
# so any floor shift is attributable to the model, not the driver.
import jax
jax.config.update("jax_enable_x64", True)   # response JVP needs f64 regardless

import sys, time
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from scripts.train import load_split
from scripts.compute_response import load_fno_x64

from src.fno import FNO

def load_fno_f64_ckpt(path, cfg, n_modes=None):
    """
    Load a checkpoint that was SAVED in f64 (e.g. from train_dino under x64-on).
    Under x64-on, FNO(...) builds f64 leaves — so the template already matches
    the file. No pin-to-f32 step needed; deserialize directly.
    Contrast with load_fno_x64 which handles f32-on-disk checkpoints.
    """
    m = n_modes if n_modes is not None else cfg.n_modes
    template = FNO(d_v=cfg.n_channels, n_modes=m,
                   n_blocks=cfg.n_fno_blocks, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, template)


def measure(ckpt_path, cfg, n_samples=200, f64_ckpt=False, n_modes=None):
    """
    Mean rel-Frobenius response error over the first n_samples TEST samples.
    load_fno_x64 loads ANY checkpoint (f32 bytes) to an f64 model — so the
    RESPONSE is always computed in f64; only the trained WEIGHTS differ between
    checkpoints. That's the correct comparison: same response arithmetic,
    different trained model.
    """
    if f64_ckpt:
        model = load_fno_f64_ckpt(ckpt_path, cfg, n_modes=n_modes)
    else:
        model = load_fno_x64(ckpt_path, cfg) 

    psi0, V, _uT = load_split("data/test.npz", nx=cfg.nx)
    psi0 = psi0.astype(jnp.complex128)
    V    = V.astype(jnp.float64)

    dt, nx, nt = cfg.t_end / cfg.nt, cfg.nx, cfg.nt

    errs = []
    t0 = time.time()
    for i in range(n_samples):
        J_true = jacobian_true(psi0[i], V[i], nx, dt, nt)
        J_fno  = jacobian_fno(model, psi0[i], V[i])
        errs.append(float(response_error(J_fno, J_true)))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    {i+1:3d}/{n_samples} | mean-so-far "
                  f"{np.mean(errs):.4f} | {el:.1f}s")
    errs = np.array(errs)
    return errs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--f64", action="store_true", dest="f64_ckpt")
    p.add_argument("--n_modes", type=int, default=None,
                   help="Override n_modes for template build. Required when "
                        "checkpoint n_modes differs from cfg default (16).")
    args = p.parse_args()
    cfg = Config()
    errs = measure(args.ckpt, cfg, f64_ckpt=args.f64_ckpt, n_modes=args.n_modes)
    print(f"\n{args.ckpt}")
    print(f"  mean response err : {errs.mean():.4f}")
    print(f"  sem               : {errs.std()/np.sqrt(len(errs)):.4f}")
    print(f"  min / max         : {errs.min():.4f} / {errs.max():.4f}")