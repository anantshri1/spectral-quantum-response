# scripts/sweep_clip_vs_retrain_f64.py — f64 clip-vs-retrain, 3-seed.
# One J_true per sample, reused across BOTH arms and all M/seeds (J_true is the
# cost driver). New f64 CSV; Phase 5 f32 artifacts untouched.
import jax
jax.config.update("jax_enable_x64", True)

import os, csv, time
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.responses import jacobian_true, jacobian_fno, response_error
from tests.measure_floor import load_fno_f64_ckpt   # f64-origin loader + n_modes override

SEEDS = [0, 1, 2]
MS    = [2, 4, 8, 12, 16]

def retrain_ckpt(seed, M):
    return f"checkpoints/dino_N2000_seed{seed}_M{M}_lam0.0_final.eqx"

def clip_base_ckpt(seed):   # the f64 forward-only M=16 model, clipped at inference
    return f"checkpoints/dino_N2000_seed{seed}_M16_lam0.0_final.eqx"

def truncate_model(model, M):
    """Zero spectral weights for modes M..15 in every block. By attribute PATH
    (value-independent) -> tree_at-safe, dtype-agnostic (works in f64)."""
    def where(m):
        leaves = []
        for b in m.blocks:
            leaves.append(b.spectral_conv.w_real)
            leaves.append(b.spectral_conv.w_imag)
        return leaves
    old = where(model)
    new = [w.at[M:].set(0.0) for w in old]   # M>=16 -> empty slice -> identity
    return eqx.tree_at(where, model, new)

if __name__ == "__main__":
    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt

    d = np.load("data/test.npz")
    u0_re, u0_im, V_all, k0_all = (d["u0_re_128"], d["u0_im_128"],
                                   d["V_128"], d["p_k0"])

    # --- preload all models once (retrain: 15, clip bases: 3) ---
    # retrain arm: genuine f64-trained at each M. M=16 override needed on loader
    # for M!=16 (shape) — load_fno_f64_ckpt takes n_modes.
    retrain = {(s, M): load_fno_f64_ckpt(retrain_ckpt(s, M), cfg, n_modes=M)
               for s in SEEDS for M in MS}
    clip_base = {s: load_fno_f64_ckpt(clip_base_ckpt(s), cfg, n_modes=16)
                 for s in SEEDS}
    clip = {(s, M): truncate_model(clip_base[s], M)
            for s in SEEDS for M in MS}   # M=16 clip == base (identity)

    # --- GATE: clip(M=16) must be bit-identical to the base model ---
    # dtype-independent identity check; replaces Phase 5's stale 0.3862 magic num.
    psi0_0 = jnp.asarray(u0_re[0] + 1j*u0_im[0], dtype=jnp.complex128)
    V_0    = jnp.asarray(V_all[0], dtype=jnp.float64)
    Jt0    = jacobian_true(psi0_0, V_0, nx, dt, nt)
    e_base = float(response_error(jacobian_fno(clip_base[0], psi0_0, V_0), Jt0))
    e_c16  = float(response_error(jacobian_fno(clip[(0,16)], psi0_0, V_0), Jt0))
    e_c4   = float(response_error(jacobian_fno(clip[(0,4)],  psi0_0, V_0), Jt0))
    print(f"clip base (seed0)     : {e_base:.4f}")
    print(f"clip(M=16) identity   : {e_c16:.4f}  [{'PASS' if abs(e_c16-e_base)<1e-12 else 'FAIL'}]")
    print(f"clip(M=4) (modes gone): {e_c4:.4f}   (expect != base)")
    # also gate retrain M=16 seed0 reproduces the known f64 floor (~0.579)
    e_r16 = float(response_error(jacobian_fno(retrain[(0,16)], psi0_0, V_0), Jt0))
    print(f"retrain(M=16) seed0   : {e_r16:.4f}  (sample-0; full-set floor ~0.579)")
    assert abs(e_c16 - e_base) < 1e-12, "clip(M=16) != base — identity broken, STOP"
    print("gate PASS — proceeding to full sweep\n")

     # --- full sweep: one J_true per sample, reused across both arms ---
    os.makedirs("results", exist_ok=True)
    n = u0_re.shape[0]
    fcsv = open("results/response_clip_vs_retrain_f64.csv", "w", newline="")
    wtr = csv.writer(fcsv)
    wtr.writerow(["idx", "k0", "seed", "M", "arm", "response_err"])
    fcsv.flush()

    t0 = time.perf_counter()
    for i in range(n):
        psi0 = jnp.asarray(u0_re[i] + 1j*u0_im[i], dtype=jnp.complex128)
        V    = jnp.asarray(V_all[i], dtype=jnp.float64)
        Jt   = jacobian_true(psi0, V, nx, dt, nt)   # EXPENSIVE — once per sample
        k0   = int(k0_all[i])

        for s in SEEDS:
            for M in MS:
                e_r = float(response_error(
                    jacobian_fno(retrain[(s, M)], psi0, V), Jt))
                wtr.writerow([i, k0, s, M, "retrain", e_r])
                e_c = float(response_error(
                    jacobian_fno(clip[(s, M)], psi0, V), Jt))
                wtr.writerow([i, k0, s, M, "clip", e_c])
        fcsv.flush()

        if i % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  sample {i:3d}/{n} | {el:.1f}s | {el/(i+1):.2f}s/sample")

    fcsv.close()
    print(f"\ndone. {n} samples x {len(SEEDS)} seeds x {len(MS)} M x 2 arms "
          f"in {time.perf_counter()-t0:.1f}s")
    print("saved results/response_clip_vs_retrain_f64.csv")