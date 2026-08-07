# scripts/precompute_dino.py — Phase 6 Block 1: offline (V, v, J_true·v) precompute
#
# x64 MUST be set before any jnp work — same discipline as compute_response.py.
# jacobian_true runs float32 silently otherwise and the 1e-10 gate fails for a
# dtype reason, not a math reason.
import jax
jax.config.update("jax_enable_x64", True)

import os, time
import numpy as np
import jax.numpy as jnp

from configs.default import Config
from src.solver import evolve
from scripts.train import load_split

K = 8   # directions per sample (locked)

def make_directions(key, n_samples, k, nx):
    """
    Isotropic Gaussian unit directions in raw-V space (R^nx).
    Shape (n_samples, k, nx), each row L2-normalized.
    Isotropic (not smooth-δV) because it's the unbiased estimator of the
    FROBENIUS response error — the exact metric we report (0.511). Sampling
    smooth directions would optimize an easier quantity than we measure.
    """
    raw = jax.random.normal(key, (n_samples, k, nx), dtype=jnp.float64)
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / norms

def main():
    cfg = Config()
    os.makedirs("results", exist_ok=True)

    # --- load training slice IDENTICALLY to compute_response.py ---
    # load_split pins complex64/float32 (Phase-3 regime); widen to the float64
    # solver's carries (bug #2: widen in driver, not in loader/solver).
    psi0, V, _uT = load_split("data/train.npz", nx=cfg.nx)
    psi0 = psi0.astype(jnp.complex128)
    V    = V.astype(jnp.float64)

    # k0 carried through for later per-|k0| breakdowns (rows positional, aligned).
    k0 = np.load("data/train.npz")["p_k0"]          # (2000,) int64

    n = psi0.shape[0]
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt                          # canonical: t_end/nt

    # --- 8 isotropic unit directions per sample, fixed seed 0 ---
    v = make_directions(jax.random.PRNGKey(0), n, K, nx)   # (n, K, nx) f64

    # --- per-sample JVP of the SAME map jacobian_true differentiates ---
    # f(V_) = evolve(psi0_i, V_, nx, dt, nt); jvp gives (psi_T, J·v) in one
    # forward pass. vmap over the K directions (one linearization reused).
    def jv_for_sample(psi0_i, V_i, v_i):
        def f(V_):
            return evolve(psi0_i, V_, nx, dt, nt)
        def one_dir(vk):
            _, jvp = jax.jvp(f, (V_i,), (vk,))      # (nx,) complex128
            return jvp
        return jax.vmap(one_dir)(v_i)               # (K, nx) complex128

    jv_for_sample = jax.jit(jv_for_sample)

    # --- loop over samples: progress + intermediate saves + wall clock ---
    Jv = np.zeros((n, K, nx), dtype=np.complex128)
    t0 = time.time()
    for i in range(n):
        Jv[i] = np.asarray(jv_for_sample(psi0[i], V[i], v[i]))
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  sample {i+1:4d}/{n} | {el:6.1f}s | {el/(i+1):.3f}s/sample")
        if (i + 1) % 500 == 0:
            np.savez("results/dino_precompute_partial.npz",
                     V=np.asarray(V), v=np.asarray(v), Jv_true=Jv,
                     k0=k0, done=i + 1)
            print(f"    [partial saved through sample {i+1}]")
    wall = time.time() - t0
    print(f"\ndone. {n} samples in {wall:.1f}s ({wall/n:.3f}s/sample)")

    # --- C_norm: global mean ||J_true·v||^2 over all n*K pairs ---
    # single global constant (not per-direction): per-dir relative normalization
    # blows up on isotropic high-k directions where ||J·v|| is tiny.
    Jv_j = jnp.asarray(Jv)
    C_norm = float(jnp.mean(jnp.sum(jnp.abs(Jv_j) ** 2, axis=-1)))   # scalar
    print(f"C_norm (mean ||J_true·v||^2) = {C_norm:.6e}")

    # --- GATE: precomputed Jv == jacobian_true(...) @ v to ~1e-10 ---
    from src.responses import jacobian_true
    print("\n[gate] precomputed J·v vs jacobian_true @ v:")
    for i in [0, 137, n - 1]:
        J = jacobian_true(psi0[i], V[i], nx, dt, nt)      # (nx,nx) complex128
        for kk in [0, K - 1]:
            ref = J @ v[i, kk]
            rel = float(jnp.linalg.norm(Jv_j[i, kk] - ref)
                        / jnp.linalg.norm(ref))
            flag = "OK" if rel < 1e-10 else "FAIL"
            print(f"  sample {i:4d} dir {kk}: rel-err {rel:.2e}  [{flag}]")
            assert rel < 1e-10, "precomputed J·v disagrees with jacobian_true — STOP"

    # --- save final ---
    np.savez("results/dino_precompute.npz",
             V=np.asarray(V), v=np.asarray(v), Jv_true=Jv,
             k0=k0, C_norm=C_norm, K=K, wall_seconds=wall,
             dir_seed=0)
    print("\nsaved results/dino_precompute.npz")
    print(f"  V        {V.shape} {V.dtype}")
    print(f"  v        {v.shape} {v.dtype}")
    print(f"  Jv_true  {Jv.shape} {Jv.dtype}")
    print(f"  C_norm   {C_norm:.6e} | K={K} | wall={wall:.1f}s")

if __name__ == "__main__":
    main()