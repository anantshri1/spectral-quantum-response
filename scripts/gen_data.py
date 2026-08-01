# scripts/gen_data.py
"""
Generate the TDSE dataset: sample (ψ₀, V) → solve at nx=1024 → subsample the
resolution ladder → normalize per-grid → save split-complex, mirroring Burgers'
file structure (one .npz per split, resolution-keyed).

Protocol (the super-res-honest one):
  solve ONLY at the finest grid, subsample the SOLUTION down. Never solve coarse
  and interpolate up. Gate: ψ_T at 1024 must have negligible energy above the
  128-grid Nyquist (k=64), else subsampling aliases.

float64 throughout — this is ground-truth numerics, run once, offline.
"""
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import argparse

from configs.default import Config
from src.data_gen import (
    sample_wavepacket_params, realize_wavepacket,
    sample_potential_spectrum, realize_potential,
)
from src.solver import evolve


def normalize_psi(psi, nx):
    """∫|ψ|²dφ = 1 on this grid. Quadrature weight dφ=2π/nx is grid-dependent,
    so normalization is applied per-resolution AFTER subsampling."""
    dphi = 2 * jnp.pi / nx
    return psi / jnp.sqrt(jnp.sum(jnp.abs(psi)**2) * dphi)


def high_k_fraction(psi, k_nyq_coarse):
    """Fraction of ψ's energy above the coarsest grid's Nyquist (k=64 for nx=128).
    In fft layout, |k|>k_nyq is the central band [k_nyq+1 : nx-k_nyq]."""
    spec = jnp.abs(jnp.fft.fft(psi))**2
    nx = psi.shape[0]
    hi = jnp.sum(spec[k_nyq_coarse+1 : nx-k_nyq_coarse])
    return float(hi / jnp.sum(spec))


def generate(cfg, n_samples, seed, out_path):
    key = jax.random.PRNGKey(seed)
    nx_fine = cfg.nx_super                              # 1024
    T = cfg.t_end
    dt = T / cfg.nt_super
    ladder = [cfg.nx, cfg.nx_mid1, cfg.nx_mid2, cfg.nx_super]   # 128,256,512,1024
    strides = {n: nx_fine // n for n in ladder}         # 8,4,2,1

    # accumulators: split-complex real arrays per resolution
    store = {f"{fld}_{n}": [] for n in ladder
             for fld in ("u0_re","u0_im","uT_re","uT_im","V")}
    params = {"x0": [], "sigma": [], "k0": [], "coeffs": []}

    worst_hi_k = 0.0
    for i in range(n_samples):
        key, kp, kv = jax.random.split(key, 3)

        # --- sample grid-free parameters ---
        x0, sigma, k0 = sample_wavepacket_params(kp, cfg)
        coeffs = sample_potential_spectrum(kv, None, cfg.v_length_scale,
                                           cfg.v_amplitude, cfg.v_n_coeffs)

        # --- realize at the finest grid ONLY, then solve ---
        psi0_fine = realize_wavepacket(x0, sigma, k0, nx_fine)
        psi0_fine = normalize_psi(psi0_fine, nx_fine)
        V_fine    = realize_potential(coeffs, nx_fine)
        psiT_fine = evolve(psi0_fine, V_fine, nx_fine, dt, cfg.nt_super)

        # --- GATE: ψ_T must be band-limited below the coarse Nyquist (k=64) ---
        hi = high_k_fraction(psiT_fine, cfg.nx // 2)
        worst_hi_k = max(worst_hi_k, hi)

        # --- subsample the SOLUTION down the ladder, normalize per grid ---
        for n in ladder:
            s = strides[n]
            u0 = normalize_psi(psi0_fine[::s], n)   # renormalize on the coarse grid
            uT = normalize_psi(psiT_fine[::s], n)                    # renormalize on the coarse grid, like u0
            V  = V_fine[::s]
            store[f"u0_re_{n}"].append(np.asarray(jnp.real(u0)))
            store[f"u0_im_{n}"].append(np.asarray(jnp.imag(u0)))
            store[f"uT_re_{n}"].append(np.asarray(jnp.real(uT)))
            store[f"uT_im_{n}"].append(np.asarray(jnp.imag(uT)))
            store[f"V_{n}"].append(np.asarray(V))

        params["x0"].append(float(x0))
        params["sigma"].append(float(sigma))
        params["k0"].append(int(k0))
        params["coeffs"].append(np.asarray(coeffs))

        if i % 200 == 0:
            print(f"  [{i:4d}/{n_samples}] worst high-k(>64) so far: {worst_hi_k:.2e}")

    # --- final gate report ---
    print(f"\nworst high-k(>64) fraction across dataset: {worst_hi_k:.2e}")
    if worst_hi_k > 1e-4:
        print("  ⚠ WARNING: ψ_T has non-trivial energy above the 128-grid Nyquist.")
        print("    Subsampling to 128 may alias. Reconsider T / length_scale / base nx.")
    else:
        print("  ✓ ψ_T band-limited below k=64 — subsampling to 128 is honest.")

    # --- save: stack lists into arrays, mirror Burgers .npz structure ---
    out = {k: np.stack(v) for k, v in store.items()}
    out["p_x0"]    = np.array(params["x0"])
    out["p_sigma"] = np.array(params["sigma"])
    out["p_k0"]    = np.array(params["k0"])
    out["p_coeffs"] = np.stack(params["coeffs"])
    np.savez(out_path, **out)
    print(f"saved {n_samples} samples → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    cfg = Config()

    if args.split == "train":
        generate(cfg, cfg.n_train_max, seed=0,   out_path="data/train.npz")
    else:
        generate(cfg, cfg.n_test,      seed=999, out_path="data/test.npz")