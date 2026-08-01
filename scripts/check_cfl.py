# scripts/check_cfl.py
"""
Gate: is nt_super large enough that ψ_T is converged at nx=1024?

Solves a handful of worst-case examples at nt_super and 2×nt_super and checks
they agree to ~float64. If they do, nt_super is sufficient — more steps wouldn't
change the answer. This turns "nt_super=10000 is a big number" into "ψ_T is
converged," which is the honest claim. Run ONCE before full generation.
"""
import jax; jax.config.update("jax_enable_x64", True)   # float64: solver precision
import jax.numpy as jnp

from configs.default import Config          # adjust to your actual config path
from src.data_gen import (
    sample_wavepacket_params, realize_wavepacket,
    sample_potential_spectrum, realize_potential,
)
from src.solver import evolve

cfg = Config()
nx  = cfg.nx_super          # 1024 — the finest grid, where CFL is tightest
T   = cfg.t_end
key = jax.random.PRNGKey(12345)

print(f"CFL convergence check at nx={nx}, T={T}")
print(f"{'ex':>3} {'|Δψ_T| (nt vs 2nt)':>20} {'norm(nt)':>12} {'norm(2nt)':>12}")

max_rel = 0.0
for ex in range(5):
    key, kp, kv = jax.random.split(key, 3)

    # worst-case-ish draw: widest sigma + largest |k0| = most spectral content
    x0, _, _ = sample_wavepacket_params(kp, cfg)
    psi0 = realize_wavepacket(x0, cfg.psi0_sigma_hi, cfg.psi0_k0_hi, nx)
    dphi = 2*jnp.pi / nx
    psi0 = psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2) * dphi)

    coeffs = sample_potential_spectrum(kv, None, cfg.v_length_scale,
                                       cfg.v_amplitude, cfg.v_n_coeffs)
    V = realize_potential(coeffs, nx)

    nt = cfg.nt_super
    psi_nt   = evolve(psi0, V, nx, T/nt,      nt)
    psi_2nt  = evolve(psi0, V, nx, T/(2*nt),  2*nt)   # reference: twice the steps

    rel = float(jnp.linalg.norm(psi_nt - psi_2nt) / jnp.linalg.norm(psi_2nt))
    n1  = float(jnp.sum(jnp.abs(psi_nt)**2)  * dphi)
    n2  = float(jnp.sum(jnp.abs(psi_2nt)**2) * dphi)
    max_rel = max(max_rel, rel)
    print(f"{ex:>3} {rel:>20.2e} {n1:>12.8f} {n2:>12.8f}")

print(f"\nmax rel diff: {max_rel:.2e}")
print("CONVERGED (nt_super sufficient)" if max_rel < 1e-6
      else "NOT CONVERGED — raise nt_super")