# scripts/check_jvp_fno.py — Phase 6 Block 2: FNO JVP primitive + gates.
# Standalone. Proves J_fno·v (single forward-mode pass) matches the full
# jacobian_fno build, AND that reverse-over-forward grad_theta is well-formed,
# BEFORE the primitive goes near a training loop.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.responses import jacobian_fno
from scripts.train import load_split
from scripts.compute_response import load_fno_x64


def jvp_fno(model, psi0, V, v):
    """
    J_fno·v in one forward-mode pass. Differentiates the SAME map jacobian_fno
    does: f(V_) = model(psi0,V_) recombined to complex (nx,), then jvp in
    direction v instead of the full jacfwd. V real => no Wirtinger.
    Returns (nx,) complex.
    """
    def f(V_):
        out = model(psi0, V_)              # (nx, 2) real
        return out[:, 0] + 1j * out[:, 1]  # (nx,) complex
    _, jv = jax.jvp(f, (V,), (v,))
    return jv


if __name__ == "__main__":
    cfg = Config()
    model = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg)

    psi0, V, _ = load_split("data/test.npz", nx=cfg.nx)
    psi0 = psi0.astype(jnp.complex128)
    V    = V.astype(jnp.float64)

    # one direction, fixed key (independent of precompute stream)
    key = jax.random.PRNGKey(1)
    v = jax.random.normal(key, (cfg.nx,), dtype=jnp.float64)
    v = v / jnp.linalg.norm(v)

    # --- GATE 1: single-JVP == full jacobian_fno @ v ---
    jv_fast = jvp_fno(model, psi0[0], V[0], v)
    J_full  = jacobian_fno(model, psi0[0], V[0])     # (nx,nx) complex
    jv_ref  = J_full @ v
    rel = float(jnp.linalg.norm(jv_fast - jv_ref) / jnp.linalg.norm(jv_ref))
    print(f"[gate 1] jvp_fno vs jacobian_fno @ v: rel-err {rel:.2e}")
    assert rel < 1e-6, "FNO JVP disagrees with full jacobian_fno — STOP"
    print("[gate 1] OK — single-pass J_fno·v matches full build.")

    # --- GATE 2: reverse-over-forward grad_theta is well-formed ---
    # toy loss: ||J_fno·v - target||^2, target = a shifted copy so grad != 0.
    target = jv_ref * 1.1
    def loss_fn(m):
        jv = jvp_fno(m, psi0[0], V[0], v)
        return jnp.sum(jnp.abs(jv - target) ** 2)

    g = eqx.filter_grad(loss_fn)(model)              # reverse-over-forward
    leaves = [x for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))]
    gnorm = float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves)))
    allfinite = bool(jnp.all(jnp.array(
        [jnp.all(jnp.isfinite(x)) for x in leaves])))
    print(f"[gate 2] grad_theta finite: {allfinite} | ‖grad‖ = {gnorm:.4e}")
    assert allfinite and gnorm > 0, "nested grad ill-formed — STOP"

    # FD check on ONE weight leaf: does grad_theta match a numerical bump?
    spec  = model.blocks[0].spectral_conv
    gspec = g.blocks[0].spectral_conv
    i0 = (0, 0, 0)
    eps = 1e-6
    def loss_with_wr(val):
        m = eqx.tree_at(lambda mm: mm.blocks[0].spectral_conv.w_real,
                        model,
                        spec.w_real.at[i0].set(val))
        return loss_fn(m)
    base = spec.w_real[i0]
    fd = (loss_with_wr(base + eps) - loss_with_wr(base - eps)) / (2 * eps)
    ad = gspec.w_real[i0]
    rel_g = float(abs(fd - ad) / (abs(fd) + 1e-30))
    print(f"[gate 2] grad w_real[0,0,0]: AD {float(ad):.6e} | FD {float(fd):.6e} | rel {rel_g:.2e}")
    assert rel_g < 1e-4, "grad_theta disagrees with FD — reverse-over-forward miswired"
    print("[gate 2] OK — nested reverse-over-forward gradient verified against FD.")

    print("\nBlock 2 gates green. JVP primitive safe to lift into train.py.")