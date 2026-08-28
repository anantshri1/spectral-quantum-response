# scripts/train_dino.py — Phase 6 Block 3: DINO response-aware training.
# NEW FILE. Imports shared pieces from scripts.train (frozen, canonical) and
# adds only DINO machinery. train.py is NOT edited — its bit-identity is the
# provenance anchor for every Phase 3/4/5 checkpoint.
import jax
jax.config.update("jax_enable_x64", True)   # response JVP needs f64, as everywhere

import os, time
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow

from configs.default import Config
from src.fno import FNO
from scripts.train import (
    load_split, relative_l2, norm_violation, eval_step,
    get_git_commit,
)


def lambda_at(epoch, dino_lambda, warmup_epochs):
    """
    DINO weight schedule. Linear ramp 0 -> dino_lambda over [0, warmup_epochs],
    then hold. warmup_epochs=0 (default) => constant dino_lambda from epoch 0.

    CRITICAL: dino_lambda=0 returns 0.0 for ALL epochs regardless of warmup,
    so the lambda=0 bit-identity gate is untouched by warmup plumbing.

    First-class (not a later patch), mirroring the lr schedule pattern. Held
    FIXED across the lambda sweep when enabled, so warmup is a pure stability
    aid that doesn't smear the Pareto frontier.
    """
    if warmup_epochs <= 0:
        return dino_lambda
    frac = jnp.minimum(epoch / warmup_epochs, 1.0)
    return dino_lambda * frac


def make_batches_dino(psi0, V, uT, v, Jv_true, batch_size, key):
    """
    Like make_batches, but carries the precomputed directions v (n,K,nx) and
    references Jv_true (n,K,nx) through the SAME permutation as (psi0,V,uT).

    ALIGNMENT TRAP: if v/Jv_true don't ride the identical perm as V, every DINO
    residual compares against the wrong sample's reference — silent, no error,
    catastrophic. The single shared `perm` is load-bearing.
    """
    n = psi0.shape[0]
    perm = jax.random.permutation(key, n)
    psi0, V, uT = psi0[perm], V[perm], uT[perm]
    v, Jv_true = v[perm], Jv_true[perm]              # SAME perm — non-negotiable
    for i in range(n // batch_size):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        yield psi0[sl], V[sl], uT[sl], v[sl], Jv_true[sl]


def jvp_fno_batched(model, psi0_b, V_b, v_b):
    """
    J_fno·v per batch element, one forward-mode pass each. Same map as
    jacobian_fno (proven in Block 2 to f64 floor). vmap over the batch.
    Returns (B, nx) complex.
    """
    def single(psi0_i, V_i, v_i):
        def f(V_):
            out = model(psi0_i, V_)              # (nx, 2) real
            return out[:, 0] + 1j * out[:, 1]    # (nx,) complex
        _, jv = jax.jvp(f, (V_i,), (v_i,))
        return jv
    return jax.vmap(single)(psi0_b, V_b, v_b)    # (B, nx) complex


@eqx.filter_jit
def train_step_dino(model, psi0_b, V_b, uT_b, v_b, Jv_true_b,
                    opt_state, optimizer, lam, C_norm):
    """
    L = rel_L2(psi_T) + lam * ||J_fno·v - J_true·v||^2 / C_norm

    - forward term: identical to train.py's loss (rel_L2 over (nx,2) [Re,Im]).
    - DINO term: batch of single-direction JVPs vs precomputed references.
    - lam threaded as a TRACED arg (not closed over) so the same compiled step
      serves every epoch as the schedule ramps — no recompile per lambda value.
    - C_norm likewise traced (a scalar); global constant from Block 1.
    """
    def loss_fn(model):
        pred = jax.vmap(model)(psi0_b, V_b)                 # (B, nx, 2)
        fwd  = relative_l2(pred, uT_b)                      # scalar, forward

        jv_fno = jvp_fno_batched(model, psi0_b, V_b, v_b)   # (B, nx) complex
        # residual per sample: ||Jfno·v - Jtrue·v||^2 summed over nx
        resid = jnp.sum(jnp.abs(jv_fno - Jv_true_b) ** 2, axis=-1)  # (B,)
        dino = jnp.mean(resid) / C_norm                    # scalar, normalized

        return fwd + lam * dino, (fwd, dino)

    (loss, (fwd, dino)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, fwd, dino

def train_dino(cfg, dino_lambda=0.0, warmup_epochs=0, seed=None, n_modes=None, n_train=None, tag=""):
    if seed is not None:
        cfg.seed = seed
    if n_modes is not None:
        cfg.n_modes = n_modes

    # --- data: raw psi0/V/uT, widened to f64 for the response JVP ---
    psi0_tr, V_tr, uT_tr = load_split("data/train.npz", cfg.nx)
    psi0_te, V_te, uT_te = load_split("data/test.npz",  cfg.nx)
    psi0_tr, V_tr = psi0_tr.astype(jnp.complex128), V_tr.astype(jnp.float64)
    psi0_te, V_te = psi0_te.astype(jnp.complex128), V_te.astype(jnp.float64)
    uT_tr, uT_te  = uT_tr.astype(jnp.float64), uT_te.astype(jnp.float64)

    # --- precomputed directions + references (Block 1) ---
    pc = np.load("results/dino_precompute.npz")
    v_all      = jnp.asarray(pc["v"]).astype(jnp.float64)        # (2000,8,128)
    Jv_all     = jnp.asarray(pc["Jv_true"]).astype(jnp.complex128)  # (2000,8,128)
    C_norm     = float(pc["C_norm"])
    K          = int(pc["K"])

    # ALIGNMENT GATE: precompute rows must match load_split rows byte-for-byte.
    # Both read train.npz positionally; assert V agrees before trusting the pair.
    assert jnp.allclose(v_all.shape[0], V_tr.shape[0]), "row count mismatch"
    assert float(jnp.max(jnp.abs(jnp.asarray(pc["V"]) - V_tr))) < 1e-12, \
        "precompute V != load_split V — row alignment broken, STOP"

    # --- N-subset for the response-vs-N sweep (additive; None => full 2000) ---
    # Contiguous prefix keeps the precompute alignment for free: pc["V"] == V_tr
    # row-for-row (gated just above), so [:n_train] on BOTH the data and the
    # references stays aligned. Nested across N, matches train.py's convention.
    # Slicing AFTER the gate means the byte-for-byte check still runs on full 2000.
    if n_train is not None:
        psi0_tr, V_tr, uT_tr = psi0_tr[:n_train], V_tr[:n_train], uT_tr[:n_train]
        v_all,   Jv_all      = v_all[:n_train],   Jv_all[:n_train]

    n_actual = psi0_tr.shape[0]
    print(f"[dino] N={n_actual} seed={cfg.seed} M={cfg.n_modes} "
          f"lambda={dino_lambda} warmup={warmup_epochs} C_norm={C_norm:.4e}")

    # --- model + optimiser: IDENTICAL construction to train.py ---
    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                n_blocks=cfg.n_fno_blocks, key=mk)

    n_batches = n_actual // cfg.batch_size
    schedule = optax.exponential_decay(
        init_value=cfg.learning_rate,
        transition_steps=cfg.lr_decay_every * n_batches,
        decay_rate=cfg.lr_decay_rate, staircase=True)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- DEDICATED dino_key: separate stream, never touches model/shuffle ---
    # This is what keeps lambda=0 bit-identical: direction-selection draws come
    # from here, so the (psi0,V,uT) shuffle key stream is byte-for-byte the
    # frozen train.py stream. Split off the top, evolved independently.
    dino_key = jax.random.PRNGKey(cfg.seed + 10_000)  # +offset: disjoint from init

    subdir = f"checkpoints/{tag}" if tag else "checkpoints"
    os.makedirs(subdir, exist_ok=True)
    wsuffix = f"_warmup{warmup_epochs}" if tag else ""   # only tag non-canonical runs

    mlflow.set_experiment(cfg.experiment_name)
    run_name = f"dino_N{n_actual}_seed{cfg.seed}_M{cfg.n_modes}_lam{dino_lambda}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": "fno_dino", "n_train": n_actual, "seed": cfg.seed,
            "n_modes": cfg.n_modes, "n_channels": cfg.n_channels,
            "n_fno_blocks": cfg.n_fno_blocks, "lr": cfg.learning_rate,
            "lr_decay_rate": cfg.lr_decay_rate, "lr_decay_every": cfg.lr_decay_every,
            "batch_size": cfg.batch_size, "n_epochs": cfg.n_epochs,
            "t_end": cfg.t_end, "dino_lambda": dino_lambda,
            "warmup_epochs": warmup_epochs, "K": K, "C_norm": C_norm,
            "dir_seed": int(pc["dir_seed"]), "git_commit": get_git_commit(),
        })

        for epoch in range(cfg.n_epochs):
            lam = lambda_at(epoch, dino_lambda, warmup_epochs)

            # shuffle key: SAME stream as train.py (key, sk = split(key))
            key, sk = jax.random.split(key)
            # direction-selection key: SEPARATE stream
            dino_key, dk = jax.random.split(dino_key)

            # one direction index per sample per epoch, from the SEPARATE stream
            idx = jax.random.randint(dk, (n_actual,), 0, K)     # (n,) in [0,K)
            v_sel  = jnp.take_along_axis(
                v_all, idx[:, None, None], axis=1).squeeze(1)   # (n,nx)
            Jv_sel = jnp.take_along_axis(
                Jv_all, idx[:, None, None], axis=1).squeeze(1)  # (n,nx)

            losses, fwds, dinos = [], [], []
            for pb, vb, ub, vv, jj in make_batches_dino(
                    psi0_tr, V_tr, uT_tr, v_sel, Jv_sel, cfg.batch_size, sk):
                model, opt_state, loss, fwd, dino = train_step_dino(
                    model, pb, vb, ub, vv, jj, opt_state, optimizer, lam, C_norm)
                losses.append(float(loss)); fwds.append(float(fwd))
                dinos.append(float(dino))

            rel, nv = eval_step(model, psi0_te, V_te, uT_te, cfg.nx)
            mlflow.log_metrics({
                "train_loss": float(np.mean(losses)),
                "train_fwd_rel_l2": float(np.mean(fwds)),
                "train_dino_resid": float(np.mean(dinos)),
                "test_rel_l2": float(rel),
                "norm_violation": float(nv),
                "lambda": float(lam),
            }, step=epoch)

            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | lam {float(lam):.3f} | "
                      f"fwd {float(np.mean(fwds)):.4f} | "
                      f"dino {float(np.mean(dinos)):.4f} | "
                      f"test {float(rel):.4f} | nv {float(nv):.4f}")
            if epoch % 50 == 0:
                eqx.tree_serialise_leaves(
                    f"{subdir}/dino_N{n_actual}_seed{cfg.seed}_M{cfg.n_modes}"
                    f"_lam{dino_lambda}{wsuffix}_epoch{epoch:04d}.eqx", model)

        eqx.tree_serialise_leaves(
            f"{subdir}/dino_N{n_actual}_seed{cfg.seed}_M{cfg.n_modes}"
            f"_lam{dino_lambda}{wsuffix}_final.eqx", model)
        print(f"  done. test rel-L2: {float(rel):.4f} | nv: {float(nv):.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dino_lambda", type=float, default=0.0)
    p.add_argument("--warmup",      type=int,   default=0, dest="warmup_epochs")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--n_modes",     type=int,   default=None)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--tag", type=str, default="",
               help="Non-canonical run tag: writes to checkpoints/<tag>/ with a "
                    "_warmup{N} filename suffix. Empty (default) = canonical path.")
    args = p.parse_args()
    train_dino(Config(), dino_lambda=args.dino_lambda,
               warmup_epochs=args.warmup_epochs, seed=args.seed,
               n_modes=args.n_modes, tag=args.tag)