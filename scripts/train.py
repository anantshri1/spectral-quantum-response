import os, sys, subprocess, time, json

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow

from configs.default import Config
from src.fno import FNO

def relative_l2(pred, true):
    """
    pred, true: (nx, 2) or (B, nx, 2) — [Re, Im] channels.
    Frobenius norm over the last two axes == complex L2 norm
    ‖ψ‖₂ = sqrt(Σ_i (Re_i² + Im_i²)), so this IS complex relative L2
    with no complex array ever formed.
    """
    diff_norm = jnp.linalg.norm(pred - true, axis=(-2, -1))
    true_norm = jnp.linalg.norm(true, axis=(-2, -1))
    return jnp.mean(diff_norm / true_norm)

def load_split(path, nx=128):
    d = np.load(path)
    # Recombine ψ₀ to COMPLEX here. FNO.__call__ does jnp.real/jnp.imag on it,
    # so it needs complex in. Pass only the real part and jnp.imag(...) is all
    # zeros SILENTLY — no error, model trains on half its input. Load-bearing line.
    psi0 = (d[f"u0_re_{nx}"] + 1j * d[f"u0_im_{nx}"]).astype(np.complex64)
    V    =  d[f"V_{nx}"].astype(np.float32)
    # Target stacked [Re, Im] to match the model's (nx, 2) output and relative_l2.
    uT   = np.stack([d[f"uT_re_{nx}"], d[f"uT_im_{nx}"]], axis=-1).astype(np.float32)
    return jnp.array(psi0), jnp.array(V), jnp.array(uT)

def make_batches(psi0, V, uT, batch_size, key):
    n = psi0.shape[0]
    perm = jax.random.permutation(key, n)          # JAX PRNG, reproducible
    psi0, V, uT = psi0[perm], V[perm], uT[perm]
    for i in range(n // batch_size):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        yield psi0[sl], V[sl], uT[sl]              # three-way yield now, not two

@eqx.filter_jit
def train_step(model, psi0_b, V_b, uT_b, opt_state, optimizer):
    def loss_fn(model):
        pred = jax.vmap(model)(psi0_b, V_b)     # (B, nx, 2), vmap over BOTH inputs
        return relative_l2(pred, uT_b)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def norm_violation(pred, nx):
    """
    Mean |∫|ψ_pred|²dφ − 1| over the batch.  dφ = 2π/nx.
    pred: (B, nx, 2) = [Re, Im].  |ψ|² = Re² + Im², summed over nx, ×dφ.
    A LEARNED surrogate has no built-in reason to conserve norm — unitary
    evolution does, so departure from 1 measures how unphysical the fit is.
    Diagnostic only: never enters the loss (we're testing emergence, not imposing it).
    """
    prob = jnp.sum(pred[..., 0]**2 + pred[..., 1]**2, axis=-1) * (2 * jnp.pi / nx)  # (B,)
    return jnp.mean(jnp.abs(prob - 1.0))

@eqx.filter_jit
def eval_step(model, psi0, V, uT, nx):
    pred = jax.vmap(model)(psi0, V)             # (B, nx, 2)
    return relative_l2(pred, uT), norm_violation(pred, nx)

def get_git_commit():
    """Log the exact code state with each run. 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def train(cfg, n_train=None, seed=None):
    if seed is not None:
        cfg.seed = seed

    psi0_tr, V_tr, uT_tr = load_split("data/train.npz", cfg.nx)
    psi0_te, V_te, uT_te = load_split("data/test.npz",  cfg.nx)

    # n_train subsetting baked in NOW so Phase 5's (modes × N) grid reuses this
    # driver unchanged — no mid-project refactor, per the Burgers lesson.
    if n_train is not None:
        psi0_tr, V_tr, uT_tr = psi0_tr[:n_train], V_tr[:n_train], uT_tr[:n_train]

    n_actual = psi0_tr.shape[0]
    print(f"[fno] training on {n_actual} samples (seed={cfg.seed}, T={cfg.t_end})")

    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes, n_blocks=cfg.n_fno_blocks, key=mk)

    n_batches = n_actual // cfg.batch_size
    schedule = optax.exponential_decay(
        init_value=cfg.learning_rate,
        transition_steps=cfg.lr_decay_every * n_batches,   # decay in EPOCH units
        decay_rate=cfg.lr_decay_rate,
        staircase=True,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    os.makedirs("checkpoints", exist_ok=True)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=f"fno_N{n_actual}_seed{cfg.seed}"):
        mlflow.log_params({
            "model_type":     "fno",
            "n_train":        n_actual,
            "seed":           cfg.seed,
            "n_modes":        cfg.n_modes,
            "n_channels":     cfg.n_channels,
            "n_fno_blocks":   cfg.n_fno_blocks,
            "lr":             cfg.learning_rate,
            "lr_decay_rate":  cfg.lr_decay_rate,     # provenance: schedule reproducible
            "lr_decay_every": cfg.lr_decay_every,
            "batch_size":     cfg.batch_size,
            "n_epochs":       cfg.n_epochs,
            "t_end":          cfg.t_end,             # log T — it's an open decision
            "git_commit":     get_git_commit(),      # provenance: exact code state
        })

        for epoch in range(cfg.n_epochs):
            key, sk = jax.random.split(key)
            losses = []
            for pb, vb, ub in make_batches(psi0_tr, V_tr, uT_tr, cfg.batch_size, sk):
                model, opt_state, loss = train_step(model, pb, vb, ub, opt_state, optimizer)
                losses.append(float(loss))

            train_loss = float(np.mean(losses))
            rel, nv = eval_step(model, psi0_te, V_te, uT_te, cfg.nx)   # Delta 2: two-number unpack

            # Delta 3: norm_violation logged as a first-class metric, not just printed —
            # its trajectory IS a result for this project.
            mlflow.log_metrics(
                {"train_rel_l2": train_loss,
                 "test_rel_l2":  float(rel),
                 "norm_violation": float(nv)},
                step=epoch,
            )

            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | train {train_loss:.4f} | "
                      f"test {float(rel):.4f} | norm-viol {float(nv):.4f}")

            # Colab-safety: checkpoint every 50 epochs. A disconnect at epoch 480
            # costs 30 epochs, not the whole run.
            if epoch % 50 == 0:
                eqx.tree_serialise_leaves(
                    f"checkpoints/fno_N{n_actual}_seed{cfg.seed}_epoch{epoch:04d}.eqx", model)

        eqx.tree_serialise_leaves(
            f"checkpoints/fno_N{n_actual}_seed{cfg.seed}_final.eqx", model)
        print(f"  done. final test rel-L2: {float(rel):.4f} | norm-viol: {float(nv):.4f}")

    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=None, help="Training set size. Default: all 2000.")
    p.add_argument("--seed",    type=int, default=None, help="Overrides cfg.seed if given.")
    args = p.parse_args()
    train(Config(), n_train=args.n_train, seed=args.seed)

"""
# Diagnostic runs

```python
cfg = Config()
psi0, V, uT = load_split("data/train.npz", cfg.nx)

print("psi0:", psi0.shape, psi0.dtype)   # (2000, 128) complex64
print("V   :", V.shape, V.dtype)         # (2000, 128) float32
print("uT  :", uT.shape, uT.dtype)       # (2000, 128, 2) float32

# ∫|ψ₀|²dφ = 1  (dφ = 2π/nx) — validates the complex recombination physically
norm0 = jnp.sum(jnp.abs(psi0[0])**2) * (2*jnp.pi/cfg.nx)
print("norm ψ₀:", float(norm0))          # expect ≈ 1.0

key = jax.random.PRNGKey(0)
model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes, n_blocks=cfg.n_fno_blocks, key=key)

out1 = model(psi0[0], V[0])                          # single sample
print("single:", out1.shape, jnp.all(jnp.isfinite(out1)).item())  # (128, 2) True

outB = jax.vmap(model)(psi0[:4], V[:4])              # vmap over BOTH inputs
print("batched:", outB.shape)                        # (4, 128, 2)

print("rel-L2 (untrained):", float(relative_l2(outB, uT[:4])))    # ~O(1)
optimizer = optax.adam(cfg.learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# one train step on a small batch
model, opt_state, loss0 = train_step(model, psi0[:32], V[:32], uT[:32], opt_state, optimizer)
print("loss after 1 step:", float(loss0))       # ~1.0, finite

# eval on a slice — returns TWO numbers now
rel, nv = eval_step(model, psi0[:64], V[:64], uT[:64], cfg.nx)
print("rel-L2      :", float(rel))               # ~1.0 untrained
print("norm-violation:", float(nv))              # finite; likely O(0.1–1) untrained
```
"""