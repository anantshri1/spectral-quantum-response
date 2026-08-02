# scripts/probe.py — disposable forward-learnability gate. NOT the real training run.
import jax, jax.numpy as jnp
import equinox as eqx
import optax

from configs.default import Config
from src.fno import FNO
from scripts.train import load_split, make_batches, train_step, eval_step

def probe(n_probe=300, n_epochs=150, batch_size=32):
    cfg = Config()

    # small fixed slice — the whole point is CHEAP
    psi0, V, uT = load_split("data/train.npz", cfg.nx)
    psi0, V, uT = psi0[:n_probe], V[:n_probe], uT[:n_probe]

    # held-out test for honest generalization signal (probe overfits 300 easily otherwise)
    psi0_te, V_te, uT_te = load_split("data/test.npz", cfg.nx)

    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes, n_blocks=cfg.n_fno_blocks, key=mk)

    # flat LR — no schedule. probe asks "does it move," not "squeeze the last %"
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print(f"probe: {n_probe} train / {psi0_te.shape[0]} test, {n_epochs} epochs, "
          f"n_modes={cfg.n_modes}, T={cfg.t_end}")

    for epoch in range(n_epochs):
        key, sk = jax.random.split(key)
        losses = []
        for pb, vb, ub in make_batches(psi0, V, uT, batch_size, sk):
            model, opt_state, loss = train_step(model, pb, vb, ub, opt_state, optimizer)
            losses.append(float(loss))
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            tr = float(jnp.mean(jnp.array(losses)))
            rel, nv = eval_step(model, psi0_te, V_te, uT_te, cfg.nx)
            print(f"  epoch {epoch:3d} | train {tr:.4f} | "
                  f"test {float(rel):.4f} | norm-viol {float(nv):.4f}")

    return model

if __name__ == "__main__":
    probe()