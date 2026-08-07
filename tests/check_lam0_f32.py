# scripts/check_lam0_f32.py — DIAGNOSTIC ONLY. Does train.py's frozen f32
# path still reproduce 0.0169? Isolates precision as the sole variable.
# Writes NO checkpoints (monkeypatches serialise to a no-op). Throwaway.
import jax
# NOTE: x64 deliberately NOT enabled — this is the f32 regime Phase 3 used.

import equinox as eqx
from configs.default import Config
import scripts.train as T

# neuter checkpoint writes so we can't clobber canonical .eqx files
eqx.tree_serialise_leaves = lambda *a, **k: None

if __name__ == "__main__":
    # train.py's own train(), unmodified path, seed 0, M16, all 2000 samples.
    # This is byte-for-byte the Phase-3 recipe.
    T.train(Config(), n_train=None, seed=0, n_modes=16)