# probe_cast.py  — throwaway, delete after
import numpy as np
import jax

# ---- Part 1: is float32 -> float64 bitwise lossless? ----
# Take a spread of "awkward" float32 values: random, plus tricky magnitudes.
rng = np.random.default_rng(0)
x32 = rng.standard_normal(100_000).astype(np.float32)
x32 = np.concatenate([x32, np.array([1/3, 0.1, 7.7, 1e-30, 1e30], np.float32)])

x64 = x32.astype(np.float64)          # widen
back = x64.astype(np.float32)         # narrow again

# Lossless <=> round-trip returns identical BITS (not just ==; catches -0.0/nan edge cases)
same_bits = (x32.view(np.uint32) == back.view(np.uint32)).all()
print("float32->float64->float32 bit-identical:", bool(same_bits))

# Direct check that the widened value equals the original exactly
print("max |x64 - x32| :", float(np.max(np.abs(x64 - x32.astype(np.float64)))))  # expect 0.0

# ---- Part 2: does the float32-trained model give the same forward in float64? ----
# Run this in a FRESH process with x64 ON.
import jax; jax.config.update("jax_enable_x64", True)   # must be before jnp work
import jax.numpy as jnp, equinox as eqx
from configs.default import Config
from src.fno import FNO
from scripts.train import load_split

cfg = Config()

# build template, PIN inexact leaves to float32 to match the file, deserialise, THEN cast up
template = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
               n_blocks=cfg.n_fno_blocks, key=jax.random.PRNGKey(0))

# split into (inexact arrays) and (everything else), cast only the arrays, recombine
arrays, static = eqx.partition(template, eqx.is_inexact_array)
arrays = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), arrays)
template = eqx.combine(arrays, static)

model32 = eqx.tree_deserialise_leaves("checkpoints/fno_N2000_seed0_final.eqx", template)

# one sample through the model, float32 weights but x64-permitted arithmetic
psi0, V, _ = load_split("data/test.npz", nx=cfg.nx)
out = model32(psi0[0], V[0])
print("forward dtype under x64:", out.dtype)   # tells us what x64 promotes to