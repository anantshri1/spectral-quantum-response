# tests/probe_x64_off.py  — throwaway, delete after
# RUN IN A FRESH PROCESS. Do NOT enable x64 anywhere in this file.
import jax                      # x64 stays OFF (default)
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from configs.default import Config
from src.fno import FNO
from scripts.train import load_split, relative_l2, eval_step

cfg = Config()

# --- load checkpoint (x64 OFF => template leaves are already float32, file matches) ---
template = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
               n_blocks=cfg.n_fno_blocks, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/fno_N2000_seed0_final.eqx", template)

# --- (1) dtype regression: spectral path must be complex64 under x64-off ---
psi0, V, uT = load_split("data/test.npz", nx=cfg.nx)
out = model(psi0[0], V[0])
print("forward dtype (x64 off):", out.dtype)          # expect float32

# peek the spectral buffer dtype directly: FFT of the real-part channel
xf = jnp.fft.rfft(jnp.real(psi0[0]))
print("rfft dtype (x64 off)   :", xf.dtype)            # expect complex64

# --- (2) number regression: forward rel-L2 unchanged from locked 0.0169 ---
rel, nv = eval_step(model, psi0, V, uT, cfg.nx)
print(f"test rel-L2 : {float(rel):.4f}   (locked: 0.0169)")
print(f"norm-viol   : {float(nv):.4f}   (locked: 0.0057)")