"""
Legacy FNO from Burgers' solver:
What this block contains:
* eqx.Module: Equinox's base class for neural network layers. It automatically registers the class as a JAX pytree, meaning JAX's jit, grad, vmap etc. can see inside it and treat its array fields as leaves (differentiable parameters). You define parameters just by declaring them as class-level type annotations and setting them in __init__.
* eqx.field(static=True): Some fields are not JAX arrays — integers, strings, booleans. JAX cannot trace through these; they must be compile-time constants. eqx.field(static=True) tells Equinox "this field is not a leaf, treat it as a static constant baked into the computation graph." We need this for n_modes because we use it as a slice index (x_ft[:self.n_modes]), which must be a concrete Python int at trace time.
* Complex arrays in JAX: jnp.fft.rfft returns complex arrays. JAX supports complex arithmetic natively and handles gradients through complex ops correctly using Wirtinger calculus — you don't need to do anything special; jax.grad just works. We store our spectral weights as complex64 arrays directly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean


class SpectralConv1d(eqx.Module):
    """
    Spectral convolution layer for 1D signals.
    Implements: x-> iFFT(W_k * FFT(x)[:n_modes])
    where W_k is a complex-valued matrix for each Fourier mode k.
   
    Spectral weights stored as two real arrays so Adam sees only real gradients.
    Complex weights are formed inside __call__ and never leave the forward pass.
    """
    w_real: jax.Array  # float32, shape (n_modes, d_in, d_out)
    w_imag: jax.Array  # float32, shape (n_modes, d_in, d_out)
    n_modes: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_out: int, n_modes: int, *, key: jax.Array):
        self.n_modes = n_modes 
        # initialize real and imaginary parts independently with small normal noise.
        # scale by 1/(d_in * d_out) <- à la original FNO paper
        scale = 1.0/(d_in * d_out)
        k1, k2 = jax.random.split(key)
        self.w_real = jax.random.normal(k1, (n_modes, d_in, d_out)) * scale
        self.w_imag = jax.random.normal(k2, (n_modes, d_in, d_out)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (nx, d_in) - single sample, no axis

        nx = x.shape[0]

        # form complex weights here, inside forward pass only
        weights = self.w_real + 1j * self.w_imag          # (n_modes, d_in, d_out), complex64


        # ---- Step 1: FFT along the spatial axis -----------------
        # rfft exploits real-valued input: output has shape (nx//2 + 1, d_in)
        # Only positive frequences are kept; the negative ones are conjugate
        # symmetric and irfft reconstructs them automatically

        x_ft = jnp.fft.rfft(x,axis=0)       # (nx//2 + 1, d_in), complex64

        # ------- Step 2: Truncate to the first n_modes Fourier modes ---
        x_ft_trunc = x_ft[:self.n_modes,:]      # (n_modes, d_in)

        # ---- Step 3: complex linear map over modes ------------
        # For each mode k: out_ft[k] = weights[k] @ x_ft_trunc[k]
        # weights[k]: (d_in x d_out) C-valued matrix
        # einsum (Einstein summation) does this for all k

        out_ft = jnp.einsum('ki, kio->ko',
                            x_ft_trunc,
                            weights
        )

        # ----- Step 4: pad back to full frequency ---------------
        # irfft expects (nx//2 + 1) freq bins.
        # high-freq bins with zero are filled (the network learns which low modes matter)

        n_ft = nx // 2 + 1
        out_ft_full = jnp.zeros((n_ft, out_ft.shape[-1]), out_ft.dtype)
        out_ft_full = out_ft_full.at[:self.n_modes, :].set(out_ft)

        # ---- Step 5: Inverse FFT -----------------------
        # n = nx IS mandatory: tells irfft the target signal length
        return jnp.fft.irfft(out_ft_full, n=nx, axis=0)   # (nx, d_out)


"""
The architecture for the FNOBlock:
Each FNO block has two parallel branches:
```
x ──┬── SpectralConv1d ──────────────┬── (+) ── GELU ── out
    └── pointwise Linear (per point) ─┘
```
The spectral branch captures global structure (mixes information across all spatial locations via Fourier modes).
The pointwise linear branch captures local structure (same matrix applied independently at each spatial point).
Together, they cover both scales.
"""

class FNOBlock(eqx.Module):
    """
    One FNO layer: spectral conv + pointwise lienar, summed, then activation.
    Input and output both have shape (nx, d_v) preserving channel width in the process.
    """
    spectral_conv: SpectralConv1d
    linear: eqx.nn.Linear

    def __init__(self, d_v: int, n_modes: int, *, key:jax.Array):
        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(d_v, d_v, n_modes, key = k1)
        self.linear = eqx.nn.Linear(d_v, d_v, use_bias=True, key = k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (nx, d_v)
        spectral_out = self.spectral_conv(x)
        linear_out = jax.vmap(self.linear)(x)
        return jax.nn.gelu(spectral_out + linear_out)



class FNO(eqx.Module):
    """
    Full FNO model for 1D operator learning.
    Maps u0: (nx, ) -> uT: (nX, )
    """
    lifting: eqx.nn.Linear
    blocks: list
    proj1: eqx.nn.Linear
    proj2: eqx.nn.Linear

    def __init__(self, d_v: int, n_modes: int, n_blocks: int, *, key:jax.Array):
        keys = jax.random.split(key, n_blocks + 3)

        # Lifting: (4, ) -> (d_v, ) [input has 4 channels: Re ψ₀, Im ψ₀, V, grid]
        self.lifting = eqx.nn.Linear(4, d_v, use_bias = True, key = keys[0])

        # FNO Blocks: each maps (nx, d_v) -> (nx, d_v)
        self.blocks = [
            FNOBlock(d_v, n_modes, key=keys[1+i])
            for i in range(n_blocks)
        ] 

        # Projection: (d_v, ) -> (128, ) -> (1, )
        self.proj1 = eqx.nn.Linear(d_v, 128, use_bias = True, key = keys[-2])
        self.proj2 = eqx.nn.Linear(128, 2, use_bias = True, key = keys[-1])

    def __call__(self, psi0: jax.Array, V: jax.Array) -> jax.Array:
        # psi0: (nx,) complex  — initial wavefunction
        # V:    (nx,) real      — potential (the Phase-4 differentiation target)
        nx = psi0.shape[0]

        # periodic coordinate channel on [0, 2π); endpoint dropped since φ=2π ≡ φ=0
        grid = jnp.linspace(0.0, 2.0 * jnp.pi, nx, endpoint=False)

        # complex → 2 real channels, right here at the model boundary
        x = jnp.stack([jnp.real(psi0), jnp.imag(psi0), V, grid], axis=-1)  # (nx, 4)

        x = jax.vmap(self.lifting)(x)      # (nx, d_v)
        for block in self.blocks:
            x = block(x)                   # (nx, d_v)
        x = jax.vmap(self.proj1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.proj2)(x)        # (nx, 2)
        return x                           # (nx, 2) = [Re ψ_T, Im ψ_T]

"""
Diagnostics
```
import jax, jax.numpy as jnp, equinox as eqx
from src.fno import FNO

key = jax.random.PRNGKey(0)
model = FNO(d_v=32, n_modes=16, n_blocks=4, key=key)

psi0 = jax.random.normal(key, (128,)) + 1j * jax.random.normal(key, (128,))
V    = jax.random.normal(key, (128,))
out  = model(psi0, V)

print("output shape :", out.shape)                       # (128, 2)
print("output finite:", jnp.all(jnp.isfinite(out)).item())
n = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
print(f"params: {n:,}")   # ~139,938 — a hair above Burgers' 139,745
```

```
Lifting:          4×32 + 32         =      160
SpectralConv ×4:   4 × 2×(16×32×32) = 131,072   ← w_real + w_imag, both float32
Linear ×4:         4 × (32×32+32)   =   4,224
proj1:             32×128+128       =   4,224
proj2:             128×2+1          =     258
─────────────────────────────────────────────
Total:                               ~139,938
```

"""