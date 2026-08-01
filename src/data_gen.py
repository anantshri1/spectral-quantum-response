import jax
import jax.numpy as jnp


def sample_potential_spectrum(key, k_max_env, length_scale, amplitude, n_coeffs):
    """
    Sample the Fourier coefficients of a smooth real periodic potential V(φ).

    V is defined by its spectrum, not grid values — so the SAME V can be
    realized at any resolution by irfft'ing onto that grid (zero-padding the
    unused high modes). This is what makes super-resolution honest: V^{1024}
    and V^{128} are the same continuous function.

    Envelope: coefficient k has std ∝ exp(-(k·length_scale)²/2). Larger
    length_scale ⇒ faster decay ⇒ only low-k survive ⇒ smoother V.

    Returns complex coefficients c_k for k=0..n_coeffs-1 (rfft layout).
    """
    k = jnp.arange(n_coeffs)                              # 0,1,...,n_coeffs-1
    envelope = jnp.exp(-(k * length_scale)**1 / 1.0)      # spectral decay

    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, (n_coeffs,)) * envelope
    im = jax.random.normal(ki, (n_coeffs,)) * envelope
    c = re + 1j * im

    # k=0 (DC) must be real: it's the spatial mean of a real function.
    # A complex DC term would make irfft produce a complex V.
    c = c.at[0].set(jnp.real(c[0]))

    # normalize so the realized V has the requested amplitude (peak-ish scale)
    return c * amplitude

"""
Trial 1: normalization issues
```python
def realize_potential(coeffs, nx):
    
    Evaluate the potential defined by `coeffs` onto an nx-point grid.

    Zero-pads (or truncates) the spectrum to nx//2+1 rfft modes, then irfft.
    Same coeffs + different nx = same function, different discretization.
    
    n_rfft = nx // 2 + 1
    n_have = coeffs.shape[0]

    if n_have < n_rfft:
        # zero-pad: higher grid has more modes, all the extra ones are empty
        padded = jnp.zeros(n_rfft, dtype=coeffs.dtype).at[:n_have].set(coeffs)
    else:
        # coarser grid: truncate to what it can represent
        padded = coeffs[:n_rfft]

    V = jnp.fft.irfft(padded, n=nx)                       # (nx,) real
    return V
```
"""

def realize_potential(coeffs, nx):
    """
    Evaluate the potential V(φ) = Σ_k Re[c_k e^{ikφ}] on an nx-point grid.

    Built as an explicit Fourier sum rather than irfft, so there is no
    normalization convention to track: the SAME coeffs give the SAME continuous
    function at any nx (the k-th term is c_k e^{ikφ}, independent of grid).
    This is what makes super-resolution honest — V^{1024} and V^{128} are
    literally the same series sampled at different densities.

    coeffs: (n_coeffs,) complex, indexed by k = 0,1,...,n_coeffs-1.
    """
    phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)   # (nx,)
    k = jnp.arange(coeffs.shape[0])                          # (n_coeffs,)

    # phase[k, x] = e^{i k φ_x}  →  (n_coeffs, nx)
    phase = jnp.exp(1j * k[:, None] * phi[None, :])

    # V(φ) = Σ_k Re[c_k e^{ikφ}]. Real part taken once over the whole sum.
    # k=0 term is Re[c_0] (constant); k≥1 terms are the oscillations.
    V = jnp.sum(jnp.real(coeffs[:, None] * phase), axis=0)   # (nx,) real
    return V


def sample_wavepacket_params(key, cfg):
    """
    Draw the three parameters defining one Gaussian wavepacket:
      x0    ∈ [x0_lo, x0_hi)   — center on the ring
      sigma ∈ [sigma_lo, sigma_hi) — real-space width
      k0    ∈ {k0_lo, ..., k0_hi}   — integer carrier momentum

    Parameters, not grid values — so (like V) the SAME packet can be realized
    at any resolution. k0 is integer because e^{ik0φ} must be single-valued on
    the ring (match at φ=0 and 2π).
    """
    kx, ks, kk = jax.random.split(key, 3)
    x0    = jax.random.uniform(kx, (), minval=cfg.psi0_x0_lo, maxval=cfg.psi0_x0_hi)
    sigma = jax.random.uniform(ks, (), minval=cfg.psi0_sigma_lo, maxval=cfg.psi0_sigma_hi)
    k0    = jax.random.randint(kk, (), cfg.psi0_k0_lo, cfg.psi0_k0_hi + 1)  # +1: randint excludes high
    return x0, sigma, k0


def realize_wavepacket(x0, sigma, k0, nx, n_images=2):
    """
    Evaluate a PERIODIC (wrapped) Gaussian wavepacket on an nx-point grid.

    ψ₀(φ) = Σ_n exp(-(φ - x0 - 2πn)²/(2σ²)) · e^{i k0 φ},   n ∈ [-n_images, n_images]

    The image sum is the honest periodic Gaussian (a theta function): each copy
    shifted by 2πn makes the packet wrap correctly instead of being truncated at
    the boundary. For σ ≪ 2π only the nearest images matter — each further one is
    suppressed by exp(-(2πn)²/2σ²) (~1e-42 at n=1, σ=0.6), so n_images=2 is exact
    to float64. This is what avoids the 1/k boundary tail that a truncated
    Gaussian would inject into the high-k modes.

    Returned UNNORMALIZED — the caller normalizes after realizing (norm is a
    grid-dependent quadrature, so it's applied per-resolution).
    """
    phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)      # (nx,)

    # envelope: sum of shifted Gaussian images
    envelope = jnp.zeros(nx)
    for n in range(-n_images, n_images + 1):
        envelope = envelope + jnp.exp(-(phi - x0 - 2*jnp.pi*n)**2 / (2*sigma**2))

    carrier = jnp.exp(1j * k0 * phi)                           # plane-wave momentum
    return (envelope * carrier).astype(jnp.complex128)         # (nx,) complex

"""
# Diagnostic runs

```
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


key = jax.random.PRNGKey(0)
coeffs = sample_potential_spectrum(key, k_max_env=None,
                                   length_scale=0.3, amplitude=1.0, n_coeffs=64)

V128  = realize_potential(coeffs, 128)
V256  = realize_potential(coeffs, 256)
V1024 = realize_potential(coeffs, 1024)

# 1. V is real (the DC fix + irfft working)
print("V128 real:", jnp.allclose(jnp.imag(V128) if jnp.iscomplexobj(V128) else 0.0, 0.0),
      "| dtype:", V128.dtype)                              # real float64

# 2. THE super-res property: coarse grid = subsampled fine grid.
#    V128 must equal V1024 sampled every 8th point — same function.
print("V128 == V1024[::8]:", jnp.allclose(V128, V1024[::8], atol=1e-10))   # True — critical

# 3. length_scale controls smoothness: larger ⇒ less high-k energy
for ls in [0.1, 0.3, 0.8]:
    c = sample_potential_spectrum(key, None, ls, 1.0, 64)
    hi = jnp.sum(jnp.abs(c[16:])**2) / jnp.sum(jnp.abs(c)**2)   # fraction of energy above k=16
    print(f"  length_scale={ls}: high-k energy fraction = {float(hi):.7f}")

key = jax.random.PRNGKey(0)
coeffs = sample_potential_spectrum(key, None, 0.3, 1.0, 64)
V128  = realize_potential(coeffs, 128)
V1024 = realize_potential(coeffs, 1024)

# is it a pure scale factor?
ratio = V1024[::8] / V128
print("ratio min/max:", float(ratio.min()), float(ratio.max()))   # if ~constant → scale bug
print("ratio ≈ 8?:", float(jnp.median(ratio)))

key = jax.random.PRNGKey(0)
print(f"{'ls':>6} {'high-k(>16) frac':>18} {'V_max (nx=128)':>16}")
for ls in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    # average over a few keys — one draw is noisy
    fracs, vmaxes = [], []
    for s in range(8):
        c = sample_potential_spectrum(jax.random.PRNGKey(s), None, ls, 1.0, 64)
        fracs.append(float(jnp.sum(jnp.abs(c[16:])**2) / jnp.sum(jnp.abs(c)**2)))
        from src.data_gen import realize_potential
        vmaxes.append(float(jnp.max(jnp.abs(realize_potential(c, 128)))))
    print(f"{ls:6.2f} {sum(fracs)/len(fracs):18.6f} {sum(vmaxes)/len(vmaxes):16.4f}")

from configs.default import Config          # adjust import to your actual path

cfg = Config()
key = jax.random.PRNGKey(0)
x0, sigma, k0 = sample_wavepacket_params(key, cfg)
print(f"sampled: x0={float(x0):.3f}, sigma={float(sigma):.3f}, k0={int(k0)}")

psi128  = realize_wavepacket(x0, sigma, k0, 128)
psi1024 = realize_wavepacket(x0, sigma, k0, 1024)

# 1. complex, right shape
print("psi128:", psi128.shape, psi128.dtype)                          # (128,) complex128

# 2. THE super-res property (same as V): coarse == fine subsampled
print("psi128 == psi1024[::8]:", jnp.allclose(psi128, psi1024[::8], atol=1e-12))  # True

# 3. periodicity: no discontinuity at the wrap. Compare endpoints via the
#    circular difference — a truncated (non-wrapped) Gaussian would show a jump.
#    Check the function is smooth across φ=0 by comparing ψ[0] to ψ[-1] scaled
#    by the expected one-step carrier phase.
step_phase = jnp.exp(1j * k0 * (2*jnp.pi/128))
print("smooth across wrap:", jnp.allclose(psi128[0], psi128[-1]*step_phase, rtol=1e-2))

# 4. band-limit: packet's spectrum should sit inside ~16 modes.
#    Center a worst-case packet (widest sigma, largest k0) and check high-k energy.
x0w, _, _ = sample_wavepacket_params(key, cfg)
psi_w = realize_wavepacket(x0w, cfg.psi0_sigma_hi, cfg.psi0_k0_hi, 128)
spec = jnp.abs(jnp.fft.fft(psi_w))**2
hi = float(jnp.sum(spec[16:113]) / jnp.sum(spec))   # energy outside lowest+highest 16 (fft layout)
print(f"worst-case high-k fraction: {hi:.4f}")      # want small, < ~0.01

# worst case: packet centered exactly on the seam
x0, sigma, k0 = 0.0, 0.324, -1          # x0=0 IS the seam
psi = realize_wavepacket(x0, sigma, k0, 128)

# adjacent-point differences, treating the array as circular
d = jnp.abs(jnp.diff(psi, append=psi[:1]))   # |ψ[i+1]-ψ[i]|, last wraps to ψ[0]
seam_jump   = float(d[-1])                    # the wrap: |ψ[0] - ψ[-1]|
typical_jump = float(jnp.median(d))           # typical adjacent difference
max_jump    = float(jnp.max(d))

print(f"seam jump:     {seam_jump:.3e}")
print(f"typical jump:  {typical_jump:.3e}")
print(f"max jump:      {max_jump:.3e}")
print("seam is normal:", seam_jump <= max_jump * 1.01)   # seam not anomalous → smooth
```
"""