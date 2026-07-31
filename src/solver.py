# src/solver.py
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

# On nx=128 points, the FFT gives you 128 frequency bins. But frequencies on a ring are signed — e^{+ikφ} and e^{-ikφ} rotate opposite ways — and they run from -64 to +63, not 0 to 127. The FFT's convention packs them in a specific order: bins 0…63 hold k = 0,1,…,63, then bins 64…127 hold the negative frequencies k = -64,-63,…,-1.

def make_kinetic_phase(nx: int, dt: float) -> jax.Array:
    """
    Fourier-space kinetic propagator for a full step: exp(-i · ½k² · dt).

    On the ring (L=2π, ℏ=m=R=1) the free-rotor kinetic operator K = -½ ∂²/∂φ²
    acts on e^{ikφ} as ½k². In the Fourier basis exp(-iK·dt) is therefore a
    pointwise multiply by exp(-i·½k²·dt).

    k comes from fftfreq with d=1/nx so the values are the exact signed integers
    0,1,…,nx/2-1,-nx/2,…,-1 — matching numpy/jax's fft bin ordering. Do NOT
    replace with arange: the top half of the array is negative frequencies.
    """
    k = jnp.fft.fftfreq(nx, d=1.0 / nx)          # (nx,) signed integer wavenumbers
    return jnp.exp(-1j * 0.5 * k**2 * dt)         # (nx,) complex


def make_potential_half_phase(V: jax.Array, dt: float) -> jax.Array:
    """
    Real-space potential propagator for a HALF step: exp(-i · V · dt/2).

    V is diagonal in real space, so exp(-iV·τ) is just a pointwise complex
    multiply. The half-step (dt/2) is the outer factor of the Strang split
    exp(-iV dt/2) · exp(-iK dt) · exp(-iV dt/2).
    """
    return jnp.exp(-1j * V * dt / 2.0)            # (nx,) complex

"""
# Diagnostic
```python
import jax.numpy as jnp

nx, dt = 128, 0.5 / 1000

kin = make_kinetic_phase(nx, dt)
V   = jnp.zeros(nx)
pot = make_potential_half_phase(V, dt)

# 1. shapes and dtypes
print("kin:", kin.shape, kin.dtype)     # (128,) complex64
print("pot:", pot.shape, pot.dtype)     # (128,) complex64

# 2. both are pure phases → |·| = 1 everywhere
print("kin unit modulus:", jnp.allclose(jnp.abs(kin), 1.0))   # True
print("pot unit modulus:", jnp.allclose(jnp.abs(pot), 1.0))   # True

# 3. V=0 ⇒ potential factor is exactly 1 (no kick)
print("V=0 ⇒ pot=1:", jnp.allclose(pot, 1.0 + 0j))            # True

# 4. k=0 bin has zero kinetic phase (DC mode doesn't rotate)
print("k=0 phase = 1:", jnp.allclose(kin[0], 1.0 + 0j))       # True

# 5. the fftfreq ordering: bin 64 must be k=-64, so k²=4096
k = jnp.fft.fftfreq(nx, d=1.0/nx)
print("k[64] =", float(k[64]), "(expect -64.0)")              # -64.0
print("k[1], k[-1] =", float(k[1]), float(k[-1]))             # 1.0, -1.0
```
"""

def strang_step(psi, kin, pot_half):
    """
    One Strang step: exp(-iV dt/2) · exp(-iK dt) · exp(-iV dt/2).

    Written to lax.scan's step signature (carry, x) -> (new_carry, y):
      carry = psi (threads through), x unused (every step identical),
      y = None (we don't record the trajectory in production).
    """
    psi = pot_half * psi                       # half potential kick  (real space)
    psi = jnp.fft.ifft(kin * jnp.fft.fft(psi)) # full kinetic drift   (Fourier space)
    psi = pot_half * psi                       # half potential kick  (real space)
    return psi

"""
# Diagnostic

```python
nx, dt = 128, 0.01
phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)

kin      = make_kinetic_phase(nx, dt)
pot_half = make_potential_half_phase(jnp.zeros(nx), dt)   # V=0 ⇒ pot_half = 1

k_test = 3
psi0 = jnp.exp(1j * k_test * phi)                          # plane wave, k=3
psi1 = strang_step(psi0, kin, pot_half)

expected_phase = jnp.exp(-1j * 0.5 * k_test**2 * dt)       # analytic one-step rotation
print("norm preserved:", jnp.allclose(jnp.abs(psi1), jnp.abs(psi0)))          # True
print("matches analytic phase:", jnp.allclose(psi1, expected_phase * psi0, atol=1e-5))  # True
```
"""

def evolve(psi0, V, nx, dt, nt, *, record=False):
    """
    Evolve psi0 under H = -½∂²/∂φ² + V for nt steps of size dt.

    Returns psi_T (final state). If record=True, also returns the full
    (nt, nx) trajectory — useful for Phase-1 debugging, off in production.
    """
    kin      = make_kinetic_phase(nx, dt)
    pot_half = make_potential_half_phase(V, dt)

    def scan_step(psi, _):                     # (carry, x) -> (new_carry, y)
        psi = strang_step(psi, kin, pot_half)
        return psi, (psi if record else None)  # y = snapshot or nothing

    psi_T, traj = lax.scan(scan_step, psi0, xs=None, length=nt)
    return (psi_T, traj) if record else psi_T

def make_state(nx):
    p = phi(nx)
    psi0 = jnp.exp(-(p - jnp.pi)**2 / (2*0.5**2)) * jnp.exp(1j * 3 * p)
    psi0 = psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2))
    V = jnp.cos(p) + 0.5*jnp.cos(2*p)
    return psi0, V


"""
# Diagnostic

```python
nx, dt, nt = 128, 0.005, 1000
phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)
key = jax.random.PRNGKey(0)

# a non-trivial normalized wavepacket
psi0 = jnp.exp(-(phi - jnp.pi)**2 / (2*0.5**2)) * jnp.exp(1j * 3 * phi)
psi0 = psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2))

# a real, smooth-ish potential
V = jnp.cos(phi) + 0.5*jnp.cos(2*phi)

psi_T = evolve(psi0, V, nx, dt, nt)

norm0 = jnp.sum(jnp.abs(psi0)**2)
normT = jnp.sum(jnp.abs(psi_T)**2)
print("norm(0):", float(norm0))                    # 1.0
print("norm(T):", float(normT))                    # ~1.0 to machine precision
print("norm drift:", float(jnp.abs(normT - norm0)))# ~1e-6 or smaller
print("psi_T finite:", jnp.all(jnp.isfinite(psi_T)).item())

phi = lambda nx: jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)
nx, T = 128, 5.0     # fix total evolved time T = nt*dt

# --- Test A: does drift scale with nt at fixed T?  (H1 vs H2) ---
for nt in [500, 1000, 2000, 4000]:
    dt = T / nt
    psi0, V = make_state(nx)
    psi_T = evolve(psi0, V, nx, dt, nt)
    drift = float(jnp.abs(jnp.sum(jnp.abs(psi_T)**2) - 1.0))
    print(f"nt={nt:5d}  dt={dt:.2e}  drift={drift:.2e}")

# --- Test B: float64 should collapse round-off (confirms H1) ---
jax.config.update("jax_enable_x64", True)
nx, dt, nt = 128, 0.005, 1000
p = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)
psi0 = jnp.exp(-(p-jnp.pi)**2/(2*0.5**2)) * jnp.exp(1j*3*p)
psi0 = (psi0/jnp.sqrt(jnp.sum(jnp.abs(psi0)**2))).astype(jnp.complex128)
V = (jnp.cos(p) + 0.5*jnp.cos(2*p)).astype(jnp.float64)
psi_T = evolve(psi0, V, nx, dt, nt)
print("float64 drift:", float(jnp.abs(jnp.sum(jnp.abs(psi_T)**2) - 1.0)))  # expect ~1e-13
```
"""