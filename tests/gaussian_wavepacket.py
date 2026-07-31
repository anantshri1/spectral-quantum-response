import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src.solver import evolve

nx = 128
phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint=False)
dphi = phi[1] - phi[0]
x0 = jnp.pi

a = 0.6                                     # wider ⇒ spreads slower ⇒ stays line-like longer
psi0 = jnp.exp(-(phi - x0)**2 / (2*a**2)).astype(jnp.complex128)
psi0 = psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2) * dphi)

sigma0_density = a / jnp.sqrt(2)

def measured_width(psi):
    p = jnp.abs(psi)**2
    p = p / jnp.sum(p * dphi)
    mean = jnp.sum(phi * p * dphi)
    var  = jnp.sum((phi - mean)**2 * p * dphi)
    return jnp.sqrt(var)

def analytic_width(t):
    return jnp.sqrt(sigma0_density**2 + (t / (2*sigma0_density))**2)

V0 = jnp.zeros(nx)
print(f"{'t':>6} {'measured σ':>12} {'analytic σ':>12} {'rel err':>10}")
for t_eval in [0.0, 0.05, 0.10, 0.15, 0.20]:
    dt, nt = 0.001, int(round(t_eval / 0.001))
    psi_t = psi0 if nt == 0 else evolve(psi0, V0, nx, dt, nt)
    m = float(measured_width(psi_t)); an = float(analytic_width(t_eval))
    print(f"{t_eval:6.2f} {m:12.5f} {an:12.5f} {abs(m-an)/an:10.2e}")