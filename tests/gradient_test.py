import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src.solver import evolve

nx = 128
phi = jnp.linspace(0.0, 2*jnp.pi, nx, endpoint = False)
dt, nt = 0.005, 200

a, x0, k0 = 0.6, jnp.pi, 3.0
psi0 = jnp.exp(-(phi-x0)**2 / (2*a**2)) * jnp.exp(1j * k0 * phi)
psi0 = (psi0 / jnp.sqrt(jnp.sum(jnp.abs(psi0)**2))).astype(jnp.complex128)

key = jax.random.PRNGKey(0)
kv, kd = jax.random.split(key)
V = jnp.cos(phi)+0.5*jnp.cos(2*phi)
dV = jax.random.normal(kv, (nx,))
dV = dV/ jnp.linalg.norm(dV)

def forward(V_):
    return evolve(psi0, V_, nx, dt, nt)

psi_T, jvp_ad = jax.jvp(forward, (V,), (dV,))

eps = 1e-6
jvp_fd = (forward(V+eps*dV) - forward(V-eps*dV))/(2*eps)

num = jnp.linalg.norm(jvp_ad-jvp_fd)
den = jnp.linalg.norm(jvp_fd)

print("‖J·dV (autodiff) − J·dV (FD)‖ / ‖J·dV (FD)‖ =", float(num/den))
print("autodiff ψ_T matches direct evolve:",
      bool(jnp.allclose(psi_T, forward(V))))    # JVP's primal output is ψ_T itself

print(f"\n{'eps':>10} {'rel err':>12}")
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    fd = (forward(V + eps*dV) - forward(V - eps*dV)) / (2*eps)
    rel = float(jnp.linalg.norm(jvp_ad - fd) / jnp.linalg.norm(fd))
    print(f"{eps:10.0e} {rel:12.2e}")


print(f"\n{'eps':>10} {'rel err':>12}")
for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
    fd = (forward(V + eps*dV) - forward(V - eps*dV)) / (2*eps)
    rel = float(jnp.linalg.norm(jvp_ad - fd) / jnp.linalg.norm(fd))
    print(f"{eps:10.0e} {rel:12.2e}")