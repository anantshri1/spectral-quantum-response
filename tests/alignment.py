# gate_block1.py — validate the linalg identities on SYNTHETIC data.
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)          # match response.py's f64 regime
from src.responses import (frob_inner, scale_ratio, cosine_alignment,
                          optimal_rescale, mode_coupling_matrix)

key = jax.random.PRNGKey(0); nx = 64
def rand_cplx(k, shape):
    kr, ki = jax.random.split(k)
    return jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)

k1, k2, key = jax.random.split(key, 3)
A, B = rand_cplx(k1, (nx, nx)), rand_cplx(k2, (nx, nx))

# Gate (a): residual == sqrt(1 - cos^2)  on independent random matrices
cos = cosine_alignment(A, B); alpha, resid = optimal_rescale(A, B)
print(f"[a] cos={cos:+.6f} alpha={alpha:+.6f} resid={resid:.6f} "
      f"sqrt(1-cos^2)={jnp.sqrt(1-cos**2):.6f} |diff|={abs(resid-jnp.sqrt(1-cos**2)):.2e}")

# Gate (b1): parallel real multiple  -> cos=1, resid=0, alpha=c
_, residP = optimal_rescale(3.7*A, A)
print(f"[b1 parallel] cos={cosine_alignment(3.7*A, A):.12f} (want 1)  resid={residP:.2e} (want 0)")

# Gate (b2): phase-rotated B=i*A. Complex-alpha would call this 'aligned';
# our REAL-alpha metric MUST call it orthogonal. This is the design in action.
aI, residI = optimal_rescale(1j*A, A)
print(f"[b2 i*A] cos={cosine_alignment(1j*A, A):.2e} (want 0)  alpha={aI:.2e} (want 0)  resid={residI:.12f} (want 1)")

# Gate (c): unitary/mode-basis invariance — same rotation on both, cos & resid unchanged
Akq, Bkq = mode_coupling_matrix(A), mode_coupling_matrix(B)
cq = cosine_alignment(Akq, Bkq); _, rq = optimal_rescale(Akq, Bkq)
print(f"[c] cos pos={cos:+.12f} kq={cq:+.12f} |diff|={abs(cos-cq):.2e}")
print(f"[c] resid pos={resid:.12f} kq={rq:.12f} |diff|={abs(resid-rq):.2e}")