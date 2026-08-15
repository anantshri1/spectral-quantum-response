# src/response.py  — Phase 4 response machinery
#
# NOTE: this module assumes x64 is already enabled by the DRIVER
# (scripts/compute_response.py sets jax_enable_x64 before importing anything).
# We do NOT set the flag here — a library shouldn't flip a global. If you import
# this in a fresh REPL without x64 on, J_true will silently run in float32.

import jax
import jax.numpy as jnp

from src.solver import evolve


def jacobian_true(psi0, V, nx, dt, nt):
    """
    J_true = d psi_T / d V, with psi0 held FIXED.

    Map differentiated:  V (real, (nx,))  ->  psi_T (complex, (nx,))
    Returns:             J (complex, (nx, nx)),  J[i, q] = d psi_T[i] / d V[q]

    Forward-mode (jacfwd): differentiates w.r.t. the FIRST positional arg of
    `f`, which we make V by closing over psi0. Forward-mode chosen because
    `evolve` is a lax.scan over nt steps — JVP carries a tangent alongside the
    primal (flat memory in nt), vs reverse-mode storing the whole trajectory.

    V is real => R^n -> C^n => no Wirtinger ambiguity; the complex J is
    exactly (d Re psi_T / d V) + i (d Im psi_T / d V), entrywise.
    """
    def f(V_):
        return evolve(psi0, V_, nx, dt, nt)   # psi0 fixed by closure

    return jax.jacfwd(f)(V)

def fd_check(psi0, V, nx, dt, nt, *, eps=1e-6, key=None):
    """
    Independent validation of jacobian_true via a directional finite difference.

    Pick a random unit direction u in V-space. Compare:
      autodiff JVP:  J @ u          (directional derivative from jacfwd)
      central FD  : [f(V+eps*u) - f(V-eps*u)] / (2*eps)
    These must agree to ~eps^2 (central FD is 2nd order) modulo float64 floor.

    Returns (rel_err, jvp_dir, fd_dir) so we can eyeball the vectors too.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    u = jax.random.normal(key, (nx,), dtype=V.dtype)
    u = u / jnp.linalg.norm(u)

    # autodiff directional derivative: (d psi_T / d V) applied to u, via a single JVP
    def f(V_):
        return evolve(psi0, V_, nx, dt, nt)
    _, jvp_dir = jax.jvp(f, (V,), (u,))          # one forward sweep, direction u

    # central finite difference in the same direction
    fd_dir = (f(V + eps * u) - f(V - eps * u)) / (2 * eps)

    rel_err = jnp.linalg.norm(jvp_dir - fd_dir) / jnp.linalg.norm(fd_dir)
    return rel_err, jvp_dir, fd_dir


def mode_coupling_matrix(J_pos):
    """
    Rotate a position-space Jacobian J_pos[i, q] = d psi_T(x_i) / d V(x_q)
    into the Fourier/mode basis: J(k, q) = d psi_hat_k / d V_hat_q.

    Rotation: FFT the OUTPUT index (i -> k), inverse-FFT the INPUT index (q -> q_mode).
    Uses the solver's FFT convention (jnp.fft.fft, integer wavenumbers on L=2pi).

      J_kq = FFT_i[ J_pos ] then IFFT along the q-axis.

    Pure linear algebra on the materialized matrix — no re-differentiation.
    For a (near-)local V, |J(k,q)| shows bands along k - q = const.
    """
    # output index i -> k : FFT down axis 0
    J_k_pos = jnp.fft.fft(J_pos, axis=0)
    # input index q -> mode : inverse FFT down axis 1
    J_kq = jnp.fft.ifft(J_k_pos, axis=1)
    return J_kq

def jacobian_fno(model, psi0, V):
    """
    J_FNO = d psi_T / d V for the trained surrogate, psi0 held FIXED.

    FNO.__call__(psi0, V) returns (nx, 2) = [Re psi_T, Im psi_T] as two REAL
    channels. We wrap it so the differentiated output is complex (nx,), making
    this byte-identical to jacobian_true's machinery:
        f(V) : R^n -> C^n,  differentiate w.r.t. real V, no Wirtinger.

    The a + 1j*b repackaging is a LINEAR map on real derivatives; jacfwd handles
    it correctly precisely because the INPUT (V) is real. Returns (nx, nx) complex.
    """
    def f(V_):
        out = model(psi0, V_)                 # (nx, 2), real
        return out[:, 0] + 1j * out[:, 1]     # (nx,), complex  <- recombine INSIDE f
    return jax.jacfwd(f)(V)

def response_error(J_fno, J_true):
    """
    Relative Frobenius error of the response Jacobian.
    ||J_fno - J_true||_F / ||J_true||_F  — a single scalar.

    Phase note: valid WITHOUT phase-alignment here because both Jacobians
    differentiate the SAME phase-convention forward (FNO trained to match the
    solver's psi_T). A derivative inherits its function's phase frame; no new
    global U(1) freedom is introduced. Would need alignment only across
    differently-phased forwards.
    """
    return jnp.linalg.norm(J_fno - J_true) / jnp.linalg.norm(J_true)

def frob_inner(A, B):
    """
    Complex Frobenius inner product  <A, B> = sum_ij conj(A_ij) B_ij.

    Convention: A occupies the conjugated ("left") slot, so <A,A> = ||A||_F^2,
    real and >= 0. Antilinear in A, linear in B. Its induced norm is exactly
    jnp.linalg.norm(., 'fro'), so it is the correct partner for response_error's
    Frobenius metric.
    """
    return jnp.sum(jnp.conj(A) * B)


def scale_ratio(J_fno, J_true):
    """
    r = ||J_fno||_F / ||J_true||_F  — the MAGNITUDE half of the error.
    r > 1: FNO Jacobian over-energetic; r < 1: under-energetic.
    (Recall raw response_error^2 = r^2 + 1 - 2 r cos, so r and cos fully
    reconstruct the scalar error.)
    """
    return jnp.linalg.norm(J_fno) / jnp.linalg.norm(J_true)


def cosine_alignment(J_fno, J_true):
    """
    Directional alignment, REAL-valued:
        cos = Re<J_fno, J_true> / (||J_fno|| ||J_true||).

    Re part is deliberate. A complex 'alignment' would smuggle a global phase
    into the metric, but response_error carries NO U(1) freedom (see its
    docstring): both Jacobians inherit the solver's fixed phase frame. Taking
    Re keeps that honest — genuine phase error stays counted as misalignment
    instead of being rotated away. cos in [-1, 1]; cos = 1 iff J_fno = c J_true
    for real c > 0.
    """
    num = jnp.real(frob_inner(J_fno, J_true))
    den = jnp.linalg.norm(J_fno) * jnp.linalg.norm(J_true)
    return num / den


def optimal_rescale(J_fno, J_true):
    """
    Best REAL scalar alpha minimizing ||alpha J_fno - J_true||_F, plus the
    residual it leaves:

        alpha* = Re<J_fno, J_true> / ||J_fno||^2
        residual = ||alpha* J_fno - J_true||_F / ||J_true||_F

    alpha REAL for the same reason cos takes Re: a complex alpha would re-absorb
    a global phase and flatter the model by hiding phase error as 'alignment'.

    Free correctness gate:  residual == sqrt(1 - cos^2)  exactly, with cos from
    cosine_alignment. Disagreement beyond ~1e-12 (f64) means the algebra or a
    dtype is wrong.

    Returns (alpha_star, residual).
    """
    alpha = jnp.real(frob_inner(J_fno, J_true)) / jnp.linalg.norm(J_fno) ** 2
    residual = jnp.linalg.norm(alpha * J_fno - J_true) / jnp.linalg.norm(J_true)
    return alpha, residual