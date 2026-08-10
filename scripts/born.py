# scripts/born.py — Phase 7, discrete Born baseline
# x64 FIRST, before any jnp work (solver + closed form both need float64)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from configs.default import Config
from src.solver import evolve
from src.responses import mode_coupling_matrix     # match compute_response import


def born_jacobian(psi0, nx, dt, nt):
    """
    Closed-form discrete Born response  J0(k,q) = ∂ψ_T/∂V |_{V=0},
    in the mode_coupling (k, q_mode) basis. Built to equal
    IFFT_q[FFT_i[J_pos]] exactly, so it gates against jacfwd(evolve) at V=0.

        J0(k,q) = -(i/nx) e^{-iE_k T} ψ̂₀(k-q) S(Δ)
        Δ = k q - q²/2,  E_k = k²/2,  T = nt·dt
        S(Δ): trapezoidal kick-sum;  Δ=0 → T,  else geometric closed form
    """
    T = nt * dt

    kf = jnp.fft.fftfreq(nx, d=1.0 / nx)          # (nx,) signed ints, fft order
    ki = kf.astype(int)                           # for index arithmetic

    # Δ[k,q] = k q - q²/2  (exact in float64 for integer k,q → == 0 test is safe)
    # OLD: Delta = kf[:, None] * kf[None, :] - kf[None, :] ** 2 / 2.0
    # NEW: E_k - E_{k-q}, but k-q must be the ALIASED wavenumber (first Brillouin
    # zone), not the raw signed difference. Raw (k-q) runs to ±(nx-1); the free
    # propagator only ever acts on the folded mode in [-nx/2, nx/2). This is the
    # whole bug: raw (k-q)² is wildly wrong at the |k-q|>nx/2 corners.
    kmq       = kf[:, None] - kf[None, :]                  # raw signed diff, ±(nx-1)
    kmq_alias = ((kmq + nx // 2) % nx) - nx // 2           # fold to [-nx/2, nx/2)
    Delta     = 0.5 * (kf[:, None] ** 2 - kmq_alias ** 2)  # = E_k - E_{(k-q) folded}
    # --- S(Δ): closed form with the double-where NaN guard ---
    r    = jnp.exp(1j * Delta * dt)
    rnt  = r ** nt
    is0  = (Delta == 0.0)                          # Δ=0 at q=0 and q=2k
    safe = jnp.where(is0, 1.0 + 0j, 1.0 - r)       # sanitize denom BEFORE dividing
    geom = dt * (0.5 + 0.5 * rnt + (r - rnt) / safe)
    S    = jnp.where(is0, T + 0j, geom)            # select AFTER (masks the dummy)

    # ψ̂₀(k-q): gather fft(psi0) at frequency (k-q), array index (k-q) mod nx
    psi0hat     = jnp.fft.fft(psi0)
    idx         = (ki[:, None] - ki[None, :]) % nx
    psi0hat_kmq = psi0hat[idx]

    E_k     = 0.5 * kf ** 2
    phase_k = jnp.exp(-1j * E_k * T)               # depends on k only

    return -(1j / nx) * phase_k[:, None] * psi0hat_kmq * S


if __name__ == "__main__":
    from scripts.train import load_split

    cfg = Config()
    nx, nt = cfg.nx, cfg.nt
    dt = cfg.t_end / cfg.nt
    T  = cfg.t_end

    psi0, V, uT = load_split("data/test.npz", nx=nx)
    psi0 = psi0.astype(jnp.complex128)
    psi0_0 = psi0[0]

    # --- Block 3a self-check ---
    J0 = born_jacobian(psi0_0, nx, dt, nt)
    print(f"[3a] J0 shape {J0.shape}, dtype {J0.dtype}")     # (128,128) complex128
    print(f"[3a] finite: {bool(jnp.all(jnp.isfinite(J0)))}") # True (NaN guard works)

    # band check: |J0| should peak at offset k-q = k0  (validates the (k-q) gather sign)
    kf = jnp.fft.fftfreq(nx, d=1/nx).astype(int)
    kk, qq = jnp.meshgrid(kf, kf, indexing="ij")
    offset = ((kk - qq + nx // 2) % nx) - nx // 2
    absJ = jnp.abs(J0)
    dvals = jnp.unique(offset.ravel())
    prof  = jnp.array([jnp.mean(absJ.ravel()[offset.ravel() == d]) for d in dvals])
    peak  = int(dvals[jnp.argmax(prof)])
    k0    = int(np.load("data/test.npz")["p_k0"][0])
    print(f"[3a] |J0| band peak at k-q = {peak}   (expect k0 = {k0})")
    assert peak == k0, "band off-center — (k-q) gather sign likely flipped"
    print("[3a] band centered — indexing sane. Proceed to gate.")

    # --- Block 3b DIAGNOSTIC: does the gate error scale with nt? ---
    print("\n[3b-diag] gate rel-err vs nt (fixed dt):")
    for nt_test in [1, 2, 10, 100, 1000]:
        J0_cf = born_jacobian(psi0_0, nx, dt, nt_test)

        def f(V_, _nt=nt_test):
            return evolve(psi0_0, V_, nx, dt, _nt)
        J0_num = mode_coupling_matrix(jax.jacfwd(f)(jnp.zeros(nx)))

        num = jnp.linalg.norm(J0_cf - J0_num)
        den = jnp.linalg.norm(J0_num)
        print(f"    nt={nt_test:5d}  T={nt_test*dt:.4f}  rel-err={float(num/den):.3e}")

    # where does the nt=1000 error live? (max single-entry, and its Δ)
    J0_cf   = born_jacobian(psi0_0, nx, dt, nt)
    J0_num  = mode_coupling_matrix(jax.jacfwd(lambda V_: evolve(psi0_0, V_, nx, dt, nt))(jnp.zeros(nx)))
    E = jnp.abs(J0_cf - J0_num)
    kf = jnp.fft.fftfreq(nx, d=1/nx)
    ka, qa = jnp.unravel_index(jnp.argmax(E), E.shape)
    kv, qv = float(kf[ka]), float(kf[qa])
    print(f"[3b-diag] max-err entry at (k={kv:.0f}, q={qv:.0f}), "
          f"Δ={kv*qv - qv**2/2:.1f}, |err|={float(E[ka,qa]):.2e}")
    
    # --- Block 3b gate: closed form vs numerical jacfwd(evolve) at V=0 ---
    def f(V_):
        return evolve(psi0_0, V_, nx, dt, nt)

    J0_num_pos = jax.jacfwd(f)(jnp.zeros(nx))          # ∂ψ_T/∂V at V=0, (nx,nx) complex
    J0_num_kq  = mode_coupling_matrix(J0_num_pos)      # same rotation the closed form matches

    rel = jnp.linalg.norm(J0 - J0_num_kq) / jnp.linalg.norm(J0_num_kq)
    print(f"\n[3b] Born closed-form vs jacfwd@V=0 rel-err: {float(rel):.2e}   (expect ~1e-12)")
    assert float(rel) < 1e-9, "closed form disagrees with linearized solver — STOP, debug"
    print("[3b] GATE PASSED — discrete Born closed form is exact against the solver.")

    # ===== Block 5 — canonical response cache (f64-trained seed0) =====
    import os, time
    from tests.measure_floor import load_fno_f64_ckpt
    from src.responses import jacobian_true, jacobian_fno, response_error
    from scipy.stats import pearsonr, spearmanr

    V = V.astype(jnp.float64)     # J_true needs f64 V
    model = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", cfg)

    N = psi0.shape[0]
    Jtrue_kq  = np.empty((N, nx, nx), dtype=np.complex128)
    born_errs = np.empty(N)
    fno_errs  = np.empty(N)

    t0 = time.time()
    for s in range(N):
        Jt_kq = mode_coupling_matrix(jacobian_true(psi0[s], V[s], nx, dt, nt))
        J0_s  = born_jacobian(psi0[s], nx, dt, nt)
        Jf_kq = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))

        Jtrue_kq[s]  = np.asarray(Jt_kq)
        born_errs[s] = float(response_error(J0_s,  Jt_kq))
        fno_errs[s]  = float(response_error(Jf_kq, Jt_kq))
        if (s + 1) % 50 == 0:
            print(f"    {s+1:3d}/{N}  Born {born_errs[:s+1].mean():.4f} "
                  f"FNO {fno_errs[:s+1].mean():.4f}  ({time.time()-t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    np.save("results/Jtrue_kq_200.npy",            Jtrue_kq)
    np.save("results/born_errs_200.npy",           born_errs)
    np.save("results/fno_errs_200_f64trained.npy", fno_errs)
    print("[5] cached Jtrue_kq_200 / born_errs_200 / fno_errs_200_f64trained")

    print(f"[5] Born mean {born_errs.mean():.4f} (locked 0.6506) | "
          f"FNO mean {fno_errs.mean():.4f} (3-seed floor 0.587)")
    assert abs(born_errs.mean() - 0.6506) < 0.01, "Born mean drifted — Born machinery changed"
    assert abs(fno_errs.mean()  - 0.587)  < 0.04, "f64-trained seed0 far from floor — check load"
    print("[5] gates passed — cache canonical.")

    r_p, p_p = pearsonr(born_errs, fno_errs)
    r_s, p_s = spearmanr(born_errs, fno_errs)
    print(f"[5] Born vs FNO per-sample:  Pearson {r_p:.3f} (p={p_p:.1e})  "
          f"Spearman {r_s:.3f} (p={p_s:.1e})")
    print("→ CORRELATED (FNO inherits PT blind spots)" if r_p > 0.4
          else "→ DECOUPLED (FNO fails by truncation, not perturbativeness) — 7A headliner")