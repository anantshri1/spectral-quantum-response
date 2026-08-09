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

    # --- Block 4a: Born vs J_true, 3-sample smoke test ---
    from src.responses import jacobian_true, response_error

    V = V.astype(jnp.float64)     # J_true needs f64 V (psi0 already complex128)

    print("\n[4a] Born vs J_true, first 3 samples:")
    for s in range(3):
        J0_s   = born_jacobian(psi0[s], nx, dt, nt)
        Jt_pos = jacobian_true(psi0[s], V[s], nx, dt, nt)
        Jt_kq  = mode_coupling_matrix(Jt_pos)
        err    = float(response_error(J0_s, Jt_kq))
        print(f"    sample {s}: Born-vs-Jtrue rel-Frob = {err:.4f}")

    # --- Block 4b: Born vs J_true across all test samples ---
    import os, time
    from src.responses import jacobian_true, response_error   # (no-op if 4a ran)

    N = psi0.shape[0]                      # 200
    FNO_FLOOR = 0.587                      # forward-only FNO response error (3-seed)

    born_errs = np.empty(N)
    t0 = time.time()
    for s in range(N):
        J0_s   = born_jacobian(psi0[s], nx, dt, nt)
        Jt_kq  = mode_coupling_matrix(jacobian_true(psi0[s], V[s], nx, dt, nt))
        born_errs[s] = float(response_error(J0_s, Jt_kq))
        if (s + 1) % 25 == 0:
            print(f"    {s+1:3d}/{N}   running mean = {born_errs[:s+1].mean():.4f}"
                  f"   ({time.time()-t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    np.save("results/born_errs_200.npy", born_errs)      # <-- SAVE (never recompute J_true)
    print(f"[4b] saved results/born_errs_200.npy")

    # --- summary: the number that picks the headline branch ---
    m, sd = born_errs.mean(), born_errs.std()
    print(f"\n[4b] Born vs J_true over {N} samples:")
    print(f"     mean {m:.4f}  std {sd:.4f}  median {np.median(born_errs):.4f}")
    print(f"     min  {born_errs.min():.4f}  max {born_errs.max():.4f}")
    print(f"     FNO forward-only floor = {FNO_FLOOR}")
    frac_born_better = float((born_errs < FNO_FLOOR).mean())
    print(f"     Born beats FNO on {frac_born_better*100:.0f}% of samples")
    print(f"     mean Born {'<' if m < FNO_FLOOR else '>'} FNO floor "
          f"→ {'FNO worse than 1st-order PT' if m < FNO_FLOOR else 'response non-perturbative'} branch")

    # --- Block 5a: canonical cache smoke test (3 samples) ---
    from scripts.compute_response import load_fno_x64
    from src.responses import jacobian_true, jacobian_fno, response_error

    model = load_fno_x64("checkpoints/fno_N2000_seed0_final.eqx", cfg)

    print("\n[5a] smoke: J_true / Born / FNO, first 3 samples:")
    for s in range(3):
        Jt_kq  = mode_coupling_matrix(jacobian_true(psi0[s], V[s], nx, dt, nt))
        J0_s   = born_jacobian(psi0[s], nx, dt, nt)
        Jf_kq  = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))
        be = float(response_error(J0_s, Jt_kq))
        fe = float(response_error(Jf_kq, Jt_kq))
        print(f"    s{s}: Born {be:.4f} | FNO {fe:.4f}")

    # --- 5a-diag: is J_fno actually f64, and does it match compute_response's own path? ---
    s = 2
    # (1) dtype of the FNO forward + its Jacobian
    out_s2 = model(psi0[s], V[s])
    Jf_s2  = jacobian_fno(model, psi0[s], V[s])
    print(f"[diag] FNO forward dtype: {out_s2.dtype}   (expect float64)")
    print(f"[diag] J_fno dtype: {Jf_s2.dtype}          (expect complex128)")

    # (2) does model here reproduce the locked test rel-L2? (same check compute_response gates on)
    from scripts.train import eval_step
    rel, nv = eval_step(model, psi0, V, uT, nx)
    print(f"[diag] FNO test rel-L2: {float(rel):.4f}   (locked: 0.0169)")

    # (3) sanity: FNO error on s2 in BOTH bases must agree (rotation is unitary)
    Jt_pos_s2 = jacobian_true(psi0[s], V[s], nx, dt, nt)
    Jf_pos_s2 = jacobian_fno(model, psi0[s], V[s])
    e_pos = float(response_error(Jf_pos_s2, Jt_pos_s2))
    e_kq  = float(response_error(mode_coupling_matrix(Jf_pos_s2), mode_coupling_matrix(Jt_pos_s2)))
    print(f"[diag] FNO s2 err  pos {e_pos:.4f} | kq {e_kq:.4f}   (must match)")

    # --- Block 5b: full 200-sample canonical cache. RUN ONLY AFTER 5a PASSES. ---
    import os, time

    N = psi0.shape[0]
    Jtrue_kq = np.empty((N, nx, nx), dtype=np.complex128)
    born_errs = np.empty(N)
    fno_errs  = np.empty(N)

    t0 = time.time()
    for s in range(N):
        Jt_kq = mode_coupling_matrix(jacobian_true(psi0[s], V[s], nx, dt, nt))
        J0_s  = born_jacobian(psi0[s], nx, dt, nt)
        Jf_kq = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))

        Jtrue_kq[s]  = np.asarray(Jt_kq)
        born_errs[s] = float(response_error(J0_s, Jt_kq))
        fno_errs[s]  = float(response_error(Jf_kq, Jt_kq))
        if (s + 1) % 25 == 0:
            print(f"    {s+1:3d}/{N}  Born {born_errs[:s+1].mean():.4f} "
                  f"FNO {fno_errs[:s+1].mean():.4f}  ({time.time()-t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    np.save("results/Jtrue_kq_200.npy", Jtrue_kq)     # 52 MB — the reusable cache
    np.save("results/born_errs_200.npy", born_errs)   # overwrites Block-4b (identical)
    np.save("results/fno_errs_200.npy",  fno_errs)
    print(f"[5b] cached Jtrue_kq_200 / born_errs_200 / fno_errs_200 to results/")

    # --- consistency gate: fno_errs must reproduce the locked 0.587 ---
    print(f"\n[5b] Born mean {born_errs.mean():.4f} (Block4b: 0.6506)")
    print(f"[5b] FNO  mean {fno_errs.mean():.4f}  (locked floor: 0.587)")
    # --- 5b-diag-2: is the 0.511 vs 0.587 gap a per-sample offset or a scale/convention? ---
    import numpy as np
    print(f"this run fno_errs: mean {fno_errs.mean():.4f} std {fno_errs.std():.4f} "
          f"min {fno_errs.min():.4f} max {fno_errs.max():.4f}")

    # (1) does ANY saved phase-5/6 array hold per-sample FNO response errors? scan results/
    import glob, os
    for p in sorted(glob.glob("results/*.npy")):
        a = np.load(p, allow_pickle=True)
        tag = ""
        if a.ndim == 1 and a.shape[0] == 200 and np.all(np.isfinite(a)) and a.max() < 3:
            tag = f"  <-- 200-vector, mean {a.mean():.4f}"
        print(f"  {p}: shape {a.shape}, dtype {a.dtype}{tag}")

    # (2) the smoking gun: is 0.5113 * const = 0.587?  (convention/scale) or additive?
    print(f"\n  ratio 0.587 / {fno_errs.mean():.4f} = {0.587/fno_errs.mean():.4f}")
    print(f"  (≈1.148 → ~15% scale; if this equals a clean factor it's a norm/basis convention)")

    # (3) recompute ONE sample's FNO error with explicit renormalization of psi_T,
    #     to test H-C (renorm convention) directly on s0
    s = 0
    def f_raw(V_):
        o = model(psi0[s], V_); return o[:,0] + 1j*o[:,1]
    def f_norm(V_):
        o = model(psi0[s], V_); c = o[:,0] + 1j*o[:,1]
        return c / jnp.sqrt(jnp.sum(jnp.abs(c)**2))          # renorm before differentiating
    Jf_raw  = jax.jacfwd(f_raw)(V[s])
    Jf_norm = jax.jacfwd(f_norm)(V[s])
    Jt = mode_coupling_matrix(jacobian_true(psi0[s], V[s], nx, dt, nt))
    e_raw  = float(response_error(mode_coupling_matrix(Jf_raw),  Jt))
    e_norm = float(response_error(mode_coupling_matrix(Jf_norm), Jt))
    print(f"\n  s0 FNO err  raw {e_raw:.4f} | renormed {e_norm:.4f}   (which matches your notes' s0?)")
    
    assert abs(fno_errs.mean() - 0.587) < 0.02, "FNO mean drifted from 0.587 — STOP, load/machinery mismatch"
    print("[5b] GATE PASSED — FNO reproduces locked floor; cache is canonical.")

    # --- the scatter's headline number: does Born predict FNO per-sample? ---
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(born_errs, fno_errs)
    r_s, _ = spearmanr(born_errs, fno_errs)
    print(f"\n[5b] per-sample corr  Pearson {r_p:.3f}  Spearman {r_s:.3f}")
    print(f"     {'CORRELATED → FNO inherits Born blind spots' if r_p > 0.4 else 'DECOUPLED → FNO fails for its own (spectral) reasons'}")