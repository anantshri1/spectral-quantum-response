# scripts/offband.py — 7A Block 1: per-sample band + off-band mass metric
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

from configs.default import Config

cfg = Config(); nx = cfg.nx
kf = np.fft.fftfreq(nx, d=1/nx).astype(int)          # signed int wavenumbers

def offset_grid(nx):
    """d = k - q folded to [-nx/2, nx/2), as a (nx,nx) int array."""
    kk, qq = np.meshgrid(kf, kf, indexing="ij")
    return ((kk - qq + nx//2) % nx) - nx//2

OFFSET = offset_grid(nx)                               # reused everywhere

def band_width_for_sample(J_true_kq, k0, frac=0.95):
    """
    Smallest radius w such that the window |d - k0| <= w captures >= frac of
    |J_true|'s total mass. Self-calibrating per sample (Option B).
    Returns w (int).
    """
    absJ  = np.abs(J_true_kq)
    total = absJ.sum()
    dist  = np.abs(OFFSET - k0)                        # distance from carrier-shifted center
    for w in range(0, nx//2 + 1):
        if absJ[dist <= w].sum() >= frac * total:
            return w
    return nx//2

def offband_mass_fraction(J_kq, k0, w):
    """Fraction of |J_kq| mass OUTSIDE the band |d - k0| <= w."""
    absJ = np.abs(J_kq)
    out  = absJ[np.abs(OFFSET - k0) > w].sum()
    return float(out / absJ.sum())


if __name__ == "__main__":
    # --- Block 1 gate: reproduce Phase-4 sample-0 mass-within-16 = 0.995 ---
    Jtrue = np.load("results/Jtrue_kq_200.npy")        # (200,128,128) cached
    k0_all = np.load("data/test.npz")["p_k0"]
    s = 0; k0 = int(k0_all[s])

    # (a) sanity: reproduce the Phase-4 number — mass of J_true within |d|<=16 (centered d=0, NOT k0)
    absJ = np.abs(Jtrue[s])
    mass16 = absJ[np.abs(OFFSET) <= 16].sum() / absJ.sum()
    print(f"[1-gate] sample {s}: J_true mass within |d|<=16 = {mass16:.3f}  (Phase-4: 0.995)")

    # (b) the actual metric: per-sample band width, and J_true's own off-band mass (should be ~0.05)
    w = band_width_for_sample(Jtrue[s], k0, frac=0.95)
    ob_true = offband_mass_fraction(Jtrue[s], k0, w)
    print(f"[1] sample {s}: k0={k0}, 95%-band radius w={w}, "
          f"J_true off-band mass={ob_true:.3f}  (expect ~0.05 by construction)")

    # --- Block 2a: off-band mass for ONE checkpoint (M16 λ0 seed0), gated ---
    import jax.numpy as jnp
    from tests.measure_floor import load_fno_f64_ckpt
    from src.responses import jacobian_fno, mode_coupling_matrix, response_error
    from scripts.train import load_split

    # reuse the same test load + f64 widen as every prior block
    psi0, V, uT = load_split("data/test.npz", nx=nx)
    psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
    N = psi0.shape[0]

    Jtrue  = np.load("results/Jtrue_kq_200.npy")          # (200,128,128), cached
    k0_all = np.load("data/test.npz")["p_k0"]

    # precompute per-sample bands ONCE (J_true is fixed; bands don't depend on the model)
    bands = np.array([band_width_for_sample(Jtrue[s], int(k0_all[s]), 0.95) for s in range(N)])
    print(f"[2a] band widths: min {bands.min()}, median {int(np.median(bands))}, max {bands.max()}")

    def offband_for_ckpt(path, m):
        model = load_fno_f64_ckpt(path, cfg, n_modes=m)
        ob = np.empty(N)
        for s in range(N):
            Jf = mode_coupling_matrix(jacobian_fno(model, psi0[s], V[s]))
            ob[s] = offband_mass_fraction(Jf, int(k0_all[s]), bands[s])
        return ob

    ob = offband_for_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", 16)
    print(f"[2a] M16 λ0 seed0: off-band mass = {ob.mean():.3f} ± {ob.std()/np.sqrt(N):.3f}")
    print(f"     J_true off-band (reference) ≈ 0.05")
    print(f"     FNO off-band / J_true off-band ratio = {ob.mean()/0.05:.1f}×")

    # --- Block 2b: off-band mass across the f64 grid, 3-seed. SAVES incrementally. ---
    import os
    os.makedirs("results", exist_ok=True)

    seeds   = [0, 1, 2]
    lam_3seed = [0.0, 0.1, 1.0, 10.0]        # 3-seed λ's, both M
    lam_extra = [0.3, 3.0]                    # seed0-only, M16 only (curve shape)

    def ckpt_path(seed, m, lam):
        return f"checkpoints/dino_N2000_seed{seed}_M{m}_lam{lam}_final.eqx"

    # build the job list: (seed, M, lam) tuples
    jobs = []
    for m in [12, 16]:
        for lam in lam_3seed:
            for seed in seeds:
                jobs.append((seed, m, lam))
    for lam in lam_extra:                     # M16 seed0 extras
        jobs.append((0, 16, lam))

    results = {}                              # (seed,M,lam) -> mean off-band
    ob_full = {}                              # (seed,M,lam) -> full 200-vector
    import time; t0 = time.time()
    for i, (seed, m, lam) in enumerate(jobs):
        path = ckpt_path(seed, m, lam)
        if not os.path.exists(path):
            print(f"  [{i+1}/{len(jobs)}] SKIPPED (not on disk): {path}")
            continue
        ob = offband_for_ckpt(path, m)        # reuses Block-2a fn + precomputed bands
        results[(seed, m, lam)] = float(ob.mean())
        ob_full[(seed, m, lam)] = ob
        # incremental save every checkpoint — disconnect-proof
        np.savez("results/offband_grid.npz",
                 keys=np.array(list(results.keys())),
                 means=np.array(list(results.values())),
                 **{f"ob_{s}_{mm}_{l}": v for (s, mm, l), v in ob_full.items()})
        print(f"  [{i+1}/{len(jobs)}] seed{seed} M{m} λ{lam}: "
              f"off-band {ob.mean():.3f}  ({time.time()-t0:.0f}s)")

    print(f"\n[2b] saved results/offband_grid.npz  ({len(results)} checkpoints)")

    # --- sign-flip table: 3-seed mean ± spread, M12 vs M16, per λ ---
    print("\n[2b] OFF-BAND MASS — 3-seed mean (spread), M12 vs M16:")
    print("  λ     M12            M16            flip?")
    for lam in lam_3seed:
        v12 = [results[(s,12,lam)] for s in seeds if (s,12,lam) in results]
        v16 = [results[(s,16,lam)] for s in seeds if (s,16,lam) in results]
        if not v12 or not v16: continue
        m12, s12 = np.mean(v12), np.std(v12)
        m16, s16 = np.mean(v16), np.std(v16)
        # "flip" = does M16 leak MORE (liability) or LESS (asset) than M12?
        rel = "M16>M12 (liability)" if m16 > m12 else "M16<M12 (asset)"
        overlap = abs(m16-m12) < (s12+s16)   # crude non-overlap check
        flag = "  ⚠overlap" if overlap else ""
        print(f"  {lam:<4}  {m12:.3f}({s12:.3f})   {m16:.3f}({s16:.3f})   {rel}{flag}")