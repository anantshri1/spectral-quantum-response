# scripts/eval_bb.py
# E3 — BB-sector hallucination amplitude vs M (mechanism) + authoritative isotropic column.
# Block 1: Jfno_kq producer + reproduce-known-number gate (seed0 M12/M16).
import jax
jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp

from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt
from src.responses import mode_coupling_matrix, jacobian_fno, response_error

cfg = Config(); nx = cfg.nx
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
N = psi0.shape[0]

Jtrue_kq = np.load("results/Jtrue_kq_200.npy")            # (N,nx,nx) truth — NEVER recompute
assert Jtrue_kq.shape == (N, nx, nx), "Jtrue_kq shape mismatch — STOP"

def jfno_kq_for(ckpt, M, chunk=25):
    """(N,nx,nx) complex FNO Jacobian in (k,q). n_modes=M REQUIRED for M!=16 (shape-mismatch bug otherwise)."""
    m = load_fno_f64_ckpt(ckpt, cfg, n_modes=M)
    one = lambda p, v: mode_coupling_matrix(jacobian_fno(m, p, v))
    outs = [jax.vmap(one)(psi0[i:i+chunk], V[i:i+chunk]) for i in range(0, N, chunk)]
    return np.asarray(jnp.concatenate(outs, axis=0))

if __name__ == "__main__":
    seed = 0
    Modes = [2, 4, 8, 12, 14, 16, 20, 24]
    kabs = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)
    bb   = np.ix_(kabs >= 16, kabs >= 16)                 # BBxBB block — the empty sector
    gate = {12: 0.2455, 16: 0.5787}                       # seed0 isotropic, reproduce-or-STOP

    jt_bb_rms = float(np.sqrt((np.abs(Jtrue_kq[:, bb[0], bb[1]])**2).mean()))
    print(f"[ref] |J_true| BB RMS = {jt_bb_rms:.3e}   (physical floor)\n")
    print(f"{'M':>3} {'iso':>8} {'priorJVP':>9} {'gap':>8} {'|Jfno|_BB RMS':>14}")

    rows = []
    for M in Modes:
        ckpt  = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam0.0_final.eqx"
        cache = f"results/Jfno_kq_seed{seed}_M{M}.npy"
        Jf = np.load(cache) if os.path.exists(cache) else jfno_kq_for(ckpt, M)
        if not os.path.exists(cache): np.save(cache, Jf)

        iso = float(np.mean([float(response_error(jnp.asarray(Jf[s]), jnp.asarray(Jtrue_kq[s]))) for s in range(N)]))
        if M in gate:
            assert abs(iso - gate[M]) < 0.02, f"M{M} iso {iso:.4f} drifted from {gate[M]} — STOP"

        bb_rms = float(np.sqrt((np.abs(Jf[:, bb[0], bb[1]])**2).mean()))
        pj  = float(np.load(f"results/jvp_prior_seed{seed}_M{M}_lam0.0.npz")["mean"])
        rows.append((M, iso, pj, iso - pj, bb_rms))
        print(f"{M:>3} {iso:>8.4f} {pj:>9.4f} {iso-pj:>8.4f} {bb_rms:>14.3e}")

    bbs  = [r[4] for r in rows]
    mono = all(bbs[i] <= bbs[i+1] + 1e-9 for i in range(len(bbs)-1))
    print(f"\n[E3] BB RMS monotone non-decreasing across all M: {mono}")
    # the load-bearing claim is the M>=12 regime (where added modes land in BB):
    hi = [r for r in rows if r[0] >= 12]
    mono_hi = all(hi[i][4] <= hi[i+1][4] + 1e-9 for i in range(len(hi)-1))
    print(f"[E3] BB RMS monotone for M>=12 (the prediction): {mono_hi}")