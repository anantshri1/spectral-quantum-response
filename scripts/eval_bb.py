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
    SEEDS = [0, 1, 2]                       # 14/20/24 have exactly 3 seeds
    Modes = [2, 4, 8, 12, 14, 16, 20, 24]
    kabs = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)
    bb   = np.ix_(kabs >= 16, kabs >= 16)
    gate = {(0,12): 0.2455, (0,16): 0.5787}

    agg = {}
    for seed in SEEDS:
        for M in Modes:
            ckpt  = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam0.0_final.eqx"
            cache = f"results/Jfno_kq_seed{seed}_M{M}.npy"
            if not os.path.exists(ckpt):
                print(f"[missing] {ckpt}"); continue
            Jf = np.load(cache) if os.path.exists(cache) else jfno_kq_for(ckpt, M)
            if not os.path.exists(cache): np.save(cache, Jf)
            iso = float(np.mean([float(response_error(jnp.asarray(Jf[s]), jnp.asarray(Jtrue_kq[s]))) for s in range(N)]))
            if (seed,M) in gate:
                assert abs(iso-gate[(seed,M)])<0.02, f"seed{seed} M{M} iso {iso:.4f} != {gate[(seed,M)]} — STOP"
            bb_rms = float(np.sqrt((np.abs(Jf[:, bb[0], bb[1]])**2).mean()))
            agg.setdefault(M, {"iso":[],"bb":[]})
            agg[M]["iso"].append(iso); agg[M]["bb"].append(bb_rms)
            print(f"[E3] seed{seed} M{M}: iso={iso:.4f}  BB_RMS={bb_rms:.3e}")

    print("\n=== E3 cross-seed (mean +/- SE) ===")
    print(f"{'M':>3} {'iso':>18} {'|Jfno|_BB RMS':>22}")
    for M in Modes:
        a = agg.get(M);  n = len(a["iso"]) if a else 0
        if not n: continue
        iso, bbv = np.array(a["iso"]), np.array(a["bb"])
        se = lambda x: x.std(ddof=1)/np.sqrt(n) if n>1 else 0.0
        print(f"{M:>3} {iso.mean():>9.4f}+/-{se(iso):.4f} {bbv.mean():>12.3e}+/-{se(bbv):.1e}")