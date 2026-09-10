# scripts/eval_jvp_prior.py
# E1 — prior-drawn JVP response error (the physically-weighted pivot).
# Block 1a: preamble + validated prior_dV + Gate A (prior spectrum sanity).
import jax
jax.config.update("jax_enable_x64", True)          # MUST be first: solver+loader+JVP are all f64
import numpy as np, jax.numpy as jnp

from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt   # the correct f64 loader (NOT load_fno_x64)
from src.solver import evolve                        # evolve(psi0, V, nx, dt, nt)

cfg = Config()
nx  = cfg.nx                    # 128
dt  = cfg.t_end / cfg.nt        # 0.5/1000 = 5e-4
nt  = cfg.nt                    # 1000
ls  = cfg.v_length_scale        # 0.2   (attribute is v_length_scale, not length_scale)

# Base points = in-distribution test potentials. load_split pins psi0->complex64,
# V->float32 (the f32 forward regime); widen both or the JVP runs mixed precision.
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128)
V    = V.astype(jnp.float64)
N    = psi0.shape[0]
print(f"[load] N={N}  psi0={psi0.dtype}  V={V.dtype}  dt={dt}  nt={nt}  ls={ls}")

# --- validated prior_dV (verbatim; do not modify) -------------------------
def prior_dV(key,ls=cfg.v_length_scale):
    kfull = np.fft.fftfreq(nx, d=1/nx)               # signed: 0,1,..,63,-64,..,-1
    env   = np.exp(-np.abs(kfull) * ls)              # exp(-|k|ls)
    cr = jax.random.normal(key, (nx,))
    ci = jax.random.normal(jax.random.fold_in(key, 1), (nx,))
    coeff = (cr + 1j*ci) * jnp.asarray(np.sqrt(env)) # amplitude ~ exp(-|k|ls/2)
    dV = jnp.real(jnp.fft.ifft(coeff)) * nx
    return (dV / jnp.linalg.norm(dV)).astype(jnp.float64)

# --- validated jvp_err_prior (verbatim; do not modify) --------------------
def jvp_err_prior(ckpt, M, ndir=8, nsamp=40):
    m = load_fno_f64_ckpt(ckpt, cfg, n_modes=M)
    ft = lambda p,v,d: jax.jvp(lambda V_: evolve(p,V_,nx,dt,nt),(v,),(d,))[1]
    ff = lambda p,v,d: jax.jvp(lambda V_:(lambda o:o[:,0]+1j*o[:,1])(m(p,V_)),(v,),(d,))[1]
    errs=[]
    for s in range(nsamp):
        for j in range(ndir):
            d = prior_dV(jax.random.PRNGKey(1000*s+j))
            a = ft(psi0[s],V[s],d); b = ff(psi0[s],V[s],d)
            errs.append(float(jnp.linalg.norm(b-a)/jnp.linalg.norm(a)))
    return np.mean(errs), np.std(errs)

# --- GATE A: where does prior_dV put its power? --------------------------
if __name__ == "__main__":
    kabs = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)   # same mask convention as SS/X/BB
    hi   = kabs >= 16

    # (i) designed spectrum, analytic: E|coeff|^2 ∝ env = exp(-|k|ls)
    env_full = np.exp(-kabs * ls)
    print(f"[Gate A] coeff-space  power frac |k|>=16 = {env_full[hi].sum()/env_full.sum():.4f}")

    # (ii) realised field, sampled: real(ifft) folds the spectrum; this is what enters evolve
    fr = []
    for i in range(200):
        d = np.asarray(prior_dV(jax.random.PRNGKey(9000 + i)))
        P = np.abs(np.fft.fft(d))**2
        fr.append(P[hi].sum() / P.sum())
    fr = np.array(fr)
    print(f"[Gate A] realised     power frac |k|>=16 = {fr.mean():.4f} +/- {fr.std():.4f}   (doc target ~0.029)")

    import os
    os.makedirs("results", exist_ok=True)
    seed, M, lam = 0, 16, 0.0
    ckpt = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam{lam}_final.eqx"
    out  = f"results/jvp_prior_seed{seed}_M{M}_lam{lam}.npz"
    if os.path.exists(out):
        z = np.load(out); mean, std = float(z["mean"]), float(z["std"]); print(f"[cache] {out}")
    else:
        mean, std = jvp_err_prior(ckpt, M=M)
        np.savez(out, mean=mean, std=std, M=M, lam=lam, seed=seed, ndir=8, nsamp=40)
    print(f"[E1 GATE] M{M} lam{lam} seed{seed}: prior-JVP = {mean:.4f} +/- {std:.4f}   (target 0.091 +/- 0.044)")

    seed, M = 0, 16
    for lam in [0.0, 0.1, 10.0]:
        ckpt = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam{lam}_final.eqx"
        out  = f"results/jvp_prior_seed{seed}_M{M}_lam{lam}.npz"
        if os.path.exists(out):
            z = np.load(out); mean, std = float(z["mean"]), float(z["std"]); tag = "cache"
        else:
            mean, std = jvp_err_prior(ckpt, M=M); tag = "run"
            np.savez(out, mean=mean, std=std, M=M, lam=lam, seed=seed, ndir=8, nsamp=40)
        print(f"[E1] M{M} lam{lam} seed{seed}: prior-JVP = {mean:.4f} +/- {std:.4f}  ({tag})")

    M = 16
    agg = {0.0: [], 0.1: [], 10.0: []}
    for seed in [0, 1, 2, 3]:
        for lam in [0.0, 0.1, 10.0]:
            ckpt = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam{lam}_final.eqx"
            out  = f"results/jvp_prior_seed{seed}_M{M}_lam{lam}.npz"
            if os.path.exists(out):
                z = np.load(out); mean, std = float(z["mean"]), float(z["std"]); tag = "cache"
            else:
                mean, std = jvp_err_prior(ckpt, M=M); tag = "run"
                np.savez(out, mean=mean, std=std, M=M, lam=lam, seed=seed, ndir=8, nsamp=40)
            agg[lam].append(mean)
            print(f"[E1] M{M} lam{lam} seed{seed}: prior-JVP = {mean:.4f} +/- {std:.4f}  ({tag})")
    print("\n[E1 4-seed] M16 prior-JVP, cross-seed mean +/- SE:")
    for lam in [0.0, 0.1, 10.0]:
        a = np.array(agg[lam]); se = a.std(ddof=1)/np.sqrt(len(a))
        print(f"  lam{lam:>4}: {a.mean():.4f} +/- {se:.4f}   seeds={np.round(a,4).tolist()}")

    Modes = [2,4,8,12,14,16,20,24]
    lam = 0.0
    agg = {0.0: []}
    for seed in [0, 1, 2]:
        for M in Modes:
            ckpt = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam{lam}_final.eqx"
            out  = f"results/jvp_prior_seed{seed}_M{M}_lam{lam}.npz"
            if os.path.exists(out):
                z = np.load(out); mean, std = float(z["mean"]), float(z["std"]); tag = "cache"
            else:
                mean, std = jvp_err_prior(ckpt, M=M); tag = "run"
                np.savez(out, mean=mean, std=std, M=M, lam=lam, seed=seed, ndir=8, nsamp=40)
            agg[lam].append(mean)
            print(f"[E1] M{M} lam{lam} seed{seed}: prior-JVP = {mean:.4f} +/- {std:.4f}  ({tag})")

    

    