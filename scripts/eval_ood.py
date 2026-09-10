# scripts/eval_ood.py — E4: response error under distribution shift (roughness ls).
# Block 1: OOD V builder + prior_dV(ls) + the ls=0.20 double-anchor gate.
import jax
jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp

from configs.default import Config
from tests.measure_floor import load_fno_f64_ckpt
from src.solver import evolve
from src.responses import mode_coupling_matrix, jacobian_fno, response_error
from src.data_gen import ood_potential, realize_potential          # verbatim shift builder
from scripts.eval_jvp_prior import prior_dV                         # now ls-parametrised

cfg = Config(); nx = cfg.nx
dt  = cfg.t_end / cfg.nt; nt = cfg.nt
N   = 200
LS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# base points EXACTLY as the OOD suite loads them (psi0 as-is, unnormalised; deterministic V)
d = np.load("data/test.npz")
psi0   = jnp.asarray(d["u0_re_128"][:N] + 1j * d["u0_im_128"][:N])
coeffs = jnp.asarray(d["p_coeffs"][:N])
V128   = jnp.asarray(d["V_128"][:N])

def Vood(ls):
    return jnp.stack([ood_potential(coeffs[i], V128[i], nx, ls) for i in range(N)])

def jfno_kq(model, V, chunk=25):
    one = lambda p, v: mode_coupling_matrix(jacobian_fno(model, p, v))
    return np.asarray(jnp.concatenate(
        [jax.vmap(one)(psi0[i:i+chunk], V[i:i+chunk]) for i in range(0, N, chunk)], axis=0))

def iso_err(Jf, Jt):
    return float(np.mean([float(response_error(jnp.asarray(Jf[s]), jnp.asarray(Jt[s]))) for s in range(N)]))

def prior_jvp_err(model, V, ls, ndir=8, nsamp=40):     # same reduction as E1, but V=Vood, delta from ls-prior
    ft = lambda p,v,dd: jax.jvp(lambda V_: evolve(p,V_,nx,dt,nt), (v,), (dd,))[1]
    ff = lambda p,v,dd: jax.jvp(lambda V_:(lambda o:o[:,0]+1j*o[:,1])(model(p,V_)), (v,), (dd,))[1]
    errs=[]
    for s in range(nsamp):
        for j in range(ndir):
            dd = prior_dV(jax.random.PRNGKey(1000*s+j), ls=ls)
            a = ft(psi0[s],V[s],dd); b = ff(psi0[s],V[s],dd)
            errs.append(float(jnp.linalg.norm(b-a)/jnp.linalg.norm(a)))
    return float(np.mean(errs)), float(np.std(errs))

if __name__ == "__main__":
    os.makedirs("results/ood", exist_ok=True)
    seed = 0
    configs = [("M16", 16, 0.0), ("M16", 16, 10.0), ("M12", 12, 0.0)]   # (label, M, lam)
    Jt = {ls: np.load(f"results/ood/Jtrue_kq_ls{ls:.2f}.npy") for ls in LS_GRID}

    for label, M, lam in configs:
        ck = f"checkpoints/dino_N2000_seed{seed}_M{M}_lam{float(lam)}_final.eqx"
        m  = load_fno_f64_ckpt(ck, cfg, n_modes=M)
        print(f"\n=== {label} lam{lam} ===")
        print(f"{'ls':>5} {'iso':>8} {'priorJVP(ls)':>13} {'priorJVP(0.20)':>15}")
        for ls in LS_GRID:
            out = f"results/ood/resperr_seed{seed}_{label}_lam{float(lam)}_ls{ls:.2f}.npz"
            if os.path.exists(out):
                z = np.load(out); iso, pj, pj0 = float(z["iso"]), float(z["pj"]), float(z["pj0"])
            else:
                V   = Vood(ls)
                Jf  = jfno_kq(m, V)
                iso = iso_err(Jf, Jt[ls])
                pj,  _ = prior_jvp_err(m, V, ls)          # delta ~ ls-prior (physical shift)
                pj0, _ = prior_jvp_err(m, V, 0.20)        # delta ~ training prior (sensitivity)
                np.savez(out, iso=iso, pj=pj, pj0=pj0, ls=ls, M=M, lam=lam, seed=seed)
            # gate: ls=0.20 must reproduce in-dist for the M16 lam0 row
            if label=="M16" and lam==0.0 and abs(ls-0.20)<1e-12:
                assert abs(iso-0.5787)<0.02 and abs(pj-0.0909)<0.02, "ls=0.20 drift — STOP"
            print(f"{ls:>5.2f} {iso:>8.4f} {pj:>13.4f} {pj0:>15.4f}")