# scripts/kq_blocks.py — concern 4: (k,q) block decomposition of M16-M12 excess. x64 FIRST.
import jax; jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp
from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt
from src.responses import jacobian_fno, mode_coupling_matrix
from src.solver import evolve

cfg = Config()
nt = cfg.nt
dt = cfg.t_end / cfg.nt

cfg = Config(); nx = cfg.nx
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
N = psi0.shape[0]
Jtrue = np.load("results/Jtrue_kq_200.npy")               # (200,128,128), FNO-independent
assert Jtrue.shape == (N, nx, nx), "J_true cache shape mismatch — STOP"

def fno_kq(ckpt, n_modes):
    m = load_fno_f64_ckpt(ckpt, cfg, n_modes=n_modes)     # n_modes override REQUIRED for M!=16
    out = np.empty((N, nx, nx), dtype=np.complex128)
    for s in range(N):
        out[s] = np.asarray(mode_coupling_matrix(jacobian_fno(m, psi0[s], V[s])))
    return out

def resp_err(Jf):
    return np.array([np.linalg.norm(Jf[s]-Jtrue[s]) / np.linalg.norm(Jtrue[s]) for s in range(N)])

p12, p16 = "results/Jfno_kq_seed0_M12.npy", "results/Jfno_kq_seed0_M16.npy"
J12 = np.load(p12) if os.path.exists(p12) else fno_kq("checkpoints/dino_N2000_seed0_M12_lam0.0_final.eqx", 12)
J16 = np.load(p16) if os.path.exists(p16) else fno_kq("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", 16)
if not os.path.exists(p12): np.save(p12, J12)
if not os.path.exists(p16): np.save(p16, J16)

e12, e16 = resp_err(J12).mean(), resp_err(J16).mean()
print(f"[gate] M12 resp {e12:.4f} (canon ~0.257) | M16 resp {e16:.4f} (canon ~0.587)")
assert abs(e16-0.587) < 0.02 and abs(e12-0.257) < 0.02, "recompute != canonical — STOP"
print("[gate] PASSED — recomputed J(k,q) reproduces canonical response errors")

# --- Block 2: (k,q) block decomposition (concern 4) ---
SHARED_CUT, EXTRA_CUT = 12, 16     # CONFIRM: n_modes=M keeps |k|<M? (see printed extra-idx)
absk = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)   # 0,1,..,64,..,1 (unshifted fft order)
S = absk < SHARED_CUT
X = (absk >= SHARED_CUT) & (absk < EXTRA_CUT)
B = absk >= EXTRA_CUT
print(f"[bands] shared:{int(S.sum())} extra:{int(X.sum())} beyond:{int(B.sum())}")
print(f"[bands] extra-mode indices (verify vs your truncation): {np.where(X)[0]}")

def bn2(A, km, qm):                # summed-over-samples squared Frobenius norm of a block
    return float(np.sum(np.abs(A[:, km][:, :, qm])**2))

def analyze(Jf, tag):
    E = Jf - Jtrue
    Etot = float(np.sum(np.abs(E)**2))
    # completeness across the full {S,X,B}^2 partition
    masks = {"S": S, "X": X, "B": B}
    acc = sum(bn2(E, masks[a], masks[b]) for a in masks for b in masks)
    assert abs(acc - Etot) / Etot < 1e-10, "block partition incomplete — STOP"
    print(f"\n=== {tag}: error-energy by (k=output, q=input) block ===")
    print(f"{'block':8s}{'|E|^2 frac':>12s}{'per-block rel-err':>20s}")
    for name, km, qm in [("SS", S, S), ("SX", S, X), ("XS", X, S), ("XX", X, X)]:
        e2, t2 = bn2(E, km, qm), bn2(Jtrue, km, qm)
        print(f"{name:8s}{e2/Etot:12.4f}{(np.sqrt(e2/t2) if t2>0 else np.nan):20.4f}")
    e_in = sum(bn2(E, masks[a], masks[b]) for a in "SX" for b in "SX")
    print(f"{'2x2 tot':8s}{e_in/Etot:12.4f}   (remainder involves |k|>=16, both models blind there)")

analyze(J12, "M12")
analyze(J16, "M16")

# --- Block 3: full 3x3 (k,q) blocks + marginals + J_true power map (concern 4, decisive) ---
J12 = np.load("results/Jfno_kq_seed0_M12.npy"); J16 = np.load("results/Jfno_kq_seed0_M16.npy")
bands = {"S": S, "X": X, "B": B}; order = ["S", "X", "B"]     # S:<12  X:[12,16)  B:>=16
def bn(A, km, qm): return float(np.sum(np.abs(A[:, km][:, :, qm])**2))

Jt_tot = float(np.sum(np.abs(Jtrue)**2))
print("\n=== J_true power fraction (row=output k, col=input q) ===")
print("    " + "".join(f"{c:>9s}" for c in order))
for a in order: print(f"{a:3s} " + "".join(f"{bn(Jtrue,bands[a],bands[b])/Jt_tot:9.4f}" for b in order))

def table(Jf, tag):
    E = Jf - Jtrue; Et = float(np.sum(np.abs(E)**2))
    print(f"\n=== {tag}: error-energy fraction (row=output k, col=input q) ===")
    print("    " + "".join(f"{c:>9s}" for c in order))
    for a in order: print(f"{a:3s} " + "".join(f"{bn(E,bands[a],bands[b])/Et:9.4f}" for b in order))
    om = {a: sum(bn(E,bands[a],bands[b]) for b in order)/Et for a in order}
    im = {b: sum(bn(E,bands[a],bands[b]) for a in order)/Et for b in order}
    print(f"  output-k marginal (==sec6): S={om['S']:.3f} X={om['X']:.3f} B={om['B']:.3f}  in-band(S+X)={om['S']+om['X']:.3f}")
    print(f"  input-q  marginal (NEW)   : S={im['S']:.3f} X={im['X']:.3f} B={im['B']:.3f}")
table(J12, "M12"); table(J16, "M16")

# --- Block 4: is BB physical or noise²/≈0 ? weight the error where J_true has power ---
# per-sample, per-entry: does |E| track |J_true|, or is it a floor independent of truth?
E16 = J16 - Jtrue
# (a) absolute error energy in BB vs the ABSOLUTE (un-ratioed) size there
bb = (np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int) >= 16)
BBk = bb[:, None] & bb[None, :]
absE_BB   = np.sqrt(np.mean(np.abs(E16[:, BBk])**2))
absJt_BB  = np.sqrt(np.mean(np.abs(Jtrue[:, BBk])**2))
absJf_BB  = np.sqrt(np.mean(np.abs(J16[:, BBk])**2))
print(f"BB block RMS magnitudes:  |J_true|={absJt_BB:.3e}  |J_fno|={absJf_BB:.3e}  |E|={absE_BB:.3e}")
print(f"  -> if |J_fno|,|E| ~ |J_true| (all tiny), BB error is noise on ~0 truth, not physical")
# (b) power-weighted global response error: weight each column q by J_true's power in that column
colpow = np.sum(np.abs(Jtrue)**2, axis=(0,1))               # (nx,) input-mode power under the data
w = colpow / colpow.sum()
num = np.sum(w[None,None,:] * np.abs(E16)**2)
den = np.sum(w[None,None,:] * np.abs(Jtrue)**2)
print(f"\nM16 power-weighted response error = {np.sqrt(num/den):.4f}   (isotropic canonical = 0.587)")
print("  >> if ~0.58: finding SURVIVES, error is where the data lives (concern 5 answered, headline holds)")
print("  >> if <<0.58: isotropic metric was inflated by empty high-q sectors (reframe needed)")

# --- Block 5: three INDEPENDENT diagnostics of the 0.587 -> 0.059 collapse ---
J12 = np.load("results/Jfno_kq_seed0_M12.npy"); J16 = np.load("results/Jfno_kq_seed0_M16.npy")
absk = np.abs(np.fft.fftfreq(nx, d=1/nx)).astype(int)

# RISK 1 — weight by the V-PRIOR envelope (independent of J_true / FNO), not by colpow.
ls = cfg.length_scale if hasattr(cfg, "length_scale") else 0.2   # CONFIRM this matches data-gen
prior = np.exp(-absk * ls); prior /= prior.sum()                 # (nx,) input-mode weight
def pw_err(Jf, w):
    num = np.sum(w[None,None,:]*np.abs(Jf-Jtrue)**2); den = np.sum(w[None,None,:]*np.abs(Jtrue)**2)
    return float(np.sqrt(num/den))
print(f"[R1] prior-weighted  M16={pw_err(J16,prior):.4f}  M12={pw_err(J12,prior):.4f}   (isotropic: 0.587 / 0.246)")

# RISK 3 — same weighting, both M: does 'more modes worse' survive data-honest weighting?
colpow = np.sum(np.abs(Jtrue)**2, axis=(0,1)); wcol = colpow/colpow.sum()
print(f"[R3] colpow-weighted M16={pw_err(J16,wcol):.4f}  M12={pw_err(J12,wcol):.4f}")

# RISK 2 — the honest downstream quantity: JVPs along PRIOR-drawn perturbations (needs models)
from src.responses import jacobian_fno, mode_coupling_matrix   # already imported above; safe
def prior_dV(key):
    kfull = np.fft.fftfreq(nx, d=1/nx)                 # signed freqs
    env = np.exp(-np.abs(kfull) * ls)                  # sampler envelope in k
    cr = jax.random.normal(key, (nx,)); ci = jax.random.normal(jax.random.fold_in(key,1), (nx,))
    coeff = (cr + 1j*ci) * jnp.asarray(np.sqrt(env))   # white * sqrt(envelope), in k-space
    dV = jnp.real(jnp.fft.ifft(coeff)) * nx            # real field with the prior spectrum
    return (dV / jnp.linalg.norm(dV)).astype(jnp.float64)

def jvp_err_prior(ckpt, M, ndir=8, nsamp=40):
    m = load_fno_f64_ckpt(ckpt, cfg, n_modes=M)
    ft = lambda p,v,d: jax.jvp(lambda V_: evolve(p,V_,nx,dt,nt),(v,),(d,))[1]  # needs evolve,dt,nt
    ff = lambda p,v,d: jax.jvp(lambda V_:(lambda o:o[:,0]+1j*o[:,1])(m(p,V_)),(v,),(d,))[1]
    errs=[]
    for s in range(nsamp):
        for j in range(ndir):
            d = prior_dV(jax.random.PRNGKey(1000*s+j))
            a = ft(psi0[s],V[s],d); b = ff(psi0[s],V[s],d)
            errs.append(float(jnp.linalg.norm(b-a)/jnp.linalg.norm(a)))
    return np.mean(errs), np.std(errs)

# GATE A: the drawn dV must actually be low-q (matches the prior), not white
d = prior_dV(jax.random.PRNGKey(0))
hik = np.abs(np.fft.fft(np.asarray(d)))**2
frac_hi = float(hik[np.abs(np.fft.fftfreq(nx,d=1/nx))>=16].sum()/hik.sum())
print(f"[gateA] dV high-|k|(>=16) power fraction = {frac_hi:.4f}  (must be SMALL, ~<0.05)")
# GATE B: for J_TRUE against itself the 'error' is 0; sanity that norms are sane & real
a = jax.jvp(lambda V_: evolve(psi0[0],V_,nx,dt,nt),(V[0],),(d,))[1]
print(f"[gateB] ||J_true.dV|| = {float(jnp.linalg.norm(a)):.3e}  (finite, O(1e-2..1e0)); dtype {a.dtype}")

# NOTE: needs `from src.solver import evolve` and dt,nt,cfg in scope — add if not present
m16 = jvp_err_prior("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", 16)
m12 = jvp_err_prior("checkpoints/dino_N2000_seed0_M12_lam0.0_final.eqx", 12)
print(f"[R2] prior-JVP error  M16={m16[0]:.4f}±{m16[1]:.4f}  M12={m12[0]:.4f}±{m12[1]:.4f}")