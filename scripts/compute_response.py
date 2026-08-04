# scripts/compute_response.py  — Phase 4 driver (grown block by block)

# --- x64 MUST be enabled before ANY jnp work, including model-internal arrays ---
# This is a global flag. Setting it here, at import time, before we touch jnp,
# is what makes FNO.__call__'s internal grid=linspace come out float64 and
# promote the whole forward pass. That promotion is the point, not a leak.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from configs.default import Config
from src.fno import FNO
from src.responses import jacobian_true, fd_check, mode_coupling_matrix, jacobian_fno, response_error
from src.solver import evolve

def load_fno_x64(path: str, cfg: Config) -> FNO:
    """
    Load a float32-trained FNO checkpoint under x64-on, returning a float64 model.

    The pin-then-cast dance avoids the deserialise dtype trap: under x64-on,
    FNO(...) builds template leaves as float64, but the file holds float32 bytes.
    Reading 4-byte data into 8-byte slots is the silent-corruption bug. So:
      1. build template (float64 leaves under x64),
      2. PIN inexact leaves down to float32 to MATCH the file exactly,
      3. deserialise (dtypes now agree — clean read),
      4. cast up to float64 (lossless; Block 0 proved this).
    """
    template = FNO(d_v=cfg.n_channels, n_modes=cfg.n_modes,
                   n_blocks=cfg.n_fno_blocks, key=jax.random.PRNGKey(0))

    # step 2: pin template inexact leaves to float32 to match the file
    arrs, static = eqx.partition(template, eqx.is_inexact_array)
    arrs = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), arrs)
    template = eqx.combine(arrs, static)

    # step 3: deserialise (float32 template <- float32 file: clean)
    model = eqx.tree_deserialise_leaves(path, template)

    # step 4: cast up to float64 (lossless widening)
    arrs, static = eqx.partition(model, eqx.is_inexact_array)
    arrs = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), arrs)
    return eqx.combine(arrs, static)


# --- GATE: prove the loaded float64 model is the same function as Phase 3 ---
if __name__ == "__main__":
    from scripts.train import load_split, eval_step

    cfg = Config()
    CKPT = "checkpoints/fno_N2000_seed0_final.eqx"

    model = load_fno_x64(CKPT, cfg)

    psi0, V, uT = load_split("data/test.npz", nx=cfg.nx)
    # --- widen to x64: load_split pins to complex64/float32 for the Phase-3 float32
    # training regime; the float64 solver needs complex128/float64 carries or lax.scan
    # rejects the dtype mismatch. Lossless widening (Block 0 proved bit-identical). ---
    psi0 = psi0.astype(jnp.complex128)
    V = V.astype(jnp.float64)

    print("psi0 dtype:", psi0.dtype, "| V dtype:", V.dtype)   # expect complex128 | float64

    out = model(psi0[0], V[0])
    print("forward dtype (x64 on):", out.dtype)          # expect float64

    rel, nv = eval_step(model, psi0, V, uT, cfg.nx)
    print(f"test rel-L2 : {float(rel):.4f}   (locked: 0.0169)")
    print(f"norm-viol   : {float(nv):.4f}   (locked: 0.0057)")
    print("output finite:", bool(jnp.all(jnp.isfinite(out))))

    # gate assertion — the number must survive the arithmetic change
    assert abs(float(rel) - 0.0169) < 1e-3, "rel-L2 drifted under x64 — STOP, do not proceed"
    print("\nGATE PASSED — float64 model matches Phase 3. Precision boundary pinned.")

     # solver params from cfg (verified signatures: evolve(psi0, V, nx, dt, nt))
    dt = cfg.t_end / cfg.nt
    nx, nt = cfg.nx, cfg.nt

    # use the first TEST sample's psi0 and V as the linearization point
    psi0_0, V_0 = psi0[0], V[0]

    # --- Block 1 gate: autodiff vs finite-difference in a random direction ---
    rel_err, jvp_dir, fd_dir = fd_check(psi0_0, V_0, nx, dt, nt)
    print(f"\n[Block 1] FD cross-check rel-err: {float(rel_err):.2e}   (expect < 1e-6)")
    assert float(rel_err) < 1e-4, "J_true disagrees with FD — STOP, map is miswired"
    print("[Block 1] J_true validated against finite difference.")

    # --- Block 2a: full position-space J_true (128 JVPs via jacfwd) ---
    J_pos = jacobian_true(psi0_0, V_0, nx, dt, nt)
    print(f"\n[Block 2a] J_pos shape: {J_pos.shape}, dtype: {J_pos.dtype}")   # (128,128) complex128
    print(f"[Block 2a] J_pos finite: {bool(jnp.all(jnp.isfinite(J_pos)))}")

    # gate: column q of the full matrix must equal the single-JVP in direction e_q.
    # reuse the validated JVP primitive as ground truth for ONE column.
    q_test = 40
    e_q = jnp.zeros(nx, dtype=V_0.dtype).at[q_test].set(1.0)
    def f(V_): return evolve(psi0_0, V_, nx, dt, nt)
    _, col_jvp = jax.jvp(f, (V_0,), (e_q,))
    col_err = jnp.linalg.norm(J_pos[:, q_test] - col_jvp) / jnp.linalg.norm(col_jvp)
    print(f"[Block 2a] column-{q_test} vs JVP rel-err: {float(col_err):.2e}   (expect < 1e-10)")
    assert float(col_err) < 1e-8, "jacfwd column disagrees with direct JVP — STOP"

    # --- rotate to Fourier space ---
    J_kq = mode_coupling_matrix(J_pos)
    print(f"[Block 2a] J_kq shape: {J_kq.shape}, dtype: {J_kq.dtype}")        # (128,128) complex128

    # --- Block 2b: is the mode-coupling structure physically sane? ---
    absJ = jnp.abs(J_kq)

    # (1) diagonal-dominance: energy should concentrate near k-q = 0.
    # roll each row so k-q=0 lands at a fixed column, then average |J| over rows
    # as a function of the OFFSET d = k - q. Peak at d=0, decay with |d|.
    k_idx = jnp.fft.fftfreq(nx, d=1/nx).astype(int)      # integer wavenumbers, fft order
    # build offset profile: mean |J(k,q)| binned by (k-q) mod nx
    kk, qq = jnp.meshgrid(k_idx, k_idx, indexing="ij")
    offset = ((kk - qq + nx//2) % nx) - nx//2            # signed k-q in [-nx/2, nx/2)
    # profile[d] = mean |J| over all (k,q) with k-q == d
    order = jnp.argsort(offset.ravel())
    off_sorted = offset.ravel()[order]
    absJ_sorted = absJ.ravel()[order]
    d_vals = jnp.unique(off_sorted)
    profile = jnp.array([jnp.mean(absJ_sorted[off_sorted == d]) for d in d_vals])

    peak_d = int(d_vals[jnp.argmax(profile)])
    # fraction of total |J| mass within |k-q| <= 16 (the FNO's mode budget)
    mass_within_cutoff = float(jnp.sum(absJ[jnp.abs(offset) <= 16]) / jnp.sum(absJ))

    k0_0 = int(np.load("data/test.npz")["p_k0"][0])   # carrier momentum of sample 0
    print(f"[Block 2b] offset profile peak at k-q = {peak_d}   (expect k0 = {k0_0})")
    assert peak_d == k0_0, "response band not centered on carrier momentum — convention drift"
    print(f"[Block 2b] |J| mass within |k-q|<=16: {mass_within_cutoff:.3f}")
    print(f"[Block 2b] |J| max: {float(absJ.max()):.3e}, "
          f"mean: {float(absJ.mean()):.3e}, "
          f"ratio: {float(absJ.max()/absJ.mean()):.1f}")

    d = np.load("data/test.npz")
    k0_all = d["p_k0"]                        # (N,) carrier momenta
    print("sample 0 k0:", int(k0_all[0]))    # <-- is it 4?

    # (2) save the matrix for plotting + Block 3 comparison (expensive to recompute)
    #import os
    #os.makedirs("results", exist_ok=True)
    #np.save("results/J_true_kq_sample0.npy", np.asarray(J_kq))
    #np.save("results/J_true_pos_sample0.npy", np.asarray(J_pos))
    #print("[Block 2b] saved J_true (kq + pos) to results/")

    # --- Block 3: J_FNO on the SAME sample 0, same machinery ---
    J_fno_pos = jacobian_fno(model, psi0_0, V_0)
    print(f"\n[Block 3] J_fno_pos shape: {J_fno_pos.shape}, dtype: {J_fno_pos.dtype}")  # (128,128) complex128
    print(f"[Block 3] J_fno_pos finite: {bool(jnp.all(jnp.isfinite(J_fno_pos)))}")

    J_fno_kq = mode_coupling_matrix(J_fno_pos)

    # same k0-centering gate as J_true: FNO should inherit the carrier shift
    absJf = jnp.abs(J_fno_kq)
    kk, qq = jnp.meshgrid(k_idx, k_idx, indexing="ij")
    offset = ((kk - qq + nx//2) % nx) - nx//2
    dv = jnp.unique(offset.ravel())
    prof = jnp.array([jnp.mean(absJf.ravel()[offset.ravel() == dd]) for dd in dv])
    peak_f = int(dv[jnp.argmax(prof)])
    mass_f = float(jnp.sum(absJf[jnp.abs(offset) <= 16]) / jnp.sum(absJf))
    print(f"[Block 3] J_FNO band peak at k-q = {peak_f}   (J_true was {k0_0})")
    print(f"[Block 3] J_FNO mass within |k-q|<=16: {mass_f:.3f}   (J_true was 0.995)")

    #np.save("results/J_fno_kq_sample0.npy", np.asarray(J_fno_kq))
    #np.save("results/J_fno_pos_sample0.npy", np.asarray(J_fno_pos))
    #print("[Block 3] saved J_FNO (kq + pos) to results/")


    # --- Block 4: response comparison, sample 0 ---
    # both already in memory (J_kq = J_true, J_fno_kq = J_FNO); saved to disk too.
    err_kq  = float(response_error(J_fno_kq, J_kq))
    err_pos = float(response_error(J_fno_pos, J_pos))
    print(f"\n[Block 4] response rel-Frobenius (k,q basis): {err_kq:.3f}")
    print(f"[Block 4] response rel-Frobenius (pos basis): {err_pos:.3f}   (should match k,q — rotation is unitary)")

    # --- k-q resolved error map: where does the error live? ---
    # error binned by offset d = k - q, same centering as the band diagnostic.
    absE = jnp.abs(J_fno_kq - J_kq)
    kk, qq = jnp.meshgrid(k_idx, k_idx, indexing="ij")
    offset = ((kk - qq + nx//2) % nx) - nx//2
    dv = jnp.unique(offset.ravel())
    err_profile = jnp.array([jnp.mean(absE.ravel()[offset.ravel() == dd]) for dd in dv])

    # where is the error concentrated, relative to the band center (k0=4)?
    err_peak_d = int(dv[jnp.argmax(err_profile)])
    print(f"[Block 4] error profile peaks at k-q = {err_peak_d}   (band center k0 = {k0_0})")

    # weak sample-0 hint: is error biased to the HIGH-k edge of the band?
    # mass of error at offsets ABOVE the band center vs BELOW
    above = float(jnp.sum(absE[offset > k0_0]))
    below = float(jnp.sum(absE[offset < k0_0]))
    print(f"[Block 4] error mass above band center: {above:.3e}, below: {below:.3e}, "
          f"ratio a/b: {above/below:.2f}")
    print("[Block 4] (sample-0 hint only — mechanism needs the k0 sweep, Phase 5)")

    np.save("results/response_err_map_sample0.npy", np.asarray(absE))
    print("[Block 4] saved error map to results/")