# scripts/timing_table.py — item (5): amortization / timing. x64 FIRST.
import jax; jax.config.update("jax_enable_x64", True)
import time, statistics, numpy as np, jax.numpy as jnp

from configs.default import Config
from scripts.train import load_split
from tests.measure_floor import load_fno_f64_ckpt
from src.solver import evolve
from src.responses import jacobian_true, jacobian_fno, mode_coupling_matrix

cfg = Config(); nx, nt = cfg.nx, cfg.nt; dt = cfg.t_end / cfg.nt
psi0, V, uT = load_split("data/test.npz", nx=nx)
psi0 = psi0.astype(jnp.complex128); V = V.astype(jnp.float64)
model = load_fno_f64_ckpt("checkpoints/dino_N2000_seed0_M16_lam0.0_final.eqx", cfg)
p0, v0 = psi0[0], V[0]                       # one representative query (shapes fix the flops)
print(f"[cfg] nx={nx} nt={nt} dt={dt:.3e}")

# --- the four ops, jitted. Closing over (nx,dt,nt,model) bakes them as compile-time
#     constants, so each jitted fn takes only the (psi0, V) arrays. ---
solve   = jax.jit(lambda p, v: evolve(p, v, nx, dt, nt))     # forward: psi_T
fno_fwd = jax.jit(lambda p, v: model(p, v))                  # forward: (nx,2)
jtrue   = jax.jit(lambda p, v: jacobian_true(p, v, nx, dt, nt))  # 128 JVPs thru scan
jfno    = jax.jit(lambda p, v: jacobian_fno(model, p, v))        # 128 JVPs thru net

def bench(fn, *args, reps=50, warmup=3):
    """Median wall-time per call. warmup absorbs compilation; block_until_ready
    defeats async dispatch. BOTH are mandatory or the number is fiction."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))          # 1st call compiles; rest settle
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))          # <-- without this you time dispatch
        ts.append(time.perf_counter() - t0)
    lo, hi = statistics.quantiles(ts, n=4)[0], statistics.quantiles(ts, n=4)[2]
    return statistics.median(ts), lo, hi

# --- DEMO: prove trap #1 to yourself on J_true (the heavy op) ---
jax.block_until_ready(jtrue(p0, v0))              # compile first, so this is fair
t0 = time.perf_counter(); y = jtrue(p0, v0);                         t_nowait = time.perf_counter() - t0
t0 = time.perf_counter(); y = jtrue(p0, v0); jax.block_until_ready(y); t_wait   = time.perf_counter() - t0
print(f"[trap#1] J_true dispatch-only (WRONG): {t_nowait*1e3:8.3f} ms")
print(f"[trap#1] J_true blocked      (RIGHT): {t_wait*1e3:8.3f} ms   ratio {t_wait/max(t_nowait,1e-9):.0f}x")

# --- Block 2: gates (timed ops compute the RIGHT thing) + four-way table ---

# gate A (the real one): timed J_true -> (k,q) must bit-match the frozen cache, sample0
Jt0_kq = np.asarray(mode_coupling_matrix(jtrue(p0, v0)))
cache0 = np.load("results/Jtrue_kq_200.npy")[0]
gA = np.max(np.abs(Jt0_kq - cache0))
print(f"[gateA] J_true(timed)->kq vs cache  max|diff| {gA:.2e}")
assert gA < 1e-8, "timed J_true != frozen cache — STOP, timing is on wrong code"

# gate B (precision-aware, soft): solver forward reproduces stored target psi_T
psiT = np.asarray(solve(p0, v0))                    # complex (nx,)
ut0  = np.asarray(uT[0])                            # (nx, 2) real channels [Re, Im]
ut0  = ut0[:, 0] + 1j * ut0[:, 1]                   # -> complex (nx,)
relB = np.linalg.norm(psiT - ut0) / np.linalg.norm(ut0)
print(f"[gateB] solver fwd vs stored uT     rel-L2  {relB:.2e}  (loose: uT is f32-origin)")
assert relB < 1e-3, "solver disagrees with stored target — STOP"

# gate C (light): J_fno well-formed; jacfwd machinery already proven by gate A
Jf0 = jfno(p0, v0)
assert Jf0.shape == (nx, nx) and np.iscomplexobj(Jf0) and np.all(np.isfinite(Jf0)), "J_fno malformed"
print(f"[gateC] J_fno {Jf0.shape} complex finite — OK")
print("[gates] PASSED\n")

# --- four-way table: median wall-time per single-sample query ---
rows = [
    ("solver_fwd", "solver forward (psi_T)",   solve),
    ("fno_fwd",    "FNO forward (psi_T)",       fno_fwd),
    ("j_true",     "J_true (128 JVPs / scan)",  jtrue),
    ("j_fno",      "J_fno  (128 JVPs / net)",   jfno),
]
times = {}
print(f"{'operation':28s} {'median (ms)':>12s} {'IQR (ms)':>20s}")
for key, name, fn in rows:
    med, lo, hi = bench(fn, p0, v0)
    times[key] = med
    print(f"{name:28s} {med*1e3:12.3f}   [{lo*1e3:7.3f}, {hi*1e3:7.3f}]")

print(f"\nforward surrogate speedup   solver/FNO    = {times['solver_fwd']/times['fno_fwd']:6.1f}x")
print(f"response surrogate speedup  J_true/J_fno   = {times['j_true']/times['j_fno']:6.1f}x  <- amortization-relevant")

# --- Block 3: amortization / break-even ---
t_true = times["j_true"]; t_fno = times["j_fno"]
saving = t_true - t_fno                       # s/query the surrogate saves (if > 0)
N_TEST = psi0.shape[0]

print("\n=== amortization ===")
print(f"per-query response operator:  J_true {t_true*1e3:6.1f} ms  vs  J_fno {t_fno*1e3:5.1f} ms   "
      f"({t_true/t_fno:.1f}x, saving {saving*1e3:.1f} ms/query)")
print(f"whole testbed direct J_true:  {N_TEST} x {t_true*1e3:.1f} ms = {N_TEST*t_true:.2f} s total")

T_TRAIN_SEC = None            # <-- fill: wall-time for ONE lambda=0.1 fit (e.g. 480 = ~8 min)
if T_TRAIN_SEC is not None:
    n_star = T_TRAIN_SEC / saving
    print(f"one training run:             {T_TRAIN_SEC:.0f} s")
    print(f"break-even  n* = T_train / saving = {T_TRAIN_SEC:.0f}s / {saving*1e3:.1f}ms "
          f"= {n_star:,.0f} response queries")
    print(f"  -> testbed has {N_TEST} queries -> n* is {n_star/N_TEST:,.0f}x the testbed volume")
    print(f"  -> direct J_true for the WHOLE testbed ({N_TEST*t_true:.1f} s) "
          f"{'<' if N_TEST*t_true < T_TRAIN_SEC else '>'} one training run ({T_TRAIN_SEC:.0f} s)")
else:
    print("T_TRAIN_SEC unset — fill it to get n* and the break-even verdict.")