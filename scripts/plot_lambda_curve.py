# scripts/plot_lambda_curve.py — Phase 6 figure #2: response + forward vs lambda.
# Two-panel, 3-seed means with spread bars, M=12 + untrained reference lines.
# Data hardcoded (LOCKED — final numbers, reproducible from PHASE6_NOTES alone).
import numpy as np
import matplotlib.pyplot as plt

# --- locked 3-seed data (seeds 0,1,2) ---
# lambda: [0, 0.1, 1, 10] have all 3 seeds; 0.3, 3 are seed-0 only (plotted faint)
lam_full = np.array([0.1, 1.0, 10.0])           # 3-seed points
resp_seeds = {                                   # response err per seed
    0.1: [0.1032, 0.1052, 0.1017],
    1.0: [0.0737, 0.0732, 0.0706],
    10.0:[0.0568, 0.0585, 0.0548],
}
fwd_seeds = {                                    # forward rel-L2 per seed
    0.1: [0.0173, 0.0169, 0.0163],
    1.0: [0.0148, 0.0146, 0.0139],
    10.0:[0.0137, 0.0138, 0.0133],
}
# lambda=0 floor (3-seed)
resp_0 = [0.5787, 0.5957, 0.5876]
fwd_0  = [0.0192, 0.0187, 0.0183]

# seed-0-only intermediate points (0.3, 3) — plotted faint for shape
lam_s0   = np.array([0.3, 3.0])
resp_s0  = np.array([0.0869, 0.0645])
fwd_s0   = np.array([0.0161, 0.0140])

# reference baselines (horizontal lines on response panel)
untrained_floor = 1.0002

m12_resp_seeds = [0.2455, 0.2951, 0.2458]
m12_resp = np.mean(m12_resp_seeds)          # 0.262
m12_resp_spread = np.std(m12_resp_seeds)    # 0.023
m12_fwd_seeds = [0.0212, 0.0209, 0.0203]
m12_fwd = np.mean(m12_fwd_seeds)            # 0.0208

def mean_spread(d):
    m = np.array([np.mean(d[l]) for l in lam_full])
    s = np.array([np.std(d[l]) for l in lam_full])
    return m, s

resp_m, resp_s = mean_spread(resp_seeds)
fwd_m,  fwd_s  = mean_spread(fwd_seeds)
resp0_m, resp0_s = np.mean(resp_0), np.std(resp_0)
fwd0_m,  fwd0_s  = np.mean(fwd_0),  np.std(fwd_0)

# x for the lambda=0 point on a log axis: place it at a small pseudo-value
# with a broken-axis marker. Cleaner: use symlog and put 0 at left, OR plot
# lambda=0 as a separate anchor. We use x=0.03 as the "lambda=0" slot on log-x
# and label it explicitly — avoids log(0).
lam0_x = 0.03

fig, (ax_r, ax_f) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})

# ===== response panel =====
# lambda=0 floor point
ax_r.errorbar([lam0_x], [resp0_m], yerr=[resp0_s], fmt="s", color="crimson",
              ms=9, capsize=4, label="λ=0 (forward-only)", zorder=5)
# 3-seed DINO points
ax_r.errorbar(lam_full, resp_m, yerr=resp_s, fmt="o-", color="C0",
              ms=7, capsize=4, lw=2, label="M=16 + DINO (3 seeds)", zorder=4)
# seed-0-only intermediate
ax_r.plot(lam_s0, resp_s0, "o", color="C0", alpha=0.35, ms=6,
          label="λ=0.3, 3 (seed 0 only)", zorder=3)
# reference lines
ax_r.axhline(untrained_floor, ls=":", color="gray", lw=1.5,
             label=f"untrained floor ({untrained_floor:.2f})")
ax_r.axhline(m12_resp, ls="--", color="darkorange", lw=1.5,
             label=f"M=12 forward-only ({m12_resp:.2f}, 3 seeds)")
ax_r.axhspan(m12_resp - m12_resp_spread, m12_resp + m12_resp_spread,
             color="darkorange", alpha=0.12, zorder=0)
ax_r.set_ylabel("response error\n‖J_fno − J_true‖ / ‖J_true‖")
ax_r.set_title("DINO recovers the response operator at no forward cost")
ax_r.legend(fontsize=8, loc="center right")
ax_r.grid(alpha=0.3)
ax_r.set_ylim(0, 1.05)

# ===== forward panel =====
ax_f.errorbar([lam0_x], [fwd0_m], yerr=[fwd0_s], fmt="s", color="crimson",
              ms=9, capsize=4, zorder=5)
ax_f.errorbar(lam_full, fwd_m, yerr=fwd_s, fmt="o-", color="C0",
              ms=7, capsize=4, lw=2, zorder=4)
ax_f.plot(lam_s0, fwd_s0, "o", color="C0", alpha=0.35, ms=6, zorder=3)
ax_f.axhline(m12_fwd, ls="--", color="darkorange", lw=1.5,
             label=f"M=12 forward ({m12_fwd:.4f})")
ax_f.set_ylabel("forward rel-L2\n(ψ_T accuracy)")
ax_f.set_xlabel("DINO weight λ  (log scale; λ=0 shown at left)")
ax_f.set_xscale("log")
ax_f.legend(fontsize=8, loc="upper right")
ax_f.grid(alpha=0.3)
ax_f.set_ylim(0, 0.022)

# x-axis: mark the pseudo-lambda=0 slot
ax_f.set_xticks([lam0_x, 0.1, 1, 10])
ax_f.set_xticklabels(["0", "0.1", "1", "10"])

plt.tight_layout()
plt.savefig("results/dino_lambda_curve.png", dpi=150, bbox_inches="tight")
print("saved results/dino_lambda_curve.png")