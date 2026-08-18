# scripts/fig_rcos_plane.py — main figure: (r,cos) plane with raw=1 boundary + contours
import glob, numpy as np, matplotlib.pyplot as plt

def cell(pattern):
    """mean (r,cos) across whatever seeds match the glob; None if nothing matches."""
    ps = sorted(glob.glob(pattern))
    if not ps: return None
    r  = np.mean([np.load(p)["r"].mean()   for p in ps])
    c  = np.mean([np.load(p)["cos"].mean() for p in ps])
    return c, r, len(ps)          # (x=cos, y=r, nseed)

Ns = [250, 500, 1000, 2000]
lam0   = [cell(f"results/align_vsN_N{n}_seed*_M16_lam0.0.npz") for n in Ns]
lam01  = [cell(f"results/align_vsN_N{n}_seed*_M16_lam0.1.npz") for n in Ns]
m12    = cell("results/align_seed*_M12_lam0.0.npz")   # 4-seed N2000
m16    = cell("results/align_seed*_M16_lam0.0.npz")   # 4-seed N2000 (= lam0 endpoint)
for name, pts in [("lam0", lam0), ("lam0.1", lam01)]:
    assert all(p is not None for p in pts), f"{name} trajectory has a missing N — STOP"

fig, ax = plt.subplots(figsize=(6.4, 5.6))

# --- analytic raw-error field: raw = sqrt(r^2 + 1 - 2 r cos) ---
cc, rr = np.meshgrid(np.linspace(0.5, 1.05, 300), np.linspace(0.85, 1.55, 300))
raw = np.sqrt(rr**2 + 1 - 2*rr*cc)
cf = ax.contourf(cc, rr, raw, levels=np.linspace(0, 1.2, 13), cmap="Greys", alpha=0.55)
plt.colorbar(cf, ax=ax, label=r"analytic response error  $\sqrt{r^2+1-2r\cos}$")
ax.contour(cc, rr, raw, levels=[0.2, 0.4, 0.6, 0.8], colors="k", linewidths=0.4, alpha=0.4)
# raw = 1 boundary (== r = 2cos): "worse than random"
ax.contour(cc, rr, raw, levels=[1.0], colors="crimson", linewidths=2.0)
ax.plot([], [], color="crimson", lw=2, label=r"$r=2\cos$  (worse than random)")

def traj(pts, color, label):
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.plot(x, y, "-", color=color, lw=1.4, alpha=0.7, zorder=3)
    ax.scatter(x, y, s=[40+55*i for i in range(len(pts))], color=color,
               edgecolor="k", linewidth=0.5, zorder=4, label=label)
    for (xi, yi), n in zip(zip(x, y), Ns):
        ax.annotate(f"N={n}", (xi, yi), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, color=color)

traj(lam0,  "#1f77b4", r"forward-only ($\lambda=0$)")
traj(lam01, "#2ca02c", r"DINO ($\lambda=0.1$)")

ax.scatter(*m12[:2], marker="D", s=90, color="#ff7f0e", edgecolor="k",
           zorder=5, label=r"M12 $\lambda=0$, N2000 (4-seed)")
ax.annotate("M12", m12[:2], textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.scatter(1, 1, marker="*", s=260, color="gold", edgecolor="k", zorder=6, label="perfect (1,1)")

ax.set_xlabel(r"directional alignment  $\cos\theta$")
ax.set_ylabel(r"scale ratio  $r=\|J_{\rm fno}\|/\|J_{\rm true}\|$")
ax.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_title("Response-operator fidelity in the (alignment, scale) plane")
fig.tight_layout()
fig.savefig("results/fig_rcos_plane.png", dpi=200)
print("saved results/fig_rcos_plane.png")
print(f"lam0  N-traj (cos,r): {[(round(p[0],3),round(p[1],3)) for p in lam0]}")
print(f"lam01 N-traj (cos,r): {[(round(p[0],3),round(p[1],3)) for p in lam01]}")
print(f"M12 {tuple(round(v,3) for v in m12[:2])}  M16 {tuple(round(v,3) for v in m16[:2])}")