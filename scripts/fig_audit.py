# scripts/fig_audit.py — what does the (r,cos) figure actually have on disk?
import glob, os, numpy as np

print("=== align_* files present ===")
for p in sorted(glob.glob("results/align_*.npz")):
    z = np.load(p)
    r, c = float(z["r"].mean()), float(z["cos"].mean())
    print(f"  {os.path.basename(p):48s}  r={r:.4f}  cos={c:.4f}")

print("\n=== response_vs_n.png exists? ===")
print("  ", os.path.exists("results/response_vs_n.png"))

print("\n=== resp_errs_vsN_* files (for the two-curve plot) ===")
for p in sorted(glob.glob("results/resp_errs_vsN_*.npy")):
    print(f"  {os.path.basename(p):48s}  mean={float(np.load(p).mean()):.4f}")