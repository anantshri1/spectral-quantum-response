# scripts/sweep_response_vs_n.py — response-vs-N sweep (Task A). DISPOSABLE.
# Forward-only (lambda=0), M16, N in {250,500,1000,2000} x seed {0,1,2}.
# N=2000 cells already exist (4-seed grid) -> they skip; included so the grid
# is self-documenting and Block 3 can eval a complete N-axis including the anchor.
import os, sys, subprocess, argparse, itertools

NS    = [250, 500, 1000, 2000]
SEEDS = [0, 1, 2]
M     = 16
LAM   = 0.0

def ckpt_path(n, s):
    # MUST match train_dino's write string exactly (lines 212-214):
    #   dino_N{n_actual}_seed{seed}_M{n_modes}_lam{dino_lambda}_final.eqx
    return f"checkpoints/dino_N{n}_seed{s}_M{M}_lam{LAM}_final.eqx"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="print the plan, launch nothing")
    ap.add_argument("--force",   action="store_true", help="re-run even if final ckpt exists")
    args = ap.parse_args()

    os.makedirs("logs/sweep_n", exist_ok=True)
    cells = list(itertools.product(NS, SEEDS))
    print(f"grid: {len(cells)} cells (N={NS} x seed={SEEDS}, M={M}, lam={LAM})\n")

    for n, s in cells:
        path = ckpt_path(n, s)
        exists = os.path.exists(path)
        tag = f"N={n:<4} seed={s}"
        if exists and not args.force:
            print(f"[skip] {tag}  (found {path})")
            continue
        if args.dry_run:
            print(f"[run ] {tag}  -> {path}")
            continue

        cmd = [sys.executable, "-m", "scripts.train_dino",
               "--n_train", str(n), "--seed", str(s),
               "--n_modes", str(M), "--dino_lambda", str(LAM)]
        log = f"logs/sweep_n/dino_N{n}_seed{s}_M{M}_lam{LAM}.log"
        print(f"[run ] {tag}  -> {path}  (log: {log})")
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"       FAILED (returncode {r.returncode}) — see {log}; continuing.")
        else:
            print(f"       done.")

if __name__ == "__main__":
    main()