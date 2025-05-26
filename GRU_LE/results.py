#!/usr/bin/env python3
# plot_le_stats.py
"""
Read a collection of *.npy Lyapunov-spectrum files, compute
per-index mean and std, print them, and plot the average spectrum
with error bars.
python plot_le_stats.py          # defaults: *.npy in CWD
python plot_le_stats.py --dir results --pattern 'GGRU*.npy'
"""
import argparse, glob, pathlib
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",     default=".",   help="directory containing *.npy")
    ap.add_argument("--pattern", default="lyap_spectrum_*.npy",
                    help="glob pattern for spectrum files")
    ap.add_argument("--outfile", default="le_spectrum_mean±std.png",
                    help="name of figure to save")
    args = ap.parse_args()

    # ---------- locate files ----------
    root = pathlib.Path(args.dir).expanduser()
    files = sorted(root.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {root}")

    # ---------- load & stack ----------
    spectra = [np.load(f) for f in files]                           # :contentReference[oaicite:0]{index=0}
    arr     = np.stack(spectra)                                     # shape (N_runs, H)

    # ---------- statistics ----------
    mean = arr.mean(axis=0)                                         # :contentReference[oaicite:1]{index=1}
    std  = arr.std(axis=0, ddof=1)                                  # unbiased estimator

    # ---------- print ----------
    print("\nIndex   mean λ_i     ±   std")
    print("-" * 32)
    for i, (m, s) in enumerate(zip(mean, std), start=1):
        print(f"{i:3d}   {m: .6f}   ± {s:.6f}")

    # ---------- plot ----------
    idx = np.arange(1, len(mean) + 1)
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(idx, mean, yerr=std, fmt="o", lw=1.2, capsize=3)   # :contentReference[oaicite:2]{index=2}
    plt.axhline(0, color="k", lw=.8, ls="--")
    plt.xlabel("Exponent index  $i$")
    plt.ylabel(r"$\langle\lambda_i\rangle \;\pm\; \sigma_i$")
    plt.title(f"Mean ± STD Lyapunov spectrum  (N = {len(arr)})")
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)
    print(f"\nSaved figure →  {args.outfile}")

if __name__ == "__main__":
    main()
