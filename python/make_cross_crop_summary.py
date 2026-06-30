"""
Aggregate per-crop PatchTST LOO results into a cross-crop comparison.

Scans --results_dir for the newest loo_<crop>_tst_* experiment per crop, reads
each crop_summary.json (written by make_crop_plots.py), and produces:

    <results_dir>/_tst_summary/cross_crop_skill.png
        (A) overall RMSE% per crop, (B) overall R² per crop,
        (C) RMSE% by LOO year, one line per crop
    <results_dir>/_tst_summary/cross_crop_table.csv
        one row per crop with headline metrics

Prints a markdown table to stdout for the written summary.

Usage:
    python make_cross_crop_summary.py [--results_dir DIR]
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CROPS = ["corn", "soy", "rice", "wheat", "sorghum"]


def load_summaries(results_dir):
    out = {}
    for crop in CROPS:
        dirs = sorted(glob.glob(os.path.join(results_dir, f"loo_{crop}_tst_*")))
        for d in reversed(dirs):  # newest first
            p = os.path.join(d, "crop_summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[crop] = json.load(f)
                out[crop]["_dir"] = d
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str,
                    default="/Users/kdoerkse/Documents/UCT/AgImpacts/results")
    args = ap.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        raise SystemExit("No crop_summary.json files found — has the run produced results yet?")

    out_dir = os.path.join(args.results_dir, "_tst_summary")
    os.makedirs(out_dir, exist_ok=True)

    # ── table ──
    rows = []
    for crop, s in summaries.items():
        o = s["overall"]
        rows.append({
            "crop": crop, "n_counties": s["n_counties"], "n_obs": s["n_obs"],
            "n_years": len(s["by_year"]),
            "mean_yield": o["mean_yield"], "rmse": o["rmse"], "mae": o["mae"],
            "rmse_pct": o["rmse_pct"], "r2": o["r2"], "bias": o["bias"],
        })
    table = pd.DataFrame(rows).sort_values("rmse_pct")
    table.to_csv(os.path.join(out_dir, "cross_crop_table.csv"), index=False)

    print("\n| crop | counties | obs | mean yield | RMSE | RMSE% | R² | bias |")
    print("|------|---------:|----:|-----------:|-----:|------:|---:|-----:|")
    for _, r in table.iterrows():
        print(f"| {r['crop']} | {r['n_counties']:,} | {r['n_obs']:,} | "
              f"{r['mean_yield']:,.0f} | {r['rmse']:,.0f} | {r['rmse_pct']:.1f}% | "
              f"{r['r2']:.3f} | {r['bias']:+,.0f} |")

    # ── figure ──
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    order = table["crop"].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
    cmap = {c: colors[i] for i, c in enumerate(order)}

    axA.bar(order, table["rmse_pct"], color=[cmap[c] for c in order])
    axA.set_ylabel("RMSE % of mean yield"); axA.set_title("(A) Overall RMSE% by crop")
    axA.tick_params(axis="x", rotation=30)

    axB.bar(order, table["r2"], color=[cmap[c] for c in order])
    axB.axhline(0, color="k", lw=0.8, ls=":")
    axB.set_ylabel("R²"); axB.set_title("(B) Overall R² by crop")
    axB.tick_params(axis="x", rotation=30)

    for crop, s in summaries.items():
        by = sorted(s["by_year"], key=lambda x: x["year"])
        yrs = [b["year"] for b in by]
        rp  = [b["rmse_pct"] for b in by]
        axC.plot(yrs, rp, "-o", ms=3, color=cmap.get(crop, "gray"), label=crop)
    axC.set_xlabel("LOO test year"); axC.set_ylabel("RMSE % of mean yield")
    axC.set_title("(C) RMSE% by year"); axC.legend(fontsize=8)

    fig.suptitle("PatchTST LOO — cross-crop comparison (1980–2007)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_crop_skill.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir}/cross_crop_skill.png and cross_crop_table.csv")


if __name__ == "__main__":
    main()
