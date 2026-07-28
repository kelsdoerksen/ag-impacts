"""Map the across-folds per-county RMSE from a completed LOO experiment.

run_loo_experiment.py writes loo_county_rmse.csv at the experiment root — one
row per adm2 county with RMSE/MAE aggregated over every LOO fold. This script
joins that table to a county shapefile (by GEOID == adm2_id) and draws a
choropleth. It is the experiment-level counterpart to plot_results.py (which
only maps a single fold).

Usage
-----
    python plot_loo_county_rmse.py \\
        --exp_dir  /path/to/results/loo_corn_tst_..._2003_2015_<ts> \\
        --county_shp /path/to/cb_2020_us_county_20m.shp

Output (written into --exp_dir)
    loo_county_<metric>_map.png
"""

import os
import argparse
import pandas as pd

# Reuse the vetted shapefile loading / choropleth helpers.
from plot_results import load_counties_geo, _choropleth
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_dir", required=True,
                   help="Experiment folder containing loo_county_rmse.csv.")
    p.add_argument("--county_shp", required=True,
                   help="Census-style county shapefile with a GEOID column "
                        "(e.g. cb_2020_us_county_20m.shp).")
    p.add_argument("--metric", default="rmse", choices=["rmse", "mae", "n_obs"],
                   help="Column from loo_county_rmse.csv to map. Default: rmse.")
    p.add_argument("--clip_pct", type=float, default=99.0,
                   help="Clip the colour scale at this percentile of the metric "
                        "so a few extreme counties don't wash out the map. "
                        "Set 100 to disable. Default: 99.")
    p.add_argument("--cmap", default=None,
                   help="Matplotlib colormap. Default: YlOrRd for rmse/mae, "
                        "viridis for n_obs.")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = os.path.join(args.exp_dir, "loo_county_rmse.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"loo_county_rmse.csv not found in {args.exp_dir!r}. "
            "Point --exp_dir at a completed run_loo_experiment.py output folder."
        )
    county_rmse = pd.read_csv(csv_path)
    if args.metric not in county_rmse.columns:
        raise ValueError(
            f"Column {args.metric!r} not in loo_county_rmse.csv "
            f"(has {list(county_rmse.columns)})."
        )
    # adm2_id must be a zero-padded 5-char FIPS string to match GEOID.
    county_rmse["adm2_id"] = county_rmse["adm2_id"].astype(str).str.zfill(5)

    counties = load_counties_geo(args.county_shp)
    merged = counties.merge(
        county_rmse[["adm2_id", args.metric]],
        left_on="GEOID", right_on="adm2_id", how="left",
    )

    n_mapped = int(merged[args.metric].notna().sum())
    print(f"Counties in shapefile (CONUS): {len(counties)}")
    print(f"Counties with a {args.metric} value: {n_mapped}")
    if n_mapped == 0:
        raise RuntimeError(
            "No counties matched between loo_county_rmse.csv and the shapefile. "
            "Check that GEOID and adm2_id use the same 5-digit FIPS codes."
        )

    # Colour-scale ceiling: clip at a percentile to keep the map readable.
    vals = merged[args.metric].dropna()
    vmax = None if args.clip_pct >= 100 else float(vals.quantile(args.clip_pct / 100))
    cmap = args.cmap or ("viridis" if args.metric == "n_obs" else "YlOrRd")
    label = {"rmse": "RMSE (kg/ha)", "mae": "MAE (kg/ha)",
             "n_obs": "n folds"}[args.metric]

    fig, ax = plt.subplots(figsize=(12, 7))
    _choropleth(ax, counties, merged, args.metric, cmap, label,
                county_level=True, vmax=vmax)

    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    overall = f"median {args.metric}={vals.median():,.0f}" if args.metric != "n_obs" else ""
    ax.set_title(f"LOO per-county {args.metric.upper()} — {exp_name}\n"
                 f"{n_mapped} counties  {overall}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    out_path = os.path.join(args.exp_dir, f"loo_county_{args.metric}_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
