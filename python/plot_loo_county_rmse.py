"""Map the across-folds per-county RMSE from a completed LOO experiment.

run_loo_experiment.py writes loo_county_rmse.csv at the experiment root — one
row per adm2 unit with RMSE/MAE aggregated over every LOO fold. This script
joins that table to an adm2 shapefile and draws a choropleth. It is the
experiment-level counterpart to plot_results.py (which only maps a single fold).

--aoi picks the id column, projection and default shapefile (see AOI_GEO in
plot_results.py), so --county_shp is only needed for a non-standard file.

Usage
-----
    python plot_loo_county_rmse.py \\
        --exp_dir  /path/to/results/loo_corn_tst_..._2003_2015_<ts>

    python plot_loo_county_rmse.py --aoi BF \\
        --exp_dir  /path/to/results/featurespace_bf/loo_corn_tst_..._<ts>

Output (written into --exp_dir)
    loo_county_<metric>_map.png
"""

import os
import argparse
import pandas as pd

# Reuse the vetted shapefile loading / choropleth helpers.
from plot_results import AOI_GEO, aoi_geo_config, load_counties_geo, _choropleth
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_dir", required=True,
                   help="Experiment folder containing loo_county_rmse.csv.")
    p.add_argument("--aoi", default="CONUS", choices=sorted(AOI_GEO),
                   help="AOI whose adm2 geometry to draw. Selects the id column, "
                        "projection and default shapefile. Default: CONUS.")
    p.add_argument("--county_shp", default=None,
                   help="adm2 shapefile. Optional — defaults to the known file for "
                        "--aoi (CONUS: cb_2020_us_county_20m.shp, "
                        "BF: cy-bench/polygons/BF/BF.shp).")
    p.add_argument("--metric", default="rmse",
                   choices=["rmse", "mae", "n_obs", "rmse_std"],
                   help="Column from loo_county_rmse.csv to map. Default: rmse. "
                        "'rmse_std' is the across-seed spread and exists only in "
                        "an aggregate_seed_runs.py output folder.")
    p.add_argument("--clip_pct", type=float, default=99.0,
                   help="Clip the colour scale at this percentile of the metric "
                        "so a few extreme counties don't wash out the map. "
                        "Set 100 to disable. Default: 99.")
    p.add_argument("--cmap", default=None,
                   help="Matplotlib colormap. Default: YlOrRd for rmse/mae, "
                        "viridis for n_obs.")
    return p.parse_args()


def plot_loo_county_map(exp_dir, counties_gdf, metric="rmse",
                        clip_pct=99.0, cmap=None, aoi="CONUS"):
    """Draw the across-folds per-county metric map for one experiment.

    Reads {exp_dir}/loo_county_rmse.csv, joins it to a pre-loaded county
    GeoDataFrame (so a batch caller loads the shapefile once), and writes
    loo_county_{metric}_map.png into exp_dir. Returns the output path.
    """
    csv_path = os.path.join(exp_dir, "loo_county_rmse.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"loo_county_rmse.csv not found in {exp_dir!r}. "
            "Point exp_dir at a completed run_loo_experiment.py output folder."
        )
    county_rmse = pd.read_csv(csv_path)
    if metric not in county_rmse.columns:
        hint = ("  rmse_std is written only by aggregate_seed_runs.py — point "
                "--exp_dir at an agg_cfg-* folder." if metric == "rmse_std" else "")
        raise ValueError(
            f"Column {metric!r} not in loo_county_rmse.csv "
            f"(has {list(county_rmse.columns)}).{hint}"
        )
    # US FIPS codes lose their leading zero through CSV round-trips, so re-pad
    # them to match GEOID. cy-bench ids are alphanumeric (zfill=0) — leave them.
    zfill = aoi_geo_config(aoi)["zfill"]
    county_rmse["adm2_id"] = county_rmse["adm2_id"].astype(str)
    if zfill:
        county_rmse["adm2_id"] = county_rmse["adm2_id"].str.zfill(zfill)

    merged = counties_gdf.merge(
        county_rmse[["adm2_id", metric]],
        left_on="GEOID", right_on="adm2_id", how="left",
    )

    n_mapped = int(merged[metric].notna().sum())
    print(f"Admin-2 units in shapefile ({aoi}): {len(counties_gdf)}")
    print(f"Admin-2 units with a {metric} value: {n_mapped}")
    if n_mapped == 0:
        raise RuntimeError(
            "No units matched between loo_county_rmse.csv and the shapefile. "
            f"Check that --aoi {aoi} matches the experiment, and that the "
            "shapefile id column and adm2_id use the same codes."
        )

    # Colour-scale ceiling: clip at a percentile to keep the map readable.
    vals = merged[metric].dropna()
    vmax = None if clip_pct >= 100 else float(vals.quantile(clip_pct / 100))
    cmap = cmap or {"n_obs": "viridis", "rmse_std": "BuPu"}.get(metric, "YlOrRd")
    label = {"rmse": "RMSE (kg/ha)", "mae": "MAE (kg/ha)",
             "n_obs": "n folds",
             "rmse_std": "RMSE std across seeds (kg/ha)"}[metric]

    fig, ax = plt.subplots(figsize=(12, 7))
    _choropleth(ax, counties_gdf, merged, metric, cmap, label,
                county_level=True, vmax=vmax)

    exp_name = os.path.basename(os.path.normpath(exp_dir))
    overall = f"median {metric}={vals.median():,.0f}" if metric != "n_obs" else ""
    ax.set_title(f"LOO per-adm2 {metric.upper()} — {exp_name}\n"
                 f"{n_mapped} units  {overall}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    out_path = os.path.join(exp_dir, f"loo_county_{metric}_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return out_path


def main():
    args = parse_args()
    counties_gdf = load_counties_geo(args.county_shp, aoi=args.aoi)
    plot_loo_county_map(args.exp_dir, counties_gdf, metric=args.metric,
                        clip_pct=args.clip_pct, cmap=args.cmap, aoi=args.aoi)


if __name__ == "__main__":
    main()
