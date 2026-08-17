"""Plot an entire completed LOO experiment in one command.

Point --exp_dir at the experiment folder — the one whose name encodes the run
(e.g. loo_corn_tst_aggregated_yield_pheno-ndvi_1980_2020_<timestamp>) and which
contains one fold_<year>/ subfolder per LOO fold plus loo_county_rmse.csv.

This walks every fold_* subfolder and runs the full per-fold plot suite
(plot_results.plot_fold) in each, then draws the experiment-level across-folds
per-county map(s) (plot_loo_county_rmse.plot_loo_county_map). Shapefiles are
loaded once and reused across all folds.

Usage
-----
    python plot_loo_all.py \\
        --exp_dir    /home/crl/kdoerksen/results/loo_corn_tst_aggregated_yield_pheno-ndvi_1980_2020_20260728_214113 \\
        --county_shp /path/to/cb_2020_us_county_20m.shp

    python plot_loo_all.py --aoi BF \\
        --exp_dir    /path/to/results/BF/loo_corn_tst_..._2002_2018_<ts>

Notes
-----
- Folds with no data (e.g. pre-2002 years when NDVI is used) have no fold_<year>/
  folder and are simply not iterated.
- A failure in one fold is reported and skipped so the rest still plot.
"""

import argparse
import glob
import os
import re
import traceback

from plot_results import (
    AOI_GEO,
    DEFAULT_SHP,
    load_plot_geometries,
    plot_fold,
)
from plot_loo_county_rmse import plot_loo_county_map


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--exp_dir", required=True,
        help="Experiment folder containing fold_<year>/ subfolders and "
             "loo_county_rmse.csv (the run_loo_experiment.py output root).",
    )
    p.add_argument(
        "--shp_path", type=str, default=DEFAULT_SHP,
        help="Global shapefile (ISO + hierid columns) for state-level maps.",
    )
    p.add_argument(
        "--aoi", type=str, default="CONUS", choices=sorted(AOI_GEO),
        help="AOI whose adm2 geometry to draw. Selects the id column, "
             "projection and default shapefile. Non-CONUS AOIs skip the "
             "US-FIPS-keyed admin1 maps. Default: CONUS.",
    )
    p.add_argument(
        "--county_shp", type=str, default=None,
        help="adm2 shapefile. Optional — defaults to the known file for --aoi "
             "(CONUS: cb_2020_us_county_20m.shp, BF: cy-bench/polygons/BF/BF.shp). "
             "Required for admin2 (county) maps and the experiment-level map.",
    )
    p.add_argument(
        "--country", type=str, default="USA",
        help="ISO country code to subset from --shp_path. Default: USA.",
    )
    p.add_argument(
        "--n_bins", type=int, default=10,
        help="Observed-yield bins for each fold's bin analysis. Default: 10.",
    )
    p.add_argument(
        "--metrics", nargs="+", default=["rmse", "mae"],
        choices=["rmse", "mae", "n_obs", "rmse_std"],
        help="Which experiment-level county maps to draw from loo_county_rmse.csv. "
             "Default: rmse mae. 'rmse_std' (across-seed spread) only exists when "
             "--exp_dir is an aggregate_seed_runs.py output folder.",
    )
    p.add_argument(
        "--skip_folds", action="store_true",
        help="Only draw the experiment-level county map(s); skip per-fold plots.",
    )
    return p.parse_args()


def find_fold_dirs(exp_dir):
    """Return fold_* subdirectories, sorted by fold year when present."""
    dirs = [d for d in glob.glob(os.path.join(exp_dir, "fold_*"))
            if os.path.isdir(d)]

    def year_key(d):
        m = re.search(r"fold_(\d+)", os.path.basename(d))
        return int(m.group(1)) if m else 0

    return sorted(dirs, key=year_key)


def main():
    args = parse_args()
    if not os.path.isdir(args.exp_dir):
        raise NotADirectoryError(f"--exp_dir not found: {args.exp_dir!r}")

    # Load shapefiles once and reuse for every fold.
    states_gdf, counties_gdf = load_plot_geometries(
        args.shp_path, args.country, args.county_shp, aoi=args.aoi)

    n_ok = 0
    failures = []

    if not args.skip_folds:
        fold_dirs = find_fold_dirs(args.exp_dir)
        print(f"\nFound {len(fold_dirs)} fold_* folders in {args.exp_dir}")
        for d in fold_dirs:
            name = os.path.basename(d)
            if not os.path.exists(os.path.join(d, "run_meta.json")):
                print(f"  [skip] {name}: no run_meta.json (fold produced no output)")
                continue
            print(f"\n{'=' * 70}\n[{name}]")
            try:
                plot_fold(d, states_gdf, counties_gdf,
                          n_bins=args.n_bins, aoi=args.aoi)
                n_ok += 1
            except Exception as exc:  # keep going through the batch
                print(f"  [error] {name}: {exc}")
                traceback.print_exc()
                failures.append((name, str(exc)))

    # Experiment-level across-folds county map(s).
    exp_maps = []
    if counties_gdf is not None:
        print(f"\n{'=' * 70}\n[experiment-level county maps]")
        for metric in args.metrics:
            try:
                exp_maps.append(plot_loo_county_map(args.exp_dir, counties_gdf,
                                                    metric=metric, aoi=args.aoi))
            except Exception as exc:
                print(f"  [error] experiment {metric} map: {exc}")
                failures.append((f"exp:{metric}", str(exc)))
    else:
        print("\n[experiment-level county map] skipped — pass --county_shp to enable.")

    # Summary
    print(f"\n{'=' * 70}\nDone. Per-fold plotted: {n_ok}"
          + (f"  |  experiment maps: {len(exp_maps)}" if exp_maps else ""))
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, err in failures:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
