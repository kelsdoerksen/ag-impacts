"""
Run a full leave-one-out experiment over a study period and compute
RMSE per county aggregated across all folds.

Data is loaded once and reused across folds for efficiency.

Directory layout created under --results_dir:
    loo_{model}_{start}_{end}_cfg-{hash}_seed{seed}_{timestamp}/
        fold_{year}/          per-fold outputs (plot_results.py compatible)
            adm2_errors.csv
            adm1_rmse.csv
            run_meta.json
        loo_all_errors.csv    all county × year predictions combined
        loo_county_rmse.csv   RMSE per county across folds
        loo_meta.json         experiment metadata + per-fold summary
        run_diagnostics.json  seed, config fingerprint, environment, metrics
        run_diagnostics.txt   the same, human-readable

Seed replicates
---------------
Every run draws a fresh seed unless --seed is given, and records it. Runs that
share a feature space / AOI / CV setup share a config hash (cfg-xxxxxxxx in the
folder name), so a set of seed replicates is identifiable by folder name alone
and can be pooled by aggregate_seed_runs.py for mean ± std reporting:

    for s in 1 2 3 4 5; do
        python run_loo_experiment.py --seed $s ...same args... ;
    done
    python aggregate_seed_runs.py --runs_dir /path/to/results

Example:
    python run_loo_experiment.py \\
        --study_years 1999 2007 \\
        --data_dir /path/to/processed_data \\
        --results_dir /path/to/results \\
        --model_type rf
"""

import os
import re
import sys
import json
import hashlib
import platform
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import root_mean_squared_error

import agro_features
from models import YieldModel, SUPPORTED_MODELS, NAN_TOLERANT_MODELS
from data_loader import (
    ANOMALY_TARGETS,
    FEATURE_MODES,
    SUPPORTED_CROPS,
    TARGET_OPTIONS,
    WEATHER_VARS,
    CountyBaseline,
    STATIC_COVARS,
    draw_seed,
    history_leak_mask,
    load_and_prepare,
    load_static_covariates,
    parse_climate_vars,
    phenology_lags,
    resolve_climate_vars,
)

# The PatchTST implementation lives in the tst/ subpackage; add it to the path
# so its reusable helpers can be imported lazily (only when --model_type tst is
# used) to avoid requiring torch/transformers for the tabular models.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tst"))


def _import_tst():
    """Import PatchTST helpers on demand (heavy torch/transformers deps)."""
    from tst_implementation import (
        load_climate_crop_data,
        build_sequence_array,
        train_predict_patchtst,
    )
    return load_climate_crop_data, build_sequence_array, train_predict_patchtst


# Phenology variable tokens for --use_phenology (tst path only). Kept here so the
# argument parser does not need to import tst_implementation (which pulls torch).
# Must stay in sync with PHENOLOGY_CHANNELS in tst_implementation.py.
PHENOLOGY_VARS = ["ndvi", "fpar", "soil_moisture"]


def _parse_phenology_arg(s: str) -> list:
    """Parse a comma-separated --use_phenology value (e.g. 'ndvi,fpar')."""
    vars_ = [v.strip() for v in s.split(",") if v.strip()]
    bad   = [v for v in vars_ if v not in PHENOLOGY_VARS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown phenology variable(s) {bad}. Choices: {PHENOLOGY_VARS}"
        )
    return vars_


def _parse_months_arg(s: str) -> list:
    """Parse a --monthly_precip_months value: '3-11' or '3,4,5,6' (or mixed)."""
    months = []
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        if "-" in tok.lstrip("-"):
            lo, hi = tok.split("-", 1)
            rng = range(int(lo), int(hi) + 1)
        else:
            rng = [int(tok)]
        for m in rng:
            if not 1 <= m <= 12:
                raise argparse.ArgumentTypeError(
                    f"Month {m} out of range; --monthly_precip_months takes 1-12."
                )
            if m not in months:
                months.append(m)
    if not months:
        raise argparse.ArgumentTypeError(
            "--monthly_precip_months must name at least one month.")
    return sorted(months)


def _parse_monthly_stats_arg(s: str) -> list:
    """Parse a comma-separated --monthly_precip_stats value (e.g. 'sum,mean')."""
    toks = [t.strip() for t in s.split(",") if t.strip()]
    bad  = [t for t in toks if t not in agro_features.MONTHLY_STATS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown monthly statistic(s) {bad}. "
            f"Choices: {agro_features.MONTHLY_STATS}")
    if not toks:
        raise argparse.ArgumentTypeError(
            "--monthly_precip_stats must name at least one statistic.")
    return [s for s in agro_features.MONTHLY_STATS if s in toks]


def _parse_static_arg(s: str) -> list:
    """Parse a comma-separated --static_covars value (e.g. 'latlon')."""
    toks = [v.strip() for v in s.split(",") if v.strip()]
    bad  = [v for v in toks if v not in STATIC_COVARS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown static covariate(s) {bad}. Choices: {sorted(STATIC_COVARS)}"
        )
    return toks


# Models selectable on the CLI: the sklearn tabular models plus the PatchTST
# sequence model ("tst"), which uses a separate (sequence) data path.
MODEL_CHOICES = SUPPORTED_MODELS + ["tst"]

# Targets valid for the tabular (sklearn) and tst (PatchTST) paths respectively.
# yield_pct_change is a tabular-only derived target; log_yield is tst-only.
TABULAR_TARGETS = TARGET_OPTIONS + ANOMALY_TARGETS
TST_TARGETS     = ["yield", "log_yield"] + ANOMALY_TARGETS
ALL_TARGETS     = TARGET_OPTIONS + ["log_yield"] + ANOMALY_TARGETS


# ──────────────────────────────────────────────────────────────────────────────
# Run identity: config fingerprint + diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def agro_settings(args) -> dict:
    """The --agro_features knobs, as kwargs for agro_features.build_agro_features.

    Also used verbatim in the config fingerprint, so two runs whose degree-day
    base or month selection differ are correctly treated as different arms
    rather than pooled as seed replicates.
    """
    return {
        "postharvest_buffer":    args.postharvest_buffer,
        "gdd_base":              args.gdd_base,
        "gdd_method":            args.gdd_method,
        "edd_threshold":         args.edd_threshold,
        "monthly_precip":        bool(args.monthly_precip),
        "monthly_precip_months": (list(args.monthly_precip_months)
                                  if args.monthly_precip_months else None),
        "monthly_precip_stats":  (list(args.monthly_precip_stats)
                                  if args.monthly_precip_stats else None),
        "monthly_min_coverage":  args.monthly_min_coverage,
        "monthly_min_days":      args.monthly_min_days,
    }


def build_config_key(args, climate_vars, study_start, study_end):
    """Everything that defines this experiment arm, EXCEPT the seed.

    Two runs that differ only in --seed produce an identical config key (and so
    an identical config hash), which is what lets aggregate_seed_runs.py pool
    them into one mean ± std. Anything that changes the feature space, the data,
    the AOI, the CV scheme or the model capacity must appear here — otherwise
    runs that are not comparable would be averaged together.

    --sample_seed is a *configuration* seed (it picks which counties are in the
    study), not a training seed, so it belongs in the key: two runs over
    different county samples are different arms.
    """
    key = {
        "crop":               args.crop,
        "model_type":         args.model_type,
        "feature_mode":       args.feature_mode,
        "target":             args.target,
        "forecast_year":      args.forecast_year,
        "aoi":                args.aoi,
        "climate_model":      args.climate_model,
        "ens_member":         args.ens_member,
        "climate_vars":       list(climate_vars),
        "preseason_buffer":   args.preseason_buffer,
        "postharvest_buffer": args.postharvest_buffer,
        "study_start":        study_start,
        "study_end":          study_end,
        "yield_history":      bool(args.yield_history),
        "yield_history_base": args.yield_history_base,
        "yield_history_len":  args.yield_history_len,
        "static_covars":      list(args.static_covars) if args.static_covars else None,
        "val_years":          args.val_years,
        "allow_history_leakage": bool(args.allow_history_leakage),
        "use_phenology":      list(args.use_phenology) if args.use_phenology else None,
        "phenology_suffix":   args.phenology_suffix,
        "phenology_lag_years": args.phenology_lag_years,
        "phenology_lookback_years": args.phenology_lookback_years,
        "cv_scheme":          args.cv_scheme,
        "agro_features":      bool(args.agro_features),
        # basename only: the same lookup at a different path on another machine
        # is still the same configuration.
        "cell_lookup":        (os.path.basename(args.cell_lookup)
                               if args.cell_lookup else None),
        "region":             args.region,
        "random_region_seed": args.random_region_seed,
        "sample_counties":    args.sample_counties,
        "sample_seed":        args.sample_seed,
    }
    if args.agro_features:
        # Only hashed when the block is on, so existing runs keep their hash.
        key["agro"] = agro_settings(args)
    if args.model_type == "tst":
        # Model capacity / optimisation settings change the arm, but only the
        # tst path reads them — including them for tabular runs would churn the
        # hash whenever an unrelated PatchTST default moves.
        key["tst_hparams"] = {
            "patch_length": args.patch_length,
            "patch_stride": args.patch_stride,
            "d_model":      args.d_model,
            "num_layers":   args.num_layers,
            "num_heads":    args.num_heads,
            "ffn_dim":      args.ffn_dim,
            "attn_dropout": args.attn_dropout,
            "head_dropout": args.head_dropout,
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
        }
    return key


def config_hash(config_key) -> str:
    """Short stable fingerprint of a config key (sorted-key JSON → sha1)."""
    blob = json.dumps(config_key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def _git_commit():
    """Current repo commit (short), or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _package_versions(include_torch):
    """Versions of the libraries that affect results, best-effort."""
    versions = {"python": platform.python_version()}
    for mod in ["numpy", "pandas", "sklearn"] + (
            ["torch", "transformers"] if include_torch else []):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def write_diagnostics(exp_dir, diagnostics):
    """Write run_diagnostics.json (+ a readable .txt twin) into exp_dir.

    Called once before the folds run — so a crashed run still records its seed
    and configuration — and again afterwards with the fold metrics filled in.
    """
    with open(os.path.join(exp_dir, "run_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    lines = [
        "LOO run diagnostics",
        "=" * 60,
        f"exp_id       : {diagnostics['exp_id']}",
        f"config_hash  : {diagnostics['config_hash']}   "
        f"(runs sharing this hash are seed replicates)",
        f"seed         : {diagnostics['seed']}  ({diagnostics['seed_source']})",
        f"started      : {diagnostics['started_at']}",
        f"finished     : {diagnostics.get('finished_at') or '— (in progress)'}",
        f"git_commit   : {diagnostics.get('git_commit')}",
        f"host         : {diagnostics.get('host')}",
        "",
        "Command",
        "-" * 60,
        diagnostics["command"],
        "",
        "Configuration (defines the seed-replicate group)",
        "-" * 60,
    ]
    for k, v in sorted(diagnostics["config"].items()):
        lines.append(f"  {k:<26}: {v}")

    lines += ["", "Environment", "-" * 60]
    for k, v in sorted(diagnostics["versions"].items()):
        lines.append(f"  {k:<26}: {v}")

    results = diagnostics.get("results")
    if results:
        lines += ["", "Results", "-" * 60,
                  f"  {'n_folds':<26}: {results['n_folds']}",
                  f"  {'overall_rmse':<26}: {results['overall_rmse']:,.2f}",
                  f"  {'overall_mae':<26}: {results['overall_mae']:,.2f}",
                  "", f"  {'fold':>12}  {'rmse':>12}  {'r2':>8}  {'n_test':>7}"]
        for fold in results["folds"]:
            r2 = fold["r2"]
            r2_str = "     nan" if r2 is None or np.isnan(r2) else f"{r2:>8.3f}"
            lines.append(f"  {fold['fold_label']:>12}  {fold['rmse']:>12,.1f}  "
                         f"{r2_str}  {fold['n_test']:>7}")

    with open(os.path.join(exp_dir, "run_diagnostics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _sample_dispersed_counties(adm2_df: pd.DataFrame, n: int, seed: int) -> list:
    """Return n adm2_ids sampled proportionally across states (Hamilton method).

    Stratifying by state ensures counties are geographically spread across
    CONUS without requiring coordinate data.  Each state gets at least 1 county
    as long as n >= number of states in the data.
    """
    county_meta  = adm2_df[["adm2_id", "adm1"]].drop_duplicates("adm2_id")
    all_counties = county_meta["adm2_id"].unique()

    if n >= len(all_counties):
        return sorted(all_counties.tolist())

    state_counts = county_meta.groupby("adm1")["adm2_id"].count()

    # Hamilton (largest-remainder) proportional allocation
    quota  = state_counts / state_counts.sum() * n
    alloc  = quota.apply(np.floor).astype(int)
    deficit = n - alloc.sum()
    top_up  = (quota - alloc).nlargest(deficit).index
    alloc[top_up] += 1
    alloc = alloc.clip(lower=1)          # guarantee ≥1 per state

    rng      = np.random.default_rng(seed)
    selected = []
    for state, k in alloc.items():
        pool   = county_meta.loc[county_meta["adm1"] == state, "adm2_id"].values
        k      = min(int(k), len(pool))
        chosen = rng.choice(pool, size=k, replace=False)
        selected.extend(chosen.tolist())

    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOO experiment over a study period with county-level RMSE."
    )
    parser.add_argument(
        "--study_years", type=int, nargs=2, metavar=("START", "END"), required=True,
        help="Inclusive start and end of the study period (e.g. --study_years 1999 2007).",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing the processed parquet files.",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Root results directory. A timestamped experiment folder is created inside.",
    )
    parser.add_argument(
        "--model_type", type=str, choices=MODEL_CHOICES, default="rf",
        help=(
            f"Model type. Choices: {MODEL_CHOICES}. Default: rf. "
            "'tst' is the PatchTST sequence model and ignores --feature_mode "
            "(it always uses raw daily climate sequences)."
        ),
    )
    parser.add_argument(
        "--crop", type=str, choices=SUPPORTED_CROPS, default="corn",
        help=f"Crop to model. Choices: {SUPPORTED_CROPS}. Default: corn.",
    )
    parser.add_argument(
        "--feature_mode", type=str, choices=FEATURE_MODES, default="aggregated",
        help=(
            "'aggregated': mean/std/sum per variable over the seasonal window. "
            "'timeseries': full daily values (Nov 1 – Nov 1) as individual features. "
            "Default: aggregated."
        ),
    )
    parser.add_argument(
        "--preseason_buffer", type=int, default=60,
        help="Days before plant_start_doy (aggregated mode only, default: 60).",
    )
    parser.add_argument(
        "--postharvest_buffer", type=int, default=14,
        help="Days after harvest_end_doy (aggregated mode only, default: 14).",
    )
    parser.add_argument(
        "--target", type=str, choices=ALL_TARGETS, default="yield",
        help=(
            "'yield': predict absolute yield. "
            "'yield_pct_change': percent change vs. prior year (tabular models only). "
            "'log_yield': log(yield), reported in original units (tst only). "
            "'yield_anomaly': deviation from the county's own baseline, in kg/ha. "
            "'yield_zscore': that deviation in county standard deviations, so a "
            "prediction of 1.0 means 'one sigma above what this county normally does'. "
            "The two anomaly targets fit their baseline on each fold's training "
            "years only and invert predictions back to kg/ha, so RMSE stays "
            "comparable to a --target yield run. Default: yield."
        ),
    )
    parser.add_argument(
        "--baseline_metric", action="store_true", default=True,
        help="Score every fold against county climatology (what the county "
             "normally yields) and report anomaly-space R². On by default: a "
             "yield model that does not beat this null is not yet useful.",
    )
    parser.add_argument(
        "--no_baseline_metric", dest="baseline_metric", action="store_false",
        help="Skip the county-climatology baseline scoring.",
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Restrict to a single adm2_id (county FIPS), e.g. '19153'. "
             "Overrides --random_region_seed.",
    )
    parser.add_argument(
        "--random_region_seed", type=int, default=None,
        help="Integer seed to randomly select one adm2 (county). "
             "Ignored if --region is set.",
    )
    parser.add_argument(
        "--sample_counties", type=int, default=None,
        help="Sample this many counties dispersed across CONUS using stratified "
             "state sampling. Applied after --region/--random_region_seed filtering. "
             "Requires --sample_seed.",
    )
    parser.add_argument(
        "--sample_seed", type=int, default=None,
        help="Random seed for --sample_counties. Required when --sample_counties is set.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed for this run: sklearn random_state / PatchTST weight "
             "init, dropout and shuffling. The same seed is used for every fold, "
             "so one run = one replicate. Default: drawn from OS entropy, so "
             "launching the same command 5 times gives 5 independent replicates "
             "to average. The value used is recorded in run_diagnostics.json and "
             "in the folder name (seed<N>) — pass it back to reproduce a run. "
             "Distinct from --sample_seed, which picks the counties.",
    )

    # ── data source (used by both paths) ──
    parser.add_argument("--climate_model", type=str, default="EC-Earth3",
                        help="Climate model subdir/file prefix. Default: EC-Earth3.")
    parser.add_argument("--ens_member", type=str, default="r1i4p1f1",
                        help="Ensemble member tag in climate filenames. Default: r1i4p1f1.")
    parser.add_argument("--forecast_year", type=int, default=1, choices=[1, 2],
                        help="Which hindcast lead-year climate product to use. "
                             "1 (default): standard product in {data_dir}/{climate_model}/ "
                             "forecasting hindcast_start+1. 2: the FY2 product in "
                             "{data_dir}/FY2/ forecasting hindcast_start+2 (initialised "
                             "~2 years ahead). --study_years is always in hindcast_start "
                             "(s-year) terms; the target/yield year is s + forecast_year.")
    parser.add_argument("--climate_vars", type=parse_climate_vars, default=None,
                        metavar="VARS",
                        help="Comma-separated daily climate variables to use "
                             "(default: all four). Short tokens tasmax,tasmin,"
                             "hursmin,pr or full column names. E.g. "
                             "--climate_vars tasmax,tasmin,pr for a "
                             "temperature + precipitation feature space. "
                             "Applies to both paths: tabular models get only "
                             "those aggregate/daily columns, tst gets only "
                             "those input channels.")
    parser.add_argument("--yield_history", action="store_true",
                        help="Add each county's own prior observed yields as an extra "
                             "input: recency-ordered annual series back to "
                             "--yield_history_base (default off). Tabular models get "
                             "lag feature columns; the tst model gets a second "
                             "time-series branch.")
    parser.add_argument("--static_covars", type=_parse_static_arg, default=None,
                        metavar="VARS",
                        help="Comma-separated time-invariant per-unit features fed "
                             "through a small side branch (tst only). Currently: "
                             "'latlon' (polygon centroid). REQUIRES an anomaly "
                             "target: under leave-one-year-out a coordinate pair "
                             "is effectively a unit ID, so against absolute yield "
                             "the model just looks up each unit's mean (lat/lon "
                             "alone scores R2 0.547 on CONUS, matching climatology, "
                             "with anomaly R2 of 0.011). Once the baseline is "
                             "removed they can condition the weather response "
                             "instead. Default: off.")
    parser.add_argument("--static_shp", type=str, default=None,
                        help="adm2 shapefile for --static_covars. Optional: the "
                             "known path for --aoi is tried first, then the same "
                             "layout under --data_dir. Set AGIMPACTS_ROOT to move "
                             "the default off this machine's paths.")
    parser.add_argument("--val_years", type=int, default=2,
                        help="Whole training years held out per fold for "
                             "checkpoint selection (tst only). Without this the "
                             "Trainer selects the best epoch on the TEST fold, "
                             "which is model selection on the test set and "
                             "optimistically biases every reported metric. Held "
                             "out by year, not by row, so the split matches the "
                             "LOO protocol. 0 disables the split and takes the "
                             "final epoch instead. Default: 2.")
    parser.add_argument("--yield_history_len", type=int, default=None,
                        help="Rolling window: keep only the N most recent prior "
                             "years of history per sample. Unlike "
                             "--yield_history_base (an absolute floor year, which "
                             "blanks the early folds entirely) this window moves "
                             "with each sample's target year. Strongly "
                             "recommended: the default reaches back to the "
                             "earliest year in the crop file (1950 on CONUS), so "
                             "the vector spans a 3.6x technology trend and the "
                             "mean of it scores R2 = -1.1 as a predictor; the "
                             "last 5 years score +0.53. Default: unlimited.")
    parser.add_argument("--yield_history_base", type=int, default=None,
                        help="Oldest calendar year of yield history to include "
                             "(default: earliest year in the crop data).")
    parser.add_argument("--aoi", type=str, default="CONUS",
                        help="Area-of-interest tag in the data filenames: "
                             "crop_<crop>_<aoi>.parquet, the climate parquets "
                             "(climate_<aoi>_<model>_s<year>-<ens>.parquet; the "
                             "original CONUS series has no AOI token) and, unless "
                             "--phenology_suffix says otherwise, the phenology "
                             "parquets. E.g. 'CONUS', 'ET', 'BF'. Default: CONUS.")
    parser.add_argument("--use_phenology", type=_parse_phenology_arg, default=None,
                        metavar="VARS",
                        help="Comma-separated CY-Bench phenology variables to add "
                             "(off by default). tst appends them as daily channels; "
                             "tabular appends prior-season summary stats. Lagged by "
                             "--phenology_lag_years (default 1, forecast-safe). "
                             f"Choices: {','.join(PHENOLOGY_VARS)}. E.g. "
                             "--use_phenology ndvi,fpar. Restricts samples to "
                             "county-years with that data (NDVI/FPAR 2001+, soil "
                             "moisture 2003+).")
    parser.add_argument("--phenology_suffix", type=str, default=None,
                        help="AOI suffix in phenology filenames "
                             "(phenology_<var>_<suffix>.parquet). Defaults to --aoi.")
    parser.add_argument("--phenology_lag_years", type=int, default=1,
                        help="Shift phenology (e.g. NDVI) back this many years so the "
                             "target season uses the PREVIOUS season's signal. Default 1 "
                             "(forecast-safe). 0 = same-year (nowcast/leaky). Applies to "
                             "both tst channels and tabular aggregate features.")
    parser.add_argument("--phenology_lookback_years", type=int, default=1,
                        help="Number of consecutive prior seasons of phenology to "
                             "feed, starting at --phenology_lag_years (default 1 = "
                             "last season only). E.g. --phenology_lookback_years 4 "
                             "gives the tst model four NDVI channels (lags 1-4), one "
                             "per season on a shared day-of-season grid, so it can "
                             "read multi-year phenology; tabular models get one set "
                             "of summary stats per season (suffixed _lag{n}). Every "
                             "county-year must have all n seasons covered, so the "
                             "usable period shrinks (NDVI starts 2001, so a 4-year "
                             "lookback needs target year 2006+, i.e. FY1 s2005+).")

    # ── agronomic (crop-calendar) features, tabular aggregated mode only ──
    agro = parser.add_argument_group(
        "Agronomic features (--feature_mode aggregated, tabular models)")
    agro.add_argument("--agro_features", action="store_true",
                      help="Add crop-calendar features on top of the window "
                           "aggregates (off by default): growing degree days "
                           "and extreme degree days accumulated from planting "
                           "through harvest_end + --postharvest_buffer, "
                           "cumulative season precipitation, and per-month "
                           "precipitation sum/mean/min/max/variance. See "
                           "agro_features.py for the exact windows.")
    agro.add_argument("--gdd_base", type=float, default=agro_features.GDD_BASE_C,
                      help="GDD lower threshold in °C, subtracted from the daily "
                           "mean of tasmax/tasmin. Default 10 (50 °F, maize).")
    agro.add_argument("--gdd_method", type=str, choices=agro_features.GDD_METHODS,
                      default="plain",
                      help="'plain' (default): GDD = max(0, (tasmax+tasmin)/2 - "
                           "--gdd_base). 'capped': the textbook maize variant, "
                           "clipping tasmax down to 30 °C and tasmin up to 10 °C "
                           "first, so extreme heat stops adding development.")
    agro.add_argument("--edd_threshold", type=float,
                      default=agro_features.EDD_THRESHOLD_C,
                      help="Extreme-degree-day threshold in °C on daily tasmax "
                           "(default 30). Emits both the count of days above it "
                           "and the summed degrees above it over the season.")
    agro.add_argument("--no_monthly_precip", dest="monthly_precip",
                      action="store_false", default=True,
                      help="Skip the per-month precipitation statistics and keep "
                           "only the degree-day and season-total features.")
    agro.add_argument("--monthly_precip_months", type=_parse_months_arg,
                      default=None, metavar="MONTHS",
                      help="Calendar months to emit precipitation statistics for, "
                           "e.g. '3-11' or '4,5,6,7'. Default: chosen from the "
                           "data by coverage (see --monthly_min_coverage).")
    agro.add_argument("--monthly_precip_stats", type=_parse_monthly_stats_arg,
                      default=None, metavar="STATS",
                      help="Which per-month precipitation statistics to emit "
                           f"(default: all of {','.join(agro_features.MONTHLY_STATS)}). "
                           "Worth trimming: monthly 'min' is exactly 0 for "
                           "78-98%% of CONUS maize county-years — almost every "
                           "month has a dry day — so it carries close to no "
                           "signal while adding one column per month.")
    agro.add_argument("--monthly_min_days", type=int, default=20,
                      help="A month's statistics are only computed for a "
                           "county-year when at least this many of its days fall "
                           "inside the seasonal window, so a two-day sliver of "
                           "October never masquerades as a monthly mean. "
                           "Default: 20.")
    agro.add_argument("--monthly_min_coverage", type=float, default=0.95,
                      help="Keep a month only when at least this fraction of "
                           "county-years clear --monthly_min_days for it. "
                           "Default: 0.95. Crop calendars vary by county, so on "
                           "CONUS maize a lower threshold buys extra shoulder "
                           "months (Mar/Oct/Nov) at the cost of NaNs — which "
                           "only --model_type xgb can keep.")

    # ── cross-validation scheme ──
    parser.add_argument("--cv_scheme", type=str, choices=["year", "cell"], default="year",
                        help="Hold-out unit per LOO fold. 'year' (default): "
                             "leave-one-year-out (temporal forecasting — trains on all "
                             "other years). 'cell': leave-one-climate-cell-out — holds "
                             "out every county sharing a coarse climate grid cell, "
                             "across all years, to remove spatial-autocorrelation "
                             "leakage between neighbouring counties. 'cell' requires "
                             "--cell_lookup.")
    parser.add_argument("--cell_lookup", type=str, default=None,
                        help="Path to a county→climate-cell parquet (columns "
                             "adm2_id, climate_cell_id) from R/county-climate-cell.R. "
                             "Required for --cv_scheme cell.")
    parser.add_argument("--allow_history_leakage", action="store_true",
                        help="Disable the --yield_history leak guard. By default "
                             "each fold suppresses the prior-yield entries that "
                             "would expose the held-out fold's own labels to "
                             "training samples (under leave-one-year-out, year "
                             "T's yields are lag features of year T+1 samples). "
                             "Set this only to reproduce the older, leaky "
                             "behaviour for comparison.")

    # ── PatchTST hyperparameters (only used when --model_type tst) ──
    tst = parser.add_argument_group("PatchTST options (--model_type tst)")
    tst.add_argument("--patch_length", type=int,   default=16,
                     help="Patch size in days (e.g. 16 ≈ 2 weeks).")
    tst.add_argument("--patch_stride", type=int,   default=8,
                     help="Step between consecutive patches (50%% overlap at default).")
    tst.add_argument("--d_model",      type=int,   default=128,
                     help="Transformer hidden dimension.")
    tst.add_argument("--num_layers",   type=int,   default=3,
                     help="Number of Transformer encoder layers.")
    tst.add_argument("--num_heads",    type=int,   default=8,
                     help="Self-attention heads (d_model must be divisible by this).")
    tst.add_argument("--ffn_dim",      type=int,   default=256,
                     help="Feed-forward network inner dimension.")
    tst.add_argument("--attn_dropout", type=float, default=0.1,
                     help="Attention dropout probability.")
    tst.add_argument("--head_dropout", type=float, default=0.1,
                     help="Regression head dropout probability.")
    tst.add_argument("--epochs",       type=int,   default=50,
                     help="Training epochs per fold.")
    tst.add_argument("--batch_size",   type=int,   default=64)
    tst.add_argument("--lr",           type=float, default=1e-4,
                     help="AdamW learning rate.")
    return parser.parse_args()


def _fit_fold_baseline(args, tr_ids, tr_y):
    """County climatology for one fold, fit on training rows only.

    Returns None when --target is not an anomaly target and no baseline metric
    was requested, so the ordinary paths pay nothing.
    """
    if args.target not in ANOMALY_TARGETS and not args.baseline_metric:
        return None
    return CountyBaseline().fit(tr_ids, tr_y)


def _baseline_scores(baseline, te_ids, y_true, y_pred, standardize):
    """Score this fold against county climatology, and in anomaly space.

    Two numbers that absolute-yield RMSE cannot give you:
      * how the model compares to simply predicting what the county normally
        yields (the null a yield forecast has to beat), and
      * R² measured on the anomaly, where the between-county variance the model
        cannot learn from weather has been removed — i.e. actual weather skill.
    """
    if baseline is None:
        return {}
    base_pred = baseline.inverse(te_ids, np.zeros(len(y_true)), standardize=False)
    base_rmse = float(np.sqrt(np.mean((base_pred - y_true) ** 2)))
    model_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    a_true = baseline.transform(te_ids, y_true, standardize)
    a_pred = baseline.transform(te_ids, y_pred, standardize)
    ss_tot = float(np.sum((a_true - a_true.mean()) ** 2))
    return {
        "baseline_climatology_rmse": base_rmse,
        "skill_vs_climatology_pct": (float(100 * (1 - model_rmse / base_rmse))
                                     if base_rmse > 0 else None),
        "beats_climatology": bool(model_rmse < base_rmse),
        "r2_anomaly": (float(1 - np.sum((a_pred - a_true) ** 2) / ss_tot)
                       if ss_tot > 0 else None),
        "corr_anomaly": (float(np.corrcoef(a_true, a_pred)[0, 1])
                         if len(a_true) > 1 and a_true.std() > 0
                         and a_pred.std() > 0 else None),
    }


def _finalize_fold(meta_te, y_true, y_pred, all_years, fold_label, model_type,
                   fold_dir, n_train, train_years, test_years,
                   cv_scheme="year", fold_value=None, extra_meta=None):
    """Build the per-fold error table + metadata, write outputs, and log a summary.

    Shared by the tabular and PatchTST fold runners so both emit identical
    adm2_errors.csv / adm1_rmse.csv / run_meta.json (plot_results.py compatible).

    A "fold" holds out one CV unit: a year ('year' scheme) or a climate cell
    ('cell' scheme). train_years / test_years are the actual years present on
    each side of the split (for the cell scheme both span the full study range).

    Parameters
    ----------
    meta_te : pd.DataFrame
        Test-set metadata with columns adm2_id, adm2, adm1, hindcast_start
        (row order aligned to y_true / y_pred).
    y_true, y_pred : array-like, shape (N,) — true and predicted yield (orig units).
    n_train : int — number of training samples (for run_meta).
    extra_meta : dict, optional — model-specific fields merged into run_meta.json.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    rmse       = float(root_mean_squared_error(y_true, y_pred))
    mean_yield = float(y_true.mean())
    # R² is undefined for a single test sample (zero variance in denominator);
    # fall back to signed % error so the fold summary is still readable.
    if len(y_true) > 1:
        ss_res    = float(np.sum((y_true - y_pred) ** 2))
        ss_tot    = float(np.sum((y_true - y_true.mean()) ** 2))
        r2        = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        pct_error = None
    else:
        r2        = float("nan")
        pct_error = float((y_pred[0] - y_true[0]) / y_true[0] * 100)

    errors = meta_te.reset_index(drop=True).copy()
    errors["y_true"]    = y_true
    errors["y_pred"]    = y_pred
    errors["residual"]  = y_pred - y_true
    errors["abs_error"] = np.abs(errors["residual"])
    errors["sq_error"]  = errors["residual"] ** 2

    adm1_rmse = (
        errors.groupby("adm1")["sq_error"]
        .mean().pipe(np.sqrt).rename("rmse").reset_index()
        .assign(rmse_pct=lambda d: 100 * d["rmse"] / mean_yield)
        .sort_values("rmse", ascending=False)
    )

    os.makedirs(fold_dir, exist_ok=True)
    errors.to_csv(os.path.join(fold_dir, "adm2_errors.csv"), index=False)
    adm1_rmse.to_csv(os.path.join(fold_dir, "adm1_rmse.csv"), index=False)

    fold_meta = {
        "run_id":      f"fold_{fold_label}",
        "fold_label":  fold_label,
        "cv_scheme":   cv_scheme,
        "fold_value":  fold_value,
        # test_year kept for plot_results.py compatibility: the single held-out
        # year under the 'year' scheme, else None (cell folds span all years).
        "test_year":   (test_years[0] if cv_scheme == "year" and len(test_years) == 1
                        else None),
        "test_years":  list(test_years),
        "train_years": list(train_years),
        "study_start": all_years[0],
        "study_end":   all_years[-1],
        "model_type":  model_type,
        "rmse_overall":    rmse,
        "r2_overall":      r2,
        "pct_error":       pct_error,
        "mean_yield":      mean_yield,
        "n_train":         int(n_train),
        "n_test":          int(len(y_true)),
        "feature_importances": None,
    }
    if extra_meta:
        fold_meta.update(extra_meta)
    with open(os.path.join(fold_dir, "run_meta.json"), "w") as f:
        json.dump(fold_meta, f, indent=2)

    skill_str = f"err%={pct_error:+.1f}%" if pct_error is not None else f"R²={r2:.3f}"
    n_masked  = fold_meta.get("history_entries_masked")
    leak_note = f"  hist_masked={n_masked:,}" if n_masked else ""
    # Anomaly R² is the number that says whether the model beat county
    # climatology (predicting "no deviation from normal" scores 0 by
    # construction), so lead with it when it is available. Absolute-yield RMSE
    # stays alongside it for continuity with earlier runs.
    r2a = fold_meta.get("r2_anomaly")
    ca  = fold_meta.get("corr_anomaly")
    anom_str = ""
    if r2a is not None:
        anom_str = f"  R²anom={r2a:+.3f}"
        if ca is not None:
            anom_str += f"  r={ca:+.3f}"
    print(
        f"  fold {fold_label}: RMSE={rmse:>10,.0f}  {skill_str}{anom_str}"
        f"  n_train={n_train}  n_test={len(y_true)}{leak_note}"
    )
    return fold_meta, errors


def _target_years(meta, forecast_year):
    """Calendar year predicted by each row: target_year if present, else derived."""
    if "target_year" in meta:
        return np.asarray(meta["target_year"], dtype=int)
    return np.asarray(meta["hindcast_start"], dtype=int) + forecast_year


def _suppress_history_leak(hist, mask, meta_tr, meta_te, forecast_year):
    """Blank the training prior-yield entries that hold the held-out fold's labels.

    Returns (hist, mask, n_masked) with fresh arrays; inputs are not mutated.
    See data_loader.history_leak_mask for why these entries are leakage.
    """
    hist = np.array(hist, copy=True)
    mask = np.array(mask, copy=True)
    blocked = zip(np.asarray(meta_te["adm2_id"]),
                  _target_years(meta_te, forecast_year))
    leak = history_leak_mask(
        np.asarray(meta_tr["adm2_id"]),
        _target_years(meta_tr, forecast_year),
        hist.shape[1],
        blocked,
    )
    # Only entries that were real observations count as suppressed; padded slots
    # are already zero/masked and flipping them changes nothing.
    leak &= mask > 0
    hist[leak] = 0.0
    mask[leak] = 0.0
    return hist, mask, int(leak.sum())


def _yield_lag_columns(feature_cols):
    """Split the prior-yield lag/mask columns out of a tabular feature list.

    Returns (lag_cols, mask_cols) ordered by lag number (lag 1 first), or
    ([], []) when --yield_history was not used.
    """
    lags  = {}
    masks = {}
    for col in feature_cols:
        m = re.fullmatch(r"yield_lag(\d+)", col)
        if m:
            lags[int(m.group(1))] = col
            continue
        m = re.fullmatch(r"yield_lagmask(\d+)", col)
        if m:
            masks[int(m.group(1))] = col
    order = sorted(lags)
    if order != sorted(masks):
        raise ValueError("Mismatched yield_lag / yield_lagmask feature columns.")
    return [lags[j] for j in order], [masks[j] for j in order]


def run_fold(adm2_df, feature_cols, all_years, fold_col, fold_value, fold_label,
             model_type, fold_dir, args_ns, cv_scheme="year", forecast_year=1,
             guard_history_leak=True, seed=42):
    """Train a tabular model leaving out one fold (a year or a climate cell).

    The held-out fold is the rows where adm2_df[fold_col] == fold_value; the
    model trains on every other row. `seed` is the estimator's random_state and
    is held constant across the folds of one run (one run = one replicate).
    """
    meta_cols   = ["adm2_id", "adm2", "adm1", "hindcast_start"]

    train_mask = adm2_df[fold_col] != fold_value
    test_mask  = adm2_df[fold_col] == fold_value

    X_tr    = adm2_df.loc[train_mask, feature_cols]
    y_tr    = adm2_df.loc[train_mask, "yield"]
    X_te    = adm2_df.loc[test_mask,  feature_cols]
    y_te    = adm2_df.loc[test_mask,  "yield"]
    meta_te = adm2_df.loc[test_mask,  meta_cols].copy()

    if len(X_te) == 0:
        print(f"  fold {fold_label}: no test data — skipping")
        return None

    # Prior-yield lag features can carry this fold's held-out labels into the
    # training rows (year T's yield is year T+1's lag 1); blank those entries.
    n_hist_masked = None
    lag_cols, mask_cols = _yield_lag_columns(feature_cols)
    if lag_cols and guard_history_leak:
        hist, hmask, n_hist_masked = _suppress_history_leak(
            X_tr[lag_cols].to_numpy(dtype=float),
            X_tr[mask_cols].to_numpy(dtype=float),
            adm2_df.loc[train_mask], adm2_df.loc[test_mask], forecast_year,
        )
        X_tr = X_tr.copy()
        X_tr[lag_cols]  = hist
        X_tr[mask_cols] = hmask

    # Anomaly targets: fit the county baseline on this fold's training rows,
    # train on the anomaly, and invert the predictions back to kg/ha so every
    # metric and CSV downstream stays in yield units.
    tr_ids   = adm2_df.loc[train_mask, "adm2_id"].to_numpy()
    te_ids   = meta_te["adm2_id"].to_numpy()
    baseline = _fit_fold_baseline(args_ns, tr_ids, y_tr.to_numpy())
    standardize = args_ns.target == "yield_zscore"

    y_fit = y_tr
    if args_ns.target in ANOMALY_TARGETS:
        y_fit = baseline.transform(tr_ids, y_tr.to_numpy(), standardize)

    model = YieldModel(model_type, random_state=seed)
    model.fit(X_tr, y_fit)
    y_pred = model.predict(X_te)
    if args_ns.target in ANOMALY_TARGETS:
        y_pred = baseline.inverse(te_ids, y_pred, standardize)

    extra_meta = {
        "feature_importances": (
            dict(zip(feature_cols, model.feature_importances_.tolist()))
            if model.feature_importances_ is not None else None
        ),
        "history_entries_masked": n_hist_masked,
        "target": args_ns.target,
        "seed": seed,
        **_baseline_scores(baseline, te_ids, y_te.to_numpy(), y_pred, standardize),
    }
    train_years = sorted(adm2_df.loc[train_mask, "hindcast_start"].unique().tolist())
    test_years  = sorted(adm2_df.loc[test_mask,  "hindcast_start"].unique().tolist())
    return _finalize_fold(
        meta_te, y_te.values, y_pred, all_years, fold_label, model_type,
        fold_dir, n_train=len(X_tr), train_years=train_years, test_years=test_years,
        cv_scheme=cv_scheme, fold_value=fold_value, extra_meta=extra_meta,
    )


def run_fold_tst(sequences, targets, sample_meta, all_years, fold_col, fold_value,
                 fold_label, fold_dir, args, cv_scheme="year",
                 yield_hist=None, yield_mask=None, static_arr=None, seed=42):
    """Train PatchTST leaving out one fold (a year or a climate cell).

    The held-out fold is the samples where sample_meta[fold_col] == fold_value;
    the model trains on every other sample. When yield_hist/yield_mask are given
    (row-aligned to sequences), the prior-yield branch is trained too. `seed`
    drives weight init/dropout/shuffling and is held constant across the folds
    of one run (one run = one replicate).
    """
    _, _, train_predict_patchtst = _import_tst()
    meta_cols  = ["adm2_id", "adm2", "adm1", "hindcast_start"]
    fold_vals  = sample_meta[fold_col].values
    train_mask = fold_vals != fold_value
    test_mask  = fold_vals == fold_value

    if test_mask.sum() == 0:
        print(f"  fold {fold_label}: no test data — skipping")
        return None

    # Carve the model-selection set out of the TRAINING years, by whole years so
    # it matches the LOO protocol — a random sample of rows would put the same
    # year on both sides of the split, and yields within a year are strongly
    # correlated, so it would not measure generalisation to an unseen season.
    val_mask = np.zeros_like(train_mask)
    val_years = []
    if args.val_years > 0 and cv_scheme == "year":
        pool = sorted(set(sample_meta.loc[train_mask, "hindcast_start"]))
        n_val = min(args.val_years, max(len(pool) - 1, 0))
        if n_val:
            # Seeded per fold so the choice is reproducible but not identical
            # across folds (which would make one year carry all the selection).
            rng = np.random.default_rng(abs(hash((seed, str(fold_value)))) % (2 ** 32))
            val_years = sorted(rng.choice(pool, size=n_val, replace=False).tolist())
            val_mask  = train_mask & sample_meta["hindcast_start"].isin(val_years).to_numpy()
            train_mask = train_mask & ~val_mask
            print(f"  fold {fold_label}: validation years {val_years} "
                  f"({int(val_mask.sum()):,} samples) held out of training")

    train_seqs, train_tgts = sequences[train_mask], targets[train_mask]
    val_seqs,   val_tgts   = sequences[test_mask],  targets[test_mask]
    meta_te = sample_meta.loc[test_mask, meta_cols].copy()
    has_val = bool(val_mask.any())

    hist_kwargs   = {}
    n_hist_masked = None
    if yield_hist is not None:
        train_hist, train_hmask = yield_hist[train_mask], yield_mask[train_mask]
        eval_hist  = yield_hist[val_mask] if has_val else None
        eval_hmask = yield_mask[val_mask] if has_val else None
        # Prior-yield history makes a held-out year's LABELS visible as other
        # samples' lag features. Both held-out sets need blocking, not just the
        # test fold:
        #   * training rows must not carry test-year OR validation-year yields —
        #     the latter would make validation loss optimistic and corrupt the
        #     checkpoint choice, which is the whole point of the split;
        #   * validation rows must not carry test-year yields, since validation
        #     drives model selection.
        # Test rows keep their full history: a county's own past observed yields
        # are legitimately known at forecast time.
        if not args.allow_history_leakage:
            blocked_tr = pd.concat([sample_meta.loc[test_mask],
                                    sample_meta.loc[val_mask]]) if has_val \
                         else sample_meta.loc[test_mask]
            train_hist, train_hmask, n_hist_masked = _suppress_history_leak(
                train_hist, train_hmask,
                sample_meta.loc[train_mask], blocked_tr,
                args.forecast_year,
            )
            if has_val:
                eval_hist, eval_hmask, _ = _suppress_history_leak(
                    eval_hist, eval_hmask,
                    sample_meta.loc[val_mask], sample_meta.loc[test_mask],
                    args.forecast_year,
                )
        hist_kwargs = dict(
            train_hist=train_hist,          train_hist_mask=train_hmask,
            val_hist=yield_hist[test_mask], val_hist_mask=yield_mask[test_mask],
            eval_hist=eval_hist,            eval_hist_mask=eval_hmask,
        )

    # Anomaly targets: baseline fit on this fold's training samples only, so the
    # held-out year never contributes to its own normalisation. The model learns
    # the anomaly; predictions are inverted to kg/ha before scoring.
    tr_ids   = sample_meta.loc[train_mask, "adm2_id"].to_numpy()
    te_ids   = meta_te["adm2_id"].to_numpy()
    y_true_orig = np.asarray(val_tgts, dtype=float).reshape(-1)
    baseline    = _fit_fold_baseline(args, tr_ids,
                                     np.asarray(train_tgts, dtype=float).reshape(-1))
    standardize = args.target == "yield_zscore"

    eval_seqs = sequences[val_mask] if has_val else None
    eval_tgts = targets[val_mask]   if has_val else None

    fit_train, fit_val, fit_eval = train_tgts, val_tgts, eval_tgts
    if args.target in ANOMALY_TARGETS:
        fit_train = baseline.transform(tr_ids, np.asarray(train_tgts).reshape(-1),
                                       standardize).reshape(-1, 1).astype(np.float32)
        fit_val   = baseline.transform(te_ids, y_true_orig, standardize).reshape(-1, 1).astype(np.float32)
        if has_val:
            fit_eval = baseline.transform(
                sample_meta.loc[val_mask, "adm2_id"].to_numpy(),
                np.asarray(eval_tgts, dtype=float).reshape(-1),
                standardize).reshape(-1, 1).astype(np.float32)

    static_kwargs = {}
    if static_arr is not None:
        static_kwargs = dict(
            train_static=static_arr[train_mask],
            val_static=static_arr[test_mask],
            eval_static=static_arr[val_mask] if has_val else None,
        )

    log_target = args.target == "log_yield"
    y_pred = train_predict_patchtst(
        train_seqs, fit_train, val_seqs, fit_val,
        eval_seqs=eval_seqs, eval_tgts=fit_eval,
        run_dir=fold_dir,
        hparams=dict(
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            attention_dropout=args.attn_dropout,
            head_dropout=args.head_dropout,
        ),
        train_cfg=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr),
        log_target=log_target,
        seed=seed,
        **hist_kwargs,
        **static_kwargs,
    )

    if args.target in ANOMALY_TARGETS:
        y_pred = baseline.inverse(te_ids, y_pred, standardize)

    extra_meta = {
        "seed":         seed,
        "target":       args.target,
        "val_years": val_years or None,
        "n_val": int(val_mask.sum()),
        **_baseline_scores(baseline, te_ids, y_true_orig,
                           np.asarray(y_pred).reshape(-1), standardize),
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "d_model":      args.d_model,
        "num_layers":   args.num_layers,
        "num_heads":    args.num_heads,
        "ffn_dim":      args.ffn_dim,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "history_entries_masked": n_hist_masked,
    }
    train_years = sorted(sample_meta.loc[train_mask, "hindcast_start"].unique().tolist())
    test_years  = sorted(sample_meta.loc[test_mask,  "hindcast_start"].unique().tolist())
    return _finalize_fold(
        meta_te, val_tgts.squeeze(), y_pred, all_years, fold_label, "tst",
        fold_dir, n_train=int(train_mask.sum()), train_years=train_years,
        test_years=test_years, cv_scheme=cv_scheme, fold_value=fold_value,
        extra_meta=extra_meta,
    )


def _select_counties(county_meta, args):
    """Resolve which adm2_ids to keep from --region / --random_region_seed / --sample_counties.

    Operates on a county-level meta table (one row per county with columns
    adm2_id, adm2, adm1) so it is shared by the tabular and sequence data paths.

    Returns (selected_ids: set[str], region: str | None).
    """
    all_ids = sorted(county_meta["adm2_id"].unique())

    region = args.region
    if region is None and args.random_region_seed is not None:
        rng    = np.random.default_rng(args.random_region_seed)
        region = str(rng.choice(all_ids))
        row    = county_meta.loc[county_meta["adm2_id"] == region].iloc[0]
        print(f"Random region   : seed={args.random_region_seed} → "
              f"{row['adm2']}, {row['adm1']} (adm2_id={region})")

    if region is not None:
        if region not in county_meta["adm2_id"].values:
            raise ValueError(f"adm2_id '{region}' not found. Available: {all_ids}")
        selected = {region}
    else:
        selected = set(all_ids)

    if args.sample_counties is not None:
        if args.sample_seed is None:
            raise ValueError("--sample_seed is required when --sample_counties is set.")
        sub      = county_meta[county_meta["adm2_id"].isin(selected)]
        chosen   = _sample_dispersed_counties(sub, args.sample_counties, args.sample_seed)
        selected = set(chosen)
        n_states = county_meta[county_meta["adm2_id"].isin(selected)]["adm1"].nunique()
        print(f"County sample   : {len(selected)} counties across {n_states} states "
              f"(seed={args.sample_seed})")

    return selected, region


def _attach_climate_cell(meta_df, lookup_path):
    """Add a climate_cell_id column to a per-sample frame by adm2_id.

    Uses a dict map (not a merge) so row order/index is preserved — the tst path
    relies on sample_meta staying positionally aligned with its sequence/target
    arrays. Errors if any present county is missing from the lookup.
    """
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(f"--cell_lookup not found: {lookup_path!r}")
    lookup = pd.read_parquet(lookup_path)
    required = {"adm2_id", "climate_cell_id"}
    if not required.issubset(lookup.columns):
        raise ValueError(
            f"Cell lookup {lookup_path!r} must contain columns {required}; "
            f"got {list(lookup.columns)}."
        )
    cell_map = dict(zip(lookup["adm2_id"].astype(str), lookup["climate_cell_id"]))

    out = meta_df.copy()
    ids = out["adm2_id"].astype(str)
    out["climate_cell_id"] = ids.map(cell_map)
    missing = sorted(ids[out["climate_cell_id"].isna()].unique().tolist())
    if missing:
        raise ValueError(
            f"{len(missing)} selected county(ies) absent from --cell_lookup "
            f"(e.g. {missing[:5]}). Regenerate the lookup (R/county-climate-cell.R) "
            "or restrict --region/--sample_counties to covered counties."
        )
    out["climate_cell_id"] = out["climate_cell_id"].astype(int)
    n_cells   = out["climate_cell_id"].nunique()
    n_county  = out["adm2_id"].nunique()
    print(f"Climate cells   : {n_cells} cells over {n_county} counties "
          f"(avg {n_county / max(n_cells, 1):.1f} counties/cell)")
    return out


def main():
    args = parse_args()
    study_start, study_end = args.study_years
    all_years = list(range(study_start, study_end + 1))

    # Validate target/model compatibility before any expensive data loading.
    if args.model_type == "tst" and args.target not in TST_TARGETS:
        raise ValueError(
            f"--target {args.target!r} is not valid for model 'tst'. "
            f"Choose from {TST_TARGETS}."
        )
    if args.static_covars:
        if args.model_type != "tst":
            raise SystemExit("--static_covars is implemented for --model_type tst only.")
        if args.target not in ANOMALY_TARGETS:
            raise SystemExit(
                f"--static_covars requires an anomaly target, got --target "
                f"{args.target!r}. Under leave-one-year-out a coordinate pair is "
                f"effectively a unit ID: against absolute yield the model uses it "
                f"to look up each unit's mean (lat/lon alone scores R2 0.547 on "
                f"CONUS, matching county climatology, at anomaly R2 0.011). Use "
                f"--target {' or '.join(ANOMALY_TARGETS)} so the baseline is "
                f"already removed and the features can only condition the "
                f"weather response."
            )
    if args.model_type != "tst" and args.target not in TABULAR_TARGETS:
        raise ValueError(
            f"--target {args.target!r} is not valid for tabular model "
            f"{args.model_type!r}. Choose from {TABULAR_TARGETS}."
        )
    if args.agro_features:
        if args.model_type == "tst":
            raise SystemExit(
                "--agro_features is a tabular feature block; --model_type tst "
                "consumes raw daily sequences and would gain nothing from it. "
                "Use --model_type xgb (or rf/gbm) with --feature_mode aggregated."
            )
        if args.feature_mode != "aggregated":
            raise SystemExit(
                f"--agro_features requires --feature_mode aggregated, got "
                f"{args.feature_mode!r}. Timeseries mode loads a Nov 1 – Nov 1 "
                f"window, which truncates the tail of a typical maize season, so "
                f"degree-day totals computed from it would be wrong."
            )
    if args.cv_scheme == "cell" and not args.cell_lookup:
        raise ValueError(
            "--cv_scheme cell requires --cell_lookup (county→climate-cell parquet "
            "from R/county-climate-cell.R)."
        )

    # Resolve the climate feature space once; both paths and the run name use it.
    climate_vars = resolve_climate_vars(args.climate_vars)

    if args.phenology_lookback_years < 1:
        raise ValueError(
            f"--phenology_lookback_years must be >= 1, got "
            f"{args.phenology_lookback_years}."
        )
    if args.phenology_lookback_years > 1 and not args.use_phenology:
        raise ValueError(
            "--phenology_lookback_years has no effect without --use_phenology "
            "(e.g. --use_phenology ndvi)."
        )

    # One seed per run, used for every fold, so a run is one clean replicate.
    seed        = args.seed if args.seed is not None else draw_seed()
    seed_source = "explicit" if args.seed is not None else "drawn"

    # Config fingerprint: identical for runs that differ only in --seed, so a
    # replicate set is recognisable from the folder name alone.
    cfg      = build_config_key(args, climate_vars, study_start, study_end)
    cfg_hash = config_hash(cfg)

    lookback_tag = ("" if args.phenology_lookback_years == 1
                    else f"-lb{args.phenology_lookback_years}")
    pheno_tag = (("_pheno-" + "-".join(args.use_phenology) + lookback_tag)
                 if args.use_phenology else "")
    cv_tag    = "" if args.cv_scheme == "year" else f"_cv-{args.cv_scheme}"
    # Tag the run name with anything that changes the feature space, so ablation
    # arms are distinguishable by folder (defaults stay untagged for continuity).
    clim_tag  = ("" if climate_vars == list(WEATHER_VARS)
                 else "_clim-" + "-".join(v.split("_")[0] for v in climate_vars))
    hist_tag  = "_yhist" if args.yield_history else ""
    if args.yield_history and args.yield_history_len:
        hist_tag += f"{args.yield_history_len}"
    agro_tag  = "_agro" if args.agro_features else ""
    exp_id  = (
        f"loo_{args.crop}_{args.model_type}_{args.feature_mode}_{args.target}"
        f"{clim_tag}{hist_tag}{agro_tag}{pheno_tag}{cv_tag}"
        f"_{study_start}_{study_end}_cfg-{cfg_hash}_seed{seed}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = os.path.join(args.results_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Written now so an interrupted run still records its seed and config, then
    # rewritten with metrics once the folds finish.
    diagnostics = {
        "exp_id":      exp_id,
        "config_hash": cfg_hash,
        "seed":        seed,
        "seed_source": seed_source,
        "started_at":  datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "command":     " ".join(sys.argv),
        "data_dir":    os.path.abspath(args.data_dir),
        "results_dir": os.path.abspath(args.results_dir),
        "git_commit":  _git_commit(),
        "host":        platform.node(),
        "versions":    _package_versions(include_torch=args.model_type == "tst"),
        "config":      cfg,
        "results":     None,
    }
    write_diagnostics(exp_dir, diagnostics)

    print(f"Experiment      : {exp_id}")
    print(f"Config hash     : {cfg_hash}  (seed replicates share this)")
    print(f"Seed            : {seed}"
          f"{'' if seed_source == 'explicit' else '  (drawn — see run_diagnostics.json)'}")
    print(f"Crop            : {args.crop}")
    print(f"Study years     : {study_start}–{study_end}  ({len(all_years)} folds, s-year)")
    print(f"Forecast year   : FY{args.forecast_year}  "
          f"(target year = s + {args.forecast_year})")
    print(f"Model           : {args.model_type}")
    print(f"Feature mode    : {args.feature_mode}")
    print(f"Target          : {args.target}")
    print(f"Climate vars    : {','.join(climate_vars)}")
    if args.agro_features:
        print(f"Agro features   : on  (GDD base {args.gdd_base:g}°C "
              f"{args.gdd_method}, EDD > {args.edd_threshold:g}°C, monthly precip "
              f"{'on' if args.monthly_precip else 'off'})")
    if args.yield_history:
        print(f"Yield history   : on  (leak guard: "
              f"{'OFF — training sees held-out labels' if args.allow_history_leakage else 'on'})")
    print(f"CV scheme       : {args.cv_scheme}"
          + (f"  (lookup: {args.cell_lookup})" if args.cv_scheme == "cell" else ""))
    if args.use_phenology:
        lags = phenology_lags(args.phenology_lag_years, args.phenology_lookback_years)
        print(f"Phenology       : {','.join(args.use_phenology)} "
              f"({len(lags)} season(s), lags {lags[0]}–{lags[-1]}yr)")
    print(f"Results dir     : {exp_dir}")

    # Load data once — reused across all folds. The PatchTST ("tst") path uses
    # raw daily climate sequences; tabular models use featurised county-year rows.
    print("\nLoading data...")
    use_tst = args.model_type == "tst"
    adm2_df = feature_cols = None
    sequences = targets = sample_meta = None
    if use_tst:
        load_climate_crop_data, build_sequence_array, _ = _import_tst()
        from tst_implementation import phenology_channel_cols
        daily_df = load_climate_crop_data(
            data_dir=args.data_dir,
            hindcast_years=all_years,
            crop=args.crop,
            climate_model=args.climate_model,
            ens_member=args.ens_member,
            aoi=args.aoi,
            use_phenology=args.use_phenology,
            phenology_suffix=args.phenology_suffix,
            phenology_lag_years=args.phenology_lag_years,
            phenology_lookback_years=args.phenology_lookback_years,
            forecast_year=args.forecast_year,
        )
        # CLIMATE_CHANNELS is the full default set; climate_vars may be a subset
        # and its order fixes the input channel order. Phenology channels follow,
        # one per variable per lookback season.
        channels = list(climate_vars) + phenology_channel_cols(
            args.use_phenology, args.phenology_lag_years,
            args.phenology_lookback_years)
        sequences, targets, sample_meta = build_sequence_array(daily_df, channels=channels)
        # Prior-yield history branch (row-aligned to sequences/sample_meta).
        yield_hist = yield_mask = None
        if args.yield_history:
            from tst_implementation import build_yield_history_arrays
            yield_hist, yield_mask, _, _ = build_yield_history_arrays(
                sample_meta, args.data_dir, args.crop, args.aoi,
                base_year=args.yield_history_base,
                max_len=args.yield_history_len)
        # Static per-unit covariates, row-aligned to sequences/sample_meta.
        static_arr = None
        if args.static_covars:
            sc = load_static_covariates(args.static_covars, aoi=args.aoi,
                                        county_shp=args.static_shp,
                                        search_dirs=(args.data_dir,))
            cols = [c for t in args.static_covars for c in STATIC_COVARS[t]]
            merged = sample_meta[["adm2_id"]].merge(sc, on="adm2_id", how="left")
            missing = int(merged[cols].isna().any(axis=1).sum())
            if missing:
                raise ValueError(
                    f"{missing:,} of {len(merged):,} samples have no static "
                    f"covariates — check the adm2_id ids in the shapefile match "
                    f"the crop parquet for --aoi {args.aoi}."
                )
            static_arr = merged[cols].to_numpy(dtype=np.float32)
            print(f"[data] Static covariates: {cols} for "
                  f"{sample_meta['adm2_id'].nunique():,} units")
        county_meta = sample_meta[["adm2_id", "adm2", "adm1"]].drop_duplicates("adm2_id")
    else:
        adm2_df, feature_cols = load_and_prepare(
            crop=args.crop,
            data_dir=args.data_dir,
            feature_mode=args.feature_mode,
            climate_model=args.climate_model,
            ens_member=args.ens_member,
            preseason_buffer=args.preseason_buffer,
            postharvest_buffer=args.postharvest_buffer,
            target=args.target,
            forecast_year=args.forecast_year,
            aoi=args.aoi,
            yield_history=args.yield_history,
            yield_history_base=args.yield_history_base,
            yield_history_len=args.yield_history_len,
            use_phenology=args.use_phenology,
            phenology_lag_years=args.phenology_lag_years,
            phenology_suffix=args.phenology_suffix,
            phenology_lookback_years=args.phenology_lookback_years,
            climate_vars=climate_vars,
            agro=agro_settings(args) if args.agro_features else None,
            # xgboost learns a default split direction for NaN, so county-years
            # whose crop calendar misses one of the emitted months can stay in
            # the study instead of being dropped.
            allow_nan_features=args.model_type in NAN_TOLERANT_MODELS,
        )
        county_meta = adm2_df[["adm2_id", "adm2", "adm1"]].drop_duplicates("adm2_id")

    # Optional county subsetting (region / random seed / dispersed sample)
    selected, region = _select_counties(county_meta, args)
    if use_tst:
        keep        = sample_meta["adm2_id"].isin(selected).values
        sequences   = sequences[keep]
        targets     = targets[keep]
        if yield_hist is not None:
            yield_hist = yield_hist[keep]
            yield_mask = yield_mask[keep]
        sample_meta = sample_meta[keep].reset_index(drop=True)
        available   = set(sample_meta["hindcast_start"].unique())
        print(f"Counties        : {sample_meta['adm2_id'].nunique()}  "
              f"({len(sample_meta)} county-year samples)")
    else:
        adm2_df   = adm2_df[adm2_df["adm2_id"].isin(selected)].copy()
        available = set(adm2_df["hindcast_start"].unique())
        print(f"Counties        : {adm2_df['adm2_id'].nunique()}  "
              f"({len(adm2_df)} county-year rows)")

    # Attach climate-cell ids for the spatial CV scheme. Done after county
    # subsetting so only the counties actually in play need lookup coverage.
    if args.cv_scheme == "cell":
        if use_tst:
            sample_meta = _attach_climate_cell(sample_meta, args.cell_lookup)
        else:
            adm2_df = _attach_climate_cell(adm2_df, args.cell_lookup)

    missing = [y for y in all_years if y not in available]
    if missing:
        print(f"WARNING: no parquet data found for years: {missing}")

    # Build the fold plan from the chosen CV scheme. Each fold is
    # (fold_value, fold_label, dir_name): hold out rows where fold_col == fold_value.
    if args.cv_scheme == "cell":
        fold_col   = "climate_cell_id"
        meta_src   = sample_meta if use_tst else adm2_df
        cell_ids   = sorted(meta_src[fold_col].unique().tolist())
        folds      = [(c, f"cell_{c}", f"fold_cell_{c}") for c in cell_ids]
    else:
        fold_col   = "hindcast_start"
        folds      = [(y, str(y), f"fold_{y}") for y in all_years]

    # LOO loop
    print(f"\nRunning {len(folds)} LOO folds ({args.cv_scheme})...\n")
    all_errors_list = []
    fold_metas      = []

    for fold_value, fold_label, dir_name in folds:
        fold_dir = os.path.join(exp_dir, dir_name)
        if use_tst:
            result = run_fold_tst(sequences, targets, sample_meta, all_years,
                                  fold_col, fold_value, fold_label, fold_dir, args,
                                  cv_scheme=args.cv_scheme,
                                  yield_hist=yield_hist, yield_mask=yield_mask,
                                  static_arr=static_arr, seed=seed)
        else:
            result = run_fold(adm2_df, feature_cols, all_years,
                              fold_col, fold_value, fold_label,
                              args.model_type, fold_dir, args,
                              cv_scheme=args.cv_scheme,
                              forecast_year=args.forecast_year,
                              guard_history_leak=not args.allow_history_leakage,
                              seed=seed)
        if result is None:
            continue
        fold_meta, errors = result
        fold_metas.append(fold_meta)
        all_errors_list.append(errors)

    if not all_errors_list:
        raise RuntimeError("No folds produced results. Check that parquet files exist.")

    # Aggregate across folds
    all_errors = pd.concat(all_errors_list, ignore_index=True)

    county_rmse = (
        all_errors
        .groupby(["adm2_id", "adm2", "adm1"])
        .agg(
            rmse  =("sq_error",  lambda x: float(np.sqrt(x.mean()))),
            mae   =("abs_error", "mean"),
            n_obs =("sq_error",  "count"),
        )
        .reset_index()
        .sort_values("rmse", ascending=False)
    )

    overall_rmse = float(np.sqrt(all_errors["sq_error"].mean()))
    overall_mae  = float(all_errors["abs_error"].mean())

    # Save experiment-level outputs
    all_errors.to_csv(os.path.join(exp_dir, "loo_all_errors.csv"), index=False)
    county_rmse.to_csv(os.path.join(exp_dir, "loo_county_rmse.csv"), index=False)

    exp_meta = {
        "exp_id":             exp_id,
        "config_hash":        cfg_hash,
        "seed":               seed,
        "seed_source":        seed_source,
        "crop":               args.crop,
        "target":             args.target,
        "region":             region,
        "sample_counties":    args.sample_counties,
        "sample_seed":        args.sample_seed,
        "forecast_year":      args.forecast_year,
        "climate_vars":       climate_vars,
        "yield_history":      args.yield_history,
        "yield_history_base": args.yield_history_base,
        "yield_history_len":  args.yield_history_len,
        "static_covars":      list(args.static_covars) if args.static_covars else None,
        "val_years":          args.val_years,
        "allow_history_leakage": args.allow_history_leakage,
        "use_phenology":      args.use_phenology,
        "phenology_lag_years": args.phenology_lag_years,
        "phenology_lookback_years": args.phenology_lookback_years,
        "preseason_buffer":   args.preseason_buffer,
        "postharvest_buffer": args.postharvest_buffer,
        "agro_features":      bool(args.agro_features),
        "agro":               agro_settings(args) if args.agro_features else None,
        "study_start":        study_start,
        "study_end":          study_end,
        "model_type":         args.model_type,
        "feature_mode":       args.feature_mode,
        "cv_scheme":          args.cv_scheme,
        "cell_lookup":        args.cell_lookup,
        "use_phenology":      args.use_phenology,
        "n_folds":            len(fold_metas),
        "overall_rmse":       overall_rmse,
        "overall_mae":        overall_mae,
        "folds":              fold_metas,
    }
    with open(os.path.join(exp_dir, "loo_meta.json"), "w") as f:
        json.dump(exp_meta, f, indent=2)

    diagnostics["finished_at"] = datetime.now().isoformat(timespec="seconds")
    diagnostics["results"] = {
        "n_folds":      len(fold_metas),
        "overall_rmse": overall_rmse,
        "overall_mae":  overall_mae,
        "n_counties":   int(all_errors["adm2_id"].nunique()),
        "n_samples":    int(len(all_errors)),
        "folds": [
            {"fold_label": fm["fold_label"], "rmse": fm["rmse_overall"],
             "r2": fm["r2_overall"], "n_test": fm["n_test"]}
            for fm in fold_metas
        ],
    }
    write_diagnostics(exp_dir, diagnostics)

    # Summary
    print(f"\n{'='*60}")
    print(f"LOO experiment complete — {args.model_type.upper()} {study_start}–{study_end}")
    print(f"{'='*60}")
    single_sample_folds = any(fm["pct_error"] is not None for fm in fold_metas)
    skill_hdr = "err%  " if single_sample_folds else "R²    "
    # Anomaly columns: R² measured after each county's own baseline is removed.
    # Predicting "no deviation from normal" scores 0 there, so R²anom > 0 means
    # the fold beat county climatology — the question absolute RMSE cannot answer.
    has_anom = any(fm.get("r2_anomaly") is not None for fm in fold_metas)
    anom_hdr = f"  {'R²anom':>7}  {'r_anom':>7}  {'vs clim':>8}" if has_anom else ""
    print(f"{'Fold':>10}  {'RMSE':>10}  {skill_hdr:>6}{anom_hdr}  {'n_test':>7}")
    for fm in fold_metas:
        if fm["pct_error"] is not None:
            skill_val = f"{fm['pct_error']:>+6.1f}%"
        else:
            skill_val = f"{fm['r2_overall']:>6.3f} "
        anom_val = ""
        if has_anom:
            def _f(v, w=7, spec="+.3f"):
                return f"{'—':>{w}}" if v is None else f"{format(v, spec):>{w}}"
            anom_val = (f"  {_f(fm.get('r2_anomaly'))}  {_f(fm.get('corr_anomaly'))}"
                        f"  {_f(fm.get('skill_vs_climatology_pct'), 8, '+.1f')}")
        print(
            f"{fm['fold_label']:>10}  {fm['rmse_overall']:>10,.0f}"
            f"  {skill_val}{anom_val}  {fm['n_test']:>7}"
        )
    print(f"{'':>10}  {'─'*10}  {'─'*6}")
    print(f"{'Overall':>10}  {overall_rmse:>10,.0f}  {'':>6}  (LOO RMSE across all folds)")
    print(f"{'':>10}  {overall_mae:>10,.0f}  {'':>6}  (LOO MAE  across all folds)")
    if has_anom:
        r2as = [fm["r2_anomaly"] for fm in fold_metas if fm.get("r2_anomaly") is not None]
        beat = sum(1 for fm in fold_metas if fm.get("beats_climatology"))
        print(f"\n{'':>10}  mean R²anom = {np.mean(r2as):+.3f}   "
              f"folds beating county climatology: {beat}/{len(fold_metas)}")
        print(f"{'':>10}  (R²anom > 0 means the model beat 'predict this county's "
              f"normal yield'.)")
    print(f"\nCounty RMSE saved to: {exp_dir}/loo_county_rmse.csv")
    print(f"All errors saved to:  {exp_dir}/loo_all_errors.csv")
    print(f"Diagnostics saved to: {exp_dir}/run_diagnostics.json (seed={seed})")
    print(
        f"\nTo plot a single fold:\n"
        f"  python plot_results.py --save_dir {exp_dir}/fold_<year> [--county_shp ...]"
    )
    print(
        f"\nTo pool seed replicates of this configuration (cfg-{cfg_hash}):\n"
        f"  python aggregate_seed_runs.py --runs_dir {args.results_dir} "
        f"--config_hash {cfg_hash}"
    )


if __name__ == "__main__":
    main()
