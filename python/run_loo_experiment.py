"""
Run a full leave-one-out experiment over a study period and compute
RMSE per county aggregated across all folds.

Data is loaded once and reused across folds for efficiency.

Directory layout created under --results_dir:
    loo_{model}_{start}_{end}_{timestamp}/
        fold_{year}/          per-fold outputs (plot_results.py compatible)
            adm2_errors.csv
            adm1_rmse.csv
            run_meta.json
        loo_all_errors.csv    all county × year predictions combined
        loo_county_rmse.csv   RMSE per county across folds
        loo_meta.json         experiment metadata + per-fold summary

Example:
    python run_loo_experiment.py \\
        --study_years 1999 2007 \\
        --data_dir /path/to/processed_data \\
        --results_dir /path/to/results \\
        --model_type rf
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import root_mean_squared_error

from models import YieldModel, SUPPORTED_MODELS
from data_loader import load_and_prepare, FEATURE_MODES, TARGET_OPTIONS, SUPPORTED_CROPS

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


# Models selectable on the CLI: the sklearn tabular models plus the PatchTST
# sequence model ("tst"), which uses a separate (sequence) data path.
MODEL_CHOICES = SUPPORTED_MODELS + ["tst"]

# Targets valid for the tabular (sklearn) and tst (PatchTST) paths respectively.
# yield_pct_change is a tabular-only derived target; log_yield is tst-only.
TABULAR_TARGETS = TARGET_OPTIONS              # ["yield", "yield_pct_change"]
TST_TARGETS     = ["yield", "log_yield"]
ALL_TARGETS     = TARGET_OPTIONS + ["log_yield"]


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
            "'yield_pct_change': predict percent change vs. prior year (tabular models only). "
            "'log_yield': predict log(yield), reported in original units (tst only). "
            "Default: yield."
        ),
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

    # ── data source (used by both paths) ──
    parser.add_argument("--climate_model", type=str, default="EC-Earth3",
                        help="Climate model subdir/file prefix. Default: EC-Earth3.")
    parser.add_argument("--ens_member", type=str, default="r1i4p1f1",
                        help="Ensemble member tag in climate filenames. Default: r1i4p1f1.")
    parser.add_argument("--aoi", type=str, default="CONUS",
                        help="Area-of-interest tag in the crop filename "
                             "(crop_<crop>_<aoi>.parquet). E.g. 'CONUS' or 'germany'. "
                             "tst path only. Default: CONUS.")
    parser.add_argument("--use_phenology", type=_parse_phenology_arg, default=None,
                        metavar="VARS",
                        help="tst path only. Comma-separated CY-Bench phenology "
                             "channels to append to the climate sequence (off by "
                             f"default). Choices: {','.join(PHENOLOGY_VARS)}. E.g. "
                             "--use_phenology ndvi,fpar. Restricts samples to "
                             "county-years with that data (NDVI/FPAR 2001+, soil "
                             "moisture 2003+).")
    parser.add_argument("--phenology_suffix", type=str, default=None,
                        help="AOI suffix in phenology filenames "
                             "(phenology_<var>_<suffix>.parquet). Defaults to --aoi.")

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


def _finalize_fold(meta_te, y_true, y_pred, all_years, test_year, model_type,
                   fold_dir, n_train, extra_meta=None):
    """Build the per-fold error table + metadata, write outputs, and log a summary.

    Shared by the tabular and PatchTST fold runners so both emit identical
    adm2_errors.csv / adm1_rmse.csv / run_meta.json (plot_results.py compatible).

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
        "run_id":      f"fold_{test_year}",
        "test_year":   test_year,
        "train_years": [y for y in all_years if y != test_year],
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
    print(
        f"  fold {test_year}: RMSE={rmse:>10,.0f}  {skill_str}"
        f"  n_train={n_train}  n_test={len(y_true)}"
    )
    return fold_meta, errors


def run_fold(adm2_df, feature_cols, all_years, test_year, model_type, fold_dir):
    """Train a tabular model on all years except test_year; evaluate on test_year."""
    train_years = [y for y in all_years if y != test_year]
    meta_cols   = ["adm2_id", "adm2", "adm1", "hindcast_start"]

    train_mask = adm2_df["hindcast_start"].isin(train_years)
    test_mask  = adm2_df["hindcast_start"] == test_year

    X_tr    = adm2_df.loc[train_mask, feature_cols]
    y_tr    = adm2_df.loc[train_mask, "yield"]
    X_te    = adm2_df.loc[test_mask,  feature_cols]
    y_te    = adm2_df.loc[test_mask,  "yield"]
    meta_te = adm2_df.loc[test_mask,  meta_cols].copy()

    if len(X_te) == 0:
        print(f"  fold {test_year}: no test data — skipping")
        return None

    model = YieldModel(model_type)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    extra_meta = {
        "feature_importances": (
            dict(zip(feature_cols, model.feature_importances_.tolist()))
            if model.feature_importances_ is not None else None
        ),
    }
    return _finalize_fold(
        meta_te, y_te.values, y_pred, all_years, test_year, model_type,
        fold_dir, n_train=len(X_tr), extra_meta=extra_meta,
    )


def run_fold_tst(sequences, targets, sample_meta, all_years, test_year, fold_dir, args):
    """Train PatchTST on all years except test_year; evaluate on test_year."""
    _, _, train_predict_patchtst = _import_tst()
    meta_cols  = ["adm2_id", "adm2", "adm1", "hindcast_start"]
    years      = sample_meta["hindcast_start"].values
    train_mask = years != test_year
    test_mask  = years == test_year

    if test_mask.sum() == 0:
        print(f"  fold {test_year}: no test data — skipping")
        return None

    train_seqs, train_tgts = sequences[train_mask], targets[train_mask]
    val_seqs,   val_tgts   = sequences[test_mask],  targets[test_mask]
    meta_te = sample_meta.loc[test_mask, meta_cols].copy()

    log_target = args.target == "log_yield"
    y_pred = train_predict_patchtst(
        train_seqs, train_tgts, val_seqs, val_tgts,
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
    )

    extra_meta = {
        "target":       args.target,
        "patch_length": args.patch_length,
        "patch_stride": args.patch_stride,
        "d_model":      args.d_model,
        "num_layers":   args.num_layers,
        "num_heads":    args.num_heads,
        "ffn_dim":      args.ffn_dim,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
    }
    return _finalize_fold(
        meta_te, val_tgts.squeeze(), y_pred, all_years, test_year, "tst",
        fold_dir, n_train=int(train_mask.sum()), extra_meta=extra_meta,
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
    if args.model_type != "tst" and args.target not in TABULAR_TARGETS:
        raise ValueError(
            f"--target {args.target!r} is not valid for tabular model "
            f"{args.model_type!r}. Choose from {TABULAR_TARGETS}."
        )
    if args.use_phenology and args.model_type != "tst":
        raise ValueError(
            "--use_phenology is only supported by --model_type tst "
            f"(got {args.model_type!r})."
        )

    pheno_tag = ("_pheno-" + "-".join(args.use_phenology)) if args.use_phenology else ""
    exp_id  = (
        f"loo_{args.crop}_{args.model_type}_{args.feature_mode}_{args.target}{pheno_tag}"
        f"_{study_start}_{study_end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = os.path.join(args.results_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment      : {exp_id}")
    print(f"Crop            : {args.crop}")
    print(f"Study years     : {study_start}–{study_end}  ({len(all_years)} folds)")
    print(f"Model           : {args.model_type}")
    print(f"Feature mode    : {args.feature_mode}")
    print(f"Target          : {args.target}")
    if args.model_type == "tst":
        print(f"Phenology       : {','.join(args.use_phenology) if args.use_phenology else 'none (climate only)'}")
    print(f"Results dir     : {exp_dir}")

    # Load data once — reused across all folds. The PatchTST ("tst") path uses
    # raw daily climate sequences; tabular models use featurised county-year rows.
    print("\nLoading data...")
    use_tst = args.model_type == "tst"
    adm2_df = feature_cols = None
    sequences = targets = sample_meta = None
    if use_tst:
        load_climate_crop_data, build_sequence_array, _ = _import_tst()
        from tst_implementation import CLIMATE_CHANNELS, phenology_channel_cols
        daily_df = load_climate_crop_data(
            data_dir=args.data_dir,
            hindcast_years=all_years,
            crop=args.crop,
            climate_model=args.climate_model,
            ens_member=args.ens_member,
            aoi=args.aoi,
            use_phenology=args.use_phenology,
            phenology_suffix=args.phenology_suffix,
        )
        channels = CLIMATE_CHANNELS + phenology_channel_cols(args.use_phenology)
        sequences, targets, sample_meta = build_sequence_array(daily_df, channels=channels)
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
        )
        county_meta = adm2_df[["adm2_id", "adm2", "adm1"]].drop_duplicates("adm2_id")

    # Optional county subsetting (region / random seed / dispersed sample)
    selected, region = _select_counties(county_meta, args)
    if use_tst:
        keep        = sample_meta["adm2_id"].isin(selected).values
        sequences   = sequences[keep]
        targets     = targets[keep]
        sample_meta = sample_meta[keep].reset_index(drop=True)
        available   = set(sample_meta["hindcast_start"].unique())
        print(f"Counties        : {sample_meta['adm2_id'].nunique()}  "
              f"({len(sample_meta)} county-year samples)")
    else:
        adm2_df   = adm2_df[adm2_df["adm2_id"].isin(selected)].copy()
        available = set(adm2_df["hindcast_start"].unique())
        print(f"Counties        : {adm2_df['adm2_id'].nunique()}  "
              f"({len(adm2_df)} county-year rows)")

    missing = [y for y in all_years if y not in available]
    if missing:
        print(f"WARNING: no parquet data found for years: {missing}")

    # LOO loop
    print(f"\nRunning {len(all_years)} LOO folds...\n")
    all_errors_list = []
    fold_metas      = []

    for test_year in all_years:
        fold_dir = os.path.join(exp_dir, f"fold_{test_year}")
        if use_tst:
            result = run_fold_tst(sequences, targets, sample_meta, all_years,
                                  test_year, fold_dir, args)
        else:
            result = run_fold(adm2_df, feature_cols, all_years, test_year,
                              args.model_type, fold_dir)
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
        "crop":               args.crop,
        "target":             args.target,
        "region":             region,
        "sample_counties":    args.sample_counties,
        "sample_seed":        args.sample_seed,
        "preseason_buffer":   args.preseason_buffer,
        "postharvest_buffer": args.postharvest_buffer,
        "study_start":        study_start,
        "study_end":          study_end,
        "model_type":         args.model_type,
        "feature_mode":       args.feature_mode,
        "use_phenology":      args.use_phenology,
        "n_folds":            len(fold_metas),
        "overall_rmse":       overall_rmse,
        "overall_mae":        overall_mae,
        "folds":              fold_metas,
    }
    with open(os.path.join(exp_dir, "loo_meta.json"), "w") as f:
        json.dump(exp_meta, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"LOO experiment complete — {args.model_type.upper()} {study_start}–{study_end}")
    print(f"{'='*60}")
    single_sample_folds = any(fm["pct_error"] is not None for fm in fold_metas)
    skill_hdr = "err%  " if single_sample_folds else "R²    "
    print(f"{'Year':>6}  {'RMSE':>10}  {skill_hdr:>6}  {'n_test':>7}")
    for fm in fold_metas:
        if fm["pct_error"] is not None:
            skill_val = f"{fm['pct_error']:>+6.1f}%"
        else:
            skill_val = f"{fm['r2_overall']:>6.3f} "
        print(
            f"{fm['test_year']:>6}  {fm['rmse_overall']:>10,.0f}"
            f"  {skill_val}  {fm['n_test']:>7}"
        )
    print(f"{'':>6}  {'─'*10}  {'─'*6}")
    print(f"{'Overall':>6}  {overall_rmse:>10,.0f}  {'':>6}  (LOO RMSE across all folds)")
    print(f"{'':>6}  {overall_mae:>10,.0f}  {'':>6}  (LOO MAE  across all folds)")
    print(f"\nCounty RMSE saved to: {exp_dir}/loo_county_rmse.csv")
    print(f"All errors saved to:  {exp_dir}/loo_all_errors.csv")
    print(
        f"\nTo plot a single fold:\n"
        f"  python plot_results.py --save_dir {exp_dir}/fold_<year> [--county_shp ...]"
    )


if __name__ == "__main__":
    main()
