"""
Train and evaluate a yield prediction model with leave-one-out temporal split.

Each run gets a unique ID of the form  {model_type}_test{test_year}_{YYYYMMDD}_{HHMMSS}
and its outputs are saved to a subfolder of --results_dir with that name.

Saves per run:
    adm1_rmse.csv   — RMSE per state
    adm2_errors.csv — per-county predictions and errors
    run_meta.json   — run ID, parameters, and overall metrics (read by plot_results.py)

Example:
    python train_eval.py \
        --crop corn \
        --test_year 2005 \
        --study_years 2000 2010 \
        --data_dir /path/to/processed_data \
        --results_dir /path/to/results \
        --model_type rf

    → saves to /path/to/results/rf_test2005_20260512_143022/
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from models import YieldModel, SUPPORTED_MODELS
from data_loader import load_and_prepare, FEATURE_MODES, TARGET_OPTIONS, SUPPORTED_CROPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a yield model with leave-one-out temporal split."
    )
    parser.add_argument(
        "--crop", type=str, choices=SUPPORTED_CROPS, default="corn",
        help=f"Crop to model. Choices: {SUPPORTED_CROPS}. Default: corn.",
    )
    parser.add_argument(
        "--test_year", type=int, required=True,
        help="Hindcast start year to hold out as the test set (e.g. 2005).",
    )
    parser.add_argument(
        "--study_years", type=int, nargs=2, metavar=("START", "END"), required=True,
        help=(
            "Inclusive start and end of the study period (e.g. --study_years 2000 2010). "
            "All years except --test_year are used for training."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing the processed parquet files.",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Root results directory. A unique run subfolder will be created inside it.",
    )
    parser.add_argument(
        "--model_type", type=str, choices=SUPPORTED_MODELS, default="rf",
        help=f"Model type. Choices: {SUPPORTED_MODELS}. Default: rf.",
    )
    parser.add_argument(
        "--feature_mode", type=str, choices=FEATURE_MODES, default="pixel_blocks",
        help=(
            "'pixel': whole-season totals/means from era5_features.py stage 3. "
            "'pixel_blocks': the same, split into season-relative windows. Both "
            "read a design matrix whose features were computed per pixel and "
            "only then averaged over cropland. Default: pixel_blocks."
        ),
    )
    parser.add_argument(
        "--target", type=str, choices=TARGET_OPTIONS, default="yield",
        help=(
            "'yield': predict absolute yield. "
            "'yield_pct_change': predict percent change vs. prior year. "
            "Default: yield."
        ),
    )
    return parser.parse_args()


def aggregate_daily(daily_df: pd.DataFrame):
    """Aggregate a daily county×day DataFrame to one row per (adm2_id, hindcast_start).

    Returns (adm2_df, weather_cols) where weather_cols is the list of raw weather
    column names used to build mean/std features.
    """
    df = daily_df.copy()
    df["temp_range_C"] = df["tasmax_C"] - df["tasmin_C"]
    weather_cols = ["tasmax_C", "tasmin_C", "hursmin", "pr_mm", "temp_range_C"]

    agg_dict = {}
    for col in weather_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"]  = (col, "std")
    agg_dict["pr_mm_sum"]       = ("pr_mm",           "sum")
    agg_dict["plant_doy"]       = ("plant_doy",        "first")
    agg_dict["harvest_doy"]     = ("harvest_doy",      "first")
    agg_dict["harvest_end_doy"] = ("harvest_end_doy",  "first")
    agg_dict["yield"]           = ("yield",            "first")
    agg_dict["adm2"]            = ("adm2",             "first")
    agg_dict["adm1"]            = ("adm1",             "first")

    adm2_df = (
        df.groupby(["adm2_id", "hindcast_start"])
        .agg(**agg_dict)
        .reset_index()
        .dropna(subset=["yield"])
    )
    print(f"(adm2, year) pairs with yield data: {len(adm2_df)}")
    return adm2_df, weather_cols



def build_features(adm2_df, weather_cols):
    feature_cols = (
        [f"{c}_mean" for c in weather_cols]
        + [f"{c}_std"  for c in weather_cols]
        + ["pr_mm_sum", "plant_doy", "harvest_doy", "harvest_end_doy"]
    )
    mask    = adm2_df[feature_cols].notna().all(axis=1)
    adm2_df = adm2_df[mask].copy()
    print(f"Samples after dropping NaN features: {len(adm2_df)}")
    return adm2_df, feature_cols


def main():
    args = parse_args()

    study_start, study_end = args.study_years
    if not (study_start <= args.test_year <= study_end):
        raise ValueError(
            f"--test_year {args.test_year} must lie within "
            f"--study_years {study_start}–{study_end}."
        )

    train_years = [y for y in range(study_start, study_end + 1) if y != args.test_year]
    test_year   = args.test_year

    run_id  = f"{args.model_type}_test{test_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run ID       : {run_id}")
    print(f"Run dir      : {run_dir}")
    print(f"Crop         : {args.crop}")
    print(f"Study years  : {study_start}–{study_end}")
    print(f"Train years  : {train_years}")
    print(f"Test year    : {test_year}")
    print(f"Model type   : {args.model_type}")
    print(f"Feature mode : {args.feature_mode}")
    print(f"Target       : {args.target}")

    adm2_df, feature_cols = load_and_prepare(
        crop=args.crop,
        data_dir=args.data_dir,
        feature_mode=args.feature_mode,
        target=args.target,
    )

    train_mask = adm2_df["hindcast_start"].isin(train_years)
    test_mask  = adm2_df["hindcast_start"] == test_year
    meta_cols  = ["adm2_id", "adm2", "adm1", "hindcast_start"]

    X_tr = adm2_df.loc[train_mask, feature_cols]
    y_tr = adm2_df.loc[train_mask, "yield"]

    X_te    = adm2_df.loc[test_mask, feature_cols]
    y_te    = adm2_df.loc[test_mask, "yield"]
    meta_te = adm2_df.loc[test_mask, meta_cols].copy()

    print(f"\nLeave-one-out split:")
    print(f"  Train: {min(train_years)}–{max(train_years)} (excl. {test_year})  n={len(X_tr)}")
    print(f"  Test:  hindcast_start={test_year}  n={len(X_te)}")
    if len(X_te) == 0:
        raise ValueError(
            f"No test samples for hindcast_start={test_year}. "
            "Check that a matching parquet file exists."
        )

    model = YieldModel(args.model_type)
    model.fit(X_tr, y_tr)

    y_pred_te    = model.predict(X_te)
    rmse_overall = root_mean_squared_error(y_te, y_pred_te)
    r2_overall   = model.score(X_te, y_te)
    mean_yield   = float(y_te.mean())

    print(f"\n{'='*60}")
    print(
        f"LOO SPLIT [{args.model_type.upper()}] — "
        f"train {min(train_years)}–{max(train_years)} (excl. {test_year}), "
        f"test {test_year}"
    )
    print(f"  Overall RMSE : {rmse_overall:,.0f}")
    print(f"  R²           : {r2_overall:.3f}")
    print(
        f"  Mean yield   : {mean_yield:,.0f}  "
        f"(RMSE%: {100 * rmse_overall / mean_yield:.1f}%)"
    )
    print(f"{'='*60}")

    errors = meta_te.copy()
    errors["y_true"]    = y_te.values
    errors["y_pred"]    = y_pred_te
    errors["residual"]  = y_pred_te - y_te.values
    errors["abs_error"] = np.abs(errors["residual"])
    errors["sq_error"]  = errors["residual"] ** 2

    adm1_rmse = (
        errors.groupby("adm1")["sq_error"]
        .mean()
        .pipe(np.sqrt)
        .rename("rmse")
        .reset_index()
        .assign(rmse_pct=lambda d: 100 * d["rmse"] / mean_yield)
        .sort_values("rmse", ascending=False)
    )

    print(f"\nRMSE by state (adm1) — hindcast_start={test_year} (yield year {test_year + 1}):")
    print(adm1_rmse.to_string(index=False, float_format="{:,.1f}".format))

    adm1_rmse.to_csv(f"{run_dir}/adm1_rmse.csv", index=False)
    errors.to_csv(f"{run_dir}/adm2_errors.csv", index=False)

    feature_importances = None
    if model.feature_importances_ is not None:
        feature_importances = dict(zip(feature_cols, model.feature_importances_.tolist()))

    run_meta = {
        "run_id":               run_id,
        "crop":                 args.crop,
        "target":               args.target,
        "test_year":            test_year,
        "train_years":          train_years,
        "study_start":          study_start,
        "study_end":            study_end,
        "model_type":           args.model_type,
        "rmse_overall":         float(rmse_overall),
        "r2_overall":           float(r2_overall),
        "mean_yield":           mean_yield,
        "n_train":              int(len(X_tr)),
        "n_test":               int(len(X_te)),
        "feature_importances":  feature_importances,
    }
    with open(f"{run_dir}/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\nAll outputs saved to {run_dir}/")
    print(f"Run plot_results.py --save_dir {run_dir} to generate plots.")


if __name__ == "__main__":
    main()
