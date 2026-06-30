"""
Load, merge, and featurise climate + crop parquets for yield model training.

Expected files in data_dir:
  Climate:  climate_{climate_model}_s{hindcast_start}_{ens_member}.parquet
  Crop:     crop_{crop}_CONUS.parquet

Two feature modes
-----------------
aggregated
    One row per (adm2_id, hindcast_start).
    Features = mean, std, and (for pr_mm) sum of each weather variable
    computed over the seasonal window using date arithmetic:
        [plant_start_date − preseason_buffer days,
         harvest_end_date + postharvest_buffer days]

timeseries
    One row per (adm2_id, hindcast_start).
    Features = daily values of each weather variable from Nov 1 of the
    hindcast_start year through Nov 1 of the yield year (~365 columns per
    variable), named  {var}_d{NNN:03d}  (NNN = 1-based day offset from Nov 1).
    Missing days are forward-filled then backward-filled within each
    county-year so no NaN values remain in the feature block.

Public API
----------
load_and_prepare(...)      → (adm2_df, feature_cols)   ready for sklearn
load_crop_climate_data(…)  → daily DataFrame            raw merged daily data
aggregate_daily(daily_df)  → (adm2_df, weather_cols)   seasonal aggregation
build_features(adm2_df, …) → (adm2_df, feature_cols)   drop-NaN step
build_timeseries_features  → (adm2_df, feature_cols)   pivoted daily features
available_crops(data_dir)  → list[str]
"""

import glob
import os
import re

import pandas as pd

WEATHER_VARS    = ["tasmax_C", "tasmin_C", "hursmin", "pr_mm"]
FEATURE_MODES   = ["aggregated", "timeseries"]
TARGET_OPTIONS  = ["yield", "yield_pct_change"]
SUPPORTED_CROPS = ["corn", "cassava", "rice", "sorghum", "soy", "wheat"]


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------

def _apply_target(crop_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Overwrite the 'yield' column with the requested target variable.

    yield            — no change (raw yield values)
    yield_pct_change — percent change vs. previous hindcast_start year for the
                       same county: (yield_t - yield_{t-1}) / yield_{t-1} * 100.
                       Rows with no prior year become NaN and are dropped later
                       by .dropna(subset=["yield"]).
    """
    if target not in TARGET_OPTIONS:
        raise ValueError(f"target must be one of {TARGET_OPTIONS}")
    if target == "yield":
        return crop_df
    df = crop_df.sort_values(["adm2_id", "hindcast_start"]).copy()
    prev = df.groupby("adm2_id")["yield"].shift(1)
    df["yield"] = (df["yield"] - prev) / prev * 100
    return df


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _parse_hindcast_year(filepath: str) -> int:
    name = os.path.basename(filepath)
    m = re.search(r"_s(\d{4})", name)
    if not m:
        raise ValueError(f"Cannot parse hindcast year from filename: {name!r}")
    return int(m.group(1))


def available_crops(data_dir: str) -> list[str]:
    """List crop names that have a crop_{name}_CONUS.parquet in data_dir."""
    files = glob.glob(os.path.join(data_dir, "crop_*_CONUS.parquet"))
    crops = []
    for f in files:
        m = re.search(r"crop_(.+)_CONUS\.parquet", os.path.basename(f))
        if m:
            crops.append(m.group(1))
    return sorted(crops)


# ---------------------------------------------------------------------------
# Raw loading + merging
# ---------------------------------------------------------------------------

def load_crop_climate_data(
    crop: str,
    data_dir: str,
    climate_model: str = "EC-Earth3",
    ens_member: str = 'r1i4p1f1',
    preseason_buffer: int = 60,
    postharvest_buffer: int = 14,
    hindcast_years: list | None = None,
    feature_mode: str = "aggregated",
    target: str = "yield",
) -> pd.DataFrame:
    """Load and merge climate + crop parquets; apply the appropriate date window.

    Aggregated mode window (date-based, year-boundary safe):
        [plant_start_date − preseason_buffer,  harvest_end_date + postharvest_buffer]

    Timeseries mode window:
        [Nov 1 of hindcast_start year,  Nov 1 of yield year]

    Returns a daily DataFrame with columns:
        adm2_id, adm2, adm1, hindcast_start, yield_year, date, doy,
        year_month, tasmax_C, tasmin_C, hursmin, pr_mm,
        plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")

    # ---- climate ----
    pattern = os.path.join(data_dir, f"{climate_model}/climate_{climate_model}_s*-{ens_member}.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No climate parquets found matching {pattern!r}")
    if hindcast_years is not None:
        hset  = set(hindcast_years)
        files = [f for f in files if _parse_hindcast_year(f) in hset]
        if not files:
            raise FileNotFoundError(
                f"No climate parquets matched hindcast_years={hindcast_years}"
            )

    print(f"Loading {len(files)} climate parquet(s)...")
    climate_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    climate_df["date"] = pd.to_datetime(climate_df["date"])
    print(
        f"  Climate rows: {len(climate_df):,}  "
        f"hindcast_start years: {sorted(climate_df['hindcast_start'].unique())}"
    )

    # ---- crop ----
    crop_file = os.path.join(data_dir, f"crop_{crop}_CONUS.parquet")
    if not os.path.exists(crop_file):
        raise FileNotFoundError(
            f"Crop parquet not found: {crop_file!r}\n"
            f"Available crops: {available_crops(data_dir)}"
        )
    crop_df = pd.read_parquet(crop_file)
    crop_df = _apply_target(crop_df, target)
    print(
        f"  Crop ({crop}) rows: {len(crop_df):,}  "
        f"hindcast_start years: {sorted(crop_df['hindcast_start'].unique())}"
    )

    # ---- merge ----
    merged = climate_df.merge(
        crop_df[["adm2_id", "hindcast_start",
                 "plant_start_doy", "plant_doy",
                 "harvest_doy", "harvest_end_doy", "yield"]],
        on=["adm2_id", "hindcast_start"],
        how="inner",
    )
    merged["yield_year"] = merged["hindcast_start"] + 1

    # ---- date-based window (year-boundary safe) ----
    # plant_start_doy and harvest_end_doy are DOYs in the yield year, so we
    # anchor them to Jan 1 of the yield year before applying buffers.
    yield_yr_jan1 = pd.to_datetime(merged["yield_year"].astype(str) + "-01-01")

    if feature_mode == "aggregated":
        window_start = yield_yr_jan1 + pd.to_timedelta(
            merged["plant_start_doy"] - preseason_buffer - 1, unit="D"
        )
        window_end = yield_yr_jan1 + pd.to_timedelta(
            merged["harvest_end_doy"] + postharvest_buffer - 1, unit="D"
        )
    else:  # timeseries: full Nov-to-Nov window
        window_start = pd.to_datetime(
            merged["hindcast_start"].astype(str) + "-11-01"
        )
        window_end = pd.to_datetime(
            merged["yield_year"].astype(str) + "-11-01"
        )

    mask   = (merged["date"] >= window_start) & (merged["date"] <= window_end)
    merged = merged[mask].copy().reset_index(drop=True)

    print(
        f"  After merge + {feature_mode} window: {len(merged):,} rows  "
        f"({merged['adm2_id'].nunique()} counties)"
    )
    return merged


# ---------------------------------------------------------------------------
# Aggregated features
# ---------------------------------------------------------------------------

def aggregate_daily(daily_df: pd.DataFrame):
    """Aggregate daily county×day data to one row per (adm2_id, hindcast_start).

    Returns (adm2_df, weather_cols).
    """
    df = daily_df.copy()
    df["temp_range_C"] = df["tasmax_C"] - df["tasmin_C"]
    weather_cols = ["tasmax_C", "tasmin_C", "hursmin", "pr_mm", "temp_range_C"]

    agg = {}
    for col in weather_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"]  = (col, "std")
    agg["pr_mm_sum"]       = ("pr_mm",           "sum")
    agg["plant_doy"]       = ("plant_doy",        "first")
    agg["harvest_doy"]     = ("harvest_doy",      "first")
    agg["harvest_end_doy"] = ("harvest_end_doy",  "first")
    agg["yield"]           = ("yield",            "first")
    agg["adm2"]            = ("adm2",             "first")
    agg["adm1"]            = ("adm1",             "first")

    adm2_df = (
        df.groupby(["adm2_id", "hindcast_start"])
        .agg(**agg)
        .reset_index()
        .dropna(subset=["yield"])
    )
    print(f"(adm2, year) pairs with yield data: {len(adm2_df)}")
    return adm2_df, weather_cols


def build_features(adm2_df: pd.DataFrame, weather_cols: list):
    """Drop rows with any NaN feature; return (adm2_df, feature_cols)."""
    feature_cols = (
        [f"{c}_mean" for c in weather_cols]
        + [f"{c}_std"  for c in weather_cols]
        + ["pr_mm_sum", "plant_doy", "harvest_doy", "harvest_end_doy"]
    )
    mask    = adm2_df[feature_cols].notna().all(axis=1)
    adm2_df = adm2_df[mask].copy()
    print(f"Samples after dropping NaN features: {len(adm2_df)}")
    return adm2_df, feature_cols


# ---------------------------------------------------------------------------
# Timeseries features
# ---------------------------------------------------------------------------

def build_timeseries_features(
    daily_df: pd.DataFrame,
    weather_vars: list = WEATHER_VARS,
) -> tuple[pd.DataFrame, list]:
    """Pivot daily data to one row per (adm2_id, hindcast_start).

    Column names: {var}_d{NNN:03d}  where NNN is the 1-based day offset
    from Nov 1 of the hindcast_start year (so d001 = Nov 1 of hindcast year,
    d366 ≈ Nov 1 of yield year).

    Missing days within each county-year are forward-filled then backward-
    filled so no NaN values remain in the feature block (RF-compatible).

    Returns (adm2_df, feature_cols).
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    season_start    = pd.to_datetime(df["hindcast_start"].astype(str) + "-11-01")
    df["day_offset"] = (df["date"] - season_start).dt.days + 1

    # Pivot each variable then concatenate column-wise
    pivots = []
    for var in weather_vars:
        piv = (
            df.pivot_table(
                index=["adm2_id", "hindcast_start"],
                columns="day_offset",
                values=var,
                aggfunc="mean",
            )
            .rename(columns=lambda d: f"{var}_d{int(d):03d}")
        )
        pivots.append(piv)

    ts_wide = pd.concat(pivots, axis=1)

    # Fill gaps within each county-year's series
    ts_wide = ts_wide.ffill(axis=1).bfill(axis=1)

    feature_cols = list(ts_wide.columns)
    ts_wide      = ts_wide.reset_index()

    # Attach metadata and yield
    meta = (
        df[["adm2_id", "hindcast_start", "adm2", "adm1", "yield"]]
        .drop_duplicates(subset=["adm2_id", "hindcast_start"])
    )
    adm2_df = (
        meta.merge(ts_wide, on=["adm2_id", "hindcast_start"], how="inner")
        .dropna(subset=["yield"])
        .reset_index(drop=True)
    )

    print(
        f"Timeseries features: {len(feature_cols)} columns  "
        f"({len(adm2_df)} county-year pairs)"
    )
    return adm2_df, feature_cols


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_and_prepare(
    crop: str,
    data_dir: str,
    feature_mode: str = "aggregated",
    climate_model: str = "EC-Earth3",
    ens_member: str = "r1i4p1f1",
    preseason_buffer: int = 60,
    postharvest_buffer: int = 14,
    hindcast_years: list | None = None,
    target: str = "yield",
) -> tuple[pd.DataFrame, list]:
    """Full pipeline: load → merge → window → featurise.

    Returns (adm2_df, feature_cols) where adm2_df has one row per
    (adm2_id, hindcast_start) with columns:
        adm2_id, adm2, adm1, hindcast_start, yield, <feature_cols>
    """
    daily_df = load_crop_climate_data(
        crop=crop,
        data_dir=data_dir,
        climate_model=climate_model,
        ens_member=ens_member,
        preseason_buffer=preseason_buffer,
        postharvest_buffer=postharvest_buffer,
        hindcast_years=hindcast_years,
        feature_mode=feature_mode,
        target=target,
    )

    if feature_mode == "aggregated":
        adm2_df, weather_cols = aggregate_daily(daily_df)
        adm2_df, feature_cols = build_features(adm2_df, weather_cols)
    else:
        adm2_df, feature_cols = build_timeseries_features(daily_df)

    return adm2_df, feature_cols
