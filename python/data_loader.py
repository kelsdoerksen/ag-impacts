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
parse_climate_vars(str)    → list[str]                 --climate_vars parsing
history_leak_mask(...)     → ndarray (N, L) bool       held-out labels in history
draw_seed()                → int                       fresh OS-entropy run seed
"""

import glob
import os
import re
import secrets

import numpy as np
import pandas as pd

# Upper bound for drawn seeds: numpy's legacy seed API and sklearn's
# random_state both cap at 2**32-1, and torch/transformers accept anything in
# that range, so a 31-bit draw is safe everywhere the seed is threaded.
SEED_MAX = 2 ** 31 - 1


def draw_seed() -> int:
    """Draw a fresh run seed from OS entropy.

    Used when no --seed is given so repeated launches of the same configuration
    are genuinely independent replicates. The drawn value is always recorded in
    the run's diagnostics, so any run can be reproduced exactly by passing it
    back via --seed.
    """
    return secrets.randbelow(SEED_MAX) + 1

WEATHER_VARS    = ["tasmax_C", "tasmin_C", "hursmin", "pr_mm"]
FEATURE_MODES   = ["aggregated", "timeseries"]
TARGET_OPTIONS  = ["yield", "yield_pct_change"]
SUPPORTED_CROPS = ["corn", "cassava", "rice", "sorghum", "soy", "wheat"]

# Short CLI tokens for the daily climate variables, so a feature-space ablation
# can select a subset (e.g. --climate_vars tasmax,tasmin,pr = temperature +
# precipitation only, dropping relative humidity). Order of the resolved list
# determines tabular feature order and TST channel index order.
CLIMATE_VAR_ALIASES = {
    "tasmax":   "tasmax_C",
    "tasmin":   "tasmin_C",
    "hursmin":  "hursmin",
    "pr":       "pr_mm",
    # full column names accepted too, so --climate_vars tasmax_C,pr_mm works
    **{v: v for v in WEATHER_VARS},
}

# CY-Bench phenology variables: token -> parquet file stem + value column(s).
# Files live at {data_dir}/phenology/phenology_{stem}_{suffix}.parquet. Canonical
# here (pandas-only) so both the tabular pipeline and the TST path share one map.
PHENOLOGY_CHANNELS = {
    "ndvi":          {"stem": "ndvi",         "cols": ["ndvi"]},
    "fpar":          {"stem": "fpar",         "cols": ["fpar"]},
    "soil_moisture": {"stem": "soilmoisture", "cols": ["ssm", "rsm"]},
}


# ---------------------------------------------------------------------------
# Climate variable selection
# ---------------------------------------------------------------------------

def parse_climate_vars(s: str) -> list[str]:
    """Parse a comma-separated --climate_vars value into full column names.

    Accepts short tokens ('tasmax,tasmin,pr') or full column names
    ('tasmax_C,tasmin_C,pr_mm'). Duplicates are collapsed, order is preserved.
    """
    import argparse
    out: list[str] = []
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        if tok not in CLIMATE_VAR_ALIASES:
            raise argparse.ArgumentTypeError(
                f"Unknown climate variable {tok!r}. "
                f"Choices: {sorted(set(CLIMATE_VAR_ALIASES))}"
            )
        col = CLIMATE_VAR_ALIASES[tok]
        if col not in out:
            out.append(col)
    if not out:
        raise argparse.ArgumentTypeError("--climate_vars must name at least one variable.")
    return out


def resolve_climate_vars(climate_vars: list | None) -> list[str]:
    """Return the climate columns to use, defaulting to all of WEATHER_VARS."""
    if not climate_vars:
        return list(WEATHER_VARS)
    unknown = [v for v in climate_vars if v not in WEATHER_VARS]
    if unknown:
        raise ValueError(
            f"Unknown climate variable(s) {unknown}. Choices: {WEATHER_VARS}"
        )
    return list(climate_vars)


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

def _read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet, falling back to fastparquet when the default engine fails.

    The FY2 climate parquets are written by a newer Arrow (R arrow 23.x) whose
    page-index format the installed pyarrow (19.x) cannot decode ("Repetition
    level histogram size mismatch"). fastparquet reads them fine, so we retry
    with it rather than requiring an environment upgrade.
    """
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            raise exc


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
# Prior-yield history (optional autoregressive feature)
# ---------------------------------------------------------------------------
#
# For a prediction at (county, target_year T), the "yield history" is the
# sequence of that county's own observed yields for the calendar years before T,
# ordered by recency: lag 1 = year T-1, lag 2 = T-2, … back to a fixed base year.
# This is shared by the tabular pipeline (as lag feature columns) and the TST
# pipeline (as a second time-series branch), so both see identical history.

def load_county_yield_series(data_dir: str, crop: str, aoi: str = "CONUS") -> pd.DataFrame:
    """Return one row per (county, calendar year) of observed yield.

    Reads the full crop parquet (all years, unfiltered by any study window) so
    the history can reach back before the experiment's study period. The crop
    parquet is keyed by hindcast_start under the FY1 convention, i.e. its yield
    is for calendar year hindcast_start + 1; we surface that as `cal_year`.
    """
    path = os.path.join(data_dir, f"crop_{crop}_{aoi}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Crop parquet not found for yield history: {path!r}. Check --crop/--aoi."
        )
    df = _read_parquet(path)[["adm2_id", "hindcast_start", "yield"]].copy()
    df = df.dropna(subset=["yield"])
    df["cal_year"] = df["hindcast_start"].astype(int) + 1
    return (
        df[["adm2_id", "cal_year", "yield"]]
        .rename(columns={"yield": "yield_obs"})
        .drop_duplicates(["adm2_id", "cal_year"])
        .reset_index(drop=True)
    )


def build_yield_history_matrix(
    meta: pd.DataFrame,
    yield_series: pd.DataFrame,
    base_year: int | None = None,
    max_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build recency-aligned prior-yield vectors for a set of prediction samples.

    Parameters
    ----------
    meta : DataFrame
        One row per sample; must have columns `adm2_id` and `target_year`
        (the calendar year being predicted). Row order is preserved.
    yield_series : DataFrame
        Output of load_county_yield_series (adm2_id, cal_year, yield_obs).
    base_year : int, optional
        Oldest calendar year to include. Defaults to the earliest year present
        in yield_series (so the history spans "all years since data start").
    max_len : int, optional
        Fixed history length L (columns). Defaults to (max target_year - base),
        so the latest-year sample fills every slot and earlier ones are padded.

    Returns
    -------
    hist : ndarray (N, L) float32
        hist[n, j] = county's observed yield for calendar year target_year-1-j,
        or 0 where that year is < base_year or has no observation.
    mask : ndarray (N, L) float32
        1 where hist[n, j] is a real observation, else 0.
    base_year, max_len : the values actually used.
    """
    targets = meta["target_year"].astype(int).to_numpy()
    aids    = meta["adm2_id"].to_numpy()

    if base_year is None:
        base_year = int(yield_series["cal_year"].min())
    if max_len is None:
        max_len = int(targets.max()) - base_year if len(targets) else 0
    max_len = max(int(max_len), 0)

    ymap = {
        (r.adm2_id, r.cal_year): r.yield_obs
        for r in yield_series.itertuples(index=False)
    }

    n = len(meta)
    hist = np.zeros((n, max_len), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    for i in range(n):
        t, aid = int(targets[i]), aids[i]
        for j in range(max_len):
            year = t - 1 - j
            if year < base_year:
                break
            v = ymap.get((aid, year))
            if v is not None:
                hist[i, j] = np.float32(v)
                mask[i, j] = 1.0
    return hist, mask, base_year, max_len


def history_leak_mask(
    adm2_ids,
    target_years,
    hist_len: int,
    blocked_labels,
) -> np.ndarray:
    """Flag prior-yield slots that would expose a held-out fold's labels.

    Under leave-one-year-out CV the held-out year's yields are still visible to
    the model through *other* samples' history: a training sample for
    (county c, target year T+1) carries lag-1 = county c's yield in year T,
    which is precisely the label of the held-out sample for (c, T). The model
    therefore trains on the test targets. This finds those slots so the caller
    can zero them (and their validity mask) before training.

    Parameters
    ----------
    adm2_ids : array-like (N,)  — county id per sample (training rows).
    target_years : array-like (N,) — calendar year predicted by each sample.
    hist_len : int — history length L; slot j holds year target_year - 1 - j.
    blocked_labels : iterable of (adm2_id, target_year)
        The (county, year) labels held out by this fold.

    Returns
    -------
    ndarray (N, L) bool — True where the slot must be suppressed.

    This covers the *training* side only: held-out labels must never be model
    inputs during fitting. Test samples keep their own history, which is the
    intended behaviour — at real forecast time a county's past observed yields
    are known. Under --cv_scheme cell the held-out unit is a set of counties and
    history is county-own, so no training entry is ever flagged (a held-out
    county's yields appear only in that county's own, held-out, samples).
    """
    n = len(adm2_ids)
    out = np.zeros((n, hist_len), dtype=bool)
    blocked = list(blocked_labels)
    if n == 0 or hist_len == 0 or not blocked:
        return out

    # Encode (county, year) as a single integer key so membership is one
    # vectorised np.isin over the (N, L) grid of history years.
    codes, uniq = pd.factorize(np.asarray(adm2_ids))
    code_of = {aid: i for i, aid in enumerate(uniq)}
    YEAR_SPAN = 10_000
    blocked_keys = np.array(
        [code_of[aid] * YEAR_SPAN + int(yr)
         for aid, yr in blocked if aid in code_of],
        dtype=np.int64,
    )
    if blocked_keys.size == 0:
        return out

    years = np.asarray(target_years, dtype=np.int64)[:, None] - 1 - np.arange(hist_len)
    keys  = codes.astype(np.int64)[:, None] * YEAR_SPAN + years
    return np.isin(keys, blocked_keys)


# ---------------------------------------------------------------------------
# Prior-season phenology (optional NDVI/FPAR/soil-moisture summary features)
# ---------------------------------------------------------------------------
#
# Phenology (e.g. NDVI) is a satellite observation that cannot be forecast, so
# for a prediction at target year T we use the PREVIOUS season's signal. Each
# phenology parquet's rows carry the season they belong to; we summarise every
# (county, season) into a handful of stats and, at feature time, attach the
# season `phenology_lag_years` before the target year. Mirrors the TST path,
# which feeds the same lagged signal as a daily channel.
#
# `phenology_lookback_years` widens that from one prior season to several
# consecutive ones (lags L, L+1, … L+n-1), so the model sees how the county's
# phenology has behaved over recent years rather than a single snapshot. Each
# season becomes its own set of columns/channels; day-of-season alignment is
# preserved, so lag-1 day d and lag-2 day d are the same point in the season.

_PHENOLOGY_STATS = ["mean", "std", "min", "max", "amp"]


def phenology_lags(lag_years: int, lookback_years: int = 1) -> list[int]:
    """Season lags to include: lag_years, lag_years+1, … (lookback_years total).

    lag_years is the most recent (smallest-lag) season — 1 by default, the
    forecast-safe "last season". A lookback of 4 with the default lag therefore
    covers the four seasons before the target year.
    """
    if lookback_years < 1:
        raise ValueError(
            f"phenology_lookback_years must be >= 1, got {lookback_years}."
        )
    if lag_years < 0:
        raise ValueError(f"phenology_lag_years must be >= 0, got {lag_years}.")
    return [lag_years + k for k in range(lookback_years)]


def phenology_lag_suffix(lag: int, lookback_years: int = 1) -> str:
    """Column/channel suffix identifying one season of a phenology lookback.

    Empty for a single-season lookback, so the original names (`ndvi`,
    `ndvi_prev_mean`, …) are unchanged when the option is not used.
    """
    return "" if lookback_years <= 1 else f"_lag{lag}"


def load_phenology_aggregates(
    data_dir: str, var: str, aoi: str = "CONUS", suffix: str | None = None,
) -> pd.DataFrame:
    """Summarise a phenology variable per (county, season target year).

    Returns one row per (adm2_id, season_target) with mean/std/min/max/amp
    columns for each of the variable's value columns. `season_target` is the
    calendar year the season ends in (Nov Y-1 → Oct Y ⇒ target Y), matching the
    FY1 yield convention so it lines up with climate target years.
    """
    if var not in PHENOLOGY_CHANNELS:
        raise ValueError(f"Unknown phenology variable {var!r}. "
                         f"Choices: {list(PHENOLOGY_CHANNELS)}")
    cfg  = PHENOLOGY_CHANNELS[var]
    stem = cfg["stem"]
    path = os.path.join(data_dir, "phenology", f"phenology_{stem}_{suffix or aoi}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Phenology parquet not found: {path!r}. "
            "Generate it with phenology-sample-gen.R or check --phenology_suffix."
        )
    cols = cfg["cols"]
    df = _read_parquet(path)
    # Season the observation belongs to: Nov/Dec fall in the *next* year's season.
    if "hindcast_start" in df.columns:
        df["season_target"] = df["hindcast_start"].astype(int) + 1
    else:
        d = pd.to_datetime(df["date"])
        df["season_target"] = d.dt.year + (d.dt.month >= 11).astype(int)

    grouped = df.groupby(["adm2_id", "season_target"])[cols]
    agg = grouped.agg(["mean", "std", "min", "max"])
    agg.columns = [f"{col}_prev_{stat}" for col, stat in agg.columns]
    for col in cols:
        agg[f"{col}_prev_amp"] = agg[f"{col}_prev_max"] - agg[f"{col}_prev_min"]
    return agg.reset_index()


def _append_phenology_features(
    adm2_df: pd.DataFrame,
    data_dir: str,
    use_phenology: list[str],
    aoi: str,
    forecast_year: int,
    lag_years: int,
    suffix: str | None,
    lookback_years: int = 1,
) -> tuple[pd.DataFrame, list]:
    """Attach lagged prior-season phenology summary features to a featurised frame.

    For each sample (adm2_id, hindcast_start) the target year is
    hindcast_start + forecast_year; we join the phenology season
    (target - lag) for every lag in the lookback. Samples with no matching
    phenology season are dropped (their features would be missing), which
    restricts the sample set to years with phenology coverage (NDVI/FPAR 2001+,
    soil moisture 2003+) — and, for a lookback of n, to years whose *oldest*
    requested season is also covered.
    """
    df = adm2_df.copy()
    df["_target_year"] = df["hindcast_start"].astype(int) + forecast_year
    lags = phenology_lags(lag_years, lookback_years)

    new_cols: list[str] = []
    for var in use_phenology:
        agg = load_phenology_aggregates(data_dir, var, aoi, suffix)
        stat_cols = [c for c in agg.columns if c not in ("adm2_id", "season_target")]
        for lag in lags:
            sfx = phenology_lag_suffix(lag, lookback_years)
            renamed = agg.rename(columns={
                "season_target": "_pheno_season",
                **{c: f"{c}{sfx}" for c in stat_cols},
            })
            df["_pheno_season"] = df["_target_year"] - lag
            df = df.merge(renamed, on=["adm2_id", "_pheno_season"], how="left")
            new_cols += [f"{c}{sfx}" for c in stat_cols]
        print(f"Phenology '{var}'   : {len(stat_cols) * len(lags)} prior-season "
              f"features ({len(lags)} season(s), lags {lags[0]}–{lags[-1]}yr)")

    before = len(df)
    df = df.dropna(subset=new_cols).reset_index(drop=True)
    print(f"  after requiring lagged phenology: {len(df)} rows "
          f"(dropped {before - len(df)} lacking prior-season coverage)")
    df = df.drop(columns=["_target_year", "_pheno_season"])
    return df, new_cols


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
    forecast_year: int = 1,
    aoi: str = "CONUS",
) -> pd.DataFrame:
    """Load and merge climate + crop parquets; apply the appropriate date window.

    Aggregated mode window (date-based, year-boundary safe):
        [plant_start_date − preseason_buffer,  harvest_end_date + postharvest_buffer]

    Timeseries mode window:
        [Nov 1 of yield year − 1,  Nov 1 of yield year]

    forecast_year selects which lead-year climate product to use:
        1 (default) — {data_dir}/{climate_model}/climate_{climate_model}_s*-{ens_member}.parquet
                      (s{Y-1} run, forecasts calendar year Y = hindcast_start + 1)
        2           — {data_dir}/FY2/climate_{aoi}_{climate_model}_fy2_s*.parquet
                      (s{Y-2} run, forecasts calendar year Y = hindcast_start + 2)
    In both cases the target (yield) year is hindcast_start + forecast_year, and
    the crop yield is matched on that target year.

    Returns a daily DataFrame with columns:
        adm2_id, adm2, adm1, hindcast_start, yield_year, date, doy,
        year_month, tasmax_C, tasmin_C, hursmin, pr_mm,
        plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")

    # ---- climate ----
    # Forecast year 1 (standard) lives under {data_dir}/{climate_model}/ and
    # carries the ens_member in the filename; forecast year >1 products live in
    # their own {data_dir}/FY{n}/ folder and are named per-AOI without ens_member.
    if forecast_year == 1:
        pattern = os.path.join(
            data_dir, f"{climate_model}/climate_{climate_model}_s*-{ens_member}.parquet")
    else:
        pattern = os.path.join(
            data_dir, f"FY{forecast_year}",
            f"climate_{aoi}_{climate_model}_fy{forecast_year}_s*.parquet")
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

    print(f"Loading {len(files)} climate parquet(s) (forecast year {forecast_year})...")
    climate_df = pd.concat([_read_parquet(f) for f in files], ignore_index=True)
    climate_df["date"] = pd.to_datetime(climate_df["date"])
    # fastparquet returns integer id columns as float; normalise so downstream
    # merges/folds see a consistent int hindcast_start regardless of read engine.
    climate_df["hindcast_start"] = climate_df["hindcast_start"].astype(int)
    # The calendar year being forecast is hindcast_start + forecast_year (FY1 → +1,
    # FY2 → +2). Compute it uniformly so the crop merge below is FY-agnostic.
    climate_df["target_year"] = climate_df["hindcast_start"] + forecast_year
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
    crop_df = _read_parquet(crop_file)
    crop_df = _apply_target(crop_df, target)
    # Crop yield is keyed by hindcast_start under the FY1 convention, i.e. it is
    # the yield for calendar year hindcast_start + 1. Express that as target_year
    # so it can be matched to any forecast-year climate product.
    crop_df["target_year"] = crop_df["hindcast_start"] + 1
    print(
        f"  Crop ({crop}) rows: {len(crop_df):,}  "
        f"hindcast_start years: {sorted(crop_df['hindcast_start'].unique())}"
    )

    # ---- merge ----
    # Match on target_year (not hindcast_start): under FY1 these are equivalent,
    # but under FY2 the climate's hindcast_start is 2 years before the target,
    # so pairing on target_year keeps each county's weather with the right yield.
    merged = climate_df.merge(
        crop_df[["adm2_id", "target_year",
                 "plant_start_doy", "plant_doy",
                 "harvest_doy", "harvest_end_doy", "yield"]],
        on=["adm2_id", "target_year"],
        how="inner",
    )
    merged["yield_year"] = merged["target_year"]

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
    else:  # timeseries: full Nov-to-Nov window ending in the yield year
        window_start = pd.to_datetime(
            (merged["yield_year"] - 1).astype(str) + "-11-01"
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

def aggregate_daily(daily_df: pd.DataFrame, weather_vars: list | None = None):
    """Aggregate daily county×day data to one row per (adm2_id, hindcast_start).

    Only the requested weather_vars are summarised (default: all of
    WEATHER_VARS). Derived columns follow their inputs: temp_range_C is built
    only when both tasmax_C and tasmin_C are selected, and the pr_mm_sum total
    only when pr_mm is selected.

    Returns (adm2_df, weather_cols).
    """
    df = daily_df.copy()
    weather_cols = resolve_climate_vars(weather_vars)
    if "tasmax_C" in weather_cols and "tasmin_C" in weather_cols:
        df["temp_range_C"] = df["tasmax_C"] - df["tasmin_C"]
        weather_cols = weather_cols + ["temp_range_C"]

    agg = {}
    for col in weather_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"]  = (col, "std")
    if "pr_mm" in weather_cols:
        agg["pr_mm_sum"]   = ("pr_mm",           "sum")
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
        + (["pr_mm_sum"] if "pr_mm" in weather_cols else [])
        + ["plant_doy", "harvest_doy", "harvest_end_doy"]
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

    # Anchor day offsets to Nov 1 of the year before the target/yield year. This
    # equals the hindcast_start year for FY1 but not for later forecast years
    # (FY2's window starts two Novembers after its hindcast_start), so prefer
    # target_year when the loader supplied it.
    anchor_year = (df["target_year"] - 1) if "target_year" in df.columns \
        else df["hindcast_start"]
    season_start    = pd.to_datetime(anchor_year.astype(str) + "-11-01")
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
    forecast_year: int = 1,
    aoi: str = "CONUS",
    yield_history: bool = False,
    yield_history_base: int | None = None,
    use_phenology: list | None = None,
    phenology_lag_years: int = 1,
    phenology_suffix: str | None = None,
    phenology_lookback_years: int = 1,
    climate_vars: list | None = None,
) -> tuple[pd.DataFrame, list]:
    """Full pipeline: load → merge → window → featurise.

    Returns (adm2_df, feature_cols) where adm2_df has one row per
    (adm2_id, hindcast_start) with columns:
        adm2_id, adm2, adm1, hindcast_start, yield, <feature_cols>

    When yield_history is True, each county's own prior observed yields are
    appended as recency-ordered lag features (yield_lag{N} = yield N years
    before the target year, with yield_lagmask{N} flagging real vs padded
    values), spanning all years back to yield_history_base (default: earliest
    year in the crop data).

    When use_phenology is a non-empty list (e.g. ["ndvi"]), prior-season
    summary stats for each variable are appended, taken from the season
    phenology_lag_years before the target year (default 1 = last season, which
    is forecast-safe). phenology_lookback_years > 1 repeats those stats for that
    many consecutive earlier seasons (suffixed _lag{n}). Samples without every
    requested season's coverage are dropped.

    climate_vars restricts which daily weather variables are featurised
    (default: all of WEATHER_VARS), so feature-space ablations can run e.g.
    temperature + precipitation only.
    """
    climate_vars = resolve_climate_vars(climate_vars)

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
        forecast_year=forecast_year,
        aoi=aoi,
    )

    if feature_mode == "aggregated":
        adm2_df, weather_cols = aggregate_daily(daily_df, weather_vars=climate_vars)
        adm2_df, feature_cols = build_features(adm2_df, weather_cols)
    else:
        adm2_df, feature_cols = build_timeseries_features(
            daily_df, weather_vars=climate_vars)

    if yield_history:
        adm2_df, hist_cols = _append_yield_history_features(
            adm2_df, data_dir, crop, aoi, forecast_year, yield_history_base)
        feature_cols = feature_cols + hist_cols

    if use_phenology:
        adm2_df, pheno_cols = _append_phenology_features(
            adm2_df, data_dir, use_phenology, aoi, forecast_year,
            phenology_lag_years, phenology_suffix,
            lookback_years=phenology_lookback_years)
        feature_cols = feature_cols + pheno_cols

    return adm2_df, feature_cols


def _append_yield_history_features(
    adm2_df: pd.DataFrame,
    data_dir: str,
    crop: str,
    aoi: str,
    forecast_year: int,
    base_year: int | None,
) -> tuple[pd.DataFrame, list]:
    """Append prior-yield lag columns (+ validity masks) to a featurised frame.

    Rows are keyed by (adm2_id, hindcast_start); the predicted calendar year is
    hindcast_start + forecast_year. Padded/missing lags are 0 with a mask of 0
    so tree models can tell them apart from a genuine low yield.
    """
    meta = pd.DataFrame({
        "adm2_id":     adm2_df["adm2_id"].to_numpy(),
        "target_year": adm2_df["hindcast_start"].astype(int) + forecast_year,
    })
    yseries = load_county_yield_series(data_dir, crop, aoi)
    hist, mask, base_year, L = build_yield_history_matrix(
        meta, yseries, base_year=base_year)

    lag_cols  = [f"yield_lag{j + 1}"     for j in range(L)]
    mask_cols = [f"yield_lagmask{j + 1}" for j in range(L)]
    # Build all history columns at once and concat to avoid frame fragmentation.
    hist_block = pd.DataFrame(
        np.hstack([hist, mask]),
        columns=lag_cols + mask_cols,
        index=adm2_df.index,
    )
    df = pd.concat([adm2_df, hist_block], axis=1)

    print(
        f"Yield history      : {L} lag features/county "
        f"(back to {base_year}); avg valid lags/sample "
        f"{mask.sum(axis=1).mean():.1f}"
    )
    return df, lag_cols + mask_cols
