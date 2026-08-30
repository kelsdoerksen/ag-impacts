"""
Load, merge, and featurise climate + crop parquets for yield model training.

Expected files in data_dir:
  Climate:  {climate_model}/climate_{aoi}_{climate_model}_s{hindcast_start}-{ens_member}.parquet
            (the original CONUS series omits the AOI token; see "Climate file
            resolution" below for every convention accepted)
  Crop:     crop_{crop}_{aoi}.parquet

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
available_crops(data_dir, aoi) → list[str]
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

from degree_days import capped_degree_days, degree_days_sine

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
FEATURE_MODES   = ["aggregated", "timeseries", "agro", "pixel"]

# ── agronomic ("agro") feature mode ──────────────────────────────────────────
# A deliberately small, physically-motivated feature set for isolating climate
# skill, in place of the generic mean/std block of "aggregated" mode.
#
# Growing and extreme degree days both use the single-sine method from
# degree_days.py -- the day's heat integrated under a sine diurnal cycle between
# tasmin and tasmax, not read off the daily mean. See that module for why the
# distinction matters (it is the difference between counting and missing the
# heat on every day the threshold falls inside the day's range).
#
# GDD is capped: heat between GDD_BASE_C and GDD_CAP_C counts toward growth,
# heat above the cap does not and is charged to EDD instead. That is the
# piecewise-linear corn response -- gains to ~30 C, losses above it.
GDD_BASE_C          = 10.0   # corn lower base temperature
EDD_THRESHOLD_C     = 30.0   # corn extreme-heat threshold
GDD_CAP_C           = None   # upper bound on beneficial heat; None = EDD_THRESHOLD_C
GDD_PRESEASON_DAYS  = 14     # GDD accumulation starts this many days pre-plant

# Sub-seasonal features are binned into fixed-length blocks measured from the
# window start rather than calendar months, so block k means the same
# phenological stage in every county (m01 is always "the month before the
# planting window opens"). Calendar months cannot do this: the CONUS corn
# window spans DOY ~0-363 and varies in length by ~80 days, so a given calendar
# month falls inside some counties' windows and outside others'.
SEASON_BLOCK_DAYS    = 30
SEASON_BLOCKS_DEFAULT = 8    # 8 x 30 = 240 days; every CONUS corn window covers it

# Variables summarised per season block, and the statistics computed for each.
SEASON_BLOCK_VARS  = ["pr_mm", "hursmin"]
SEASON_BLOCK_STATS = ["mean", "min", "max", "std", "var"]

# Variables summarised over the whole window ("entire time series") with the
# same five statistics.
WHOLE_WINDOW_STATS_VARS = ["hursmin"]
TARGET_OPTIONS  = ["yield", "yield_pct_change"]

# Targets expressed relative to a county's own history. Unlike the entries in
# TARGET_OPTIONS these CANNOT be applied when the crop table is loaded: their
# baseline must be fit on each fold's training years only, or the test year
# leaks into its own normalisation. They are applied inside the fold runners
# (see CountyBaseline) and inverted before scoring, so every metric, CSV and
# plot downstream stays in kg/ha.
ANOMALY_TARGETS = ["yield_anomaly", "yield_zscore"]
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

def drop_invalid_yield(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Drop rows whose observed yield is missing or non-positive.

    The cy-bench yield files encode "no crop was harvested here" as yield 0 with
    harvest_area/production of 0 or NaN — whole countries (FI, IE, EE, LV) are
    nothing but such rows. They are structural zeros, not bad harvests: in the
    CONUS parquet the 1st percentile of positive yields is ~940 kg/ha against a
    median of ~5,600, so a 0 is a different kind of record rather than the left
    tail of the distribution. Training on them teaches the model that a county
    produced nothing, and log(0) = -inf makes --target log_yield diverge on the
    first optimizer step.
    """
    n_before = len(df)
    out      = df[df["yield"].notna() & (df["yield"] > 0)].copy()
    n_zero   = int((df["yield"].notna() & (df["yield"] <= 0)).sum())
    if n_zero:
        where = f" [{context}]" if context else ""
        print(f"[data]{where} Dropped {n_zero:,} rows with non-positive yield "
              f"(of {n_before:,}); these are 'no crop harvested' records.")
    return out


class CountyBaseline:
    """Per-county expected yield, fit on training rows only.

    This is the "county climatology" reference: what a county normally yields,
    estimated from its own other years. It is both the honest null model for a
    yield forecast (beating it is the minimum bar) and the offset that turns an
    absolute-yield target into a weather-skill target.

    Fitting on training rows only is the whole point. Under leave-one-year-out
    the test year must not contribute to its own baseline, so a new instance is
    fit inside each fold.

    Counties too thin to fit on their own fall back to the pooled training
    baseline, and sigma is floored so a near-constant county cannot produce
    explosive z-scores.

    Detrending
    ----------
    With detrend=True the baseline is a per-county straight line in year rather
    than a flat mean:  base(year) = level + slope * (year - year0).

    This matters because CONUS maize yields carry a large technology trend
    (~+140 kg/ha/yr) that a climate-only model cannot possibly predict: under
    leave-one-year-out it does not know which year it is forecasting, so it is
    anchored near the whole-period mean and systematically over-predicts early
    years and under-predicts late ones. A flat mean leaves that bias in the
    anomaly and it swamps the weather signal. Removing it per county is the
    difference between measuring weather skill and measuring the trend.

    Counties with fewer than min_trend_years distinct years, or with no spread
    in year, fall back to a flat mean (their trend is not estimable); counties
    below min_years fall back to the pooled baseline, which is itself
    trend-fitted when detrend=True.
    """

    def __init__(self, min_years: int = 3, sigma_floor_frac: float = 0.1,
                 detrend: bool = False, min_trend_years: int = 8):
        self.min_years        = max(int(min_years), 2)
        self.sigma_floor_frac = float(sigma_floor_frac)
        self.detrend          = bool(detrend)
        # A line through very few noisy points overfits: the residual sigma
        # collapses and the extrapolated slope is wild. Require a real series.
        self.min_trend_years  = max(int(min_trend_years), 3)
        # aid -> (level, slope, year0, sigma); slope/year0 are 0.0 when flat.
        self.params: dict = {}
        self.global_params: tuple = (0.0, 0.0, 0.0, 1.0)
        self.n_fallback       = 0   # counties too thin for their own baseline
        self.n_flat           = 0   # counties detrended-mode but fit flat

    @staticmethod
    def _fit_flat(vals, min_years):
        """Return (level, slope=0, year0=0, sigma) for one county, or None."""
        n = len(vals)
        if n < min_years:
            return None
        mean = float(vals.mean())
        # ddof=1 where possible: these are estimates, not the population.
        sigma = float((vals - mean).std(ddof=1)) if n > 1 else 0.0
        return mean, 0.0, 0.0, sigma

    @staticmethod
    def _fit_line(years, vals, min_trend_years):
        """Return (level, slope, year0, resid_sigma), or None if not estimable.

        Centred on the county's mean training year so `level` stays the county
        mean and the slope term is a small correction — extrapolating one year
        past the training span (the endpoint LOO folds) stays well conditioned.
        """
        if len(np.unique(years)) < min_trend_years:
            return None
        year0 = float(years.mean())
        x     = years.astype(float) - year0
        if x.std() == 0:
            return None
        slope, level = np.polyfit(x, vals, 1)
        resid = vals - (level + slope * x)
        # dof = n - 2: two parameters were fit.
        sigma = float(resid.std(ddof=2)) if len(vals) > 2 else 0.0
        return float(level), float(slope), year0, sigma

    def _fit_county(self, years, vals):
        if self.detrend:
            fit = self._fit_line(years, vals, self.min_trend_years)
            if fit is not None:
                return fit, False
        fit = self._fit_flat(vals, self.min_years)
        return fit, (fit is not None and self.detrend)

    def fit(self, adm2_ids, y, years=None):
        """Fit per-county baselines. `years` is required when detrend=True."""
        adm2_ids = np.asarray(adm2_ids)
        y        = np.asarray(y, dtype=float)
        if self.detrend and years is None:
            raise ValueError("CountyBaseline(detrend=True).fit needs `years`")
        years = (np.asarray(years, dtype=float) if years is not None
                 else np.zeros(len(y)))

        # Pooled fallback, trend-fitted too so thin counties still shed the
        # national trend rather than inheriting a flat whole-period mean.
        g = None
        if self.detrend:
            g = self._fit_line(years, y, self.min_trend_years)
        if g is None:
            g = self._fit_flat(y, 2) or (float(y.mean()), 0.0, 0.0, float(y.std()))
        floor = max(g[3] * self.sigma_floor_frac, 1e-6)

        self.n_fallback = 0
        self.n_flat     = 0
        for aid in np.unique(adm2_ids):
            sel = adm2_ids == aid
            fit, was_flat = self._fit_county(years[sel], y[sel])
            if fit is None:
                self.n_fallback += 1
                continue
            level, slope, year0, sigma = fit
            self.n_flat += int(was_flat)
            self.params[aid] = (level, slope, year0, max(sigma, floor))
        self.global_params = (g[0], g[1], g[2], max(g[3], floor))
        return self

    def _lookup(self, adm2_ids, years=None):
        """Vectorised (baseline, sigma) per row, falling back where unfit."""
        gl, gsl, gy0, gsig = self.global_params
        n     = len(adm2_ids)
        years = (np.asarray(years, dtype=float) if years is not None
                 else np.zeros(n))
        base  = np.empty(n, dtype=float)
        sigma = np.empty(n, dtype=float)
        for i, aid in enumerate(adm2_ids):
            level, slope, year0, sig = self.params.get(aid, (gl, gsl, gy0, gsig))
            base[i]  = level + slope * (years[i] - year0)
            sigma[i] = sig
        return base, sigma

    def transform(self, adm2_ids, y, standardize: bool, years=None):
        """kg/ha -> anomaly (or z-score). Inverse of `inverse`."""
        base, sigma = self._lookup(np.asarray(adm2_ids), years)
        anom = np.asarray(y, dtype=float) - base
        return anom / sigma if standardize else anom

    def inverse(self, adm2_ids, z, standardize: bool, years=None):
        """anomaly (or z-score) -> kg/ha, so every metric downstream is in yield units."""
        base, sigma = self._lookup(np.asarray(adm2_ids), years)
        z = np.asarray(z, dtype=float)
        return (z * sigma + base) if standardize else (z + base)


# Modes for --within_county_features. "none" leaves the design matrix alone.
WITHIN_COUNTY_MODES = ["none", "center", "zscore"]


class WithinCountyScaler:
    """Re-express each feature as a deviation from that county's own normal.

    Why this exists
    ---------------
    A seasonal climate aggregate is mostly a description of *where* a county is,
    not of *what happened* there this year. Measured on CONUS corn, the share of
    each feature's variance that lies between counties rather than within a
    county across years is 0.93 for season-mean tasmax/tasmin, 0.89 for
    GDD/EDD/hot-day counts, and 0.23 even for per-block precipitation. A tree
    model handed those columns can identify the county and predict its long-run
    mean yield — spatial regression dressed up as forecasting. It scores well on
    absolute yield (R2 up to ~0.7 here) while having no anomaly skill at all.

    Subtracting each county's own training-period feature mean removes that
    channel: every column becomes "how unusual was this season, for this
    county", so whatever skill survives is genuinely weather-driven. This is the
    feature-side counterpart of what CountyBaseline does to the target.

    Fitting on training rows only is essential for the same reason it is in
    CountyBaseline: under leave-one-year-out the held-out year must not
    contribute to the mean it is centred on.

    Modes
    -----
    center  x - mean_county(x)                    keeps each feature's units
    zscore  (x - mean_county(x)) / sd_county(x)   puts every county's variability
                                                  on one scale, so a wet year in
                                                  Arizona and in Iowa read alike

    Counties unseen in training fall back to the pooled training mean/sd.
    Features that never vary within a county (plant_doy, and the yield_lagmask
    indicator columns, which must stay 0/1) can be excluded via `skip_cols`.
    """

    def __init__(self, mode: str = "center", skip_cols=None, sd_floor_frac: float = 1e-3):
        if mode not in WITHIN_COUNTY_MODES:
            raise ValueError(f"mode must be one of {WITHIN_COUNTY_MODES}, got {mode!r}")
        self.mode          = mode
        self.skip_cols     = set(skip_cols or [])
        self.sd_floor_frac = float(sd_floor_frac)
        self.cols: list    = []
        self.means         = None   # DataFrame: index adm2_id, columns self.cols
        self.sds           = None
        self.global_mean   = None   # Series over self.cols
        self.global_sd     = None
        self.n_unseen      = 0

    def fit(self, X, adm2_ids):
        """Learn per-county means (and sds) from training rows only."""
        self.cols = [c for c in X.columns if c not in self.skip_cols]
        ids = pd.Series(np.asarray(adm2_ids), index=X.index, name="_aid")
        sub = X[self.cols]

        g = sub.groupby(ids)
        self.means = g.mean()
        self.global_mean = sub.mean()
        if self.mode == "zscore":
            # ddof=1: a county's spread across its training years is an estimate.
            self.sds = g.std(ddof=1)
            self.global_sd = sub.std(ddof=1)
            # A county with one training year, or a feature constant within the
            # county, gives sd 0 or NaN. Floor against the pooled spread so the
            # division cannot blow up or produce NaN.
            floor = (self.global_sd.abs() * self.sd_floor_frac).clip(lower=1e-9)
            self.global_sd = self.global_sd.fillna(0.0).clip(lower=floor)
            self.sds = self.sds.fillna(0.0).clip(lower=floor, axis=1)
        return self

    def transform(self, X, adm2_ids):
        """Centre (and optionally scale) X against its counties' training normals."""
        if self.mode == "none":
            return X
        ids = pd.Series(np.asarray(adm2_ids), index=X.index, name="_aid")
        # reindex aligns each row to its county's stats; unseen counties -> NaN,
        # which we then fill with the pooled values.
        mu = self.means.reindex(ids.to_numpy())
        self.n_unseen = int(mu.isna().all(axis=1).sum())
        mu = mu.fillna(self.global_mean)
        mu.index = X.index

        out = X.copy()
        out[self.cols] = X[self.cols].to_numpy() - mu[self.cols].to_numpy()
        if self.mode == "zscore":
            sd = self.sds.reindex(ids.to_numpy()).fillna(self.global_sd)
            sd.index = X.index
            out[self.cols] = out[self.cols].to_numpy() / sd[self.cols].to_numpy()
        return out


def _apply_target(crop_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Overwrite the 'yield' column with the requested target variable.

    yield            — no change (raw yield values)
    yield_pct_change — percent change vs. previous hindcast_start year for the
                       same county: (yield_t - yield_{t-1}) / yield_{t-1} * 100.
                       Rows with no prior year become NaN and are dropped later
                       by .dropna(subset=["yield"]).
    """
    # Anomaly targets are fit per fold on training rows only, so the crop table
    # keeps raw yield here and the fold runners do the transform.
    if target in ANOMALY_TARGETS:
        return crop_df
    if target not in TARGET_OPTIONS:
        raise ValueError(f"target must be one of {TARGET_OPTIONS + ANOMALY_TARGETS}")
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


# ---------------------------------------------------------------------------
# Climate file resolution
# ---------------------------------------------------------------------------
#
# Climate parquets have accumulated several naming conventions:
#
#   climate_{model}_s{year}-{ens}.parquet          original CONUS FY1 series
#   climate_{aoi}_{model}_s{year}-{ens}.parquet    AOI series with ens suffix
#   climate_{aoi}_{model}_fy{n}_s{year}.parquet    AOI series, forecast-year tag
#
# All three are accepted, most specific first. The AOI is matched explicitly so
# a non-CONUS run can never silently pick up the CONUS series (or another AOI's)
# out of a shared folder — that would train on the wrong region's weather while
# looking perfectly healthy.

# AOI tags naming the original CONUS series, whose FY1 files predate the
# AOI-in-filename convention and therefore carry no AOI token.
CONUS_ALIASES = {"CONUS", "USA", "US"}


def _is_conus(aoi: str) -> bool:
    return str(aoi).upper() in CONUS_ALIASES


def climate_dir(data_dir: str, climate_model: str, forecast_year: int = 1) -> str:
    """Folder holding the climate parquets for one forecast-year product.

    FY1 lives in {data_dir}/{climate_model}/; later forecast years get their own
    {data_dir}/FY{n}/ folder.

    FY0 is the observed (reanalysis) case -- ERA5 has no initialization year and
    no lead time, so hindcast_start is the target year itself. There is only ever
    one such product per source, so it shares the {climate_model} folder rather
    than getting an FY0/ of its own.
    """
    return os.path.join(
        data_dir,
        climate_model if forecast_year in (0, 1) else f"FY{forecast_year}")


def climate_name_candidates(climate_model: str, ens_member: str, aoi: str,
                            forecast_year: int = 1, year="*") -> list[str]:
    """Ordered basename patterns for one hindcast year's climate parquet.

    `year` may be a literal year or "*" to build glob patterns for a whole
    series. AOI-tagged names come first; the AOI-less legacy name is offered
    only for CONUS, so other AOIs cannot resolve to it.
    """
    fy = forecast_year
    # Most specific first: AOI + forecast-year tag + s-year + ensemble member.
    # That is the canonical form ML-sample-gen.R now writes for every AOI and
    # forecast year; the rest are older variants kept readable so existing
    # folders do not have to be renamed.
    names = [
        f"climate_{aoi}_{climate_model}_fy{fy}_s{year}-{ens_member}.parquet",
        f"climate_{aoi}_{climate_model}_fy{fy}_s{year}.parquet",
    ]
    if fy == 1:
        # The FY1 series predates the fy token, so those spellings are FY1-only.
        names += [
            f"climate_{aoi}_{climate_model}_s{year}-{ens_member}.parquet",
            f"climate_{aoi}_{climate_model}_s{year}.parquet",
        ]
    if _is_conus(aoi):
        # The original CONUS series carries no AOI token at all.
        names.append(f"climate_{climate_model}_fy{fy}_s{year}.parquet")
        if fy == 1:
            names += [
                f"climate_{climate_model}_s{year}-{ens_member}.parquet",
                f"climate_{climate_model}_s{year}.parquet",
            ]
    return names


def _aoi_prefix_ok(basename: str, climate_model: str, aoi: str) -> bool:
    """Does this filename belong to `aoi`?

    Non-CONUS AOIs must carry their tag as the segment right after 'climate_'.
    CONUS accepts its own tag or the legacy AOI-less form (climate_{model}_…),
    but not a file tagged with some other AOI.
    """
    if basename.startswith(f"climate_{aoi}_"):
        return True
    if not _is_conus(aoi):
        return False
    return (basename.startswith(f"climate_{climate_model}")
            or any(basename.startswith(f"climate_{a}_") for a in CONUS_ALIASES))


def _climate_dir_listing(cdir: str, limit: int = 8) -> str:
    """A short 'what is actually in there' hint for error messages."""
    if not os.path.isdir(cdir):
        return f"    (directory does not exist: {cdir})"
    found = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(cdir, "*.parquet")))
    if not found:
        return "    (no .parquet files in that directory)"
    shown = "\n".join(f"    {n}" for n in found[:limit])
    if len(found) > limit:
        shown += f"\n    … and {len(found) - limit} more"
    return shown


def find_climate_file(data_dir: str, climate_model: str, year: int,
                      ens_member: str = "r1i4p1f1", aoi: str = "CONUS",
                      forecast_year: int = 1) -> str:
    """Locate the climate parquet for one hindcast year, honouring the AOI.

    Tries every known naming convention for `aoi` in order, then falls back to
    any parquet in the folder that carries both this AOI and an _s{year} token.
    Raises FileNotFoundError listing what was tried and what is present.
    """
    cdir = climate_dir(data_dir, climate_model, forecast_year)
    candidates = climate_name_candidates(
        climate_model, ens_member, aoi, forecast_year, year)

    for name in candidates:
        path = os.path.join(cdir, name)
        if os.path.exists(path):
            return path

    # Fallback: an unanticipated variant of the name, but still this AOI and year.
    year_re = re.compile(rf"(?:^|[_-])s{year}(?:[^0-9]|$)")
    loose = sorted(
        p for p in glob.glob(os.path.join(cdir, "*.parquet"))
        if year_re.search(os.path.basename(p))
        and _aoi_prefix_ok(os.path.basename(p), climate_model, aoi)
    )
    if loose:
        return loose[0]

    raise FileNotFoundError(
        f"No climate parquet for AOI {aoi!r}, year {year} in {cdir!r}.\n"
        f"  Tried:\n" + "".join(f"    {n}\n" for n in candidates) +
        f"  Present in that directory:\n{_climate_dir_listing(cdir)}\n"
        f"  Check --data_dir, --climate_model, --ens_member, --aoi and the year range."
    )


def find_climate_files(data_dir: str, climate_model: str,
                       ens_member: str = "r1i4p1f1", aoi: str = "CONUS",
                       forecast_year: int = 1,
                       hindcast_years: list | None = None) -> list[str]:
    """Locate the climate parquets for a whole series, one per hindcast year.

    With hindcast_years given, each year is resolved individually (so a series
    whose naming changed partway through still loads). Without it, every file
    in the folder belonging to this AOI is returned.
    """
    if hindcast_years is not None:
        return [find_climate_file(data_dir, climate_model, y, ens_member,
                                  aoi, forecast_year)
                for y in sorted(hindcast_years)]

    cdir  = climate_dir(data_dir, climate_model, forecast_year)
    found = {}
    for pattern in climate_name_candidates(
            climate_model, ens_member, aoi, forecast_year, year="*"):
        for path in sorted(glob.glob(os.path.join(cdir, pattern))):
            # First (most specific) pattern to claim a year wins.
            found.setdefault(_parse_hindcast_year(path), path)
    if not found:
        raise FileNotFoundError(
            f"No climate parquets for AOI {aoi!r} in {cdir!r}.\n"
            f"  Present in that directory:\n{_climate_dir_listing(cdir)}"
        )
    return [found[y] for y in sorted(found)]


def available_crops(data_dir: str, aoi: str = "CONUS") -> list[str]:
    """List crop names that have a crop_{name}_{aoi}.parquet in data_dir."""
    files = glob.glob(os.path.join(data_dir, f"crop_*_{aoi}.parquet"))
    crops = []
    for f in files:
        m = re.search(rf"crop_(.+)_{re.escape(aoi)}\.parquet", os.path.basename(f))
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
    df = drop_invalid_yield(df, context="yield history")
    df["cal_year"] = df["hindcast_start"].astype(int) + 1
    return (
        df[["adm2_id", "cal_year", "yield"]]
        .rename(columns={"yield": "yield_obs"})
        .drop_duplicates(["adm2_id", "cal_year"])
        .reset_index(drop=True)
    )


STATIC_COVARS = {
    # token -> the columns load_static_covariates() must produce
    "latlon": ["lat", "lon"],
}


def load_static_covariates(covars, aoi="CONUS", county_shp=None,
                           search_dirs=()) -> pd.DataFrame:
    """Time-invariant per-admin-unit features: adm2_id plus one column each.

    Currently only 'latlon' (polygon centroid, degrees). Centroids are taken in
    the AOI's equal-area projection and then converted to lat/lon, so they are
    not distorted by the projection.

    A WARNING on what these can do. Under leave-one-YEAR-out every unit appears
    in training, so a precise coordinate pair is effectively a unit ID, and a
    model will happily use it to look up that unit's mean yield: on CONUS a
    random forest given ONLY lat/lon scores RMSE 1,707 / R2 0.547, matching
    county climatology (1,716 / 0.542) with a correlation of 0.994 — and an
    anomaly R2 of 0.011, i.e. no forecast skill at all. Against an absolute
    yield target these features therefore inflate the headline metric without
    adding anything. They are only meaningful once the county baseline has been
    removed (an anomaly target), where they can condition *how* a unit responds
    to weather rather than *what it usually yields*. run_loo_experiment.py
    enforces that pairing.
    """
    unknown = [c for c in covars if c not in STATIC_COVARS]
    if unknown:
        raise ValueError(f"Unknown static covariate(s) {unknown}. "
                         f"Choices: {sorted(STATIC_COVARS)}")
    # Lazy: pulls geopandas (and matplotlib via plot_results), which the rest of
    # this module deliberately does not require.
    from plot_results import load_counties_geo, resolve_aoi_shp

    # Resolve before loading so a missing file names every path tried rather
    # than surfacing as a bare pyogrio DataSourceError.
    shp = resolve_aoi_shp(aoi, county_shp, search_dirs=search_dirs)
    gdf = load_counties_geo(shp, aoi=aoi)                 # equal-area projection
    cen = gdf.geometry.centroid.to_crs("EPSG:4326")
    out = pd.DataFrame({"adm2_id": gdf["GEOID"].astype(str),
                        "lat": cen.y.to_numpy(), "lon": cen.x.to_numpy()})
    cols = [c for tok in covars for c in STATIC_COVARS[tok]]
    return out[["adm2_id"] + cols].drop_duplicates("adm2_id").reset_index(drop=True)


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

    Aggregated / agro mode window (date-based, year-boundary safe):
        [plant_start_date − preseason_buffer,  harvest_end_date + postharvest_buffer]
        Where the crop calendar crosses the new year (plant_start_doy >
        harvest_end_doy, as in southern-hemisphere maize), plant_start_date is
        anchored in yield_year − 1 and that year's climate is pulled in too, so
        the window covers the whole season rather than collapsing to nothing.
        County-seasons the loaded climate does not cover end to end are dropped.

    Timeseries mode window:
        [Nov 1 of yield year − 1,  Nov 1 of yield year]

    forecast_year selects which lead-year climate product to use:
        1 (default) — {data_dir}/{climate_model}/  (s{Y-1} run, forecasts
                      calendar year Y = hindcast_start + 1)
        2           — {data_dir}/FY2/             (s{Y-2} run, forecasts
                      calendar year Y = hindcast_start + 2)
    In both cases the target (yield) year is hindcast_start + forecast_year, and
    the crop yield is matched on that target year. Filenames are resolved by
    find_climate_files, which accepts every naming convention in use and matches
    the AOI explicitly — see the "Climate file resolution" section above.

    Returns a daily DataFrame with columns:
        adm2_id, adm2, adm1, hindcast_start, yield_year, date, doy,
        year_month, tasmax_C, tasmin_C, hursmin, pr_mm,
        plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")

    # ---- climate ----
    # Resolved per hindcast year and per AOI, so a folder holding more than one
    # AOI's series cannot cross-contaminate a run.
    files = find_climate_files(data_dir, climate_model, ens_member, aoi,
                               forecast_year, hindcast_years)

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
    crop_file = os.path.join(data_dir, f"crop_{crop}_{aoi}.parquet")
    if not os.path.exists(crop_file):
        raise FileNotFoundError(
            f"Crop parquet not found: {crop_file!r}\n"
            f"Available crops for AOI {aoi!r}: {available_crops(data_dir, aoi)}"
        )
    crop_df = _read_parquet(crop_file)
    # Filter structural zeros on the RAW yield, before _apply_target — once the
    # target is yield_pct_change the column holds percent changes, which are
    # legitimately negative. Doing it first also stops pct_change dividing by a
    # zero prior year.
    crop_df = drop_invalid_yield(crop_df, context=f"{crop}/{aoi}")
    crop_df = _apply_target(crop_df, target)
    # Crop yield is keyed by hindcast_start under the FY1 convention, i.e. it is
    # the yield for calendar year hindcast_start + 1. Express that as target_year
    # so it can be matched to any forecast-year climate product.
    crop_df["target_year"] = crop_df["hindcast_start"] + 1
    print(
        f"  Crop ({crop}) rows: {len(crop_df):,}  "
        f"hindcast_start years: {sorted(crop_df['hindcast_start'].unique())}"
    )

    # ---- cross-year seasons ----
    # A southern-hemisphere season is planted in one calendar year and harvested
    # in the next (Zambian maize: sos ~ DOY 326, eos ~ DOY 134), so half its
    # weather sits in a climate file whose own target_year is a year earlier and
    # would never reach this season's rows. Offer every climate row to the season
    # ending in the FOLLOWING year as well; the date window below keeps only the
    # days that genuinely fall inside a season, so the extra candidates cost rows
    # in the merge and nothing in the result. Gated on the calendar actually
    # crossing, since for a same-year AOI like CONUS this would double the merge
    # to no purpose.
    crosses_new_year = bool(
        (crop_df["plant_start_doy"] > crop_df["harvest_end_doy"]).any())
    # Only the observed (FY0) product needs stitching. Its files are calendar
    # years, so a cross-year season straddles two of them -- but they are two
    # slices of one continuous reanalysis, so joining them is just reading the
    # record. A hindcast file is not: s{Y} runs Nov Y -> Oct Y+1, so it already
    # holds its target season end to end, and reaching into s{Y-1} for a few
    # extra pre-season days would splice a DIFFERENT initialization's weather
    # into the same feature row. The pre-season buffer is clipped at the
    # initialization date instead, which is what the hindcast actually knows.
    stitch_prev_year = crosses_new_year and forecast_year == 0
    if stitch_prev_year:
        print("  Crop calendar crosses the new year (plant_start_doy > "
              "harvest_end_doy): pairing each season with the previous "
              "calendar year's observations as well.")
        climate_df = pd.concat(
            [climate_df,
             climate_df.assign(target_year=climate_df["target_year"] + 1)],
            ignore_index=True,
        )
    elif crosses_new_year:
        print(f"  Crop calendar crosses the new year; forecast year "
              f"{forecast_year} files span Nov-Oct and already cover the "
              f"season, so any part of the {preseason_buffer}-day pre-season "
              f"buffer that falls before the initialization date is clipped.")

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
    # Re-key on the season rather than on the calendar year the climate row came
    # from. These are the same thing for a same-year season, but a cross-year one
    # is served by two climate files and every fold, groupby and yield-history
    # lookup downstream keys on hindcast_start expecting target_year =
    # hindcast_start + forecast_year — so the November half must carry the
    # hindcast_start of the season it belongs to, not of the file it came from.
    merged["hindcast_start"] = merged["target_year"] - forecast_year

    # ---- date-based window (year-boundary safe) ----
    # harvest_end_doy is a DOY in the yield year; plant_start_doy is a DOY in the
    # yield year too for a same-year season, but in the year BEFORE it when the
    # calendar crosses the new year. Anchor each end to its own Jan 1.
    yield_yr_jan1 = pd.to_datetime(merged["yield_year"].astype(str) + "-01-01")

    if feature_mode in ("aggregated", "agro"):
        window_start = _plant_year_jan1(merged) + pd.to_timedelta(
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

    merged["_win_start"] = window_start
    merged["_win_end"]   = window_end
    mask   = (merged["date"] >= merged["_win_start"]) & (merged["date"] <= merged["_win_end"])
    merged = merged[mask].copy().reset_index(drop=True)

    if stitch_prev_year:
        # A stitched season needs two climate files, so the first and last
        # season of any series are half-covered — no planting weather for the
        # earliest, no grain fill for the latest. Their aggregates would be taken
        # over a truncated window and arrive looking like ordinary rows, so drop
        # any county-season the loaded climate does not cover end to end. A
        # same-year season cannot hit this (its window sits inside the one
        # calendar year its climate file already covers), and a hindcast season
        # is clipped at its initialization date by design, so the check belongs
        # to the stitched case alone rather than to every AOI and product.
        keys  = ["adm2_id", "hindcast_start"]
        cover = merged.groupby(keys).agg(
            first=("date", "min"), last=("date", "max"),
            want_first=("_win_start", "first"), want_last=("_win_end", "first"))
        complete = cover.index[(cover["first"] <= cover["want_first"])
                               & (cover["last"] >= cover["want_last"])]
        if len(complete) < len(cover):
            short = cover.drop(index=complete)
            print(f"  Dropping {len(short)} county-season(s) whose window the "
                  f"loaded climate does not cover end to end (years: "
                  f"{sorted(short.index.get_level_values('hindcast_start').unique())})")
            merged = merged[
                pd.MultiIndex.from_frame(merged[keys]).isin(complete)
            ].copy().reset_index(drop=True)
    merged = merged.drop(columns=["_win_start", "_win_end"])

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
# Agronomic features ("agro" feature mode)
# ---------------------------------------------------------------------------

def _plant_year_jan1(df: pd.DataFrame) -> pd.Series:
    """Jan 1 of the calendar year each row's plant_start_doy is measured in.

    Usually that is the yield year. Where the crop calendar crosses the new year
    — plant_start_doy after harvest_end_doy, as in southern-hemisphere maize —
    planting happened in the year BEFORE the harvest, so the DOY is anchored a
    year earlier. Anchoring both ends to the yield year there would produce a
    window that ends months before it starts, and silently select no days at all.
    """
    plant_year = (df["yield_year"].astype(int)
                  - (df["plant_start_doy"] > df["harvest_end_doy"]).astype(int))
    return pd.to_datetime(plant_year.astype(str) + "-01-01")


def _window_start_dates(df: pd.DataFrame, preseason_buffer: int) -> pd.Series:
    """Nominal first day of each row's seasonal window.

    Recomputed from the crop calendar rather than taken as each group's earliest
    observed date, so the season-block grid is anchored identically for every
    county-year even where the climate series is short a leading day.
    """
    return _plant_year_jan1(df) + pd.to_timedelta(
        df["plant_start_doy"] - preseason_buffer - 1, unit="D")


def build_agro_features(
    daily_df: pd.DataFrame,
    preseason_buffer: int = 60,
    season_blocks: int = SEASON_BLOCKS_DEFAULT,
    gdd_base: float = GDD_BASE_C,
    edd_threshold: float = EDD_THRESHOLD_C,
    gdd_preseason_days: int = GDD_PRESEASON_DAYS,
    gdd_cap: float | None = GDD_CAP_C,
) -> tuple[pd.DataFrame, list]:
    """Build the curated agronomic feature set, one row per (adm2_id, hindcast_start).

    Features (all computed inside the seasonal window the loader already applied,
    i.e. [plant_start − preseason_buffer, harvest_end + postharvest_buffer]):

      Heat accumulation (single-sine; see degree_days.py)
        gdd                 Growing degree days between gdd_base and gdd_cap,
                            summed over [plant_start − gdd_preseason_days,
                            window end]. Each day's value integrates the heat
                            under a sine diurnal cycle running between tasmin_C
                            and tasmax_C, so a day whose mean sits below the
                            base still accumulates the hours that were above it.
                            Heat beyond gdd_cap stops counting here and is
                            charged to edd instead. Defaults: base 10 C (corn
                            lower threshold), cap = edd_threshold.
        edd                 Extreme degree days above edd_threshold, same
                            single-sine integral — the heat load above 30 C,
                            not just its frequency, and not zero merely because
                            the daily mean happens to land on the threshold.
        days_over_{T}C      Count of days in the window with tasmax_C above the
                            threshold. Reported alongside edd because "number of
                            extreme degree days" is used for both quantities.

      Season temperature
        tasmax_C_mean       Mean daily maximum over the full window.
        tasmin_C_mean       Mean daily minimum over the full window.

      Whole-window humidity
        hursmin_{mean,min,max,std,var}

      Sub-seasonal precipitation and humidity
        {pr_mm,hursmin}_m{NN}_{mean,min,max,std,var}
                            for NN = 01..season_blocks, where block NN covers
                            days [(NN−1)·30 + 1, NN·30] measured from the window
                            start. The final block absorbs any remaining days, so
                            a longer window lengthens the last block rather than
                            adding a partly-empty one.

    Notes
    -----
    GDD accumulation starts gdd_preseason_days before plant_start_doy, which is
    only fully inside the loaded window when preseason_buffer >= that value; a
    warning is printed otherwise since the sum would then be truncated.

    Returns (adm2_df, feature_cols).
    """
    if season_blocks < 1:
        raise ValueError(f"season_blocks must be >= 1, got {season_blocks}")
    missing = [c for c in ("tasmax_C", "tasmin_C", "hursmin", "pr_mm")
               if c not in daily_df.columns]
    if missing:
        raise ValueError(
            f"agro feature mode needs daily columns {missing}, which are absent. "
            "It uses the full weather set, so do not restrict --climate_vars."
        )
    if preseason_buffer < gdd_preseason_days:
        print(
            f"WARNING: --preseason_buffer {preseason_buffer} is shorter than the "
            f"{gdd_preseason_days}-day GDD pre-plant lead-in, so GDD is "
            f"accumulated from the window start instead and undercounts."
        )

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    keys = ["adm2_id", "hindcast_start"]

    # ---- daily derived quantities ----
    # Single-sine degree days, shared with era5_features.py so gdd and edd name
    # the same quantity in both pipelines. NaN temperatures are carried through
    # explicitly: numpy comparisons against NaN are False, which would otherwise
    # turn a missing day into a zero-degree-day one and bias the season sum low.
    cap    = edd_threshold if gdd_cap is None else gdd_cap
    tmin_a = df["tasmin_C"].to_numpy(dtype="float64")
    tmax_a = df["tasmax_C"].to_numpy(dtype="float64")
    valid  = np.isfinite(tmin_a) & np.isfinite(tmax_a)

    df["_gdd_day"] = np.where(
        valid, capped_degree_days(tmin_a, tmax_a, gdd_base, cap), np.nan)
    df["_edd_day"] = np.where(
        valid, degree_days_sine(tmin_a, tmax_a, edd_threshold), np.nan)
    df["_hot_day"] = (df["tasmax_C"] > edd_threshold).astype(float).where(
        df["tasmax_C"].notna())

    # GDD accumulates only from gdd_preseason_days before the planting window
    # opens, so zero out the earlier part of the window rather than sub-setting
    # rows (keeps one pass over the frame and one groupby).
    gdd_start = _plant_year_jan1(df) + pd.to_timedelta(
        df["plant_start_doy"] - gdd_preseason_days - 1, unit="D")
    df["_gdd_day"] = df["_gdd_day"].where(df["date"] >= gdd_start, 0.0)

    # ---- season block index ----
    # Days elapsed since the nominal window start, bucketed into fixed-length
    # blocks. The last block is clipped so it absorbs the variable-length tail.
    elapsed = (df["date"] - _window_start_dates(df, preseason_buffer)).dt.days
    df["_block"] = (elapsed // SEASON_BLOCK_DAYS).clip(lower=0,
                                                       upper=season_blocks - 1)

    # ---- whole-window aggregates ----
    hot_col = f"days_over_{edd_threshold:g}C"
    agg = {
        "gdd":            ("_gdd_day",  "sum"),
        "edd":            ("_edd_day",  "sum"),
        hot_col:          ("_hot_day",  "sum"),
        "tasmax_C_mean":  ("tasmax_C",  "mean"),
        "tasmin_C_mean":  ("tasmin_C",  "mean"),
    }
    for var in WHOLE_WINDOW_STATS_VARS:
        for stat in SEASON_BLOCK_STATS:
            agg[f"{var}_{stat}"] = (var, stat)
    agg["yield"] = ("yield", "first")
    agg["adm2"]  = ("adm2",  "first")
    agg["adm1"]  = ("adm1",  "first")

    adm2_df = df.groupby(keys).agg(**agg).reset_index()

    whole_cols = (
        ["gdd", "edd", hot_col, "tasmax_C_mean", "tasmin_C_mean"]
        + [f"{v}_{s}" for v in WHOLE_WINDOW_STATS_VARS
           for s in SEASON_BLOCK_STATS]
    )

    # ---- per-block aggregates ----
    # Aggregate once per (county-year, block), then pivot blocks into columns.
    block_stats = (
        df.groupby(keys + ["_block"])[SEASON_BLOCK_VARS]
        .agg(SEASON_BLOCK_STATS)
    )
    # MultiIndex columns (var, stat) x block index -> flat {var}_m{NN}_{stat}
    block_wide = block_stats.unstack("_block")
    block_wide.columns = [
        f"{var}_m{int(blk) + 1:02d}_{stat}"
        for var, stat, blk in block_wide.columns
    ]
    block_cols = [
        f"{var}_m{b + 1:02d}_{stat}"
        for var in SEASON_BLOCK_VARS
        for b in range(season_blocks)
        for stat in SEASON_BLOCK_STATS
    ]
    # Name every expected column even if no county-year reached that block, so
    # the feature matrix has a fixed width and the NaN filter below reports it.
    block_wide = block_wide.reindex(columns=block_cols)
    adm2_df = adm2_df.merge(block_wide.reset_index(), on=keys, how="left")

    feature_cols = whole_cols + block_cols

    # A single-day block makes std/var undefined; that is a real coverage gap in
    # the window, so drop those county-years rather than imputing a zero spread.
    adm2_df = adm2_df.dropna(subset=["yield"])
    before  = len(adm2_df)
    adm2_df = adm2_df[adm2_df[feature_cols].notna().all(axis=1)].reset_index(drop=True)
    if before != len(adm2_df):
        print(f"  Dropped {before - len(adm2_df):,} county-years with incomplete "
              f"season-block coverage")

    print(
        f"Agro features      : {len(feature_cols)} columns "
        f"({len(whole_cols)} whole-window + {len(block_cols)} sub-seasonal over "
        f"{season_blocks} x {SEASON_BLOCK_DAYS}d blocks); "
        f"{len(adm2_df):,} county-year pairs"
    )
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

# ---------------------------------------------------------------------------
# Pixel-first season features (feature_mode "pixel")
# ---------------------------------------------------------------------------
#
# Everything above computes features from admin-2 DAILY means: the climate is
# averaged over the unit first, and GDD/EDD/VPD are taken of that average. That
# ordering is wrong for any feature nonlinear in its inputs, because
#   mean(max(0, T - Tbase))  !=  max(0, mean(T) - Tbase)
# and averaging first flattens exactly the tails those features exist to
# measure. R/era5-pixel-store.R + python/era5_features.py fix the ordering by
# computing each feature per pixel and only then taking the cropland-weighted
# mean over the unit. Their stage-3 output is already one row per
# (unit, harvest year) with the features as columns -- a finished design matrix
# -- so this path does no featurising at all. It reads that table, attaches the
# yield target and hands the result to the same fold machinery every other
# feature mode uses.

# cy-bench spells maize; --crop and the crop parquets use the US name. Stage 3
# tags its filenames with whatever era5_features.py --crop was given, which is
# the cy-bench spelling.
CY_CROP_NAMES = {"corn": "maize"}

# Written by aggregate_season() in era5_features.py; everything else in the
# table is a feature.
_SEASON_META_COLS = (
    "adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start",
    "forecast_year", "target_year", "season_start", "season_end",
)


def pixel_features_dirs(data_dir: str, climate_model: str,
                        features_dir: str | None = None) -> list:
    """Candidate stage-3 output directories, in resolution order.

    The default tree is {data_dir}/{climate_model}/features_admin2, matching
    Layout in era5_features.py. But that Layout also honours its own --out and
    the ERA5_OUT_DIR environment variable (for ERA5 only), which is how a store
    kept outside the AOI tree gets built -- so look there too rather than
    reporting a missing file the user can plainly see on disk.
    """
    cands = []
    if features_dir:
        # An explicit path may name either the store root or features_admin2
        # itself; accept both so --pixel_features_dir is hard to get wrong.
        cands += [features_dir, os.path.join(features_dir, "features_admin2")]
    env = os.environ.get("ERA5_OUT_DIR") if climate_model == "ERA5" else None
    if env:
        cands.append(os.path.join(env, "features_admin2"))
    cands.append(os.path.join(data_dir, climate_model, "features_admin2"))
    seen, out = set(), []
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pixel_features_dir(data_dir: str, climate_model: str,
                       features_dir: str | None = None) -> str:
    """First candidate stage-3 directory that exists, else the default."""
    cands = pixel_features_dirs(data_dir, climate_model, features_dir)
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[-1]


def find_season_feature_files(data_dir: str, climate_model: str, aoi: str,
                              crop: str, features_dir: str | None = None
                              ) -> tuple[list, str | None]:
    """Stage-3 season parquets for this AOI/crop.

    Tries the crop tag under both spellings, then "fixed" for a run that used
    era5_features.py --season SOS:EOS instead of a crop calendar.
    """
    tags = [crop]
    if crop in CY_CROP_NAMES:
        tags.append(CY_CROP_NAMES[crop])
    tags.append("fixed")
    for fdir in pixel_features_dirs(data_dir, climate_model, features_dir):
        for tag in tags:
            paths = sorted(glob.glob(
                os.path.join(fdir, f"features_{aoi}_{tag}_season_*.parquet")))
            if paths:
                return paths, tag
    return [], None


def load_pixel_season_features(
    crop: str,
    data_dir: str,
    aoi: str,
    climate_model: str,
    forecast_year: int,
    target: str,
    hindcast_years: list | None = None,
    features_dir: str | None = None,
) -> tuple[pd.DataFrame, list]:
    """Read stage-3 season features and join the yield target onto them.

    Returns (adm2_df, feature_cols) in the same shape as the other feature
    modes: one row per (adm2_id, hindcast_start) carrying adm2_id, adm2, adm1,
    hindcast_start, target_year, yield and the feature columns.
    """
    paths, tag = find_season_feature_files(
        data_dir, climate_model, aoi, crop, features_dir)
    if not paths:
        looked = pixel_features_dirs(data_dir, climate_model, features_dir)
        raise FileNotFoundError(
            f"No pixel-first season features for AOI {aoi!r}, crop {crop!r}.\n"
            f"Looked in:\n" + "".join(f"  {d}\n" for d in looked) +
            f"Point --pixel_features_dir (or ERA5_OUT_DIR) at the store, or "
            f"build it with:\n"
            f"  Rscript R/era5-pixel-store.R --aoi={aoi} <years>\n"
            f"  python python/era5_features.py --aoi={aoi} --source={climate_model} "
            f"--years=<years> --reduce=season "
            f"--crop={CY_CROP_NAMES.get(crop, crop)}"
        )

    clim = pd.concat([_read_parquet(pth) for pth in paths], ignore_index=True)
    feature_cols = [c for c in clim.columns if c not in _SEASON_META_COLS]
    if not feature_cols:
        raise ValueError(f"Season features in {paths[0]!r} carry no feature columns.")

    # Stage 3 stamps the lead time it was built at. Silently mixing an FY0
    # reanalysis with an FY1 hindcast would compare two different experiments,
    # so make the mismatch loud.
    stamped = sorted(int(v) for v in clim["forecast_year"].dropna().unique())
    if stamped != [forecast_year]:
        raise ValueError(
            f"Season features were built at forecast year(s) {stamped} but the "
            f"run asked for {forecast_year}. Pass --forecast_year {stamped[0]} "
            f"or rebuild stage 3 with --forecast-year {forecast_year}."
        )

    print(f"  Pixel-first season features: {len(clim):,} rows from {len(paths)} "
          f"file(s) [{tag}], harvest years "
          f"{int(clim['target_year'].min())}-{int(clim['target_year'].max())}")
    print(f"  Features ({len(feature_cols)}): {','.join(feature_cols)}")

    crop_file = os.path.join(data_dir, f"crop_{crop}_{aoi}.parquet")
    if not os.path.exists(crop_file):
        raise FileNotFoundError(
            f"Crop parquet not found: {crop_file!r}\n"
            f"Available crops for AOI {aoi!r}: {available_crops(data_dir, aoi)}"
        )
    crop_df = _read_parquet(crop_file)
    # Same order as the daily path: drop structural zeros on the RAW yield,
    # before the target transform turns the column into percent changes that are
    # legitimately negative (and before pct_change can divide by a zero).
    crop_df = drop_invalid_yield(crop_df, context=f"{crop}/{aoi}")
    crop_df = _apply_target(crop_df, target)
    # Crop yield is keyed by hindcast_start under the FY1 convention, i.e. the
    # yield for calendar year hindcast_start + 1.
    crop_df["target_year"] = crop_df["hindcast_start"].astype(int) + 1

    merged = clim.drop(columns=["hindcast_start", "forecast_year"]).merge(
        crop_df[["adm2_id", "target_year", "yield"]],
        on=["adm2_id", "target_year"], how="inner")
    # Re-key to this product's lead time, so folds and --study_years mean the
    # same thing here as on the daily path.
    merged["hindcast_start"] = merged["target_year"].astype(int) - forecast_year
    if hindcast_years:
        merged = merged[merged["hindcast_start"].isin(hindcast_years)]

    before = len(merged)
    merged = merged.dropna(subset=["yield"]).dropna(subset=feature_cols)
    if before != len(merged):
        print(f"  Dropped {before - len(merged):,} row(s) with a missing target "
              f"or feature")
    if merged.empty:
        raise ValueError(
            "No rows survived the season-feature/yield merge. Check that the "
            "harvest years in features_admin2 overlap the crop parquet's "
            "years and that adm2_id spellings match."
        )

    lead = ["adm2_id", "adm2", "adm1", "hindcast_start", "target_year", "yield"]
    out = merged[lead + feature_cols].sort_values(
        ["adm2_id", "hindcast_start"]).reset_index(drop=True)
    print(f"  Samples: {len(out):,} over {out['adm2_id'].nunique()} units, "
          f"years {out['hindcast_start'].min()}-{out['hindcast_start'].max()}")
    return out, feature_cols


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
    yield_history_len: int | None = None,
    use_phenology: list | None = None,
    phenology_lag_years: int = 1,
    phenology_suffix: str | None = None,
    phenology_lookback_years: int = 1,
    climate_vars: list | None = None,
    season_blocks: int = SEASON_BLOCKS_DEFAULT,
    gdd_base: float = GDD_BASE_C,
    edd_threshold: float = EDD_THRESHOLD_C,
    gdd_preseason_days: int = GDD_PRESEASON_DAYS,
    gdd_cap: float | None = GDD_CAP_C,
    pixel_features_dir: str | None = None,
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
    temperature + precipitation only. It does not apply to feature_mode "agro",
    whose feature list is fixed by construction.

    feature_mode "agro" replaces the generic mean/std block with the curated
    agronomic set built by build_agro_features: corn growing degree days,
    extreme degree days above edd_threshold (both single-sine, see
    degree_days.py), season mean tasmax/tasmin,
    whole-window humidity statistics, and per-season-block precipitation and
    humidity statistics. season_blocks / gdd_base / edd_threshold / gdd_cap /
    gdd_preseason_days are forwarded to it.
    """
    climate_vars = resolve_climate_vars(climate_vars)

    # feature_mode "pixel" reads a design matrix that stage 3 already built, so
    # it skips the daily load and every builder below. Returns early rather than
    # threading a None daily_df through code that assumes one exists.
    if feature_mode == "pixel":
        adm2_df, feature_cols = load_pixel_season_features(
            crop=crop,
            data_dir=data_dir,
            aoi=aoi,
            climate_model=climate_model,
            forecast_year=forecast_year,
            target=target,
            hindcast_years=hindcast_years,
            features_dir=pixel_features_dir,
        )
        if yield_history:
            adm2_df, hist_cols = _append_yield_history_features(
                adm2_df, data_dir, crop, aoi, forecast_year, yield_history_base,
                max_len=yield_history_len)
            feature_cols = feature_cols + hist_cols
        if use_phenology:
            adm2_df, pheno_cols = _append_phenology_features(
                adm2_df, data_dir, use_phenology, aoi, forecast_year,
                phenology_lag_years, phenology_suffix,
                lookback_years=phenology_lookback_years)
            feature_cols = feature_cols + pheno_cols
        return adm2_df, feature_cols

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
    elif feature_mode == "agro":
        adm2_df, feature_cols = build_agro_features(
            daily_df,
            preseason_buffer=preseason_buffer,
            season_blocks=season_blocks,
            gdd_base=gdd_base,
            edd_threshold=edd_threshold,
            gdd_preseason_days=gdd_preseason_days,
            gdd_cap=gdd_cap,
        )
    else:
        adm2_df, feature_cols = build_timeseries_features(
            daily_df, weather_vars=climate_vars)

    if yield_history:
        adm2_df, hist_cols = _append_yield_history_features(
            adm2_df, data_dir, crop, aoi, forecast_year, yield_history_base,
            max_len=yield_history_len)
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
    max_len: int | None = None,
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
        meta, yseries, base_year=base_year, max_len=max_len)

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
