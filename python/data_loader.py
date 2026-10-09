"""
Load, merge, and featurise climate + crop parquets for yield model training.

Expected files in data_dir:
  Climate:  {climate_model}/climate_{aoi}_{climate_model}_s{hindcast_start}-{ens_member}.parquet
            (the original CONUS series omits the AOI token; see "Climate file
            resolution" below for every convention accepted)
  Crop:     crop_{crop}_{aoi}.parquet

Feature modes
-------------
Both read a design matrix that stage 3 has already built, so the feature
definition lives in python/era5_features.py and this module only joins it to
the yield targets. Features are computed per pixel and only then averaged over
cropland -- see the README for why that ordering is the whole point.

pixel
    One row per (adm2_id, harvest_year). Features are whole-season totals or
    means per climate variable, from features_{AOI}_{...}_season_{year}.parquet.

pixel_blocks
    The same, split into consecutive season-relative windows, giving one column
    per feature per block (gdd_m01, pr_m04, ...). A season total cannot separate
    a dry spell at flowering from the same shortfall at maturity; blocks can.

Public API
----------
load_and_prepare(...)      → (adm2_df, feature_cols)   ready for sklearn
load_pixel_season_features → (adm2_df, feature_cols)   stage-3 design matrix
load_pixel_daily(...)      → daily DataFrame            stage-2 per-pixel series
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
FEATURE_MODES   = ["pixel", "pixel_blocks"]

# ── degree-day defaults ──────────────────────────────────────────────────────
# Defaults for the degree-day features, mirrored by era5_features.py's registry
# so a run that overrides them here and a stage-2 cube built elsewhere cannot
# disagree silently.
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
    "n_blocks",
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
                              crop: str, features_dir: str | None = None,
                              kind: str = "season") -> tuple[list, str | None]:
    """Stage-3 parquets for this AOI/crop.

    `kind` is "season" (one column per feature, accumulated over the whole
    growing window) or "blocks" (one column per feature per season-relative
    window, e.g. pr_m04). Tries the crop tag under both spellings, then "fixed"
    for a run that used era5_features.py --season SOS:EOS instead of a calendar.
    """
    tags = [crop]
    if crop in CY_CROP_NAMES:
        tags.append(CY_CROP_NAMES[crop])
    tags.append("fixed")
    for fdir in pixel_features_dirs(data_dir, climate_model, features_dir):
        for tag in tags:
            paths = sorted(glob.glob(
                os.path.join(fdir, f"features_{aoi}_{tag}_{kind}_*.parquet")))
            if paths:
                return paths, tag
    return [], None


# The stage-3 daily store names its columns after the FEATURES that produced
# them (tmax, tmin, hurs, pr -- the identity features in era5_features.py). Every
# daily consumer here, and the PatchTST path, was written against the raw ERA5
# column names. Rename on read rather than in era5_features: the feature names
# are correct on their own side, and a compatibility shim belongs with the code
# that needs compatibility.
_RAW_FEATURE_ALIASES = {
    "tmax": "tasmax_C",
    "tmin": "tasmin_C",
    "hurs": "hursmin",
    "pr":   "pr_mm",
}


def find_pixel_daily_files(data_dir: str, climate_model: str, aoi: str,
                           features_dir: str | None = None,
                           years: list | None = None) -> list:
    """Stage-3 daily parquets (era5_features.py --reduce=daily) for this AOI."""
    out = []
    for fdir in pixel_features_dirs(data_dir, climate_model, features_dir):
        paths = sorted(glob.glob(
            os.path.join(fdir, f"features_{aoi}_daily_*.parquet")))
        if paths:
            out = paths
            break
    if years:
        want = set(int(y) for y in years)
        # Keep the neighbouring year too: a cross-year season reaches back into
        # the previous calendar file, and the window filter downstream discards
        # whatever it does not need.
        want |= {y - 1 for y in want} | {y + 1 for y in want}
        out = [p for p in out
               if int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]) in want]
    return out


def load_pixel_daily(data_dir: str, climate_model: str, aoi: str,
                     features_dir: str | None = None,
                     years: list | None = None) -> pd.DataFrame:
    """Daily admin-2 climate from the PIXEL-FIRST store, in raw-column schema.

    Same shape as the aggregate-first climate parquets, so it is a drop-in for
    them -- but the numbers differ in ordering: these were area-weighted from
    per-pixel values, the others were computed on an already-averaged unit. For
    a linear pass-through variable (tmax, pr) the two orderings agree exactly;
    they diverge only once something nonlinear is applied downstream, which is
    the whole reason this store exists.
    """
    paths = find_pixel_daily_files(data_dir, climate_model, aoi, features_dir, years)
    if not paths:
        looked = pixel_features_dirs(data_dir, climate_model, features_dir)
        raise FileNotFoundError(
            f"No pixel-first DAILY features for AOI {aoi!r}.\n"
            f"Looked in:\n" + "".join(f"  {d}\n" for d in looked) +
            f"Build them with:\n"
            f"  python python/era5_features.py --aoi={aoi} --source={climate_model} "
            f"--years=<years> --reduce=daily --features=tmax,tmin,hurs,pr"
        )
    df = pd.concat([_read_parquet(pth) for pth in paths], ignore_index=True)
    have = {k: v for k, v in _RAW_FEATURE_ALIASES.items() if k in df.columns}
    df = df.rename(columns=have)
    missing = [v for v in _RAW_FEATURE_ALIASES.values() if v not in df.columns]
    if missing:
        raise ValueError(
            f"Pixel-first daily store is missing {missing}. It holds "
            f"{[c for c in df.columns if c not in ('adm2_id','adm2','adm1','date','doy','year_month','hindcast_start','forecast_year','target_year')]}.\n"
            f"Rebuild stage 3 with --features=tmax,tmin,hurs,pr (identity "
            f"features that reproduce the raw ERA5 variables)."
        )
    print(f"  Pixel-first daily store: {len(df):,} rows from {len(paths)} file(s)")
    return df


def load_pixel_season_features(
    crop: str,
    data_dir: str,
    aoi: str,
    climate_model: str,
    forecast_year: int,
    target: str,
    hindcast_years: list | None = None,
    features_dir: str | None = None,
    kind: str = "season",
) -> tuple[pd.DataFrame, list]:
    """Read stage-3 features and join the yield target onto them.

    Returns (adm2_df, feature_cols) in the same shape as the other feature
    modes: one row per (adm2_id, hindcast_start) carrying adm2_id, adm2, adm1,
    hindcast_start, target_year, yield and the feature columns.
    """
    paths, tag = find_season_feature_files(
        data_dir, climate_model, aoi, crop, features_dir, kind)
    if not paths:
        looked = pixel_features_dirs(data_dir, climate_model, features_dir)
        raise FileNotFoundError(
            f"No pixel-first {kind} features for AOI {aoi!r}, crop {crop!r}.\n"
            f"Looked in:\n" + "".join(f"  {d}\n" for d in looked) +
            f"Point --pixel_features_dir (or ERA5_OUT_DIR) at the store, or "
            f"build it with:\n"
            f"  Rscript R/era5-pixel-store.R --aoi={aoi} <years>\n"
            f"  python python/era5_features.py --aoi={aoi} --source={climate_model} "
            f"--years=<years> --reduce={kind} "
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

    print(f"  Pixel-first {kind} features: {len(clim):,} rows from {len(paths)} "
          f"file(s) [{tag}], harvest years "
          f"{int(clim['target_year'].min())}-{int(clim['target_year'].max())}")
    print(f"  Features ({len(feature_cols)}): {','.join(feature_cols)}")

    crop_file = os.path.join(data_dir, f"crop_{crop}_{aoi}.parquet")
    if not os.path.exists(crop_file):
        # This mode reads two directories -- the climate store (resolved above,
        # possibly via --pixel_features_dir) and --data_dir for the yield table.
        # Pointing both at the climate store is the easy mistake, and the bare
        # "not found" gives no hint that they are different things.
        hint = ""
        if os.path.abspath(os.path.dirname(paths[0])).startswith(
                os.path.abspath(data_dir)):
            hint = (f"\n--data_dir is the AOI processed-data directory holding the "
                    f"cy-bench yield table, NOT the climate store. It looks like "
                    f"both are pointing at the same place; pass the climate store "
                    f"as --pixel_features_dir instead.")
        raise FileNotFoundError(
            f"Crop parquet not found: {crop_file!r}\n"
            f"Available crops for AOI {aoi!r}: {available_crops(data_dir, aoi)}"
            f"{hint}\n"
            f"Build it with: Rscript R/crop-data-gen-cybench.R {aoi}"
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
    feature_mode: str = "pixel_blocks",
    climate_model: str = "EC-Earth3",
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
    Only the stage-3 design matrix is read here; the feature set and its
    thresholds were fixed when era5_features.py stage 2 ran, so climate_vars
    selects which of those columns to keep rather than how they were built.
    """
    climate_vars = resolve_climate_vars(climate_vars)

    # feature_mode "pixel" reads a design matrix that stage 3 already built, so
    # it skips the daily load and every builder below. Returns early rather than
    # threading a None daily_df through code that assumes one exists.
    if feature_mode in ("pixel", "pixel_blocks"):
        adm2_df, feature_cols = load_pixel_season_features(
            crop=crop,
            data_dir=data_dir,
            aoi=aoi,
            climate_model=climate_model,
            forecast_year=forecast_year,
            target=target,
            hindcast_years=hindcast_years,
            features_dir=pixel_features_dir,
            kind="blocks" if feature_mode == "pixel_blocks" else "season",
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
