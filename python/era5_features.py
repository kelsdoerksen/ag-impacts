#!/usr/bin/env python3
"""Per-pixel ERA5 features, then the cropland-weighted admin-2 aggregate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

# The single-sine degree-day kernel is shared with data_loader.py
# feature mode, so gdd/edd mean the same thing whichever pipeline produced them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from degree_days import degree_days_sine  # noqa: E402

DEFAULT_ROOT = "/Users/kdoerkse/Documents/UCT/AgImpacts" # This is local, need to update this to lab/set via export in lab PC
# Lead time in years: how far ahead of its initialisation a run is being used.
# An initialised (hindcast) source is labelled by the November it started from,
# so s{Y} at fy=1 is its first full season and targets harvest year Y+1; the
# same file kind at fy=2 would be a year-older initialisation aimed at the same
# harvest, i.e. a harder forecast.
#
# An observed product has no initialisation and no lead time, so it gets the
# degenerate fy=0: the file for 1983 IS 1983. It carries the column purely so an
# observed feature table drops into code already reading the hindcast schema.
#
# Which of the two a source is, is a property of the source rather than a choice
# to make per run, so it is defaulted here instead of being a flag you have to
# remember. --forecast-year still overrides (e.g. an fy2 DCPP arm).
OBSERVED_SOURCES = {"ERA5", "ERA5-Land", "AgERA5", "MERRA2"}
DEFAULT_FORECAST_YEAR = 0


def default_forecast_year(source: str) -> int:
    """0 for a reanalysis, 1 for an initialised forecast/hindcast run."""
    return 0 if source in OBSERVED_SOURCES else 1

# ============================== feature registry ==============================
class Feature(NamedTuple):
    name: str
    deps: tuple[str, ...]
    reduce: str  # how stage 3 collapses the daily series: "sum" or "mean"
    unit: str
    doc: str
    fn: Callable[..., xr.DataArray]
    defaults: dict


REGISTRY: dict[str, Feature] = {}
def feature(name, deps, reduce, unit, doc, **defaults):
    """Register a per-pixel daily feature.

    `reduce` is how the daily series collapses to one number in the aggregation
     -- "sum" for anything accumulated (degree days, rainfall, day
    counts), "mean" for a state variable (VPD, mean temperature). Adding a
    feature is adding one function here; nothing else in the pipeline changes.
    """

    def wrap(fn):
        REGISTRY[name] = Feature(name, tuple(deps), reduce, unit, doc.strip(), fn, defaults)
        return fn

    return wrap


def _dd(ds, threshold):
    return xr.apply_ufunc(
        degree_days_sine, ds["tasmin_C"], ds["tasmax_C"], kwargs={"threshold": threshold}
    )


@feature(
    "gdd",
    deps=("tasmax_C", "tasmin_C"),
    reduce="sum",
    unit="degC-day",
    tbase=10.0,
    tcap=30.0,
    doc="""
    Growing degree days between tbase and tcap, single-sine method. Computed as the
    degree days above tbase minus those above tcap, so heat beyond tcap stops
    counting rather than being clipped at the daily mean.
    """,
)
def _gdd(ds, tbase=10.0, tcap=30.0):
    return _dd(ds, tbase) - _dd(ds, tcap)


@feature(
    "edd",
    deps=("tasmax_C", "tasmin_C"),
    reduce="sum",
    unit="degC-day",
    tcrit=30.0,
    doc="""
    Extreme degree days above tcrit, single-sine. The damage term
    paired with gdd in the piecewise-linear yield response; tcrit=29-30 for
    maize, ~32-34 for other crops.
    """,
)
def _edd(ds, tcrit=30.0):
    return _dd(ds, tcrit)


@feature(
    "tavg",
    deps=("tasmax_C", "tasmin_C"),
    reduce="mean",
    unit="degC",
    doc="Mean daily temperature, (tasmax + tasmin) / 2.",
)
def _tavg(ds):
    return (ds["tasmax_C"] + ds["tasmin_C"]) / 2.0


@feature(
    "vpd",
    deps=("tasmax_C", "hursmin"),
    reduce="mean",
    unit="kPa",
    doc="""
    Daily maximum vapour pressure deficit, from saturation vapour pressure at
    tasmax (Tetens) and the daily minimum relative humidity.
    """,
)
def _vpd(ds):
    es = 0.6108 * np.exp(17.27 * ds["tasmax_C"] / (ds["tasmax_C"] + 237.3))
    return es * (1.0 - ds["hursmin"] / 100.0)


@feature(
    "pr",
    deps=("pr_mm",),
    reduce="sum",
    unit="mm",
    doc="Precipitation. Summed over the window, so the season value is total rainfall.",
)
def _pr(ds):
    return ds["pr_mm"]


# Comment out dry days feature until the threshold is decided: 1.0 mm is the WMO "wet day" convention
# borrowed as-is, and it is not obvious that it is the right cut for maize water
# stress in any particular AOI. Uncomment to register it again
# @feature(
#     "dry_days",
#     deps=("pr_mm",),
#     reduce="sum",
#     unit="days",
#     threshold=1.0,
#     doc="Days with less than `threshold` mm of rain.",
# )
# def _dry_days(ds, threshold=1.0):
#     return (ds["pr_mm"] < threshold).astype("float32").where(ds["pr_mm"].notnull())


@feature(
    "hot_days",
    deps=("tasmax_C",),
    reduce="sum",
    unit="days",
    threshold=35.0,
    doc="Days with tasmax above `threshold` degC.",
)
def _hot_days(ds, threshold=35.0):
    return (ds["tasmax_C"] > threshold).astype("float32").where(ds["tasmax_C"].notnull())


@feature(
    "tmax",
    deps=("tasmax_C",),
    reduce="mean",
    unit="degC",
    doc="""
    Daily maximum temperature. Reduced by MEAN, so a block value is the mean of
    that window's daily maxima -- not the single hottest day, which is what
    hot_days and edd are for.
    """,
)
def _tmax(ds):
    return ds["tasmax_C"]


@feature(
    "tmin",
    deps=("tasmin_C",),
    reduce="mean",
    unit="degC",
    doc="Daily minimum temperature, averaged over the window.",
)
def _tmin(ds):
    return ds["tasmin_C"]


@feature(
    "pr_mean",
    deps=("pr_mm",),
    reduce="mean",
    unit="mm/day",
    doc="""
    Mean daily precipitation. Paired with `pr` (which accumulates): over a fixed
    window the two are proportional, but blocks differ in length by a day or two,
    so the mean is the one that compares cleanly across them.
    """,
)
def _pr_mean(ds):
    return ds["pr_mm"]


@feature(
    "hurs",
    deps=("hursmin",),
    reduce="mean",
    unit="percent",
    doc="""
    Daily minimum relative humidity, passed through unchanged. An identity
    feature: with tmax, tmin and pr it reconstructs the four raw ERA5 variables
    as an area-weighted admin-2 series, which is what a sequence model wants as
    input. Cheap to carry and it keeps the raw series inside the pixel-first
    ordering rather than requiring a separate aggregate-first pipeline.
    """,
)
def _hurs(ds):
    return ds["hursmin"]


# The four raw ERA5 variables, area-weighted per day: the input a sequence model
# takes. --reduce=daily --features=RAW_FEATURES writes one row per unit per day.
RAW_FEATURES = ["tmax", "tmin", "hurs", "pr"]

# The season default stays as it was: these three exist for --reduce blocks and
# are opt-in via --features, so adding them does not silently change any run.
DEFAULT_FEATURES = ["gdd", "edd", "tavg", "vpd", "pr", "hot_days"]
BLOCK_FEATURES = ["gdd", "edd", "tmax", "tmin", "pr", "pr_mean", "vpd"]


# ================================== paths ====================================


class Layout:
    """Where stage 1 put things, and where stages 2 and 3 put theirs.

    One tree per climate source, because the weight table is grid-specific: it
    encodes cell ids and cropland fractions for one grid, and ERA5's 0.25 deg
    weights are meaningless on the coarser EC-Earth3 grid. Keeping the sources
    in sibling directories makes mixing them a missing-file error rather than a
    silently wrong answer.
    """

    def __init__(self, aoi: str, root: str | None = None, out: str | None = None,
                 source: str = "ERA5", forecast_year: int | None = None):
        self.aoi = aoi
        self.source = source
        self.forecast_year = (default_forecast_year(source)
                              if forecast_year is None else forecast_year)
        self.root = Path(root or os.environ.get("AGIMPACTS_ROOT", DEFAULT_ROOT))
        # ERA5_OUT_DIR is honoured only for ERA5 -- it names one product, and
        # letting a hindcast run inherit it would drop EC-Earth3 cubes into the
        # ERA5 tree next to a weight table built on the wrong grid.
        base = out or (os.environ.get("ERA5_OUT_DIR") if source == "ERA5" else None)
        self.out = Path(base) if base else self.root / f"{aoi}_processed_data" / source
        self.pixel_daily = self.out / "pixel_daily"
        self.grid = self.out / "grid"
        self.pixel_features = self.out / "pixel_features"
        self.admin2 = self.out / "features_admin2"

    def _year_tag(self, year):
        # An initialised run is identified by BOTH its lead time and its
        # initialisation year: the same s-year at two lead times is two
        # different forecasts, verifying against two different harvest years.
        # Leaving the lead out made them share a filename, so a second lead
        # would have been skipped as "already written" and silently inherited
        # the first one's data. Mirrors the fy{n}_s{year} convention the climate
        # parquets already use. An observed run (forecast_year 0) keeps the bare
        # calendar year, so existing ERA5 files are untouched.
        return f"fy{self.forecast_year}_s{year}" if self.forecast_year else f"{year}"

    def cube(self, year):
        if self.source == "ERA5":
            # Spelled as R/era5-pixel-store.R writes it; renaming would orphan
            # cubes already on disk for no gain.
            return self.pixel_daily / f"era5_pixel_{self.aoi}_{year}.nc"
        return self.pixel_daily / f"pixel_{self.source}_{self.aoi}_{self._year_tag(year)}.nc"

    def features(self, year):
        return self.pixel_features / f"features_{self.aoi}_{self._year_tag(year)}.nc"

    @property
    def stage1_hint(self):
        """The stage-1 command that produces what a caller just found missing."""
        if self.source == "ERA5":
            return f"Rscript R/era5-pixel-store.R --aoi={self.aoi}"
        return f"Rscript R/dcpp-pixel-store.R --aoi={self.aoi} --model={self.source}"

    @property
    def weights(self):
        return self.grid / f"weights_{self.aoi}.parquet"

    @property
    def manifest(self):
        return self.grid / f"grid_{self.aoi}.json"


def parse_years(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        elif part:
            years.add(int(part))
    return sorted(years)


# ============================ stage 2: per-pixel =============================


def open_nc(path, **kw) -> xr.Dataset:
    """Open a cube without CF time-decoding surprises.

    A feature carrying units="days" (dry_days, hot_days) is a count, not a
    duration, but xarray's CF rules read any time-like unit string as a
    timedelta and hand back timedelta64 -- which then refuses to multiply by a
    float weight. Turning the decoding off keeps counts numeric.
    """
    return xr.open_dataset(path, decode_timedelta=False, **kw)


def compute_features(ds: xr.Dataset, names: list[str], params: dict) -> xr.Dataset:
    """Per-pixel DAILY features. No spatial or temporal reduction happens here."""
    out = {}
    for name in names:
        spec = REGISTRY[name]
        missing = [d for d in spec.deps if d not in ds]
        if missing:
            raise SystemExit(f"feature '{name}' needs {missing}, which the cube does not have")
        kwargs = {**spec.defaults, **params.get(name, {})}
        da = spec.fn(ds, **kwargs).astype("float32")
        da.attrs = {
            "units": spec.unit,
            "reduce": spec.reduce,  # stage 3 reads this
            "long_name": " ".join(spec.doc.split())[:200],
            "params": json.dumps(kwargs),
        }
        out[name] = da
    return xr.Dataset(out)


def stage_features(lay: Layout, years, names, params, overwrite=False):
    lay.pixel_features.mkdir(parents=True, exist_ok=True)
    for year in years:
        src, dst = lay.cube(year), lay.features(year)
        if not src.exists():
            raise SystemExit(
                f"missing per-pixel cube: {src}\n"
                f"Run: {lay.stage1_hint} {year}"
            )
        if dst.exists() and not overwrite:
            print(f"  {year}: features exist, skipping ({dst.name})")
            continue
        with open_nc(src) as ds:
            feats = compute_features(ds, names, params)
            feats.attrs = {
                "aoi": lay.aoi,
                "year": year,
                "source_cube": src.name,
                "features": ",".join(names),
                "note": "per-pixel daily features; not yet aggregated to admin units",
            }
            enc = {v: {"zlib": True, "complevel": 5} for v in feats.data_vars}
            tmp = dst.with_suffix(".part.nc")
            feats.to_netcdf(tmp, encoding=enc)
        tmp.replace(dst)
        print(f"  {year}: wrote {dst.name} ({dst.stat().st_size / 1e6:.1f} MB, "
              f"{len(names)} features)")


# ========================== stage 3: aggregation =============================

WEIGHT_COLUMNS = {"cropland": "combined_weight", "area": "coverage_frac", "none": None}


def load_weights(lay: Layout, weighting: str) -> pd.DataFrame:
    if not lay.weights.exists():
        raise SystemExit(
            f"missing weight table: {lay.weights}\n"
            f"Run: {lay.stage1_hint} --grid-only <year>\n"
            f"Weights are grid-specific -- a {lay.source} run needs its own, "
            f"not the ones built for another source."
        )
    w = pd.read_parquet(lay.weights)
    col = WEIGHT_COLUMNS[weighting]
    w["w"] = 1.0 if col is None else w[col]
    # A zero weight contributes nothing to either side of the weighted mean, so
    # dropping these is exact, not an approximation -- it just avoids carrying
    # the pixels through every array op.
    w = w[w["w"] > 0].copy()
    # row/col are grid indices and get used to index numpy arrays directly, so
    # they have to be an integer dtype. They arrive as float64: terra returns an
    # NA cell for a polygon lying outside the raster (the "clipped" units the
    # grid manifest counts), arrow stores the column as a nullable int, and
    # pandas reads that back as float. Filtering on weight already removes those
    # rows -- NaN > 0 is False -- but the dtype stays float, and numpy refuses a
    # float index. Drop any survivors defensively, then cast.
    idx = ["row", "col", "cell"]
    bad = w[idx].isna().any(axis=1)
    if bad.any():
        w = w[~bad].copy()
    w[idx] = w[idx].astype("int64")
    # Contiguous groups let the whole aggregation run as one reduceat rather
    # than a per-unit Python loop.
    w.sort_values(["adm2_id", "cell"], inplace=True)
    w.reset_index(drop=True, inplace=True)
    return w


def check_grid(ds: xr.Dataset, lay: Layout, w: pd.DataFrame | None = None):
    """The weights join by grid POSITION, so a mismatch is a silently wrong answer.

    Shape is the cheap half. The dangerous half is orientation: the weight table
    holds terra's 1-based row/col with row 1 = northernmost, and pair_values
    indexes the cube with them directly. If the cube's latitude axis were ever
    written ascending instead of descending, every row index would address the
    mirror-image pixel -- same shape, same value range, plausible-looking output,
    entirely wrong pixel-to-unit assignment. Nothing downstream could detect it.

    The weight table also stores each pixel's lon/lat, so the check is exact
    rather than a convention we hope both sides kept: index the cube's coordinate
    axes the way pair_values will and confirm the coordinates that come back are
    the ones the weights were built from.
    """
    if lay.manifest.exists():
        g = json.loads(lay.manifest.read_text())["grid"]
        got = (ds.sizes.get("latitude"), ds.sizes.get("longitude"))
        if got != (g["nrow"], g["ncol"]):
            raise SystemExit(
                f"grid mismatch: cube is {got[0]}x{got[1]}, weights were built on "
                f"{g['nrow']}x{g['ncol']}. Rebuild with "
                f"'Rscript R/era5-pixel-store.R --aoi={lay.aoi} --grid-only --overwrite <year>'."
            )

    if w is None or not {"lat", "lon"} <= set(w.columns):
        return
    lat = np.asarray(ds["latitude"].values, dtype="float64")
    lon = np.asarray(ds["longitude"].values, dtype="float64")
    rows = w["row"].to_numpy() - 1
    cols = w["col"].to_numpy() - 1
    if rows.max() >= lat.size or cols.max() >= lon.size:
        raise SystemExit(
            f"weight table indexes row {rows.max() + 1}/col {cols.max() + 1}, "
            f"outside the cube's {lat.size}x{lon.size} grid.")
    # Half a cell: comfortably inside a 0.25 deg grid step, but far tighter than
    # the whole-grid span a flip would move a pixel by.
    tol_lat = abs(np.diff(lat)).min() / 2 if lat.size > 1 else 1e-6
    tol_lon = abs(np.diff(lon)).min() / 2 if lon.size > 1 else 1e-6
    d_lat = np.abs(lat[rows] - w["lat"].to_numpy())
    d_lon = np.abs(lon[cols] - w["lon"].to_numpy())
    if d_lat.max() > tol_lat or d_lon.max() > tol_lon:
        flipped = np.abs(lat[::-1][rows] - w["lat"].to_numpy()).max() <= tol_lat
        raise SystemExit(
            f"grid orientation mismatch: cube latitude is "
            f"{'ascending' if lat[0] < lat[-1] else 'descending'} "
            f"({lat[0]:.3f} -> {lat[-1]:.3f}), but the weight table's row indices "
            f"land up to {d_lat.max():.3f} deg lat / {d_lon.max():.3f} deg lon away "
            f"from the pixels they name."
            + ("\nThe latitude axis appears REVERSED relative to the weights -- the "
               "cube was written ascending where the weight table assumes row 1 = "
               "northernmost." if flipped else "")
            + f"\nRebuild the weights on this cube's grid: "
              f"'Rscript R/era5-pixel-store.R --aoi={lay.aoi} --grid-only --overwrite <year>'."
        )


def require_features(ds: xr.Dataset, names, lay: Layout, year):
    """Stage-2 files hold whatever --features was passed when they were written.

    Asking stage 3 for a feature stage 2 never computed is a common sequencing
    slip -- especially after new features are added to the registry -- and the
    raw KeyError from xarray does not say which stage to re-run.
    """
    missing = [n for n in names if n not in ds]
    if missing:
        raise SystemExit(
            f"per-pixel features {missing} are not in {lay.features(year).name}, "
            f"which holds {sorted(str(v) for v in ds.data_vars)}.\n"
            f"Stage 2 writes only the features it was asked for. Re-run it with "
            f"them included:\n"
            f"  python python/era5_features.py --aoi={lay.aoi} --source={lay.source} "
            f"--stage=features --features={','.join(names)} --years=... --overwrite")


def pair_values(da: xr.DataArray, w: pd.DataFrame) -> np.ndarray:
    """(n_pairs, n_time): the feature at each (admin unit, pixel) pair.

    A pixel straddling two units appears once per unit,
    each unit weights it by its own share of that pixel.
    """
    arr = da.transpose("time", "latitude", "longitude").values
    return arr[:, w["row"].to_numpy() - 1, w["col"].to_numpy() - 1].T


def group_starts(w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = w["adm2_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return starts, ids[starts]


def weighted_mean_by_unit(vals: np.ndarray, wt: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Cropland-weighted mean over each unit's pixels, for every time step.

    NaN pixels drop out of both numerator and denominator, so a unit with
    partial coverage is still averaged over what it does have rather than
    NaN
    """
    wcol = wt[:, None]
    valid = ~np.isnan(vals)
    num = np.add.reduceat(np.where(valid, vals * wcol, 0.0), starts, axis=0)
    den = np.add.reduceat(np.where(valid, np.broadcast_to(wcol, vals.shape), 0.0), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def unit_meta(w: pd.DataFrame) -> pd.DataFrame:
    return w.groupby("adm2_id", sort=True)[["adm2", "adm1"]].first().reset_index()


def aggregate_daily(lay: Layout, year, w, names) -> pd.DataFrame:
    """One row per (admin unit, day): the weighted mean of each daily feature.

    Same shape as the existing climate parquets, so this drops straight into
    anything already consuming those. Values are features computed per
    pixel first
    """
    with open_nc(lay.features(year)) as ds:
        check_grid(ds, lay, w)
        starts, unit_ids = group_starts(w)
        wt = w["w"].to_numpy()
        dates = pd.to_datetime(ds["time"].values)

        cols = {}
        for name in names:
            cols[name] = weighted_mean_by_unit(pair_values(ds[name], w), wt, starts)

    n_units, n_time = len(unit_ids), len(dates)
    df = pd.DataFrame(
        {
            "adm2_id": np.repeat(unit_ids, n_time),
            "date": np.tile(dates.values, n_units),
            **{k: v.ravel() for k, v in cols.items()},
        }
    )
    df = df.merge(unit_meta(w), on="adm2_id", how="left")
    df["doy"] = df["date"].dt.dayofyear
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    df["hindcast_start"] = year
    df["forecast_year"] = lay.forecast_year
    df["target_year"] = year + lay.forecast_year
    lead = ["adm2_id", "adm2", "adm1", "hindcast_start", "forecast_year", "target_year",
            "date", "doy", "year_month"]
    return df[lead + names]


def load_crop_calendar(lay: Layout, crop: str, cy_country: str) -> pd.DataFrame:
    """cy-bench sowing/harvest day-of-year, one row per admin unit.

    sos > eos means the season crosses new year, so the window for harvest year Y
    starts in Y-1.
    """
    path = (lay.root / "cy-bench" / "cybench-data" / crop / cy_country
            / f"crop_calendar_{crop}_{cy_country}.csv")
    if not path.exists():
        # The calendar hangs off --root (the DATA root: crop parquets and
        # cy-bench), not --out (the climate store). Those are usually different
        # directories on a shared machine, and --root falling back to a
        # developer's laptop path is not something the bare filename reveals.
        via = ("--root" if os.environ.get("AGIMPACTS_ROOT") is None
               else "$AGIMPACTS_ROOT")
        raise SystemExit(
            f"no crop calendar at {path}\n"
            f"  root = {lay.root}  (from {via})\n"
            f"Set the DATA root -- where cy-bench and the crop parquets live, "
            f"not the climate store passed to --out:\n"
            f"  export AGIMPACTS_ROOT=/path/to/AgImpacts\n"
            f"  # or: --root /path/to/AgImpacts\n"
            f"Find it with: find / -name 'crop_calendar_{crop}_{cy_country}.csv' 2>/dev/null\n"
            f"Or bypass the calendar entirely with --season=SOS:EOS.")
    cal = pd.read_csv(path)
    cal = cal.rename(columns={"adm_id": "adm2_id"})[["adm2_id", "sos", "eos"]]
    return cal.set_index("adm2_id")


def season_windows(unit_ids, cal, year, fixed=None):
    """(start, end) timestamps per unit for harvest year `year`."""
    out = {}
    for uid in unit_ids:
        if fixed is not None:
            sos, eos = fixed
        elif uid in cal.index:
            sos, eos = float(cal.at[uid, "sos"]), float(cal.at[uid, "eos"])
        else:
            continue
        # cy-bench ships a calendar row per unit but not always a filled one;
        # int(round(nan)) raises, which would take down the whole year for one
        # bad unit. Skipping matches the "not in cal.index" branch above: the
        # unit gets no window and drops out with a NaN row.
        if not (np.isfinite(sos) and np.isfinite(eos)):
            continue
        sos_i, eos_i = int(round(sos)), int(round(eos))
        start_year = year - 1 if sos_i > eos_i else year
        start = pd.Timestamp(f"{start_year}-01-01") + pd.Timedelta(days=sos_i - 1)
        end = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=eos_i - 1)
        out[uid] = (start, end)
    return out


def block_windows(unit_ids, cal, year, fixed=None, n_blocks: int = 6):
    """Split each unit's season into n_blocks consecutive windows.

    WHY SEASON-RELATIVE RATHER THAN CALENDAR MONTHS
    Districts sow up to three weeks apart (ZM maize sos spans doy 307-329), so
    "January" is a different growth stage in different districts and averaging
    it across them blurs the very thing these features exist to resolve. Blocks
    are defined as fractions of each unit's OWN season, so m04 is the flowering
    window everywhere.

    WHY EQUAL FRACTIONS RATHER THAN FIXED 30-DAY WINDOWS
    Season length varies by unit (157-203 days for Zambian maize), so fixed
    30-day windows would give some units 6 blocks and others 7 -- a ragged
    design matrix whose trailing column is missing for a subset of units, which
    then drop out of the model entirely. Equal fractions keep the matrix
    rectangular; blocks come out near the requested length and the CLI reports
    the spread.

    WHAT THIS IS NOT
    An equal split is a PROXY for phenology, not phenology. Real crop
    development runs on accumulated heat, so in a cool year the flowering window
    arrives later than a fixed fraction of the season implies. cy-bench gives
    only sos and eos -- no per-stage dates -- so a thermal-time staging would
    need a phenology product this pipeline does not have. Worth revisiting if
    the block features prove informative.
    """
    out = {}
    for uid, (start, end) in season_windows(unit_ids, cal, year, fixed).items():
        total = (end - start).days + 1
        if total < n_blocks:
            continue                      # too short to split meaningfully
        edges = np.linspace(0, total, n_blocks + 1).round().astype(int)
        out[uid] = [(start + pd.Timedelta(days=int(edges[i])),
                     start + pd.Timedelta(days=int(edges[i + 1]) - 1))
                    for i in range(n_blocks)]
    return out


def blocks_from_days(unit_ids, cal, year, fixed, block_days: int) -> int:
    """How many equal blocks give windows closest to `block_days` long."""
    lens = [(e - s).days + 1
            for s, e in season_windows(unit_ids, cal, year, fixed).values()]
    if not lens:
        return 1
    return max(1, int(round(float(np.median(lens)) / block_days)))


def aggregate_blocks(lay: Layout, year, w, names, cal, fixed=None,
                     n_blocks: int = 6) -> pd.DataFrame:
    """One row per (unit, harvest year); one column per (feature, block).

    Same pixel-first ordering as aggregate_season -- the feature was computed per
    pixel upstream, here each pixel's own block window is reduced along time and
    only then are pixels averaged by cropland weight. The only difference is that
    the window is a sixth of the season rather than all of it, so a dry spell at
    flowering is a different number from the same shortfall at maturity.
    """
    starts, unit_ids = group_starts(w)
    bounds = np.r_[starts, len(w)]
    wt = w["w"].to_numpy()
    windows = block_windows(unit_ids, cal, year, fixed, n_blocks)
    if not windows:
        raise SystemExit(
            f"no unit has a usable season window for harvest year {year} -- "
            f"check the crop calendar covers these adm2_ids.")

    wanted = season_source_years(year, lay.forecast_year)
    earliest = min(b[0][0] for b in windows.values())
    if lay.forecast_year == 0 and earliest.year >= year:
        wanted = [year]
    have = [y for y in wanted if lay.features(y).exists()]
    if not have:
        raise SystemExit(
            f"no per-pixel features for harvest year {year} "
            f"(looked for source year(s) {wanted}); run --stage=features first")

    cols = [f"{nm}_m{b + 1:02d}" for nm in names for b in range(n_blocks)]
    results = {c: np.full(len(unit_ids), np.nan) for c in cols}

    with contextlib.ExitStack() as stack:
        dsets = [stack.enter_context(open_nc(lay.features(y))) for y in have]
        ds = xr.concat(dsets, dim="time", data_vars="all",
                       combine_attrs="override") if len(dsets) > 1 else dsets[0]
        check_grid(ds, lay, w)
        # Per FILE, not on the concatenation: xr.concat fills a variable that is
        # absent from one input with NaN, so checking the joined dataset would
        # pass and the missing year would surface only as unexplained NaN rows.
        # Stage-2 files written at different times can hold different --features.
        for _y, _d in zip(have, dsets):
            require_features(_d, names, lay, _y)
        times = pd.to_datetime(ds["time"].values)

        latest = max(b[-1][1] for b in windows.values())
        if earliest < times.min() or latest > times.max():
            missing = [y for y in wanted if y not in have]
            raise SystemExit(
                f"harvest year {year}: season windows run outside the loaded days "
                f"({times.min().date()} to {times.max().date()}); source year(s) "
                f"{missing} are missing. Run --stage=features for them first.")

        for name in names:
            spec = REGISTRY[name]
            vals = pair_values(ds[name].load(), w)
            for gi, uid in enumerate(unit_ids):
                if uid not in windows:
                    continue
                pw = wt[bounds[gi]:bounds[gi + 1]]
                for bi, (b0, b1) in enumerate(windows[uid]):
                    sel = (times >= b0) & (times <= b1)
                    if not sel.any():
                        continue
                    block = vals[bounds[gi]:bounds[gi + 1], sel]
                    per_pixel = season_reduce(block, spec.reduce)
                    ok = ~np.isnan(per_pixel) & (pw > 0)
                    if ok.any():
                        results[f"{name}_m{bi + 1:02d}"][gi] = np.average(
                            per_pixel[ok], weights=pw[ok])
            del vals

    df = pd.DataFrame({"adm2_id": unit_ids, **results})
    df = df.merge(unit_meta(w), on="adm2_id", how="left")
    df["harvest_year"] = year
    df["season_start"] = [windows[u][0][0] if u in windows else pd.NaT for u in unit_ids]
    df["season_end"] = [windows[u][-1][1] if u in windows else pd.NaT for u in unit_ids]
    df["n_blocks"] = n_blocks
    df["hindcast_start"] = year - lay.forecast_year
    df["forecast_year"] = lay.forecast_year
    df["target_year"] = year
    lead = ["adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start", "forecast_year",
            "target_year", "season_start", "season_end", "n_blocks"]
    return df[lead + cols]


def season_source_years(harvest_year: int, forecast_year: int) -> list[int]:
    """Source (cube) years a harvest season is drawn from.

    Observed (fy=0): the cubes are calendar years, so a season that crosses the
    new year straddles two of them and both are read -- they are two slices of
    one continuous reanalysis.

    Hindcast (fy>=1): the s{Y} run spans Nov Y -> Oct Y+1 and so already holds
    its target season end to end. Exactly ONE file is read. Reaching into
    s{Y-1} for a few extra pre-season days would splice a different
    initialisation's weather into the same feature row, which is not a lead time
    the forecast ever had.
    """
    if forecast_year == 0:
        return [harvest_year - 1, harvest_year]
    return [harvest_year - forecast_year]


def harvest_year_candidates(source_years, forecast_year: int) -> list[int]:
    """Harvest years these source years could possibly speak to, covered or not.

    At fy=0 a calendar year can serve its own harvest and (for a cross-year
    calendar) the next one. At fy>=1 an initialisation speaks to exactly one
    harvest year, so a source year is never itself a candidate.
    """
    have = set(source_years)
    if forecast_year == 0:
        return sorted(have | {y + 1 for y in have})
    return sorted(y + forecast_year for y in have)


def harvest_years_from_sources(source_years, forecast_year: int,
                               crosses_new_year: bool) -> list[int]:
    """Harvest years fully covered by the source years on hand.

    `--years` always names SOURCE years -- calendar years for an observed
    product, initialisation years for a hindcast -- because that is what stage 1
    produced and what stage 2 consumes. Stage 3 derives the harvest years rather
    than letting one flag mean two different things in two stages.
    """
    have = set(source_years)
    candidates = harvest_year_candidates(source_years, forecast_year)
    out = []
    for y in candidates:
        if forecast_year == 0 and not crosses_new_year:
            need = {y}                      # a same-year season never reaches back
        else:
            need = set(season_source_years(y, forecast_year))
        if need <= have:
            out.append(y)
    return out


def season_reduce(block: np.ndarray, how: str) -> np.ndarray:
    """Collapse each pixel's season window along time, NaN if it is not covered.

    np.nansum is the wrong tool here and quietly so: over an all-NaN slice it
    returns 0.0, not NaN. A masked pixel would therefore contribute a season
    total of "no degree days, no rain" to its unit's weighted mean, pulling the
    unit's value down in exact proportion to that pixel's weight, with nothing
    NaN anywhere to reveal it. A partially covered window is the same error in
    smaller print: the sum of 135 of a season's 174 days is not the season
    total, it is a number that looks like one.

    So an accumulated feature requires EVERY day of its window. ERA5 has no
    missing days, so a gap means a masked pixel or a window running off the end
    of the loaded cubes -- both of which should surface as NaN and be dropped,
    not be averaged in. State variables (means) only need one valid day, since
    a mean over what is there is still a mean.
    """
    valid = ~np.isnan(block)
    n_valid = valid.sum(axis=1)
    if how == "sum":
        return np.where(n_valid == block.shape[1], np.nansum(block, axis=1), np.nan)
    with np.errstate(invalid="ignore"):
        totals = np.nansum(block, axis=1)
    return np.where(n_valid > 0, totals / np.maximum(n_valid, 1), np.nan)


def aggregate_season(lay: Layout, year, w, names, cal, fixed=None) -> pd.DataFrame:
    """One row per (admin unit, harvest year), accumulated over its own window.

    Per pixel: reduce the daily feature over that unit's growing season (sum for
    accumulated features, mean for state variables). Then take the weighted mean
    across the unit's pixels. Both reductions are linear so the order between
    them is free -- what is *not* free is that the feature itself was already
    computed per pixel, upstream of both.
    """
    starts, unit_ids = group_starts(w)
    bounds = np.r_[starts, len(w)]
    wt = w["w"].to_numpy()
    windows = season_windows(unit_ids, cal, year, fixed)
    if not windows:
        raise SystemExit(
            f"no unit has a usable season window for harvest year {year} -- "
            f"check the crop calendar covers these adm2_ids.")

    # Which source years the windows actually reach into. At fy=0 a cross-year
    # calendar needs year-1 as well, but a same-year one does not, and loading a
    # cube nothing will read doubles the working set for nothing.
    wanted = season_source_years(year, lay.forecast_year)
    if lay.forecast_year == 0 and min(s for s, _ in windows.values()).year >= year:
        wanted = [year]
    have = [y for y in wanted if lay.features(y).exists()]
    if not have:
        raise SystemExit(
            f"no per-pixel features for harvest year {year} "
            f"(looked for source year(s) {wanted}); run --stage=features first")

    # Concatenated rather than open_mfdataset()'d: mfdataset pulls in dask, which
    # buys nothing here (each feature is loaded one at a time below) and is one
    # more thing that has to be installed on the lab machine.
    with contextlib.ExitStack() as stack:
        dsets = [stack.enter_context(open_nc(lay.features(y))) for y in have]
        ds = xr.concat(dsets, dim="time", data_vars="all", combine_attrs="override") if len(dsets) > 1 else dsets[0]
        check_grid(ds, lay, w)
        # Per FILE, not on the concatenation: xr.concat fills a variable that is
        # absent from one input with NaN, so checking the joined dataset would
        # pass and the missing year would surface only as unexplained NaN rows.
        # Stage-2 files written at different times can hold different --features.
        for _y, _d in zip(have, dsets):
            require_features(_d, names, lay, _y)
        times = pd.to_datetime(ds["time"].values)

        # A window running off the end of what is loaded used to be warned about
        # and then silently truncated into a smaller-but-plausible season total.
        # season_reduce now returns NaN for those rows instead, so the warning
        # says what actually happens to them.
        missing = [y for y in wanted if y not in have]
        covered = (times.min(), times.max())
        short = [u for u, (s, e) in windows.items() if s < covered[0] or e > covered[1]]
        if short:
            raise SystemExit(
                f"harvest year {year}: {len(short)} unit(s) have a season window "
                f"outside the loaded days ({covered[0].date()} to {covered[1].date()}); "
                f"source year(s) {missing} are missing.\n"
                f"Run --stage=features for {missing or wanted} first, or drop {year} "
                f"from --years.")

        results = {name: np.full(len(unit_ids), np.nan) for name in names}
        for name in names:
            spec = REGISTRY[name]
            vals = pair_values(ds[name].load(), w)  # (n_pairs, n_time)
            for gi, uid in enumerate(unit_ids):
                if uid not in windows:
                    continue
                start, end = windows[uid]
                sel = (times >= start) & (times <= end)
                if not sel.any():
                    continue
                block = vals[bounds[gi]:bounds[gi + 1], sel]
                # Per pixel first: collapse this unit's window along time...
                per_pixel = season_reduce(block, spec.reduce)
                # ...then weight those per-pixel season values by cropland area.
                pw = wt[bounds[gi]:bounds[gi + 1]]
                ok = ~np.isnan(per_pixel) & (pw > 0)
                if ok.any():
                    results[name][gi] = np.average(per_pixel[ok], weights=pw[ok])
            del vals

    df = pd.DataFrame({"adm2_id": unit_ids, **results})
    df = df.merge(unit_meta(w), on="adm2_id", how="left")
    df["harvest_year"] = year
    df["season_start"] = [windows.get(u, (pd.NaT, pd.NaT))[0] for u in unit_ids]
    df["season_end"] = [windows.get(u, (pd.NaT, pd.NaT))[1] for u in unit_ids]
    df["hindcast_start"] = year - lay.forecast_year
    df["forecast_year"] = lay.forecast_year
    df["target_year"] = year
    lead = ["adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start", "forecast_year",
            "target_year", "season_start", "season_end"]
    return df[lead + names]


def stage_aggregate(lay, years, names, reduce, weighting, crop, cy_country, season,
                    overwrite, block_days: int = 30, n_blocks: int | None = None):
    lay.admin2.mkdir(parents=True, exist_ok=True)
    w = load_weights(lay, weighting)
    print(f"  weights: {len(w)} pixel-unit pairs over {w.adm2_id.nunique()} units "
          f"({weighting}-weighted)")

    fixed = tuple(float(x) for x in season.split(":")) if season else None
    seasonal = reduce in ("season", "blocks")
    cal = pd.DataFrame() if fixed else (
        load_crop_calendar(lay, crop, cy_country) if seasonal else pd.DataFrame())

    if seasonal:
        # `years` arrived as SOURCE years; a season is keyed by its harvest year,
        # and which harvest years those sources can cover depends on the lead
        # time and on whether the crop calendar crosses the new year.
        crosses = bool(fixed[0] > fixed[1]) if fixed else bool((cal["sos"] > cal["eos"]).any())
        harvest_years = harvest_years_from_sources(years, lay.forecast_year, crosses)
        if not harvest_years:
            raise SystemExit(
                f"source years {years[0]}-{years[-1]} cover no complete season at "
                f"forecast year {lay.forecast_year}"
                + (" (the calendar crosses the new year, so harvest year Y needs "
                   "source years Y-1 and Y)" if crosses and not lay.forecast_year else ""))
        dropped = [y for y in harvest_year_candidates(years, lay.forecast_year)
                   if y not in harvest_years]
        print(f"  harvest years {harvest_years[0]}-{harvest_years[-1]} "
              f"({len(harvest_years)}) from source years {years[0]}-{years[-1]}"
              + (f"; {len(dropped)} candidate year(s) lack a complete season" if dropped else ""))
        years = harvest_years

    # Block count is fixed ONCE, from the first harvest year, and reused for
    # every year of the run. Letting it float per year would give different
    # years different columns, which is not a design matrix.
    if reduce == "blocks":
        _, uids = group_starts(w)
        if n_blocks is None:
            n_blocks = blocks_from_days(uids, cal, years[0], fixed, block_days)
        lens = [(e - s).days + 1 for s, e in
                season_windows(uids, cal, years[0], fixed).values()]
        if lens:
            print(f"  {n_blocks} blocks/season; season length "
                  f"{min(lens)}-{max(lens)} days -> blocks of "
                  f"~{min(lens)//n_blocks}-{max(lens)//n_blocks} days")

    # Lead-time marker on the admin-2 outputs too: two leads verifying against
    # the same harvest year are two different predictions and must not share a
    # filename. Empty for an observed run, so ERA5 filenames do not change.
    fy_tag = f"fy{lay.forecast_year}_" if lay.forecast_year else ""
    for year in years:
        if reduce == "daily":
            out = lay.admin2 / f"features_{lay.aoi}_{fy_tag}daily_{year}.parquet"
        else:
            tag = "fixed" if fixed else crop
            kind = "blocks" if reduce == "blocks" else "season"
            out = lay.admin2 / f"features_{lay.aoi}_{tag}_{fy_tag}{kind}_{year}.parquet"
        if out.exists() and not overwrite:
            print(f"  {year}: exists, skipping ({out.name})")
            continue
        if reduce == "daily":
            df = aggregate_daily(lay, year, w, names)
        elif reduce == "blocks":
            df = aggregate_blocks(lay, year, w, names, cal, fixed, n_blocks)
        else:
            df = aggregate_season(lay, year, w, names, cal, fixed)
        df.to_parquet(out, index=False)
        value_cols = [c for c in df.columns if c.split("_m")[0] in names or c in names]
        filled = df[value_cols].notna().all(axis=1).sum()
        print(f"  {year}: wrote {out.name} ({len(df)} rows, {filled} complete)")


# ==================================== cli ====================================


def parse_params(pairs):
    """--param gdd.tbase=8 --param edd.tcrit=29"""
    out: dict[str, dict] = {}
    for p in pairs or []:
        key, _, val = p.partition("=")
        feat, _, arg = key.partition(".")
        if not arg or feat not in REGISTRY:
            raise SystemExit(f"--param expects FEATURE.arg=value with a known feature, got '{p}'")
        out.setdefault(feat, {})[arg] = float(val)
    return out


def list_features():
    print(f"{'feature':<10} {'reduce':<7} {'unit':<10} {'params':<28} description")
    print("-" * 100)
    for name, f in REGISTRY.items():
        params = ", ".join(f"{k}={v}" for k, v in f.defaults.items()) or "-"
        print(f"{name:<10} {f.reduce:<7} {f.unit:<10} {params:<28} {' '.join(f.doc.split())[:60]}")
    print(f"\ndefault set: {','.join(DEFAULT_FEATURES)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", help="AOI tag, matching R/era5-pixel-store.R (e.g. ZM, CONUS)")
    p.add_argument("--source", default="ERA5",
                   help="climate source: the stage-1 tree to read and the folder "
                        "under {AOI}_processed_data. ERA5 (default) or a hindcast "
                        "model such as EC-Earth3. Each source has its own weight "
                        "table, since those are built on that source's grid.")
    p.add_argument("--forecast-year", type=int, default=None,
                   help="Lead time in years -- how far ahead of its initialisation "
                        "a run is used. Defaults from --source and rarely needs "
                        "setting: 0 for a reanalysis (the file for 1983 IS 1983), "
                        "1 for an initialised run (s1982, started Nov 1982, is the "
                        "first full season ahead and targets harvest 1983). Pass 2 "
                        "for a longer-lead DCPP arm: s1981 aimed at the same 1983 "
                        "harvest.")
    p.add_argument("--years", help="SOURCE years -- calendar years when "
                        "--forecast-year=0, initialisation (s) years otherwise. "
                        "1980-2017, or 1991,1995. With --reduce=season the "
                        "harvest years covered are derived from these.")
    p.add_argument("--stage", choices=["features", "aggregate", "both"], default="both")
    p.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                   help="comma-separated; see --list-features")
    p.add_argument("--param", action="append", metavar="FEAT.arg=value",
                   help="override a feature parameter, e.g. gdd.tbase=8")
    p.add_argument("--reduce", choices=["daily", "season", "blocks"], default="daily",
                   help="daily: weighted mean per unit per day. season: accumulate over "
                        "the whole crop-calendar window, one row per unit per harvest "
                        "year. blocks: the same, but split into consecutive "
                        "season-relative windows (~--block-days long), giving one column "
                        "per feature per block (gdd_m01, pr_m04, ...). Season totals "
                        "cannot separate a dry spell at flowering from the same shortfall "
                        "at maturity; blocks can.")
    p.add_argument("--block-days", type=int, default=30, metavar="N",
                   help="Target block length for --reduce=blocks. The season is split "
                        "into equal fractions, so actual blocks land near this. "
                        "Default: 30.")
    p.add_argument("--n-blocks", type=int, default=None, metavar="N",
                   help="Exact number of blocks, overriding --block-days.")
    p.add_argument("--weight", choices=list(WEIGHT_COLUMNS), default="cropland",
                   help="cropland: cropland-area weighted (default). area: polygon overlap "
                        "only. none: unweighted pixel mean")
    p.add_argument("--crop", default="maize", help="crop calendar to use with --reduce=season")
    p.add_argument("--cy-country", help="cy-bench country code (defaults to --aoi; US for CONUS)")
    p.add_argument("--season", metavar="SOS:EOS",
                   help="fixed day-of-year window for every unit, instead of a crop calendar")
    p.add_argument("--root", help="AgImpacts root (default $AGIMPACTS_ROOT)")
    p.add_argument("--out", help="ERA5 output dir (default $ERA5_OUT_DIR or {root}/{AOI}_processed_data/ERA5)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list-features", action="store_true")
    a = p.parse_args(argv)

    if a.list_features:
        list_features()
        return 0
    if not a.aoi or not a.years:
        p.error("--aoi and --years are required (or use --list-features)")

    names = [n.strip() for n in a.features.split(",") if n.strip()]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        p.error(f"unknown feature(s): {unknown}. See --list-features")

    lay = Layout(a.aoi, a.root, a.out, source=a.source,
                 forecast_year=a.forecast_year)
    years = parse_years(a.years)
    params = parse_params(a.param)
    cy_country = a.cy_country or ("US" if a.aoi == "CONUS" else a.aoi)

    fy = lay.forecast_year
    reading = ("observed calendar years; the file for a year IS that year"
               if fy == 0 else
               f"initialisation years; s{{Y}} targets harvest year Y+{fy}")
    print(f"{a.source} features - AOI {a.aoi}, features: {','.join(names)}")
    print(f"  forecast year {fy} ({reading})")
    print(f"  --years {years[0]}-{years[-1]} read as {reading.split(';')[0]}")
    print(f"  root: {lay.root}"
          + ("" if lay.root.exists() else "   <-- DOES NOT EXIST"))
    print(f"  out: {lay.out}")
    if not lay.root.exists() and a.reduce in ("season", "blocks") and not a.season:
        # Stage 2 can take a long time; failing on a missing calendar only after
        # it finishes wastes all of it.
        print(f"  ! --root does not exist, and --reduce={a.reduce} needs the "
              f"cy-bench crop calendar under it.\n"
              f"    Set AGIMPACTS_ROOT or --root to the data root before this "
              f"reaches stage 3.")

    if a.stage in ("features", "both"):
        print("\nStage 2: per-pixel daily features")
        stage_features(lay, years, names, params, a.overwrite)
    if a.stage in ("aggregate", "both"):
        print(f"\nStage 3: admin-2 aggregation ({a.reduce})")
        stage_aggregate(lay, years, names, a.reduce, a.weight, a.crop, cy_country,
                        a.season, a.overwrite, a.block_days, a.n_blocks)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
