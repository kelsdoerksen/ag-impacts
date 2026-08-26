#!/usr/bin/env python3
"""Per-pixel ERA5 features, then the cropland-weighted admin-2 aggregate.

WHY THE ORDER MATTERS

R/era5-sample-gen.R aggregates first: it writes the cropland-weighted admin-2
mean of the raw daily variables, so anything computed from that parquet is a
feature *of the admin-2 mean*. That is only correct for features linear in the
inputs, and the interesting ones are not::

    mean_i( max(0, T_i - Tbase) )  !=  max(0, mean_i(T_i) - Tbase)

Averaging first clips the tails, so a unit holding both a cool and a hot pixel
loses the heat the hot pixel actually accumulated. Degree days, extreme degree
days, dry-day counts, VPD and every threshold feature are affected, which is why
the causal crop literature computes the feature per pixel and only then takes
the cropland-area-weighted mean. This script is stages 2 and 3 of that::

    1. R/era5-pixel-store.R      daily grids -> per-pixel daily cube (NetCDF)
                                             -> pixel/admin-2 weights (parquet)
    2. this script --stage=features    cube -> per-pixel DAILY features
    3. this script --stage=aggregate   features + weights -> admin-2 parquet

Stage 2 stays per-pixel and daily. It does not collapse time, because the
temporal window is an admin-2 property (a crop calendar's sowing and harvest
dates differ by unit), so the season reduction belongs with the aggregation in
stage 3.

Examples::

    python era5_features.py --aoi=ZM --years=1980-2017 --stage=both
    python era5_features.py --aoi=ZM --years=1991 --stage=aggregate --reduce=season --crop=maize
    python era5_features.py --list-features
    python era5_features.py --aoi=ZM --years=1991 --features=gdd,edd --param gdd.tbase=8

Needs: xarray, netCDF4, pandas, numpy, pyarrow.
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

DEFAULT_ROOT = "/Users/kdoerkse/Documents/UCT/AgImpacts"
# Fills the hindcast columns the EC-Earth3 parquets carry, so an ERA5 feature
# table is a drop-in for anything already reading those. ERA5 is observed: no
# initialisation year, no lead time.
FORECAST_YEAR = 0


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

    `reduce` is how the daily series collapses to one number per season in
    stage 3 -- "sum" for anything accumulated (degree days, rainfall, day
    counts), "mean" for a state variable (VPD, mean temperature). Adding a
    feature is adding one function here; nothing else in the pipeline changes.
    """

    def wrap(fn):
        REGISTRY[name] = Feature(name, tuple(deps), reduce, unit, doc.strip(), fn, defaults)
        return fn

    return wrap


def degree_days_sine(tmin, tmax, threshold):
    """Degree days above `threshold` for one day, single-sine method.

    Integrates max(T - threshold, 0) over the day, taking the diurnal cycle to
    be a sine wave running between tmin and tmax. This is the standard
    construction in the crop-impacts literature (Schlenker & Roberts and
    descendants) and is meaningfully different from the naive
    max(mean(tmin, tmax) - threshold, 0) whenever the threshold falls inside the
    day's range -- which for a heat threshold is exactly the case that matters.
    """
    tmin = np.asarray(tmin, dtype="float64")
    tmax = np.asarray(tmax, dtype="float64")
    mid = (tmax + tmin) / 2.0
    amp = (tmax - tmin) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip((threshold - mid) / np.where(amp > 0, amp, np.nan), -1.0, 1.0)
        theta = np.arcsin(ratio)
        # Area of the sine above the threshold, for a threshold strictly inside
        # the day's range.
        partial = ((mid - threshold) * (np.pi / 2.0 - theta) + amp * np.cos(theta)) / np.pi

    out = np.where(
        threshold <= tmin, mid - threshold, np.where(threshold >= tmax, 0.0, partial)
    )
    # A day with no diurnal range degenerates to a point, not a sine.
    return np.where(amp <= 0, np.maximum(mid - threshold, 0.0), out)


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
    Growing degree days between tbase and tcap, single-sine. Computed as the
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
    Extreme (killing) degree days above tcrit, single-sine. The damage term
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
    tasmax (Tetens) and the daily minimum relative humidity. Pairing the day's
    hottest, driest moments is deliberate -- it is the peak atmospheric demand,
    not the daily average, that drives stomatal closure.
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


@feature(
    "dry_days",
    deps=("pr_mm",),
    reduce="sum",
    unit="days",
    threshold=1.0,
    doc="Days with less than `threshold` mm of rain.",
)
def _dry_days(ds, threshold=1.0):
    return (ds["pr_mm"] < threshold).astype("float32").where(ds["pr_mm"].notnull())


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


DEFAULT_FEATURES = ["gdd", "edd", "tavg", "vpd", "pr", "dry_days", "hot_days"]


# ================================== paths ====================================


class Layout:
    """Where stage 1 put things, and where stages 2 and 3 put theirs."""

    def __init__(self, aoi: str, root: str | None = None, out: str | None = None):
        self.aoi = aoi
        self.root = Path(root or os.environ.get("AGIMPACTS_ROOT", DEFAULT_ROOT))
        base = out or os.environ.get("ERA5_OUT_DIR")
        self.out = Path(base) if base else self.root / f"{aoi}_processed_data" / "ERA5"
        self.pixel_daily = self.out / "pixel_daily"
        self.grid = self.out / "grid"
        self.pixel_features = self.out / "pixel_features"
        self.admin2 = self.out / "features_admin2"

    def cube(self, year):
        return self.pixel_daily / f"era5_pixel_{self.aoi}_{year}.nc"

    def features(self, year):
        return self.pixel_features / f"features_{self.aoi}_{year}.nc"

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
                f"Run: Rscript R/era5-pixel-store.R --aoi={lay.aoi} {year}"
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
            f"Run: Rscript R/era5-pixel-store.R --aoi={lay.aoi} --grid-only <year>"
        )
    w = pd.read_parquet(lay.weights)
    col = WEIGHT_COLUMNS[weighting]
    w["w"] = 1.0 if col is None else w[col]
    # A zero weight contributes nothing to either side of the weighted mean, so
    # dropping these is exact, not an approximation -- it just avoids carrying
    # the pixels through every array op.
    w = w[w["w"] > 0].copy()
    # Contiguous groups let the whole aggregation run as one reduceat rather
    # than a per-unit Python loop.
    w.sort_values(["adm2_id", "cell"], inplace=True)
    w.reset_index(drop=True, inplace=True)
    return w


def check_grid(ds: xr.Dataset, lay: Layout):
    """The weights join by grid position, so a shape mismatch is a wrong answer."""
    if not lay.manifest.exists():
        return
    g = json.loads(lay.manifest.read_text())["grid"]
    got = (ds.sizes.get("latitude"), ds.sizes.get("longitude"))
    if got != (g["nrow"], g["ncol"]):
        raise SystemExit(
            f"grid mismatch: cube is {got[0]}x{got[1]}, weights were built on "
            f"{g['nrow']}x{g['ncol']}. Rebuild with "
            f"'Rscript R/era5-pixel-store.R --aoi={lay.aoi} --grid-only --overwrite <year>'."
        )


def pair_values(da: xr.DataArray, w: pd.DataFrame) -> np.ndarray:
    """(n_pairs, n_time): the feature at each (admin unit, pixel) pair.

    A pixel straddling two units appears once per unit, which is the point --
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
    poisoned to NaN.
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
    anything already consuming those -- but the values are features computed per
    pixel first, which is the whole point of the rework.
    """
    with open_nc(lay.features(year)) as ds:
        check_grid(ds, lay)
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
    df["forecast_year"] = FORECAST_YEAR
    df["target_year"] = year
    lead = ["adm2_id", "adm2", "adm1", "hindcast_start", "forecast_year", "target_year",
            "date", "doy", "year_month"]
    return df[lead + names]


def load_crop_calendar(lay: Layout, crop: str, cy_country: str) -> pd.DataFrame:
    """cy-bench sowing/harvest day-of-year, one row per admin unit.

    sos > eos means the season crosses new year (Zambian maize is sown around
    doy 327 and harvested around doy 135), so the window for harvest year Y
    starts in Y-1. Handled explicitly below rather than silently producing an
    empty window.
    """
    path = (lay.root / "cy-bench" / "cybench-data" / crop / cy_country
            / f"crop_calendar_{crop}_{cy_country}.csv")
    if not path.exists():
        raise SystemExit(f"no crop calendar at {path}\nUse --season=SOS:EOS for a fixed window.")
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
        sos_i, eos_i = int(round(sos)), int(round(eos))
        start_year = year - 1 if sos_i > eos_i else year
        start = pd.Timestamp(f"{start_year}-01-01") + pd.Timedelta(days=sos_i - 1)
        end = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=eos_i - 1)
        out[uid] = (start, end)
    return out


def aggregate_season(lay: Layout, year, w, names, cal, fixed=None) -> pd.DataFrame:
    """One row per (admin unit, harvest year), accumulated over its own window.

    Per pixel: reduce the daily feature over that unit's growing season (sum for
    accumulated features, mean for state variables). Then take the weighted mean
    across the unit's pixels. Both reductions are linear so the order between
    them is free -- what is *not* free is that the feature itself was already
    computed per pixel, upstream of both.
    """
    paths = [lay.features(y) for y in (year - 1, year) if lay.features(y).exists()]
    if not paths:
        raise SystemExit(f"no per-pixel features for {year}; run --stage=features first")
    # Concatenated rather than open_mfdataset()'d: mfdataset pulls in dask, which
    # buys nothing here (each feature is loaded one at a time below) and is one
    # more thing that has to be installed on the lab machine.
    with contextlib.ExitStack() as stack:
        dsets = [stack.enter_context(open_nc(pth)) for pth in paths]
        ds = xr.concat(dsets, dim="time", data_vars="all", combine_attrs="override") if len(dsets) > 1 else dsets[0]
        check_grid(ds, lay)
        times = pd.to_datetime(ds["time"].values)
        starts, unit_ids = group_starts(w)
        bounds = np.r_[starts, len(w)]
        wt = w["w"].to_numpy()
        windows = season_windows(unit_ids, cal, year, fixed)

        wrapping = [u for u, (s, _) in windows.items() if s.year < year]
        if wrapping and lay.features(year - 1) not in paths:
            print(f"  ! {len(wrapping)} unit(s) have a season starting in {year - 1}, "
                  f"whose features are missing -- those rows will be truncated.")

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
                per_pixel = (np.nansum if spec.reduce == "sum" else np.nanmean)(block, axis=1)
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
    df["hindcast_start"] = year
    df["forecast_year"] = FORECAST_YEAR
    df["target_year"] = year
    lead = ["adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start", "forecast_year",
            "target_year", "season_start", "season_end"]
    return df[lead + names]


def stage_aggregate(lay, years, names, reduce, weighting, crop, cy_country, season, overwrite):
    lay.admin2.mkdir(parents=True, exist_ok=True)
    w = load_weights(lay, weighting)
    print(f"  weights: {len(w)} pixel-unit pairs over {w.adm2_id.nunique()} units "
          f"({weighting}-weighted)")

    fixed = tuple(float(x) for x in season.split(":")) if season else None
    cal = pd.DataFrame() if fixed else (
        load_crop_calendar(lay, crop, cy_country) if reduce == "season" else pd.DataFrame())

    for year in years:
        if reduce == "daily":
            out = lay.admin2 / f"features_{lay.aoi}_daily_{year}.parquet"
        else:
            tag = "fixed" if fixed else crop
            out = lay.admin2 / f"features_{lay.aoi}_{tag}_season_{year}.parquet"
        if out.exists() and not overwrite:
            print(f"  {year}: exists, skipping ({out.name})")
            continue
        df = (aggregate_daily(lay, year, w, names) if reduce == "daily"
              else aggregate_season(lay, year, w, names, cal, fixed))
        df.to_parquet(out, index=False)
        filled = df[names].notna().all(axis=1).sum()
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
    p.add_argument("--years", help="1980-2017, or 1991,1995")
    p.add_argument("--stage", choices=["features", "aggregate", "both"], default="both")
    p.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                   help="comma-separated; see --list-features")
    p.add_argument("--param", action="append", metavar="FEAT.arg=value",
                   help="override a feature parameter, e.g. gdd.tbase=8")
    p.add_argument("--reduce", choices=["daily", "season"], default="daily",
                   help="daily: weighted mean per unit per day. season: accumulate over "
                        "the crop-calendar window, one row per unit per harvest year")
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

    lay = Layout(a.aoi, a.root, a.out)
    years = parse_years(a.years)
    params = parse_params(a.param)
    cy_country = a.cy_country or ("US" if a.aoi == "CONUS" else a.aoi)

    print(f"ERA5 features - AOI {a.aoi}, years {years[0]}-{years[-1]}, "
          f"features: {','.join(names)}")
    print(f"  out: {lay.out}")

    if a.stage in ("features", "both"):
        print("\nStage 2: per-pixel daily features")
        stage_features(lay, years, names, params, a.overwrite)
    if a.stage in ("aggregate", "both"):
        print(f"\nStage 3: admin-2 aggregation ({a.reduce})")
        stage_aggregate(lay, years, names, a.reduce, a.weight, a.crop, cy_country,
                        a.season, a.overwrite)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
