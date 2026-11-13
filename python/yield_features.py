"""
Assemble the yield design matrix from ONE named feature source.

The stage-3 stores are the finished design matrices -- one row per
(admin-2 unit, harvest year), features already computed per pixel and then
averaged over cropland. This module's whole job is to pick which store to read,
attach the yield target and the cy-bench covariates, and hand back a frame that
is keyed identically no matter which source produced it. That last property is
the point: an ERA5 run and a DCPP run must differ ONLY in the climate columns,
or the comparison measures plumbing rather than forecast skill.

SOURCES

    era5                observed reanalysis, forecast_year 0. The verification
                        target and the skill ceiling.
    dcpp:<member>       one EC-Earth3 realisation at lead F, e.g.
                        dcpp:r1i4p1f1.
    dcpp:ens            the ensemble mean of every member found on disk,
                        averaged FEATURE BY FEATURE after stage 3.

Why the ensemble mean is taken here and not earlier: averaging the members'
daily fields first and then computing GDD would average away the tails that a
threshold feature exists to measure -- the same mean-of-nonlinear argument that
makes the whole pipeline pixel-first. Averaging the finished features is the
defensible order, and it is what compare_dcpp_era5.py scores.

TREES ON DISK

    era5        {era5_root}/{AOI}/features_admin2/
    dcpp        {dcpp_root}/{AOI}_processed_data/EC-Earth3_{member}/features_admin2/
    targets     {share}/{AOI}_processed_data/crop_{crop}_{AOI}.parquet
                {share}/{AOI}_processed_data/phenology/phenology_{stem}_{AOI}.parquet

The filenames carry the lead time (features_{AOI}_{crop}_fy{F}_season_{Y}.parquet,
with the fy token absent for observed runs), so a store cannot be read at the
wrong lead by accident.

KEYS

Everything here is keyed on (adm2_id, harvest_year). harvest_year is the
calendar year the season ENDS in, which is what stage 3 stamps and what
compare_dcpp_era5.py joins on. The cy-bench crop tables are keyed on
hindcast_start under an FY1 convention, so harvest_year = hindcast_start + 1 is
applied on read and the ambiguous column is dropped immediately.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Reused rather than re-derived: these resolve the same trees compare_dcpp_era5
# reads, and a second copy of the path algebra is a second thing to get wrong.
from compare_dcpp_era5 import (
    DEFAULT_SHARE,
    MODEL,
    _resolve_share,
    dcpp_dir,
    discover_members,
    era5_dir,
)
from era5_features import DEFAULT_SLICE_POLICY, SLICE_POLICIES, lead_tag

# cy-bench spells the crop "maize"; the yield tables and --crop use the US name.
# Stage 3 tags its filenames with whatever era5_features.py --crop was given.
CY_CROP_NAMES = {"corn": "maize"}

# Written by aggregate_season()/aggregate_blocks(); everything else is a feature.
META_COLS = (
    "adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start",
    "forecast_year", "target_year", "season_start", "season_end", "n_blocks",
)

# Prior-season phenology from cy-bench. Mirrors data_loader.PHENOLOGY_CHANNELS.
PHENOLOGY = {
    "ndvi":          {"stem": "ndvi",         "cols": ["ndvi"]},
    "fpar":          {"stem": "fpar",         "cols": ["fpar"]},
    "soil_moisture": {"stem": "soilmoisture", "cols": ["ssm", "rsm"]},
}
COVARIATES = list(PHENOLOGY) + ["yield_history"]


# ============================== where things live ==============================

@dataclass
class Roots:
    """The three independent trees a run reads.

    They are separate because they are built by separate drivers: ERA5 by
    run_cybench_era5.py --out, DCPP by run-dcpp.sh, and the yield/phenology
    tables by the R scripts into the AOI processed-data directories.
    """

    share: Path
    era5: Path
    dcpp: Path
    targets: Path

    @classmethod
    def resolve(cls, share=DEFAULT_SHARE, era5=None, dcpp=None, targets=None):
        s = _resolve_share(str(share))
        return cls(share=s,
                   era5=Path(era5) if era5 else s / "era5",
                   dcpp=Path(dcpp) if dcpp else s / "dcpp",
                   targets=Path(targets) if targets else s)

    def aoi_targets(self, aoi: str) -> Path:
        """The AOI's processed-data directory.

        ET's was created as ET_preprocessed_data rather than ET_processed_data
        (R/crop-data-gen-cybench.R). Accepted here rather than renamed, because
        renaming a directory on the share would orphan files other scripts
        already point at.
        """
        for name in (f"{aoi}_processed_data", f"{aoi}_preprocessed_data"):
            p = self.targets / name
            if p.is_dir():
                return p
        return self.targets / f"{aoi}_processed_data"


def parse_source(source: str) -> tuple[str, str | None]:
    """'era5' | 'dcpp:r1i4p1f1' | 'dcpp:ens' -> (kind, member).

    'dcpp' alone is rejected: with several members on disk there is no sensible
    default, and silently taking the first would make two runs with identical
    command lines disagree.
    """
    s = source.strip().lower()
    if s == "era5":
        return "era5", None
    if s.startswith("dcpp"):
        _, _, mem = s.partition(":")
        if not mem:
            raise ValueError(
                "name the DCPP member explicitly: --source dcpp:r1i4p1f1 for one "
                "realisation, or --source dcpp:ens for the ensemble mean.")
        return "dcpp", mem
    raise ValueError(f"unknown source {source!r}; expected era5 or dcpp:<member>")


def features_dir(roots: Roots, aoi: str, kind: str, member: str) -> Path:
    return era5_dir(roots.era5, aoi) if kind == "era5" \
        else dcpp_dir(roots.dcpp, aoi, member)


def members_for(roots: Roots, aoi: str, member: str) -> list[str]:
    """Expand 'ens' to every member with a stage-1 tree for this AOI."""
    if member != "ens":
        return [member]
    found = discover_members(roots.dcpp, aoi)
    if not found:
        raise FileNotFoundError(
            f"no {MODEL} members found under {roots.dcpp}/{aoi}_processed_data")
    return found


def season_files(fdir: Path, aoi: str, crop: str, forecast_year: int,
                 reduce: str = "season",
                 slice_policy: str = DEFAULT_SLICE_POLICY
                 ) -> tuple[list[Path], str | None]:
    """Stage-3 parquets for this AOI/crop/lead/policy, and the crop tag matched.

    The lead token is part of the filename, so an FY1 store cannot be picked up
    by an FY2 run, and a newest-run store (fy1n_) cannot be picked up by a
    fixed-init run at the same lead. 'fixed' is tried last, for a store built
    with era5_features.py --season SOS:EOS instead of a crop calendar.
    """
    fy = lead_tag(forecast_year, slice_policy)
    kind = "blocks" if reduce == "blocks" else "season"
    tags = [crop] + ([CY_CROP_NAMES[crop]] if crop in CY_CROP_NAMES else []) + ["fixed"]
    for tag in tags:
        hits = sorted((fdir / "features_admin2").glob(
            f"features_{aoi}_{tag}_{fy}{kind}_*.parquet"))
        if hits:
            return hits, tag
    return [], None


# ============================== reading the stores ==============================

def _read_store(fdir: Path, aoi: str, crop: str, forecast_year: int,
                reduce: str, years=None, label: str = "",
                slice_policy: str = DEFAULT_SLICE_POLICY) -> pd.DataFrame:
    """One source's stage-3 table, concatenated over years."""
    paths, tag = season_files(fdir, aoi, crop, forecast_year, reduce, slice_policy)
    if not paths:
        fy = lead_tag(forecast_year, slice_policy)
        kind = "blocks" if reduce == "blocks" else "season"
        listing = sorted(p.name for p in (fdir / "features_admin2").glob("*.parquet")) \
            if (fdir / "features_admin2").is_dir() else []
        raise FileNotFoundError(
            f"no stage-3 {kind} features for {aoi}/{crop} at forecast year "
            f"{forecast_year}{f' [{label}]' if label else ''}.\n"
            f"  looked for : {fdir}/features_admin2/"
            f"features_{aoi}_<crop>_{fy}{kind}_*.parquet\n"
            f"  found there: {', '.join(listing[:8]) or '(nothing)'}"
            + (f" ... {len(listing)} file(s)" if len(listing) > 8 else "") +
            f"\n  build it with: python python/era5_features.py --aoi={aoi} "
            f"--stage=aggregate --reduce={kind} "
            f"--crop={CY_CROP_NAMES.get(crop, crop)} "
            f"--forecast-year={forecast_year}")

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if "harvest_year" not in df.columns:
        raise ValueError(f"{paths[0].name} has no harvest_year column")
    df["harvest_year"] = df["harvest_year"].astype(int)

    # Stage 3 stamps the lead it was built at. Mixing an FY0 reanalysis with an
    # FY1 hindcast would silently merge two different experiments.
    if "forecast_year" in df.columns:
        stamped = sorted({int(v) for v in df["forecast_year"].dropna().unique()})
        if stamped and stamped != [forecast_year]:
            raise ValueError(
                f"{fdir.name}: features are stamped forecast year {stamped} but "
                f"the run asked for {forecast_year}")
    if years is not None:
        df = df[df["harvest_year"].isin(set(years))]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def load_climate(roots: Roots, aoi: str, source: str, crop: str,
                 forecast_year: int, reduce: str = "season",
                 years=None, verbose: bool = True,
                 slice_policy: str = DEFAULT_SLICE_POLICY
                 ) -> tuple[pd.DataFrame, list[str], dict]:
    """Climate features for one AOI from one source, keyed (adm2_id, harvest_year).

    For 'dcpp:ens' every member is read and the feature columns are averaged on
    the shared key. Averaging is restricted to units/years ALL members have, so
    the ensemble mean is never a mean over a varying membership -- that would
    put a step in the series wherever one member's year was missing and show up
    downstream as spurious interannual signal.
    """
    kind, member = parse_source(source)
    if kind == "era5":
        fdir = features_dir(roots, aoi, kind, "")
        # The ERA5 side is observed (fy=0), so no policy applies to it.
        df = _read_store(fdir, aoi, crop, forecast_year, reduce, years, "ERA5")
        feats = feature_columns(df)
        info = {"source": "era5", "members": [], "n_members": 0, "dir": str(fdir)}
        if verbose:
            print(f"  [{aoi}] era5: {len(df):,} rows, {len(feats)} feature(s), "
                  f"harvest years {df.harvest_year.min()}-{df.harvest_year.max()}")
        return df, feats, info

    mems = members_for(roots, aoi, member)
    frames = []
    for mem in mems:
        fdir = features_dir(roots, aoi, kind, mem)
        d = _read_store(fdir, aoi, crop, forecast_year, reduce, years, mem,
                        slice_policy)
        frames.append(d.assign(_member=mem))
    both = pd.concat(frames, ignore_index=True)
    feats = [c for c in feature_columns(both) if c != "_member"]

    if len(mems) == 1:
        out = both.drop(columns="_member")
    else:
        key = ["adm2_id", "harvest_year"]
        n = both.groupby(key)["_member"].nunique()
        complete = set(n[n == len(mems)].index)
        dropped = len(n) - len(complete)
        both = both[both.set_index(key).index.isin(complete)]
        meta = [c for c in META_COLS if c in both.columns]
        out = (both.groupby(key, as_index=False)
               .agg({**{c: "mean" for c in feats},
                     **{c: "first" for c in meta if c not in key}}))
        if dropped and verbose:
            print(f"  [{aoi}] dropped {dropped:,} unit-year(s) not present in all "
                  f"{len(mems)} member(s) before averaging")

    info = {"source": f"dcpp:{member}", "members": mems, "n_members": len(mems),
            "slice_policy": slice_policy,
            "dir": str(features_dir(roots, aoi, kind, mems[0]))}
    if verbose:
        print(f"  [{aoi}] dcpp {lead_tag(forecast_year, slice_policy).rstrip('_')} "
              f"({len(mems)} member(s)"
              f"{': mean' if len(mems) > 1 else f': {mems[0]}'}): {len(out):,} rows, "
              f"{len(feats)} feature(s), harvest years "
              f"{out.harvest_year.min()}-{out.harvest_year.max()}")
    return out, feats, info


# ============================== targets and covariates ==============================

def load_yield(roots: Roots, aoi: str, crop: str, shift: int = 0) -> pd.DataFrame:
    """The cy-bench yield table, re-keyed onto harvest_year.

    The parquet is keyed on hindcast_start under an FY1 convention -- the yield
    is for calendar year hindcast_start + 1 -- so that column is converted and
    dropped here. Leaving both in the frame is how a lead-time off-by-one gets
    made downstream.

    `shift` is added to harvest_year to correct a source whose labels are off.
    Brazil needs -1: python/check_harvest_offset.py puts documented drought
    losses at label H+1, and ERA5 season features of year Y-1 correlate with
    the yield labelled Y far more strongly than those of Y, in most states.
    Phenology needs no matching shift; its seasons come from observation dates.
    """
    path = roots.aoi_targets(aoi) / f"crop_{crop}_{aoi}.parquet"
    if not path.exists():
        have = sorted(p.name for p in path.parent.glob("crop_*.parquet")) \
            if path.parent.is_dir() else []
        raise FileNotFoundError(
            f"yield table not found: {path}\n"
            f"  present here  : {', '.join(have) or '(nothing)'}\n"
            f"  build it with : Rscript R/crop-data-gen-cybench.R  (country = {aoi})")
    df = pd.read_parquet(path)
    df["harvest_year"] = df["hindcast_start"].astype(int) + 1 + int(shift)
    df = df.drop(columns=[c for c in ("hindcast_start", "forecast_year") if c in df])
    if shift:
        print(f"  [{aoi}] yield harvest_year shifted by {int(shift):+d}")

    # Structural zeros are missing data, not a failed harvest, and they drag a
    # unit's climatology down far enough to distort its anomalies.
    bad = (~np.isfinite(df["yield"])) | (df["yield"] <= 0)
    if bad.any():
        print(f"  [{aoi}] dropped {int(bad.sum()):,} row(s) with non-positive yield")
    keep = ["adm2_id", "harvest_year", "yield"]
    return df.loc[~bad, keep].drop_duplicates(["adm2_id", "harvest_year"])


def phenology_features(roots: Roots, aoi: str, var: str, lag: int,
                       suffix: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Prior-season summary stats for one cy-bench phenology variable.

    `lag` is in seasons before the harvest year being predicted, and must be
    >= 1: the current season's NDVI is not observable when the forecast is
    issued, so using it would be label leakage dressed as a covariate.
    """
    if var not in PHENOLOGY:
        raise ValueError(f"unknown phenology variable {var!r}; have {list(PHENOLOGY)}")
    if lag < 1:
        raise ValueError(f"phenology lag must be >= 1 season, got {lag}")
    cfg = PHENOLOGY[var]
    path = roots.aoi_targets(aoi) / "phenology" / \
        f"phenology_{cfg['stem']}_{suffix or aoi}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"phenology table not found: {path}\n"
            f"  build it with: Rscript R/phenology-sample-gen.R  (country = {aoi})")

    df = pd.read_parquet(path)
    # The season an observation belongs to: Nov/Dec fall in the NEXT year's.
    if "hindcast_start" in df.columns:
        df["season"] = df["hindcast_start"].astype(int) + 1
    else:
        d = pd.to_datetime(df["date"])
        df["season"] = d.dt.year + (d.dt.month >= 11).astype(int)

    cols = cfg["cols"]
    agg = df.groupby(["adm2_id", "season"])[cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{c}_prev{lag}_{s}" for c, s in agg.columns]
    for c in cols:
        agg[f"{c}_prev{lag}_amp"] = agg[f"{c}_prev{lag}_max"] - agg[f"{c}_prev{lag}_min"]
    agg = agg.reset_index()
    # Shift onto the harvest year this row will be joined to.
    agg["harvest_year"] = agg["season"].astype(int) + lag
    names = [c for c in agg.columns if c not in ("adm2_id", "season", "harvest_year")]
    return agg[["adm2_id", "harvest_year"] + names], names


def yield_lag_features(yields: pd.DataFrame, n_lags: int,
                       min_lag: int) -> tuple[pd.DataFrame, list[str]]:
    """A unit's own past yields as lag columns, plus a validity mask each.

    min_lag is the forecast horizon, and it is not cosmetic. A lead-F forecast
    is issued before harvest year H - F + 1 is in the ground, so the most recent
    yield the forecaster can possibly know is H - F. Allowing lag 1 at F = 2
    would hand the model an observation from the future.

    Missing lags are filled with the unit's mean over the lags it does have and
    flagged 0 in the mask, so a tree can learn "this value is padding" instead
    of treating a sentinel as a real low yield.
    """
    out = yields[["adm2_id", "harvest_year"]].copy()
    lags = list(range(min_lag, min_lag + n_lags))
    src = yields.set_index(["adm2_id", "harvest_year"])["yield"]
    names = []
    for k in lags:
        probe = pd.MultiIndex.from_arrays(
            [out["adm2_id"], out["harvest_year"] - k])
        v = src.reindex(probe).to_numpy()
        out[f"yield_lag{k}"] = v
        out[f"yield_lag{k}_ok"] = np.isfinite(v).astype(float)
        names += [f"yield_lag{k}", f"yield_lag{k}_ok"]
    cols = [f"yield_lag{k}" for k in lags]
    rowmean = out[cols].mean(axis=1)
    for c in cols:
        out[c] = out[c].fillna(rowmean)
    # A unit with no usable history at all stays NaN and is dropped by the
    # caller, rather than being imputed from other units.
    return out, names


# ============================== assembly ==============================

def reference_yields(yields: pd.DataFrame, lags) -> tuple[pd.DataFrame, list[str]]:
    """y[u, t-k] for each requested k, as `yref{k}` columns.

    These are TARGET metadata, not features: the direction label is
    sign(y[u,t] - y[u,t-k]), so the reference has to be on the row before a
    label can be formed. It is taken from the full yield table rather than from
    the feature-joined frame, because year t-k often has no climate features
    (outside --years, or skipped for a missing DCPP slice) while its yield is
    perfectly well observed. Joining on the restricted frame would silently
    drop those rows.

    Whether y[u,t-k] may also be used as an INPUT is a separate question, and
    the answer is yes -- it is known when a lead-F forecast is issued as long
    as k >= F. That is what --add yield_history does; it is kept distinct so
    the label and the feature set cannot be confused.
    """
    out = yields[["adm2_id", "harvest_year"]].copy()
    src = yields.set_index(["adm2_id", "harvest_year"])["yield"]
    names = []
    for k in sorted({int(x) for x in lags}):
        probe = pd.MultiIndex.from_arrays([out["adm2_id"], out["harvest_year"] - k])
        out[f"yref{k}"] = src.reindex(probe).to_numpy()
        names.append(f"yref{k}")
    return out, names


def build_aoi(roots: Roots, aoi: str, source: str, crop: str, forecast_year: int,
              reduce: str = "season", years=None, covariates=(),
              phenology_lag: int = 1, n_yield_lags: int = 3,
              phenology_suffix: str | None = None, ref_lags=(),
              verbose: bool = True, slice_policy: str = DEFAULT_SLICE_POLICY,
              yield_shift: int = 0) -> tuple[pd.DataFrame, list[str], dict]:
    """One AOI's design matrix: climate + yield + requested covariates.

    `ref_lags` additionally attaches yref{k} columns for the direction target.
    They are returned on the frame but never in the feature list.
    """
    clim, feats, info = load_climate(roots, aoi, source, crop, forecast_year,
                                     reduce, years, verbose, slice_policy)
    y = load_yield(roots, aoi, crop, yield_shift)

    keep = ["adm2_id", "harvest_year"] + \
        [c for c in ("adm2", "adm1") if c in clim.columns] + feats
    df = clim[keep].merge(y, on=["adm2_id", "harvest_year"], how="inner")
    if df.empty:
        cu, yu = set(clim.adm2_id.unique()), set(y.adm2_id.unique())
        raise ValueError(
            f"[{aoi}] climate and yield tables share no (adm2_id, harvest_year).\n"
            f"  climate: {len(cu)} unit(s), years "
            f"{clim.harvest_year.min()}-{clim.harvest_year.max()}\n"
            f"  yield  : {len(yu)} unit(s), years {y.harvest_year.min()}-{y.harvest_year.max()}\n"
            f"  ids in common: {len(cu & yu)}\n"
            f"  zero ids in common means the adm2_id spellings differ between the "
            f"polygons and the cy-bench yield table.")

    if ref_lags:
        refs, refnames = reference_yields(y, ref_lags)
        df = df.merge(refs, on=["adm2_id", "harvest_year"], how="left")
        if verbose:
            for nm in refnames:
                have = int(df[nm].notna().sum())
                print(f"  [{aoi}] {nm}: {have:,}/{len(df):,} rows have a "
                      f"reference year")

    cols = list(feats)
    for cov in covariates:
        if cov == "yield_history":
            # min_lag from the lead time: see yield_lag_features.
            lagdf, names = yield_lag_features(
                y, n_yield_lags, min_lag=max(1, forecast_year))
            df = df.merge(lagdf, on=["adm2_id", "harvest_year"], how="left")
        else:
            lagdf, names = phenology_features(
                roots, aoi, cov, phenology_lag, phenology_suffix)
            df = df.merge(lagdf, on=["adm2_id", "harvest_year"], how="left")
        cols += names
        before = len(df)
        df = df.dropna(subset=names)
        if verbose:
            print(f"  [{aoi}] + {cov}: {len(names)} column(s), "
                  f"{len(df):,} rows (dropped {before - len(df):,} without coverage)")

    before = len(df)
    df = df.dropna(subset=cols + ["yield"])
    if before != len(df) and verbose:
        print(f"  [{aoi}] dropped {before - len(df):,} row(s) with a missing "
              f"feature or target")
    df = df.assign(aoi=aoi).sort_values(["adm2_id", "harvest_year"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"[{aoi}] no rows survived assembly")
    if verbose:
        print(f"  [{aoi}] -> {len(df):,} samples, {df.adm2_id.nunique()} units, "
              f"{df.harvest_year.nunique()} years, {len(cols)} features")
    info = {**info, "aoi": aoi, "n_rows": len(df),
            "n_units": int(df.adm2_id.nunique()),
            "years": [int(df.harvest_year.min()), int(df.harvest_year.max())]}
    return df, cols, info


def build(roots: Roots, aois, source: str, crop: str, forecast_year: int,
          reduce: str = "season", years=None, covariates=(),
          phenology_lag: int = 1, n_yield_lags: int = 3,
          phenology_suffix: str | None = None, ref_lags=(),
          verbose: bool = True, slice_policy: str = DEFAULT_SLICE_POLICY,
          yield_shift: dict | None = None
          ) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Several AOIs stacked into one frame with an `aoi` column.

    The feature set is intersected across AOIs, not unioned. A union would
    introduce all-NaN columns for whichever AOI lacks them, and dropping those
    rows would quietly delete that AOI from the pooled fit.
    """
    frames, colsets, infos = [], [], []
    for aoi in aois:
        df, cols, info = build_aoi(roots, aoi, source, crop, forecast_year, reduce,
                                   years, covariates, phenology_lag, n_yield_lags,
                                   phenology_suffix, ref_lags, verbose, slice_policy,
                                   (yield_shift or {}).get(aoi, 0))
        frames.append(df)
        colsets.append(cols)
        infos.append(info)

    shared = [c for c in colsets[0] if all(c in s for s in colsets)]
    if not shared:
        raise ValueError("the AOIs share no feature columns; build their stage-3 "
                         "stores with the same --features and --reduce")
    for aoi, cols in zip(aois, colsets):
        extra = [c for c in cols if c not in shared]
        if extra and verbose:
            print(f"  note: {aoi} has {len(extra)} feature(s) the others lack, "
                  f"dropped: {', '.join(extra[:6])}"
                  f"{' ...' if len(extra) > 6 else ''}")

    lead = ["aoi", "adm2_id", "harvest_year", "yield"]
    opt = [c for c in ("adm2", "adm1") if all(c in f.columns for f in frames)]
    opt += [f"yref{int(k)}" for k in sorted({int(x) for x in ref_lags})]
    out = pd.concat([f[lead + opt + shared] for f in frames], ignore_index=True)
    # adm2_id is unique across cy-bench countries, but not guaranteed to be, and
    # a collision would merge two units' climatologies into one.
    out["unit"] = out["aoi"] + ":" + out["adm2_id"].astype(str)
    if verbose:
        print(f"  pooled: {len(out):,} samples, {out.unit.nunique()} units, "
              f"{len(aois)} AOI(s), {len(shared)} features")
    return out, shared, infos
