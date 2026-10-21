#!/usr/bin/env python3
"""Compare EC-Earth3 DCPP hindcast features against ERA5 features, per AOI.

Builds cropland-weighted admin-2 features for both sources through
`era5_features.py` (so there is exactly one feature definition in the repo),
joins them on (adm2_id, harvest_year), and reports how well each DCPP ensemble
member -- and the ensemble mean -- reproduces the ERA5 value for the same
region and season.

    stage 1 (already done)   daily per-pixel cubes + cropland weight tables
    stage 2+3 (this script)  era5_features.py -> features_admin2/*.parquet
    comparison (this script) join on (adm2_id, harvest_year) -> csv + json

The cropland GeoTIFF is NOT read here. It is consumed once at stage 1, where
`R/pixel-store.R` turns it into `grid/weights_{AOI}.parquet`
(`combined_weight = cropland_frac * coverage_frac`). Stages 2 and 3 read that
weight table, so the weighting is already baked into every value below. Each
source has its own table because the grids differ (ERA5 0.25 deg vs EC-Earth3
~0.7031 deg), which is why the two trees are siblings rather than shared.

Four correlations are reported per (AOI, feature, member):

  r_pooled    all (unit, year) pairs at once. INFLATED: units differ in mean
              climate, so a model that gets only the spatial pattern right
              scores well without any interannual skill. Reported for
              completeness, not as the skill number.
  r_anom      same pairs after removing each unit's own mean from both sides.
              This is the interannual signal -- the honest headline.
  r_time      per unit across years, then averaged over units. Same quantity as
              r_anom but unit-by-unit, so its spread shows whether skill is
              uniform or carried by a few units.
  r_space     per year across units, then averaged over years. Spatial pattern
              skill within a season.

Usage:
    python python/compare_dcpp_era5.py --aois ZM --harvest-years 2002-2022
    python python/compare_dcpp_era5.py --aois ZM,ET,BF --go
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from era5_features import (  # noqa: E402
    DEFAULT_FEATURES,
    parse_years,
    season_source_years,
)

FEATURES_SCRIPT = HERE / "era5_features.py"

# Defaults match the CRL machine's mount
DEFAULT_SHARE = "/mnt/CRL_storage/personal_folders/kdoerksen"
MODEL = "EC-Earth3" # default, will run with other models in the future as well

# Columns stage 3 writes as metadata; everything else in the parquet is a value.
# `n_blocks` is written by aggregate_blocks(); without it here it would be
# picked up as if it were a feature value.
META = {
    "adm2_id", "adm2", "adm1", "harvest_year", "hindcast_start",
    "forecast_year", "target_year", "season_start", "season_end", "n_blocks",
}

# aggregate_blocks() names its columns "{feature}_m01" .. "_mNN".
BLOCK_COL = re.compile(r"^(?P<base>.+)_m\d{2}$")

# Setting minimums of points to pool over
MIN_N_POOLED = 10
MIN_N_PER_GROUP = 4

# ============================== filepaths =============================

def _resolve_share(path: str) -> Path:
    """Accept either capitalisation of the CRL_storage mount."""
    p = Path(path)
    if p.exists():
        return p
    for cand in (Path(str(p).replace("/CRL_storage/", "/crl_storage/")),
                 Path(str(p).replace("/crl_storage/", "/CRL_storage/"))):
        if cand.exists():
            print(f"note: {p} not found, using {cand}")
            return cand
    raise SystemExit(
        f"share not found: {p}\n"
        f"Pass --share (or --era5-root/--dcpp-root/--root individually).\n"
        f"Mounted filesystems:\n  "
        + "\n  ".join(sorted(str(x) for x in Path("/mnt").glob("*")) or ["(nothing under /mnt)"])
    )


def era5_dir(era5_root: Path, aoi: str) -> Path:
    """run_cybench_era5.py writes each AOI to {out}/{AOI}."""
    return era5_root / aoi


def dcpp_dir(dcpp_root: Path, aoi: str, member: str) -> Path:
    """run-dcpp.sh writes {out}/{AOI}_processed_data/{MODEL}_{member}."""
    return dcpp_root / f"{aoi}_processed_data" / f"{MODEL}_{member}"


def discover_members(dcpp_root: Path, aoi: str) -> list[str]:
    """Members that actually have a stage-1 tree for this AOI."""
    base = dcpp_root / f"{aoi}_processed_data"
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.glob(f"{MODEL}_*")):
        if (d / "pixel_daily").is_dir() or (d / "grid").is_dir():
            out.append(d.name[len(MODEL) + 1:])
    return out


def season_parquet(out_dir: Path, aoi: str, crop: str, forecast_year: int,
                   harvest_year: int, reduce: str, fixed_season: bool) -> Path:
    """Mirror of the filename stage_aggregate() builds in era5_features.py.

    Kept as one function so a change there is a single edit here, and verified
    by _load_side() which lists the directory when a file is missing.
    """
    fy_tag = f"fy{forecast_year}_" if forecast_year else ""
    tag = "fixed" if fixed_season else crop
    kind = "blocks" if reduce == "blocks" else "season"
    name = f"features_{aoi}_{tag}_{fy_tag}{kind}_{harvest_year}.parquet"
    return out_dir / "features_admin2" / name


# ================================ build variables =================================


def build_cmd(a, aoi: str, source: str, out_dir: Path, forecast_year: int,
              source_years: list[int]) -> list[str]:
    cmd = [
        a.python, str(FEATURES_SCRIPT),
        f"--aoi={aoi}",
        f"--source={source}",
        f"--forecast-year={forecast_year}",
        f"--years={source_years[0]}-{source_years[-1]}",
        f"--stage={a.stage}",
        f"--reduce={a.reduce}",
        f"--weight={a.weight}",
        f"--features={a.features}",
        f"--crop={a.crop}",
        f"--out={out_dir}",
        f"--root={a.root}",
    ]
    # cy-bench country code: defaults to the AOI tag, except CONUS which is US.
    cmd.append(f"--cy-country={a.cy_country or ('US' if aoi == 'CONUS' else aoi)}")
    if a.season:
        cmd.append(f"--season={a.season}")
    if a.reduce == "blocks":
        cmd.append(f"--block-days={a.block_days}")
        if a.n_blocks:
            cmd.append(f"--n-blocks={a.n_blocks}")
    for p in a.param or []:
        cmd.append(f"--param={p}")
    if a.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_build(cmd: list[str], log: Path) -> bool:
    """Run one era5_features.py invocation"""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", buffering=1) as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"    FAILED (rc={rc}) -- see {log}")
    return rc == 0


# ================================ load & join =================================


def value_columns(df: pd.DataFrame) -> list[str]:
    """Every column that is not metadata."""
    return [c for c in df.columns if c not in META]


def match_columns(df: pd.DataFrame, feats: list[str], reduce: str) -> list[str]:
    """Value columns in `df` that belong to the requested features.

    --reduce season gives one column per feature ("gdd"); --reduce blocks gives
    one per feature per season block ("gdd_m01" ... "gdd_m06"). Matching the
    bare feature name against a blocks file finds nothing, which is why this is
    resolved from the data rather than from the --features string.
    """
    vals = value_columns(df)
    if reduce != "blocks":
        return [c for c in vals if c in feats]
    out = []
    for c in vals:
        m = BLOCK_COL.match(c)
        if m and m.group("base") in feats:
            out.append(c)
    return out


def _load_side(out_dir: Path, aoi: str, crop: str, fy: int, years: list[int],
               reduce: str, fixed: bool, label: str) -> pd.DataFrame:
    frames, missing = [], []
    for hy in years:
        p = season_parquet(out_dir, aoi, crop, fy, hy, reduce, fixed)
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append(hy)
    if missing:
        have = sorted(x.name for x in (out_dir / "features_admin2").glob("*.parquet")) \
            if (out_dir / "features_admin2").is_dir() else []
        print(f"    {label}: missing {len(missing)} year(s): "
              f"{missing[0]}..{missing[-1]}" if len(missing) > 1
              else f"    {label}: missing {missing}")
        if not frames:
            print(f"      nothing loaded from {out_dir / 'features_admin2'}")
            if have:
                print(f"      directory holds: {', '.join(have[:6])}"
                      + (f" ... (+{len(have) - 6})" if len(have) > 6 else ""))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pair_long(era5: pd.DataFrame, dcpp: pd.DataFrame, feats: list[str],
              member: str) -> pd.DataFrame:
    """Inner-join one member to ERA5 to (unit, year, feature) rows.
    """
    keys = ["adm2_id", "harvest_year"]
    e = era5[keys + feats].rename(columns={f: f"{f}__era5" for f in feats})
    d = dcpp[keys + feats].rename(columns={f: f"{f}__dcpp" for f in feats})
    j = e.merge(d, on=keys, how="inner")
    rows = []
    for f in feats:
        sub = j[keys + [f"{f}__era5", f"{f}__dcpp"]].copy()
        sub.columns = keys + ["era5", "dcpp"]
        sub["feature"] = f
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["member"] = member
    return out.dropna(subset=["era5", "dcpp"])


# ================================ statistics ==================================


def _r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, NaN when either side is constant"""
    if len(x) < 2:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _grouped_r(df: pd.DataFrame, by: str) -> tuple[float, float, int]:
    """Mean and sd of within-group r, plus the number of usable groups."""
    rs = []
    for _, g in df.groupby(by):
        if len(g) >= MIN_N_PER_GROUP:
            r = _r(g["era5"].to_numpy(), g["dcpp"].to_numpy())
            if np.isfinite(r):
                rs.append(r)
    if not rs:
        return np.nan, np.nan, 0
    return float(np.mean(rs)), float(np.std(rs)), len(rs)


def metrics(df: pd.DataFrame) -> dict:
    """All four correlations plus the error terms needed to interpret them."""
    e = df["era5"].to_numpy(dtype=float)
    d = df["dcpp"].to_numpy(dtype=float)
    out = {
        "n_pairs": int(len(df)),
        "n_units": int(df["adm2_id"].nunique()),
        "n_years": int(df["harvest_year"].nunique()),
        "era5_mean": float(np.mean(e)) if len(e) else np.nan,
        "dcpp_mean": float(np.mean(d)) if len(d) else np.nan,
        "era5_sd": float(np.std(e)) if len(e) else np.nan,
        "dcpp_sd": float(np.std(d)) if len(d) else np.nan,
    }
    if len(df) < MIN_N_POOLED:
        out.update(r_pooled=np.nan, r_anom=np.nan, bias=np.nan, mae=np.nan,
                   rmse=np.nan, sd_ratio_anom=np.nan, sd_ratio_raw=np.nan,
                   r_time=np.nan, r_time_sd=np.nan, n_units_scored=0,
                   r_space=np.nan, r_space_sd=np.nan, n_years_scored=0)
        return out

    out["r_pooled"] = _r(e, d)
    # Anomalies: remove each unit's own mean from BOTH sides. What is left is
    # the interannual variation, which is what a decadal hindcast is judged on.
    g = df.groupby("adm2_id")
    ea = (df["era5"] - g["era5"].transform("mean")).to_numpy(dtype=float)
    da = (df["dcpp"] - g["dcpp"].transform("mean")).to_numpy(dtype=float)
    out["r_anom"] = _r(ea, da)
    out["bias"] = float(np.mean(d - e))
    out["mae"] = float(np.mean(np.abs(d - e)))
    out["rmse"] = float(np.sqrt(np.mean((d - e) ** 2)))
    # Dispersion on ANOMALIES. The raw ratio is dominated by between-unit
    # climatology -- on a panel where units differ in mean climate it sits at
    # ~1.0 however badly the hindcast's interannual amplitude is scaled, which
    # is the same reason r_pooled is uninformative. sd_ratio_anom is the one
    # that answers "is the hindcast underdispersed".
    out["sd_ratio_anom"] = (float(np.std(da) / np.std(ea))
                            if np.std(ea) > 0 else np.nan)
    out["sd_ratio_raw"] = (float(np.std(d) / np.std(e)) if np.std(e) > 0 else np.nan)
    rt, rts, nu = _grouped_r(df, "adm2_id")
    out.update(r_time=rt, r_time_sd=rts, n_units_scored=nu)
    rs, rss, ny = _grouped_r(df, "harvest_year")
    out.update(r_space=rs, r_space_sd=rss, n_years_scored=ny)
    return out


def ensemble_mean(pairs: pd.DataFrame) -> pd.DataFrame:
    """Average the members, then compare. ERA5 is identical across members, so
    it is taken as the first value rather than averaged."""
    g = pairs.groupby(["adm2_id", "harvest_year", "feature"], as_index=False)
    out = g.agg(era5=("era5", "first"),
                dcpp=("dcpp", "mean"),
                n_members=("member", "nunique"))
    out["member"] = "ENSMEAN"
    return out


# =================================== main =====================================


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Correlate DCPP hindcast features against ERA5, per AOI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--aois", required=True, help="comma-separated AOI tags, e.g. ZM,ET")
    p.add_argument("--members", help="comma-separated DCPP members; default: discover")
    p.add_argument("--forecast-year", type=int, default=1)
    p.add_argument("--harvest-years", default="2002-2022",
                   help="HARVEST years to compare (source years are derived)")
    p.add_argument("--share", default=DEFAULT_SHARE,
                   help="base of the lab share; --era5-root/--dcpp-root/--root override")
    p.add_argument("--era5-root", help="default {share}/era5")
    p.add_argument("--dcpp-root", help="default {share}/dcpp")
    p.add_argument("--root", help="DATA root holding cy-bench (default {share})")
    p.add_argument("--out-dir", help="where to write csv/json (default {share}/comparisons)")
    p.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    # No "daily": aggregate_daily() emits one row per unit per DAY, keyed on
    # date with no harvest_year column, so it cannot be joined per region-year.
    p.add_argument("--reduce", choices=["season", "blocks"], default="season")
    p.add_argument("--weight", default="cropland", choices=["cropland", "area", "none"])
    p.add_argument("--crop", default="maize")
    p.add_argument("--cy-country", help="cy-bench country code (default: the AOI tag)")
    p.add_argument("--season", metavar="SOS:EOS", help="fixed window, bypass the calendar")
    p.add_argument("--block-days", type=int, default=30)
    p.add_argument("--n-blocks", type=int, default=None)
    p.add_argument("--param", action="append", metavar="FEAT.arg=value")
    p.add_argument("--stage", choices=["features", "aggregate", "both"], default="both")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-build", action="store_true",
                   help="only read existing parquets and correlate")
    p.add_argument("--overwrite", action="store_true", help="rebuild stage 2/3 outputs")
    p.add_argument("--tag", default=None, help="suffix for the output filenames")
    p.add_argument("--go", action="store_true", help="without this, print the plan and exit")
    a = p.parse_args(argv)

    share = _resolve_share(a.share)
    era5_root = Path(a.era5_root) if a.era5_root else share / "era5"
    dcpp_root = Path(a.dcpp_root) if a.dcpp_root else share / "dcpp"
    a.root = str(a.root or share)
    out_dir = Path(a.out_dir) if a.out_dir else share / "comparisons"

    aois = [x.strip() for x in a.aois.split(",") if x.strip()]
    feats = [x.strip() for x in a.features.split(",") if x.strip()]
    fixed = bool(a.season)

    # era5_features.parse_years handles "2002-2022", "2002,2005", and mixtures.
    # int("2002,2005") used to raise a bare ValueError here.
    harvest_years = parse_years(a.harvest_years)
    if not harvest_years:
        raise SystemExit(f"--harvest-years parsed to nothing: {a.harvest_years!r}")
    if a.forecast_year < 1:
        raise SystemExit(
            f"--forecast-year must be >= 1 for a hindcast (got {a.forecast_year}).\n"
            "At 0 the DCPP outputs would take the observed naming convention and "
            "collide with the ERA5 tree's filenames.")

    # Source years each side needs. era5_features.py takes SOURCE years and
    # derives the harvest years itself, so these only have to be a superset.
    era5_src = sorted({y for h in harvest_years for y in season_source_years(h, 0)})
    dcpp_src = sorted({y for h in harvest_years
                       for y in season_source_years(h, a.forecast_year)})

    print("=" * 78)
    print(f"DCPP (fy{a.forecast_year}) vs ERA5 -- cropland-weighted admin-2 features")
    print("=" * 78)
    print(f"  ERA5 tree   : {era5_root}/{{AOI}}")
    print(f"  DCPP tree   : {dcpp_root}/{{AOI}}_processed_data/{MODEL}_{{member}}")
    print(f"  data root   : {a.root}  (cy-bench crop calendar)")
    print(f"  out         : {out_dir}")
    print(f"  harvest yrs : {harvest_years[0]}-{harvest_years[-1]} ({len(harvest_years)})")
    print(f"  ERA5 source : {era5_src[0]}-{era5_src[-1]}")
    print(f"  DCPP source : {dcpp_src[0]}-{dcpp_src[-1]}")
    print(f"  features    : {','.join(feats)}   reduce={a.reduce} weight={a.weight}")
    print()

    plan = []
    for aoi in aois:
        mems = ([x.strip() for x in a.members.split(",")] if a.members
                else discover_members(dcpp_root, aoi))
        if not mems:
            print(f"  {aoi}: no DCPP members found under "
                  f"{dcpp_root / (aoi + '_processed_data')} -- skipping")
            continue
        e_dir = era5_dir(era5_root, aoi)
        if not e_dir.is_dir():
            print(f"  {aoi}: no ERA5 tree at {e_dir} -- skipping")
            continue
        plan.append((aoi, mems))
        print(f"  {aoi}: {len(mems)} member(s): {', '.join(mems)}")
    if not plan:
        raise SystemExit("nothing to do")

    n_builds = sum(1 + len(m) for _, m in plan)
    print(f"\n  {len(plan)} AOI(s), {n_builds} era5_features.py invocation(s)")
    # Echo the exact commands: this is the thing worth reviewing before a long
    # run, and reconstructing it from the flags above is error-prone.
    aoi0, mems0 = plan[0]
    print("\n  commands (first AOI shown; one per source):")
    print("    " + " ".join(build_cmd(a, aoi0, "ERA5", era5_dir(era5_root, aoi0), 0, era5_src)))
    print("    " + " ".join(build_cmd(a, aoi0, f"{MODEL}_{mems0[0]}",
                                      dcpp_dir(dcpp_root, aoi0, mems0[0]),
                                      a.forecast_year, dcpp_src)))
    if not a.go:
        print("\n  dry run -- add --go to execute")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"fy{a.forecast_year}_{a.reduce}_{harvest_years[0]}-{harvest_years[-1]}"

    all_pairs, rows, failures = [], [], []
    for aoi, mems in plan:
        print(f"\n--- {aoi} ---")
        e_dir = era5_dir(era5_root, aoi)

        if not a.skip_build:
            print("  building ERA5 features ...")
            cmd = build_cmd(a, aoi, "ERA5", e_dir, 0, era5_src)
            if not run_build(cmd, out_dir / "logs" / f"{aoi}_ERA5.log"):
                failures.append((aoi, "ERA5", "build"))
                continue

        era5 = _load_side(e_dir, aoi, a.crop, 0, harvest_years, a.reduce, fixed,
                          f"{aoi}/ERA5")
        if era5.empty:
            failures.append((aoi, "ERA5", "no output"))
            continue
        have = match_columns(era5, feats, a.reduce)
        base = {BLOCK_COL.match(c).group("base") if BLOCK_COL.match(c) else c
                for c in have}
        if base != set(feats):
            print(f"    note: ERA5 lacks {sorted(set(feats) - base)} "
                  f"-- rerun with --overwrite to widen stage 2")
        if not have:
            failures.append((aoi, "ERA5", "no requested features"))
            continue
        print(f"    ERA5: {len(era5)} rows, {era5.adm2_id.nunique()} units, "
              f"{era5.harvest_year.nunique()} years")

        aoi_pairs = []
        for mem in mems:
            source = f"{MODEL}_{mem}"
            d_dir = dcpp_dir(dcpp_root, aoi, mem)
            if not a.skip_build:
                print(f"  building {source} ...")
                cmd = build_cmd(a, aoi, source, d_dir, a.forecast_year, dcpp_src)
                if not run_build(cmd, out_dir / "logs" / f"{aoi}_{source}.log"):
                    failures.append((aoi, source, "build"))
                    continue
            dcpp = _load_side(d_dir, aoi, a.crop, a.forecast_year, harvest_years,
                              a.reduce, fixed, f"{aoi}/{source}")
            if dcpp.empty:
                failures.append((aoi, source, "no output"))
                continue
            usable = [c for c in have if c in value_columns(dcpp)]
            if not usable:
                failures.append((aoi, source, "no shared features"))
                continue
            pr = pair_long(era5, dcpp, usable, mem)
            if pr.empty:
                failures.append((aoi, source, "empty join"))
                continue
            dropped = era5.adm2_id.nunique() - pr.adm2_id.nunique()
            print(f"    {source}: {len(pr)} paired values, {pr.adm2_id.nunique()} units"
                  + (f" ({dropped} ERA5 unit(s) absent from DCPP)" if dropped > 0 else ""))
            aoi_pairs.append(pr)

        if not aoi_pairs:
            continue
        pairs = pd.concat(aoi_pairs, ignore_index=True)
        pairs.insert(0, "aoi", aoi)
        ens = ensemble_mean(pairs.drop(columns="aoi"))
        ens.insert(0, "aoi", aoi)
        all_pairs.append(pairs)
        all_pairs.append(ens)

        for member, mdf in pd.concat([pairs, ens], ignore_index=True).groupby("member"):
            for feat, fdf in mdf.groupby("feature"):
                rec = {"aoi": aoi, "member": member, "feature": feat,
                       "forecast_year": a.forecast_year, "reduce": a.reduce,
                       "weight": a.weight, "crop": a.crop}
                if member == "ENSMEAN":
                    rec["n_members"] = int(fdf["n_members"].max())
                rec.update(metrics(fdf))
                rows.append(rec)

    if not rows:
        raise SystemExit("no comparable data produced -- see the logs in "
                         f"{out_dir / 'logs'}")

    summary = pd.DataFrame(rows)
    lead = ["aoi", "member", "feature", "forecast_year", "r_anom", "r_time",
            "r_space", "r_pooled", "bias", "rmse", "sd_ratio_anom",
            "n_pairs", "n_units", "n_years"]
    summary = summary[[c for c in lead if c in summary.columns]
                      + [c for c in summary.columns if c not in lead]]
    summary = summary.sort_values(["aoi", "feature", "member"])

    csv_p = out_dir / f"dcpp_era5_corr_{tag}.csv"
    json_p = out_dir / f"dcpp_era5_corr_{tag}.json"
    pairs_p = out_dir / f"dcpp_era5_pairs_{tag}.csv.gz"
    summary.to_csv(csv_p, index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(pairs_p, index=False)
    with open(json_p, "w") as fh:
        json.dump({
            "run": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "argv": sys.argv,
                "model": MODEL,
                "forecast_year": a.forecast_year,
                "harvest_years": [harvest_years[0], harvest_years[-1]],
                "era5_source_years": [era5_src[0], era5_src[-1]],
                "dcpp_source_years": [dcpp_src[0], dcpp_src[-1]],
                "features_requested": feats,
                "reduce": a.reduce, "weight": a.weight, "crop": a.crop,
                "era5_root": str(era5_root), "dcpp_root": str(dcpp_root),
                "data_root": a.root,
                "aois": {aoi: mems for aoi, mems in plan},
            },
            "metric_notes": {
                "r_anom": "unit means removed from both sides; interannual skill (headline)",
                "r_time": "mean of per-unit temporal r; r_time_sd is its spread across units",
                "r_space": "mean of per-year spatial r across units",
                "r_pooled": "raw pooled r; inflated by between-unit climatology",
                "sd_ratio_anom": "sd(DCPP anom)/sd(ERA5 anom); <1 = underdispersed interannual amplitude",
                "sd_ratio_raw": "same on raw values; climatology-dominated, kept only for reference",
                "bias": "mean(DCPP - ERA5) in the feature's own units",
            },
            "failures": [{"aoi": a_, "source": s, "why": w} for a_, s, w in failures],
            "results": json.loads(summary.to_json(orient="records")),
        }, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"wrote {csv_p}")
    print(f"      {json_p}")
    print(f"      {pairs_p}  (every paired value, for re-analysis)")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for a_, s, w in failures:
            print(f"  {a_:6} {s:24} {w}")
    show = ["aoi", "member", "feature", "r_anom", "r_time", "r_space",
            "r_pooled", "bias", "sd_ratio_anom", "n_pairs"]
    print("\nr_anom is the headline; r_pooled is climatology-inflated by design:")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
