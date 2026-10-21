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


def discover_leads(dcpp_root: Path, aoi: str, members: list[str]) -> list[int]:
    """Lead times that actually have stage-1 output for this AOI.

    Stage 1 writes pixel_{tag}_{AOI}_fy{n}_s{year}.nc, so the leads present are
    readable off the filenames. Checked across every member and unioned: a lead
    processed for only some members is still worth comparing, and the per-member
    loop reports the gaps.

    Falls back to pixel_features/ when a tree has been cleaned of stage-1 cubes
    but keeps stage-2 output, which is a state a --stage=features rerun leaves.
    """
    leads: set[int] = set()
    for mem in members:
        base = dcpp_dir(dcpp_root, aoi, mem)
        for sub, pat in (("pixel_daily", f"pixel_*_{aoi}_fy*_s*.nc"),
                         ("pixel_features", f"features_{aoi}_fy*_s*.nc")):
            hits = list((base / sub).glob(pat)) if (base / sub).is_dir() else []
            for f in hits:
                m = re.search(r"_fy(\d+)_s\d+", f.name)
                if m:
                    leads.add(int(m.group(1)))
            if hits:
                break
    return sorted(leads)


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


def _grouped_r(df: pd.DataFrame, by: str, xcol: str = "era5",
               ycol: str = "dcpp") -> tuple[float, float, int]:
    """Mean and sd of within-group r, plus the number of usable groups."""
    rs = []
    for _, g in df.groupby(by):
        if len(g) >= MIN_N_PER_GROUP:
            r = _r(g[xcol].to_numpy(dtype=float), g[ycol].to_numpy(dtype=float))
            if np.isfinite(r):
                rs.append(r)
    if not rs:
        return np.nan, np.nan, 0
    return float(np.mean(rs)), float(np.std(rs)), len(rs)


def add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each unit's own temporal mean from each side separately.

    Each source is centred on ITS OWN climatology, not on a shared one, so a
    constant offset between the two (DCPP's warm bias, say) cannot leak into
    any anomaly correlation. The bias is reported separately instead.
    """
    g = df.groupby("adm2_id")
    return df.assign(e_anom=df["era5"] - g["era5"].transform("mean"),
                     d_anom=df["dcpp"] - g["dcpp"].transform("mean"))


def yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per harvest year: the cross-unit (spatial) skill for that year.

    r_acc is the anomaly correlation coefficient in the forecast-verification
    sense: within a single year, correlate the two anomaly FIELDS across admin
    units. r_space is the same correlation on raw values, which is dominated by
    the shared climatological pattern and therefore nearly always high.
    """
    d = add_anomalies(df)
    rows = []
    for yr, g in d.groupby("harvest_year"):
        n = len(g)
        ok = n >= MIN_N_PER_GROUP
        e, p = g["era5"].to_numpy(float), g["dcpp"].to_numpy(float)
        rows.append({
            "harvest_year": int(yr),
            "n_units": int(n),
            "r_acc": _r(g["e_anom"].to_numpy(float), g["d_anom"].to_numpy(float)) if ok else np.nan,
            "r_space": _r(e, p) if ok else np.nan,
            "bias": float(np.mean(p - e)),
            "rmse": float(np.sqrt(np.mean((p - e) ** 2))),
            "era5_mean": float(np.mean(e)),
            "dcpp_mean": float(np.mean(p)),
            "era5_sd": float(np.std(e)),
            "dcpp_sd": float(np.std(p)),
        })
    return pd.DataFrame(rows)


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
                   r_space=np.nan, r_space_sd=np.nan, n_years_scored=0,
                   r_acc=np.nan, r_acc_sd=np.nan, n_years_acc=0)
        return out

    out["r_pooled"] = _r(e, d)
    # Anomalies: remove each unit's own mean from BOTH sides. What is left is
    # the interannual variation, which is what a decadal hindcast is judged on.
    anom = add_anomalies(df)
    ea = anom["e_anom"].to_numpy(dtype=float)
    da = anom["d_anom"].to_numpy(dtype=float)
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
    # Within one unit, removing that unit's mean subtracts a constant, which
    # leaves correlation unchanged -- so r_time is already an anomaly
    # correlation and needs no separate raw/anomaly pair.
    rt, rts, nu = _grouped_r(df, "adm2_id")
    out.update(r_time=rt, r_time_sd=rts, n_units_scored=nu)
    # Within one YEAR the unit means differ across units, so raw and anomaly
    # versions are genuinely different quantities. Both are reported.
    rs, rss, ny = _grouped_r(df, "harvest_year")
    out.update(r_space=rs, r_space_sd=rss, n_years_scored=ny)
    ra, ras, nya = _grouped_r(anom, "harvest_year", "e_anom", "d_anom")
    out.update(r_acc=ra, r_acc_sd=ras, n_years_acc=nya)
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


LEAD_COLUMNS = ["aoi", "member", "feature", "forecast_year", "r_anom", "r_time",
                "r_acc", "r_space", "r_pooled", "bias", "rmse", "sd_ratio_anom",
                "n_pairs", "n_units", "n_years"]


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Headline metrics first, everything else after, nothing dropped."""
    return df[[c for c in LEAD_COLUMNS if c in df.columns]
              + [c for c in df.columns if c not in LEAD_COLUMNS]]


METRIC_NOTES = {
    "r_anom": "unit means removed from both sides; interannual skill (headline)",
    "r_time": "mean of per-unit temporal r; r_time_sd is its spread across units",
    "r_acc": "anomaly correlation coefficient: per-year cross-unit r on anomalies, averaged over years",
    "r_space": "per-year cross-unit r on RAW values, averaged; climatology-dominated",
    "r_pooled": "raw pooled r; inflated by between-unit climatology",
    "sd_ratio_anom": "sd(DCPP anom)/sd(ERA5 anom); <1 = underdispersed interannual amplitude",
    "sd_ratio_raw": "same on raw values; climatology-dominated, kept only for reference",
    "bias": "mean(DCPP - ERA5) in the feature's own units",
}


def write_aoi_outputs(aoi_out: Path, tag: str, rows: list[dict],
                      yearly: pd.DataFrame, pairs: pd.DataFrame, meta: dict,
                      failures: list) -> list[Path]:
    """One AOI's four tables, side by side with that AOI's logs."""
    aoi_out.mkdir(parents=True, exist_ok=True)
    summary = order_columns(pd.DataFrame(rows)).sort_values(["feature", "member"])
    csv_p = aoi_out / f"dcpp_era5_corr_{tag}.csv"
    json_p = aoi_out / f"dcpp_era5_corr_{tag}.json"
    year_p = aoi_out / f"dcpp_era5_year_{tag}.csv"
    pairs_p = aoi_out / f"dcpp_era5_pairs_{tag}.csv.gz"
    summary.to_csv(csv_p, index=False)
    if not yearly.empty:
        yearly.sort_values(["feature", "member", "harvest_year"]).to_csv(year_p, index=False)
    pairs.to_csv(pairs_p, index=False)
    with open(json_p, "w") as fh:
        json.dump({"run": meta, "metric_notes": METRIC_NOTES,
                   "failures": [{"aoi": x[0], "source": x[1], "why": x[2]}
                                for x in failures],
                   "results": json.loads(summary.to_json(orient="records"))},
                  fh, indent=2)
    return [csv_p, json_p, year_p, pairs_p] if not yearly.empty else [csv_p, json_p, pairs_p]


# =================================== main =====================================


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Correlate DCPP hindcast features against ERA5, per AOI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--aois", required=True, help="comma-separated AOI tags, e.g. ZM,ET")
    p.add_argument("--members", help="comma-separated DCPP members; default: discover")
    p.add_argument("--forecast-years", default=None,
                   help="comma-separated DCPP lead times, e.g. 1,2. "
                        "Default: every lead found in the DCPP tree.")
    p.add_argument("--forecast-year", type=int, default=None,
                   help="deprecated single-lead alias for --forecast-years")
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
    if a.forecast_year is not None:          # deprecated alias wins if given
        a.forecast_years = str(a.forecast_year)
    fy_list = parse_years(a.forecast_years) if a.forecast_years else None
    if fy_list is not None:
        if not fy_list:
            raise SystemExit(f"--forecast-years parsed to nothing: {a.forecast_years!r}")
        if min(fy_list) < 1:
            raise SystemExit(
                f"lead times must be >= 1 for a hindcast (got {sorted(fy_list)}).\n"
                "At 0 the DCPP outputs would take the observed naming convention "
                "and collide with the ERA5 tree's filenames.")

    # Source years each side needs. era5_features.py takes SOURCE years and
    # derives the harvest years itself, so these only have to be a superset.
    era5_src = sorted({y for h in harvest_years for y in season_source_years(h, 0)})
    def src_for(fy):
        return sorted({y for h in harvest_years for y in season_source_years(h, fy)})

    print("=" * 78)
    print("DCPP vs ERA5 -- cropland-weighted admin-2 features")
    print("=" * 78)
    print(f"  ERA5 tree   : {era5_root}/{{AOI}}")
    print(f"  DCPP tree   : {dcpp_root}/{{AOI}}_processed_data/{MODEL}_{{member}}")
    print(f"  data root   : {a.root}  (cy-bench crop calendar)")
    print(f"  out         : {out_dir}")
    print(f"  harvest yrs : {harvest_years[0]}-{harvest_years[-1]} ({len(harvest_years)})")
    print(f"  ERA5 source : {era5_src[0]}-{era5_src[-1]}")
    print(f"  DCPP leads  : "
          + (", ".join(f"FY{f}" for f in fy_list) if fy_list is not None
             else "discovered per AOI from the DCPP tree"))
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
        fys = fy_list if fy_list is not None else discover_leads(dcpp_root, aoi, mems)
        if not fys:
            print(f"  {aoi}: no fy*_s* files under "
                  f"{dcpp_dir(dcpp_root, aoi, mems[0])} -- skipping "
                  f"(run run-dcpp.sh, or pass --forecast-years)")
            continue
        plan.append((aoi, mems, fys))
        print(f"  {aoi}: {len(mems)} member(s): {', '.join(mems)}"
              f"   leads: {', '.join(f'FY{f}' for f in fys)}"
              + ("" if fy_list is not None else "  (discovered)"))
    if not plan:
        raise SystemExit("nothing to do")

    n_builds = sum(1 + len(m) * len(f) for _, m, f in plan)
    print(f"\n  {len(plan)} AOI(s), {n_builds} era5_features.py invocation(s)")
    # Echo the exact commands: this is the thing worth reviewing before a long
    # run, and reconstructing it from the flags above is error-prone.
    aoi0, mems0, fys0 = plan[0]
    print("\n  commands (first AOI shown; one per source):")
    print("    " + " ".join(build_cmd(a, aoi0, "ERA5", era5_dir(era5_root, aoi0), 0, era5_src)))
    print("    " + " ".join(build_cmd(a, aoi0, f"{MODEL}_{mems0[0]}",
                                      dcpp_dir(dcpp_root, aoi0, mems0[0]),
                                      fys0[0], src_for(fys0[0]))))
    if not a.go:
        print("\n  dry run -- add --go to execute")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, failures, written = [], [], []
    for aoi, mems, fys in plan:
        print(f"\n--- {aoi} ---")
        e_dir = era5_dir(era5_root, aoi)
        # One directory per AOI, holding that AOI's tables and its build logs.
        # report_dcpp_era5.py reads the same layout back by AOI name.
        aoi_out = out_dir / aoi
        (aoi_out / "logs").mkdir(parents=True, exist_ok=True)

        # ERA5 is lead-independent: built and loaded ONCE per AOI, then reused
        # for every lead. Rebuilding it per lead would spawn a process per lead
        # that does nothing but print "exists, skipping".
        if not a.skip_build:
            print("  building ERA5 features ...")
            cmd = build_cmd(a, aoi, "ERA5", e_dir, 0, era5_src)
            if not run_build(cmd, aoi_out / "logs" / f"{aoi}_ERA5.log"):
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

        for fy in fys:
            a.forecast_year = fy          # build_cmd reads this
            dcpp_src = src_for(fy)
            tag = a.tag or f"fy{fy}_{a.reduce}_{harvest_years[0]}-{harvest_years[-1]}"
            print(f"  -- FY{fy}  (initialisations s{dcpp_src[0]}-{dcpp_src[-1]}) --")

            aoi_pairs = []
            for mem in mems:
                source = f"{MODEL}_{mem}"
                d_dir = dcpp_dir(dcpp_root, aoi, mem)
                label = f"{source} FY{fy}"
                if not a.skip_build:
                    print(f"    building {label} ...")
                    cmd = build_cmd(a, aoi, source, d_dir, fy, dcpp_src)
                    if not run_build(cmd, aoi_out / "logs" / f"{aoi}_{source}_fy{fy}.log"):
                        failures.append((aoi, label, "build"))
                        continue
                dcpp = _load_side(d_dir, aoi, a.crop, fy, harvest_years,
                                  a.reduce, fixed, f"{aoi}/{source}/FY{fy}")
                if dcpp.empty:
                    failures.append((aoi, label, "no output"))
                    continue
                usable = [c for c in have if c in value_columns(dcpp)]
                if not usable:
                    failures.append((aoi, label, "no shared features"))
                    continue
                pr = pair_long(era5, dcpp, usable, mem)
                if pr.empty:
                    failures.append((aoi, label, "empty join"))
                    continue
                dropped = era5.adm2_id.nunique() - pr.adm2_id.nunique()
                print(f"      {source}: {len(pr)} paired values, "
                      f"{pr.adm2_id.nunique()} units"
                      + (f" ({dropped} ERA5 unit(s) absent from DCPP)"
                         if dropped > 0 else ""))
                aoi_pairs.append(pr)

            if not aoi_pairs:
                continue
            pairs = pd.concat(aoi_pairs, ignore_index=True)
            pairs.insert(0, "aoi", aoi)
            ens = ensemble_mean(pairs.drop(columns="aoi"))
            ens.insert(0, "aoi", aoi)
            # Pairs are written per (AOI, lead) below, so nothing accumulates
            # across AOIs: a 10-member x 21-year x 1900-unit AOI is millions
            # of rows on its own.
            both = pd.concat([pairs, ens], ignore_index=True)

            rows, yrows = [], []
            for member, mdf in both.groupby("member"):
                for feat, fdf in mdf.groupby("feature"):
                    key = {"aoi": aoi, "member": member, "feature": feat,
                           "forecast_year": fy}
                    rec = dict(key, reduce=a.reduce, weight=a.weight, crop=a.crop)
                    if member == "ENSMEAN":
                        rec["n_members"] = int(fdf["n_members"].max())
                    rec.update(metrics(fdf))
                    rows.append(rec)
                    ym = yearly_metrics(fdf)
                    for k, v in reversed(list(key.items())):
                        ym.insert(0, k, v)
                    yrows.append(ym)
            all_rows.extend(rows)
            yearly = pd.concat(yrows, ignore_index=True) if yrows else pd.DataFrame()

            meta = dict(
                created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), argv=sys.argv,
                model=MODEL, aoi=aoi, members=mems, forecast_year=fy,
                harvest_years=[harvest_years[0], harvest_years[-1]],
                era5_source_years=[era5_src[0], era5_src[-1]],
                dcpp_source_years=[dcpp_src[0], dcpp_src[-1]],
                features_requested=feats, compared_columns=sorted(set(have)),
                reduce=a.reduce, weight=a.weight, crop=a.crop,
                era5_dir=str(e_dir), dcpp_root=str(dcpp_root), data_root=a.root,
            )
            paths = write_aoi_outputs(aoi_out, tag, rows, yearly, both, meta,
                                      [f for f in failures if f[0] == aoi])
            written.extend(paths)
            for q in paths:
                print(f"      -> {q.name}")

    if not all_rows:
        raise SystemExit("no comparable data produced -- see the logs in "
                         f"{out_dir / 'logs'}")

    summary = pd.DataFrame(all_rows)
    summary = order_columns(summary).sort_values(["aoi", "feature", "member"])

    print("\n" + "=" * 78)
    print(f"wrote {len(written)} file(s) under {out_dir}/<AOI>/")
    print(f"report with: python python/report_dcpp_era5.py --dir {out_dir} "
          f"--aois {','.join(aoi for aoi, _, _ in plan)}")
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
