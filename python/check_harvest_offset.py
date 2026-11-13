#!/usr/bin/env python
"""Is a yield table's harvest_year label aligned with the season it measures?

The pipeline passes cy-bench's harvest_year straight through
(R/crop-data-gen-cybench.R writes hindcast_start = harvest_year - 1,
yield_features.load_yield adds the 1 back). So if cy-bench labels a state's
yields one year late, every climate feature is paired with the wrong season and
nothing downstream will say so -- the model just sees noise. This checks the
labels against two independent references, per adm-1 unit (state):

  1. KNOWN EVENTS. Documented drought seasons, given as the calendar year of
     harvest. For each, the state-median yield anomaly at H-1, H and H+1. A
     correct label puts the loss at H; a +1 offset puts it at H+1 every time.

  2. CLIMATE LAG (needs ERA5 stage-3 features). Pool every unit-year in the
     state and correlate the yield anomaly labelled Y with the ERA5 season
     feature for harvest Y+k, k in {-1, 0, +1}. The k with the strongest
     correlation of the expected sign is the alignment the data supports.
     k = -1 winning means the yield labelled Y belongs to season Y-1, i.e. the
     label is one year late.

Anomalies are per unit: a linear trend is fit to each unit's yield and the
anomaly is 100 (y - trend) / trend. The trend absorbs technology, which would
otherwise correlate with any trending climate variable at every lag equally.

Neither test is proof alone. Events are few; the lag test needs a variable
that actually limits yield in that state (precipitation in rainfed southern
and north-eastern Brazil works; it will be flat in irrigated or
non-water-limited areas). Agreement between the two is the signal.

    python python/check_harvest_offset.py --aoi BR
    python python/check_harvest_offset.py --aoi BR --yield-table ~/.../crop_corn_BR.parquet --no-climate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Documented drought seasons, by calendar year of the (summer) maize harvest.
# adm1 keys are the IBGE state code as R/crop-data-gen-cybench.R writes it.
EVENTS = {
    "BR": {
        "BR-43": [1991, 2005, 2012, 2020, 2022],   # RS: La Nina droughts
        "BR-41": [2022],                           # PR: 2021/22 drought
        "BR-50": [2022],                           # MS: 2021/22 drought
        "BR-23": [1983, 1993, 1998, 2012],         # CE: north-east droughts
        "BR-26": [1983, 1993, 1998, 2012],         # PE
        "BR-22": [1983, 1993, 1998, 2012],         # PI
        "BR-24": [1983, 1993, 1998, 2012],         # RN
        "BR-25": [1983, 1993, 1998, 2012],         # PB
        "BR-31": [2014],                           # MG: 2013/14 south-east drought
    },
}


def adm1_of(adm2_id: pd.Series) -> pd.Series:
    """Same parent-code rule as cybench_adm1() in R/crop-data-gen-cybench.R."""
    s = adm2_id.astype(str)
    out = s.str[:3]
    synth = s.str.match(r"^[A-Z]{2}-A2-")
    fips = s.str.match(r"^[A-Z]{2}-[0-9]+-")
    ibge = s.str.match(r"^[A-Z]{2}[0-9]")
    out[synth] = s[synth].str[:2] + "-A1-" + s[synth].str[6:8]
    out[fips] = s[fips].str.extract(r"^([A-Z]{2}-[0-9]+)-", expand=False)[fips]
    out[ibge] = s[ibge].str[:2] + "-" + s[ibge].str[2:4]
    return out


def detrended_anomaly(df: pd.DataFrame, col: str, min_years: int = 8) -> pd.Series:
    """Per-unit 100 (x - linear trend) / trend; NaN for short or tiny series."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("adm2_id"):
        if g["harvest_year"].nunique() < min_years:
            continue
        c = np.polyfit(g["harvest_year"], g[col], 1)
        f = np.polyval(c, g["harvest_year"])
        ok = f > 0
        out.loc[g.index[ok]] = 100 * (g[col].to_numpy()[ok] - f[ok]) / f[ok]
    return out


def residual(df: pd.DataFrame, col: str, min_years: int = 8) -> pd.Series:
    """Per-unit linear detrend in the variable's own units (for climate)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("adm2_id"):
        if g["harvest_year"].nunique() < min_years or g[col].isna().any():
            continue
        c = np.polyfit(g["harvest_year"], g[col], 1)
        out.loc[g.index] = g[col].to_numpy() - np.polyval(c, g["harvest_year"])
    return out


def load_yields(path: Path, shift: int = 0) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["harvest_year"] = d["hindcast_start"].astype(int) + 1 + shift
    d = d[np.isfinite(d["yield"]) & (d["yield"] > 0)].copy()
    d["adm1"] = adm1_of(d["adm2_id"])
    d["y_anom"] = detrended_anomaly(d, "yield")
    return d[["adm2_id", "adm1", "harvest_year", "yield", "y_anom"]]


def event_table(y: pd.DataFrame, events: dict) -> pd.DataFrame:
    med = y.groupby(["adm1", "harvest_year"])["y_anom"].median()
    n = y.groupby("adm1")["adm2_id"].nunique()
    rows = []
    for adm1, years in events.items():
        for h in years:
            vals = {k: med.get((adm1, h + k), np.nan) for k in (-1, 0, 1)}
            if all(np.isnan(v) for v in vals.values()):
                continue
            worst = min(vals, key=lambda k: np.inf if np.isnan(vals[k]) else vals[k])
            rows.append({"adm1": adm1, "units": int(n.get(adm1, 0)), "event": h,
                         "H-1": vals[-1], "H": vals[0], "H+1": vals[1],
                         "lowest_at": {-1: "H-1", 0: "H", 1: "H+1"}[worst]})
    return pd.DataFrame(rows)


def lag_table(y: pd.DataFrame, clim: pd.DataFrame, feats, min_units: int) -> pd.DataFrame:
    """Pooled corr of yield anomaly(Y) with climate residual(Y+k), per state."""
    c = clim[["adm2_id", "harvest_year"] + feats].copy()
    for f in feats:
        c[f] = residual(c, f)
    rows = []
    for adm1, g in y.dropna(subset=["y_anom"]).groupby("adm1"):
        if g["adm2_id"].nunique() < min_units:
            continue
        for f in feats:
            r = {}
            for k in (-1, 0, 1):
                cc = c[["adm2_id", "harvest_year", f]].assign(
                    harvest_year=lambda d: d.harvest_year - k)   # climate of Y+k onto label Y
                j = g.merge(cc, on=["adm2_id", "harvest_year"]).dropna(subset=[f])
                r[k] = (float(np.corrcoef(j["y_anom"], j[f])[0, 1])
                        if len(j) > 30 and j[f].std() > 0 else np.nan)
            best = max(r, key=lambda k: -np.inf if np.isnan(r[k]) else abs(r[k]))
            rows.append({"adm1": adm1, "units": g["adm2_id"].nunique(), "feature": f,
                         "r(k=-1)": r[-1], "r(k=0)": r[0], "r(k=+1)": r[1],
                         "strongest": {-1: "k=-1", 0: "k=0", 1: "k=+1"}[best]})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", required=True)
    p.add_argument("--crop", default="corn")
    p.add_argument("--share", default=None, help="data root (default: yield_features.DEFAULT_SHARE)")
    p.add_argument("--yield-table", default=None,
                   help="crop_{crop}_{AOI}.parquet; default {share}/{AOI}_processed_data/")
    p.add_argument("--features", default="pr,edd,vpd",
                   help="ERA5 season features for the lag test")
    p.add_argument("--min-units", type=int, default=10,
                   help="skip states with fewer units in the lag test")
    p.add_argument("--no-climate", action="store_true", help="events test only")
    p.add_argument("--shift", type=int, default=0,
                   help="apply this harvest_year correction first, to verify it "
                        "(same as run_yield_experiment.py --yield-shift)")
    a = p.parse_args(argv)

    from yield_features import DEFAULT_SHARE, Roots, load_climate
    pd.set_option("display.width", 160)

    roots = None
    if a.yield_table:
        ypath = Path(a.yield_table).expanduser()
    else:
        roots = Roots.resolve(a.share or DEFAULT_SHARE)
        ypath = roots.aoi_targets(a.aoi) / f"crop_{a.crop}_{a.aoi}.parquet"
    y = load_yields(ypath, a.shift)
    print(f"{ypath}\n  {len(y):,} rows, {y.adm2_id.nunique():,} units, "
          f"harvest years {y.harvest_year.min()}-{y.harvest_year.max()}\n")

    ev = EVENTS.get(a.aoi)
    if ev:
        t = event_table(y, ev)
        print("1 · KNOWN EVENTS -- state-median detrended yield anomaly (%) around each")
        print("    documented drought harvest H. Correct labels put the low at H.\n")
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.0f}"))
        print()
        for adm1, g in t.groupby("adm1"):
            c = g["lowest_at"].value_counts()
            print(f"  {adm1}: lowest at " + ", ".join(f"{k} x{v}" for k, v in c.items())
                  + f"  ({len(g)} event(s))")
        print()
    else:
        print(f"(no event list for {a.aoi}; add one to EVENTS)\n")

    if a.no_climate:
        return 0
    roots = roots or Roots.resolve(a.share or DEFAULT_SHARE)
    clim, feats, _ = load_climate(roots, a.aoi, "era5", a.crop, 0)
    want = [f for f in a.features.split(",") if f in feats]
    if not want:
        raise SystemExit(f"none of {a.features} in the ERA5 store; have {feats}")
    t = lag_table(y, clim, want, a.min_units)
    print("\n2 · CLIMATE LAG -- corr of yield anomaly labelled Y with ERA5 season")
    print("    feature of harvest Y+k. k=0 strongest = aligned; k=-1 strongest =")
    print("    the yield belongs to the season before its label (label 1 year late).\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    if not t.empty:
        votes = (t.groupby("adm1")["strongest"]
                 .agg(lambda s: s.value_counts().idxmax()).value_counts())
        print("\n  states by majority alignment: "
              + ", ".join(f"{k} {v}" for k, v in votes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
