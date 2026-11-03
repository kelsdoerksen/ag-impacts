#!/usr/bin/env python
"""Per-AOI crop calendar summary, and the DCPP slices each season needs.

Reports the same quantities compare_dcpp_era5.calendar_fit() tests and
era5_features.season_windows() builds, so the numbers here explain the
SKIPPED/note lines that run prints rather than approximating them.

  python python/crop_calendar_table.py --root /mnt/crl_storage/personal_folders/kdoerksen
  python python/crop_calendar_table.py --root ... --harvest-year 2003 --slices

A slice is identified as fy{L}_s{S} and spans 1 Nov (S+L-1) -> 31 Oct (S+L),
so slice_span(S, L) == slice_span(S+1, L-1): the same calendar window reached
at two different lead times. That identity is why "combine s2001 with s2002"
has to say which window each run supplies.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NOV1, OCT31 = 305, 304          # DOY edges of a published slice
MONTH = 31                   # for reporting leads in months
DEFAULT_AOIS = "AR,BF,BR,ET,FR,HU,MG,ML,MW,RO,SN,US,ZM"
# FIPS state codes outside the contiguous 48. The US AOI bbox is CONUS
# (lon -124.98..-71.36, lat 25.34..49.88), so these units have no polygon
# inside it and never reach stage 3 -- but calendar_fit() reads the raw
# calendar and warns about them anyway.
NON_CONUS = ("02", "15", "60", "66", "69", "72", "78")


def doy_date(doy: float) -> str:
    """DOY as a day-month label, on a non-leap reference year."""
    if not np.isfinite(doy):
        return "--"
    return (pd.Timestamp("2001-01-01")
            + pd.Timedelta(days=int(round(doy)) - 1)).strftime("%d %b")


def drop_non_conus(c: pd.DataFrame) -> pd.DataFrame:
    """Keep only contiguous-48 counties, by FIPS state code in US-SS-CCC."""
    st = c.adm_id.astype(str).str.split("-").str[1]
    return c[~st.isin(NON_CONUS)]


def calendar(root: Path, crop: str, aoi: str,
             conus: bool = False) -> pd.DataFrame | None:
    p = (root / "cy-bench" / "cybench-data" / crop / aoi
         / f"crop_calendar_{crop}_{aoi}.csv")
    if not p.exists():
        return None
    c = pd.read_csv(p)
    if not {"sos", "eos"} <= set(c.columns):
        return None
    c = c[np.isfinite(c.sos) & np.isfinite(c.eos)]
    if conus and aoi == "US" and "adm_id" in c.columns:
        c = drop_non_conus(c)
    return c


def span(s_year: int, lead: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(year=s_year + lead - 1, month=11, day=1),
            pd.Timestamp(year=s_year + lead, month=10, day=31))


def window(sos: int, eos: int, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Season window for harvest year `year`, as season_windows() anchors it."""
    start_year = year - 1 if sos > eos else year
    return (pd.Timestamp(f"{start_year}-01-01") + pd.Timedelta(days=sos - 1),
            pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=eos - 1))


def option_b(year: int, fy: int, lo, hi) -> list[tuple[int, int]]:
    """Slices under one fixed initialisation S = year - fy, leads varying."""
    s = year - fy
    return [(s, L) for L in range(1, 11)
            if span(s, L)[1] >= lo and span(s, L)[0] <= hi]


def lead_table(c: pd.DataFrame, fys, floor_mo: float, H: int = 2010):
    """Per-lead horizon stats: days from the 1 Nov initialisation to sowing
    and to harvest, per unit. Year-invariant, so H is arbitrary.

    `floor_mo` is a minimum acceptable lead on HARVEST, applied per unit --
    a median above the floor says nothing about the earliest-harvesting
    district, which is the one that breaks an operational claim.
    """
    sos = c.sos.round().astype(int).to_numpy()
    eos = c.eos.round().astype(int).to_numpy()
    wins = [window(s, e, H) for s, e in zip(sos, eos)]
    floor_d = floor_mo * MONTH
    out = {}
    for fy in fys:
        init = pd.Timestamp(year=H - fy, month=11, day=1)
        plant = np.array([(a - init).days for a, _ in wins], float)
        harv = np.array([(b - init).days for _, b in wins], float)
        # Coverable == the season does not start before lead 1 of its own run.
        cov = np.mean([span(H - fy, 1)[0] <= a for a, _ in wins]) * 100
        out[fy] = dict(plant_med=np.median(plant), harv_med=np.median(harv),
                       harv_min=harv.min(), harv_p05=np.percentile(harv, 5),
                       pass_floor=np.mean(harv >= floor_d) * 100, cov=cov)
    return out


def init_vs_planting(rows, H: int) -> pd.DataFrame:
    """Per-AOI FY1 initialisation date against the planting dates it precedes.

    The binding date is the EARLIEST-planting unit, not the median: a forecast
    is only "before planting" if it beats the first field in the ground.
    """
    init = pd.Timestamp(year=H - 1, month=11, day=1)
    out = []
    for aoi, c, _e, _l in rows:
        sos = c.sos.round().astype(int).to_numpy()
        eos = c.eos.round().astype(int).to_numpy()
        plant = pd.DatetimeIndex([window(x, y, H)[0] for x, y in zip(sos, eos)])
        d = (plant - init).days.to_numpy()
        after = float((d > 0).mean() * 100)
        out.append(dict(
            aoi=aoi, n_units=len(c), harvest_year=H,
            dcpp_slice=f"fy1_s{H-1}", init_date=init.date(),
            slice_start=init.date(), slice_end=pd.Timestamp(f"{H}-10-31").date(),
            first_planting=plant.min().date(),
            median_planting=plant.sort_values()[len(plant) // 2].date(),
            last_planting=plant.max().date(),
            days_init_to_first_planting=int(d.min()),
            days_init_to_median_planting=int(np.median(d)),
            pct_units_planted_after_init=round(after, 2),
            before_planting=("yes_whole_aoi" if d.min() > 0 else
                             "no_majority_already_planted" if after < 50 else
                             "partly_some_already_planted")))
    return pd.DataFrame(out)


def horizon_by_lead(rows, fys, floor_mo: float, H: int = 2010) -> pd.DataFrame:
    """Per-AOI, per-lead forecast horizon, with the floor applied per unit."""
    out = []
    for aoi, c, _e, _l in rows:
        t = lead_table(c, fys, floor_mo, H)
        pick = next((f for f in fys
                     if t[f]["cov"] == 100 and t[f]["pass_floor"] == 100), None)
        sos = c.sos.round().astype(int).to_numpy()
        eos = c.eos.round().astype(int).to_numpy()
        wins = [window(x, y, H) for x, y in zip(sos, eos)]
        for fy in fys:
            i = pd.Timestamp(year=H - fy, month=11, day=1)
            pl = np.array([(x - i).days for x, _ in wins], float)
            hv = np.array([(y - i).days for _, y in wins], float)
            r = t[fy]
            out.append(dict(
                aoi=aoi, n_units=len(c), lead=f"FY{fy}",
                mo_pre_plant_min=round(pl.min() / MONTH, 2),
                mo_pre_plant_median=round(np.median(pl) / MONTH, 2),
                mo_pre_harvest_min=round(hv.min() / MONTH, 2),
                mo_pre_harvest_p05=round(np.percentile(hv, 5) / MONTH, 2),
                mo_pre_harvest_median=round(np.median(hv) / MONTH, 2),
                mo_pre_harvest_max=round(hv.max() / MONTH, 2),
                pct_units_above_floor=round(r["pass_floor"], 2),
                pct_units_coverable=round(r["cov"], 2),
                usable=bool(r["cov"] == 100 and r["pass_floor"] == 100),
                shortest_usable_lead=(fy == pick),
                floor_months=floor_mo))
    return pd.DataFrame(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="share root holding cy-bench/")
    p.add_argument("--crop", default="maize")
    p.add_argument("--aois", default=DEFAULT_AOIS)
    p.add_argument("--harvest-year", type=int, default=None,
                   help="also print the window and slices for this harvest year")
    p.add_argument("--forecast-years", default="1,2",
                   help="leads to resolve slices at (with --harvest-year)")
    p.add_argument("--conus", action="store_true",
                   help="restrict the US AOI to the contiguous 48 (drops HI, PR "
                        "and other territories, which lie outside the US AOI "
                        "bbox and never reach stage 3)")
    p.add_argument("--csv-dir", default=None,
                   help="also write both tables as CSVs into this directory")
    p.add_argument("--leads", action="store_true",
                   help="horizon of each lead, in months before planting and "
                        "before harvest, per unit")
    p.add_argument("--floor-months", type=float, default=5.0,
                   help="minimum acceptable months before harvest, applied "
                        "PER UNIT (default 5)")
    p.add_argument("--slices", action="store_true",
                   help="print the per-lead slice list under both schemes")
    a = p.parse_args(argv)

    root = Path(a.root)
    aois = [x.strip() for x in a.aois.split(",") if x.strip()]
    fys = [int(x) for x in a.forecast_years.split(",") if x.strip()]

    hdr = (f"{'AOI':<5} {'units':>6} {'median sos':>14} {'median eos':>14} "
           f"{'len':>5} {'cross':>7} {'sow<1Nov':>9} {'eos>31Oct':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for aoi in aois:
        c = calendar(root, a.crop, aoi, conus=a.conus)
        if c is None or c.empty:
            print(f"{aoi:<5} {'--':>6}   no usable calendar")
            continue
        cross = (c.sos > c.eos)
        early = cross & (c.sos < NOV1)
        late = (c.eos > OCT31)
        length = np.where(cross, 365 - c.sos + c.eos, c.eos - c.sos)
        sos_m, eos_m = c.sos.median(), c.eos.median()
        print(f"{aoi:<5} {len(c):>6} "
              f"{doy_date(sos_m) + f' ({sos_m:.0f})':>14} "
              f"{doy_date(eos_m) + f' ({eos_m:.0f})':>14} "
              f"{np.median(length):>5.0f} "
              f"{cross.mean() * 100:>6.1f}% "
              f"{early.mean() * 100:>8.1f}% "
              f"{late.mean() * 100:>9.1f}%")
        rows.append((aoi, c, early, late))

    if a.leads:
        fl = a.floor_months
        print(f"\nforecast horizon per lead (months). floor = {fl:.0f} mo "
              f"before harvest, applied per unit")
        print(f"{'AOI':<4} {'lead':>4} | {'pre-plant':>9} | "
              f"{'pre-harvest: min':>16} {'p05':>6} {'median':>7} "
              f"{'>=floor':>8} | {'coverable':>9}")
        print("-" * 78)
        for aoi, c, _e, _l in rows:
            t = lead_table(c, fys, fl)
            pick = next((f for f in fys
                         if t[f]["cov"] == 100 and t[f]["pass_floor"] == 100), None)
            for fy in fys:
                r = t[fy]
                star = " <-- arm lead" if fy == pick else ""
                print(f"{aoi:<4} {'FY' + str(fy):>4} | "
                      f"{r['plant_med'] / MONTH:>9.1f} | "
                      f"{r['harv_min'] / MONTH:>16.1f} "
                      f"{r['harv_p05'] / MONTH:>6.1f} "
                      f"{r['harv_med'] / MONTH:>7.1f} "
                      f"{r['pass_floor']:>7.0f}% | {r['cov']:>8.1f}%{star}")
            if pick is None:
                print(f"{aoi:<4}      |   no lead in {fys} clears both the floor "
                      f"and full coverage")
        print("\n  a lead is usable only if coverable == 100% AND >=floor == 100%")

    if a.csv_dir:
        out = Path(a.csv_dir); out.mkdir(parents=True, exist_ok=True)
        suffix = "_conus" if a.conus else ""
        H = a.harvest_year or 2003
        t1 = init_vs_planting(rows, H)
        t2 = horizon_by_lead(rows, fys, a.floor_months)
        for nm, df in (("dcpp_fy1_init_vs_planting", t1),
                       ("dcpp_forecast_horizon_by_lead", t2)):
            fp = out / f"{nm}{suffix}.csv"
            df.to_csv(fp, index=False)
            print(f"\n  wrote {fp}  ({len(df)} rows)")

    if a.harvest_year is None:
        return 0

    H = a.harvest_year
    print(f"\nharvest year {H}: season envelope across all units, "
          f"and the slices it needs")
    for aoi, c, early, late in rows:
        lo = min(window(int(round(s)), int(round(e)), H)[0]
                 for s, e in zip(c.sos, c.eos))
        hi = max(window(int(round(s)), int(round(e)), H)[1]
                 for s, e in zip(c.sos, c.eos))
        print(f"\n  {aoi}: {lo.date()} -> {hi.date()} "
              f"({(hi - lo).days} d envelope, {len(c)} units)")
        if not a.slices:
            continue
        for fy in fys:
            b = option_b(H, fy, lo, hi)
            spans = ", ".join(f"fy{L}_s{S} [{span(S, L)[0].date()}..{span(S, L)[1].date()}]"
                              for S, L in b)
            print(f"    FY{fy} one-initialisation stitch: {spans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
