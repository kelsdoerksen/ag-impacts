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
DEFAULT_AOIS = "AR,BF,BR,ET,FR,HU,MG,ML,RO,SN,US,ZM"


def doy_date(doy: float) -> str:
    """DOY as a day-month label, on a non-leap reference year."""
    if not np.isfinite(doy):
        return "--"
    return (pd.Timestamp("2001-01-01")
            + pd.Timedelta(days=int(round(doy)) - 1)).strftime("%d %b")


def calendar(root: Path, crop: str, aoi: str) -> pd.DataFrame | None:
    p = (root / "cy-bench" / "cybench-data" / crop / aoi
         / f"crop_calendar_{crop}_{aoi}.csv")
    if not p.exists():
        return None
    c = pd.read_csv(p)
    if not {"sos", "eos"} <= set(c.columns):
        return None
    return c[np.isfinite(c.sos) & np.isfinite(c.eos)]


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
        c = calendar(root, a.crop, aoi)
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
