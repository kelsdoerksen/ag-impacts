#!/usr/bin/env python3
"""Drive ERA5 stages 1 and 2 over many cy-bench AOIs, one after another.

Stage 1 (R/era5-pixel-store.R) downloads and stores the per-pixel cubes and the
weight table; stage 2 (python/era5_features.py) computes per-pixel daily
features. Neither is parallelised here on purpose: CDS queues per account, so
two AOIs downloading at once share the same slots and finish no sooner.

SIZE THIS BEFORE YOU START IT. The whole cy-bench set is ~1200 AOI-years, and
CDS bills roughly one queued request per month of data, so the run is measured
in weeks. `--dry-run` (the default) prints the plan and an estimate; pass
`--go` to actually launch.

    python tools/run_cybench_era5.py --out /Volumes/PRO-G40/UCT-postdoc/era5
    python tools/run_cybench_era5.py --out ... --aois ZM,ET,MW,MZ --go
    python tools/run_cybench_era5.py --out ... --stage 2 --go     # features only

Resumption is free: both stages skip years that are already on disk, and this
driver records the AOIs it has finished in `_driver_state.json` under --out. A
failed AOI is logged and the run carries on to the next one rather than
stopping the queue.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "R" / "cybench-aois.json"
DEFAULT_ROOT = os.environ.get("AGIMPACTS_ROOT", "/Users/kdoerkse/Documents/UCT/AgImpacts")
# Queue time per request, which is almost all of the cost and varies enormously
# with how busy CDS is and how much the account has recently consumed. Measured
# on this setup: 3.5 min/request at best (Madagascar, single months, quiet
# queue), 20-25 min/request at worst (Burkina Faso, 3-month requests, after
# heavy use). The default is the pessimistic end, because a plan that finishes
# early is a better surprise than one that does not finish at all. Override with
# --min-per-request. Ethiopia measured ~25 min/month while a
# second run competed for the same queue slots, so treat this as the
# uncontended figure and expect worse if two jobs overlap. Queue time dominates,
# not AOI size -- a 600-cell country costs about what a 3000-cell one does.
MIN_PER_REQUEST = 20.0


def load_plan(root: Path, only=None, keep_order=False):
    """AOI -> (first_year, last_year), from the crop coverage in the registry.

    A season whose eos falls before its sos runs into the next calendar year, so
    harvest year Y needs climate from Y-1. Those AOIs get one extra year at the
    front -- the same reason Ethiopia starts at 1998 for yields beginning 1999.
    """
    import pandas as pd

    reg = json.loads(REGISTRY.read_text())
    plan = []
    for cc, x in sorted(reg.items()):
        if only and cc not in set(only):
            continue
        years = set()
        wraps = False
        for crop, k in x["crops"].items():
            years |= set(range(k["year_min"], k["year_max"] + 1))
            cal = root / "cy-bench" / "cybench-data" / crop / cc / f"crop_calendar_{crop}_{cc}.csv"
            if cal.exists():
                c = pd.read_csv(cal)
                wraps = wraps or bool((c["eos"] < c["sos"]).any())
        if not years:
            continue
        lo, hi = min(years), max(years)
        plan.append({"aoi": cc, "y0": lo - 1 if wraps else lo, "y1": hi,
                     "cells": x["grid_cells"], "units": x["polygons"], "wraps": wraps})
    # AOIs that stage 1 knows but the cy-bench registry does not -- CONUS above
    # all, which is the built-in US counties AOI rather than a cy-bench country.
    # There is no crop coverage to infer years from, so they are only usable
    # with an explicit --years, and the caller is told so rather than being left
    # with "no AOIs matched".
    if only:
        extra = [cc for cc in only if cc not in reg]
        for cc in extra:
            plan.append({"aoi": cc, "y0": None, "y1": None, "cells": 0,
                         "units": 0, "wraps": False, "off_registry": True})

    if keep_order and only:
        # An explicit --aois list is a priority order: a time-boxed run wants the
        # AOIs it cares about most to finish first, not the cheapest ones.
        rank = {cc: i for i, cc in enumerate(only)}
        plan.sort(key=lambda p: rank.get(p["aoi"], len(rank)))
    else:
        # Smallest first: the cheap AOIs finish early and prove the setup before
        # a big one ties the queue up for days.
        plan.sort(key=lambda p: (p["y1"] - p["y0"] + 1) * p["cells"])
    return plan


STAGE2_IMPORTS = ("xarray", "netCDF4", "pandas", "numpy", "pyarrow")


def check_stage2_python(python: str) -> list[str]:
    """Which of stage 2's imports this interpreter is missing.

    Checked up front because stage 1 runs first and takes hours: finding out at
    the end of an overnight download that the feature stage cannot import xarray
    is a bad way to spend a night. conda's `ag-deeplearning` carries pandas and
    pyarrow but neither xarray nor netCDF4, which is exactly the trap.
    """
    probe = ("import importlib.util;"
             f"print(','.join(m for m in {STAGE2_IMPORTS!r}"
             " if importlib.util.find_spec(m) is None))")
    try:
        out = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=60)
    except Exception as e:
        return [f"could not run {python}: {e}"]
    if out.returncode != 0:
        return [f"{python} failed to start: {out.stderr.strip()[:200]}"]
    return [m for m in out.stdout.strip().split(",") if m]


def run(cmd, log_path: Path, env) -> bool:
    """Run one stage, streaming into its own log. True if it exited clean.

    The exit status is written into the log as well as returned. R buffers its
    output when stdout is a file, so a process that dies abruptly can lose its
    error message entirely -- leaving a log that just stops mid-request with no
    explanation. The status line is written by this process, after the child is
    gone, so it always survives. A negative code means killed by that signal
    (-9 = SIGKILL, -15 = SIGTERM), which distinguishes "something killed it"
    from "it raised an error".
    """
    with log_path.open("a") as fh:
        fh.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    with log_path.open("a") as fh:
        if rc == 0:
            fh.write(f"\n[driver] {time.strftime('%Y-%m-%d %H:%M:%S')} exit 0 (ok)\n")
        elif rc < 0:
            fh.write(f"\n[driver] {time.strftime('%Y-%m-%d %H:%M:%S')} KILLED by signal "
                     f"{-rc} -- not an error raised by the job itself\n")
        else:
            fh.write(f"\n[driver] {time.strftime('%Y-%m-%d %H:%M:%S')} exit {rc} "
                     f"-- the job raised an error; its message may be above, or may have "
                     f"been lost to output buffering if it died abruptly\n")
        fh.flush()
    return rc == 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="root for the per-AOI output trees")
    p.add_argument("--root", default=DEFAULT_ROOT, help="AGIMPACTS_ROOT (inputs live here)")
    p.add_argument("--aois", help="comma-separated subset; default every cy-bench AOI")
    p.add_argument("--stage", default="both", choices=["1", "2", "both"])
    p.add_argument("--from-year", type=int, metavar="YEAR",
                   help="earliest year to fetch; each AOI keeps its own end year. Use 2001 to "
                        "match the cy-bench meteo/NDVI/FPAR series, which all start then. The "
                        "wrap-around extra year is still applied, so a season crossing new "
                        "year fetches YEAR-1 as well.")
    p.add_argument("--years", help="override the registry year span for every AOI in this "
                                   "run, e.g. 1980-2015. Cuts download time proportionally; "
                                   "the wrap-around extra year is NOT added on top, so pass "
                                   "the earlier start yourself if the season crosses new year.")
    p.add_argument("--features", default="gdd,edd,tavg,vpd,pr,hot_days")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter for stage 2 (needs xarray, netCDF4, pandas, pyarrow)")
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--source", default="cds", choices=["cds", "arco"])
    p.add_argument("--months-per-request", type=int, metavar="N",
                   help="months per CDS request, passed to stage 1. Unset lets it size from "
                        "the AOI and halve on a rejection. Pin it (1 is the long-proven "
                        "value) when CDS is refusing requests on cost grounds.")
    p.add_argument("--local-time", action="store_true",
                   help="bin daily values on local solar days rather than UTC days")
    p.add_argument("--min-per-request", type=float, default=MIN_PER_REQUEST,
                   metavar="MIN", help=f"minutes per CDS request for the estimate "
                                       f"(default {MIN_PER_REQUEST}, measured uncontended)")
    p.add_argument("--go", action="store_true", help="actually run; otherwise print the plan")
    a = p.parse_args(argv)

    out_root = Path(a.out)
    only = [c.strip() for c in a.aois.split(",")] if a.aois else None
    plan = load_plan(Path(a.root), only, keep_order=True)
    if not plan:
        sys.exit("no AOIs matched")
    if a.years:
        lo, _, hi = a.years.partition("-")
        for q in plan:
            q["y0"], q["y1"] = int(lo), int(hi or lo)
    off_reg = [q["aoi"] for q in plan if q.get("off_registry") and q["y0"] is None]
    if off_reg and not a.years:
        sys.exit(f"{', '.join(off_reg)} is not a cy-bench AOI, so its years cannot be "
                 f"derived from crop coverage. Pass --years explicitly, e.g. "
                 f"--years 2001-2023.")

    if a.from_year:
        # Clamp the start, keep each AOI's own end. A wrapping season still needs
        # the year before the first harvest year, so the floor is from_year - 1.
        plan = [q for q in plan if q["y1"] is None or q["y1"] >= a.from_year]
        for q in plan:
            if q["y0"] is None:
                continue
            q["y0"] = max(q["y0"], a.from_year - 1 if q["wraps"] else a.from_year)

    total_years = sum(p_["y1"] - p_["y0"] + 1 for p_ in plan)
    # Requests, not months: CDS charges queue time per request, and stage 1 asks
    # for several months at once. With the span pinned we know the count exactly;
    # without it, stage 1 sizes the span itself and 12 is the pessimistic bound.
    per_year = -(-12 // a.months_per_request) if a.months_per_request else 12
    reqs = total_years * per_year
    est = timedelta(minutes=reqs * a.min_per_request)

    print(f"{'AOI':<5}{'years':>13}{'n':>4}{'cells':>8}  wraps")
    for q in plan:
        span = f"{q['y0']}-{q['y1']}"
        print(f"{q['aoi']:<5}{span:>13}{q['y1']-q['y0']+1:>4}"
              f"{q['cells']:>8}  {'yes' if q['wraps'] else ''}")
    print(f"\n{len(plan)} AOIs, {total_years} AOI-years, "
          f"{per_year} request(s)/year, ~{reqs:,} CDS requests")
    hours = est.total_seconds() / 3600
    print(f"stage 1 estimate at {a.min_per_request:.1f} min/request: {hours:.0f} h "
          f"= {hours / 24:.1f} days. Stage 2 is local and adds minutes, not days.")
    print(f"output root: {out_root}")

    if a.stage in ("2", "both"):
        missing = check_stage2_python(a.python)
        if missing:
            print(f"\nSTAGE 2 CANNOT RUN with {a.python}")
            print(f"  missing: {', '.join(missing)}")
            print("  Use an interpreter that has them, e.g.")
            print("    --python /opt/anaconda3/envs/ag/bin/python")
            print("  or pass --stage 1 to download now and build features later.")
            if a.go:
                return 2
        else:
            print(f"stage 2 interpreter OK: {a.python}")

    if not a.go:
        print("\nDry run. Re-run with --go to start, and wrap it in caffeinate:")
        print(f"  caffeinate -is python {Path(__file__).relative_to(Path.cwd()) if Path.cwd() in Path(__file__).parents else __file__} "
              f"--out {out_root} --go")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "_driver_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    env = {**os.environ, "AGIMPACTS_ROOT": a.root}
    # ERA5_OUT_DIR names one AOI's tree, so it must not leak across the loop --
    # every AOI gets an explicit --out instead.
    env.pop("ERA5_OUT_DIR", None)

    t_start = time.time()
    for i, q in enumerate(plan, 1):
        aoi, y0, y1 = q["aoi"], q["y0"], q["y1"]
        if state.get(aoi) == "done" and a.stage == "both":
            print(f"[{i}/{len(plan)}] {aoi}: already done, skipping")
            continue

        aoi_out = out_root / aoi
        aoi_out.mkdir(parents=True, exist_ok=True)
        log = aoi_out / f"driver_{aoi}.log"
        years = f"{y0}-{y1}"
        elapsed = timedelta(seconds=int(time.time() - t_start))
        print(f"[{i}/{len(plan)}] {aoi} {years} -> {aoi_out}  (elapsed {elapsed})", flush=True)

        ok = True
        if a.stage in ("1", "both"):
            cmd = [a.rscript, str(REPO / "R" / "era5-pixel-store.R"),
                   f"--aoi={aoi}", f"--source={a.source}", f"--out={aoi_out}"]
            if a.months_per_request:
                cmd.append(f"--months-per-request={a.months_per_request}")
            if a.local_time:
                cmd.append("--local-time")
            cmd.append(years)
            ok = run(cmd, log, env)
            if not ok:
                print(f"    stage 1 FAILED - see {log}", flush=True)
                state[aoi] = "stage1-failed"

        if ok and a.stage in ("2", "both"):
            ok = run([a.python, str(REPO / "python" / "era5_features.py"),
                      f"--aoi={aoi}", f"--years={years}", "--stage=features",
                      f"--features={a.features}", f"--out={aoi_out}", f"--root={a.root}"],
                     log, env)
            if not ok:
                print(f"    stage 2 FAILED - see {log}", flush=True)
                state[aoi] = "stage2-failed"

        if ok:
            state[aoi] = "done"
            print(f"    done", flush=True)
        state_path.write_text(json.dumps(state, indent=2))

    failed = {k: v for k, v in state.items() if v != "done"}
    print(f"\nFinished in {timedelta(seconds=int(time.time() - t_start))}. "
          f"{sum(1 for v in state.values() if v == 'done')} AOIs done, {len(failed)} failed.")
    if failed:
        print("failed:", ", ".join(f"{k} ({v})" for k, v in failed.items()))
        print("Re-run the same command; finished years are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
