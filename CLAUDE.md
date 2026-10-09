# CLAUDE.md

Context for working on this repo. `README.md` says what each script does; this
says what the conventions are, what has already been decided, and what trips
people up. Read both.

## What this project is

Predicting crop yield from climate, at admin-2 resolution, for two climate
sources that must be directly comparable:

- **ERA5** reanalysis — observed, the skill ceiling
- **EC-Earth3 DCPP** hindcasts — forecasts at 1–5 year lead, the thing being evaluated

Yield targets come from **cy-bench** (42 maize AOIs). The research question is
how much of interannual yield variation a decadal forecast can recover, so the
two sources have to be processed identically — a difference in processing would
be indistinguishable from a difference in forecast skill.

## The one rule that is not up for discussion

**Features are computed per pixel, then averaged over cropland. Never the
reverse.** Aggregating first is only valid for features linear in their inputs:

```
mean_i( max(0, T_i - Tbase) )  !=  max(0, mean_i(T_i) - Tbase)
```

Averaging first clips the tails, so a unit holding both a cool and a hot pixel
loses the heat the hot pixel actually accumulated. This affects GDD, EDD, VPD
and every threshold feature. The aggregate-first code paths were deleted in
Oct 2026 (`era5-sample-gen.R`, `ML-sample-gen.R`'s `main()`, and
`data_loader.py`'s `aggregated`/`timeseries`/`agro` modes). **Do not propose
reintroducing them.**

```
stage 1  R/{era5,dcpp}-pixel-store.R   daily climate -> per-pixel cube + weight table
stage 2  python/era5_features.py       cube -> per-pixel DAILY features   (no weighting)
stage 3  python/era5_features.py       features + weights -> admin-2      (weighting HERE)
```

`R/pixel-store.R` is the shared stage-1 engine; the two front-ends differ only
in how they obtain a year's daily rasters. Keep it that way — it is what stops
the two sources drifting apart.

## Conventions that bite

- **`hindcast_start` is FY1**: the yield on a row is for calendar year
  `hindcast_start + 1`. Season windows are Nov(H) → Oct(H+1).
- **DCPP target year = initialisation year + forecast year.** A file for s-year
  Y at lead F spans 1 Nov (Y+F-1) to 31 Oct (Y+F). Wanting harvest years
  2002–2022 at FY1 means **s2001–s2021**. `run-dcpp.sh` does this arithmetic;
  check its printed table before any long run.
- **`--years` means SOURCE years** (initialisation years for a hindcast,
  calendar years for ERA5), not harvest years.
- **`US` and `CONUS` are different AOIs.** `US` = cy-bench polygons, `adm2_id`
  like `US-19-169`, joins to cy-bench yield. `CONUS` = Census counties,
  `adm2_id` like `19169`, does **not** join. Use `US` for anything involving
  cy-bench yield.
- **Units.** Cubes store analysis units (°C, %, mm/day); the GeoTIFF download
  cache stays in raw units (K, precip flux). Yield is kg/ha everywhere —
  cy-bench's t/ha is multiplied by 1000 on the way in.
- **One AOI registry.** `AOI_BOUNDS` in `R/era5.R`, built from
  `R/cybench-aois.json` via `R/build-cybench-aois.R`. DCPP derives its 0–360
  window from the same registry with 1.5° padding. Do not add a second list.

## Where things live

| | |
|---|---|
| `AGIMPACTS_ROOT` | the data root; everything else derives from it |
| Mac | `/Users/kdoerkse/Documents/UCT/AgImpacts` |
| Lab PC (`crl@137.158.31.55`) | `/mnt/crl_storage/personal_folders/kdoerksen` |
| DCPP archive | `/mnt/crl_storage/all_datasets/dcpp_daily/EC-Earth3/{member}/day` |
| DCPP output | `{root}/dcpp/{AOI}_processed_data/EC-Earth3_{member}` |

Pass `--root` / `--out` on the command line rather than exporting
`AGIMPACTS_ROOT`: a `tmux new` attached to an existing server does **not**
inherit the parent shell's exports, which has silently produced Mac paths on the
lab PC more than once.

## Environment

- **Lab PC stage 1**: `conda activate agimpacts-r`. Without it you get the
  system R and `reticulate` fails on an ABI mismatch (`undefined symbol:
  SETLENGTH`).
- **Mac stages 2–3**: the `ag` env. `ag-deeplearning` has a broken `netCDF4`
  (links `libjpeg.8`, which is absent) — use `--python
  /opt/anaconda3/envs/ag/bin/python`.
- `caffeinate -is` on macOS only; it is an unrelated package on Linux.
- macOS blocks newly-mounted volumes until the terminal app restarts (TCC).

## Decisions already made — do not re-litigate

- **Daily extremes are on UTC days, not local solar days.** `--local-time`
  exists but is unused. Verified: for every AOI in play the UTC day brackets
  both the local sunrise minimum and the afternoon maximum. The only exception
  is 2.4% of US counties west of 120°W, representing **0.33% of US maize area**.
  UTC also keeps ERA5 symmetric with DCPP, which has no equivalent shift.
- **ARCO for large AOIs, CDS for small ones.** CDS charges queue time per
  request (18–25 min) and refuses large ones on cost; ARCO has no queue but
  transfers global fields, so its cost depends only on the number of years.
  A >20,000-cell month will be refused by CDS.
- **DCPP `hursmin` is not ERA5's `hursmin`.** ERA5's is a true daily minimum of
  hourly RH; EC-Earth3's is RH from daily-mean T and dewpoint. Same column name,
  different quantity, and VPD inherits it. This is a property of the input —
  note it in methods, do not try to "fix" it.
- **Degree-day thresholds live at stage 2**, set with `--param gdd.tbase=8`, and
  are recorded in each cube's attributes. There is deliberately no fit-time
  flag, because by then the feature is a finished column.

## Known gotchas

- **cy-bench bounding boxes** are padded 0.5° around the polygons that have
  *yield data*, not the whole shapefile. A yield-less polygon can fall outside
  the box and be clipped — `units_clipped` in `grid_{AOI}.json` counts them.
  Measured: ET 1, BR 1, US 164, all with zero yield-bearing units affected.
- **`EE`, `FI`, `IE`, `LV` maize yields are 100% zeros** in cy-bench. Unusable;
  exclude them.
- **Brazil's shapefile has no `.prj`.** `load_cybench_boundaries()` assumes
  EPSG:4326 when the CRS is missing and the coordinates are degrees.
- **ARCO has no intra-year cache.** Killing it mid-year loses that year back to
  January. CDS months land in `cds_nc/` and survive.
- `jsonlite::write_json` defaults to 4 decimal places — the grid manifest needs
  `digits = NA` or resumed runs fail their own grid check.

## Priorities

Measured per-AOI climate signal: detrend each unit's yield, then look at CV
(how much interannual variation exists) and mean pairwise correlation of unit
anomalies (how spatially coherent the shock is).

1. **RO, HU** — strongest coherent signal in cy-bench (mean_r 0.68/0.76),
   rainfed continental maize, and small enough to be 1 CDS request per year.
2. **AR** — 409 units, 54 years, 17k observations, ENSO-driven. Worth its full
   1970– record rather than 2001–.
3. **FR, ML** — moderate; FR is irrigated in the southwest, which damps signal.
4. **BR, US** — huge panels but low spatial coherence (mean_r 0.05/0.14); the
   response is identified within-unit over time, not from national shock years.

Deprioritise small admin-1 AOIs (MZ, LS, AO, TD — ≤26 units) as robustness
checks only, and anything with mean_r below ~0.1 (CN, EL, PT, IT, IN).

## Working style

- Verify against real data rather than asserting. Every fix in this repo's
  recent history was found by running it, not by reading it.
- Long runs go in tmux, resumable, with per-AOI logs. Both stages skip work
  already on disk, and a failure is logged and stepped over.
- Flag the arithmetic before a multi-day run. The target-year/initialisation-year
  offset and the CDS request count are the two things worth re-reading.
