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

### ML experiments (`yield_features.py` + `run_yield_experiment.py`)

- **Folds are whole years, never rows.** Districts share a year's weather, so
  their residuals are correlated (r ~ 0.28 for ZM maize). A random row split
  leaves ~70 near-duplicates of each test row in training, and the score it
  reports has nothing to do with forecasting. Every CV inside a model
  (`GroupedRidgeCV`, `GroupedMLP`, `logreg`) holds out years too.
- **The feature source is the only thing an ERA5 run and a DCPP run may
  differ in.** `--source era5 | dcpp:<member> | dcpp:ens`, same keys, same
  covariates, same folds. Otherwise the difference in score measures plumbing.
- **The DCPP ensemble mean is taken AFTER stage 3**, feature by feature, and
  only over unit-years every member has. Averaging daily fields first would
  flatten the tails a threshold feature exists to measure; a varying
  membership would put a step in the series that reads as interannual signal.
- **Lagged inputs and the direction reference must be at least the lead
  time.** A lead-F forecast is issued before harvest year H-F+1 exists, so at
  FY2 the most recent observable yield is H-2. `yield_lag{k}` starts at
  `max(1, F)` and `--target-lags` is refused below F.
- **`sd_ratio_anom < 1` is not a defect.** For a calibrated conditional-mean
  predictor under squared error, `sd(pred)/sd(obs) ~= r`. Measured on a
  synthetic panel: mlp 0.287 vs r_anom 0.297 (calibrated), rf-anomaly 0.442 vs
  0.266 and rf-absolute 1.682 vs 0.901 (both over-dispersed). An MLP's weight
  decay does not change this -- pinning alpha over six decades moved
  sd_ratio_anom only 0.281-0.289. Wanting full amplitude means a different
  loss, or the `direction` target, which discards amplitude.
- **The direction band is `tau * sigma(u,k)`, fit on training rows only.**
  Per unit, because a fixed kg/ha band means different things in a district
  that swings 40% a year and one that swings 5%. Per lag, because the spread
  of a k-year difference grows ~sqrt(k). Training-only, or the held-out year
  sets its own class boundary. `tau = 0.5` gives roughly balanced thirds.
- **Never quote direction accuracy alone.** Widening the band to 2 sigma took
  accuracy from 0.318 to 0.934 purely by growing the flat class to 93% of
  rows; the base-rate null rose with it and `hss` stayed at 0. Report
  `acc_major`, `bacc` and `hss` beside it.

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
- **`data_loader.find_season_feature_files` cannot see a DCPP store.** It globs
  `features_{aoi}_{tag}_{kind}_*` while stage 3 writes
  `features_{aoi}_{tag}_fy{F}_{kind}_{year}`, so it silently finds nothing at
  any lead. Observed (FY0) stores still match. `yield_features.py` globs the
  lead token; the older runners (`run_loo_experiment.py`, the TST arms) are
  still FY0-only.
- **FR and US shapefiles span most of a hemisphere.** FR ships `FRY10/20/30/40/50`
  (Guadeloupe, Martinique, Guyane, Réunion, Mayotte) and US ships FIPS 02/15/60/66/69
  plus the Aleutians across the antimeridian. Drawing a map to the shapefile's own
  extent puts metropolitan France at 12% of the frame and CONUS at 16%. Restrict
  to the units in the data first; `report_dcpp_era5.py` then trims what is left by
  MAD (k=6, the only value that touches nothing but FR and US).
- **ET's processed-data directory is `ET_preprocessed_data`**, not
  `ET_processed_data` (`R/crop-data-gen-cybench.R`). `Roots.aoi_targets`
  accepts both rather than renaming a directory other scripts point at.

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

## Onboarding a collaborator for ERA5 processing

AOIs are independent — each writes to its own `{AOI}_processed_data/ERA5`
directory — so two people can process different AOIs into the same tree with no
coordination beyond agreeing who takes which. Claim a block from the table
below and say so.

### 1. Get your own CDS account

**Do not share an account.** The CDS queue penalty is per-account and
compounds with recent usage: two people on one account slow each other down
badly, and large requests start being refused on cost. With separate accounts
you genuinely parallelise.

1. Register at <https://cds.climate.copernicus.eu> and verify the email.
2. Open the **reanalysis-era5-single-levels** dataset, go to its *Download*
   tab, scroll to the licence box and **accept the Copernicus licence**. This
   is the step people miss — the API returns 403 until it is accepted, and the
   error does not say why.
3. Copy your **Personal Access Token** from your profile page (a 36-character
   UUID).
4. Write `~/.cdsapirc`:

   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-36-char-token>
   ```

   `chmod 600 ~/.cdsapirc`. Note the current API takes the token alone — the
   older `UID:APIKEY` form is for the retired endpoint.

### 2. Set up on the lab PC

```sh
ssh crl@137.158.31.55
cd ~/kdoerksen/code/ag-impacts && git pull
conda activate agimpacts-r          # NOT the base env: see Environment above

# prove the setup before committing to a batch -- seconds, one live CDS fetch
python python/run_cybench_era5.py     --out /mnt/crl_storage/personal_folders/kdoerksen/era5     --root /mnt/crl_storage/personal_folders/kdoerksen     --aois CZ --from-year 2001 --python $CONDA_PREFIX/bin/python
```

That prints the plan without downloading. Then one AOI for real, in tmux:

```sh
tmux new -s era5_cz
python python/run_cybench_era5.py     --out /mnt/crl_storage/personal_folders/kdoerksen/era5     --root /mnt/crl_storage/personal_folders/kdoerksen     --aois CZ,PL,BG,SK,BE,AT --from-year 2001     --python $CONDA_PREFIX/bin/python --go
# Ctrl-b d to detach
```

Pass `--root` and `--out` on the command line, never via export — a `tmux new`
attached to an existing server does not inherit the parent shell's exports.

Explicit `--aois` runs in the order given, so put the ones you care about
first. Everything is resumable: both stages skip what is already on disk, and a
failed AOI is logged and stepped over rather than stopping the queue.

### 3. AOIs still to do, in priority order

`mean_r` is the mean pairwise correlation of detrended unit yield anomalies —
how spatially coherent the annual climate shock is. `CV` is how much
interannual variation exists to explain. Both measured from the cy-bench yield
tables; see Priorities above for the method.

| | units | yrs | obs | CV | mean_r | req/yr | notes |
|---|---|---|---|---|---|---|---|
| DE | 321 | 43 | 8,945 | 0.13 | 0.32 | 1 | biggest remaining panel |
| CZ | 14 | 16 | 216 | 0.18 | 0.81 | 1 | strongest coherence anywhere |
| PL | 17 | 18 | 301 | 0.14 | 0.62 | 1 | |
| IT | 101 | 26 | 2,318 | 0.17 | 0.11 | 2 | |
| BG | 6 | 30 | 180 | 0.27 | 0.64 | 1 | |
| SK | 8 | 12 | 94 | 0.23 | 0.76 | 1 | |
| BE | 10 | 10 | 81 | 0.18 | 0.79 | 1 | |
| AT | 9 | 46 | 414 | 0.12 | 0.45 | 1 | longest record here |
| LT | 10 | 21 | 181 | 0.35 | 0.47 | 1 | |
| LS | 10 | 41 | 393 | **0.50** | 0.46 | 1 | highest CV, cheapest AOI (168 cells) |
| MZ | 10 | 22 | 196 | 0.42 | 0.53 | 2 | |
| AO | 18 | 21 | 378 | 0.30 | 0.43 | 2 | |
| IN | 501 | 28 | 11,930 | 0.33 | 0.09 | 6 | monsoon; second-biggest panel |
| NE | 30 | 40 | 530 | 0.41 | 0.27 | 2 | |
| ES | 53 | 23 | 1,146 | 0.17 | 0.13 | 3 | |
| ZA | 9 | 36 | 316 | 0.20 | 0.33 | 4 | |
| EL | 52 | 11 | 559 | 0.20 | 0.03 | 1 | weak signal |
| NL | 10 | 13 | 119 | 0.17 | 0.15 | 1 | weak signal |
| TD | 17 | 31 | 425 | 0.38 | 0.05 | 1 | weak signal |
| PT | 7 | 35 | 243 | 0.14 | −0.02 | 3 | weak signal |
| CN | 31 | 33 | 998 | 0.10 | 0.03 | — | **`--source arco`**: 36,105 cells, CDS refuses |

Suggested coherent blocks: **Central Europe** (CZ, PL, BG, SK, BE, AT, DE, LT,
NL, EL) — ten AOIs, nearly all 1 request/year, no ARCO needed. **Africa**
(LS, MZ, AO, NE, TD, ZA). **Large/awkward** (IN, ES, IT, PT, CN) — CN must use
ARCO.

**Do not process**: `EE`, `FI`, `IE`, `LV` (100% zero yields in cy-bench);
`DK` (7 units, median 2 years each), `SE` (1 unit), `HR` (2 units), `MX` (4
distinct years) — too thin for a panel.

### 4. Expectations and failure modes

- **`--months-per-request`**: leave it off so each AOI sizes itself. If CDS
  starts refusing with `403 ... cost limits exceeded`, pin `--months-per-request 3`.
- **Queue time dominates.** 18–25 min of waiting per request for ~2 min of work,
  and it degrades with recent account usage. A 1-request/year AOI over 20 years
  is a few hours; the request count in the plan is the number that matters.
- **Large AOIs need ARCO** (`--source arco`). A month over more than ~20,000
  cells is refused by CDS on cost. ARCO has no queue but transfers global fields
  and has **no intra-year cache**, so do not kill it mid-year.
- If a run dies, `cat {out}/{AOI}/last_error.txt` has the reason even when the
  log looks truncated.

## Working style

- Verify against real data rather than asserting. Every fix in this repo's
  recent history was found by running it, not by reading it.
- Long runs go in tmux, resumable, with per-AOI logs. Both stages skip work
  already on disk, and a failure is logged and stepped over.
- Flag the arithmetic before a multi-day run. The target-year/initialisation-year
  offset and the CDS request count are the two things worth re-reading.
