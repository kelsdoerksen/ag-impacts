# DCPP forecast arms: lead times, slice stitching, and what is on disk

Written 2026-11-04. The reference for "which DCPP files does a given look-ahead
time need, and which do I already have". Numbers here were measured against the
cy-bench maize calendars (13 AOIs, harvest 2002–2022, US restricted to CONUS)
and the DCPP tree on `crl_storage`, not assumed.

## The constraint everything follows from

EC-Earth3 `dcppA-hindcast` initialises **once a year, on 1 November**. A slice
`fy{F}_s{Y}` spans **1 Nov (Y+F−1) → 31 Oct (Y+F)**, so the published slices
tile the calendar into 12-month windows and slice `(s, F)` *is* the window
`W(k) = 1 Nov k … 31 Oct k+1` with `k = s + F − 1`. Two consequences:

1. The gap between initialisation and planting is fixed by each AOI's crop
   calendar, and the only dial is a **12-month step**.
2. `fy1_s{H−1}` and `fy2_s{H−2}` are **the same twelve months** seen from two
   different start dates. Combining them gives one window twice, not two pieces
   of a season. (This is the arithmetic that kills the obvious-looking
   "FY1 of 2002 + FY2 of 2001" pairing.)

Gap from the 1 Nov initialisation to **median sowing**, in months:

| AOI | FY1 | FY2 | FY3 | | AOI | FY1 | FY2 | FY3 |
|---|---|---|---|---|---|---|---|---|
| AR | −1.0 | 11.0 | 23.0 | | ML | 7.2 | 19.2 | 31.2 |
| BF | 7.2 | 19.2 | 31.2 | | MW | 0.7 | 12.7 | 24.7 |
| BR | 1.9 | 13.9 | 25.9 | | RO | 5.9 | 17.9 | 29.9 |
| ET | −3.7 | 8.3 | 20.4 | | SN | 7.9 | 19.9 | 31.9 |
| FR | 5.4 | 17.4 | 29.4 | | US | 5.7 | 17.7 | 29.7 |
| HU | 5.7 | 17.7 | 29.7 | | ZM | 0.6 | 12.6 | 24.6 |
| MG | 0.8 | 12.8 | 24.9 | | | | | |

**FY1 is not a one-year-ahead forecast.** Across AOIs it ranges from 0.6 to 7.9
months before planting — a factor of thirteen — so a cross-AOI skill comparison
within the FY1 arm is partly a comparison of lead times. Report the horizon
column beside every FY1 skill number.

**AR and ET are initialised after their crop is sown** (−30 d and −109 d from
median planting). Under the original one-initialisation stitching that makes FY1
unbuildable for them: 7664 of 9639 AR and 1050 of 1764 ET district-years are
uncoverable. Brazil loses 72 of 4704 municipalities, all in Rio Grande do Sul
(sowing 15 Aug – 31 Oct, genuine, not a calendar artifact) — that is 14.5% of
RS and 0% of every other state, and the survivors are biased toward late
planting in the one state where planting date varies most.

A 5-month-before-harvest floor, checked per unit rather than per median, is
slack everywhere it matters: MG (5.7 min), MW (6.1), ZM (5.9) and US (10.9) all
clear it at FY1. It bites only on AR (2.0 min) and ET (2.0), which coverage
already excludes.

## The four arms

| arm | slices | what it answers |
|---|---|---|
| **A** — FY1 | `fy1_s{H−1}` + `fy2_s{H−1}` tail | closest to planting; 0.6–7.9 mo, bimodal |
| **B** — FY2 | `fy1/fy2/fy3_s{H−2}` | ~1 year before planting; 8.3–19.9 mo, zero losses |
| **C** — FY3 | leads 2,3,4 of `s{H−3}` | ~2 years before planting; 20.4–31.9 mo |
| **D** — FY1 maximised | `fy1_s{H−2}` + `fy1_s{H−1}` + `fy2_s{H−1}` | arm A's decision date with the whole season at lead 1 |

Arm D is the standing preference (see below). It has the **same decision date as
arm A** — nothing is read from a run started after 1 Nov (H−1) — and it is
lossless where arm A is not: AR and ET become buildable, and Brazil keeps its 72
RS municipalities. It moves 88% of Argentina's season days and 51% of Ethiopia's
from lead 2 to lead 1 and drops lead 3 entirely.

Arm B's slice composition, CONUS-corrected — only **AR, BR and ET** need the
lead-1 slice; HU, MG, MW and ZM need nothing but lead 2:

```
AOI      n | lead-1 units  lead-2  lead-3 | mo pre-plant (min)
AR     459 |          364     459      13 |  8.5
BF      45 |            0      45      24 | 18.4
BR    4704 |           72    4704       0 |  9.4
ET      84 |           50      84      34 |  6.8
FR      97 |            0      97       2 | 12.8
HU      20 |            0      20       0 | 17.6
MG      22 |            0      22       0 | 12.5
ML      27 |            0      27      27 | 18.5
MW      27 |            0      27       0 | 12.5
RO      42 |            0      42       5 | 17.7
SN      45 |            0      45      45 | 19.2
US    3016 |            0    3016     766 | 16.5
ZM      74 |            0      74       0 | 12.1
```

Ten of thirteen clear 12 months before planting on every unit. AR (11.0 median),
ET (8.3) and BR's RS tail (9.4) cannot, because their calendars sit between two
1 November initialisations. Do not promote them to FY3 to fix it — that buys
strict compliance at the cost of lead-4 downloads and a mixed-lead design in
which an AR skill number is no longer comparable to a Hungarian one.

## Standing preference: maximise FY1

**Use lead-1 slices for as much of every season as the decision date allows.**
Lead 1 is the best DCPP data there is; lead degradation is the thing being
measured, not something to absorb silently into the feature set. So:

- At the FY1 horizon, run **arm D** (`--slice-policy newest-run`), not arm A.
  Same issue date, strictly more of the season at lead 1, no NaN'd districts.
- At longer horizons, keep the arm defined by its lead (FY2, FY3) so cross-AOI
  skill stays comparable, and accept that the main season sits at lead F there.
  `newest-run` is a no-op from F = 2: the pre-season window already belongs to
  `s{H−F}` under both policies.
- Never stitch across initialisations *inside* the main season. The splice is
  allowed only at a 12-month window boundary, upstream of sowing, which is where
  `newest-run` makes it.

## `--slice-policy`

`fixed-init` (default) pins the run at `s = H − F` and varies the lead.
`newest-run` takes each 12-month window from the newest run that is still no
newer than `s = H − F`, so the pre-season window comes from lead 1 of
`s = H − F − 1` instead of being uncoverable.

- Implemented in `era5_features.season_slices()` / `season_source_years()` /
  `harvest_year_candidates()`, threaded through `compare_dcpp_era5.py`,
  `yield_features.py`, `run_yield_experiment.py` and `run-dcpp.sh`.
- **Stage 1 and 2 are policy-blind.** A slice is just (initialisation, lead) and
  both policies read the same files, so no cube or per-pixel feature file needs
  rebuilding to switch policy.
- **Stage 3 output is tagged.** `fy1_` for fixed-init, `fy1n_` for newest-run, in
  the admin-2 parquets, the comparison tables and the experiment `run_id`. The
  two never collide, and `calendar_fit()` stops refusing AR and ET at FY1 under
  `newest-run`.
- Verified: `fixed-init` reproduces the previous `season_slices()` exactly on
  5040 (harvest year × lead × window) cases, so arms A, B and C are unaffected.
- `report_dcpp_era5.py` pivots on the lead, so it refuses a comparisons
  directory holding both policies until you pass `--slice-policy`; otherwise it
  would average arm A with arm D and label the result FY1. The default HTML
  filename carries the policy when it is not `fixed-init`.

`compare_dcpp_era5.py --stage both` builds stage 2 only for the lead it was
given, while `newest-run` also reads lead F+1 of `s{H-F}`; on this share leads
1–3 already exist for every AOI, so that is a no-op, and `run-dcpp.sh` is what
fills a gap if one appears.

```sh
# arm D, stage 3 + feature comparison (AR and ET now buildable at FY1)
python python/compare_dcpp_era5.py --aois AR,BF,BR,ET,FR,HU,MG,ML,RO,SN,US,ZM \
  --forecast-years 1 --slice-policy newest-run --harvest-years 2002-2022 --go

# arm D, the yield models
python python/run_yield_experiment.py --aois US,HU,RO,ZM,BR,FR,AR,ET \
  --source dcpp:ens --forecast-year 1 --slice-policy newest-run \
  --years 2002-2022 --loo --go
```

## What is on disk (verified 2026-11-04)

`{share}/dcpp/{AOI}_processed_data/EC-Earth3_{member}`, 10 members, 12 AOIs
(**MW has no directory at all** — it would be a from-scratch build; 27
districts, lead 1 only, cheap).

```
            L1          L2          L3          L4
cubes    10/10 full  10/10 full  10/10 full     --
features 10/10 full  10/10 full  10/10 full     --
```

Leads 1–3 cover **s2000–s2022 (23 initialisations)**; lead 4 is empty. Reliable
for AR, BF, BR, ET, FR, HU, MG, ML, RO, SN. **US and ZM were scanned while the
SMB mount was failing and are unconfirmed**, as is ET at lead 3 across all ten
members (an earlier FY3 run died at s2010; `r1i4p1f1` shows 23 files, so it was
probably finished later).

| arm | needs | status |
|---|---|---|
| A (FY1) | L1 s2001-2021, L2 s2000-2021 | already built |
| B (FY2) | L1,2,3 s2000-2020 | already built |
| D (FY1 maximised) | L1 s2000-2021, L2 s2001-2021 | already built |
| C (FY3) | L2,3,4 s1999-2019 | **lead 4 missing everywhere** |

Arm C over harvest **2003**–2022 instead of 2002–2022 needs only **lead 4 over
s2000–2019** — one harvest year to avoid building `s1999` across every AOI and
lead. Worth taking. Remaining stage-1/2 work, 80 jobs / 1600 member-years:

```sh
tmux new -s dcpp
cd ~/kdoerksen/code/ag-impacts && git pull
conda activate agimpacts-r        # NOT base: system R fails on reticulate
./run-dcpp.sh --aois AR,BF,ET,FR,ML,RO,SN,US --forecast-years 4 --init-years 2000-2019
./run-dcpp.sh --aois AR,BF,ET,FR,ML,RO,SN,US --forecast-years 4 --init-years 2000-2019 --go
```

BR, HU, MG and ZM are absent because their seasons never reach lead 4.

## Open items

- Confirm the US and ZM member inventories, and ET at lead 3, when the mount is
  stable. A lead complete in one member but short in another still fails: the
  ensemble mean is taken only over unit-years every member has.
- A harvest-relative 7–9 month arm **cannot be built from DCPP**. Taking each
  AOI's `(gap mod 12)`, only BR lands in [7, 9] at any lead. That horizon is
  seasonal-forecast territory (SEAS5, C3S multi-system, monthly initialisation).
- cy-bench gives Rio Grande do Sul a 245-day median maize season (12 Oct –
  15 Jun), which is long enough to suspect `eos` is tracking senescence or a
  second cycle. It inflates the aggregation window for all 497 RS units.
