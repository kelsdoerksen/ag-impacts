# ERA5 admin-2 climate samples — setup

Generates the same per-county daily climate parquets as `ML-sample-gen.R`, but
from observed ERA5 instead of the EC-Earth3 hindcasts. Same schema, same units,
same cropland-area weighting, so downstream code does not change.

Files:

- `era5-aoi.R` — the AOI registry, the daily-grid cache and the argument
  helpers, shared by both pipelines below. Not run directly.
- `era5.R` — pulls hourly ERA5 and reduces it to daily gridded rasters
  (`tasmax`, `tasmin`, `hursmin`, `pr`).
- `era5-sample-gen.R` — applies the cropland mask and aggregates to admin-2
  units, writing one parquet per year. **Aggregates before any feature is
  computed** — see §10 for when that is the wrong order.
- `era5-pixel-store.R` — the same download, stored per pixel instead, with a
  pixel/admin-2 weight table. Feeds `python/era5_features.py`.

Both pipelines share `{out_dir}/daily_grids`, so a year fetched by one is free
for the other. Nothing needs re-downloading to switch between them.

**The EC-Earth3 SMB share is not needed.** The ERA5 path never touches
`/Volumes/CRL_storage`, so the lab PC does not need that mount.

---

## 1-2. R and Python

Two routes. On a **shared machine, use conda** — `install.packages()` writes to
a personal R library, which is shared by everyone using the same account, so
installing or upgrading there is visible to other people. A conda env is not.

### Route A — conda env (recommended on shared machines)

Self-contained: its own R, its own Python, its own GDAL. No `sudo`, no system
`-dev` packages, no effect on anyone else.

```sh
conda env create -f environment-r.yml
conda activate agimpacts-r
conda env config vars set RETICULATE_PYTHON=$CONDA_PREFIX/bin/python
conda env config vars set R_LIBS_USER=$CONDA_PREFIX/lib/R/library
conda deactivate && conda activate agimpacts-r     # apply the variables
```

`R_LIBS_USER` matters on a **shared account**. R appends a per-user library
(`~/R/x86_64-*-library/4.x`) to its search path, and on a shared login that
directory belongs to everyone: an `install.packages()` there upgrades packages
for every user of the account, and an interrupted one can leave a package
half-replaced for them too. Pinning `R_LIBS_USER` into the env keeps both
reading and writing inside the environment.

Confirm the isolation held — every path printed should be inside the env:

```sh
Rscript -e 'print(.libPaths())'
```

`era5.R` reads `RETICULATE_PYTHON`, so nothing in the code needs changing.
Setting it as an env config var means it travels with the environment and you
cannot forget it. With the env active, `Rscript` is the env's R and its library
path is inside the env, so the system R and its user library are bypassed
entirely.

### Route B — system R (fine on a machine you own)

```r
install.packages(c("reticulate", "terra", "sf", "dplyr", "tidyr", "arrow"))
```

```r
library(reticulate)
virtualenv_create("r-reticulate", python = "/usr/bin/python3")
virtualenv_install("r-reticulate",
  c("numpy", "xarray", "gcsfs", "zarr", "netCDF4", "dask", "cdsapi"))
```

On Linux, `terra`/`sf` build from source and need GDAL/GEOS/PROJ headers
(`sudo apt install libgdal-dev libgeos-dev libproj-dev libudunits2-dev`).
Run `conda deactivate` first if conda is active — system R picking up conda's
libraries while building against system headers is a common source of link
errors. Pass `python =` explicitly for the same reason.

### Selecting the Python

`era5.R` resolves it in this order, all with `required = TRUE` so a missing
environment is an error rather than a silent fall back to a Python without
xarray:

| Setting | Effect |
|---|---|
| `RETICULATE_PYTHON=/path/to/bin/python` | use that interpreter |
| `ERA5_CONDA_ENV=agimpacts-r` | use that conda env by name |
| *(neither)* | the `r-reticulate` virtualenv |

### Checking what a machine already has

```sh
Rscript -e 'cat(R.version.string, "\n"); cat("libPaths:", .libPaths()[1], "\n");
for (p in c("reticulate","terra","sf","dplyr","tidyr","arrow"))
  cat(sprintf("  %-11s %s\n", p, requireNamespace(p, quietly = TRUE)))'
which R; which python3; python3 --version
gdal-config --version 2>/dev/null || echo "(no gdal-config)"
```

An interrupted `install.packages()` can leave `00LOCK-<pkg>` directories that
block later installs, and occasionally a half-replaced package. A
`00LOCK-<pkg>` holds R's backup of the *previous* version, so check what is
broken before deleting anything.

This reports and changes nothing. If a package does need restoring it prints
the exact command, with the real package name already filled in:

```sh
Rscript -e '
lib <- .libPaths()[1]
cat("library:", lib, "\n")
locks <- list.files(lib, pattern = "^00LOCK")
cat("locks:", if (length(locks)) paste(locks, collapse=", ") else "none", "\n")
broken <- character(0)
for (p in c("reticulate","terra","sf","dplyr","tidyr","arrow","Rcpp","vctrs","rlang","cli")) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("  %-12s %s\n", p, ok)); if (!ok) broken <- c(broken, p)
}
if (!length(broken)) {
  cat("Nothing broken.\n")
} else for (p in broken) {
  src <- file.path(lib, paste0("00LOCK-", p), p)
  if (dir.exists(src))
    cat(sprintf("restore: file.copy(\"%s\", \"%s\", recursive=TRUE)\n", src, lib))
  else
    cat(sprintf("reinstall: install.packages(\"%s\")\n", p))
}'
```

Once everything loads, clear the locks:

```sh
Rscript -e 'unlink(list.files(.libPaths()[1], pattern = "^00LOCK", full.names = TRUE),
                   recursive = TRUE)'
```

## 3. CDS credentials

ERA5 comes from the Copernicus Climate Data Store, which subsets server-side.

1. Free account at <https://cds.climate.copernicus.eu>.
2. Copy the personal access token from your profile page into `~/.cdsapirc`
   (on Windows: `C:\Users\<you>\.cdsapirc`):

   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-token>
   ```

   The token is per-account, not per-machine — the same one works everywhere,
   but the file itself does not travel with the repo, so each machine needs its
   own copy. `chmod 600 ~/.cdsapirc` on a shared server.

3. **Accept the licence**, or every request fails with a 403:
   <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences>

Check it works before starting a long run:

```r
reticulate::use_virtualenv("r-reticulate", required = TRUE)
reticulate::import("cdsapi")$Client()$retrieve(
  "reanalysis-era5-single-levels",
  list(product_type = "reanalysis", variable = list("2m_temperature"),
       year = "2000", month = "06", day = list("01"), time = list("12:00"),
       area = c(12.9, -2.1, 11.9, -1.1),
       data_format = "netcdf", download_format = "unarchived"),
  "probe.nc")
```

A licence problem shows up as `403 ... required licences not accepted`; a token
problem shows up as `401`.

## 4. Data files and paths

Two independent knobs, so inputs and outputs can live in different places:

| Variable | Controls | Default |
|---|---|---|
| `AGIMPACTS_ROOT` | where the **inputs** are read from | `/Users/kdoerkse/Documents/UCT/AgImpacts` |
| `ERA5_OUT_DIR` | where the **output** is written | `{AGIMPACTS_ROOT}/CONUS_processed_data/ERA5` |

`AGIMPACTS_ROOT` alone moves the whole tree. Set `ERA5_OUT_DIR` as well when the
inputs sit on a shared/read-only volume and the output belongs somewhere with
room. `--out=` on the command line overrides both.

The ERA5 path needs exactly two inputs under `AGIMPACTS_ROOT`:

| Path | Used for |
|---|---|
| `CroplandPastureArea2000_Geotiff/cropland2000_area.tif` | cropland-area weights |
| `us_county_2020_shapefile/cb_2020_us_county_20m.shp` | CONUS county boundaries |

Copy those two from the Mac (or re-download them) — nothing else is required.

## 5. Running

### Lab PC, start to finish

```sh
git clone <repo> && cd ag-impacts          # or git pull

# 1. point at the data (inputs and output separately)
export AGIMPACTS_ROOT=/data/AgImpacts
export ERA5_OUT_DIR=/data/era5_out

# 2. verify everything -- seconds, includes one live fetch
Rscript R/era5-sample-gen.R --check

# 3. run it detached, overnight
nohup Rscript R/era5-sample-gen.R 1981-2025 > era5_conus.log 2>&1 &

# 4. watch
tail -f era5_conus.log
```

`--check` verifies, in this order:

1. both input files exist (naming the missing one, and the `AGIMPACTS_ROOT` it
   came from)
2. the output directory is writable — by writing to it, not just by looking, so
   a read-only mount is caught
3. `~/.cdsapirc` has both a `url:` and a `key:` line
4. **one live request** to CDS — which is the only thing that can prove the
   token is valid, the licence is accepted, and the machine can reach the
   service. A well-formed credentials file proves none of those.

Steps 1–3 also run automatically at the start of every real run, so a
misconfigured job dies in seconds rather than partway through the night.

### All the forms

```sh
Rscript R/era5-sample-gen.R 2000                       # one year
Rscript R/era5-sample-gen.R 2001 2002 2005             # specific years
Rscript R/era5-sample-gen.R 1981-2025                  # a range
Rscript R/era5-sample-gen.R --out=/data/era5 1981-2025 # explicit output dir
Rscript R/era5-sample-gen.R --source=arco 2000         # no-account fallback
Rscript R/era5-sample-gen.R --check                    # verify setup, fetch nothing
```

Years already written are skipped, so re-running after an interruption picks up
where it stopped. On Windows use `setx AGIMPACTS_ROOT C:\path\to\AgImpacts`
(then reopen the shell); `nohup ... &` becomes `Start-Process` or just a
detached terminal.

## 6. Caches and disk

The deliverable is the parquet, same as your other climate data — about
**36 MB per year**, matching the EC-Earth3 series. Two caches sit behind it:

| Cache | Contents | Size | Kept? |
|---|---|---|---|
| `.../ERA5/cds_nc/` | raw hourly months from CDS | ~1.5 GB/year | **deleted per year** |
| `.../ERA5/daily_grids/` | daily rasters, GeoTIFF per variable per year | ~80 MB/year | kept |

The hourly months exist only so an interrupted year resumes at the month it
stopped on. Once a year is reduced to daily grids they are deleted, so `cds_nc`
holds at most the year in flight instead of growing with the series. Set
`era5_keep_hourly <- TRUE` in `era5-sample-gen.R` to keep them — worth it only
if you plan to re-derive some sub-daily statistic the daily rasters don't carry.

The daily GeoTIFFs are written with `COMPRESS=DEFLATE, PREDICTOR=3`, which is
lossless (verified bit-identical against an uncompressed write) and takes about
a third off.

**Total for a 45-year run: roughly 7 GB** — ~3.5 GB of daily grids, ~1.6 GB of
parquets, plus ~1.5 GB transient for the year being fetched.

> Storing the hourly data as parquet instead was measured and is *worse*:
> 2.4 GB/year against 1.5 GB/year as netCDF. Long-format parquet repeats
> time/lat/lon on every row and loses the array locality netCDF compresses well.
> Parquet is the right format for the county-aggregated output, which is what it
> is already used for; it is the wrong one for dense gridded data.

## 7. Two sources

`era5_daily_rasters(aoi, year, source = ...)`:

- **`"cds"`** (default) — subsets server-side, so a CONUS month is ~125 MB.
  Requests are queued, so wall time is unpredictable: roughly 40–60 min per year
  submitting months serially, mostly spent waiting rather than transferring.
- **`"arco"`** — Google's ARCO-ERA5 mirror. No account, no queue, but its chunks
  are one *global* hourly field each, so cropping to an AOI saves no bytes at
  all: ~50 GB and ~3.5 h per CONUS year, measured.

Both produce byte-identical grids (verified: same extent, same cells, same
values to 3 decimals), so they can be mixed across years freely.

## 8. Gotchas already handled

Listed because they are non-obvious if something breaks later:

- **`CERTIFICATE_VERIFY_FAILED`** — python.org framework builds ship no CA
  bundle. `era5.R` points OpenSSL at `certifi` *before* importing `gcsfs`;
  aiohttp builds its SSL context at import time, so setting it afterwards is
  ignored.
- **CDS returns a zip, not a netCDF** — a request mixing instantaneous
  (`2m_temperature`, `2m_dewpoint_temperature`) with accumulated
  (`total_precipitation`) variables always comes back as two files,
  `stepType-instant` and `stepType-accum`, regardless of `download_format`.
  `cds_month_nc()` sniffs the magic bytes and unpacks; `open_cds_month()`
  merges them back on their shared hourly axis.
- **`valid_time` vs `time`** — the current CDS backend renamed the time
  coordinate. Both are accepted.
- **Short vs long variable names** — CDS netCDF uses `t2m`/`d2m`/`tp`; the rest
  of the code uses the ARCO long names. Mapped in `open_cds_month()`.

## 9. Output

`climate_CONUS_ERA5_fy0_s{year}-obs.parquet`, one row per county-day:

```
adm2_id, adm2, adm1, hindcast_start, forecast_year, target_year,
date, doy, year_month, tasmax_C, tasmin_C, hursmin, pr_mm
```

ERA5 is observed, so it has no initialization year or lead time:
`hindcast_start = target_year = year` and `forecast_year = 0`. On the Python
side, `climate_dir()` files FY0 alongside the model rather than in an `FY0/`
folder, and `--forecast_year` accepts `0`:

```sh
python python/run_loo_experiment.py --climate_model ERA5 --ens_member obs \
    --forecast_year 0 --aoi CONUS ...
```

---

## 9b. How many CDS requests a year costs

CDS charges **queue time per request, not per byte**. Measured on a real
Burkina Faso run:

```
accepted → running     18-20 min     queue wait
running  → successful      2 min     actual work
```

So a request for one month and a request for twelve cost about the same. The
downloader therefore asks for as many months at once as memory allows, sized
automatically from the AOI's cell count (`.cds_auto_span()` in `era5.R`):

| AOI | cells | months/request | requests/year |
|---|---|---|---|
| CONUS, BR, AU | 23k–32k | 1 | 12 (unchanged) |
| IN | 12k | 2 | 6 |
| ET | 2.8k | 10 | 2 |
| ZM, BF, ML, SN, MG | 0.2k–2.2k | 12 | **1** |

For a small country that is a 12× reduction in queue waits — a year drops from
roughly four hours to under one. Override with `--months-per-request=N` if you
need to (N=1 restores the old behaviour).

**Resuming is unaffected.** Months already cached individually — from a run that
predates batching, or one interrupted partway — are reused as-is, and only the
gaps are re-requested. A year with months 1–5 on disk becomes five cache hits
plus one request for 6–12.

**ARCO ignores all of this.** It has no queue but transfers a global field per
hour regardless of AOI, so its cost is purely per byte and batching buys
nothing; it stays monthly. That also makes ARCO the better source for large
AOIs and for anything run while CDS is throttling you — it has no per-account
fair-share penalty at all.

## 10. Per-pixel features (the other pipeline)

### Why

`era5-sample-gen.R` takes the cropland-weighted admin-2 mean of the raw daily
variables and writes that. Any feature computed from that parquet is therefore a
feature *of the admin-2 mean*, which is only correct for features linear in the
inputs. The ones that matter are not:

```
mean_i( max(0, T_i − Tbase) )  ≠  max(0, mean_i(T_i) − Tbase)
```

Averaging first flattens the tails, so a unit holding both a cool pixel and a
hot one loses the heat the hot pixel actually accumulated. Degree days, extreme
degree days, dry-day counts, VPD and every threshold feature are affected. On a
Zambian admin-2 unit over one maize season, computing EDD per pixel first gave a
season total **73% higher** than computing it from the unit mean; GDD moved 1.6%
the other way. The causal crop-yield literature computes the feature per pixel
and only then takes the cropland-area-weighted mean, which is what this pipeline
does.

### The three stages

```sh
# 1. download + store per pixel  (the expensive, network-bound part)
Rscript R/era5-pixel-store.R --aoi=ZM 1980-2017

# 2. per-pixel daily features    (cheap, re-runnable, no network)
python python/era5_features.py --aoi=ZM --years=1980-2017 --stage=features

# 3. cropland-weighted admin-2 aggregate
python python/era5_features.py --aoi=ZM --years=1980-2017 --stage=aggregate \
    --reduce=season --crop=maize
```

Stage 1 is deliberately feature-free, so it can run now and be reused unchanged
however the feature set evolves. Stages 2 and 3 take seconds to minutes and can
be re-run as often as the feature list changes.

### Layout under `{out_dir}`

```
daily_grids/     GeoTIFF download cache, shared with era5-sample-gen.R
cds_nc/          raw hourly months, discarded per year
pixel_daily/     era5_pixel_{AOI}_{year}.nc    per-pixel daily cube
grid/            weights_{AOI}.parquet         pixel × admin-2 × weight
                 cells_{AOI}.parquet           the pixel set, with lon/lat
                 grid_{AOI}.json               grid geometry + provenance
pixel_features/  features_{AOI}_{year}.nc      per-pixel DAILY features
features_admin2/ features_{AOI}_*.parquet      the final admin-2 table
```

The cube is NetCDF, not parquet: the data is dense and gridded, so a long table
would repeat a cell id and a date on ~8.7M rows per CONUS year to hold the same
numbers, and NetCDF is what xarray opens natively. This is not the netCDF that
was rejected as too large — that was the raw **hourly** CDS download (~1.2 GB per
year). A daily, four-variable, float32, deflate-compressed cube is ~50 MB per
CONUS year and ~10 MB per Zambia year. The weight table stays parquet: it is
tabular, has no time dimension, and is a join key on the pandas side.

### The weight table

One row per (admin unit, pixel it overlaps), with the two weighting terms kept
apart so the feature stage can choose:

| column | meaning |
|---|---|
| `coverage_frac` | share of the pixel inside the admin polygon (terra exact intersection) |
| `cropland_frac` | harvested-area fraction of that pixel |
| `combined_weight` | the product — the cropland-area weight, and what `--weight=cropland` uses |

Built once per AOI, not once per year. `row`/`col` are terra's 1-based grid
indices and index the cube directly (row 1 = northernmost, matching its
descending latitude axis); `lon`/`lat` are there to verify that. The grid
geometry is recorded in `grid_{AOI}.json` and checked against every year, because
the join is by grid position — a shifted grid would be silently wrong rather
than loudly broken.

### Features

`python python/era5_features.py --list-features`

| feature | reduce | unit | params |
|---|---|---|---|
| `gdd` | sum | °C-day | `tbase=10`, `tcap=30` |
| `edd` | sum | °C-day | `tcrit=30` |
| `tavg` | mean | °C | |
| `vpd` | mean | kPa | |
| `pr` | sum | mm | |
| `dry_days` | sum | days | `threshold=1.0` |
| `hot_days` | sum | days | `threshold=35.0` |

`gdd` and `edd` use the single-sine method — integrating `max(T − threshold, 0)`
over a sine diurnal cycle between `tasmin` and `tasmax` — which is the standard
construction in this literature and differs materially from
`max(mean(tmin,tmax) − threshold, 0)` whenever the threshold falls inside the
day's range. Override a parameter with `--param gdd.tbase=8`; the values used
are recorded in the output cube's attributes.

Adding a feature is adding one decorated function in `era5_features.py`. The
`reduce` field says how stage 3 collapses the daily series over a season — `sum`
for anything accumulated, `mean` for a state variable.

### Season windows

`--reduce=season` reads the cy-bench crop calendar
(`crop_calendar_{crop}_{CC}.csv`, columns `sos`/`eos` as day-of-year) and gives
each admin unit its own window. Zambian maize has `sos` 327 and `eos` 135, so the
season for harvest year Y starts in Y−1 — handled explicitly, and it means
stage 2 must have been run for Y−1 as well. `--season=SOS:EOS` applies one fixed
window to every unit instead.

`--reduce=daily` (the default) skips the temporal reduction and writes one row
per unit per day, the same shape as the existing climate parquets — the values
are just features computed per pixel first.

### Running stage 1 on the lab PC

`era5-pixel-store.R` writes the GeoTIFF daily grids *and* the NetCDF cubes, so it
is the only thing to run there. Point `ERA5_OUT_DIR` at the directory that
already holds `daily_grids/` for that AOI and every year already downloaded is
free — it is read from cache, not re-fetched.

```sh
cd ~/kdoerksen/code/ag-impacts && git pull      # era5-aoi.R is new; needed to run
conda activate agimpacts-r

export AGIMPACTS_ROOT=/home/crl/kdoerksen/data
export ERA5_OUT_DIR=/home/crl/kdoerksen/data/era5_zm    # per AOI; era5/ for CONUS

# 1. prove the setup (seconds; includes one live CDS fetch and a boundary load)
Rscript R/era5-pixel-store.R --aoi=ZM --check

# 2. build the weight table from a year already on disk (no download)
Rscript R/era5-pixel-store.R --aoi=ZM --grid-only 1980

# 3. the real run, detached
tmux new -s zm_pixel
Rscript R/era5-pixel-store.R --aoi=ZM 1980-2017 2>&1 | tee era5_pixel_zm.log
#   Ctrl-b d to detach; tmux attach -t zm_pixel to come back
```

Years already written are skipped, so an interrupted run resumes where it
stopped. Running `era5-sample-gen.R` afterwards costs no extra download — it
reads the same `daily_grids/` cache.

### Environment

Stages 2 and 3 need `xarray`, `netCDF4`, `pandas`, `numpy`, `pyarrow`. The `ag`
conda env has all five. `dask` is deliberately not required.
