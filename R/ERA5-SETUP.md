# ERA5 admin-2 climate samples — setup

Generates the same per-county daily climate parquets as `ML-sample-gen.R`, but
from observed ERA5 instead of the EC-Earth3 hindcasts. Same schema, same units,
same cropland-area weighting, so downstream code does not change.

Two files:

- `era5.R` — pulls hourly ERA5 and reduces it to daily gridded rasters
  (`tasmax`, `tasmin`, `hursmin`, `pr`).
- `era5-sample-gen.R` — applies the cropland mask and aggregates to admin-2
  units, writing one parquet per year.

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
