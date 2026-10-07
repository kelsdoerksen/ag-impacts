# Query ERA5 climate data, subset it to an area of interest (AOI) and a set
# of variables, and save the result.
#
# Data source: Google's public ARCO-ERA5 archive on Google Cloud Storage.
# It's read anonymously, so no login or API key is needed.
#
# R doesn't have a good native way to read this kind of cloud data store, so
# this script uses reticulate to call the Python libraries that do (xarray +
# gcsfs). One-time setup, if you haven't already:
#   reticulate::virtualenv_create("r-reticulate", python = "3.11.9")
#   reticulate::virtualenv_install("r-reticulate",
#     c("numpy", "xarray", "gcsfs", "zarr", "netCDF4", "dask"))
# dask is what lets a chunked read fetch several store chunks at once; see
# era5_daily_rasters() for why that matters.

library(reticulate)
library(terra)

# Directory holding this file, so the AOI registry resolves regardless of the
# working directory (source() records the path on its call frame; Rscript only
# reports it in --file=).
.era5_dir <- local({
  f <- NULL
  for (i in seq_len(sys.nframe())) {
    ofile <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(ofile)) f <- ofile
  }
  if (is.null(f)) {
    a <- commandArgs(trailingOnly = FALSE)
    m <- sub("^--file=", "", a[grep("^--file=", a)])
    if (length(m)) f <- m[1]
  }
  if (is.null(f)) getwd() else dirname(normalizePath(f))
})

# Which Python reticulate should use. The default is the "r-reticulate"
# virtualenv, but on a shared machine it is usually better to put R and Python
# in one self-contained conda environment, so both are selectable without
# editing this file:
#   RETICULATE_PYTHON=/path/to/env/bin/python   any interpreter, conda or not
#   ERA5_CONDA_ENV=agimpacts-r                  a conda env by name
# required = TRUE throughout, so a missing environment is an error rather than a
# silent fall back to some other Python that lacks xarray/gcsfs.
local({
  py_path  <- Sys.getenv("RETICULATE_PYTHON")
  conda_nm <- Sys.getenv("ERA5_CONDA_ENV")
  if (nzchar(py_path)) {
    use_python(py_path, required = TRUE)
  } else if (nzchar(conda_nm)) {
    use_condaenv(conda_nm, required = TRUE)
  } else {
    use_virtualenv("r-reticulate", required = TRUE)
  }
})

# Avoids a hard crash on macOS when R and numpy each try to initialize their
# own OpenMP runtime.
Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

# A python.org framework build (which is what virtualenv_create() picks up on
# macOS unless told otherwise) ships no CA bundle, so every https request to
# Google Cloud Storage dies with CERTIFICATE_VERIFY_FAILED. certifi comes along
# with gcsfs; point OpenSSL at it. This has to run BEFORE gcsfs is imported:
# aiohttp builds its default SSL context at import time, so anything set
# afterwards is ignored. (Set through Python's os.environ rather than
# Sys.setenv() so it reaches the C-level getenv() OpenSSL actually reads.)
py_run_string(paste(
  "import os, certifi",
  "os.environ.setdefault('SSL_CERT_FILE', certifi.where())",
  "os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())",
  sep = "\n"))

xr    <- import("xarray")
gcsfs <- import("gcsfs")
np    <- import("numpy")
dask  <- import("dask")
# Only the CDS path needs this, and it is an extra pip install, so it is loaded
# on first use rather than at source time.
cdsapi <- import("cdsapi", delay_load = TRUE)
py    <- import_builtins()  # gives us Python's builtin slice(), used below

ERA5_PATH <- "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Bounding boxes for each area of interest. Add a new AOI by adding a row here.
# Longitude is in the usual -180..180 range; it gets converted to ERA5's
# 0..360 convention automatically.
AOI_BOUNDS <- list(
  conus        = list(lat_min = 24.5,  lat_max = 49.5,  lon_min = -125.0, lon_max = -66.5),
  burkina_faso = list(lat_min = 9.4,   lat_max = 15.1,  lon_min = -5.6,   lon_max = 2.5),
  south_africa = list(lat_min = -35.0, lat_max = -22.0, lon_min = 16.3,   lon_max = 33.0),
  # ~1x1 degree box around Ouagadougou, Burkina Faso — for quick smoke tests.
  test         = list(lat_min = 11.9,  lat_max = 12.9,  lon_min = -2.1,   lon_max = -1.1)
)

# The three variables the ML pipeline needs. CDS requests spell variables this
# way too, so the same vector drives both sources; only the netCDF that comes
# back from CDS uses the GRIB short names, which open_cds_month() maps away.
ERA5_ML_VARS <- c("2m_temperature", "2m_dewpoint_temperature", "total_precipitation")

CDS_DATASET <- "reanalysis-era5-single-levels"

CDS_SHORT_TO_LONG <- c(t2m = "2m_temperature",
                       d2m = "2m_dewpoint_temperature",
                       tp  = "total_precipitation")

# ERA5 stores these as running totals since the last timestep rather than
# instantaneous values, so they need to be summed (not averaged) when
# resampling to a coarser time step.
# Every cy-bench country, loaded from R/cybench-aois.json and keyed by its
# two-letter code (AO, BF, ... ZM). The registry is generated from the polygon
# shapefiles themselves by tools/build-cybench-aois.R, so the boxes cannot drift
# from the geometry they are meant to cover -- rebuild it rather than editing
# boxes by hand. Hand-written AOIs above win on a name clash.
local({
  reg <- file.path(.era5_dir, "cybench-aois.json")
  if (!file.exists(reg)) {
    warning("cy-bench AOI registry not found at ", reg,
            "; only the built-in AOIs are available", call. = FALSE)
    return(invisible(NULL))
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    warning("jsonlite not installed; cy-bench AOIs unavailable", call. = FALSE)
    return(invisible(NULL))
  }
  entries <- jsonlite::read_json(reg, simplifyVector = FALSE)
  for (cc in names(entries)) {
    if (!is.null(AOI_BOUNDS[[cc]])) next
    b <- entries[[cc]]$bbox
    AOI_BOUNDS[[cc]] <<- list(lat_min = b$lat_min, lat_max = b$lat_max,
                              lon_min = b$lon_min, lon_max = b$lon_max)
  }
})

ACCUMULATED_VARS <- c(
  "total_precipitation", "large_scale_precipitation", "convective_precipitation",
  "evaporation", "runoff", "surface_runoff", "sub_surface_runoff", "snowfall"
)

# Open the ERA5 dataset. This is lazy: no data is downloaded yet, only the
# metadata needed to know what's available.
# `chunks = NULL` reads eagerly, one store chunk at a time. Pass a named list
# (e.g. list(time = 24L)) to hand the read to dask instead, which fetches chunks
# concurrently -- worth roughly 3x on this store, whose chunks are one global
# field per hour.
open_era5 <- function(chunks = NULL) {
  gcs <- gcsfs$GCSFileSystem(token = "anon")
  xr$open_zarr(gcs$get_mapper(ERA5_PATH), chunks = chunks)
}

# Crop the dataset down to one AOI's bounding box.
crop_to_aoi <- function(ds, aoi_name) {
  box <- AOI_BOUNDS[[aoi_name]]

  # ERA5 stores longitude as 0..360, not -180..180, so convert the box.
  lon_min <- box$lon_min %% 360
  lon_max <- box$lon_max %% 360

  if (lon_min <= lon_max) {
    ds <- ds$sel(longitude = py$slice(lon_min, lon_max))
  } else {
    # The box crosses the 0/360 line (e.g. it spans the Prime Meridian), so
    # it has to be selected in two pieces and stitched back together.
    west_part <- ds$sel(longitude = py$slice(lon_min, 360))
    east_part <- ds$sel(longitude = py$slice(0, lon_max))
    ds <- xr$concat(list(west_part, east_part), dim = "longitude")
  }

  # ERA5 stores latitude from north to south (90 down to -90), so the slice
  # must go from the max down to the min, not min to max.
  ds$sel(latitude = py$slice(box$lat_max, box$lat_min))
}

# Keep only the requested variables. Needs a Python list, not an R vector,
# or reticulate tries to treat it as a single (invalid) key.
keep_variables <- function(ds, var_names) {
  ds[as.list(var_names)]
}

# Coarsen from ERA5's native hourly resolution to daily/monthly/yearly.
# Accumulated variables (see ACCUMULATED_VARS) are summed over the window;
# everything else (temperatures, pressures, ...) is averaged.
resample_time <- function(ds, resample, var_names) {
  if (resample == "hourly") return(ds)

  freq <- switch(resample,
    daily   = "1D",
    monthly = "1MS",
    yearly  = "1YS",
    stop("resample must be one of 'hourly', 'daily', 'monthly', 'yearly'")
  )

  sum_vars  <- intersect(var_names, ACCUMULATED_VARS)
  mean_vars <- setdiff(var_names, sum_vars)

  parts <- list()
  if (length(mean_vars)) parts <- c(parts, ds[as.list(mean_vars)]$resample(time = freq)$mean())
  if (length(sum_vars))  parts <- c(parts, ds[as.list(sum_vars)]$resample(time = freq)$sum())

  if (length(parts) == 1) parts[[1]] else xr$merge(parts)
}

# Convert one xarray DataArray (dims time, latitude, longitude, ERA5's native
# north-to-south lat order) into a daily-layered terra SpatRaster. Mirrors
# ML-sample-gen.R's arrays_to_rast(), which every downstream step (harvested-
# area masking, extract_daily_by_county) expects its climate rasters shaped as.
era5_da_to_rast <- function(da) {
  lons <- da$longitude$values
  lats <- da$latitude$values
  dates <- as.Date(da$time$values)
  vals <- da$values  # [time, latitude, longitude], numpy -> R array via reticulate

  # Shift ERA5's 0..360 longitude back to -180..180 for terra.
  lons <- ifelse(lons > 180, lons - 360, lons)
  dx <- abs(diff(lons)[1])
  dy <- abs(diff(lats)[1])
  nlyr <- length(dates)

  r <- rast(nrows = length(lats), ncols = length(lons), nlyrs = nlyr,
            xmin = min(lons) - dx / 2, xmax = max(lons) + dx / 2,
            ymin = min(lats) - dy / 2, ymax = max(lats) + dy / 2,
            crs = "EPSG:4326")

  # [time, lat, lon] -> [lon, lat, time] -> flat [ncell, nlyr], lon fastest,
  # rows north-to-south first: terra's convention, matching arrays_to_rast().
  vals <- aperm(vals, c(3, 2, 1))
  # Both ARCO and CDS hand back descending latitudes, but a source that did not
  # would come out vertically mirrored with no error anywhere, so it is checked
  # rather than assumed (arrays_to_rast() guards the same way).
  if (lats[1] < lats[length(lats)]) vals <- vals[, length(lats):1L, , drop = FALSE]
  dim(vals) <- c(length(lons) * length(lats), nlyr)
  values(r) <- vals

  time(r) <- dates
  r
}

# Relative humidity (%) from temperature and dewpoint (both Celsius). Same
# Magnus-Tetens approximation as Emmanuel's relative_humidity.py:
#   RH = 100 * (exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T)))
# ERA5 has no direct humidity variable, so this is derived rather than read.
# Unlike relative_humidity.py (which runs the formula on daily-mean T/Td to
# get one daily "hurs" value), this is applied hourly and the caller takes
# the daily min of the result — giving a true hursmin, not an RH derived
# from an already-averaged daily temperature.
era5_relative_humidity <- function(temp_c, dewpoint_c) {
  es <- np$exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
  e0 <- np$exp((17.625 * temp_c)     / (243.04 + temp_c))
  100 * es / e0
}

# ===== CDS (Copernicus Climate Data Store) =====
#
# The alternative to the ARCO archive, and the reason it exists here: CDS
# subsets server-side. `area` is applied before anything is transmitted, so a
# CONUS month costs tens of MB. The ARCO store cannot do that at any price --
# its chunks are one *global* hourly field each, so cropping to an AOI saves no
# bytes at all and a CONUS year moves ~50 GB. Over a multi-year series that is
# the difference between hours and days.
#
# What you give up is predictability: CDS requests are queued, so a month may
# come back in a minute or sit for an hour, and that is outside our control.
#
# Setup, once:
#   1. Make a free account at https://cds.climate.copernicus.eu
#   2. Copy the personal access token from your profile page into ~/.cdsapirc:
#        url: https://cds.climate.copernicus.eu/api
#        key: <your-token>
#   3. Open the reanalysis-era5-single-levels dataset's Download tab and accept
#      the licence. Until you do, every request fails on licence terms.
#   4. reticulate::virtualenv_install("r-reticulate", "cdsapi")

# Issue one CDS request and land it as a directory of netCDFs.
#
# Built under a temporary name and moved into place only once complete, so an
# interrupted transfer cannot leave a half-written directory that a cache check
# would trust as finished.
.cds_fetch <- function(request, dest, label) {
  tmp <- paste0(dest, ".part")
  unlink(tmp, recursive = TRUE)
  dir.create(tmp, recursive = TRUE)
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  target <- file.path(tmp, "download")
  # retry_max defaults to 500, which is right for a flaky connection and badly
  # wrong for a permanent rejection: a 403 "cost limits exceeded" was retried
  # for the better part of an hour before surfacing. A handful of retries still
  # rides out a network blip but lets a refusal through in minutes.
  cdsapi$Client(retry_max = 5L)$retrieve(CDS_DATASET, request, target)
  if (!file.exists(target)) stop("CDS retrieve returned no file for ", label)

  # Sniff the magic bytes rather than trusting the requested download_format.
  if (identical(readBin(target, "raw", 2L), as.raw(c(0x50, 0x4b)))) {  # "PK"
    unzip(target, exdir = tmp)
    unlink(target)
  } else {
    file.rename(target, file.path(tmp, "data.nc"))
  }

  if (!length(list.files(tmp, pattern = "\\.nc$"))) {
    stop("CDS returned no netCDF for ", label)
  }
  if (!dir.exists(dirname(dest))) dir.create(dirname(dest), recursive = TRUE)
  file.rename(tmp, dest)
  dest
}

.cds_area <- function(aoi) {
  box <- AOI_BOUNDS[[aoi]]
  # CDS wants the box as North, West, South, East, in -180..180 -- which is how
  # AOI_BOUNDS already stores it, so no 0..360 conversion here.
  c(box$lat_max, box$lon_min, box$lat_min, box$lon_max)
}

# Local solar time offset for an AOI, in whole hours: longitude / 15.
#
# Solar, not civil: the diurnal cycle that puts the minimum near sunrise and the
# maximum mid-afternoon tracks the sun, not a government's time zone, and is
# unaffected by daylight saving. One offset per AOI, because a per-pixel offset
# would give every pixel a different day boundary and the result could no longer
# be one daily raster with one time axis. That is exact for a country under an
# hour of longitude wide (most of cy-bench) and leaves a residual at the edges
# of a wide one -- CONUS spans 3.9 h, so its coasts keep about +/-2 h.
era5_solar_offset <- function(aoi) {
  box <- AOI_BOUNDS[[aoi]]
  as.integer(round(((box$lon_min + box$lon_max) / 2) / 15))
}

# The single UTC day of padding a shifted block needs, or NULL if none.
#
# Shifting the hourly axis by +h means a local day begins h hours into the
# previous UTC day, so an eastward AOI needs the UTC day *before* the block; a
# westward one needs the day *after*. One day either way covers any offset, |h|
# never exceeding 12.
.era5_pad_date <- function(year, months, utc_offset) {
  if (utc_offset == 0) return(NULL)
  first <- as.Date(sprintf("%d-%02d-01", year, min(months)))
  last  <- seq(as.Date(sprintf("%d-%02d-01", year, max(months))),
               by = "month", length.out = 2L)[2] - 1
  if (utc_offset > 0) first - 1 else last + 1
}

# One UTC day of hourly ERA5, as the pad for a shifted block. 3 variables x 24
# hours = 72 fields, so it is a trivial request -- but it IS a request, and CDS
# charges queue time per request, so it is cached under a name that the
# per-year hourly cleanup does not match. Pads are a few hundred KB and get
# reused by the next year, which is worth far more than the space.
cds_pad_day_nc <- function(aoi, date, cache_dir, verbose = TRUE) {
  dest <- file.path(cache_dir, sprintf("era5_cds_%s_pad_%s", aoi, format(date, "%Y%m%d")))
  if (dir.exists(dest) && length(list.files(dest, pattern = "\\.nc$"))) return(dest)
  if (verbose) cat(sprintf("+pad %s, ", format(date, "%Y-%m-%d")))

  .cds_fetch(list(
    product_type    = "reanalysis",
    variable        = as.list(ERA5_ML_VARS),
    year            = format(date, "%Y"),
    month           = format(date, "%m"),
    day             = format(date, "%d"),
    time            = as.list(sprintf("%02d:00", 0:23)),
    area            = .cds_area(aoi),
    data_format     = "netcdf",
    download_format = "unarchived"
  ), dest, paste("pad", format(date)))
}

# Move the hourly time axis onto local solar time, so the daily resample that
# follows bins local days rather than UTC days.
.era5_shift_time <- function(ds, utc_offset) {
  if (utc_offset == 0) return(ds)
  ds$assign_coords(time = ds[["time"]] + np$timedelta64(as.integer(utc_offset), "h"))
}

# Keep only whole local days belonging to this block. Shifting leaves a partial
# day hanging off the far end from the pad, which would otherwise be written as
# a real day built from a fraction of its hours.
.era5_keep_dates <- function(rasters, from, to) {
  lapply(rasters, function(r) {
    keep <- which(time(r) >= from & time(r) <= to)
    out  <- r[[keep]]
    time(out) <- time(r)[keep]
    out
  })
}

# Cache directory name for one request's worth of months. A single month keeps
# the original {year}-{mm} spelling, so caches written before request batching
# existed are still found and never re-downloaded; a run adds its last month.
# Both match era5-aoi.R's "^era5_cds_{aoi}_{year}-" discard pattern.
.cds_dir_name <- function(aoi, year, months) {
  if (length(months) == 1) sprintf("era5_cds_%s_%d-%02d", aoi, year, months)
  else sprintf("era5_cds_%s_%d-%02d-%02d", aoi, year, min(months), max(months))
}

.cds_label <- function(year, months) {
  if (length(months) == 1) sprintf("%d-%02d", year, months)
  else sprintf("%d-%02d..%02d", year, min(months), max(months))
}

# How many months to ask for in one request, from the AOI's size.
#
# Waiting is per request; processing is per byte. So the span is made as large
# as memory allows: era5_month_rasters() holds several intermediates of the
# hourly array at once for the humidity calculation, so the raw block is capped
# near `budget` and the peak lands around four times that.
#
# CONUS comes out at 1 month, which is exactly what this did before spans
# existed; a country the size of Burkina Faso comes out at 12, collapsing a
# whole year into a single request.
.cds_auto_span <- function(aoi, budget = 250e6) {
  box   <- AOI_BOUNDS[[aoi]]
  ncell <- ceiling((box$lon_max - box$lon_min) / 0.25) *
           ceiling((box$lat_max - box$lat_min) / 0.25)
  per_month <- 744 * ncell * 4 * length(ERA5_ML_VARS)   # hours x cells x float32 x vars
  max(1L, min(12L, as.integer(floor(budget / per_month))))
}

# Split a year into the runs of months that will each become one CDS request.
#
# Months already cached on their own -- from a run predating batching, or one
# interrupted partway -- are emitted as single-month groups, so a resumed year
# reuses them instead of paying to fetch them again inside a larger run.
.cds_plan_groups <- function(aoi, year, cache_dir, span) {
  have <- vapply(1:12, function(m) {
    d <- file.path(cache_dir, .cds_dir_name(aoi, year, m))
    dir.exists(d) && length(list.files(d, pattern = "\\.nc$")) > 0
  }, logical(1))

  groups <- list()
  run    <- integer(0)
  for (m in 1:12) {
    if (have[m]) {
      if (length(run)) { groups <- c(groups, list(run)); run <- integer(0) }
      groups <- c(groups, list(m))
    } else {
      run <- c(run, m)
      if (length(run) >= span) { groups <- c(groups, list(run)); run <- integer(0) }
    }
  }
  if (length(run)) groups <- c(groups, list(run))
  groups
}

# Download a run of consecutive months of hourly ERA5 over one AOI, returning
# the path to a directory holding the netCDF(s) for them. Runs already in
# cache_dir are reused, which is what makes an interrupted multi-year run cheap
# to resume.
#
# WHY A RUN AND NOT A MONTH. CDS charges queue time per *request*, not per byte:
# a measured Burkina Faso year spent 18-20 minutes waiting for every 2 minutes
# of actual processing, whatever the month. Asking for twelve months in one
# request therefore costs about what one month costs, and turns a year from
# twelve waits into one. The only thing bounding the span is memory during the
# daily reduction -- see .cds_auto_span().
#
# A directory rather than a file because CDS decides how many files to send:
# `download_format = "unarchived"` gets a plain netCDF for a single variable,
# but a request mixing instantaneous variables (2m_temperature,
# 2m_dewpoint_temperature) with accumulated ones (total_precipitation) always
# comes back as a zip of two -- stepType-instant and stepType-accum -- because
# the two have different time semantics. Both shapes are normalised to "a
# directory of .nc files" here so open_cds_month() only has one case to handle.
cds_period_nc <- function(aoi, year, months, cache_dir, verbose = TRUE) {
  months <- sort(as.integer(months))
  dest <- file.path(cache_dir, .cds_dir_name(aoi, year, months))
  if (dir.exists(dest) && length(list.files(dest, pattern = "\\.nc$"))) {
    if (verbose) cat("cached, ")
    return(dest)
  }

  # The longest month in the run. CDS drops date combinations that do not exist
  # (31 February and so on) rather than failing, so asking every month for the
  # same day list is safe -- and for a single month this is exactly the day
  # count the per-month version used, so that path is unchanged.
  ndays <- max(vapply(months, function(m) {
    first <- as.Date(sprintf("%d-%02d-01", year, m))
    as.integer(format(seq(first, by = "month", length.out = 2L)[2] - 1, "%d"))
  }, integer(1)))

  request <- list(
    product_type    = "reanalysis",
    variable        = as.list(ERA5_ML_VARS),
    year            = as.character(year),
    month           = as.list(sprintf("%02d", months)),
    day             = as.list(sprintf("%02d", seq_len(ndays))),
    time            = as.list(sprintf("%02d:00", 0:23)),
    # CDS wants the box as North, West, South, East, in -180..180 -- which is
    # how AOI_BOUNDS already stores it, so no 0..360 conversion here.
    area            = .cds_area(aoi),
    data_format     = "netcdf",
    download_format = "unarchived"
  )

  # CDS enforces a per-request cost limit, and rejects anything over it with a
  # 403 rather than trimming it. The ceiling is not documented per dataset and
  # differs by account, so rather than hard-coding a guess, an over-large run is
  # halved and retried until it fits. Each half is cached under its own name, so
  # the discovery is paid for once and the next year reuses the working size.
  out <- tryCatch(.cds_fetch(request, dest, .cds_label(year, months)),
                  error = function(e) {
    msg <- conditionMessage(e)
    too_big <- grepl("cost limit|too large|request is too", msg, ignore.case = TRUE)
    if (!too_big || length(months) < 2) stop(e)
    mid <- length(months) %/% 2
    cat(sprintf("\n    CDS refused %d months as too large; splitting into %d + %d\n",
                length(months), mid, length(months) - mid))
    flush(stdout())
    c(cds_period_nc(aoi, year, months[seq_len(mid)], cache_dir, verbose),
      cds_period_nc(aoi, year, months[-seq_len(mid)], cache_dir, verbose))
  })
  out
}

# The smallest real request there is: one variable, one hour, a 1x1 degree box.
# Proves credentials, licence acceptance and connectivity together in a few
# seconds, which is the cheap way to find out that something is wrong -- the
# alternative is discovering it an hour into an unattended run.
cds_probe <- function(target = tempfile(fileext = ".nc")) {
  cdsapi$Client()$retrieve(CDS_DATASET, list(
    product_type    = "reanalysis",
    variable        = list("2m_temperature"),
    year = "2000", month = "06", day = list("01"), time = list("12:00"),
    area            = c(12.9, -2.1, 11.9, -1.1),
    data_format     = "netcdf",
    download_format = "unarchived"), target)
  if (!file.exists(target)) stop("CDS probe returned no file")
  target
}

# Open a month downloaded by cds_month_nc() in the same shape the ARCO path
# produces: long variable names, a coordinate called `time`, no stray singleton
# dimensions. Several netCDFs in the directory are merged -- CDS splits
# instantaneous and accumulated variables into separate files on the same hourly
# axis, so they recombine exactly.
open_cds_month <- function(nc_dir) {
  files <- list.files(nc_dir, pattern = "\\.nc$", full.names = TRUE)
  if (!length(files)) stop("No netCDF files in ", nc_dir)

  parts <- lapply(files, function(f) xr$open_dataset(f))
  # compat/join are passed explicitly rather than left to xarray's defaults,
  # which it warns are about to change. join = "exact" is the load-bearing one:
  # the split files must share an hourly axis, and an outer join would quietly
  # pad a mismatch with NaN instead of failing.
  ds    <- if (length(parts) == 1) parts[[1]] else {
    xr$merge(parts, compat = "override", join = "exact")
  }

  # The current CDS backend calls the time coordinate valid_time; the older one
  # called it time. Accept either.
  if ("valid_time" %in% iterate(ds$variables$keys())) {
    ds <- ds$rename(list(valid_time = "time"))
  }

  # Reanalysis output carries `number` (ensemble member) and `expver`
  # coordinates that era5_da_to_rast()'s [time, lat, lon] reshape would trip
  # over if they ever arrive as dimensions rather than scalars.
  ds <- ds$squeeze(drop = TRUE)

  present <- iterate(ds$data_vars$keys())
  missing <- setdiff(names(CDS_SHORT_TO_LONG), present)
  if (length(missing)) {
    stop("CDS month in ", nc_dir, " is missing ", paste(missing, collapse = ", "),
         "; got ", paste(present, collapse = ", "))
  }
  ds$rename(as.list(CDS_SHORT_TO_LONG))
}


# Open one or more cached CDS directories as a single hourly dataset.
#
# cds_period_nc() returns more than one directory when a run had to be split to
# fit the per-request cost limit. Those pieces are consecutive in time and share
# a grid, so they concatenate; sortby makes the result independent of the order
# the pieces came back in.
open_cds_dirs <- function(dirs) {
  parts <- lapply(dirs, open_cds_month)
  if (length(parts) == 1) parts[[1]] else xr$concat(parts, dim = "time")$sortby("time")
}


# ===== DAILY RASTERS =====

# Daily tasmax/tasmin/hursmin/pr for one month of hourly data. Takes an xarray
# Dataset carrying ERA5_ML_VARS on dims (time, latitude, longitude), which is
# what both sources are normalised to, so the aggregation and the unit
# conventions below are defined exactly once and cannot drift apart.
era5_month_rasters <- function(ds) {
  temp_k     <- ds[["2m_temperature"]]
  dewpoint_c <- ds[["2m_dewpoint_temperature"]] - 273.15
  rh_pct     <- era5_relative_humidity(temp_k - 273.15, dewpoint_c)

  list(
    tasmax  = era5_da_to_rast(temp_k$resample(time = "1D")$max()),
    tasmin  = era5_da_to_rast(temp_k$resample(time = "1D")$min()),
    hursmin = era5_da_to_rast(rh_pct$resample(time = "1D")$min()),
    # Daily total precip (m) -> mm -> a kg/m2/s-style rate, so that multiplying
    # by 86400 downstream (as EC-Earth3's pr already is) reproduces mm/day.
    pr      = era5_da_to_rast(
      (ds[["total_precipitation"]]$resample(time = "1D")$sum() * 1000) / 86400)
  )
}

# Daily tasmax/tasmin/hursmin/pr rasters for one AOI and one calendar year,
# built to match the *raw* units ML-sample-gen.R's extract_daily_by_county()
# already expects from the EC-Earth3 series (see its scale_factor/unit_offset
# comments), so the same downstream extraction code works unchanged:
#   tasmax, tasmin  Kelvin        (ERA5's 2m_temperature is already Kelvin)
#   hursmin         percent       (0-100, matches EC-Earth3's hursmin)
#   pr              kg/m2/s-style flux (so the existing *86400 -> mm/day works)
# forecast_year has no ERA5 equivalent (it's observed, not a hindcast), so
# treat `year` as both the hindcast_start and the target year when calling
# this from era5-sample-gen.R.
#
# source picks where the hourly data comes from. Both produce identical output;
# they differ only in what they cost:
#   "cds"  server-side area subsetting, tens of MB per CONUS month, but queued.
#          The right default for a multi-year series. Needs credentials -- see
#          the CDS section above.
#   "arco" anonymous, no account, no queue, but transfers a global field per
#          hour regardless of AOI: ~50 GB and several hours per CONUS year.
era5_daily_rasters <- function(aoi, year, source = c("cds", "arco"),
                               cds_cache_dir = file.path("era5_cds_cache", aoi),
                               verbose = TRUE, months_per_request = NULL,
                               utc_offset = 0L) {
  source <- match.arg(source)
  utc_offset <- as.integer(utc_offset)

  # Walk the year in blocks rather than selecting it in one go. A CONUS year of
  # hourly ERA5 is ~830 MB per variable and the humidity calculation holds
  # several intermediates of that size at once, so the whole-year form needs
  # many GB before it produces its first number. Only the daily aggregates (a
  # few hundred layers, ~35 MB per variable) are kept, so blocks bound memory.
  #
  # How big a block is depends on the source, because their costs are shaped
  # differently. CDS charges a long queue wait per request and almost nothing
  # for size, so blocks are made as large as memory allows and a small country
  # becomes one request for the whole year. ARCO has no queue at all but
  # transfers a global field per hour regardless of AOI, so its cost is purely
  # per byte and there is nothing to gain by batching -- it stays monthly, which
  # also keeps a slow transfer reporting progress twelve times a year.
  month_starts <- as.Date(sprintf("%d-%02d-01", year, 1:12))
  month_ends   <- c(month_starts[-1], as.Date(sprintf("%d-01-01", year + 1))) - 1

  span <- if (source == "arco") 1L
          else if (is.null(months_per_request)) .cds_auto_span(aoi)
          else max(1L, min(12L, as.integer(months_per_request)))
  groups <- if (source == "arco") as.list(1:12)
            else .cds_plan_groups(aoi, year, cds_cache_dir, span)
  if (verbose && source == "cds") {
    cat(sprintf("  %d CDS request(s) for %d, up to %d month(s) each%s\n",
                length(groups), year, span,
                if (utc_offset == 0) "" else
                  sprintf(" (+1 pad day each, local solar time UTC%+d)", utc_offset)))
  } else if (verbose && utc_offset != 0) {
    cat(sprintf("  local solar time UTC%+d\n", utc_offset))
  }

  if (source == "arco") {
    # chunks = list(time = 24) hands the read to dask so those per-hour global
    # chunks are fetched concurrently; a day per chunk lines up exactly with the
    # daily resample, so nothing is read twice.
    dask$config$set(scheduler = "threads", num_workers = 32L)
    ds_all <- open_era5(chunks = list(time = 24L))
    # Subset variables before anything else -- see the note in the script block
    # below on why cropping a 270-variable, 150-year dataset is a trap.
    ds_all <- keep_variables(ds_all, ERA5_ML_VARS)
  }

  blocks <- lapply(groups, function(ms) {
    if (verbose) cat(sprintf("  ERA5 %s (%s) ... ", .cds_label(year, ms), source))
    t0 <- Sys.time()

    want_from <- month_starts[min(ms)]
    want_to   <- month_ends[max(ms)]
    pad_date  <- .era5_pad_date(year, ms, utc_offset)

    ds <- if (source == "arco") {
      # ARCO slices arbitrary time ranges, so the padding a shift needs is just
      # a wider window -- no extra request, because there are no requests.
      lo <- if (is.null(pad_date)) want_from else min(want_from, pad_date)
      hi <- if (is.null(pad_date)) want_to   else max(want_to,   pad_date)
      crop_to_aoi(ds_all$sel(time = py$slice(format(lo), format(hi))), aoi)
    } else {
      main <- open_cds_dirs(cds_period_nc(aoi, year, ms, cds_cache_dir, verbose))
      if (is.null(pad_date)) main else {
        pad <- open_cds_dirs(cds_pad_day_nc(aoi, pad_date, cds_cache_dir, verbose))
        # sortby rather than relying on concat order, so this is correct whether
        # the pad belongs before the block (eastward) or after it (westward).
        xr$concat(list(main, pad), dim = "time")$sortby("time")
      }
    }

    ds  <- .era5_shift_time(ds, utc_offset)
    out <- era5_month_rasters(ds)
    # Drop the partial local day the shift leaves hanging off the far end.
    out <- .era5_keep_dates(out, want_from, want_to)

    if (verbose) {
      cat(sprintf("%d days, %.1f min\n", nlyr(out$tasmax),
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
      # Flush per block, so a redirected log reports progress as it happens
      # rather than a buffer at a time -- the difference between watching a run
      # and guessing at it.
      flush(stdout())
    }
    out
  })

  # Stitch the months back into one raster per variable. terra's c() concatenates
  # layers but does not always carry the per-layer time along, so it is
  # reassigned from the blocks explicitly.
  vars <- c("tasmax", "tasmin", "hursmin", "pr")
  out  <- lapply(vars, function(v) {
    r <- do.call(c, lapply(blocks, `[[`, v))
    time(r) <- do.call(c, lapply(blocks, function(b) time(b[[v]])))
    r
  })
  names(out) <- vars
  out
}

# Run only when executed as a script (not when sourced for its functions, as
# ML-sample-gen.R does for era5_daily_rasters()).
if (sys.nframe() == 0L) {
  # (currently set small on purpose, as a quick smoke test — widen these back
  # out to CONUS / a full year / etc. once you've confirmed it works)
  aoi       <- "test"
  variables <- c("2m_temperature", "total_precipitation")
  start     <- "2015-01-01"
  end       <- "2015-01-01"
  resample  <- "daily"  # one of "hourly" (raw, no resampling), "daily", "monthly", "yearly"
  out_file  <- "era5_test.nc"

  ds <- open_era5()
  print(iterate(ds$data_vars$keys()))  # all variables available in the ERA5 store
  # Subset variables AND time before cropping the AOI -- for a seam-crossing
  # AOI (e.g. burkina_faso), crop_to_aoi()'s xr$concat() does not stay lazy
  # for this store, so running it against the full dataset (~270 variables,
  # 86 years of hourly data) instead of an already-small slice is dramatically
  # (minutes-to-hours vs. seconds) slower. See era5_daily_rasters() for more.
  ds <- keep_variables(ds, variables)
  ds <- ds$sel(time = py$slice(start, end))
  ds <- crop_to_aoi(ds, aoi)
  ds <- resample_time(ds, resample, variables)

  print(ds)

  message("Writing to ", out_file)
  ds$to_netcdf(out_file)
}
