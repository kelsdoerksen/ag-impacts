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
#   reticulate::virtualenv_install("r-reticulate", c("numpy", "xarray", "gcsfs", "zarr", "netCDF4"))

library(reticulate)
library(terra)
use_virtualenv("r-reticulate", required = TRUE)

# Avoids a hard crash on macOS when R and numpy each try to initialize their
# own OpenMP runtime.
Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

xr    <- import("xarray")
gcsfs <- import("gcsfs")
np    <- import("numpy")
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

# ERA5 stores these as running totals since the last timestep rather than
# instantaneous values, so they need to be summed (not averaged) when
# resampling to a coarser time step.
ACCUMULATED_VARS <- c(
  "total_precipitation", "large_scale_precipitation", "convective_precipitation",
  "evaporation", "runoff", "surface_runoff", "sub_surface_runoff", "snowfall"
)

# Open the ERA5 dataset. This is lazy: no data is downloaded yet, only the
# metadata needed to know what's available.
open_era5 <- function() {
  gcs <- gcsfs$GCSFileSystem(token = "anon")
  xr$open_zarr(gcs$get_mapper(ERA5_PATH), chunks = NULL)
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

# Daily tasmax/tasmin/hursmin/pr rasters for one AOI and one calendar year,
# built to match the *raw* units ML-sample-gen.R's extract_daily_by_county()
# already expects from the EC-Earth3 series (see its scale_factor/unit_offset
# comments), so the same downstream extraction code works unchanged:
#   tasmax, tasmin  Kelvin        (ERA5's 2m_temperature is already Kelvin)
#   hursmin         percent       (0-100, matches EC-Earth3's hursmin)
#   pr              kg/m2/s-style flux (so the existing *86400 -> mm/day works)
# forecast_year has no ERA5 equivalent (it's observed, not a hindcast), so
# treat `year` as both the hindcast_start and the target year when calling
# this from ML-sample-gen.R (see its climate_source == "ERA5" branch).
era5_daily_rasters <- function(aoi, year) {
  ds <- open_era5()
  # Order matters a lot here for a seam-crossing AOI (e.g. Burkina Faso):
  # crop_to_aoi()'s xr$concat() across the 0/360 line does not stay fully
  # lazy for this zarr-backed store -- it ends up eagerly touching data
  # during the concat. Doing it against the full ~270-variable, 86-year
  # (1940-present) dataset instead of the 3 variables / 1 year we actually
  # want turns a few-second operation into one that runs for over an hour
  # and pulls gigabytes into memory. So: subset variables and time FIRST,
  # crop the AOI (and its concat) LAST, once the dataset is already small.
  ds <- keep_variables(ds, c("2m_temperature", "2m_dewpoint_temperature", "total_precipitation"))
  ds <- ds$sel(time = py$slice(sprintf("%d-01-01", year), sprintf("%d-12-31", year)))
  ds <- crop_to_aoi(ds, aoi)

  temp_k     <- ds[["2m_temperature"]]
  temp_c     <- temp_k - 273.15
  dewpoint_c <- ds[["2m_dewpoint_temperature"]] - 273.15
  rh_pct     <- era5_relative_humidity(temp_c, dewpoint_c)

  tasmax <- temp_k$resample(time = "1D")$max()
  tasmin <- temp_k$resample(time = "1D")$min()
  hursmin <- rh_pct$resample(time = "1D")$min()

  # Daily total precip (m) -> mm -> a kg/m2/s-style rate, so that multiplying
  # by 86400 downstream (as EC-Earth3's pr already is) reproduces mm/day.
  pr <- (ds[["total_precipitation"]]$resample(time = "1D")$sum() * 1000) / 86400

  list(
    tasmax  = era5_da_to_rast(tasmax),
    tasmin  = era5_da_to_rast(tasmin),
    hursmin = era5_da_to_rast(hursmin),
    pr      = era5_da_to_rast(pr)
  )
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
