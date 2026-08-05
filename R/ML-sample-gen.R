# ML Sample Generation — batch over hindcast years (climate data only)

# ===== LOAD PACKAGES =====
library(terra)
library(sf)
library(dplyr)
library(tidyr)
if (!requireNamespace("arrow",  quietly = TRUE)) install.packages("arrow")
library(arrow)


# ===== MAIN VARIABLES TO UPDATE ACCORDINGLY =====

country            <- "CONUS"     # which country to generate samples for: see COUNTRY_CONFIG below
climate_model      <- "EC-Earth3"
member             <- "r1i4p1f1"
target_yrs         <- 2021:2026   # climate (yield) years to generate output for

# Forecast lead year of the hindcast product to use.
#   1 = initialized ~13 months before the target year   (s{Y-1}, covering {Y-1}1101-{Y}1031)
#   2 = initialized ~25 months before the target year   (s{Y-2}, covering {Y-1}1101-{Y}1031)
# In both cases the file covers Nov (Y-1) - Oct (Y); only the initialization
# date differs. e.g. target year 2000 at FY2 uses the s1998 file
# pr_..._s1998-r1i4p1f1_gr_19991101-20001031.nc
forecast_year <- 1

# The hindcast initialization years implied by the targets above.
hindcast_start_yrs <- target_yrs - forecast_year

# Server mount (smb://137.158.30.70/CRL_storage mounted at /Volumes/CRL_storage)
day_data_dir <- "/Volumes/CRL_storage/all_datasets/dcpp_daily/EC-Earth3/r1i4p1f1/day"

# Version directories to search, newest first (a file present in several wins
# from the newest). The dcppA-hindcast runs and the s2021-s2023 dcppB-forecast
# runs sit under v20210910; the s2024/s2025 dcppB-forecast runs were published
# later under v20250108.
nc_versions  <- c("v20250108", "v20210910")

# Retry policy for reads off the SMB share (see read_nc_region). Seconds to wait
# after each failed attempt; the last value repeats if attempts exceed its length.
nc_read_attempts <- 5L
nc_read_backoff  <- c(5, 15, 30, 60)

harv_tif   <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CroplandPastureArea2000_Geotiff/cropland2000_area.tif"
county_shp <- "/Users/kdoerkse/Documents/UCT/AgImpacts/us_county_2020_shapefile/cb_2020_us_county_20m.shp"
nuts_gpkg  <- "/Users/kdoerkse/Documents/UCT/AgImpacts/NUTS_RG_60M_2024_3035.gpkg"
cybench_poly_dir <- "/Users/kdoerkse/Documents/UCT/AgImpacts/cy-bench/polygons"
proc_root  <- "/Users/kdoerkse/Documents/UCT/AgImpacts"
out_dir    <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data/EC-Earth3"

# ===== COUNTRY CONFIG =====
# Each country defines:
#   lon_range / lat_range : climate-data subset window, in the NetCDF's 0-360 lon space
#                           (lon_range[1] > lon_range[2] means the window straddles
#                           the prime meridian -- see read_nc_region)
#   out_dir               : where this country's parquets are written
#   load_boundaries       : zero-arg function returning list(sf, vect) of admin units,
#                           reprojected to EPSG:4326, with standardized columns
#                           adm2 (finest unit name), adm2_id (unique id), adm1 (parent unit)
COUNTRY_CONFIG <- list(
  usa = list(
    lon_range       = c(233, 298),   # CONUS in 0-360 lon space
    lat_range       = c(23, 52),
    out_dir         = file.path(proc_root, "CONUS_processed_data", "EC-Earth3"),
    load_boundaries = function() load_us_boundaries(county_shp)
  ),
  germany = list(
    lon_range       = c(5, 16),      # Germany ~6-15 deg E
    lat_range       = c(46, 56),     # Germany ~47-55 deg N
    out_dir         = file.path(proc_root, "Germany_processed_data", "EC-Earth3"),
    load_boundaries = function() load_nuts_boundaries(nuts_gpkg, cntr_code = "DE", level = 3)
  )
)

# cy-bench admin-2 countries. Windows are the shapefile bounding box padded by
# ~1.5 deg (about two EC-Earth3 cells) so edge polygons keep full cell overlap.
# Burkina Faso is the wrap case: 5.5W-2.4E becomes 353.0 -> 4.0 through zero.
CYBENCH_COUNTRIES <- list(
  BF = list(lon_range = c(353.0, 4.0),   lat_range = c(8.0, 16.5)),    # Burkina Faso
  ET = list(lon_range = c(31.5, 49.5),   lat_range = c(2.0, 16.5)),    # Ethiopia
  MG = list(lon_range = c(41.5, 52.0),   lat_range = c(-27.0, -10.5)), # Madagascar
  MW = list(lon_range = c(31.0, 37.5),   lat_range = c(-18.5, -8.0))   # Malawi
)

for (.cc in names(CYBENCH_COUNTRIES)) {
  local({
    cc <- .cc
    COUNTRY_CONFIG[[cc]] <<- list(
      lon_range       = CYBENCH_COUNTRIES[[cc]]$lon_range,
      lat_range       = CYBENCH_COUNTRIES[[cc]]$lat_range,
      out_dir         = file.path(proc_root, paste0(cc, "_processed_data"), "EC-Earth3"),
      load_boundaries = function() {
        load_cybench_boundaries(file.path(cybench_poly_dir, cc, paste0(cc, ".shp")))
      }
    )
  })
}
rm(.cc)

# "CONUS" is an alias for the usa config: the data dir, crop AOI tag, and output
# filenames all use the CONUS label, so accept it as a country key too.
COUNTRY_CONFIG$CONUS <- COUNTRY_CONFIG$usa

if (!country %in% names(COUNTRY_CONFIG)) {
  stop("Unknown country '", country, "'. Options: ", paste(names(COUNTRY_CONFIG), collapse = ", "))
}
cfg       <- COUNTRY_CONFIG[[country]]
lon_range <- cfg$lon_range
lat_range <- cfg$lat_range
out_dir   <- cfg$out_dir


# ===== FUNCTIONS =====

# List every NetCDF for one variable across all version dirs, newest version
# first. The SMB share is slow to stat, so each variable is listed once and
# cached for the rest of the session.
.nc_index <- new.env(parent = emptyenv())

nc_listing <- function(var) {
  if (!is.null(.nc_index[[var]])) return(.nc_index[[var]])
  files <- unlist(lapply(nc_versions, function(v) {
    d <- file.path(day_data_dir, var, "gr", v)
    if (!dir.exists(d)) return(character(0))
    file.path(d, list.files(d, pattern = "\\.nc$"))
  }), use.names = FALSE)
  if (!length(files)) stop("No NetCDFs found for variable '", var, "' under ", day_data_dir)
  .nc_index[[var]] <- files
  files
}

# Resolve the NetCDF for one variable / initialization year / lead year.
#
# A run initialized in November of year `hstart` produces one file per lead year:
# forecast year `fy` covers {hstart+fy-1}1101 - {hstart+fy}1031, i.e. it forecasts
# calendar year hstart+fy. So FY1 of s1998 is 19981101-19991031 (forecasts 1999)
# and FY2 of s1998 is 19991101-20001031 (forecasts 2000).
#
# The filename is matched rather than constructed, because two parts of it vary
# across the archive and neither is derivable from hstart:
#   - experiment tag: the dcppA-hindcast stream stops at s2020; initializations
#     from s2021 on are dcppB-forecast, the real-time continuation of the same
#     lead-year layout.
#   - initialization label: usually s{YYYY}, but some later runs (e.g. s2024)
#     carry the full init date, s{YYYYMMDD}.
# The lead-year time span is the unambiguous part, so it does the work.
nc_path <- function(var, hstart, fy = forecast_year) {
  files <- nc_listing(var)
  base  <- basename(files)

  ok <- startsWith(base, paste0(var, "_day_", climate_model, "_dcpp")) &
        grepl(sprintf("_s%d(1101)?-%s_", hstart, member), base) &
        grepl(sprintf("_gr_%d1101-%d1031\\.nc$", hstart + fy - 1L, hstart + fy), base)

  if (!any(ok)) {
    stop(sprintf("No NetCDF for %s s%d FY%d (expected span %d1101-%d1031)",
                 var, hstart, fy, hstart + fy - 1L, hstart + fy))
  }
  # nc_versions is newest-first, so on the rare duplicate the newer publication wins.
  files[ok][1]
}

# Experiment tag (dcppA-hindcast / dcppB-forecast) backing a given target year,
# read off the resolved tasmax filename. Reported per year so a mixed-stream
# batch is visible in the log.
nc_experiment <- function(hstart, fy = forecast_year) {
  sub(sprintf("^tasmax_day_%s_(dcpp[AB]-[a-z]+)_.*$", climate_model), "\\1",
      basename(nc_path("tasmax", hstart, fy)))
}

# Read a NetCDF file on the SMB mount via a subprocess.
#
# terra::rast() and ncdf4::nc_close() both segfault when the HDF5 library tries
# to clean up file handles on this SMB share (invalid-permissions crash at 0x5b0).
# The fix: run ncdf4 in a child Rscript, have it saveRDS() the CONUS array slice
# BEFORE any cleanup runs, then read the RDS back in the parent session.
# The child process may crash on exit but the data is already on disk.
#
# Subsets a lon/lat window given in the NetCDF's 0-360 lon space. The window
# is taken from the globally-selected country (lon_range / lat_range from cfg).
#
# A window may straddle the prime meridian (e.g. Burkina Faso, 5.5W-2.4E). In
# 0-360 space that is two blocks, 354.5-360 and 0-2.4, which cannot be written
# as one lon >= a & lon <= b interval, and ncvar_get cannot read a
# non-contiguous hyperslab. Such a window is flagged by lon_range[1] >
# lon_range[2] ("from a, eastward through 0, to b"); the selection then becomes
# lon >= a | lon <= b, and each contiguous run of indices is read separately and
# bound back together along the lon axis, western block first so the longitudes
# are ascending once arrays_to_rast() shifts them to -180-180.
read_nc_region <- function(f, varname, lon_range = get("lon_range", envir = globalenv()),
                           lat_range = get("lat_range", envir = globalenv())) {
  tmp_rds <- tempfile(fileext = ".rds")
  tmp_r   <- tempfile(fileext = ".R")

  wraps <- lon_range[1] > lon_range[2]

  script <- c(
    "library(ncdf4)",
    paste0("nc      <- nc_open(", deparse(f), ")"),
    "lon     <- ncvar_get(nc, 'lon')",
    "lat     <- ncvar_get(nc, 'lat')",
    if (wraps) {
      sprintf("li      <- c(which(lon >= %s), which(lon <= %s))", lon_range[1], lon_range[2])
    } else {
      sprintf("li      <- which(lon >= %s & lon <= %s)", lon_range[1], lon_range[2])
    },
    sprintf("ki      <- which(lat >= %s & lat <= %s)", lat_range[1], lat_range[2]),
    "if (!length(li) || !length(ki)) stop('empty lon/lat selection')",
    "t_raw   <- ncvar_get(nc, 'time')",
    "t_units <- ncatt_get(nc, 'time', 'units')$value",
    # Split li into contiguous runs; one read per run, collapse_degen = FALSE so a
    # single-column run keeps its lon dimension instead of being dropped.
    "runs    <- split(li, cumsum(c(1L, diff(li) != 1L)))",
    "vals    <- array(NA_real_, dim = c(length(li), length(ki), length(t_raw)))",
    "off     <- 0L",
    "for (ix in runs) {",
    paste0("  blk <- ncvar_get(nc, ", deparse(varname),
           ", start = c(ix[1], ki[1], 1L), count = c(length(ix), length(ki), -1L),",
           " collapse_degen = FALSE)"),
    "  vals[(off + 1L):(off + length(ix)), , ] <- blk",
    "  off <- off + length(ix)",
    "}",
    paste0("saveRDS(list(vals = vals, lon = lon[li], lat = lat[ki],",
           " t_raw = t_raw, t_units = t_units), ", deparse(tmp_rds), ")")
  )
  writeLines(script, tmp_r)
  on.exit(unlink(c(tmp_r, tmp_rds)), add = TRUE)

  # The SMB share drops reads intermittently on a weak link. Retry with backoff
  # rather than letting one blip abort a multi-hour batch. A share that has gone
  # away entirely fails immediately instead of burning retries on every file.
  for (attempt in seq_len(nc_read_attempts)) {
    unlink(tmp_rds)
    system(paste("Rscript", shQuote(tmp_r)), ignore.stdout = TRUE, ignore.stderr = TRUE)

    if (file.exists(tmp_rds)) {
      d <- tryCatch(readRDS(tmp_rds), error = function(e) NULL)  # truncated write
      if (!is.null(d)) return(d)
    }
    if (!dir.exists(day_data_dir)) {
      stop("Data share is not mounted: ", day_data_dir)
    }
    if (attempt < nc_read_attempts) {
      wait <- nc_read_backoff[min(attempt, length(nc_read_backoff))]
      cat(sprintf("  read failed (attempt %d/%d), retrying in %ds: %s\n",
                  attempt, nc_read_attempts, wait, basename(f)))
      Sys.sleep(wait)
    }
  }
  stop("Failed to read after ", nc_read_attempts, " attempts: ", basename(f))
}

# Convert the list returned by read_nc_region() into an in-memory SpatRaster.
# ncdf4 returns arrays as [lon, lat, time]; terra expects cells in row-major
# order (lon varies fastest) with the northernmost row first.
arrays_to_rast <- function(d) {
  # shift 0-360 → -180-180, but only for lons in the western hemisphere
  # (>180). Eastern-hemisphere lons (e.g. Germany, 5-16°E) are unchanged.
  lons <- ifelse(d$lon > 180, d$lon - 360, d$lon)
  lats <- d$lat
  dx   <- abs(diff(d$lon)[1])
  dy   <- abs(diff(d$lat)[1])
  nlyr <- length(d$t_raw)

  r <- rast(nrows = length(lats), ncols = length(lons), nlyrs = nlyr,
            xmin  = min(lons) - dx / 2,  xmax = max(lons) + dx / 2,
            ymin  = min(lats) - dy / 2,  ymax = max(lats) + dy / 2,
            crs   = "EPSG:4326")

  vals <- d$vals
  # Flip lat to north-to-south if stored south-to-north
  if (lats[1] < lats[length(lats)]) {
    vals <- vals[, length(lats):1L, , drop = FALSE]
  }
  # Reshape [lon, lat, time] → [ncell, nlyr] (lon varies fastest = terra convention)
  dim(vals) <- c(length(lons) * length(lats), nlyr)
  values(r) <- vals

  t_origin <- as.Date(sub(".*since\\s+", "", d$t_units))
  time(r)  <- t_origin + as.integer(d$t_raw)
  r
}

# Load one forecast year of the s{hstart} run (see nc_path for the lead-year layout).
load_temperature_data <- function(day_data_dir, hindcast_start_year, region_vect,
                                   climate_model, member = "r1i4p1f1",
                                   nc_version = "v20210910", fy = forecast_year) {
  load_one <- function(var) {
    arrays_to_rast(read_nc_region(nc_path(var, hindcast_start_year, fy), var))
  }

  tasmax <- load_one("tasmax")
  tasmin <- load_one("tasmin")

  cat("=== TASMAX ===\n")
  cat("  Daily layers:", nlyr(tasmax), "  Time range:", format(range(time(tasmax))), "\n")
  cat("=== TASMIN ===\n")
  cat("  Daily layers:", nlyr(tasmin), "  Time range:", format(range(time(tasmin))), "\n")

  list(tasmax = tasmax, tasmin = tasmin)
}

load_precip_data_daily <- function(day_data_dir, hindcast_start_year, region_vect,
                                    climate_model, member = "r1i4p1f1",
                                    nc_version = "v20210910", fy = forecast_year) {
  pr <- arrays_to_rast(read_nc_region(nc_path("pr", hindcast_start_year, fy), "pr"))

  cat("\n=== PR daily ===\n")
  cat("  Daily layers:", nlyr(pr), "  Time range:", format(range(time(pr))), "\n")
  pr
}

load_hursmin_data <- function(day_data_dir, hindcast_start_year, region_vect,
                               climate_model, member = "r1i4p1f1",
                               nc_version = "v20210910", fy = forecast_year) {
  f <- nc_path("hursmin", hindcast_start_year, fy)
  hursmin <- arrays_to_rast(read_nc_region(f, "hursmin"))

  cat("\n=== HURSMIN ===\n")
  cat("  Daily layers:", nlyr(hursmin), "\n")
  cat("  Time range:", format(range(time(hursmin))), "\n")
  hursmin
}

# Admin-boundary loaders. Each returns list(sf, vect) reprojected to EPSG:4326
# with standardized columns: adm2 (finest unit name), adm2_id (unique id),
# adm1 (parent unit name). Downstream code only ever reads these three columns.

# Load CONUS county boundaries from a local shapefile, excluding non-contiguous territories.
load_us_boundaries <- function(shp_path) {
  us_counties <- st_read(shp_path, quiet = TRUE) %>%
    filter(!STUSPS %in% c("AK", "HI", "PR", "GU", "VI", "MP", "AS")) %>%
    st_transform(crs = 4326) %>%
    mutate(adm2 = NAME, adm2_id = GEOID, adm1 = STATE_NAME)
  list(sf = us_counties, vect = vect(us_counties))
}

# Load admin units from the Eurostat NUTS geopackage for one country.
# level 3 = NUTS-3 (e.g. German Kreise) as the finest (adm2) unit; the parent
# NUTS-1 region (e.g. Bundesland) is matched on the first 3 chars of NUTS_ID
# and used as adm1.
# Load cy-bench admin-2 polygons (BF, ET, MG, MW).
#
# These shapefiles carry a single attribute, adm_id, formatted {CC}-A2-{AA}{BB}
# where AA indexes the parent admin-1 unit and BB the admin-2 unit within it
# (verified against the real administrative structure: BF 13 regions / 45
# provinces, MW 3 regions / 28 districts, MG 6 provinces / 22 regions).
# cy-bench ships no unit names anywhere, so adm2 carries the id itself and adm1
# is the reconstructed parent code {CC}-A1-{AA} -- both are codes, not names.
# adm_id is also the join key to the cy-bench yield csvs.
load_cybench_boundaries <- function(shp_path) {
  regions <- st_read(shp_path, quiet = TRUE) %>%
    st_transform(crs = 4326) %>%
    mutate(
      adm2_id = adm_id,
      adm2    = adm_id,
      adm1    = paste0(substr(adm_id, 1, 2), "-A1-",
                       substr(sub("^[A-Z]{2}-A2-", "", adm_id), 1, 2))
    )
  list(sf = regions, vect = vect(regions))
}

load_nuts_boundaries <- function(gpkg_path, cntr_code, level = 3) {
  nuts <- st_read(gpkg_path, quiet = TRUE)

  nuts1 <- nuts %>%
    st_drop_geometry() %>%
    filter(LEVL_CODE == 1) %>%
    select(adm1_id = NUTS_ID, adm1 = NUTS_NAME)

  regions <- nuts %>%
    filter(CNTR_CODE == cntr_code, LEVL_CODE == level) %>%
    mutate(adm2 = NUTS_NAME, adm2_id = NUTS_ID, adm1_id = substr(NUTS_ID, 1, 3)) %>%
    left_join(nuts1, by = "adm1_id") %>%
    st_transform(crs = 4326)

  list(sf = regions, vect = vect(regions))
}

# Load the cropland area raster and resample it to the target grid.
load_harvested_area_raster <- function(tif_path, region_vect, target_rast) {
  harv_frac_r <- resample(
    crop(rast(tif_path), region_vect),
    target_rast,
    method = "bilinear"
  )
  cat("\nHarvested area raster loaded.\n")
  cat("  Pixels with cropland > 0:", sum(values(harv_frac_r) > 0, na.rm = TRUE), "\n")
  cat("  Pixels with no cropland: ",
      sum(is.na(values(harv_frac_r)) | values(harv_frac_r) == 0), "\n")
  harv_frac_r
}

# Extract daily cropland-area-weighted county means for all layers in a SpatRaster.
# harv_extracted must be the data frame produced in the static loading block via
#   terra::extract(harv_frac_r, region$vect, weights=TRUE, exact=TRUE, cells=TRUE)
# with a combined_weight column = cropland_frac * cell_overlap_fraction.
# scale_factor and unit_offset:
#   temperature:   scale_factor = 1,     unit_offset = -273.15, col_suffix = "_C"
#   hursmin:       scale_factor = 1,     unit_offset = 0,       col_suffix = ""
#   precipitation: scale_factor = 86400, unit_offset = 0,       col_suffix = "_mm"
# Returns a long data frame with one row per county-date.
extract_daily_by_county <- function(masked_raster, var_name, regions,
                                    harv_extracted, unit_offset = -273.15, col_suffix = "_C",
                                    scale_factor = 1) {
  dates <- time(masked_raster)
  doys  <- as.integer(format(dates, "%j"))

  cat("Extracting", var_name, "—", length(dates), "daily layers\n")

  daily_list <- lapply(seq_along(dates), function(i) {
    # Extract by pre-computed cell numbers: avoids re-running polygon intersection
    # geometry each day. Same grid as harv_frac_r so cell indices are valid.
    cell_vals <- terra::extract(masked_raster[[i]], harv_extracted$cell)[, 1]

    county_wmeans <- tapply(seq_len(length(cell_vals)), harv_extracted$ID, function(idx) {
      v    <- cell_vals[idx]
      w    <- harv_extracted$combined_weight[idx]
      keep <- !is.na(v) & !is.na(w) & w > 0
      if (!any(keep)) return(NA_real_)
      weighted.mean(v[keep], w[keep])
    })

    data.frame(
      adm2    = regions$adm2,
      adm2_id = regions$adm2_id,
      adm1    = regions$adm1,
      date    = dates[i],
      doy     = doys[i],
      value   = as.numeric(county_wmeans) * scale_factor + unit_offset
    )
  })

  result <- bind_rows(daily_list)
  names(result)[names(result) == "value"] <- paste0(var_name, col_suffix)
  result
}


# ===== MAIN =====
# Wrapped in a function so the file can be sourced to reuse its functions
# (e.g. for testing) without triggering the static load + batch loop. Run as a
# script (Rscript ML-sample-gen.R) it executes via the guard at the bottom.
main <- function() {

# ----- LOAD STATIC DATA (once, before the batch loop) -----

cat("\nGenerating samples for country:", country, "\n")
cat("  Forecast year:  FY", forecast_year, "\n", sep = "")
cat("  Target years:   ", min(target_yrs), "-", max(target_yrs), "\n", sep = "")
cat("  Hindcast starts:", min(hindcast_start_yrs), "-", max(hindcast_start_yrs), "\n", sep = "")
cat("  Output dir:     ", out_dir, "\n", sep = "")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

region <- cfg$load_boundaries()
cat("  Admin units (adm2):", nrow(region$sf), "\n")

# Reference temperature load for grid alignment (harvested area resampling)
cat("\nLoading reference temperature grid for year", hindcast_start_yrs[1], "...\n")
ref_rast    <- arrays_to_rast(read_nc_region(
  nc_path("tasmax", hindcast_start_yrs[1]), "tasmax"))[[1]]
harv_frac_r <- load_harvested_area_raster(harv_tif, region$vect, ref_rast)

# Compute pixel-to-county mapping with exact cell overlap fractions once.
# Column order from terra: ID, cropland_frac, weight (overlap fraction), cell.
harv_extracted           <- terra::extract(harv_frac_r, region$vect,
                                            weights = TRUE, exact = TRUE, cells = TRUE)
names(harv_extracted)[2] <- "cropland_frac"
harv_extracted$combined_weight <- harv_extracted$cropland_frac * harv_extracted$weight

harv_mask   <- ifel(harv_frac_r > 0, 1, NA)

rm(ref_rast)
gc()

cat("\nStatic data loaded. Starting batch loop over hindcast years.\n")


# ----- BATCH LOOP -----

for (hindcast_start in hindcast_start_yrs) {
  year_of_interest <- hindcast_start + forecast_year

  cat("\n\n========================================\n")
  cat("hindcast_start =", hindcast_start, "  FY", forecast_year,
      "  target year =", year_of_interest, "\n")
  cat("experiment     =", nc_experiment(hindcast_start), "\n")
  cat("========================================\n\n")

  # Two naming conventions are already in use, and data_loader.py globs for both:
  # the FY1 CONUS series is climate_{model}_s{hstart}-{member}.parquet inside a
  # {model}/ folder, everything else is climate_{aoi}_{model}_fy{n}_s{hstart}.parquet.
  # Match whichever applies so new years join the existing series.
  out_file <- file.path(out_dir, if (forecast_year == 1 && country %in% c("CONUS", "usa")) {
    sprintf("climate_%s_s%d-%s.parquet", climate_model, hindcast_start, member)
  } else {
    sprintf("climate_%s_%s_fy%d_s%d.parquet",
            country, climate_model, forecast_year, hindcast_start)
  })

  if (file.exists(out_file)) {
    cat("Output already exists, skipping:", basename(out_file), "\n")
    next
  }

  # ----- Load climate data for this hindcast year -----

  temp     <- load_temperature_data(day_data_dir, hindcast_start, region$vect,
                                     climate_model, member, nc_version, forecast_year)
  pr_raw   <- load_precip_data_daily(day_data_dir, hindcast_start, region$vect,
                                      climate_model, member, nc_version, forecast_year)
  hurs_raw <- load_hursmin_data(day_data_dir, hindcast_start, region$vect,
                                 climate_model, member, nc_version, forecast_year)

  # ----- Apply cropland mask -----

  tasmax_full  <- temp$tasmax  * harv_mask
  tasmin_full  <- temp$tasmin  * harv_mask
  pr_full      <- pr_raw       * harv_mask
  hursmin_full <- hurs_raw     * harv_mask

  rm(temp, pr_raw, hurs_raw)
  gc()

  # ----- Extract by county (cropland-area-weighted mean) -----

  tasmax_daily  <- extract_daily_by_county(tasmax_full,  "tasmax",  region$sf, harv_extracted)
  tasmin_daily  <- extract_daily_by_county(tasmin_full,  "tasmin",  region$sf, harv_extracted)
  hursmin_daily <- extract_daily_by_county(hursmin_full, "hursmin", region$sf, harv_extracted,
                                            unit_offset = 0, col_suffix = "")
  pr_daily      <- extract_daily_by_county(pr_full, "pr", region$sf, harv_extracted,
                                            unit_offset = 0, col_suffix = "_mm", scale_factor = 86400)

  rm(tasmax_full, tasmin_full, pr_full, hursmin_full)
  gc()

  # ----- Build full hindcast timeseries (no crop-season filter) -----

  hstart <- hindcast_start
  fy     <- forecast_year
  tyear  <- year_of_interest

  ts_data <- tasmax_daily %>%
    left_join(tasmin_daily  %>% select(adm2_id, date, tasmin_C), by = c("adm2_id", "date")) %>%
    left_join(hursmin_daily %>% select(adm2_id, date, hursmin),  by = c("adm2_id", "date")) %>%
    left_join(pr_daily      %>% select(adm2_id, date, pr_mm),    by = c("adm2_id", "date")) %>%
    mutate(year_month = format(date, "%Y-%m"), hindcast_start = hstart,
           forecast_year = fy, target_year = tyear) %>%
    select(adm2_id, adm2, adm1, hindcast_start, forecast_year, target_year,
           date, doy, year_month, tasmax_C, tasmin_C, hursmin, pr_mm)

  cat("\nFull hindcast climate dataset — hindcast_start:", hindcast_start,
      " FY:", forecast_year, " target year:", year_of_interest, "\n")
  cat("  Rows (county × day):", nrow(ts_data), "\n")
  cat("  Unique counties:    ", n_distinct(ts_data$adm2_id), "\n")
  cat("  Date range:         ", format(range(ts_data$date)), "\n")

  write_parquet(ts_data, out_file)
  cat("\nWritten:", out_file, "\n")

  rm(tasmax_daily, tasmin_daily, hursmin_daily, pr_daily, ts_data)
  gc()
}

cat("\n\nAll years processed.\n")

}  # end main()


# ===== BATCH DRIVER =====

# Target years per cy-bench country, matched to that country's maize yield
# coverage in cybench-data/maize/{CC}/yield_maize_{CC}.csv so every generated
# sample has an observed yield to pair with.
CYBENCH_TARGET_YRS <- list(
  BF = 1984:2019,
  ET = 1999:2020,
  MG = 1990:2010,
  MW = 2018:2023
)

# Run main() for several countries in one session. main() and read_nc_region()
# read country, lon_range, lat_range, out_dir and the year vectors from the
# global environment, so each country's settings are installed there before its
# turn. `spec` is a named list of country -> target years.
run_countries <- function(spec) {
  for (cn in names(spec)) {
    cc <- COUNTRY_CONFIG[[cn]]
    if (is.null(cc)) stop("Unknown country '", cn, "'")
    yrs <- spec[[cn]]
    assign("country",            cn,                  envir = globalenv())
    assign("cfg",                cc,                  envir = globalenv())
    assign("lon_range",          cc$lon_range,        envir = globalenv())
    assign("lat_range",          cc$lat_range,        envir = globalenv())
    assign("out_dir",            cc$out_dir,          envir = globalenv())
    assign("target_yrs",         yrs,                 envir = globalenv())
    assign("hindcast_start_yrs", yrs - forecast_year, envir = globalenv())
    main()
  }
}

# Run only when executed as a script (not when sourced for its functions).
# With no arguments, runs the single country configured at the top of the file.
# With country codes as arguments (e.g. `Rscript ML-sample-gen.R BF ET MG MW`),
# runs each in turn using its CYBENCH_TARGET_YRS range and its own out_dir.
if (sys.nframe() == 0L) {
  .args <- commandArgs(trailingOnly = TRUE)
  if (length(.args)) {
    .unknown <- setdiff(.args, names(CYBENCH_TARGET_YRS))
    if (length(.unknown)) {
      stop("No target years defined for: ", paste(.unknown, collapse = ", "))
    }
    run_countries(CYBENCH_TARGET_YRS[.args])
  } else {
    main()
  }
}
