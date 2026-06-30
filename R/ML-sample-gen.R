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
hindcast_start_yrs <- 1980:2023     # generates output for yield years defined

# Server mount (smb://137.158.30.70/CRL_storage mounted at /Volumes/CRL_storage)
day_data_dir <- "/Volumes/CRL_storage/all_datasets/dcpp_daily/EC-Earth3/r1i4p1f1/day"
nc_version   <- "v20210910"

harv_tif   <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CroplandPastureArea2000_Geotiff/cropland2000_area.tif"
county_shp <- "/Users/kdoerkse/Documents/UCT/AgImpacts/us_county_2020_shapefile/cb_2020_us_county_20m.shp"
nuts_gpkg  <- "/Users/kdoerkse/Documents/UCT/AgImpacts/NUTS_RG_60M_2024_3035.gpkg"
out_dir    <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"

# ===== COUNTRY CONFIG =====
# Each country defines:
#   lon_range / lat_range : climate-data subset window, in the NetCDF's 0-360 lon space
#   load_boundaries       : zero-arg function returning list(sf, vect) of admin units,
#                           reprojected to EPSG:4326, with standardized columns
#                           adm2 (finest unit name), adm2_id (unique id), adm1 (parent unit)
COUNTRY_CONFIG <- list(
  usa = list(
    lon_range       = c(233, 298),   # CONUS in 0-360 lon space
    lat_range       = c(23, 52),
    load_boundaries = function() load_us_boundaries(county_shp)
  ),
  germany = list(
    lon_range       = c(5, 16),      # Germany ~6-15 deg E
    lat_range       = c(46, 56),     # Germany ~47-55 deg N
    load_boundaries = function() load_nuts_boundaries(nuts_gpkg, cntr_code = "DE", level = 3)
  )
)

# "CONUS" is an alias for the usa config: the data dir, crop AOI tag, and output
# filenames all use the CONUS label, so accept it as a country key too.
COUNTRY_CONFIG$CONUS <- COUNTRY_CONFIG$usa

if (!country %in% names(COUNTRY_CONFIG)) {
  stop("Unknown country '", country, "'. Options: ", paste(names(COUNTRY_CONFIG), collapse = ", "))
}
cfg       <- COUNTRY_CONFIG[[country]]
lon_range <- cfg$lon_range
lat_range <- cfg$lat_range


# ===== FUNCTIONS =====

# Build the path to a dcppA hindcast NetCDF file.
nc_path <- function(var, hstart) {
  file.path(day_data_dir, var, "gr", nc_version, sprintf(
    "%s_day_%s_dcppA-hindcast_s%d-%s_gr_%d1101-%d1031.nc",
    var, climate_model, hstart, member, hstart, hstart + 1))
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
read_nc_region <- function(f, varname, lon_range = get("lon_range", envir = globalenv()),
                           lat_range = get("lat_range", envir = globalenv())) {
  tmp_rds <- tempfile(fileext = ".rds")
  tmp_r   <- tempfile(fileext = ".R")

  script <- c(
    "library(ncdf4)",
    paste0("nc      <- nc_open(", deparse(f), ")"),
    "lon     <- ncvar_get(nc, 'lon')",
    "lat     <- ncvar_get(nc, 'lat')",
    sprintf("li      <- which(lon >= %s & lon <= %s)", lon_range[1], lon_range[2]),
    sprintf("ki      <- which(lat >= %s & lat <= %s)", lat_range[1], lat_range[2]),
    paste0("vals    <- ncvar_get(nc, ", deparse(varname),
           ", start = c(li[1], ki[1], 1L), count = c(length(li), length(ki), -1L))"),
    "t_raw   <- ncvar_get(nc, 'time')",
    "t_units <- ncatt_get(nc, 'time', 'units')$value",
    paste0("saveRDS(list(vals = vals, lon = lon[li], lat = lat[ki],",
           " t_raw = t_raw, t_units = t_units), ", deparse(tmp_rds), ")")
  )
  writeLines(script, tmp_r)

  system(paste("Rscript", shQuote(tmp_r)), ignore.stdout = TRUE, ignore.stderr = TRUE)
  file.remove(tmp_r)

  if (!file.exists(tmp_rds)) stop("Failed to read: ", basename(f))
  d <- readRDS(tmp_rds)
  file.remove(tmp_rds)
  d
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

# Load forecast year 1 only: s{hstart} file covering {hstart}1101–{hstart+1}1031.
load_temperature_data <- function(day_data_dir, hindcast_start_year, region_vect,
                                   climate_model, member = "r1i4p1f1",
                                   nc_version = "v20210910") {
  load_one <- function(var) {
    arrays_to_rast(read_nc_region(nc_path(var, hindcast_start_year), var))
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
                                    nc_version = "v20210910") {
  pr <- arrays_to_rast(read_nc_region(nc_path("pr", hindcast_start_year), "pr"))

  cat("\n=== PR daily ===\n")
  cat("  Daily layers:", nlyr(pr), "  Time range:", format(range(time(pr))), "\n")
  pr
}

load_hursmin_data <- function(day_data_dir, hindcast_start_year, region_vect,
                               climate_model, member = "r1i4p1f1",
                               nc_version = "v20210910") {
  f <- nc_path("hursmin", hindcast_start_year)
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
  year_of_interest <- hindcast_start + 1

  cat("\n\n========================================\n")
  cat("hindcast_start =", hindcast_start, "  yield year =", year_of_interest, "\n")
  cat("========================================\n\n")

  out_file <- file.path(out_dir, sprintf(
    "climate_%s_%s_s%d.parquet",
    country, climate_model, hindcast_start))

  if (file.exists(out_file)) {
    cat("Output already exists, skipping:", basename(out_file), "\n")
    next
  }

  # ----- Load climate data for this hindcast year -----

  temp     <- load_temperature_data(day_data_dir, hindcast_start, region$vect,
                                     climate_model, member, nc_version)
  pr_raw   <- load_precip_data_daily(day_data_dir, hindcast_start, region$vect,
                                      climate_model, member, nc_version)
  hurs_raw <- load_hursmin_data(day_data_dir, hindcast_start, region$vect,
                                 climate_model, member, nc_version)

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

  ts_data <- tasmax_daily %>%
    left_join(tasmin_daily  %>% select(adm2_id, date, tasmin_C), by = c("adm2_id", "date")) %>%
    left_join(hursmin_daily %>% select(adm2_id, date, hursmin),  by = c("adm2_id", "date")) %>%
    left_join(pr_daily      %>% select(adm2_id, date, pr_mm),    by = c("adm2_id", "date")) %>%
    mutate(year_month = format(date, "%Y-%m"), hindcast_start = hstart) %>%
    select(adm2_id, adm2, adm1, hindcast_start,
           date, doy, year_month, tasmax_C, tasmin_C, hursmin, pr_mm)

  cat("\nFull hindcast climate dataset — hindcast_start:", hindcast_start, "\n")
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


# Run only when executed as a script (not when sourced for its functions).
if (sys.nframe() == 0L) {
  main()
}
