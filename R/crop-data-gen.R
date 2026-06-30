# Crop Data Generation — crop calendar DOYs + yield per admin-unit × year
#
# Generates one parquet per crop: crop_{crop}_{suffix}.parquet
# Output columns (one row per adm2_id × hindcast_start):
#   adm2_id, adm2, adm1, hindcast_start, crop,
#   plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield
#
# Multi-country via COUNTRY_CONFIG (set `country` below). The output schema is
# identical across countries; only the admin unit (adm2) differs:
#   usa     -> county (adm2), state  (adm1)  — yield joined by name
#   germany -> NUTS-3 (adm2), NUTS-1 (adm1)  — yield is NUTS-2, mapped down to
#              NUTS-3 so adm2_id matches the climate data
#
# Season buffer columns are intentionally omitted — computed at Python load time.

# ===== LOAD PACKAGES =====
library(terra)
library(sf)
library(haven)
library(dplyr)
if (!requireNamespace("tigris", quietly = TRUE)) install.packages("tigris")
if (!requireNamespace("arrow",  quietly = TRUE)) install.packages("arrow")
library(tigris)
library(arrow)


# ===== MAIN VARIABLES TO UPDATE ACCORDINGLY =====

country            <- "CONUS"     # which country: see COUNTRY_CONFIG below

# All inputs below are local — this script does not touch the SMB share. Crop
# calendar DOY modes are computed on the native 0.5° calendar grid, so no
# climate reference grid (and no NetCDF read) is needed.
cal_dir      <- "/Users/kdoerkse/Documents/UCT/AgImpacts/ALL_CROPS_netCDF_0.5deg_unfilled"
yield_dir    <- "/Users/kdoerkse/Documents/UCT/AgImpacts/historical_yields"
nuts_gpkg    <- "/Users/kdoerkse/Documents/UCT/AgImpacts/NUTS_RG_60M_2024_3035.gpkg"
out_dir      <- "/Users/kdoerkse/Documents/UCT/AgImpacts/Germany_processed_data"


# ===== CROP CONFIG =====
# Map from crop key → crop calendar .nc stem → yield .dta prefix
CROP_CONFIG <- list(
  corn    = list(cal = "Maize",    yield_file = "corn"),
  soy     = list(cal = "Soybeans", yield_file = "soy"),
  wheat   = list(cal = "Wheat",    yield_file = "wheat"),
  rice    = list(cal = "Rice",     yield_file = "rice"),
  sorghum = list(cal = "Sorghum",  yield_file = "sorghum"),
  cassava = list(cal = "Cassava",  yield_file = "cassava")
)


# ===== COUNTRY CONFIG =====
# Each country defines:
#   iso_code        : ISO code used in the yield .dta `iso` column
#   out_suffix      : suffix in the output filename crop_{crop}_{out_suffix}.parquet
#   load_boundaries : zero-arg fn returning list(sf, vect) reprojected to EPSG:4326
#                     with standardized columns adm2, adm2_id, adm1, adm1_id
#   map_yield       : fn(yield_all, region) -> data.frame(adm2_id, hindcast_start, yield)
#                     mapping raw yield rows onto the country's adm2 ids
COUNTRY_CONFIG <- list(
  usa = list(
    iso_code        = "USA",
    out_suffix      = "CONUS",
    load_boundaries = function() load_us_boundaries(),
    # USA yield rows carry county/state names but a non-GEOID adm2_id; join on
    # lowercased names to recover the GEOID-based adm2_id used everywhere else.
    map_yield = function(yield_all, region) {
      county_meta <- region$sf %>%
        st_drop_geometry() %>%
        transmute(adm2_id, adm2_lower = tolower(adm2), adm1_lower = tolower(adm1))
      yield_all %>%
        select(-any_of("adm2_id")) %>%
        mutate(adm2_lower     = tolower(adm2),
               adm1_lower     = tolower(adm1),
               hindcast_start = as.integer(year) - 1L) %>%
        left_join(county_meta, by = c("adm2_lower", "adm1_lower")) %>%
        select(adm2_id, hindcast_start, yield)
    }
  ),
  germany = list(
    iso_code        = "DEU",
    out_suffix      = "germany",
    # Per-crop calendar overrides: Germany grows winter wheat, so the plain
    # "Wheat" (spring) calendar has no coverage — use the winter calendar.
    cal_override    = list(wheat = "Wheat.Winter"),
    # NUTS-3 to match the climate samples (ML-sample-gen.R uses level 3).
    load_boundaries = function() load_nuts_boundaries(nuts_gpkg, cntr_code = "DE", level = 3),
    # German yield resolves only to NUTS-2 (adm2_id, 4-char, matching NUTS_ID).
    # Map each NUTS-2 yield value down to all its child NUTS-3 units (the first 4
    # chars of a NUTS-3 id are its parent NUTS-2 code) so the output keys on the
    # same NUTS-3 adm2_id as the climate data. The yield value is repeated across
    # children — true yield resolution stays NUTS-2.
    map_yield = function(yield_all, region) {
      nuts3 <- region$sf %>%
        st_drop_geometry() %>%
        transmute(adm2_id, nuts2_id = substr(adm2_id, 1, 4))
      yield_all %>%
        transmute(nuts2_id = adm2_id, hindcast_start = as.integer(year) - 1L, yield) %>%
        inner_join(nuts3, by = "nuts2_id", relationship = "many-to-many") %>%
        select(adm2_id, hindcast_start, yield)
    }
  )
)

if (!country %in% names(COUNTRY_CONFIG)) {
  stop("Unknown country '", country, "'. Options: ", paste(names(COUNTRY_CONFIG), collapse = ", "))
}
cfg <- COUNTRY_CONFIG[[country]]


# ===== FUNCTIONS =====

# Admin-boundary loaders. Each returns list(sf, vect) reprojected to EPSG:4326
# with standardized columns: adm2 (finest unit), adm2_id (unique id), adm1
# (parent unit name), adm1_id (parent unit id used to group the crop calendar).

# CONUS county boundaries from tigris, excluding non-contiguous territories.
load_us_boundaries <- function(boundary_year = 2020) {
  us_counties <- counties(cb = TRUE, resolution = "20m", year = boundary_year) %>%
    filter(!STUSPS %in% c("AK", "HI", "PR", "GU", "VI", "MP", "AS")) %>%
    st_transform(crs = 4326) %>%
    mutate(adm2 = NAME, adm2_id = GEOID, adm1 = STATE_NAME, adm1_id = STUSPS)
  list(sf = us_counties, vect = vect(us_counties))
}

# Admin units from the Eurostat NUTS geopackage for one country at `level`.
# adm2 = the requested level (e.g. NUTS-2); adm1 = parent NUTS-1, matched on the
# first 3 chars of NUTS_ID.
load_nuts_boundaries <- function(gpkg_path, cntr_code, level = 2) {
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

# Resolve a crop calendar .nc path, decompressing a .nc.gz to a temp file if needed.
resolve_cal_path <- function(cal_dir, cal_stem) {
  nc_path <- file.path(cal_dir, paste0(cal_stem, ".crop.calendar.nc"))
  gz_path <- paste0(nc_path, ".gz")
  if (file.exists(nc_path)) return(nc_path)
  if (file.exists(gz_path)) {
    tmp <- tempfile(fileext = ".nc")
    system2("gunzip", args = c("-c", shQuote(gz_path)), stdout = tmp)
    cat("  Decompressed", basename(gz_path), "→ temp file\n")
    return(tmp)
  }
  stop("Calendar file not found for: ", cal_stem,
       "\n  Looked for:\n  ", nc_path, "\n  ", gz_path)
}

# Load a crop calendar at its native 0.5° resolution, cropped to the region.
# DOY modes are taken per admin unit downstream, so no climate-grid resample
# is needed (and the native grid samples more pixels per unit).
load_crop_calendar_from_path <- function(cal_path, region_vect, preseason_days = 60) {
  crop_cal     <- rast(cal_path)
  crop_cal_reg <- crop(crop_cal, region_vect)

  plant_doy_r       <- crop_cal_reg[["plant"]]
  harvest_doy_r     <- crop_cal_reg[["harvest"]]
  plant_start_doy_r <- crop_cal_reg[["plant.start"]]
  harvest_end_doy_r <- crop_cal_reg[["harvest.end"]]
  preseason_start_r <- max(plant_start_doy_r - preseason_days, 1)

  list(
    plant_doy_r       = plant_doy_r,
    harvest_doy_r     = harvest_doy_r,
    plant_start_doy_r = plant_start_doy_r,
    harvest_end_doy_r = harvest_end_doy_r,
    preseason_start_r = preseason_start_r
  )
}

# For each adm1 region compute the mode of each crop calendar variable across
# all pixels within that region, returning a data frame keyed by adm1_id.
# The SAGE calendars are coarse/country-level and "unfilled", so some adm1
# regions contain no calendar pixel (common for marginal crops, e.g. maize in
# Germany). Any adm1 with an NA mode is backfilled with the country-wide mode.
compute_adm1_mode_calendar <- function(cal_list, region_sf) {
  mode_val <- function(x) {
    x <- round(x[!is.na(x)])
    if (length(x) == 0) return(NA_real_)
    as.numeric(names(sort(table(x), decreasing = TRUE))[1])
  }

  cal_stack <- c(
    cal_list$plant_start_doy_r,
    cal_list$plant_doy_r,
    cal_list$harvest_doy_r,
    cal_list$harvest_end_doy_r
  )
  names(cal_stack) <- c("plant_start_doy", "plant_doy", "harvest_doy", "harvest_end_doy")

  adm1_sf <- region_sf %>%
    group_by(adm1_id) %>%
    summarise(.groups = "drop")

  ex <- terra::extract(cal_stack, vect(adm1_sf))

  cal_cols  <- names(cal_stack)
  natl_mode <- sapply(ex[cal_cols], mode_val)

  adm1_mode <- ex %>%
    mutate(adm1_id = adm1_sf$adm1_id[ID]) %>%
    select(-ID) %>%
    group_by(adm1_id) %>%
    summarise(across(everything(), mode_val), .groups = "drop")

  # Backfill adm1 regions with no calendar pixel using the national mode.
  n_filled <- 0L
  for (col in cal_cols) {
    miss <- is.na(adm1_mode[[col]])
    n_filled <- n_filled + sum(miss)
    adm1_mode[[col]][miss] <- natl_mode[[col]]
  }
  if (n_filled > 0)
    cat("  Backfilled", n_filled, "missing adm1 calendar values with national mode.\n")

  adm1_mode
}

# Load crop yield data from a .dta file for a given country (iso_code).
load_yield_data <- function(dta_path, iso_code, year_range = NULL) {
  yield <- read_dta(dta_path) %>%
    filter(iso == iso_code)
  if (!is.null(year_range))
    yield <- yield %>% filter(year >= year_range[1], year <= year_range[2])
  if (nrow(yield) == 0) {
    cat("  Yield rows (", iso_code, "): 0\n")
    return(yield)
  }
  cat("  Yield rows (", iso_code, "):", nrow(yield),
      "  Years:", paste(range(yield$year), collapse = " – "), "\n")
  yield
}


# ===== LOAD STATIC DATA =====

cat("\nGenerating crop data for country:", country, "  (iso:", cfg$iso_code, ")\n")
region <- cfg$load_boundaries()
cat("  Admin units (adm2):", nrow(region$sf), "\n")

cat("\nStatic data ready. Processing crops.\n")


# ===== CROP LOOP =====

for (crop_key in names(CROP_CONFIG)) {
  ccfg     <- CROP_CONFIG[[crop_key]]
  dta_path <- file.path(yield_dir, paste0(ccfg$yield_file, "_gmfd_v1_ready.dta"))
  out_file <- file.path(out_dir, sprintf("crop_%s_%s.parquet", crop_key, cfg$out_suffix))

  cat("\n\n========================================\n")
  cat("Crop:", crop_key, "  Calendar:", ccfg$cal, "\n")
  cat("========================================\n\n")

  if (file.exists(out_file)) {
    cat("Output already exists, skipping:", basename(out_file), "\n")
    next
  }

  if (!file.exists(dta_path)) {
    cat("Yield file not found, skipping:", dta_path, "\n")
    next
  }

  # ----- Yield (all years for this country) -----
  yield_all <- load_yield_data(dta_path, cfg$iso_code)
  if (nrow(yield_all) == 0) {
    cat("No yield rows for", cfg$iso_code, "— skipping crop.\n")
    next
  }

  # ----- Crop calendar -----
  cal_stem      <- if (!is.null(cfg$cal_override[[crop_key]])) cfg$cal_override[[crop_key]] else ccfg$cal
  if (cal_stem != ccfg$cal) cat("  Using calendar override:", cal_stem, "\n")
  cal_path      <- resolve_cal_path(cal_dir, cal_stem)
  cal           <- load_crop_calendar_from_path(cal_path, region$vect)
  adm1_cal_mode <- compute_adm1_mode_calendar(cal, region$sf)

  adm2_cal <- region$sf %>%
    st_drop_geometry() %>%
    select(adm2_id, adm2, adm1, adm1_id) %>%
    left_join(adm1_cal_mode, by = "adm1_id") %>%
    select(-adm1_id)

  cat("  Calendar DOY ranges:\n")
  cat("    plant_start_doy:", range(adm1_cal_mode$plant_start_doy, na.rm = TRUE), "\n")
  cat("    plant_doy:      ", range(adm1_cal_mode$plant_doy,       na.rm = TRUE), "\n")
  cat("    harvest_doy:    ", range(adm1_cal_mode$harvest_doy,     na.rm = TRUE), "\n")
  cat("    harvest_end_doy:", range(adm1_cal_mode$harvest_end_doy, na.rm = TRUE), "\n")

  # ----- Map yield onto adm2 ids -----
  yield_mapped <- cfg$map_yield(yield_all, region)

  # ----- Join calendar × yield -----
  crop_df <- adm2_cal %>%
    left_join(yield_mapped, by = "adm2_id") %>%
    mutate(crop = crop_key) %>%
    filter(!is.na(hindcast_start), !is.na(yield)) %>%
    select(adm2_id, adm2, adm1, hindcast_start, crop,
           plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield)

  if (nrow(crop_df) == 0) {
    cat("No matched county × year rows after join — skipping write.\n")
    next
  }

  cat("\nCrop parquet — ", crop_key, "\n")
  cat("  Rows (adm2 × year):  ", nrow(crop_df), "\n")
  cat("  Unique adm2:         ", n_distinct(crop_df$adm2_id), "\n")
  cat("  hindcast_start range:", paste(range(crop_df$hindcast_start), collapse = " – "), "\n")

  write_parquet(crop_df, out_file)
  cat("\nWritten:", out_file, "\n")

  rm(cal, adm1_cal_mode, adm2_cal, yield_all, yield_mapped, crop_df)
  gc()
}

cat("\n\nAll crops processed.\n")
