# ML Sample Generation

# ===== LOAD PACKAGES =====
library(terra)
library(sf)
library(haven)
library(dplyr)
library(tidyr)
library(ggplot2)
if (!requireNamespace("tigris", quietly = TRUE)) install.packages("tigris")
library(tigris)
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
library(arrow)


# ===== MAIN VARIABLES TO UPDATE ACCORDINGLY =====

# Crop name aliases — calendar files use "Maize", yield files use "corn"
CROP_CAL_NAME   <- c(maize = "Maize", corn = "Maize")
CROP_YIELD_NAME <- c(maize = "corn",  corn = "corn")

climate_model  <- "EC-Earth3"  # Climate model name as it appears in filenames
hindcast_start <- 2005         # Initialization year of the climate model
                         # (e.g. "s2005" in hindcast_s2005-r1i4p1f1)
# Target crop yield year: always hindcast_start + 1
# (e.g. hindcast_start = 2005 → file 20051101-20061031.nc → 2006 crop yield)
year_of_interest <- hindcast_start + 1
crop_name <- "Maize"


data_dir  <- "/Users/kdoerkse/Documents/UCT/AgImpacts/dcpp_data"
cal_dir   <- "/Users/kdoerkse/Documents/UCT/AgImpacts/ALL_CROPS_netCDF_0.5deg_unfilled"
harv_tif  <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CroplandPastureArea2000_Geotiff/cropland2000_area.tif"
yield_dta <- file.path("/Users/kdoerkse/Documents/UCT/AgImpacts/historical_yields",
                       sprintf("%s_gmfd_v1_ready.dta", CROP_YIELD_NAME[tolower(crop_name)]))


# ===== FUNCTIONS =====

# Load CONUS county boundaries, excluding non-contiguous territories.
# Returns list(sf, vect).
load_conus_boundaries <- function(boundary_year = 2020) {
  us_counties <- counties(cb = TRUE, resolution = "20m", year = boundary_year) %>%
    filter(!STUSPS %in% c("AK", "HI", "PR", "GU", "VI", "MP", "AS")) %>%
    st_transform(crs = 4326)
  list(sf = us_counties, vect = vect(us_counties))
}


# Load daily tasmax and tasmin for a target crop yield year.
# Loads only the single dcppA hindcast file spanning hindcast_start_year Nov 1
# through crop_yield_year Oct 31 (e.g. s2005 → 20051101-20061031.nc for 2006
# crop yield), rotates longitude to -180/180, and crops to region_vect.
# Returns list(tasmax, tasmin).
load_temperature_data <- function(data_dir, hindcast_start_year, region_vect,
                                   climate_model, member = "r1i4p1f1") {
  yr0 <- hindcast_start_year        # e.g. 2005  (the "s2005" initialisation year)
  yr1 <- yr0 + 1                    # e.g. 2006  (the target crop yield year)

  nc_file <- function(var) {
    file.path(data_dir, sprintf(
      "%s_day_%s_dcppA-hindcast_s%d-%s_gr_%d1101-%d1031.nc",
      var, climate_model, yr0, member, yr0, yr1))
  }

  load_var <- function(var) {
    crop(rotate(rast(nc_file(var))), region_vect)
  }

  tasmax <- load_var("tasmax")
  tasmin <- load_var("tasmin")

  cat("=== TASMAX ===\n")
  cat("Resolution (deg):", res(tasmax), "\n")
  cat("Daily layers:", nlyr(tasmax), "\n")
  cat("Time range:", format(range(time(tasmax))), "\n")
  cat("Units:", units(tasmax)[1], "\n")

  cat("\n=== TASMIN ===\n")
  cat("Resolution (deg):", res(tasmin), "\n")
  cat("Daily layers:", nlyr(tasmin), "\n")
  cat("Time range:", format(range(time(tasmin))), "\n")
  cat("Units:", units(tasmin)[1], "\n")

  list(tasmax = tasmax, tasmin = tasmin)
}


# Load monthly precipitation (pr) for a target crop yield year.
# Loads the single dcppA hindcast Amon file spanning hindcast_start_year Nov
# through crop_yield_year Oct (e.g. s2005 → 200511-200610.nc for 2006 crop
# yield), rotates longitude to -180/180, and crops to region_vect.
# Returns the monthly pr SpatRaster (one layer per month, units kg m-2 s-1).
load_precip_data <- function(data_dir, hindcast_start_year, region_vect,
                              climate_model, member = "r1i4p1f1") {
  yr0 <- hindcast_start_year        # e.g. 2005  (the "s2005" initialisation year)
  yr1 <- yr0 + 1                    # e.g. 2006  (the target crop yield year)

  f <- file.path(data_dir, sprintf(
    "pr_Amon_%s_dcppA-hindcast_s%d-%s_gr_%d11-%d10.nc",
    climate_model, yr0, member, yr0, yr1))

  pr <- crop(rotate(rast(f)), region_vect)

  cat("\n=== PR ===\n")
  cat("Resolution (deg):", res(pr), "\n")
  cat("Monthly layers:", nlyr(pr), "\n")
  cat("Time range:", format(range(time(pr))), "\n")
  cat("Units:", units(pr)[1], "\n")

  pr
}


# Load the crop calendar NetCDF for crop_name (e.g. "Maize"), resample all
# DOY layers to the temperature grid, and compute the preseason start
# (plant.start minus preseason_days, floored at DOY 1).
# year sets the reference calendar year used in the printed summary.
# Returns a named list of DOY SpatRasters plus cal_df (summary data frame).
load_crop_calendar <- function(cal_dir, crop_name, region_vect, target_rast,
                                year, preseason_days = 90) {
  cal_path <- file.path(cal_dir, paste0(CROP_CAL_NAME[tolower(crop_name)], ".crop.calendar.nc"))
  crop_cal     <- rast(cal_path)
  crop_cal_reg <- crop(crop_cal, ext(region_vect))

  resamp <- function(layer) {
    resample(crop(crop_cal_reg[[layer]], region_vect), target_rast, method = "near")
  }

  plant_doy_r       <- resamp("plant")
  harvest_doy_r     <- resamp("harvest")
  plant_start_doy_r <- resamp("plant.start")
  harvest_end_doy_r <- resamp("harvest.end")
  preseason_start_r <- max(plant_start_doy_r - preseason_days, 1)

  cat("\nResolution after resampling:\n")
  cat("  Crop calendar (resampled):", res(plant_doy_r), "\n")
  cat("  Temperature:              ", res(target_rast), "\n")

  # Summary data frame (0.5-deg grid, useful for QA)
  origin_date <- as.Date(paste0(year, "-01-01"))
  cal_df <- as.data.frame(crop_cal_reg[["plant"]], xy = TRUE, na.rm = TRUE) %>%
    rename(plant_doy = plant) %>%
    left_join(
      as.data.frame(crop_cal_reg[["harvest"]], xy = TRUE, na.rm = TRUE) %>%
        rename(harvest_doy = harvest), by = c("x", "y")) %>%
    left_join(
      as.data.frame(crop_cal_reg[["tot.days"]], xy = TRUE, na.rm = TRUE) %>%
        rename(season_length = tot.days), by = c("x", "y")) %>%
    mutate(
      plant_date   = format(as.Date(plant_doy   - 1, origin = origin_date), "%b %d"),
      harvest_date = format(as.Date(harvest_doy - 1, origin = origin_date), "%b %d")
    )

  cat("\n===", crop_name, "Crop Calendar — CONUS (", year, "reference) ===\n")
  cat("Grid cells with data:", nrow(cal_df), "\n\n")
  cat("Mean Planting (day of year):\n");    print(summary(cal_df$plant_doy))
  cat("\nMean Harvest (day of year):\n");   print(summary(cal_df$harvest_doy))
  cat("\nGrowing season length (days):\n"); print(summary(cal_df$season_length))
  cat("\nMost common planting window: ~",
      format(as.Date(round(median(cal_df$plant_doy)) - 1,          origin = origin_date), "%B %d"),
      "to",
      format(as.Date(round(quantile(cal_df$plant_doy, 0.75)) - 1,  origin = origin_date), "%B %d"), "\n")
  cat("Most common harvest window: ~",
      format(as.Date(round(median(cal_df$harvest_doy)) - 1,         origin = origin_date), "%B %d"),
      "to",
      format(as.Date(round(quantile(cal_df$harvest_doy, 0.75)) - 1, origin = origin_date), "%B %d"), "\n")

  list(
    plant_doy_r       = plant_doy_r,
    harvest_doy_r     = harvest_doy_r,
    plant_start_doy_r = plant_start_doy_r,
    harvest_end_doy_r = harvest_end_doy_r,
    preseason_start_r = preseason_start_r,
    cal_df            = cal_df
  )
}


# Load the cropland area raster (absolute area per pixel), resample to the
# temperature grid, and apply a binary mask to all supplied climate rasters so
# that zero-cropland pixels are NA before any further processing.
# pr_conus is optional; include it to mask precipitation alongside temperature.
# Returns list(harv_frac_r, tasmax, tasmin, pr) where harv_frac_r retains the
# cropland area values used as weights during extraction.
load_harvested_area <- function(tif_path, region_vect, target_rast,
                                 tasmax_conus, tasmin_conus, pr_conus = NULL) {
  harv_frac_r <- resample(
    crop(rast(tif_path), region_vect),
    target_rast,
    method = "bilinear"
  )

  harv_mask <- ifel(harv_frac_r > 0, 1, NA)

  cat("\nCropland area mask applied:\n")
  cat("  Pixels retained (cropland area > 0):",
      sum(values(harv_mask) == 1, na.rm = TRUE), "\n")
  cat("  Pixels excluded (cropland area = 0 or NA):",
      sum(is.na(values(harv_mask))), "\n")

  result <- list(
    harv_frac_r = harv_frac_r,
    tasmax      = tasmax_conus * harv_mask,
    tasmin      = tasmin_conus * harv_mask
  )
  if (!is.null(pr_conus)) result$pr <- pr_conus * harv_mask
  result
}


# Load crop yield data from a .dta file, filter to iso_code, and optionally
# subset to years covered by the climate data so the table matches the
# crop calendar period of interest.
load_yield_data <- function(dta_path, iso_code, year_range = NULL) {
  yield <- read_dta(dta_path) %>%
    filter(iso == iso_code)

  if (!is.null(year_range)) {
    yield <- yield %>% filter(year >= year_range[1], year <= year_range[2])
  }

  cat("\nCrop yield (", iso_code, "):", nrow(yield), "rows\n")
  cat("Years:", paste(range(yield$year), collapse = " - "), "\n")
  yield
}


# Apply the crop season DOY window to a daily raster, setting out-of-season
# pixels to NA.  Works for both temperature and precipitation.
apply_season_mask <- function(daily_raster, preseason_start_r, harvest_end_r, var_name) {
  dates <- time(daily_raster)
  doys  <- as.integer(format(dates, "%j"))

  cat("Masking", var_name, "—", nlyr(daily_raster), "daily layers\n")

  masked <- lapply(seq_along(dates), function(i) {
    doy  <- doys[i]
    mask <- ifel(doy >= preseason_start_r & doy <= harvest_end_r, 1, NA)
    daily_raster[[i]] * mask
  })

  result <- rast(masked)
  time(result) <- dates
  names(result) <- paste0(var_name, "_", format(dates, "%Y%m%d"))
  result
}


# Extract season-masked monthly precipitation per county, weighted by cropland
# area fraction.  Input pr_masked has one layer per month (kg m-2 s-1 mean
# rate); convert to mm/month by multiplying by seconds in each month.
# Returns a long data frame with one row per county-month (adm2_id, year_month, pr_mm).
extract_monthly_precip_by_county <- function(pr_masked, counties_vect, us_counties,
                                              harv_frac_r) {
  dates <- time(pr_masked)

  # Extract weights once
  harv_vals <- terra::extract(harv_frac_r, counties_vect)
  names(harv_vals)[2] <- "weight"

  cat("Extracting precipitation —", nlyr(pr_masked), "monthly layers\n")

  monthly_list <- lapply(seq_len(nlyr(pr_masked)), function(i) {
    # kg m-2 s-1 → mm/month: multiply by seconds in the month
    month_start  <- as.Date(format(dates[i], "%Y-%m-01"))
    next_month   <- as.Date(format(month_start + 32, "%Y-%m-01"))
    seconds_in_mo <- as.numeric(difftime(next_month, month_start, units = "secs"))

    vals_raw <- terra::extract(pr_masked[[i]] * seconds_in_mo, counties_vect)
    names(vals_raw)[2] <- "value"

    county_wmeans <- tapply(seq_len(nrow(vals_raw)), vals_raw$ID, function(idx) {
      v    <- vals_raw$value[idx]
      w    <- harv_vals$weight[idx]
      keep <- !is.na(v) & !is.na(w) & w > 0
      if (sum(keep) == 0) return(NA_real_)
      weighted.mean(v[keep], w[keep])
    })

    data.frame(
      adm2_id    = us_counties$GEOID,
      year_month = format(dates[i], "%Y-%m"),
      pr_mm      = as.numeric(county_wmeans)
    )
  })

  bind_rows(monthly_list)
}


# Extract daily values by county using harvested-area-fraction weighted mean.
# Returns a long data frame with one row per county-date.
extract_daily_by_county <- function(masked_raster, var_name, counties_vect, us_counties, harv_frac_r) {
  dates <- time(masked_raster)
  doys  <- as.integer(format(dates, "%j"))

  # Extract harvested area fraction weights once — same geometry for every day
  harv_vals <- terra::extract(harv_frac_r, counties_vect)
  names(harv_vals)[2] <- "weight"

  cat("Extracting", var_name, "—", length(dates), "daily layers\n")

  daily_list <- lapply(seq_along(dates), function(i) {
    vals_raw <- terra::extract(masked_raster[[i]], counties_vect)
    names(vals_raw)[2] <- "value"

    county_wmeans <- tapply(seq_len(nrow(vals_raw)), vals_raw$ID, function(idx) {
      v    <- vals_raw$value[idx]
      w    <- harv_vals$weight[idx]
      keep <- !is.na(v) & !is.na(w) & w > 0
      if (sum(keep) == 0) return(NA_real_)
      weighted.mean(v[keep], w[keep])
    })

    data.frame(
      adm2    = us_counties$NAME,
      adm2_id = us_counties$GEOID,
      adm1    = us_counties$STATE_NAME,
      date    = dates[i],
      doy     = doys[i],
      value_C = as.numeric(county_wmeans) - 273.15
    )
  })

  result <- bind_rows(daily_list)
  names(result)[names(result) == "value_C"] <- paste0(var_name, "_C")
  result
}


# ----- Load data -----

conus <- load_conus_boundaries()

temp  <- load_temperature_data(data_dir, hindcast_start, conus$vect, climate_model)
pr    <- load_precip_data(data_dir, hindcast_start, conus$vect, climate_model)

cal <- load_crop_calendar(
  cal_dir, crop_name, conus$vect,
  target_rast    = temp$tasmax[[1]],
  year           = year_of_interest,
  preseason_days = 60
)

harv <- load_harvested_area(
  harv_tif, conus$vect, temp$tasmax[[1]],
  tasmax_conus = temp$tasmax,
  tasmin_conus = temp$tasmin,
  pr_conus     = pr
)

yield <- load_yield_data(yield_dta, "USA", year_range = c(year_of_interest, year_of_interest))


# ----- Filter all climate to year_of_interest -----

in_year <- function(r) which(format(time(r), "%Y") == as.character(year_of_interest))

tasmax_yr <- harv$tasmax[[in_year(harv$tasmax)]]
tasmin_yr <- harv$tasmin[[in_year(harv$tasmin)]]
pr_yr     <- harv$pr[[in_year(harv$pr)]]


# ----- Apply season mask -----
# Temperature: preseason start through mean harvest DOY
# Precipitation: preseason start through harvest end DOY (full window for monthly totals)

tasmax_masked <- apply_season_mask(tasmax_yr, cal$preseason_start_r, cal$harvest_doy_r,     "tasmax")
tasmin_masked <- apply_season_mask(tasmin_yr, cal$preseason_start_r, cal$harvest_doy_r,     "tasmin")
pr_masked     <- apply_season_mask(pr_yr,     cal$preseason_start_r, cal$harvest_end_doy_r, "pr")

cat("\nSeason-masked layers (", year_of_interest, "):\n")
cat("  tasmax:", sum(!is.nan(global(tasmax_masked, "mean", na.rm = TRUE)[[1]])), "days\n")
cat("  tasmin:", sum(!is.nan(global(tasmin_masked, "mean", na.rm = TRUE)[[1]])), "days\n")
cat("  pr:    ", sum(!is.nan(global(pr_masked,     "mean", na.rm = TRUE)[[1]])), "days\n")


# ----- Extract and aggregate by county -----

tasmax_daily <- extract_daily_by_county(tasmax_masked, "tasmax", conus$vect, conus$sf, harv$harv_frac_r)
tasmin_daily <- extract_daily_by_county(tasmin_masked, "tasmin", conus$vect, conus$sf, harv$harv_frac_r)
pr_monthly   <- extract_monthly_precip_by_county(pr_masked, conus$vect, conus$sf, harv$harv_frac_r)


# ----- Build long-format timeseries -----

# Join daily temperature streams on county + date
temp_daily <- tasmax_daily %>%
  left_join(tasmin_daily %>% select(adm2_id, date, tasmin_C),
            by = c("adm2_id", "date")) %>%
  mutate(year_month = format(date, "%Y-%m"))

# Join monthly precip — repeated for every day within the same month
ts_data <- temp_daily %>%
  left_join(pr_monthly, by = c("adm2_id", "year_month"))

# Map yield to adm2_id via county name matching
county_meta <- ts_data %>%
  select(adm2, adm2_id, adm1) %>%
  distinct() %>%
  mutate(adm2_lower = tolower(adm2),
         adm1_lower = tolower(adm1))

yield_mapped <- yield %>%
  select(-any_of("adm2_id")) %>%           # drop if already present to avoid join conflict
  mutate(adm2_lower = tolower(adm2),
         adm1_lower = tolower(adm1)) %>%
  left_join(county_meta %>% select(adm2_id, adm2_lower, adm1_lower),
            by = c("adm2_lower", "adm1_lower")) %>%
  select(adm2_id, yield)

ts_data <- ts_data %>%
  left_join(yield_mapped, by = "adm2_id") %>%
  mutate(hindcast_start = hindcast_start) %>%
  select(adm2_id, adm2, adm1, hindcast_start,
         date, doy, year_month, tasmax_C, tasmin_C, pr_mm, yield)

cat("\nFinal timeseries dataset — year:", year_of_interest, "\n")
cat("  Rows (county × day):       ", nrow(ts_data), "\n")
cat("  Unique counties:           ", n_distinct(ts_data$adm2_id), "\n")
cat("  Date range:                ", format(range(ts_data$date)), "\n")
cat("  Counties with yield:       ", sum(!is.na(ts_data$yield)) > 0, "\n")
head(ts_data)


# ----- Export -----

out_file <- sprintf("/Users/kdoerkse/Documents/UCT//AgImpacts/processed_data/ml_timeseries_%s_s%d_%d.parquet",
                    climate_model, hindcast_start, year_of_interest)
write_parquet(ts_data, out_file)
cat("\nWritten:", out_file, "\n")


# ===== PLOT: Temperature time series for a random county =====

set.seed(250)
random_county_id <- sample(unique(tasmax_daily$adm2_id), 1)
county_label <- paste0(
  conus$sf$NAME[conus$sf$GEOID == random_county_id], ", ",
  conus$sf$STATE_NAME[conus$sf$GEOID == random_county_id]
)

county_ts <- tasmax_daily %>%
  filter(adm2_id == random_county_id) %>%
  select(date, tasmax_C) %>%
  left_join(tasmin_daily %>% filter(adm2_id == random_county_id) %>% select(date, tasmin_C),
            by = "date")

county_idx <- which(conus$sf$GEOID == random_county_id)
cal_extracted <- terra::extract(
  c(cal$plant_start_doy_r, cal$harvest_end_doy_r, cal$preseason_start_r),
  conus$vect[county_idx]
)
plant_start_val   <- round(mean(cal_extracted[[2]], na.rm = TRUE))
harvest_end_val   <- round(mean(cal_extracted[[3]], na.rm = TRUE))
preseason_doy_val <- round(mean(cal_extracted[[4]], na.rm = TRUE))

origin_yr <- paste0(year_of_interest, "-01-01")
vlines <- data.frame(
  date  = as.Date(c(preseason_doy_val, plant_start_val, harvest_end_val) - 1,
                  origin = origin_yr),
  event = c("plant - 90 days", "Plant start", "Harvest end")
)

ggplot(county_ts, aes(x = date)) +
  geom_vline(data = vlines, aes(xintercept = date, colour = event),
             linetype = "dashed", linewidth = 0.5) +
  geom_line(aes(y = tasmax_C, colour = "tasmax"), linewidth = 0.6) +
  geom_line(aes(y = tasmin_C, colour = "tasmin"),  linewidth = 0.6) +
  scale_colour_manual(
    values = c(tasmax = "#d73027", tasmin = "#4575b4",
               `Plant start` = "#1a9641", `Harvest end` = "#f97b2a",
               `plant - 90 days` = "#984ea3"),
    labels = c(tasmax = "Max temp", tasmin = "Min temp",
               `Plant start` = "Plant start (DOY)", `Harvest end` = "Harvest end (DOY)",
               `plant - 90 days` = "Plant start − 90 days")
  ) +
  labs(title = paste0("Daily Temperature — ", county_label, " (", year_of_interest, ")"),
       x = "Date", y = "Temperature (°C)", colour = NULL) +
  theme_minimal()
