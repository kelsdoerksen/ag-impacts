# EC-Earth Climate Data Analysis

# ===== LOAD PACKAGES =====
library(terra)
library(sf)
library(haven)
library(dplyr)
library(tidyr)
library(ggplot2)
if (!requireNamespace("tigris", quietly = TRUE)) install.packages("tigris")
library(tigris)   # US county boundaries
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
library(arrow)    # for parquet export

# ===== LOAD DATA =====

# 1. Load climate NC file (daily tasmax, Kelvin, 365 layers Nov 1960 - Oct 1961)
tasmax_data <- rast("/Users/kdoerkse/Documents/UCT/climate_data/tasmax_day_EC-Earth3_dcppA-hindcast_s1960-r1i4p1f1_gr_19601101-19611031.nc")

# The raster uses 0-360 longitude — rotate to -180/180 for use with US boundaries
tasmax_data <- rotate(tasmax_data)

# 2. Load US county boundaries (ADM2, contiguous US only)
us_counties <- counties(cb = TRUE, resolution = "20m", year = 2020) %>%
  filter(!STUSPS %in% c("AK", "HI", "PR", "GU", "VI", "MP", "AS")) %>%
  st_transform(crs = 4326)

counties_vect <- vect(us_counties)

# Crop raster to CONUS extent and filter to 1960 layers only
tasmax_conus <- crop(tasmax_data, counties_vect)
tasmax_conus <- tasmax_conus[[which(format(time(tasmax_conus), "%Y") == "1960")]]

# 3. Load corn yield data
corn_yield <- read_dta("/Users/kdoerkse/Documents/UCT/historical_yields/corn_gmfd_v1_ready.dta")

corn_yield_usa <- corn_yield %>%
  filter(iso == "USA")


# ===== AGGREGATE DAILY TASMAX TO MONTHLY MEANS BY COUNTY =====

raster_dates  <- time(tasmax_conus)
raster_months <- format(raster_dates, "%Y-%m")
unique_months <- unique(raster_months)

monthly_county_list <- list()

for (mo in unique_months) {
  cat("Processing", mo, "\n")

  month_layers   <- tasmax_conus[[which(raster_months == mo)]]
  monthly_mean   <- mean(month_layers, na.rm = TRUE)

  county_vals_raw <- terra::extract(monthly_mean, counties_vect)
  county_means    <- tapply(county_vals_raw[[2]], county_vals_raw$ID, mean, na.rm = TRUE)

  result <- data.frame(
    adm2       = us_counties$NAME,
    adm2_id    = us_counties$GEOID,
    adm1       = us_counties$STATE_NAME,
    year_month = mo,
    year       = as.integer(substr(mo, 1, 4)),
    month      = as.integer(substr(mo, 6, 7)),
    tasmax_K   = as.numeric(county_means),
    tasmax_C   = as.numeric(county_means) - 273.15
  )

  monthly_county_list[[mo]] <- result
}

county_monthly_tasmax <- bind_rows(monthly_county_list)
head(county_monthly_tasmax)


# ===== RESHAPE TO WIDE FORMAT (one row per county-year) =====

county_tasmax_wide <- county_monthly_tasmax %>%
  mutate(month_name = month.abb[month]) %>%
  select(adm2, adm2_id, adm1, year, month_name, tasmax_C) %>%
  pivot_wider(names_from = month_name, values_from = tasmax_C,
              names_prefix = "tasmax_C_")


# ===== MERGE WITH CORN YIELD DATA =====

county_tasmax_wide <- county_tasmax_wide %>%
  mutate(adm2_lower = tolower(adm2),
         adm1_lower = tolower(adm1))

corn_yield_usa <- corn_yield_usa %>%
  mutate(adm2_lower = tolower(adm2),
         adm1_lower = tolower(adm1))

combined_data <- corn_yield_usa %>%
  left_join(county_tasmax_wide %>% select(-adm2, -adm1),
            by = c("adm2_lower", "adm1_lower", "year"))

cat("Rows in corn yield (USA):", nrow(corn_yield_usa), "\n")
cat("Rows after merge:        ", nrow(combined_data), "\n")
cat("Matched rows (non-NA):   ", sum(!is.na(combined_data$tasmax_C_Nov)), "\n")

head(combined_data)


# ===== MAPS =====

# --- Map 1: Mean corn yield by US county ---

yield_by_county <- corn_yield_usa %>%
  group_by(adm2_lower, adm1_lower) %>%
  summarise(mean_yield = mean(yield, na.rm = TRUE), .groups = "drop")

us_counties_yield <- us_counties %>%
  mutate(adm2_lower = tolower(NAME),
         adm1_lower = tolower(STATE_NAME)) %>%
  left_join(yield_by_county, by = c("adm2_lower", "adm1_lower"))

ggplot(us_counties_yield) +
  geom_sf(aes(fill = mean_yield), color = NA) +
  geom_sf(data = us_counties %>% group_by(STATEFP) %>% summarise(),
          fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(option = "YlGn", na.value = "grey80",
                       name = "Mean Yield\n(t/ha)") +
  labs(title = "Mean Corn Yield by US County",
       subtitle = "Source: GMFD historical yields") +
  theme_void() +
  theme(legend.position = "right")

# --- Plot: Time series of corn yield by US state (ADM1) ---

yield_by_state_year <- corn_yield_usa %>%
  group_by(adm1, year) %>%
  summarise(mean_yield = mean(yield, na.rm = TRUE), .groups = "drop")

ggplot(yield_by_state_year, aes(x = year, y = mean_yield, color = adm1, group = adm1)) +
  geom_line(linewidth = 0.6, alpha = 0.8) +
  labs(title = "Corn Yield by US State Over Time",
       x = "Year", y = "Mean Yield (t/ha)", color = "State") +
  theme_minimal() +
  theme(legend.key.size = unit(0.4, "cm"),
        legend.text = element_text(size = 7))


# --- Plot: Time series of tasmax by US county (ADM2) ---

ggplot(county_monthly_tasmax, aes(x = as.Date(paste0(year_month, "-01")),
                                   y = tasmax_C,
                                   color = adm2, group = adm2)) +
  geom_line(linewidth = 0.5, alpha = 0.6) +
  labs(title = "Monthly Mean Tasmax by US County",
       x = "Month", y = "Tasmax (°C)") +
  theme_minimal() +
  theme(legend.position = "none")  # too many counties for a legend


# --- Map 2: Mean tasmax across 1960 (Nov-Dec) ---

tasmax_mean_1960 <- mean(tasmax_conus, na.rm = TRUE) - 273.15  # convert to Celsius

tasmax_df <- as.data.frame(tasmax_mean_1960, xy = TRUE)
names(tasmax_df)[3] <- "tasmax_C"

ggplot() +
  geom_raster(data = tasmax_df, aes(x = x, y = y, fill = tasmax_C)) +
  geom_sf(data = us_counties %>% group_by(STATEFP) %>% summarise(),
          fill = NA, color = "white", linewidth = 0.3) +
  scale_fill_viridis_c(option = "inferno", na.value = "grey80",
                       name = "Tasmax (°C)") +
  coord_sf(xlim = c(-125, -66), ylim = c(24, 50)) +
  labs(title = "Mean Daily Maximum Temperature — 1960 (Nov–Dec)",
       subtitle = "Source: EC-Earth3 dcppA hindcast") +
  theme_void() +
  theme(legend.position = "right")


# --- Map: Global cropland fraction 2015 (ERA5 grid) ---

load("/Users/kdoerkse/Documents/UCT/cropland_world_2015_era5.rda")

# Filter to cells with non-zero cropland
cropland_plot <- cropland_world_2015_era5[cropland_world_2015_era5$weight > 0, ]

ggplot(cropland_plot, aes(x = x, y = y, fill = weight)) +
  geom_raster() +
  scale_fill_viridis_c(option = "YlGn", name = "Cropland\nFraction",
                       limits = c(0, 1)) +
  coord_equal() +
  labs(title = "Global Cropland Fraction 2015",
       subtitle = "Source: ERA5 grid",
       x = NULL, y = NULL) +
  theme_void() +
  theme(legend.position = "right")


# ===== EXPORT =====

write_parquet(combined_data, "corn_yield_tasmax_features.parquet")
write.csv(combined_data, "corn_yield_tasmax_features.csv", row.names = FALSE)
