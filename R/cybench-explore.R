# CY-Bench data exploration — US maize, admin level 2 (counties)
# 1. Map of maize yield in 2023 across CONUS counties
# 2. Pick a random county with maize yield and plot its NDVI + soil moisture time series
#
# Data source: https://github.com/WUR-AI/AgML-CY-Bench
# adm_id format: US-<state FIPS>-<county FIPS>, joins polygons <-> CSV tables.

# ===== LOAD PACKAGES =====
library(sf)
library(dplyr)
library(readr)
library(ggplot2)
library(scales)

# ===== PATHS =====
cybench    <- "/Users/kdoerkse/Documents/UCT/AgImpacts/cy-bench"
poly_shp   <- file.path(cybench, "polygons/US/US.shp")
data_dir   <- file.path(cybench, "cybench-data/maize/US")
fig_dir    <- "/Users/kdoerkse/code/ag-impacts/R/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

yield_year <- 2023
set.seed(42)   # reproducible random county pick

# ===== LOAD POLYGONS =====
# US county polygons in WGS84 (EPSG:4326). adm_id is the join key.
# Restrict the basemap to CONUS: drop Alaska (02), Hawaii (15) and
# territories (>= 60) so the Albers map isn't distorted by far-flung geometry.
non_conus <- c("02", "15", "60", "66", "69", "72", "78")
polys <- st_read(poly_shp, quiet = TRUE) %>%
  mutate(state_fips = substr(adm_id, 4, 5)) %>%
  filter(!state_fips %in% non_conus)
message("Loaded ", nrow(polys), " CONUS county polygons")

# ===== LOAD YIELD =====
yield <- read_csv(file.path(data_dir, "yield_maize_US.csv"),
                  show_col_types = FALSE)

yield_yr <- yield %>%
  filter(harvest_year == yield_year, !is.na(yield)) %>%
  mutate(yield = yield * 1000) %>%   # CY-Bench yield is t/ha -> convert to kg/ha
  select(adm_id, yield, harvest_area, production)
message(nrow(yield_yr), " counties with maize yield in ", yield_year)

# Join yield onto polygons (left join keeps full CONUS basemap)
map_df <- polys %>% left_join(yield_yr, by = "adm_id")

# ===== MAP 1: MAIZE YIELD 2023 =====
p_map <- ggplot(map_df) +
  geom_sf(aes(fill = yield), color = "grey80", linewidth = 0.05) +
  scale_fill_gradientn(
    colours = c("#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"),
    na.value = "grey92",
    name = "Yield\n(kg/ha)",
    labels = label_comma()
  ) +
  coord_sf(crs = 5070) +   # CONUS Albers Equal Area for an undistorted map
  labs(
    title    = paste0("US maize yield by county — ", yield_year),
    subtitle = paste0(nrow(yield_yr), " counties reporting (CY-Bench)"),
    caption  = "Source: CY-Bench (WUR-AI/AgML-CY-Bench)"
  ) +
  theme_void(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold"),
    legend.position = c(0.92, 0.25)
  )

ggsave(file.path(fig_dir, paste0("maize_yield_map_", yield_year, ".png")),
       p_map, width = 11, height = 7, dpi = 200, bg = "white")
message("Wrote maize_yield_map_", yield_year, ".png")

# ===== PICK A RANDOM COUNTY WITH MAIZE YIELD =====
sel_id <- sample(yield_yr$adm_id, 1)
sel_yield <- yield_yr$yield[yield_yr$adm_id == sel_id]
message("Selected county: ", sel_id, "  (", yield_year, " yield = ",
        round(sel_yield), " kg/ha)")

# ===== TIME SERIES: NDVI =====
ndvi <- read_csv(file.path(data_dir, "ndvi_maize_US.csv"),
                 show_col_types = FALSE) %>%
  filter(adm_id == sel_id) %>%
  mutate(date = as.Date(as.character(date), format = "%Y%m%d")) %>%
  arrange(date)

p_ndvi <- ggplot(ndvi, aes(date, ndvi)) +
  geom_line(color = "#2ca25f", linewidth = 0.4) +
  scale_x_date(date_breaks = "2 years", labels = date_format("%Y")) +
  labs(title = paste0("NDVI time series — ", sel_id),
       x = NULL, y = "NDVI") +
  theme_minimal(base_size = 12)

ggsave(file.path(fig_dir, paste0("ndvi_ts_", sel_id, ".png")),
       p_ndvi, width = 11, height = 4, dpi = 200, bg = "white")
message("Wrote ndvi_ts_", sel_id, ".png  (", nrow(ndvi), " obs)")

# ===== TIME SERIES: SOIL MOISTURE =====
# ssm = surface soil moisture, rsm = root-zone (subsurface) soil moisture
sm <- read_csv(file.path(data_dir, "soil_moisture_maize_US.csv"),
               show_col_types = FALSE) %>%
  filter(adm_id == sel_id) %>%
  mutate(date = as.Date(as.character(date), format = "%Y%m%d")) %>%
  arrange(date)

# ssm and rsm are on very different scales, so facet with free y-axes
sm_long <- sm %>%
  tidyr::pivot_longer(c(ssm, rsm), names_to = "layer", values_to = "value") %>%
  mutate(layer = recode(layer,
                        ssm = "Surface soil moisture (ssm)",
                        rsm = "Root-zone soil moisture (rsm)"))

p_sm <- ggplot(sm_long, aes(date, value, color = layer)) +
  geom_line(linewidth = 0.4, show.legend = FALSE) +
  facet_wrap(~ layer, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c("Surface soil moisture (ssm)" = "#3182bd",
                                "Root-zone soil moisture (rsm)" = "#08519c")) +
  scale_x_date(date_breaks = "2 years", labels = date_format("%Y")) +
  labs(title = paste0("Soil moisture time series — ", sel_id),
       x = NULL, y = "Soil moisture") +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"))

ggsave(file.path(fig_dir, paste0("soil_moisture_ts_", sel_id, ".png")),
       p_sm, width = 11, height = 6, dpi = 200, bg = "white")
message("Wrote soil_moisture_ts_", sel_id, ".png  (", nrow(sm), " obs)")

message("\nDone. Figures in: ", fig_dir)
