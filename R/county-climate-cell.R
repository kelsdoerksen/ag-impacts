# County → climate-grid-cell lookup
#
# Derives, for each county (adm2), the coarse climate grid cell that dominates
# its cropland-weighted climate feature value. Counties sharing a dominant cell
# share (nearly) identical climate inputs, so this id is the natural grouping
# unit for spatial ("leave-one-climate-cell-out") cross-validation that removes
# the spatial-autocorrelation leakage between neighbouring counties.
#
# Reuses the overlay machinery in ML-sample-gen.R WITHOUT regenerating samples:
# it only reproduces the one-off static overlay (county polygons vs. the climate
# grid) and writes a small parquet. Run it in the same environment that runs
# ML-sample-gen.R (needs the county shapefile + one climate NetCDF + the
# cropland raster). Output lands next to the samples in `out_dir` and is read by
# run_loo_experiment.py via --cell_lookup.
#
#   Rscript county-climate-cell.R
#
# Output: {out_dir}/county_climate_cell_{country}.parquet
#   columns: adm2_id, climate_cell_id, cell_lon, cell_lat, fallback

# Source ML-sample-gen.R for its functions + cfg/paths. The `sys.nframe()` guard
# at the bottom of that file means sourcing does NOT run the batch loop.
source("ML-sample-gen.R")
library(dplyr)

cat("\nBuilding county→climate-cell lookup for country:", country, "\n")

# ----- Reproduce the static overlay (same grid as ML-sample-gen.R) -----

region   <- cfg$load_boundaries()
cat("  Admin units (adm2):", nrow(region$sf), "\n")

# Native climate grid (one reference layer is enough to define the cells).
ref_rast <- arrays_to_rast(read_nc_region(
  nc_path("tasmax", hindcast_start_yrs[1]), "tasmax"))[[1]]

# Cropland fraction resampled ONTO the climate grid, so `cell` indices below are
# climate-grid cells — identical to the cells used to weight the feature means.
harv_frac_r <- load_harvested_area_raster(harv_tif, region$vect, ref_rast)

ext <- terra::extract(harv_frac_r, region$vect,
                      weights = TRUE, exact = TRUE, cells = TRUE)
names(ext)[2] <- "cropland_frac"
ext$combined_weight <- ext$cropland_frac * ext$weight

# NA-safe weights for ranking.
ext$cw <- ifelse(is.na(ext$combined_weight), 0, ext$combined_weight)  # cropland-weighted
ext$gw <- ifelse(is.na(ext$weight),          0, ext$weight)           # geometric overlap

# ----- Pick one dominant cell per county -----
# Primary: the cell contributing the most cropland-weighted overlap (i.e. the
# cell that dominates the county's feature value). Fallback for counties with no
# cropland overlap: the cell with the largest geometric overlap, so every county
# still gets a cell. The `fallback` flag records which rule was used.

any_crop <- ext %>%
  group_by(ID) %>%
  summarise(has_crop = sum(cw) > 0, .groups = "drop")

dom_crop <- ext %>%
  inner_join(filter(any_crop, has_crop),  by = "ID") %>%
  group_by(ID) %>% slice_max(cw, n = 1, with_ties = FALSE) %>% ungroup() %>%
  mutate(fallback = FALSE)

dom_geo <- ext %>%
  inner_join(filter(any_crop, !has_crop), by = "ID") %>%
  group_by(ID) %>% slice_max(gw, n = 1, with_ties = FALSE) %>% ungroup() %>%
  mutate(fallback = TRUE)

dom <- bind_rows(dom_crop, dom_geo) %>% arrange(ID)

# terra's ID is the 1-based row index into region$vect, which preserves the
# region$sf row order — so adm2_id maps straight across.
xy <- terra::xyFromCell(ref_rast, dom$cell)
lookup <- data.frame(
  adm2_id         = region$sf$adm2_id[dom$ID],
  climate_cell_id = as.integer(dom$cell),
  cell_lon        = xy[, 1],
  cell_lat        = xy[, 2],
  fallback        = dom$fallback,
  stringsAsFactors = FALSE
)

out_file <- file.path(out_dir, sprintf("county_climate_cell_%s.parquet", country))
arrow::write_parquet(lookup, out_file)

cat("\nWritten:", out_file, "\n")
cat("  Counties:          ", nrow(lookup), "\n")
cat("  Distinct cells:    ", dplyr::n_distinct(lookup$climate_cell_id), "\n")
cat("  Avg counties/cell: ",
    round(nrow(lookup) / dplyr::n_distinct(lookup$climate_cell_id), 2), "\n")
cat("  Fallback (no crop):", sum(lookup$fallback), "\n")
