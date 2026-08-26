# Crop Data Generation — CY-Bench AOIs (BF, ET, ZM)
#
# Companion to crop-data-gen.R, which builds the US/Germany crop parquets from
# .dta yield files plus SAGE calendar rasters. The cy-bench AOIs ship their own
# yield table and crop calendar as CSVs keyed by adm_id, so they need neither
# source — this script just reshapes them into the same schema:
#
#   adm2_id, adm2, adm1, hindcast_start, crop,
#   plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield
#
# Output: {out_dir}/crop_{crop}_{aoi}.parquet, keyed on hindcast_start under the
# FY1 convention (yield is for calendar year hindcast_start + 1).
#
# Two conversions worth knowing about:
#   - UNITS. CY-Bench reports yield in t/ha; the CONUS parquets are kg/ha. Yield
#     is multiplied by 1000 so cross-AOI error metrics are in the same units.
#   - CALENDAR. CY-Bench gives one sos/eos pair per admin unit, where SAGE gives
#     four distinct DOYs. plant_start/plant both take sos and harvest/harvest_end
#     both take eos, so the aggregated-mode buffers still have something to work
#     from. The transformer path ignores these columns entirely.

# ===== LOAD PACKAGES =====
library(sf)
library(dplyr)
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
library(arrow)


# ===== MAIN VARIABLES TO UPDATE ACCORDINGLY =====

country <- "BF"        # see COUNTRY_CONFIG below

# Same knob as ML-sample-gen.R / era5-sample-gen.R, so one export points the
# whole R side at the data on a machine that is not the Mac it was written on.
proc_root    <- Sys.getenv("AGIMPACTS_ROOT", "/Users/kdoerkse/Documents/UCT/AgImpacts")
cybench_dir  <- file.path(proc_root, "cy-bench", "cybench-data")
cybench_poly <- file.path(proc_root, "cy-bench", "polygons")


# ===== COUNTRY CONFIG =====
# Each country defines:
#   cy_country : CY-Bench country subfolder code
#   cy_crop    : CY-Bench crop subfolder ("maize", "wheat")
#   crop       : crop key used in the output filename. Must be in the python
#                SUPPORTED_CROPS list (data_loader.py) — cy-bench "maize" is
#                written as "corn" so --crop corn works across AOIs.
#   aoi        : filename suffix, matching that AOI's climate parquets
#   out_dir    : processed-data root
COUNTRY_CONFIG <- list(
  BF = list(
    cy_country = "BF",
    cy_crop    = "maize",
    crop       = "corn",
    aoi        = "BF",
    out_dir    = file.path(proc_root, "BF_processed_data")
  ),
  ET = list(
    cy_country = "ET",
    cy_crop    = "maize",
    crop       = "corn",
    aoi        = "ET",
    out_dir    = file.path(proc_root, "ET_preprocessed_data")
  ),
  # Zambia. Southern hemisphere, so its maize season crosses the new year:
  # sos ~ DOY 326 (late November), eos ~ DOY 134 (mid-May) of the FOLLOWING
  # calendar year, i.e. plant_start_doy > harvest_end_doy in the output. The
  # python loader detects that and opens the seasonal window in yield_year - 1;
  # nothing needs doing here beyond passing the calendar through unchanged.
  # Directory matches what era5-sample-gen.R writes for --aoi=ZM.
  ZM = list(
    cy_country = "ZM",
    cy_crop    = "maize",
    crop       = "corn",
    aoi        = "ZM",
    out_dir    = file.path(proc_root, "ZM_processed_data")
  )
)

# t/ha -> kg/ha, to match the CONUS/Germany parquets.
YIELD_SCALE <- 1000


# ===== FUNCTIONS =====

# Admin-2 units from the cy-bench polygons, keyed by the raw adm_id — the same
# convention ML-sample-gen.R writes into the climate parquets. cy-bench ships no
# unit names, so adm2 carries the id and adm1 is the parent code {CC}-A1-{AA}.
load_cybench_boundaries <- function(shp_path) {
  st_read(shp_path, quiet = TRUE) %>%
    st_drop_geometry() %>%
    transmute(
      adm2_id = adm_id,
      adm2    = adm_id,
      adm1    = paste0(substr(adm_id, 1, 2), "-A1-",
                       substr(sub("^[A-Z]{2}-A2-", "", adm_id), 1, 2))
    )
}

# sos/eos (day of year) per admin unit. One pair per unit, no year dimension.
load_cybench_calendar <- function(cybench_dir, cy_crop, cy_country) {
  f <- file.path(cybench_dir, cy_crop, cy_country,
                 sprintf("crop_calendar_%s_%s.csv", cy_crop, cy_country))
  if (!file.exists(f)) stop("CY-Bench calendar not found: ", f)
  read_csv_arrow(f) %>%
    transmute(
      adm2_id         = adm_id,
      plant_start_doy = round(sos),
      plant_doy       = round(sos),
      harvest_doy     = round(eos),
      harvest_end_doy = round(eos)
    )
}

load_cybench_yield <- function(cybench_dir, cy_crop, cy_country) {
  f <- file.path(cybench_dir, cy_crop, cy_country,
                 sprintf("yield_%s_%s.csv", cy_crop, cy_country))
  if (!file.exists(f)) stop("CY-Bench yield file not found: ", f)
  read_csv_arrow(f) %>%
    filter(!is.na(yield)) %>%
    transmute(
      adm2_id        = adm_id,
      hindcast_start = as.integer(harvest_year) - 1L,
      yield          = yield * YIELD_SCALE
    )
}


# ===== MAIN =====
main <- function(country = get("country", envir = globalenv())) {

if (!country %in% names(COUNTRY_CONFIG)) {
  stop("Unknown country '", country, "'. Options: ", paste(names(COUNTRY_CONFIG), collapse = ", "))
}
cfg      <- COUNTRY_CONFIG[[country]]
out_file <- file.path(cfg$out_dir, sprintf("crop_%s_%s.parquet", cfg$crop, cfg$aoi))

dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\nGenerating crop data — country:", country,
    " cy-bench crop:", cfg$cy_crop, "-> crop key:", cfg$crop, "\n")

region   <- load_cybench_boundaries(file.path(cybench_poly, country,
                                              paste0(country, ".shp")))
calendar <- load_cybench_calendar(cybench_dir, cfg$cy_crop, cfg$cy_country)
yields   <- load_cybench_yield(cybench_dir, cfg$cy_crop, cfg$cy_country)

cat("  Boundary units:", nrow(region),
    "  calendar units:", nrow(calendar),
    "  yield rows:", nrow(yields), "\n")

crop_df <- yields %>%
  inner_join(region,   by = "adm2_id") %>%
  inner_join(calendar, by = "adm2_id") %>%
  mutate(crop = cfg$crop) %>%
  select(adm2_id, adm2, adm1, hindcast_start, crop,
         plant_start_doy, plant_doy, harvest_doy, harvest_end_doy, yield) %>%
  arrange(adm2_id, hindcast_start)

if (nrow(crop_df) == 0) stop("No rows survived the joins — check adm_id conventions.")

dropped <- nrow(yields) - nrow(crop_df)
cat("\nCrop parquet —", cfg$crop, "\n")
cat("  Rows (adm2 x year):  ", nrow(crop_df), "  (dropped in join:", dropped, ")\n")
cat("  Unique adm2:         ", n_distinct(crop_df$adm2_id), "\n")
cat("  hindcast_start range:", paste(range(crop_df$hindcast_start), collapse = " - "),
    " (harvest years ", paste(range(crop_df$hindcast_start) + 1, collapse = " - "), ")\n", sep = "")
cat("  Yield (kg/ha):        min", round(min(crop_df$yield), 1),
    " mean", round(mean(crop_df$yield), 1),
    " max", round(max(crop_df$yield), 1), "\n")

write_parquet(crop_df, out_file)
cat("\nWritten:", out_file, "\n")

}  # end main()


# Run only when executed as a script (not when sourced for its functions).
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0L) main() else for (cn in args) main(cn)
}
