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

country <- "BF"        # fallback when no country is passed on the command line

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
# Every cy-bench country takes the same shape -- the code IS the AOI tag and the
# directory name -- so the config is generated rather than listed. Adding an AOI
# used to mean editing this file; now it means having the cy-bench files on disk.
# OUT_DIR_OVERRIDE carries the one directory that does not follow the pattern.
OUT_DIR_OVERRIDE <- list(ET = "ET_preprocessed_data")

cy_config <- function(country, cy_crop = "maize", crop = "corn") {
  dirname <- OUT_DIR_OVERRIDE[[country]]
  if (is.null(dirname)) dirname <- paste0(country, "_processed_data")
  list(
    cy_country = country,
    cy_crop    = cy_crop,
    # cy-bench spells the crop "maize"; the parquets and python --crop use the
    # US name, so one --crop corn works across every AOI.
    crop       = crop,
    aoi        = country,
    out_dir    = file.path(proc_root, dirname)
  )
}

# Countries that actually have the crop's yield + calendar CSVs on disk.
cy_available <- function(cy_crop = "maize") {
  d <- file.path(cybench_dir, cy_crop)
  if (!dir.exists(d)) return(character(0))
  cands <- list.dirs(d, full.names = FALSE, recursive = FALSE)
  cands <- cands[nzchar(cands)]
  keep <- vapply(cands, function(cc) {
    file.exists(file.path(d, cc, sprintf("yield_%s_%s.csv", cy_crop, cc))) &&
    file.exists(file.path(d, cc, sprintf("crop_calendar_%s_%s.csv", cy_crop, cc)))
  }, logical(1))
  sort(cands[keep])
}

# t/ha -> kg/ha, to match the CONUS/Germany parquets.
YIELD_SCALE <- 1000


# ===== FUNCTIONS =====

# Admin-2 units from the cy-bench polygons, keyed by the raw adm_id — the same
# convention ML-sample-gen.R writes into the climate parquets. cy-bench ships no
# unit names, so adm2 carries the id and adm1 is the parent code {CC}-A1-{AA}.
# cy-bench uses four different adm_id conventions, and the parent code has to
# be cut differently from each one:
#   ZM-A2-0101  cy-bench synthetic   -> ZM-A1-01   (first 2 of the numeric tail)
#   US-21-007   state-county FIPS    -> US-21      (the state field)
#   BR1100015   IBGE municipality    -> BR-11      (first 2 digits = state)
#   FRK23 HU232 RO111  NUTS-3        -> FRK        (first 3 chars = NUTS-1)
# Getting this wrong is not fatal for the ML path -- yield_features.py keeps
# only adm2_id/harvest_year/yield and takes adm1 from the climate table -- but
# train_eval.py and plot_results.py group per-state errors on it.
cybench_adm1 <- function(adm_id) {
  ifelse(grepl("^[A-Z]{2}-A2-", adm_id),
         paste0(substr(adm_id, 1, 2), "-A1-",
                substr(sub("^[A-Z]{2}-A2-", "", adm_id), 1, 2)),
  ifelse(grepl("^[A-Z]{2}-[0-9]+-", adm_id),
         sub("^([A-Z]{2}-[0-9]+)-.*$", "\\1", adm_id),
  ifelse(grepl("^[A-Z]{2}[0-9]", adm_id),
         paste0(substr(adm_id, 1, 2), "-", substr(adm_id, 3, 4)),
         substr(adm_id, 1, 3))))
}

load_cybench_boundaries <- function(shp_path) {
  st_read(shp_path, quiet = TRUE) %>%
    st_drop_geometry() %>%
    transmute(
      adm2_id = adm_id,
      adm2    = adm_id,
      adm1    = cybench_adm1(adm_id)
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

avail <- cy_available()
if (!country %in% avail) {
  stop("No cy-bench maize yield/calendar for '", country, "' under ",
       file.path(cybench_dir, "maize"), ".\n  Available: ",
       paste(avail, collapse = ", "))
}
shp <- file.path(cybench_poly, country, paste0(country, ".shp"))
if (!file.exists(shp)) stop("No polygons for '", country, "': ", shp)
cfg      <- cy_config(country)
out_file <- file.path(cfg$out_dir, sprintf("crop_%s_%s.parquet", cfg$crop, cfg$aoi))

dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\nGenerating crop data — country:", country,
    " cy-bench crop:", cfg$cy_crop, "-> crop key:", cfg$crop, "\n")

region   <- load_cybench_boundaries(shp)
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

if (nrow(crop_df) == 0) {
  stop("No rows survived the joins for '", country, "'.\n",
       "  yield ids e.g.:    ", paste(head(unique(yields$adm2_id), 3), collapse = ", "), "\n",
       "  polygon ids e.g.:  ", paste(head(unique(region$adm2_id), 3), collapse = ", "), "\n",
       "  calendar ids e.g.: ", paste(head(unique(calendar$adm2_id), 3), collapse = ", "), "\n",
       "  Zero overlap means the id spellings differ between the files.")
}
matched <- length(intersect(unique(yields$adm2_id), unique(region$adm2_id)))
cat("  adm2_id overlap yield x polygons:", matched, "of",
    n_distinct(yields$adm2_id), "yield unit(s)\n")

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
