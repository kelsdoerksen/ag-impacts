# Phenology Sample Generation — CY-Bench observed time series per admin unit
#
# Produces SEPARATE parquets, one per phenology variable, so the transformer can
# OPTIONALLY include NDVI / FPAR / soil moisture (off by default). Phenology has
# shorter and irregular temporal coverage than the EC-Earth3 climate, so it is
# kept out of the climate parquets rather than padded with NAs.
#
# Source: CY-Bench (WUR-AI/AgML-CY-Bench), maize, US.
#   ndvi_maize_US.csv          ~8-day composites,  2001–2023
#   fpar_maize_US.csv          ~10-day (dekadal),  2001–2023
#   soil_moisture_maize_US.csv daily (ssm, rsm),   2003–2023
# Cadence is preserved at native resolution — NOT resampled to daily.
#
# Keys align with the climate samples (ML-sample-gen.R): adm2_id (5-digit GEOID)
# and hindcast_start, where harvest_year = hindcast_start + 1. Each observation is
# assigned to the Nov(H)–Oct(H+1) window (EC-Earth3 forecast year 1), so phenology
# joins climate on (adm2_id, hindcast_start, date).

# ===== LOAD PACKAGES =====
library(sf)
library(dplyr)
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
library(arrow)


# ===== MAIN VARIABLES TO UPDATE ACCORDINGLY =====

country   <- "usa"                              # fallback when no args are passed
variables <- c("ndvi", "fpar", "soil_moisture") # phenology variables to generate

proc_root    <- "/Users/kdoerkse/Documents/UCT/AgImpacts"
cybench_dir  <- file.path(proc_root, "cy-bench", "cybench-data")
cybench_poly <- file.path(proc_root, "cy-bench", "polygons")
county_shp   <- file.path(proc_root, "us_county_2020_shapefile", "cb_2020_us_county_20m.shp")


# ===== COUNTRY CONFIG =====
# Each country defines:
#   cy_country      : CY-Bench country subfolder code
#   crop            : CY-Bench crop subfolder
#   aoi             : filename suffix, must match that country's crop/climate parquets
#   out_dir         : processed-data root; phenology lands in {out_dir}/phenology
#   load_boundaries : zero-arg fn returning a data frame keyed by adm2_id with
#                     adm2/adm1, matching the climate set
#   adm_id_to_adm2  : fn(adm_id) -> adm2_id, mapping the CY-Bench id onto the
#                     adm2_id convention this AOI's climate parquets use
# `usa` is the LEGACY CONUS pipeline: it remaps cy-bench's US-SS-CCC ids onto
# 5-digit GEOIDs and writes CONUS_processed_data, to match the climate parquets
# ML-sample-gen.R produced. It is NOT interchangeable with the cy-bench "US"
# AOI below, whose crop parquet and stage-3 features both key on the raw adm_id
# -- mixing the two gives an empty join, not a wrong number.
#
# Every cy-bench AOI is otherwise identical: identity id mapping, boundaries
# from its own polygon file, output beside its crop parquet. So those are
# generated rather than listed.
LEGACY_CONFIG <- list(
  usa = list(
    cy_country      = "US",
    crop            = "maize",
    aoi             = "CONUS",
    out_dir         = file.path(proc_root, "CONUS_processed_data"),
    load_boundaries = function() load_us_boundaries(county_shp),
    # "US-SS-CCC" -> 5-digit GEOID
    adm_id_to_adm2  = function(adm_id) paste0(substr(adm_id, 4, 5), substr(adm_id, 7, 9))
  )
)

OUT_DIR_OVERRIDE <- list(ET = "ET_preprocessed_data")

cy_config <- function(country, crop = "maize") {
  dirname <- OUT_DIR_OVERRIDE[[country]]
  if (is.null(dirname)) dirname <- paste0(country, "_processed_data")
  list(
    cy_country      = country,
    crop            = crop,
    aoi             = country,
    out_dir         = file.path(proc_root, dirname),
    load_boundaries = function() load_cybench_boundaries(
      file.path(cybench_poly, country, paste0(country, ".shp"))),
    adm_id_to_adm2  = function(adm_id) adm_id
  )
}

resolve_config <- function(country) {
  if (!is.null(LEGACY_CONFIG[[country]])) return(LEGACY_CONFIG[[country]])
  shp <- file.path(cybench_poly, country, paste0(country, ".shp"))
  if (!file.exists(shp)) {
    stop("No polygons for '", country, "': ", shp,
         "\n  Legacy AOIs: ", paste(names(LEGACY_CONFIG), collapse = ", "))
  }
  cy_config(country)
}

# ===== VARIABLE CONFIG =====
# value_cols : numeric columns kept from each CY-Bench CSV
VAR_CONFIG <- list(
  ndvi          = list(value_cols = "ndvi"),
  fpar          = list(value_cols = "fpar"),
  soil_moisture = list(value_cols = c("ssm", "rsm"))
)


# ===== FUNCTIONS =====

# CONUS county names from the cb_2020 shapefile, keyed by GEOID (adm2_id), with
# non-contiguous territories excluded. Restricts phenology to the same county set
# as the climate samples; CY-Bench counties not present here are dropped. -> Keeping this for consistency since CY-Bench is only from 2003
load_us_boundaries <- function(shp_path) {
  st_read(shp_path, quiet = TRUE) %>%
    st_drop_geometry() %>%
    filter(!STUSPS %in% c("AK", "HI", "PR", "GU", "VI", "MP", "AS")) %>%
    transmute(adm2_id = GEOID, adm2 = NAME, adm1 = STATE_NAME)
}

# Admin-2 units from the cy-bench polygons (BF, ET, MG, MW), keyed by the raw
# adm_id — the same convention ML-sample-gen.R's load_cybench_boundaries() writes
# into the climate parquets. cy-bench ships no unit names, so adm2 carries the id
# itself and adm1 is the reconstructed parent code {CC}-A1-{AA}.
# Four adm_id conventions across the cy-bench polygons; see the matching
# cybench_adm1() in crop-data-gen-cybench.R.
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
    transmute(adm2_id = adm_id, adm2 = adm_id, adm1 = cybench_adm1(adm_id))
}

# Read one CY-Bench phenology CSV and shape it into the standard phenology schema:
#   adm2_id, adm2, adm1, hindcast_start, date, doy, year_month, <value_cols...>
# - adm_id -> adm2_id via the AOI's `adm_id_to_adm2` (GEOID for the US, identity
#   for the cy-bench AOIs)
# - hindcast_start: Nov/Dec fall into the NEXT year's growing-season window
# - drops rows with no value in any value_col, and counties outside `county_meta`
load_phenology_csv <- function(cybench_dir, crop, cy_country, var, value_cols, county_meta,
                               adm_id_to_adm2) {
  f <- file.path(cybench_dir, crop, cy_country,
                 sprintf("%s_%s_%s.csv", var, crop, cy_country))
  if (!file.exists(f)) stop("CY-Bench file not found: ", f)

  raw <- read_csv_arrow(f, col_select = c("adm_id", "date", all_of(value_cols)))

  df <- raw %>%
    mutate(
      adm2_id = adm_id_to_adm2(adm_id),
      date    = as.Date(as.character(date), format = "%Y%m%d")
    ) %>%
    filter(if_any(all_of(value_cols), ~ !is.na(.))) %>%
    mutate(
      yr             = as.integer(format(date, "%Y")),
      mo             = as.integer(format(date, "%m")),
      doy            = as.integer(format(date, "%j")),
      hindcast_start = ifelse(mo >= 11L, yr, yr - 1L),
      year_month     = format(date, "%Y-%m")
    ) %>%
    inner_join(county_meta, by = "adm2_id") %>%
    select(adm2_id, adm2, adm1, hindcast_start, date, doy, year_month,
           all_of(value_cols)) %>%
    arrange(adm2_id, date)

  attr(df, "n_raw") <- nrow(raw)
  df
}


# ===== MAIN =====
# Wrapped so the file can be sourced for its functions without running the loop.
main <- function(country = get("country", envir = globalenv())) {

cfg       <- resolve_config(country)
crop      <- cfg$crop
aoi       <- cfg$aoi
pheno_dir <- file.path(cfg$out_dir, "phenology")  # separate from the climate parquets

dir.create(pheno_dir, showWarnings = FALSE, recursive = TRUE)

cat("\nGenerating phenology samples — country:", country, " crop:", crop, " aoi:", aoi, "\n")
county_meta <- cfg$load_boundaries()
cat("  Admin-2 units (boundary set):", nrow(county_meta), "\n")

for (var in variables) {
  if (!var %in% names(VAR_CONFIG))
    stop("Unknown variable '", var, "'. Options: ", paste(names(VAR_CONFIG), collapse = ", "))
  vcfg     <- VAR_CONFIG[[var]]
  out_name <- if (var == "soil_moisture") "soilmoisture" else var
  out_file <- file.path(pheno_dir, sprintf("phenology_%s_%s.parquet", out_name, aoi))

  cat("\n\n========================================\n")
  cat("Phenology variable:", var, "\n")
  cat("========================================\n\n")

  if (file.exists(out_file)) {
    cat("Output already exists, skipping:", basename(out_file), "\n")
    next
  }

  df <- load_phenology_csv(cybench_dir, crop, cfg$cy_country, var,
                           vcfg$value_cols, county_meta, cfg$adm_id_to_adm2)

  cat("  Raw rows:", attr(df, "n_raw"), " -> kept:", nrow(df), "\n")
  cat("  Unique admin-2 units:", n_distinct(df$adm2_id), "\n")
  cat("  hindcast_start range:", paste(range(df$hindcast_start), collapse = " – "),
      " (harvest years ", paste(range(df$hindcast_start) + 1, collapse = " – "), ")\n", sep = "")
  cat("  Date range:         ", format(range(df$date)), "\n")
  # Median days between consecutive observations for one county (native cadence).
  one <- df %>% filter(adm2_id == df$adm2_id[1]) %>% arrange(date)
  cat("  Native cadence (median day gap):", median(diff(as.integer(one$date))), "days\n")

  write_parquet(df, out_file)
  cat("\nWritten:", out_file, "\n")

  rm(df, one); gc()
}

cat("\n\nAll phenology variables processed.\n")
cat("Parquets in:", pheno_dir, "\n")

}  # end main()


# Run only when executed as a script (not when sourced for its functions).
# With no arguments, runs the single country configured at the top of the file.
# With country codes as arguments (e.g. `Rscript phenology-sample-gen.R BF`),
# runs each in turn.
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0L) main() else for (cn in args) main(cn)
}
