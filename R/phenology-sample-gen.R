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

country   <- "usa"                              # see COUNTRY_CONFIG below
aoi       <- "CONUS"                            # filename suffix (matches crop/climate parquets)
crop      <- "maize"                            # CY-Bench crop subfolder
variables <- c("ndvi", "fpar", "soil_moisture") # phenology variables to generate

cybench_dir <- "/Users/kdoerkse/Documents/UCT/AgImpacts/cy-bench/cybench-data"
county_shp  <- "/Users/kdoerkse/Documents/UCT/AgImpacts/us_county_2020_shapefile/cb_2020_us_county_20m.shp"
out_dir     <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
pheno_dir   <- file.path(out_dir, "phenology")  # phenology parquets live here, separate from climate


# ===== COUNTRY CONFIG =====
# Each country defines:
#   cy_country      : CY-Bench country subfolder code
#   load_boundaries : zero-arg fn returning a data frame keyed by GEOID adm2_id
#                     with county/state names (adm2, adm1), matching the climate set
COUNTRY_CONFIG <- list(
  usa = list(
    cy_country      = "US",
    load_boundaries = function() load_us_boundaries(county_shp)
  )
)

if (!country %in% names(COUNTRY_CONFIG)) {
  stop("Unknown country '", country, "'. Options: ", paste(names(COUNTRY_CONFIG), collapse = ", "))
}
cfg <- COUNTRY_CONFIG[[country]]


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

# Read one CY-Bench phenology CSV and shape it into the standard phenology schema:
#   adm2_id, adm2, adm1, hindcast_start, date, doy, year_month, <value_cols...>
# - adm_id "US-SS-CCC" -> 5-digit GEOID adm2_id
# - hindcast_start: Nov/Dec fall into the NEXT year's growing-season window
# - drops rows with no value in any value_col, and counties outside `county_meta`
load_phenology_csv <- function(cybench_dir, crop, cy_country, var, value_cols, county_meta) {
  f <- file.path(cybench_dir, crop, cy_country,
                 sprintf("%s_%s_%s.csv", var, crop, cy_country))
  if (!file.exists(f)) stop("CY-Bench file not found: ", f)

  raw <- read_csv_arrow(f, col_select = c("adm_id", "date", all_of(value_cols)))

  df <- raw %>%
    mutate(
      adm2_id = paste0(substr(adm_id, 4, 5), substr(adm_id, 7, 9)),
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
main <- function() {

dir.create(pheno_dir, showWarnings = FALSE, recursive = TRUE)

cat("\nGenerating phenology samples — country:", country, " crop:", crop, "\n")
county_meta <- cfg$load_boundaries()
cat("  CONUS counties (boundary set):", nrow(county_meta), "\n")

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
                           vcfg$value_cols, county_meta)

  cat("  Raw rows:", attr(df, "n_raw"), " -> kept:", nrow(df), "\n")
  cat("  Unique counties:    ", n_distinct(df$adm2_id), "\n")
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
if (sys.nframe() == 0L) {
  main()
}
