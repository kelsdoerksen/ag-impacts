# Sanity check: pair climate sequences with maize yields and report the number
# of trainable county-year samples (rows that have BOTH a climate sequence and a
# yield label). Join key: adm2_id + hindcast_start  (harvest_year = hindcast_start + 1).
#
# Reads only the key columns from the climate parquets (not the daily values), so
# it's cheap to run.

# ===== LOAD PACKAGES =====
suppressMessages({
  library(arrow)
  library(dplyr)
})

# ===== PATHS =====
base_dir     <- "/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
yield_file   <- file.path(base_dir, "crop_corn_CONUS.parquet")
climate_dir  <- file.path(base_dir, "EC-Earth3")
climate_glob <- "climate_EC-Earth3_s*-r1i4p1f1.parquet"

# ===== YIELD KEYS =====
yield <- read_parquet(yield_file) %>%
  transmute(adm2_id, hindcast_start, harvest_year = hindcast_start + 1L, yield)
cat("Yield (crop_corn_CONUS):\n")
cat("  county-year rows:", nrow(yield),
    "  harvest years:", paste(range(yield$harvest_year), collapse = " – "), "\n\n")

# ===== CLIMATE KEYS =====
# Distinct (adm2_id, hindcast_start) present in each climate parquet. We only need
# to know which county-years have a sequence, not the daily values.
climate_files <- list.files(climate_dir, pattern = glob2rx(climate_glob), full.names = TRUE)
stopifnot(length(climate_files) > 0)

climate_keys <- lapply(climate_files, function(f) {
  read_parquet(f, col_select = c("adm2_id", "hindcast_start")) %>% distinct()
}) %>% bind_rows() %>% distinct() %>%
  mutate(harvest_year = hindcast_start + 1L)

cat("Climate (EC-Earth3):\n")
cat("  files:", length(climate_files),
    "  harvest years:", paste(range(climate_keys$harvest_year), collapse = " – "),
    "  county-years with a sequence:", nrow(climate_keys), "\n\n")

# ===== JOIN: trainable samples =====
# Inner join = county-years with BOTH a climate sequence and a yield label.
trainable <- inner_join(
  climate_keys,
  yield %>% select(adm2_id, hindcast_start, yield),
  by = c("adm2_id", "hindcast_start")
)

cat("========================================\n")
cat("TRAINABLE SAMPLES (climate + yield):", nrow(trainable), "\n")
cat("  unique counties:", n_distinct(trainable$adm2_id), "\n")
cat("  harvest years:  ", paste(range(trainable$harvest_year), collapse = " – "), "\n")
cat("========================================\n\n")

# ----- Per-year breakdown -----
overlap_yrs <- sort(intersect(yield$harvest_year, climate_keys$harvest_year))

per_year <- data.frame(harvest_year = overlap_yrs) %>%
  left_join(climate_keys %>% count(harvest_year, name = "climate"),  by = "harvest_year") %>%
  left_join(yield        %>% count(harvest_year, name = "yield"),    by = "harvest_year") %>%
  left_join(trainable    %>% count(harvest_year, name = "trainable"),by = "harvest_year") %>%
  mutate(
    yield_no_climate = yield     - trainable,   # have a label but no climate sequence
    climate_no_yield = climate   - trainable    # have a sequence but no label
  )

cat("Per harvest-year (overlapping range only):\n")
print(per_year, row.names = FALSE)

cat("\nNotes:\n")
cat("  trainable        = county-years usable for training (climate + yield)\n")
cat("  yield_no_climate = labeled county-years with no climate sequence (outside hindcast range / masked)\n")
cat("  climate_no_yield = county-years with a sequence but no yield report that year\n")
