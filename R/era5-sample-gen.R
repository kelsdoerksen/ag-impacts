# ERA5 -> admin-2 daily climate samples.
#
# era5.R produces gridded daily rasters straight from the ARCO-ERA5 archive;
# ML-sample-gen.R knows how to turn gridded daily rasters into the cropland-area-
# weighted, per-admin-unit parquet the ML pipeline trains on. This script is the
# join between them: it swaps ERA5 in where the EC-Earth3 hindcast NetCDFs
# normally sit and writes a parquet with exactly the same schema, so nothing
# downstream has to know which climate source it came from.
#
# ERA5 is observed, not a hindcast, so it has no initialization year and no
# lead time. The three columns that describe those in the EC-Earth3 series are
# filled in with the degenerate case: hindcast_start = target_year = the
# calendar year, forecast_year = 0.
#
#   Rscript R/era5-sample-gen.R                    # the years set below
#   Rscript R/era5-sample-gen.R 2000 2001          # years on the command line
#   Rscript R/era5-sample-gen.R 1990-2020          # or a range
#   Rscript R/era5-sample-gen.R --source=arco 2000 # override the hourly source
#
# Two caches, both worth keeping:
#   {out_dir}/daily_grids  the daily gridded rasters, one GeoTIFF per variable
#                          per year. A year present here needs no download at
#                          all, so re-running the extraction is free.
#   {out_dir}/cds_nc       the raw hourly months from CDS, one directory per
#                          month (CDS splits instantaneous and accumulated
#                          variables into separate netCDFs). Survives an
#                          interrupted run, so a multi-year series resumes at
#                          the month it stopped on rather than the start -- and
#                          is deleted per year once that year's daily grids are
#                          written, so it never accumulates across the series.
#                          Set era5_keep_hourly to keep it.
#
# On the hourly source (see the CDS section of era5.R for the full story): "cds"
# subsets server-side, so a CONUS month is tens of MB but sits in a queue;
# "arco" needs no account and never queues, but transfers a global field per
# hour whatever the AOI -- ~50 GB and several hours per CONUS year. For one year
# either is fine; for a multi-year series that gap is the whole cost of the job.

# Directory holding this file, so the two sources below resolve regardless of
# the working directory. source() records the path on its own call frame, which
# covers being sourced; Rscript only reports it in --file=.
.this_dir <- local({
  f <- NULL
  for (i in seq_len(sys.nframe())) {
    ofile <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(ofile)) f <- ofile
  }
  if (is.null(f)) {
    a <- commandArgs(trailingOnly = FALSE)
    m <- sub("^--file=", "", a[grep("^--file=", a)])
    if (length(m)) f <- m[1]
  }
  if (is.null(f)) getwd() else dirname(normalizePath(f))
})

source(file.path(.this_dir, "era5.R"))          # era5_daily_rasters()
# Defines load_us_boundaries(), load_harvested_area_raster() and
# extract_daily_by_county(). Its own batch loop is behind a sys.nframe() guard,
# so sourcing it only installs the functions and its config globals.
source(file.path(.this_dir, "ML-sample-gen.R"))

library(arrow)


# ===== CONFIG =====

era5_aoi      <- "conus"    # key into era5.R's AOI_BOUNDS
era5_aoi_tag  <- "CONUS"    # AOI token in the output filename
era5_model    <- "ERA5"     # climate_model token
era5_member   <- "obs"      # ensemble-member token; ERA5 is observed, so there is one
era5_years    <- 2000
era5_source   <- "cds"      # "cds" (server-side subset, queued) or "arco" (no account)

# Keep the raw hourly months after a year has been reduced to daily grids?
# They are only a resume aid, and they are by far the largest thing on disk
# (~1.2 GB per year against ~60 MB of daily grids), so they are discarded once
# the year they belong to is complete. Set TRUE to keep them -- worth it only if
# you intend to re-derive something the daily rasters do not carry, e.g. a
# different sub-daily statistic.
era5_keep_hourly <- FALSE

# Years may be given as single years or as ranges, so a long series does not
# have to be spelled out: `1990-2020` and `1990:2020` both work.
.parse_years <- function(x) {
  sort(unique(unlist(lapply(x, function(a) {
    p <- strsplit(a, "[-:]")[[1]]
    if (length(p) == 2) seq(as.integer(p[1]), as.integer(p[2])) else as.integer(p)
  }))))
}

.args <- commandArgs(trailingOnly = TRUE)
.src  <- grep("^--source=", .args, value = TRUE)
if (length(.src)) {
  era5_source <- sub("^--source=", "", .src[1])
  .args <- setdiff(.args, .src)
}
if (length(.args)) era5_years <- .parse_years(.args)

if (!era5_source %in% c("cds", "arco")) {
  stop("--source must be 'cds' or 'arco', got '", era5_source, "'")
}

era5_out_dir   <- file.path(proc_root, "CONUS_processed_data", era5_model)
era5_cache_dir <- file.path(era5_out_dir, "daily_grids")
era5_cds_dir   <- file.path(era5_out_dir, "cds_nc")

ERA5_VARS <- c("tasmax", "tasmin", "hursmin", "pr")


# ===== FUNCTIONS =====

# Daily gridded rasters for one year, from the GeoTIFF cache if it is there and
# from ARCO-ERA5 (slow -- see the header) if it is not.
#
# GeoTIFF carries no per-layer time, so it is reattached on read as the calendar
# year's days. era5_daily_rasters() resamples to "1D" over a whole calendar year,
# so a complete year is exactly that day sequence; a layer count that disagrees
# means the cached file is not a whole year and is treated as unusable rather
# than silently mislabelled.
era5_year_rasters <- function(aoi, year, cache_dir) {
  dates <- seq(as.Date(sprintf("%d-01-01", year)),
               as.Date(sprintf("%d-12-31", year)), by = "day")
  paths <- file.path(cache_dir, sprintf("era5_%s_%s_%d.tif", ERA5_VARS, era5_aoi_tag, year))
  names(paths) <- ERA5_VARS

  if (all(file.exists(paths))) {
    cached <- lapply(paths, rast)
    if (all(vapply(cached, nlyr, integer(1)) == length(dates))) {
      cat("Using cached daily grids for", year, "in", cache_dir, "\n")
      return(lapply(cached, function(r) { time(r) <- dates; r }))
    }
    cat("Cached grids for", year, "are not a full year; re-downloading.\n")
  }

  cat("Fetching ERA5", year, "for AOI", aoi, "via", era5_source,
      "- one month at a time:\n")
  t0 <- Sys.time()
  clim <- era5_daily_rasters(aoi, year, source = era5_source,
                             cds_cache_dir = era5_cds_dir)
  cat(sprintf("ERA5 %d downloaded in %.1f min\n", year,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
  # DEFLATE with the floating-point predictor is lossless (verified bit-identical
  # against an uncompressed write) and takes about a third off, which matters
  # once this is the only copy of the year being kept.
  for (v in ERA5_VARS) {
    writeRaster(clim[[v]], paths[[v]], overwrite = TRUE,
                gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=6"))
  }
  cat("Cached daily grids to", cache_dir, "\n")

  # The hourly months have served their purpose now that the year is reduced and
  # on disk: they exist so an interrupted year resumes at the month it stopped
  # on, not as an archive. Dropping them here caps peak usage at roughly one
  # year rather than accumulating ~1.2 GB per year for the whole series.
  if (!era5_keep_hourly && era5_source == "cds") {
    spent <- list.files(era5_cds_dir, full.names = TRUE,
                        pattern = sprintf("^era5_cds_%s_%d-", aoi, year))
    if (length(spent)) {
      freed <- sum(file.size(list.files(spent, recursive = TRUE, full.names = TRUE)),
                   na.rm = TRUE)
      unlink(spent, recursive = TRUE)
      cat(sprintf("Discarded %d hourly month(s) for %d, freeing %.1f GB\n",
                  length(spent), year, freed / 1073741824))
    }
  }

  clim
}

# One year of ERA5 -> one parquet, same schema and units as ML-sample-gen.R's
# EC-Earth3 output. `statics` carries the admin units and the cropland weights,
# which depend on the ERA5 grid and so are built once by main() below.
era5_year_to_parquet <- function(year, statics, out_dir) {
  out_file <- file.path(out_dir, sprintf("climate_%s_%s_fy0_s%d-%s.parquet",
                                         era5_aoi_tag, era5_model, year, era5_member))
  if (file.exists(out_file)) {
    cat("Output already exists, skipping:", basename(out_file), "\n")
    return(invisible(out_file))
  }

  clim <- era5_year_rasters(era5_aoi, year, era5_cache_dir)

  # Same cropland mask the EC-Earth3 path applies, on the ERA5 grid.
  masked <- lapply(clim[ERA5_VARS], function(r) r * statics$harv_mask)
  rm(clim); gc()

  tasmax_daily  <- extract_daily_by_county(masked$tasmax,  "tasmax",  statics$sf,
                                            statics$harv_extracted)
  tasmin_daily  <- extract_daily_by_county(masked$tasmin,  "tasmin",  statics$sf,
                                            statics$harv_extracted)
  hursmin_daily <- extract_daily_by_county(masked$hursmin, "hursmin", statics$sf,
                                            statics$harv_extracted,
                                            unit_offset = 0, col_suffix = "")
  pr_daily      <- extract_daily_by_county(masked$pr,      "pr",      statics$sf,
                                            statics$harv_extracted,
                                            unit_offset = 0, col_suffix = "_mm",
                                            scale_factor = 86400)
  rm(masked); gc()

  # ERA5 is observed: no initialization year, no lead time. Fill the hindcast
  # columns with the degenerate case so the schema still matches.
  ts_data <- tasmax_daily %>%
    left_join(tasmin_daily  %>% select(adm2_id, date, tasmin_C), by = c("adm2_id", "date")) %>%
    left_join(hursmin_daily %>% select(adm2_id, date, hursmin),  by = c("adm2_id", "date")) %>%
    left_join(pr_daily      %>% select(adm2_id, date, pr_mm),    by = c("adm2_id", "date")) %>%
    mutate(year_month = format(date, "%Y-%m"), hindcast_start = year,
           forecast_year = 0L, target_year = year) %>%
    select(adm2_id, adm2, adm1, hindcast_start, forecast_year, target_year,
           date, doy, year_month, tasmax_C, tasmin_C, hursmin, pr_mm)

  cat("\nERA5 climate dataset - year", year, "\n")
  cat("  Rows (admin unit x day):", nrow(ts_data), "\n")
  cat("  Unique admin units:     ", n_distinct(ts_data$adm2_id), "\n")
  cat("  Date range:             ", format(range(ts_data$date)), "\n")
  cat("  tasmax_C range:         ", round(range(ts_data$tasmax_C, na.rm = TRUE), 2), "\n")
  cat("  tasmin_C range:         ", round(range(ts_data$tasmin_C, na.rm = TRUE), 2), "\n")
  cat("  hursmin range:          ", round(range(ts_data$hursmin,  na.rm = TRUE), 2), "\n")
  cat("  pr_mm range:            ", round(range(ts_data$pr_mm,    na.rm = TRUE), 3), "\n")
  cat("  Rows with any NA:       ", sum(!complete.cases(ts_data)), "\n")

  write_parquet(ts_data, out_file)
  cat("Written:", out_file, "\n")
  invisible(out_file)
}


# ===== MAIN =====

era5_main <- function() {
  cat("\nERA5 -> admin-2 samples\n")
  cat("  AOI:        ", era5_aoi, "(", era5_aoi_tag, ")\n")
  cat("  Years:      ", paste(era5_years, collapse = ", "), "\n")
  cat("  Source:     ", era5_source, "\n")
  cat("  Output dir: ", era5_out_dir, "\n\n")

  if (!dir.exists(era5_out_dir)) dir.create(era5_out_dir, recursive = TRUE)

  region <- load_us_boundaries(county_shp)
  cat("Admin units (adm2):", nrow(region$sf), "\n")

  # The cropland weights are tied to the climate grid, so they are built from an
  # ERA5 layer rather than an EC-Earth3 one -- ERA5's 0.25 deg is about three
  # times finer, so these weights are not interchangeable with the EC-Earth3
  # run's. The first year is fetched here and reused below from the cache.
  ref_year <- era5_years[1]
  ref_rast <- era5_year_rasters(era5_aoi, ref_year, era5_cache_dir)$tasmax[[1]]

  harv_frac_r <- load_harvested_area_raster(harv_tif, region$vect, ref_rast)

  harv_extracted <- terra::extract(harv_frac_r, region$vect,
                                    weights = TRUE, exact = TRUE, cells = TRUE)
  names(harv_extracted)[2] <- "cropland_frac"
  harv_extracted$combined_weight <- harv_extracted$cropland_frac * harv_extracted$weight

  statics <- list(sf = region$sf, harv_extracted = harv_extracted,
                  harv_mask = ifel(harv_frac_r > 0, 1, NA))
  rm(ref_rast); gc()

  for (y in era5_years) era5_year_to_parquet(y, statics, era5_out_dir)

  cat("\nAll years processed.\n")
}

if (sys.nframe() == 0L) era5_main()
