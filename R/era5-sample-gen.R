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

# Areas of interest. Each names the bounding box in era5.R's AOI_BOUNDS, the
# files it needs on disk, and how to load its admin-2 units -- the loaders come
# from ML-sample-gen.R, so the boundary handling is shared with the EC-Earth3
# path rather than reimplemented here.
#
# The key is also the AOI token in output filenames and the {key}_processed_data
# folder name, matching the convention the EC-Earth3 runs already use.
ERA5_AOIS <- list(
  CONUS = list(
    era5_key = "conus",
    inputs   = function() c(harv_tif, county_shp),
    loader   = function() load_us_boundaries(county_shp)
  )
)

# Every cy-bench country becomes an AOI automatically, from the same registry
# era5.R reads for its bounding boxes. Adding a country is a matter of
# regenerating the registry, not editing this file.
#
# They all share one loader: load_cybench_boundaries() (from ML-sample-gen.R)
# parses the {CC}-A2-{AA}{BB} adm_id scheme every cy-bench shapefile uses.
local({
  reg <- file.path(.this_dir, "cybench-aois.json")
  if (!file.exists(reg) || !requireNamespace("jsonlite", quietly = TRUE)) return(invisible(NULL))
  entries <- jsonlite::read_json(reg, simplifyVector = FALSE)
  for (cc in names(entries)) {
    if (!is.null(ERA5_AOIS[[cc]])) next
    local({
      code <- cc
      rel  <- entries[[code]]$shapefile
      ERA5_AOIS[[code]] <<- list(
        era5_key = code,
        inputs   = function() c(harv_tif, file.path(proc_root, rel)),
        loader   = function() load_cybench_boundaries(file.path(proc_root, rel))
      )
    })
  }
})

era5_aoi_tag  <- "CONUS"    # which ERA5_AOIS entry to run; --aoi= overrides
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
  # Parse first and complain at the top level of this function -- raising from
  # inside the lapply buries the message under R's "error in evaluating the
  # argument" chain, which is exactly when you least want to read one.
  parts <- lapply(x, function(a) suppressWarnings(as.integer(strsplit(a, "[-:]")[[1]])))
  bad_args <- x[vapply(parts, anyNA, logical(1))]
  if (length(bad_args)) {
    stop("cannot read a year from: ", paste(bad_args, collapse = ", "), call. = FALSE)
  }

  y <- sort(unique(unlist(lapply(parts, function(p) {
    if (length(p) == 2) seq(p[1], p[2]) else p
  }))))
  # Catch a typo before it becomes an overnight run that quietly does nothing,
  # or one that asks CDS for years ERA5 does not cover.
  bad <- y[y < 1940 | y > as.integer(format(Sys.Date(), "%Y"))]
  if (length(bad)) {
    stop("year(s) out of range for ERA5 (1940-present): ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  y
}

# Pull "--flag=value" off the argument list, leaving the years behind.
.take_flag <- function(flag, default) {
  hit <- grep(paste0("^--", flag, "="), .args, value = TRUE)
  if (!length(hit)) return(default)
  .args <<- setdiff(.args, hit)
  sub(paste0("^--", flag, "="), "", hit[1])
}

.args        <- commandArgs(trailingOnly = TRUE)
.check_only  <- "--check" %in% .args
.list_only   <- "--list-aois" %in% .args
.args        <- setdiff(.args, c("--check", "--list-aois"))
era5_source  <- .take_flag("source", era5_source)
era5_aoi_tag <- .take_flag("aoi", era5_aoi_tag)
.out_flag    <- .take_flag("out", "")
if (length(.args)) era5_years <- .parse_years(.args)

if (!era5_source %in% c("cds", "arco")) {
  stop("--source must be 'cds' or 'arco', got '", era5_source, "'")
}
if (!era5_aoi_tag %in% names(ERA5_AOIS)) {
  stop("--aoi must be one of: ", paste(names(ERA5_AOIS), collapse = ", "),
       "; got '", era5_aoi_tag, "'")
}

era5_cfg <- ERA5_AOIS[[era5_aoi_tag]]
era5_aoi <- era5_cfg$era5_key

# Where everything is written. Three ways to set it, most specific first, so an
# overnight run can be pointed at a scratch disk without touching the file:
#   Rscript R/era5-sample-gen.R --out=/data/era5 1981-2025
#   ERA5_OUT_DIR=/data/era5 Rscript R/era5-sample-gen.R 1981-2025
#   (default) {AGIMPACTS_ROOT}/CONUS_processed_data/ERA5
# AGIMPACTS_ROOT alone moves the whole tree, inputs included; ERA5_OUT_DIR moves
# only the output, which is what you want when the inputs live on a shared
# volume and the output belongs somewhere with room.
era5_out_dir <- if (nzchar(.out_flag)) {
  .out_flag
} else {
  Sys.getenv("ERA5_OUT_DIR",
             file.path(proc_root, paste0(era5_aoi_tag, "_processed_data"), era5_model))
}
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
    # numeric(1), not integer(1): terra::nlyr() returns a double.
    if (all(vapply(cached, nlyr, numeric(1)) == length(dates))) {
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

# Fail before the first download rather than hours in. Everything checked here
# is something that would otherwise surface as an obscure error partway through
# an unattended overnight run.
era5_preflight <- function() {
  problems <- character(0)

  for (f in era5_cfg$inputs()) {
    if (!file.exists(f)) {
      problems <- c(problems, sprintf("missing input: %s", f))
    }
  }
  if (length(problems)) {
    problems <- c(problems,
      sprintf("  (paths derive from AGIMPACTS_ROOT, currently %s)", proc_root))
  }

  # Writability, checked by actually writing -- dir.exists() says nothing about
  # permission, and a read-only mount is a normal way for this to go wrong.
  if (!dir.exists(era5_out_dir)) {
    ok <- dir.create(era5_out_dir, recursive = TRUE, showWarnings = FALSE)
    if (!ok) problems <- c(problems, sprintf("cannot create output dir: %s", era5_out_dir))
  }
  if (dir.exists(era5_out_dir)) {
    probe <- file.path(era5_out_dir, ".write_probe")
    ok <- tryCatch({ writeLines("x", probe); unlink(probe); TRUE }, error = function(e) FALSE)
    if (!ok) problems <- c(problems, sprintf("output dir is not writable: %s", era5_out_dir))
  }

  # CDS credentials: a missing or malformed ~/.cdsapirc only shows up on the
  # first retrieve, which is after the boundaries and weights have been built.
  if (era5_source == "cds") {
    rc <- Sys.getenv("CDSAPI_RC", path.expand("~/.cdsapirc"))
    if (!file.exists(rc) && !nzchar(Sys.getenv("CDSAPI_KEY"))) {
      problems <- c(problems, sprintf("no CDS credentials: %s not found", rc))
    } else if (file.exists(rc)) {
      rc_txt <- readLines(rc, warn = FALSE)
      if (!any(grepl("^\\s*url\\s*:", rc_txt)) || !any(grepl("^\\s*key\\s*:", rc_txt))) {
        problems <- c(problems, sprintf("%s needs both a 'url:' and a 'key:' line", rc))
      }
    }
  }

  if (length(problems)) {
    stop("preflight failed:\n  ", paste(problems, collapse = "\n  "), call. = FALSE)
  }

  # Rough disk guidance rather than a hard gate. Scaled off the AOI's actual cell
  # count, because the same year costs ~70 MB for CONUS and ~6 MB for a country
  # the size of Zambia -- a CONUS-shaped estimate is wildly wrong for small AOIs.
  box   <- AOI_BOUNDS[[era5_aoi]]
  ncell <- ((box$lon_max - box$lon_min) / 0.25) * ((box$lat_max - box$lat_min) / 0.25)
  # 4 variables x 366 days x float32, halved for DEFLATE (measured ~2x on real fields).
  gb_yr <- 4 * 366 * ncell * 4 / 2 / 1073741824
  cat(sprintf("Preflight OK. %d year(s), %.0f grid cells; ~%.0f MB/year of daily grids, ~%.1f GB total in %s\n",
              length(era5_years), ncell, gb_yr * 1024, gb_yr * length(era5_years), era5_out_dir))
}

# --list-aois: what can be run, and what yield data each AOI has to pair with.
# Reads the registry rather than the loaded AOI table so it can show the crop
# coverage too, which is usually what decides whether an AOI is worth running.
era5_list_aois <- function() {
  reg <- file.path(.this_dir, "cybench-aois.json")
  cat(sprintf("%-6s %8s %7s  %-28s %s\n",
              "AOI", "cells", "units", "bbox lon/lat", "crops (years)"))
  cat(strrep("-", 100), "\n")
  cat(sprintf("%-6s %8s %7s  %-28s %s\n", "CONUS", "23735", "3108",
              "US counties (built-in)", "-- EC-Earth3 series AOI"))
  if (!file.exists(reg) || !requireNamespace("jsonlite", quietly = TRUE)) {
    cat("(cy-bench registry unavailable)\n"); return(invisible(NULL))
  }
  e <- jsonlite::read_json(reg, simplifyVector = FALSE)
  for (cc in names(e)) {
    x <- e[[cc]]
    crops <- paste(vapply(names(x$crops), function(cr) {
      k <- x$crops[[cr]]
      sprintf("%s %s %d-%d (%dy, %d units, e.g. %s)", cr, k$admin,
              k$year_min, k$year_max, k$n_years, k$units,
              if (is.null(k$adm_id_example)) "?" else k$adm_id_example)
    }, character(1)), collapse = "; ")
    cat(sprintf("%-6s %8d %7d  %-28s %s\n", cc, x$grid_cells, x$polygons,
                sprintf("[%.0f,%.0f] [%.0f,%.0f]", x$bbox$lon_min, x$bbox$lon_max,
                        x$bbox$lat_min, x$bbox$lat_max), crops))
  }
  cat("\nRun one with:  Rscript R/era5-sample-gen.R --aoi=ZM 1980-2017\n")
}

# --check: everything the real run depends on, proved in seconds. Preflight
# covers configuration; this adds one live fetch, because a well-formed
# ~/.cdsapirc still says nothing about whether the token is valid, the licence
# is accepted, or the machine can reach the service.
era5_check <- function() {
  era5_preflight()

  if (era5_source == "cds") {
    cat("Live CDS request (1 variable, 1 hour, 1x1 degree) ... ")
    f <- cds_probe()
    cat(sprintf("OK, %.1f KB\n", file.size(f) / 1024))
    unlink(f)
  } else {
    cat("Live ARCO read (1 variable, 1 hour) ... ")
    ds <- open_era5()
    ds <- crop_to_aoi(ds[["2m_temperature"]]$sel(
      time = py$slice("2000-06-01T00", "2000-06-01T00"))$to_dataset(), "test")
    v <- ds[["2m_temperature"]]$values
    cat(sprintf("OK, mean %.1f K\n", mean(v)))
  }

  cat("\nAll checks passed. Ready to run.\n")
}

era5_main <- function() {
  era5_preflight()

  cat("\nERA5 -> admin-2 samples\n")
  cat("  AOI:        ", era5_aoi_tag, "(bbox", era5_aoi, ")\n")
  cat("  Years:      ", paste(era5_years, collapse = ", "), "\n")
  cat("  Source:     ", era5_source, "\n")
  cat("  Output dir: ", era5_out_dir, "\n\n")

  region <- era5_cfg$loader()
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

if (sys.nframe() == 0L) {
  if (.list_only)        era5_list_aois()
  else if (.check_only)  era5_check()
  else                   era5_main()
}
