# Stage 1 for ERA5: hourly reanalysis -> per-pixel daily cubes + weight table.
#
# A front-end over pixel-store.R, which holds everything this shares with
# dcpp-pixel-store.R. What is ERA5-specific and therefore lives here: where the
# hourly data comes from (CDS or ARCO), how it is reduced to daily, and the
# local-solar-time option. Everything after the daily rasters -- the grid check,
# the weight table, the cube layout, the resume logic -- is shared, so the
# reanalysis and the hindcasts cannot drift apart.
#
#   Rscript R/era5-pixel-store.R --aoi=ZM 1980-2017
#   Rscript R/era5-pixel-store.R --aoi=ZM --check          # prove the setup works
#   Rscript R/era5-pixel-store.R --aoi=ZM --grid-only 2001 # weights only
#   Rscript R/era5-pixel-store.R --aoi=US --source=arco 2000-2023
#   Rscript R/era5-pixel-store.R --aoi=BF --months-per-request=3 2001-2019
#   Rscript R/era5-pixel-store.R --aoi=ZM --local-time 2001-2017
#
# LAYOUT under {out_dir}  (default {AGIMPACTS_ROOT}/{AOI}_processed_data/ERA5)
#
#   daily_grids/  GeoTIFF download cache -- a year here costs nothing to redo
#   cds_nc/       raw hourly months, discarded per year (resume aid only)
#   pixel_daily/  era5_pixel_{AOI}_{year}.nc     <- the per-pixel daily cube
#   grid/         weights_{AOI}.parquet          <- pixel x admin-2 x weight
#                 cells_{AOI}.parquet, grid_{AOI}.json

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

# The AOI registry, the daily-grid cache, the argument helpers and the shared
# stage-1 engine. Sources era5.R and ML-sample-gen.R in turn.
source(file.path(.this_dir, "era5-aoi.R"))


# ===== CONFIG =====

era5_aoi_tag  <- "CONUS"
era5_model    <- "ERA5"
era5_years    <- 2000
era5_source   <- "cds"      # "cds" (server-side subset, queued) or "arco" (no account)

# Keep the raw hourly months after a year is reduced to daily grids? They are a
# resume aid, not an archive, and by far the largest thing on disk.
era5_keep_hourly <- FALSE

.args        <- commandArgs(trailingOnly = TRUE)
.check_only  <- "--check"      %in% .args
.grid_only   <- "--grid-only"  %in% .args
.overwrite   <- "--overwrite"  %in% .args
.local_time  <- "--local-time" %in% .args
.args        <- setdiff(.args, c("--check", "--grid-only", "--overwrite", "--local-time"))
era5_source  <- era5_take_flag("source", era5_source)
era5_aoi_tag <- era5_take_flag("aoi", era5_aoi_tag)
.out_flag    <- era5_take_flag("out", "")

# Months per CDS request. Unset means size it from the AOI: the queue wait is
# per request and dwarfs the work, so a small country asks for a whole year at
# once while a continental AOI still goes month by month to bound memory.
.mpr <- era5_take_flag("months-per-request", "")
era5_months_per_request <- if (nzchar(.mpr)) as.integer(.mpr) else NULL

# Daily reduction on UTC days (default) or local solar days. Opt-in, because
# switching it on changes the numbers: years already downloaded stay UTC unless
# refetched, and the daily_grids filenames carry the convention so the two
# cannot mix.
.tz_flag        <- era5_take_flag("utc-offset", "")
era5_utc_offset <- if (nzchar(.tz_flag)) as.integer(.tz_flag) else 0L

if (length(.args)) era5_years <- era5_parse_years(.args)

if (!era5_source %in% c("cds", "arco")) {
  stop("--source must be 'cds' or 'arco', got '", era5_source, "'")
}
if (!era5_aoi_tag %in% names(ERA5_AOIS)) {
  stop("--aoi must be one of: ", paste(names(ERA5_AOIS), collapse = ", "),
       "; got '", era5_aoi_tag, "'")
}

era5_cfg <- ERA5_AOIS[[era5_aoi_tag]]
era5_aoi <- era5_cfg$era5_key

if (.local_time) {
  if (nzchar(.tz_flag)) stop("use either --local-time or --utc-offset=, not both")
  era5_utc_offset <- era5_solar_offset(era5_aoi)
}
if (abs(era5_utc_offset) > 12L) stop("--utc-offset must be between -12 and 12")

era5_out_dir   <- era5_resolve_out_dir(era5_aoi_tag, .out_flag, era5_model)
era5_cache_dir <- file.path(era5_out_dir, "daily_grids")
era5_cds_dir   <- file.path(era5_out_dir, "cds_nc")
era5_nc_dir    <- file.path(era5_out_dir, "pixel_daily")
era5_grid_dir  <- file.path(era5_out_dir, "grid")


# ===== MAIN =====

era5_preflight <- function() {
  pixel_preflight(era5_cfg, era5_out_dir, source = era5_source)

  # Rough disk guidance rather than a hard gate, scaled off the AOI's actual
  # cell count. Two costs: the GeoTIFF download cache and the NetCDF cubes,
  # which hold the same numbers in different shapes and are both kept.
  box   <- AOI_BOUNDS[[era5_aoi]]
  ncell <- ((box$lon_max - box$lon_min) / 0.25) * ((box$lat_max - box$lat_min) / 0.25)
  mb_yr <- 4 * 366 * ncell * 4 / 2 / 1048576
  cat(sprintf(paste0("Preflight OK. %d year(s), ~%.0f grid cells.\n",
                     "  daily grids ~%.0f MB/year, pixel cubes ~%.0f MB/year",
                     " -> ~%.1f GB total in %s\n"),
              length(era5_years), ncell, mb_yr, mb_yr,
              2 * mb_yr * length(era5_years) / 1024, era5_out_dir))
}

# --check: everything the real run depends on, proved in seconds. Preflight
# covers configuration; this adds one live fetch, because a well-formed
# ~/.cdsapirc says nothing about whether the token is valid, the licence is
# accepted, or the machine can reach the service.
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
    cat(sprintf("OK, mean %.1f K\n", mean(ds[["2m_temperature"]]$values)))
  }
  cat("Loading admin units ... ")
  cat(nrow(era5_cfg$loader()$sf), "units OK\n")
  cat("\nAll checks passed. Ready to run.\n")
}

era5_pixel_main <- function() {
  era5_preflight()
  cat("  Source:     ", era5_source, "\n")
  cat("  Output dir: ", era5_out_dir, "\n")

  pixel_store_run(
    aoi_tag    = era5_aoi_tag,
    source_tag = era5_model,
    years      = era5_years,
    loader     = era5_cfg$loader,
    get_year   = function(y) era5_year_rasters(
                   era5_aoi, y, era5_cache_dir, aoi_tag = era5_aoi_tag,
                   source = era5_source, cds_dir = era5_cds_dir,
                   keep_hourly = era5_keep_hourly,
                   months_per_request = era5_months_per_request,
                   utc_offset = era5_utc_offset),
    cube_path  = function(y) file.path(era5_nc_dir,
                   sprintf("era5_pixel_%s_%d.nc", era5_aoi_tag, y)),
    grid_dir   = era5_grid_dir,
    overwrite  = .overwrite,
    grid_only  = .grid_only,
    utc_offset = era5_utc_offset,
    extra_manifest = list(utc_offset = as.integer(era5_utc_offset),
                          hourly_source = era5_source)
  )
}

if (sys.nframe() == 0L) {
  withCallingHandlers(
    tryCatch({
      if (.check_only) era5_check() else era5_pixel_main()
    }, error = function(e) pixel_die(e, era5_out_dir)),
    condition = function(c) {
      if (inherits(c, c("message", "warning", "restart"))) return(invisible(NULL))
      if (!inherits(c, "error")) pixel_die(c, era5_out_dir)
    })
}
