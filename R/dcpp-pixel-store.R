# Stage 1 for the DCPP hindcasts: EC-Earth3 daily NetCDFs -> per-pixel daily
# cubes + weight table.
#
# A front-end over pixel-store.R, the twin of era5-pixel-store.R. Everything
# after the daily rasters is shared with the reanalysis path, which is the whole
# point: a comparison between a hindcast and ERA5 should measure forecast skill,
# not a difference in how the two were processed.
#
# WHAT IS DIFFERENT FROM ERA5, AND UNAVOIDABLY SO
#
#   Already daily.  The hindcasts are published as daily tasmax/tasmin/hursmin/pr,
#                   so there is no hourly reduction here and no day-boundary
#                   choice to make -- the model's own daily fields are taken as
#                   given. ERA5's --local-time has no counterpart; leave ERA5 on
#                   UTC days when comparing, or the two differ by a shift the
#                   hindcast cannot match.
#   Coarser grid.   EC-Earth3 is roughly three times coarser than ERA5's 0.25
#                   deg, so its weight table is built on its own grid and the
#                   two are not interchangeable. Each source keeps its own
#                   grid/ directory; mixing them is a missing-file error rather
#                   than a silently wrong answer.
#   hursmin.        The published hursmin is relative humidity derived from
#                   daily-mean temperature and dewpoint, whereas ERA5's is a
#                   true daily minimum of hourly RH. Same column name, different
#                   quantity. Nothing here can fix that; it is a property of the
#                   input and belongs in a methods note.
#   Nov-Oct years.  A file for initialisation year Y at forecast year F spans
#                   1 Nov (Y+F-1) to 31 Oct (Y+F), not a calendar year. Cubes
#                   are named by initialisation year with an s prefix, matching
#                   what python/era5_features.py expects.
#
#   Rscript R/dcpp-pixel-store.R --aoi=CONUS 1982-2018
#   Rscript R/dcpp-pixel-store.R --aoi=ZM --forecast-year=2 1990-2010
#   Rscript R/dcpp-pixel-store.R --aoi=CONUS --check
#   Rscript R/dcpp-pixel-store.R --aoi=CONUS --grid-only 1982
#
# LAYOUT under {out_dir}  (default {AGIMPACTS_ROOT}/{AOI}_processed_data/{model})
#
#   pixel_daily/  pixel_{model}_{AOI}_s{year}.nc
#   grid/         weights_{AOI}.parquet, cells_{AOI}.parquet, grid_{AOI}.json

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

# Brings the AOI registry, the argument helpers, the shared stage-1 engine, and
# -- via ML-sample-gen.R -- the DCPP NetCDF readers nc_path(), read_nc_region()
# and arrays_to_rast(), plus the boundary loaders both sources share.
source(file.path(.this_dir, "era5-aoi.R"))


# ===== CONFIG =====

dcpp_aoi_tag <- "CONUS"
dcpp_years   <- 1982        # initialisation (s) years
dcpp_fy      <- 1L          # forecast year: lead time in years
dcpp_model   <- "EC-Earth3"
dcpp_member  <- "r1i4p1f1"

.args        <- commandArgs(trailingOnly = TRUE)
.check_only  <- "--check"     %in% .args
.grid_only   <- "--grid-only" %in% .args
.overwrite   <- "--overwrite" %in% .args
.args        <- setdiff(.args, c("--check", "--grid-only", "--overwrite"))
dcpp_aoi_tag <- era5_take_flag("aoi", dcpp_aoi_tag)
dcpp_model   <- era5_take_flag("model", dcpp_model)
dcpp_member  <- era5_take_flag("member", dcpp_member)
.fy_flag     <- era5_take_flag("forecast-year", "")
.out_flag    <- era5_take_flag("out", "")
.dir_flag    <- era5_take_flag("data-dir", "")
if (nzchar(.fy_flag)) dcpp_fy <- as.integer(.fy_flag)
if (length(.args))    dcpp_years <- era5_parse_years(.args)

if (!dcpp_aoi_tag %in% names(ERA5_AOIS)) {
  stop("--aoi must be one of: ", paste(names(ERA5_AOIS), collapse = ", "),
       "; got '", dcpp_aoi_tag, "'")
}
if (dcpp_fy < 1L || dcpp_fy > 10L) stop("--forecast-year must be between 1 and 10")

dcpp_cfg <- ERA5_AOIS[[dcpp_aoi_tag]]
dcpp_key <- dcpp_cfg$era5_key

dcpp_out_dir  <- era5_resolve_out_dir(dcpp_aoi_tag, .out_flag, dcpp_model)
dcpp_nc_dir   <- file.path(dcpp_out_dir, "pixel_daily")
dcpp_grid_dir <- file.path(dcpp_out_dir, "grid")

# ML-sample-gen.R's readers take these from the global environment, so they are
# set here rather than threaded through. The window comes from the one AOI
# registry both sources use -- AOI_BOUNDS in era5.R -- converted to the NetCDF's
# 0-360 longitude space and padded by about two EC-Earth3 cells so edge polygons
# keep full cell overlap. Deriving it means a new AOI is usable by both sources
# the moment it enters the registry, with no second list to keep in step.
local({
  pad <- 1.5
  b   <- AOI_BOUNDS[[dcpp_key]]
  if (is.null(b)) stop("no AOI_BOUNDS entry for '", dcpp_key, "'")
  lon <- c(b$lon_min - pad, b$lon_max + pad) %% 360
  lat <- c(max(b$lat_min - pad, -90), min(b$lat_max + pad, 90))
  # lon_range[1] > lon_range[2] marks a window straddling zero, which
  # read_nc_region() reads in two pieces -- the Burkina Faso case.
  lon_range     <<- lon
  lat_range     <<- lat
  climate_model <<- dcpp_model
  member        <<- dcpp_member
  forecast_year <<- dcpp_fy
  if (nzchar(.dir_flag)) day_data_dir <<- .dir_flag
})


# ===== MAIN =====

# One initialisation year's four daily variables, on the model's own grid.
dcpp_year_rasters <- function(hstart) {
  cat(sprintf("Reading %s s%d FY%d (%s) ...\n", dcpp_model, hstart, dcpp_fy, dcpp_member))
  out <- lapply(ERA5_VARS, function(v) {
    r <- arrays_to_rast(read_nc_region(nc_path(v, hstart, dcpp_fy), v))
    cat(sprintf("  %-8s %3d daily layers  %s\n", v, nlyr(r),
                paste(format(range(time(r))), collapse = " -> ")))
    r
  })
  names(out) <- ERA5_VARS
  # The four variables must share one time axis, or the cube's days would not
  # line up across variables and nothing downstream would notice.
  nd <- vapply(out, nlyr, numeric(1))
  if (length(unique(nd)) != 1L) {
    stop("variables disagree on day count for s", hstart, ": ",
         paste(sprintf("%s=%d", names(nd), nd), collapse = ", "), call. = FALSE)
  }
  out
}

dcpp_preflight <- function() {
  extra <- character(0)
  # Listable, not merely present: a network share can be mounted and still
  # refuse to be read, and dir.exists() says nothing about that. Checked by
  # actually listing it, the same way the output directory is checked by
  # actually writing to it.
  readable <- dir.exists(day_data_dir) &&
    !inherits(try(length(list.files(day_data_dir)), silent = TRUE), "try-error") &&
    length(list.files(day_data_dir)) > 0
  if (!readable) {
    extra <- c(extra,
      sprintf("DCPP data directory missing, empty or unreadable: %s", day_data_dir),
      "  (mount the CRL_storage share and check it can be listed, or pass --data-dir=)")
  }
  pixel_preflight(dcpp_cfg, dcpp_out_dir, source = "none", extra_problems = extra)
  cat(sprintf("Preflight OK. %d initialisation year(s), FY%d, window lon %.1f-%.1f lat %.1f-%.1f\n",
              length(dcpp_years), dcpp_fy, lon_range[1], lon_range[2],
              lat_range[1], lat_range[2]))
}

# --check: prove the setup in seconds. The expensive failure here is not a
# credential but a missing file for one initialisation year, so the first year's
# four paths are resolved without reading them.
dcpp_check <- function() {
  dcpp_preflight()
  cat("Resolving NetCDF paths for s", dcpp_years[1], " FY", dcpp_fy, ":\n", sep = "")
  for (v in ERA5_VARS) {
    cat(sprintf("  %-8s %s\n", v, basename(nc_path(v, dcpp_years[1], dcpp_fy))))
  }
  cat("Loading admin units ... ")
  cat(nrow(dcpp_cfg$loader()$sf), "units OK\n")
  cat("\nAll checks passed. Ready to run.\n")
}

dcpp_pixel_main <- function() {
  dcpp_preflight()
  cat("  Model:      ", dcpp_model, "member", dcpp_member, "FY", dcpp_fy, "\n")
  cat("  Output dir: ", dcpp_out_dir, "\n")

  pixel_store_run(
    aoi_tag    = dcpp_aoi_tag,
    source_tag = dcpp_model,
    years      = dcpp_years,
    loader     = dcpp_cfg$loader,
    get_year   = dcpp_year_rasters,
    cube_path  = function(y) file.path(dcpp_nc_dir,
                   sprintf("pixel_%s_%s_s%d.nc", dcpp_model, dcpp_aoi_tag, y)),
    grid_dir   = dcpp_grid_dir,
    overwrite  = .overwrite,
    grid_only  = .grid_only,
    # The hindcasts are published as daily fields, so there is no shift to make.
    utc_offset = 0L,
    extra_manifest = list(model = dcpp_model, member = dcpp_member,
                          forecast_year = dcpp_fy,
                          lon_range = lon_range, lat_range = lat_range)
  )
}

if (sys.nframe() == 0L) {
  withCallingHandlers(
    tryCatch({
      if (.check_only) dcpp_check() else dcpp_pixel_main()
    }, error = function(e) pixel_die(e, dcpp_out_dir)),
    condition = function(c) {
      if (inherits(c, c("message", "warning", "restart"))) return(invisible(NULL))
      if (!inherits(c, "error")) pixel_die(c, dcpp_out_dir)
    })
}
