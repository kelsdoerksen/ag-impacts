# ERA5 -> per-pixel daily cubes, plus the pixel/admin-2 cropland weight table
# that turns them into admin-2 numbers later.
#
# WHY THIS EXISTS, AND HOW IT DIFFERS FROM era5-sample-gen.R
#
# era5-sample-gen.R aggregates first: it takes the cropland-weighted admin-2
# mean of the raw daily variables and writes that, so any feature computed
# downstream is a feature *of the admin-2 mean*. That is only correct for
# features that are linear in the inputs. Growing degree days are not:
#
#   mean(max(0, T - Tbase))  !=  max(0, mean(T) - Tbase)
#
# Averaging first flattens the tails, so a unit containing both a cool and a hot
# pixel loses the heat accumulation the hot pixel actually saw -- and the same
# argument applies to extreme degree days, dry-day counts, VPD and every other
# threshold or nonlinear feature. The causal-inference crop literature therefore
# computes the feature per pixel and only then takes the cropland-area-weighted
# mean over the unit. This script produces the inputs for that ordering.
#
# THE THREE STAGES
#
#   1. this script         daily grids -> per-pixel daily cube (NetCDF)
#                                      -> pixel/admin-2 weight table (parquet)
#   2. python/era5_features.py --stage=features
#                          per-pixel daily cube -> per-pixel daily features
#                          (GDD, EDD, VPD, ...) -- the part still being designed
#   3. python/era5_features.py --stage=aggregate
#                          per-pixel features + weights -> admin-2 parquet,
#                          cropland-area weighted, optionally accumulated over a
#                          crop-calendar season window
#
# Stage 1 is deliberately feature-free: it is the expensive, network-bound part,
# so it can run now and be reused unchanged however the feature set evolves.
#
#   Rscript R/era5-pixel-store.R --aoi=ZM 1980-2017
#   Rscript R/era5-pixel-store.R --aoi=ZM --check         # prove the setup works
#   Rscript R/era5-pixel-store.R --aoi=ZM --grid-only     # weights only, no cubes
#   Rscript R/era5-pixel-store.R --aoi=CONUS --source=cds 2000-2024
#   Rscript R/era5-pixel-store.R --aoi=BF --months-per-request=3 2001-2019
#
# LAYOUT under {out_dir}  (default {AGIMPACTS_ROOT}/{AOI}_processed_data/ERA5)
#
#   daily_grids/  GeoTIFF download cache, SHARED with era5-sample-gen.R. A year
#                 either script has already fetched costs nothing here.
#   cds_nc/       raw hourly months, discarded per year (resume aid only)
#   pixel_daily/  era5_pixel_{AOI}_{year}.nc     <- the per-pixel daily cube
#   grid/         weights_{AOI}.parquet          <- pixel x admin-2 x weight
#                 cells_{AOI}.parquet            <- the pixel set, with lon/lat
#                 grid_{AOI}.json                <- grid geometry + provenance
#
# The cube is NetCDF rather than parquet on purpose. The data is dense and
# gridded (time x lat x lon), so a long table would repeat a cell id and a date
# on every one of ~8.7M rows per CONUS year to store the same numbers; NetCDF
# stores the array. It is also what xarray opens natively, which is where the
# feature stage lives. This is not the netCDF that was rejected as too large
# earlier -- that was the raw *hourly* CDS download (~1.2 GB/year). A daily,
# four-variable, float32, deflate-compressed cube is ~50 MB per CONUS year and
# ~5 MB per Zambia year.
#
# The weight table stays parquet: it is genuinely tabular, has no time
# dimension, and is a join key on the pandas side.

# Directory holding this file, so the shared configuration below resolves
# regardless of the working directory. source() records the path on its own
# call frame, which covers being sourced; Rscript only reports it in --file=.
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

# The AOI registry, the daily-grid cache and the argument helpers, shared with
# era5-sample-gen.R. Sources era5.R and ML-sample-gen.R in turn.
source(file.path(.this_dir, "era5-aoi.R"))


# ===== CONFIG =====

era5_aoi_tag  <- "CONUS"    # which ERA5_AOIS entry to run; --aoi= overrides
era5_model    <- "ERA5"
era5_years    <- 2000
era5_source   <- "cds"      # "cds" (server-side subset, queued) or "arco" (no account)

# Keep the raw hourly months after a year has been reduced to daily grids? They
# are a resume aid, not an archive, and are by far the largest thing on disk.
era5_keep_hourly <- FALSE

.args        <- commandArgs(trailingOnly = TRUE)
.check_only  <- "--check"     %in% .args
.grid_only   <- "--grid-only" %in% .args
.overwrite   <- "--overwrite" %in% .args
.args        <- setdiff(.args, c("--check", "--grid-only", "--overwrite"))
era5_source  <- era5_take_flag("source", era5_source)
era5_aoi_tag <- era5_take_flag("aoi", era5_aoi_tag)
.out_flag    <- era5_take_flag("out", "")
# Months per CDS request. Unset means size it from the AOI: the queue wait is
# per request and dwarfs the work, so a small country asks for a whole year at
# once while CONUS still goes month by month to bound memory.
era5_months_per_request <- era5_take_flag("months-per-request", "")
era5_months_per_request <- if (nzchar(era5_months_per_request)) {
  as.integer(era5_months_per_request)
} else NULL

# Time convention for the daily reduction. ERA5 is hourly on UTC and the daily
# resample bins whatever axis it is given, so by default a "day" is a UTC day.
# --local-time shifts the hourly axis onto local solar time first, so tasmax and
# tasmin are the local day's extremes -- which is what they are meant to be, and
# what matters for degree days. Opt-in, because switching it on changes the
# numbers: years already downloaded stay UTC unless refetched.
#   --local-time        solar offset from the AOI's centre longitude
#   --utc-offset=N      an explicit whole-hour offset instead
.local_time  <- "--local-time" %in% .args
.args        <- setdiff(.args, "--local-time")
.tz_flag     <- era5_take_flag("utc-offset", "")
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


# ===== GRID IDENTITY =====

# The weight table joins to the cube by grid position, so every year in a series
# must sit on exactly the same grid. It always should -- era5.R asks CDS for a
# fixed bounding box at a fixed resolution -- but "should" is not a guarantee
# worth betting a 46-year series on: a changed AOI_BOUNDS entry or a re-derived
# cy-bench bbox would shift the grid, and the join would then be silently wrong
# rather than loudly broken. So the geometry is recorded once and checked
# against every year.
era5_grid_signature <- function(r) {
  e <- as.vector(ext(r))          # xmin, xmax, ymin, ymax
  list(
    xmin  = round(unname(e[1]), 6), xmax = round(unname(e[2]), 6),
    ymin  = round(unname(e[3]), 6), ymax = round(unname(e[4]), 6),
    res_x = round(res(r)[1], 8),    res_y = round(res(r)[2], 8),
    nrow  = as.integer(nrow(r)),    ncol = as.integer(ncol(r)),
    ncell = as.integer(ncell(r)),
    crs   = trimws(crs(r, proj = TRUE))
  )
}

era5_assert_grid <- function(r, expected, what) {
  got <- era5_grid_signature(r)
  bad <- names(expected)[vapply(names(expected), function(k) {
    !isTRUE(all.equal(expected[[k]], got[[k]], tolerance = 1e-8))
  }, logical(1))]
  if (length(bad)) {
    stop(sprintf(
      "grid mismatch in %s: %s differ from grid_%s.json.\n",
      what, paste(bad, collapse = ", "), era5_aoi_tag),
      "The weight table is keyed on grid position, so joining these would be ",
      "silently wrong. Either the AOI bounds changed since the grid was built, ",
      "or this cache belongs to a different AOI. Rebuild with --grid-only ",
      "--overwrite (and re-run the feature stage) if the change was intended.",
      call. = FALSE)
  }
  invisible(TRUE)
}


# ===== GRID ARTIFACTS =====

# The pixel/admin-2 weight table: one row per (admin unit, pixel it overlaps),
# carrying both weighting terms separately so the feature stage can choose.
#
#   coverage_frac    how much of the pixel falls inside the admin polygon (0-1),
#                    from terra's exact polygon/cell intersection
#   cropland_frac    the harvested-area fraction of that pixel
#   combined_weight  the product -- the cropland-area weight, and the one that
#                    reproduces what era5-sample-gen.R does today
#
# Keeping them apart means an unweighted (plain area) admin-2 mean is still
# available from the same table, which matters while the feature set is being
# worked out: some features want cropland weighting and some want the unit's
# whole-area climate.
#
# This is built once per AOI, not once per year: it depends on the grid, the
# admin polygons and the cropland raster, none of which vary by year.
era5_build_grid <- function(cfg, aoi_tag, ref_rast, grid_dir, overwrite = FALSE) {
  w_path <- file.path(grid_dir, sprintf("weights_%s.parquet", aoi_tag))
  c_path <- file.path(grid_dir, sprintf("cells_%s.parquet",   aoi_tag))
  m_path <- file.path(grid_dir, sprintf("grid_%s.json",       aoi_tag))

  if (all(file.exists(c(w_path, c_path, m_path))) && !overwrite) {
    man <- jsonlite::read_json(m_path, simplifyVector = TRUE)
    cat("Grid artifacts already built (", man$n_pairs, "pixel-unit pairs ) in", grid_dir, "\n")
    era5_assert_grid(ref_rast, man$grid, "the cached daily grids")
    return(man)
  }

  if (!dir.exists(grid_dir)) dir.create(grid_dir, recursive = TRUE)

  region <- cfg$loader()
  cat("Admin units (adm2):", nrow(region$sf), "\n")

  # The cropland weights are tied to the climate grid, so they are built from an
  # ERA5 layer rather than an EC-Earth3 one -- ERA5's 0.25 deg is about three
  # times finer, so these weights are not interchangeable with the EC-Earth3
  # run's.
  harv_frac_r <- load_harvested_area_raster(harv_tif, region$vect, ref_rast)

  # weights + exact: terra returns ID, <value>, cell, weight -- the same call
  # era5-sample-gen.R makes, so the two pipelines weight identically.
  ex <- terra::extract(harv_frac_r, region$vect, weights = TRUE, exact = TRUE, cells = TRUE)
  names(ex)[2] <- "cropland_frac"
  # NA here means "no cropland raster value", which for weighting purposes is
  # the same as none: era5-sample-gen.R drops NA weights, and a zero weight is
  # dropped by the same filter. Storing 0 keeps the parquet free of NAs.
  ex$cropland_frac[is.na(ex$cropland_frac)] <- 0

  xy <- terra::xyFromCell(ref_rast, ex$cell)
  rc <- terra::rowColFromCell(ref_rast, ex$cell)

  weights <- data.frame(
    adm2_id         = as.character(region$sf$adm2_id[ex$ID]),
    adm2            = as.character(region$sf$adm2[ex$ID]),
    adm1            = as.character(region$sf$adm1[ex$ID]),
    cell            = as.integer(ex$cell),
    row             = as.integer(rc[, 1]),
    col             = as.integer(rc[, 2]),
    lon             = as.numeric(xy[, 1]),
    lat             = as.numeric(xy[, 2]),
    coverage_frac   = as.numeric(ex$weight),
    cropland_frac   = as.numeric(ex$cropland_frac),
    combined_weight = as.numeric(ex$cropland_frac) * as.numeric(ex$weight),
    stringsAsFactors = FALSE
  )

  cells <- unique(weights[, c("cell", "row", "col", "lon", "lat")])
  cells <- cells[order(cells$cell), ]

  # Units with no cropland at all produce NA for every cropland-weighted feature
  # -- they are not an error, but they are the kind of thing that is much better
  # discovered now than as a column of NAs after a 46-year run.
  tot   <- tapply(weights$combined_weight, weights$adm2_id, sum, na.rm = TRUE)
  empty <- names(tot)[is.na(tot) | tot <= 0]

  # An admin unit reaching past the edge of the ERA5 box silently gets only the
  # pixels that fall inside it. The cy-bench bounding boxes are padded around the
  # polygons that have yield data, which is not always the whole shapefile
  # (Ethiopia ships 84 polygons but only 68 have maize), so a unit outside that
  # subset can sit partly outside the grid. Not necessarily a problem -- usually
  # it is a unit with no yield to model anyway -- but it should be said out loud
  # rather than discovered as an oddly low weight sum later.
  g <- ext(ref_rast)
  clipped <- vapply(seq_len(nrow(region$sf)), function(i) {
    e <- ext(region$vect[i])
    xmin(e) < xmin(g) || xmax(e) > xmax(g) || ymin(e) < ymin(g) || ymax(e) > ymax(g)
  }, logical(1))

  write_parquet(weights, w_path)
  write_parquet(cells,   c_path)

  man <- list(
    aoi_tag        = aoi_tag,
    era5_key       = cfg$era5_key,
    grid           = era5_grid_signature(ref_rast),
    n_admin_units  = nrow(region$sf),
    n_pairs        = nrow(weights),
    n_cells_used   = nrow(cells),
    units_no_crop  = length(empty),
    units_clipped  = sum(clipped),
    utc_offset     = as.integer(era5_utc_offset),
    vars           = unname(vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$name, "")),
    var_units      = unname(vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$unit, "")),
    cropland_raster = harv_tif,
    weights_file   = basename(w_path),
    cells_file     = basename(c_path),
    created        = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    note = paste("combined_weight = cropland_frac * coverage_frac. Join to the",
                 "pixel_daily cubes on (lat, lon); row/col are terra's 1-based",
                 "grid indices, row 1 = northernmost, matching the cube's",
                 "descending latitude axis.")
  )
  jsonlite::write_json(man, m_path, auto_unbox = TRUE, pretty = TRUE)

  cat("\nGrid artifacts written to", grid_dir, "\n")
  cat("  pixel-unit pairs: ", nrow(weights), "\n")
  cat("  distinct pixels:  ", nrow(cells), "of", ncell(ref_rast), "in the AOI grid\n")
  cat("  units with no cropland:", length(empty),
      if (length(empty)) sprintf("(e.g. %s)", paste(head(empty, 5), collapse = ", ")) else "", "\n")
  if (any(clipped)) {
    cat(sprintf("  ! %d unit(s) extend past the ERA5 box and are clipped to it: %s\n",
                sum(clipped),
                paste(head(as.character(region$sf$adm2_id[clipped]), 5), collapse = ", ")))
    cat("    (the AOI bbox comes from cybench-aois.json; rebuild it with",
        "tools/build-cybench-aois.R if these units matter)\n")
  }
  man
}


# ===== PER-PIXEL DAILY CUBE =====

# Is this a complete cube? Asked through ncdf4 rather than terra/GDAL, so it
# answers the same way on a machine whose GDAL cannot read netCDF at all.
era5_nc_is_complete <- function(path, varnames, ndays) {
  if (!file.exists(path) || !requireNamespace("ncdf4", quietly = TRUE)) return(FALSE)
  nc <- tryCatch(ncdf4::nc_open(path), error = function(e) NULL)
  if (is.null(nc)) return(FALSE)
  on.exit(ncdf4::nc_close(nc), add = TRUE)
  all(varnames %in% names(nc$var)) &&
    !is.null(nc$dim[["time"]]) && nc$dim[["time"]]$len == ndays
}

# One year of daily grids -> one NetCDF cube, in analysis units. The GeoTIFF
# cache stays in era5.R's raw units (Kelvin, precip flux) because that is what
# era5-sample-gen.R's extraction expects; the conversion happens here, once, on
# the way out, so the feature stage never has to think about it.
# The time convention a cube was written on, from its global attributes. A cube
# with no such attribute predates local time and is therefore UTC.
era5_nc_utc_offset <- function(path) {
  if (!file.exists(path) || !requireNamespace("ncdf4", quietly = TRUE)) return(NA_integer_)
  nc <- tryCatch(ncdf4::nc_open(path), error = function(e) NULL)
  if (is.null(nc)) return(NA_integer_)
  on.exit(ncdf4::nc_close(nc), add = TRUE)
  a <- tryCatch(ncdf4::ncatt_get(nc, 0, "era5_utc_offset"),
                error = function(e) list(hasatt = FALSE))
  if (isTRUE(a$hasatt)) as.integer(a$value) else 0L
}

era5_write_pixel_year <- function(clim, year, aoi_tag, nc_dir, overwrite = FALSE,
                                  utc_offset = 0L) {
  out_nc <- file.path(nc_dir, sprintf("era5_pixel_%s_%d.nc", aoi_tag, year))
  if (file.exists(out_nc) && !overwrite) {
    # The cube filename carries no time tag, because the Python feature stage
    # looks it up by name. So the convention is checked instead of encoded: a
    # cube built on a different one is a hard error, never a silent skip.
    have <- era5_nc_utc_offset(out_nc)
    if (!is.na(have) && have != as.integer(utc_offset)) {
      stop(sprintf(paste0("%s already exists but was built on UTC%+d, and this run is ",
                          "UTC%+d.\nMixing time conventions in one pixel_daily would be ",
                          "silent and wrong. Re-run with --overwrite to replace it, or ",
                          "point --out at a separate tree."),
                   basename(out_nc), have, as.integer(utc_offset)), call. = FALSE)
    }
    cat("Cube already exists, skipping:", basename(out_nc), "\n")
    return(invisible(out_nc))
  }
  if (!dir.exists(nc_dir)) dir.create(nc_dir, recursive = TRUE)

  dates <- time(clim[[ERA5_VARS[1]]])

  layers <- lapply(ERA5_VARS, function(v) {
    conv <- ERA5_UNIT_CONV[[v]]
    r <- clim[[v]] * conv$scale + conv$offset
    time(r) <- dates
    r
  })
  names(layers) <- vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$name, "")

  d <- terra::sds(layers)
  names(d)     <- names(layers)
  units(d)     <- unname(vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$unit, ""))
  longnames(d) <- c("daily maximum 2m temperature",
                    "daily minimum 2m temperature",
                    "daily minimum relative humidity",
                    "daily total precipitation")

  # Written to a scratch name first: an interrupted year would otherwise leave a
  # truncated .nc that looks complete to the skip check above. The .nc extension
  # is kept on the scratch name -- writeCDF picks its driver from it.
  part <- file.path(nc_dir, sprintf(".era5_pixel_%s_%d.part.nc", aoi_tag, year))
  atts <- c(sprintf("era5_utc_offset=%d", as.integer(utc_offset)),
            sprintf("era5_time_convention=%s",
                    if (utc_offset == 0) "UTC day" else "local solar day"))

  # terra::writeCDF() writes the file through ncdf4 and then re-opens it with
  # sds() purely to build its return value. That read goes through GDAL, which
  # needs a netCDF/HDF5 driver -- absent from some conda GDAL builds, where it
  # reports the file it has just written as "not recognized as being in a
  # supported file format". The write has already succeeded by then, so a
  # failure here is about the reader, not the file. Verify with ncdf4 (the
  # library that wrote it, and a hard dependency of writeCDF) rather than
  # letting a missing GDAL plugin end an overnight run.
  ok <- tryCatch({
    writeCDF(d, filename = part, overwrite = TRUE, compression = 6, atts = atts)
    TRUE
  }, error = function(e) {
    if (era5_nc_is_complete(part, names(layers), length(dates))) {
      cat("(terra could not re-open the file it wrote -- GDAL here has no netCDF\n",
          "   driver; verified with ncdf4 instead)\n", sep = "")
      TRUE
    } else {
      stop("writeCDF failed for ", basename(out_nc), ": ", conditionMessage(e),
           call. = FALSE)
    }
  })
  stopifnot(isTRUE(ok))
  file.rename(part, out_nc)

  cat(sprintf("Wrote %s  (%d days x %d pixels x %d vars, %.1f MB)\n",
              basename(out_nc), length(dates), ncell(clim[[1]]), length(ERA5_VARS),
              file.size(out_nc) / 1048576))
  invisible(out_nc)
}


# ===== MAIN =====

era5_preflight <- function() {
  problems <- era5_check_inputs(era5_cfg, era5_out_dir, era5_source)
  if (length(problems)) {
    stop("preflight failed:\n  ", paste(problems, collapse = "\n  "), call. = FALSE)
  }

  # Rough disk guidance rather than a hard gate, scaled off the AOI's actual
  # cell count -- a CONUS-shaped estimate is wildly wrong for a country the size
  # of Zambia. Two costs: the GeoTIFF download cache and the NetCDF cubes, which
  # hold the same numbers in different shapes and are both kept.
  box   <- AOI_BOUNDS[[era5_aoi]]
  ncell <- ((box$lon_max - box$lon_min) / 0.25) * ((box$lat_max - box$lat_min) / 0.25)
  mb_yr <- 4 * 366 * ncell * 4 / 2 / 1048576   # float32, ~2x from deflate
  cat(sprintf(paste0("Preflight OK. %d year(s), ~%.0f grid cells.\n",
                     "  daily grids ~%.0f MB/year, pixel cubes ~%.0f MB/year",
                     " -> ~%.1f GB total in %s\n"),
              length(era5_years), ncell, mb_yr, mb_yr,
              2 * mb_yr * length(era5_years) / 1024, era5_out_dir))
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
    cat(sprintf("OK, mean %.1f K\n", mean(ds[["2m_temperature"]]$values)))
  }
  # The boundary and cropland files are the other half of the run and are just
  # as capable of failing late, so load them here rather than trusting that they
  # exist -- a shapefile can be present and still unreadable.
  cat("Loading admin units ... ")
  region <- era5_cfg$loader()
  cat(nrow(region$sf), "units OK\n")
  cat("\nAll checks passed. Ready to run.\n")
}

era5_pixel_main <- function() {
  era5_preflight()

  cat("\nERA5 -> per-pixel daily cubes\n")
  cat("  AOI:        ", era5_aoi_tag, "(bbox", era5_aoi, ")\n")
  cat("  Years:      ", paste(era5_years, collapse = ", "), "\n")
  cat("  Source:     ", era5_source, "\n")
  cat("  Day:        ", if (era5_utc_offset == 0) "UTC" else
        sprintf("local solar (UTC%+d)", era5_utc_offset), "\n")
  cat("  Output dir: ", era5_out_dir, "\n\n")

  get_year <- function(y) {
    era5_year_rasters(era5_aoi, y, era5_cache_dir, aoi_tag = era5_aoi_tag,
                      source = era5_source, cds_dir = era5_cds_dir,
                      keep_hourly = era5_keep_hourly,
                      months_per_request = era5_months_per_request,
                      utc_offset = era5_utc_offset)
  }

  # The first year does double duty: it supplies the grid template the weight
  # table is built on, and it is the first cube written.
  first <- era5_years[1]
  clim  <- get_year(first)
  man   <- era5_build_grid(era5_cfg, era5_aoi_tag, clim[[1]][[1]], era5_grid_dir,
                           overwrite = .overwrite)

  if (.grid_only) {
    cat("\n--grid-only: weight table built, no cubes written.\n")
    return(invisible(NULL))
  }

  era5_write_pixel_year(clim, first, era5_aoi_tag, era5_nc_dir, overwrite = .overwrite,
                        utc_offset = era5_utc_offset)
  rm(clim); gc()

  for (y in era5_years[-1]) {
    clim <- get_year(y)
    era5_assert_grid(clim[[1]][[1]], man$grid, sprintf("the %d daily grids", y))
    era5_write_pixel_year(clim, y, era5_aoi_tag, era5_nc_dir, overwrite = .overwrite,
                          utc_offset = era5_utc_offset)
    rm(clim); gc()
  }

  cat("\nAll years written to", era5_nc_dir, "\n")
  cat("Next:  python/era5_features.py --aoi=", era5_aoi_tag, " --stage=features\n", sep = "")
}

# R buffers stdout when it is redirected to a file, so an error raised deep in a
# long run can be lost entirely if the process ends before the buffer is
# flushed -- which is exactly how this pipeline's first failures presented: a
# log that simply stops mid-request with no message. Catching it here and
# flushing explicitly means the reason always reaches the log, whatever it was.
.era5_die <- function(e) {
  msg <- tryCatch(conditionMessage(e), error = function(...) "<condition message unreadable>")
  cls <- tryCatch(paste(class(e), collapse = "/"), error = function(...) "?")

  # Two independent records, because each can fail on its own. The file is
  # written through its own connection and closed immediately, so it survives
  # both stdout buffering and a handler that dies partway; the console copy is
  # what shows up in the driver log when everything behaves. useBytes guards
  # against a message carrying bytes that would make a normal write throw --
  # which is the likeliest reason an earlier failure left no trace at all.
  try({
    con <- file(file.path(era5_out_dir, "last_error.txt"), open = "wt")
    writeLines(c(format(Sys.time()), paste("condition:", cls), msg), con, useBytes = TRUE)
    close(con)
  }, silent = TRUE)
  try({
    cat("\n", strrep("!", 70), "\nERROR [", cls, "]: ", msg, "\n",
        strrep("!", 70), "\n", sep = "")
    flush(stdout())
  }, silent = TRUE)
  quit(status = 1, save = "no")
}

if (sys.nframe() == 0L) {
  # `condition`, not just `error`: the last failure exited 1 without tripping an
  # error handler at all, so anything that can end the run -- including a
  # non-error condition -- gets recorded rather than vanishing.
  withCallingHandlers(
    tryCatch({
      if (.check_only) era5_check() else era5_pixel_main()
    }, error = .era5_die),
    condition = function(c) {
      if (inherits(c, c("message", "warning", "restart"))) return(invisible(NULL))
      if (!inherits(c, "error")) .era5_die(c)
    })
}
