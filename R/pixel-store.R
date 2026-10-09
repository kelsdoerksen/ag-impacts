# Stage 1, shared: daily gridded climate -> per-pixel daily cubes, plus the
# pixel/admin-2 cropland weight table that turns them into admin-2 numbers.
#
# WHY PER PIXEL
#
# Aggregating first and computing features second is only correct for features
# linear in the inputs, and the interesting ones are not:
#
#   mean_i( max(0, T_i - Tbase) )  !=  max(0, mean_i(T_i) - Tbase)
#
# Averaging first clips the tails, so a unit holding both a cool and a hot pixel
# loses the heat the hot pixel actually accumulated. Degree days, extreme degree
# days, dry-day counts, VPD and every threshold feature are affected. So this
# stage stops at the pixel, and python/era5_features.py computes features there
# before taking the cropland-area-weighted mean.
#
# WHY SHARED
#
# The reanalysis and the hindcasts must be processed identically or a comparison
# between them measures the processing as much as the forecast. Everything that
# could drift -- the grid identity check, the weight table, the unit conversion,
# the cube layout, the resume logic -- lives here once and is used by both:
#
#   era5-pixel-store.R   ERA5 reanalysis, hourly -> daily reduced here
#   dcpp-pixel-store.R   EC-Earth3 DCPP hindcasts, already daily
#
# A front-end supplies three things and nothing else: how to get one year's
# daily rasters, where its cubes live, and what to call them.

library(arrow)


# ===== GRID IDENTITY =====

# The weight table joins to the cubes by grid position, so every year in a
# series must sit on exactly the same grid. It always should, but "should" is
# not worth betting a multi-decade series on: a changed bounding box or a
# re-derived registry entry would shift the grid and the join would then be
# silently wrong rather than loudly broken. So the geometry is recorded once and
# checked against every year.
pixel_grid_signature <- function(r) {
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

pixel_assert_grid <- function(r, expected, what, aoi_tag) {
  got <- pixel_grid_signature(r)
  bad <- names(expected)[vapply(names(expected), function(k) {
    !isTRUE(all.equal(expected[[k]], got[[k]], tolerance = 1e-8))
  }, logical(1))]
  if (length(bad)) {
    stop(sprintf("grid mismatch in %s: %s differ from grid_%s.json.\n",
                 what, paste(bad, collapse = ", "), aoi_tag),
         "The weight table is keyed on grid position, so joining these would be ",
         "silently wrong. Either the AOI bounds changed since the grid was built, ",
         "or this cache belongs to a different AOI or a different climate source. ",
         "Rebuild with --grid-only --overwrite if the change was intended.",
         call. = FALSE)
  }
  invisible(TRUE)
}


# ===== WEIGHT TABLE =====

# One row per (admin unit, pixel it overlaps), carrying both weighting terms
# separately so the feature stage can choose:
#
#   coverage_frac    how much of the pixel falls inside the admin polygon (0-1),
#                    from terra's exact polygon/cell intersection
#   cropland_frac    the harvested-area fraction of that pixel
#   combined_weight  the product -- the cropland-area weight
#
# Keeping them apart means a plain area-weighted mean is still available from the
# same table, which matters while a feature set is being worked out.
#
# Built once per AOI *per climate source*, never once per year: it depends on the
# grid, the admin polygons and the cropland raster, none of which vary by year --
# but the grid does vary by source, so ERA5 weights are meaningless on the
# coarser EC-Earth3 grid and each source keeps its own.
pixel_build_grid <- function(loader, aoi_tag, ref_rast, grid_dir, source_tag,
                             overwrite = FALSE, extra = list()) {
  w_path <- file.path(grid_dir, sprintf("weights_%s.parquet", aoi_tag))
  c_path <- file.path(grid_dir, sprintf("cells_%s.parquet",   aoi_tag))
  m_path <- file.path(grid_dir, sprintf("grid_%s.json",       aoi_tag))

  if (all(file.exists(c(w_path, c_path, m_path))) && !overwrite) {
    man <- jsonlite::read_json(m_path, simplifyVector = TRUE)
    cat("Grid artifacts already built (", man$n_pairs, "pixel-unit pairs ) in", grid_dir, "\n")
    pixel_assert_grid(ref_rast, man$grid, "the cached daily grids", aoi_tag)
    return(man)
  }

  if (!dir.exists(grid_dir)) dir.create(grid_dir, recursive = TRUE)

  region <- loader()
  cat("Admin units (adm2):", nrow(region$sf), "\n")

  # The cropland weights are tied to this source's climate grid, so they are
  # built from one of its own layers.
  harv_frac_r <- load_harvested_area_raster(harv_tif, region$vect, ref_rast)

  # weights + exact: terra returns ID, <value>, cell, weight.
  ex <- terra::extract(harv_frac_r, region$vect, weights = TRUE, exact = TRUE, cells = TRUE)
  names(ex)[2] <- "cropland_frac"
  # NA here means "no cropland raster value", which for weighting is the same as
  # none. Storing 0 keeps the parquet free of NAs in the weight columns.
  ex$cropland_frac[is.na(ex$cropland_frac)] <- 0
  # A polygon lying outside the raster yields an NA cell. Those rows cannot be
  # joined to anything, and left in they make row/col a nullable column that
  # arrives in pandas as float and cannot index an array. Dropped here so the
  # parquet's grid indices are honest integers.
  no_cell <- is.na(ex$cell)
  if (any(no_cell)) {
    cat("  dropping", sum(no_cell), "pixel-unit row(s) with no grid cell",
        "(polygon outside the climate grid)\n")
    ex <- ex[!no_cell, , drop = FALSE]
  }

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

  # Units with no cropland produce NA for every cropland-weighted feature. Not
  # an error, but much better discovered now than as a column of NAs later.
  tot   <- tapply(weights$combined_weight, weights$adm2_id, sum, na.rm = TRUE)
  empty <- names(tot)[is.na(tot) | tot <= 0]

  # A unit reaching past the edge of the climate box silently gets only the
  # pixels inside it. Registry bounding boxes are padded around the polygons
  # that have yield data, which is not always the whole shapefile, so a unit
  # outside that subset can sit partly outside the grid.
  g <- ext(ref_rast)
  clipped <- vapply(seq_len(nrow(region$sf)), function(i) {
    e <- ext(region$vect[i])
    xmin(e) < xmin(g) || xmax(e) > xmax(g) || ymin(e) < ymin(g) || ymax(e) > ymax(g)
  }, logical(1))

  write_parquet(weights, w_path)
  write_parquet(cells,   c_path)

  man <- c(list(
    aoi_tag         = aoi_tag,
    source          = source_tag,
    grid            = pixel_grid_signature(ref_rast),
    n_admin_units   = nrow(region$sf),
    n_pairs         = nrow(weights),
    n_cells_used    = nrow(cells),
    units_no_crop   = length(empty),
    units_clipped   = sum(clipped),
    vars            = unname(vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$name, "")),
    var_units       = unname(vapply(ERA5_VARS, function(v) ERA5_UNIT_CONV[[v]]$unit, "")),
    cropland_raster = harv_tif,
    weights_file    = basename(w_path),
    cells_file      = basename(c_path),
    created         = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    note = paste("combined_weight = cropland_frac * coverage_frac. Join to the",
                 "pixel_daily cubes on (lat, lon); row/col are terra's 1-based",
                 "grid indices, row 1 = northernmost, matching the cube's",
                 "descending latitude axis.")
  ), extra)
  # digits = NA for full double precision. write_json() defaults to 4 decimal
  # places, which silently truncated the grid signature: 20.039062 was stored as
  # 20.0391, and pixel_assert_grid() compares at 1e-8, so every RESUMED run
  # failed with a grid mismatch against its own manifest. ERA5 escaped it only
  # because its extents are exact multiples of 0.25; EC-Earth3's 0.7031 deg grid
  # does not survive four decimals.
  jsonlite::write_json(man, m_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)

  cat("\nGrid artifacts written to", grid_dir, "\n")
  cat("  pixel-unit pairs: ", nrow(weights), "\n")
  cat("  distinct pixels:  ", nrow(cells), "of", ncell(ref_rast), "in the AOI grid\n")
  cat("  units with no cropland:", length(empty),
      if (length(empty)) sprintf("(e.g. %s)", paste(head(empty, 5), collapse = ", ")) else "", "\n")
  if (any(clipped)) {
    cat(sprintf("  ! %d unit(s) extend past the climate box and are clipped to it: %s\n",
                sum(clipped),
                paste(head(as.character(region$sf$adm2_id[clipped]), 5), collapse = ", ")))
  }
  man
}


# ===== PER-PIXEL DAILY CUBE =====

# Is this a complete cube? Asked through ncdf4 rather than terra/GDAL, so it
# answers the same way on a machine whose GDAL cannot read netCDF at all.
pixel_nc_is_complete <- function(path, varnames, ndays) {
  if (!file.exists(path) || !requireNamespace("ncdf4", quietly = TRUE)) return(FALSE)
  nc <- tryCatch(ncdf4::nc_open(path), error = function(e) NULL)
  if (is.null(nc)) return(FALSE)
  on.exit(ncdf4::nc_close(nc), add = TRUE)
  all(varnames %in% names(nc$var)) &&
    !is.null(nc$dim[["time"]]) && nc$dim[["time"]]$len == ndays
}

# The time convention a cube was written on, from its global attributes. A cube
# with no such attribute predates local time and is therefore UTC.
pixel_nc_utc_offset <- function(path) {
  if (!file.exists(path) || !requireNamespace("ncdf4", quietly = TRUE)) return(NA_integer_)
  nc <- tryCatch(ncdf4::nc_open(path), error = function(e) NULL)
  if (is.null(nc)) return(NA_integer_)
  on.exit(ncdf4::nc_close(nc), add = TRUE)
  a <- tryCatch(ncdf4::ncatt_get(nc, 0, "era5_utc_offset"),
                error = function(e) list(hasatt = FALSE))
  if (isTRUE(a$hasatt)) as.integer(a$value) else 0L
}

# Is this year already written, and on this run's time convention? Used to
# decide whether to read a year at all. Without it a resumed run re-reads every
# year's input to discover it has nothing to write -- which for a local NetCDF
# archive costs nearly as much as the original pass.
pixel_year_done <- function(out_nc, utc_offset = 0L) {
  if (!file.exists(out_nc)) return(FALSE)
  have <- pixel_nc_utc_offset(out_nc)
  # A mismatch is NOT "done": it must reach pixel_write_year(), which raises
  # rather than letting two conventions share one pixel_daily.
  is.na(have) || have == as.integer(utc_offset)
}

# One year of daily rasters -> one NetCDF cube, in analysis units.
#
# Both sources hand over the same four variables in the same raw units (Kelvin,
# percent, precip flux), so the conversion is defined once in ERA5_UNIT_CONV and
# applied here, on the way out. Nothing downstream has to know which source the
# numbers came from.
pixel_write_year <- function(clim, out_nc, source_tag, overwrite = FALSE, utc_offset = 0L) {
  if (file.exists(out_nc) && !overwrite) {
    # The cube filename carries no time tag, because the Python feature stage
    # looks it up by name. So the convention is checked instead of encoded: a
    # cube built on a different one is a hard error, never a silent skip.
    have <- pixel_nc_utc_offset(out_nc)
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
  nc_dir <- dirname(out_nc)
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

  part <- file.path(nc_dir, paste0(".", basename(out_nc), ".part.nc"))
  atts <- c(sprintf("era5_utc_offset=%d", as.integer(utc_offset)),
            sprintf("era5_time_convention=%s",
                    if (utc_offset == 0) "UTC day" else "local solar day"),
            sprintf("climate_source=%s", source_tag))

  # terra::writeCDF() writes through ncdf4 and then re-opens the file with sds()
  # purely to build its return value. That read goes through GDAL, which needs a
  # netCDF/HDF5 driver -- absent from some conda GDAL builds, where it reports
  # the file it has just written as unreadable. The write has already succeeded
  # by then, so verify with ncdf4 rather than letting a missing plugin end an
  # overnight run.
  ok <- tryCatch({
    writeCDF(d, filename = part, overwrite = TRUE, compression = 6, atts = atts)
    TRUE
  }, error = function(e) {
    if (pixel_nc_is_complete(part, names(layers), length(dates))) {
      cat("(terra could not re-open the file it wrote -- GDAL here has no netCDF\n",
          "   driver; verified with ncdf4 instead)\n", sep = "")
      TRUE
    } else {
      stop("writeCDF failed for ", basename(out_nc), ": ", conditionMessage(e), call. = FALSE)
    }
  })
  stopifnot(isTRUE(ok))
  file.rename(part, out_nc)

  cat(sprintf("Wrote %s  (%d days x %d pixels x %d vars, %.1f MB)\n",
              basename(out_nc), length(dates), ncell(clim[[1]]), length(ERA5_VARS),
              file.size(out_nc) / 1048576))
  invisible(out_nc)
}


# ===== PREFLIGHT AND THE RUN LOOP =====

# R buffers stdout when it is redirected to a file, so an error raised deep in a
# long run can be lost entirely if the process ends before the buffer is
# flushed. Two independent records are written, because each can fail on its
# own: a file through its own connection, closed immediately, and the console
# copy that shows up in a driver log.
pixel_die <- function(e, out_dir) {
  msg <- tryCatch(conditionMessage(e), error = function(...) "<condition message unreadable>")
  cls <- tryCatch(paste(class(e), collapse = "/"), error = function(...) "?")
  try({
    con <- file(file.path(out_dir, "last_error.txt"), open = "wt")
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

# Everything that can be proved wrong before the first byte is fetched.
pixel_preflight <- function(cfg, out_dir, source = "none", extra_problems = character(0)) {
  problems <- c(era5_check_inputs(cfg, out_dir, source = source), extra_problems)
  if (length(problems)) {
    stop("preflight failed:\n  ", paste(problems, collapse = "\n  "), call. = FALSE)
  }
  invisible(TRUE)
}

# The run loop both front-ends share.
#
#   get_year(year)  -> named list of four daily SpatRasters in raw units
#   cube_path(year) -> where that year's cube belongs
pixel_store_run <- function(aoi_tag, source_tag, years, loader, get_year, cube_path,
                            grid_dir, overwrite = FALSE, grid_only = FALSE,
                            utc_offset = 0L, extra_manifest = list()) {
  cat("\n", source_tag, " -> per-pixel daily cubes\n", sep = "")
  cat("  AOI:        ", aoi_tag, "\n")
  cat("  Years:      ", paste(years, collapse = ", "), "\n")
  cat("  Day:        ", if (utc_offset == 0) "UTC" else
        sprintf("local solar (UTC%+d)", utc_offset), "\n\n")

  # Which years actually need work, decided before any input is read.
  todo <- if (overwrite) years else {
    years[!vapply(years, function(y) pixel_year_done(cube_path(y), utc_offset), logical(1))]
  }
  have_grid <- file.exists(file.path(grid_dir, sprintf("grid_%s.json", aoi_tag)))
  if (length(todo) < length(years)) {
    cat(sprintf("%d of %d year(s) already written; skipping those without reading them.\n",
                length(years) - length(todo), length(years)))
  }
  if (!length(todo) && have_grid && !grid_only) {
    cat("Nothing to do.\n")
    return(invisible(NULL))
  }

  # One year supplies the grid template the weight table is built on. Prefer a
  # year that needs writing anyway, so a rebuild of the grid alone does not pay
  # to read a year twice.
  first <- if (length(todo)) todo[1] else years[1]
  clim  <- get_year(first)
  man   <- pixel_build_grid(loader, aoi_tag, clim[[1]][[1]], grid_dir, source_tag,
                            overwrite = overwrite, extra = extra_manifest)

  if (grid_only) {
    cat("\n--grid-only: weight table built, no cubes written.\n")
    return(invisible(NULL))
  }

  if (first %in% todo) {
    pixel_write_year(clim, cube_path(first), source_tag, overwrite, utc_offset)
  }
  rm(clim); gc()

  for (y in setdiff(todo, first)) {
    clim <- get_year(y)
    pixel_assert_grid(clim[[1]][[1]], man$grid, sprintf("the %d daily grids", y), aoi_tag)
    pixel_write_year(clim, cube_path(y), source_tag, overwrite, utc_offset)
    rm(clim); gc()
  }

  cat("\nAll years written.\n")
  cat("Next:  python/era5_features.py --aoi=", aoi_tag,
      " --source=", source_tag, " --stage=features\n", sep = "")
}
