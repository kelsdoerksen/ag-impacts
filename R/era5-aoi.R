# Shared ERA5 configuration: the AOI registry, the daily-grid cache, and the
# argument helpers every ERA5 entry script needs.
#
# Two pipelines read this file, and they differ in where they aggregate:
#
#   era5-sample-gen.R   grids -> cropland-weighted admin-2 mean of the RAW
#                       variables -> parquet. Features are computed downstream,
#                       on the admin-2 means.
#   era5-pixel-store.R  grids -> per-pixel daily cube + a pixel/admin-2 weight
#                       table. Features are computed per pixel (see
#                       python/era5_features.py) and only then weighted, which
#                       is what the causal-model literature does and what a
#                       nonlinear feature like growing degree days requires.
#
# Both walk the same download path and share the same {out_dir}/daily_grids
# cache, so a year fetched by one is free for the other.

# Directory holding this file, so the sources below resolve regardless of the
# working directory. source() records the path on its own call frame; Rscript
# only reports it in --file=.
ERA5_R_DIR <- local({
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

source(file.path(ERA5_R_DIR, "era5.R"))          # era5_daily_rasters(), AOI_BOUNDS
# Defines load_us_boundaries(), load_cybench_boundaries(),
# load_harvested_area_raster() and extract_daily_by_county(), plus the config
# globals proc_root / harv_tif / county_shp. Its own batch loop is behind a
# sys.nframe() guard, so sourcing it only installs the functions.
source(file.path(ERA5_R_DIR, "ML-sample-gen.R"))

library(arrow)


# ===== AOI REGISTRY =====

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
  reg <- file.path(ERA5_R_DIR, "cybench-aois.json")
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

# The four daily variables both pipelines carry, in era5.R's raw units:
#   tasmax, tasmin  Kelvin
#   hursmin         percent (0-100)
#   pr              kg/m2/s-style flux (x86400 -> mm/day)
ERA5_VARS <- c("tasmax", "tasmin", "hursmin", "pr")

# Analysis units, applied by whichever stage first hands the data to something
# that is not a raster. Kept here so the two pipelines cannot drift apart on it.
ERA5_UNIT_CONV <- list(
  tasmax  = list(name = "tasmax_C",  scale = 1,     offset = -273.15, unit = "degC"),
  tasmin  = list(name = "tasmin_C",  scale = 1,     offset = -273.15, unit = "degC"),
  hursmin = list(name = "hursmin",   scale = 1,     offset = 0,       unit = "percent"),
  pr      = list(name = "pr_mm",     scale = 86400, offset = 0,       unit = "mm/day")
)


# ===== ARGUMENT HELPERS =====

# Years may be given as single years or as ranges, so a long series does not
# have to be spelled out: `1990-2020` and `1990:2020` both work.
era5_parse_years <- function(x) {
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

# Pull "--flag=value" off the argument list, leaving the years behind. Reads and
# writes `.args` in the calling script's global environment.
era5_take_flag <- function(flag, default) {
  hit <- grep(paste0("^--", flag, "="), .args, value = TRUE)
  if (!length(hit)) return(default)
  .args <<- setdiff(.args, hit)
  sub(paste0("^--", flag, "="), "", hit[1])
}

# Where everything for one AOI is written. Three ways to set it, most specific
# first, so an overnight run can be pointed at a scratch disk without touching
# any file:
#   --out=/data/era5   >   ERA5_OUT_DIR=/data/era5   >   {proc_root}/{AOI}_processed_data/ERA5
# AGIMPACTS_ROOT alone moves the whole tree, inputs included; ERA5_OUT_DIR moves
# only the output, which is what you want when the inputs live on a shared
# volume and the output belongs somewhere with room.
era5_resolve_out_dir <- function(aoi_tag, out_flag = "", model = "ERA5") {
  if (nzchar(out_flag)) return(out_flag)
  Sys.getenv("ERA5_OUT_DIR",
             file.path(proc_root, paste0(aoi_tag, "_processed_data"), model))
}


# ===== DAILY GRID CACHE =====

# Daily gridded rasters for one year, from the GeoTIFF cache if it is there and
# from the network (slow) if it is not. This is the one download path: both
# pipelines call it, so a year either has already been paid for or is paid for
# once.
#
# GeoTIFF carries no per-layer time, so it is reattached on read as the calendar
# year's days. era5_daily_rasters() resamples to "1D" over a whole calendar year,
# so a complete year is exactly that day sequence; a layer count that disagrees
# means the cached file is not a whole year and is treated as unusable rather
# than silently mislabelled.
era5_year_rasters <- function(aoi, year, cache_dir, aoi_tag = aoi,
                              source = "cds", cds_dir = NULL, keep_hourly = FALSE) {
  dates <- seq(as.Date(sprintf("%d-01-01", year)),
               as.Date(sprintf("%d-12-31", year)), by = "day")
  paths <- file.path(cache_dir, sprintf("era5_%s_%s_%d.tif", ERA5_VARS, aoi_tag, year))
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

  cat("Fetching ERA5", year, "for AOI", aoi, "via", source,
      "- one month at a time:\n")
  t0 <- Sys.time()
  clim <- era5_daily_rasters(aoi, year, source = source, cds_cache_dir = cds_dir)
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
  if (!keep_hourly && source == "cds" && !is.null(cds_dir)) {
    spent <- list.files(cds_dir, full.names = TRUE,
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


# ===== PREFLIGHT =====

# Everything that can be proved wrong before the first download: inputs on disk,
# an output directory that actually accepts a write, and CDS credentials that at
# least have the right shape. Returns the problems rather than raising, so a
# caller can add its own and report them all at once -- an unattended overnight
# run should fail on the first second, not four hours in.
era5_check_inputs <- function(cfg, out_dir, source = "cds") {
  problems <- character(0)

  for (f in cfg$inputs()) {
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
  if (!dir.exists(out_dir)) {
    ok <- dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    if (!ok) problems <- c(problems, sprintf("cannot create output dir: %s", out_dir))
  }
  if (dir.exists(out_dir)) {
    probe <- file.path(out_dir, ".write_probe")
    ok <- tryCatch({ writeLines("x", probe); unlink(probe); TRUE }, error = function(e) FALSE)
    if (!ok) problems <- c(problems, sprintf("output dir is not writable: %s", out_dir))
  }

  # CDS credentials: a missing or malformed ~/.cdsapirc only shows up on the
  # first retrieve, which is after the boundaries and weights have been built.
  if (source == "cds") {
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

  problems
}
