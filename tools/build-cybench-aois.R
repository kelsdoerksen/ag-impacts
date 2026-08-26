# Regenerate R/cybench-aois.json from the cy-bench polygon shapefiles.
#
# Run this when cy-bench adds countries or crops:
#   Rscript tools/build-cybench-aois.R
#
# The bounding boxes are derived from the shapefiles rather than typed by hand,
# so they cannot drift from the geometry they are meant to cover. Each box is
# padded by PAD degrees (~2 ERA5 cells) so units on the edge of a country still
# see full cell overlap when the cropland weighting is applied.
suppressMessages({library(sf); library(jsonlite)})
root  <- Sys.getenv("AGIMPACTS_ROOT", "/Users/kdoerkse/Documents/UCT/AgImpacts")
poly  <- file.path(root, "cy-bench", "polygons")
cdata <- file.path(root, "cy-bench", "cybench-data")
crops <- list.dirs(cdata, recursive = FALSE, full.names = FALSE)
PAD   <- 0.5   # ~2 ERA5 cells, so edge units keep full cell overlap

`%||%` <- function(a, b) if (is.null(a)) b else a
entries <- list()
for (cc in sort(list.dirs(poly, recursive = FALSE, full.names = FALSE))) {
  shp <- file.path(poly, cc, paste0(cc, ".shp"))
  if (!file.exists(shp)) next
  s <- st_read(shp, quiet = TRUE)

  # Restrict to units that actually carry yield data for some crop. This is the
  # difference between a usable box and a useless one for countries whose
  # shapefile includes far-flung territories: the US polygon set spans Alaska,
  # Hawaii and American Samoa, giving a bbox from 179W to 180E (501,000 cells,
  # crossing the antimeridian), while the units with maize yields are CONUS
  # alone (~24,000 cells). Units with no yield would produce climate columns
  # nothing could ever be joined to.
  yield_ids <- character(0)
  for (cr in crops) {
    yf <- file.path(cdata, cr, cc, sprintf("yield_%s_%s.csv", cr, cc))
    if (file.exists(yf)) {
      yield_ids <- union(yield_ids, utils::read.csv(yf, stringsAsFactors = FALSE)$adm_id)
    }
  }
  # Only trust the filter when the two id sets actually correspond. Malawi is the
  # counterexample: its shapefile numbers districts within each region
  # (0101-0106, 0201-0209, ...) while its yield file numbers them continuously
  # (0101-0110, 0211-0217, ...). Same 28 districts, incompatible ids, 18%
  # accidental overlap -- filtering on that would have silently produced a box
  # covering only northern Malawi. Below the threshold, keep the full extent and
  # record the mismatch so it is visible rather than quietly wrong.
  match_rate <- if (length(yield_ids)) {
    length(intersect(yield_ids, s$adm_id)) / length(yield_ids)
  } else NA_real_
  if (!is.na(match_rate) && match_rate >= 0.8) {
    s <- s[s$adm_id %in% yield_ids, ]
  }
  # BR ships without a .prj. Its coordinates are already WGS84 lon/lat (verified
  # against Brazil's true extent), as are every other cy-bench polygon set, so an
  # undefined CRS is assumed to be 4326 rather than failing the whole build.
  no_crs <- is.na(st_crs(s))
  if (no_crs) st_crs(s) <- 4326
  s <- st_transform(s, 4326)
  b <- as.numeric(st_bbox(s))
  if (any(abs(b[c(1,3)]) > 180) || any(abs(b[c(2,4)]) > 90)) {
    stop("bbox for ", cc, " is not lon/lat degrees: ", paste(round(b,2), collapse=", "))
  }

  # Admin level and unit count come from the yield files, per crop.
  per_crop <- list()
  for (cr in crops) {
    y <- file.path(cdata, cr, cc, sprintf("yield_%s_%s.csv", cr, cc))
    if (!file.exists(y)) next
    d <- utils::read.csv(y, stringsAsFactors = FALSE)
    lvl <- unique(regmatches(d$adm_id, regexpr("-A[0-9]-", d$adm_id)))
    # cy-bench uses two id conventions: an explicit {CC}-A1-/{CC}-A2- scheme
    # (the African countries), and native national codes elsewhere (NUTS for the
    # EU, state-county FIPS for the US, IBGE municipality codes for Brazil).
    # Only the first states its admin level, so rather than guess at the others
    # the raw id is carried through and the unit count indicates granularity.
    per_crop[[cr]] <- list(
      units     = length(unique(d$adm_id)),
      admin     = if (length(lvl)) gsub("-", "", lvl[1]) else "native",
      adm_id_example = as.character(d$adm_id[1]),
      year_min  = min(d$harvest_year), year_max = max(d$harvest_year),
      n_years   = length(unique(d$harvest_year)), obs = nrow(d))
  }
  if (!length(per_crop)) next

  entries[[cc]] <- list(
    country_code = cc,
    shapefile    = file.path("cy-bench", "polygons", cc, paste0(cc, ".shp")),
    polygons     = nrow(s),
    crs_assumed  = no_crs,
    id_match_rate = if (is.na(match_rate)) NULL else round(match_rate, 3),
    id_join_ok    = !is.na(match_rate) && match_rate >= 0.8,
    bbox         = list(lon_min = round(b[1] - PAD, 2), lat_min = round(b[2] - PAD, 2),
                        lon_max = round(b[3] + PAD, 2), lat_max = round(b[4] + PAD, 2)),
    crosses_antimeridian = (b[1] < -170 && b[3] > 170),
    grid_cells   = round(((b[3]+PAD)-(b[1]-PAD))/0.25) * round(((b[4]+PAD)-(b[2]-PAD))/0.25),
    crops        = per_crop)
}
# Surface the boxes that are larger than their unit count suggests -- almost
# always an overseas territory sitting inside a mainland admin unit (South
# Africa's Prince Edward Islands, Spain's Canaries, Portugal's Azores). These
# still work, they just download more sea than necessary; the download is
# queue-bound rather than volume-bound, so the cost is small. Override a box by
# editing the JSON directly if it matters for a given country.
cells <- vapply(entries, function(e) e$grid_cells, numeric(1))
cat("\nLargest AOI boxes (cells):\n")
for (cc in names(sort(cells, decreasing = TRUE))[1:8]) {
  e <- entries[[cc]]
  cat(sprintf("  %-4s %7d cells  lon[%.1f,%.1f] lat[%.1f,%.1f]  %d units\n",
              cc, e$grid_cells, e$bbox$lon_min, e$bbox$lon_max,
              e$bbox$lat_min, e$bbox$lat_max, e$polygons))
}
bad <- names(entries)[!vapply(entries, function(e) isTRUE(e$id_join_ok), logical(1))]
if (length(bad)) {
  cat("\nWARNING -- shapefile adm_id does not match yield adm_id (cannot be joined):\n")
  for (cc in bad) {
    cat(sprintf("  %-4s match rate %.0f%%  (full country extent used for the AOI box)\n",
                cc, 100 * (entries[[cc]]$id_match_rate %||% 0)))
  }
}

wrapped <- names(entries)[vapply(entries, function(e) isTRUE(e$crosses_antimeridian), logical(1))]
if (length(wrapped)) {
  cat("\nWARNING: box crosses the antimeridian (era5.R handles the seam, but check):",
      paste(wrapped, collapse = ", "), "\n")
}

write_json(entries, file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])), "..", "R", "cybench-aois.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
cat("wrote R/cybench-aois.json with", length(entries), "countries\n")
