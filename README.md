# ag-impacts

Climate-impacted crop yield prediction with ML.

Daily climate — observed (ERA5) or forecast (EC-Earth3 DCPP hindcasts) — is
turned into agronomic features per admin-2 unit and used to predict crop yield.

## The one design decision worth knowing

**Features are computed per pixel, then averaged over cropland.** Not the other
way round. Averaging first and computing features second is only valid for
features linear in their inputs, and the useful ones are not:

```
mean_i( max(0, T_i - Tbase) )  !=  max(0, mean_i(T_i) - Tbase)
```

Averaging first clips the tails, so a unit containing both a cool and a hot pixel
loses the heat the hot pixel actually accumulated. Growing and extreme degree
days, VPD, dry-day and hot-day counts are all affected. Both climate sources go
through the same three stages, in the same order, so a comparison between them
measures forecast skill rather than a difference in processing.

```
stage 1   daily gridded climate  ->  per-pixel daily cube (NetCDF)
                                 ->  pixel x admin-2 cropland weight table
stage 2   cube                   ->  per-pixel DAILY features   (no weighting yet)
stage 3   features + weights     ->  admin-2 table              (weighting here)
```

---

## Climate pipeline

### Stage 1 — download and store per pixel (R)

| file | what it does |
|---|---|
| `R/pixel-store.R` | **The shared stage-1 engine.** Grid-identity check, the cropland weight table, unit conversion, the cube writer, resume logic and the run loop. Not run directly — both front-ends below are thin wrappers over it, so the two sources cannot drift apart. |
| `R/era5-pixel-store.R` | **Stage 1 for ERA5.** Front-end: fetches hourly ERA5 (CDS or ARCO), reduces it to daily, optionally on local solar days (`--local-time`). |
| `R/dcpp-pixel-store.R` | **Stage 1 for the DCPP hindcasts.** Front-end: reads EC-Earth3 daily NetCDFs for one initialisation year and lead time (`--forecast-year`). |
| `R/era5.R` | ERA5 acquisition. Two sources — CDS (server-side subsetting, queued, batched into multi-month requests) and ARCO-ERA5 on GCS (no account, no queue, transfers global fields). Reduces hourly to daily `tasmax`/`tasmin`/`hursmin`/`pr`. Holds `AOI_BOUNDS`, the single registry of AOI bounding boxes. |
| `R/era5-aoi.R` | Shared configuration: the AOI registry (`ERA5_AOIS` — every cy-bench country plus CONUS), the GeoTIFF daily-grid cache, argument parsing, preflight. Sourced by both front-ends. |
| `R/ML-sample-gen.R` | The DCPP NetCDF readers (`nc_path`, `read_nc_region`, `arrays_to_rast`) and the admin-boundary loaders both sources share. A library, not a script — it has no `main()`. |

```sh
# ERA5
Rscript R/era5-pixel-store.R --aoi=ZM --check            # prove the setup
Rscript R/era5-pixel-store.R --aoi=ZM 2001-2017          # CDS (small AOIs)
Rscript R/era5-pixel-store.R --aoi=US --source=arco 2000-2023   # ARCO (large AOIs)

# DCPP hindcasts
Rscript R/dcpp-pixel-store.R --aoi=CONUS --forecast-year=1 1982-2018
```

### Stages 2 and 3 — features and aggregation (Python)

| file | what it does |
|---|---|
| `python/era5_features.py` | **Stages 2 and 3 for both sources.** Stage 2 computes per-pixel daily features; stage 3 takes the cropland-area-weighted admin-2 mean, optionally accumulated over a crop-calendar season (`--reduce season`) or season-relative blocks (`--reduce blocks`). `--source` selects the climate tree; `--forecast-year` the lead time. `--list-features` shows the registry. |
| `python/degree_days.py` | The single-sine degree-day integral, shared by `era5_features.py` and `data_loader.py` so the feature definition cannot differ between the two. |
| `python/run_cybench_era5.py` | Driver: runs stages 1 and 2 over many AOIs in sequence, deriving each AOI's year span from its cy-bench crop coverage. Dry-run by default; resumable; logs per AOI. |

```sh
python python/era5_features.py --aoi=US --years=2000-2023 --stage=features
python python/era5_features.py --aoi=US --years=2000-2023 --stage=aggregate \
    --reduce=blocks --crop=maize
```

### Supporting data

| file | what it does |
|---|---|
| `R/build-cybench-aois.R` | Generates `R/cybench-aois.json` — every cy-bench country's bounding box, polygon count, and per-crop year coverage — from the shapefiles themselves, so the boxes cannot drift from the geometry. Re-run it rather than editing the JSON. |
| `R/crop-data-gen.R`, `R/crop-data-gen-cybench.R` | Yield and crop-area target tables (cy-bench and other sources). |
| `R/phenology-sample-gen.R` | NDVI / FPAR / soil-moisture phenology series per admin unit. |
| `R/county-climate-cell.R` | County-to-climate-cell lookups used by the experiment runner. |

---

## ML pipeline

| file | what it does |
|---|---|
| `python/data_loader.py` | Assembles the design matrix: joins stage 3's admin-2 features to yield targets, and optionally appends yield history and phenology. `FEATURE_MODES` is `pixel` (whole-season) and `pixel_blocks` (season-relative windows) — both read what stage 3 built, so the feature set is fixed upstream and no flag here can change it. |
| `python/models.py`, `python/train_eval.py` | Model definitions and the train/evaluate loop. |
| `python/run_loo_experiment.py` | Leave-one-out experiment runner. |
| `python/tst/` | Time-series transformer arms. |
| `python/climate_yield_information.py` | Information-theoretic analysis of climate–yield signal. |
| `python/compare_feature_spaces.py`, `python/bin_analysis.py` | Feature-space and binned-response comparisons. |
| `python/aggregate_seed_runs.py` | Pools replicate seeds into one result table. |
| `python/plot_*.py`, `python/make_*.py` | Figures and summary tables. |

---

## Environment

- **R** (stage 1): `terra`, `sf`, `dplyr`, `tidyr`, `arrow`, `jsonlite`, `ncdf4`,
  `reticulate`. See `environment-r.yml`; on a shared machine use the conda env so
  `install.packages()` does not write to a library other people share.
- **Python for reticulate** (ERA5 acquisition): `xarray`, `cdsapi`, `gcsfs`,
  `zarr`, `dask`, `netCDF4`.
- **Python** (stages 2–3 and ML): `xarray`, `netCDF4`, `pandas`, `numpy`,
  `pyarrow`. See `environment.yml`.
- `AGIMPACTS_ROOT` points at the data root; `ERA5_OUT_DIR` overrides just the
  ERA5 output tree.

`R/ERA5-SETUP.md` has the full setup, CDS credentials, cache layout, request
batching and per-pixel feature documentation.
