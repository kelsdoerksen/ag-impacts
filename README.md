# 👩‍🌾 ag-impacts

Climate-impacted crop yield prediction with AI.

Utilizing daily climate data either observed (ERA5) or forecasted (EC-Earth3 DCPP hindcasts), processed into agronomic features per admin-2 unit and used as inputs into ML models to predict crop yield 1-5 years ahead.

## Aggregated Feature Design

**Features are computed per pixel, then averaged over cropland.** 

```
stage 1   daily gridded climate  ->  per-pixel daily cube (NetCDF)
                                 ->  pixel x admin-2 cropland weight table
stage 2   cube                   ->  per-pixel DAILY features   (no weighting yet)
stage 3   features + weights     ->  admin-2 table              (weighting here)
```

---

## Climate pipeline

### Stage 1 — download and store per pixel (R)

| script | functionality |
|---|---|
| `R/pixel-store.R` | **Shared stage-1 engine.** Grid-identity check, the cropland weight table, unit conversion, the cube writer, resume logic and the run loop.|
| `R/era5-pixel-store.R` | **Stage 1 for ERA5.** Front-end: fetches hourly ERA5 (CDS or ARCO), reduces it to daily, optionally on local solar days (`--local-time`). |
| `R/dcpp-pixel-store.R` | **Stage 1 for the DCPP hindcasts.** Front-end: reads EC-Earth3 daily NetCDFs for one initialisation year and lead time (`--forecast-year`). |
| `R/era5.R` | ERA5 acquisition. Available at two sources: CDS (server-side subsetting, queued, batched into multi-month requests) and ARCO-ERA5 on GCS (no account, no queue, transfers global fields). Reduces hourly to daily `tasmax`/`tasmin`/`hursmin`/`pr`. Holds `AOI_BOUNDS`, the single registry of AOI bounding boxes. |
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

### Stages 2 and 3: generate features and aggregate (Python)

| script | functionality |
|---|---|
| `python/era5_features.py` | **Stages 2 and 3 for both sources.** Stage 2 computes per-pixel daily features; stage 3 takes the cropland-area-weighted admin-2 mean, optionally accumulated over a crop-calendar season (`--reduce season`) or season-relative blocks (`--reduce blocks`). `--source` selects the climate tree; `--forecast-year` the lead time. `--list-features` shows the registry. |
| `python/degree_days.py` | The single-sine degree-day integral, shared by `era5_features.py` and `data_loader.py` so the feature definition cannot differ between the two. |
| `python/run_cybench_era5.py` | Driver: runs stages 1 and 2 over many AOIs in sequence, deriving each AOI's year span from its cy-bench crop coverage. Dry-run by default; resumable; logs per AOI. |
| `python/compare_dcpp_era5.py` | **Checks the hindcast against the reanalysis.** Drives `era5_features.py` for ERA5 and for each EC-Earth3 member, joins the two on `(adm2_id, harvest_year)`, and reports how well each member -- and the ensemble mean -- reproduces the ERA5 feature. `r_anom` (each unit's own mean removed from both sides) is the interannual skill; `r_pooled` is reported beside it but is inflated by between-unit climatology. Writes csv + json + every paired value. Members auto-discover from the DCPP tree; dry-run by default. |
| `python/report_dcpp_era5.py` | Turns one or more `dcpp_era5_corr_*.csv` into a single self-contained HTML report -- headline skill per AOI, decay with lead time, what the ensemble mean adds, and calibration -- plus a tidy csv for slides. No plotting dependency; the bars are inline CSS. |

```sh
python python/era5_features.py --aoi=US --years=2000-2023 --stage=features
python python/era5_features.py --aoi=US --years=2000-2023 --stage=aggregate \
    --reduce=blocks --crop=maize

# DCPP fy1 vs ERA5: plan and the exact commands first, then run
python python/compare_dcpp_era5.py --aois ZM --harvest-years 2002-2022
python python/compare_dcpp_era5.py --aois ZM --harvest-years 2002-2022 --go
python python/report_dcpp_era5.py --dir /mnt/crl_storage/personal_folders/kdoerksen/comparisons
```

### Supporting data

| script | functionality |
|---|---|
| `R/build-cybench-aois.R` | Generates `R/cybench-aois.json`: every cy-bench country's bounding box, polygon count, and per-crop year coverage from the AOI shapefiles, so the boxes cannot drift from the geometry. Re-run it rather than editing the JSON. |
| `R/crop-data-gen.R`, `R/crop-data-gen-cybench.R` | Yield and crop-area target tables (cy-bench and other sources). |
| `R/phenology-sample-gen.R` | NDVI / FPAR / soil-moisture phenology series per admin unit. |
| `R/county-climate-cell.R` | County-to-climate-cell lookups used by the experiment runner. |

---

## ML pipeline

| script | functionality |
|---|---|
| `python/data_loader.py` | Assembles the design matrix: joins stage 3's admin-2 features to yield targets, and optionally appends yield history and phenology. `FEATURE_MODES` is `pixel` (whole-season) and `pixel_blocks` (season-relative windows) are fixed upstream. |
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
