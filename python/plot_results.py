"""
Generate all plots from a completed train_eval.py run.

Reads from --save_dir:
    run_meta.json   — run parameters and overall metrics
    adm1_rmse.csv   — RMSE per state
    adm2_errors.csv — per-county predictions and errors

Two shapefile arguments:
  --shp_path   Global shapefile (ISO + hierid columns).  Used for state-level
               RMSE map and state border overlays on county maps.
  --county_shp Census-style county shapefile with a GEOID column that matches
               adm2_id.  Required for admin2 observed/predicted yield maps.
               The tigris R package caches these; point to any
               cb_20XX_us_county_*.shp or tl_20XX_us_county.shp.

Outputs written to --save_dir:
    scatter.png
    feature_importance.png
    rmse_map_adm1.png        (RMSE aggregated per state)
    mae_adm1.png             (MAE aggregated per state)
    observed_yield_adm2.png  (requires --county_shp)
    predicted_yield_adm2.png (requires --county_shp)
    mae_adm2.png             (MAE per county; requires --county_shp)
    observed_yield_adm1.png  (state-level fallback when --county_shp omitted)
    rmse_by_bin.png          (RMSE and bias per observed-yield bin)
    yield_density.png        (KDE of observed vs predicted yield)
    bin_stats.csv            (per-bin n, mean_yield, rmse, mae, bias)

Example:
    python plot_results.py \\
        --save_dir /path/to/results/rf_test2005_20260512_143022 \\
        --shp_path /path/to/agglomerated-world-new.shp \\
        --county_shp /path/to/cb_2020_us_county_20m.shp \\
        --country USA
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from shapely.geometry import box as _shapely_box

_CONUS_EXCLUDE_STUSPS = {"AK", "HI", "PR", "GU", "VI", "MP", "AS"}
_CONUS_EXCLUDE_STATEFP = {"02", "15", "72", "66", "78", "60", "69"}
_CONUS_EXCLUDE_STATE_FIPS_INT = {2, 15}   # AK, HI — used for global shp

# Bounding box (WGS84) that covers the contiguous US only.
# Anything outside this is dropped regardless of attribute columns,
# which definitively removes Hawaii, Alaska, and territories.
_CONUS_BBOX = _shapely_box(-125.0, 24.0, -66.0, 50.0)


def _clip_to_conus(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return rows whose geometry intersects the CONUS bounding box."""
    in_crs = gdf.crs
    mask = gdf.to_crs("EPSG:4326").geometry.intersects(_CONUS_BBOX)
    return gdf[mask].copy()


# ===== AOI GEOMETRY CONFIG =====
# Per-AOI adm2 shapefile handling, so the same plotting code serves CONUS and the
# cy-bench AOIs. Each entry gives:
#   adm2_col   : shapefile column holding the id that matches adm2_id in the
#                results CSVs. Renamed to GEOID on load so every downstream join
#                (which is written against GEOID) works unchanged.
#   crs        : equal-area projection to draw in, chosen for that region.
#   clip_conus : drop geometries outside the CONUS bbox — only ever right for the
#                US shapefile, which ships Alaska/Hawaii/territories.
#   zfill      : zero-pad adm2_id to this width before joining. US FIPS codes lose
#                their leading zero through CSV round-trips; the cy-bench ids are
#                alphanumeric and must be left alone.
AOI_GEO = {
    "CONUS": dict(adm2_col="GEOID",  crs="EPSG:5070",    clip_conus=True,  zfill=5),
    # Africa Albers Equal Area Conic. BF straddles UTM zones 30N/31N, so a
    # single UTM zone would skew one side of the country.
    "BF":    dict(adm2_col="adm_id", crs="ESRI:102022",  clip_conus=False, zfill=0),
}

# Default adm2 shapefile per AOI, so --county_shp can be omitted for known AOIs.
_PROC_ROOT = "/Users/kdoerkse/Documents/UCT/AgImpacts"
AOI_DEFAULT_SHP = {
    "CONUS": f"{_PROC_ROOT}/us_county_2020_shapefile/cb_2020_us_county_20m.shp",
    "BF":    f"{_PROC_ROOT}/cy-bench/polygons/BF/BF.shp",
}


def aoi_geo_config(aoi: str) -> dict:
    if aoi not in AOI_GEO:
        raise ValueError(
            f"No geometry config for AOI {aoi!r}. Known: {sorted(AOI_GEO)}. "
            "Add an entry to AOI_GEO in plot_results.py."
        )
    return AOI_GEO[aoi]

DEFAULT_SHP = (
    "/Users/kdoerkse/Documents/UCT/AgImpacts/ir_shapefile/agglomerated-world-new.shp"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from a completed train_eval.py run."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory containing run_meta.json, adm1_rmse.csv, adm2_errors.csv.",
    )
    parser.add_argument(
        "--shp_path", type=str, default=DEFAULT_SHP,
        help="Global shapefile (ISO + hierid columns) for state-level maps.",
    )
    parser.add_argument(
        "--aoi", type=str, default="CONUS", choices=sorted(AOI_GEO),
        help="AOI whose adm2 geometry to draw. Selects the id column, "
             "projection and default shapefile. Non-CONUS AOIs skip the "
             "US-FIPS-keyed admin1 maps. Default: CONUS.",
    )
    parser.add_argument(
        "--county_shp", type=str, default=None,
        help=(
            "adm2 shapefile. Optional — defaults to the known file for --aoi "
            "(CONUS: cb_2020_us_county_20m.shp from tigris, with a GEOID "
            "column; BF: cy-bench/polygons/BF/BF.shp). Required for admin2 maps."
        ),
    )
    parser.add_argument(
        "--country", type=str, default="USA",
        help="ISO country code to subset from --shp_path. Default: USA.",
    )
    parser.add_argument(
        "--n_bins", type=int, default=10,
        help="Number of equal-width observed-yield bins for bin analysis. Default: 10.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(save_dir, aoi="CONUS"):
    meta_path = os.path.join(save_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"run_meta.json not found in {save_dir}. Run train_eval.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    adm1_rmse = pd.read_csv(os.path.join(save_dir, "adm1_rmse.csv"))
    errors    = pd.read_csv(os.path.join(save_dir, "adm2_errors.csv"))
    # US FIPS codes lose their leading zero through CSV round-trips, so re-pad
    # them to match GEOID. cy-bench ids are alphanumeric (zfill=0) — leave them.
    zfill = aoi_geo_config(aoi)["zfill"]
    errors["adm2_id"] = errors["adm2_id"].astype(str)
    if zfill:
        errors["adm2_id"] = errors["adm2_id"].str.zfill(zfill)
    return meta, adm1_rmse, errors


def load_states_geo(shp_path, country):
    """
    Load global shapefile, subset to country, dissolve to state level.
    Returns a GeoDataFrame with columns: state_fips (int), geometry.
    For USA the hierid state number equals the state FIPS code.
    """
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf["ISO"] == country].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf["state_fips"] = gdf["hierid"].str.split(".").str[1].astype(int)

    if country == "USA":
        gdf = gdf[~gdf["state_fips"].isin(_CONUS_EXCLUDE_STATE_FIPS_INT)].copy()

    dissolved = (
        gdf.dissolve(by="state_fips", as_index=False)[["state_fips", "geometry"]]
    )
    return _clip_to_conus(dissolved).to_crs("EPSG:5070")


def attach_state_names(states_gdf, errors):
    """
    Derive state_fips → adm1 (state name) from the first two digits of adm2_id,
    then join onto states_gdf.
    """
    lookup = (
        errors.assign(state_fips=errors["adm2_id"].str[:2].astype(int))
        [["state_fips", "adm1"]]
        .drop_duplicates(subset="state_fips")
    )
    return states_gdf.merge(lookup, on="state_fips", how="left")


def load_counties_geo(county_shp=None, aoi="CONUS"):
    """
    Load an adm2 shapefile for `aoi`, filter and reproject per AOI_GEO.

    CONUS expects a Census-style shapefile with a GEOID column (5-char FIPS);
    cy-bench AOIs (e.g. BF) use their adm_id column, which is renamed to GEOID
    so the downstream joins are AOI-agnostic. county_shp defaults to the known
    shapefile for that AOI (AOI_DEFAULT_SHP).
    """
    cfg        = aoi_geo_config(aoi)
    county_shp = county_shp or AOI_DEFAULT_SHP.get(aoi)
    if not county_shp:
        raise ValueError(f"No default shapefile for AOI {aoi!r} — pass --county_shp.")

    gdf = gpd.read_file(county_shp)
    if cfg["adm2_col"] not in gdf.columns:
        raise ValueError(
            f"Shapefile {county_shp!r} has no {cfg['adm2_col']!r} column "
            f"(has {list(gdf.columns)}). Check --county_shp matches --aoi {aoi}."
        )
    if cfg["adm2_col"] != "GEOID":
        gdf = gdf.rename(columns={cfg["adm2_col"]: "GEOID"})
    if cfg["clip_conus"]:
        gdf = _clip_to_conus(gdf)
    return gdf.to_crs(cfg["crs"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _county_merge(counties_gdf, errors, value_col):
    return counties_gdf.merge(
        errors[["adm2_id", value_col]],
        left_on="GEOID", right_on="adm2_id",
        how="left",
    )


def _choropleth(ax, base_gdf, data_gdf, value_col, cmap, label, border_gdf=None,
                county_level=True, vmin=None, vmax=None):
    """Draw a choropleth on ax; base_gdf shows the grey 'no data' background."""
    lw = 0.2 if county_level else 0.5
    base_gdf.plot(ax=ax, color="#d3d3d3", edgecolor="white", linewidth=lw)
    has_data = data_gdf[data_gdf[value_col].notna()]
    if len(has_data):
        _vmin = vmin if vmin is not None else has_data[value_col].min()
        _vmax = vmax if vmax is not None else has_data[value_col].max()
        has_data.plot(
            column=value_col, ax=ax, cmap=cmap,
            vmin=_vmin, vmax=_vmax,
            edgecolor="white", linewidth=lw, legend=True,
            legend_kwds={"label": label, "shrink": 0.6},
        )
    if border_gdf is not None:
        border_gdf.boundary.plot(ax=ax, edgecolor="#555555", linewidth=0.6)
    return len(has_data)


# ---------------------------------------------------------------------------
# Non-geographic plots (no shapefile needed)
# ---------------------------------------------------------------------------

def plot_scatter(meta, errors, save_dir):
    test_year    = meta["test_year"]
    model_type   = meta["model_type"]
    rmse_overall = meta["rmse_overall"]
    r2_overall   = meta["r2_overall"]
    y_te         = errors["y_true"].values
    y_pred_te    = errors["y_pred"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        y_te, y_pred_te,
        c=y_pred_te - y_te, cmap="coolwarm",
        alpha=0.6, edgecolors="none", s=25,
    )
    plt.colorbar(sc, ax=ax, label="Residual (pred − obs)")
    lims = [min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Observed yield", fontsize=12)
    ax.set_ylabel("Predicted yield", fontsize=12)
    ax.set_title(
        f"{model_type.upper()} maize yield — LOO test (hindcast_start={test_year})\n"
        f"RMSE={rmse_overall:,.0f}  R²={r2_overall:.3f}  n={len(y_te)}",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter.png"), dpi=150)
    plt.close()
    print("  scatter.png")


def plot_feature_importance(meta, save_dir):
    if not meta.get("feature_importances"):
        print("  feature_importance.png — skipped (model has no feature_importances_)")
        return

    importance = (
        pd.Series(meta["feature_importances"])
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    importance[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature importance (mean decrease impurity)")
    ax.set_title(
        f"Top 15 feature importances — {meta['model_type'].upper()} LOO split"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print("  feature_importance.png")


# ---------------------------------------------------------------------------
# Admin2 (county-level) maps  — requires --county_shp
# ---------------------------------------------------------------------------

def plot_adm2_yield(meta, errors, counties_gdf, states_gdf, value_col,
                    title_prefix, cmap, fname, save_dir, vmin=None, vmax=None):
    test_year = meta["test_year"]
    merged    = _county_merge(counties_gdf, errors, value_col)
    has_data  = merged[merged[value_col].notna()]

    fig, ax = plt.subplots(figsize=(14, 8))
    n = _choropleth(
        ax, merged, merged, value_col, cmap, "Yield (kg/ha)",
        border_gdf=states_gdf, county_level=True, vmin=vmin, vmax=vmax,
    )
    ax.set_title(
        f"{title_prefix} — hindcast_start={test_year} (yield year={test_year + 1})\n"
        f"(grey = no data, n={len(has_data)} counties)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {fname}")


def plot_adm2_mae(meta, errors, counties_gdf, states_gdf, save_dir):
    test_year = meta["test_year"]
    merged    = _county_merge(counties_gdf, errors, "abs_error")
    has_data  = merged[merged["abs_error"].notna()]

    fig, ax = plt.subplots(figsize=(14, 8))
    _choropleth(
        ax, merged, merged, "abs_error", "YlOrRd",
        "MAE (yield units)", border_gdf=states_gdf, county_level=True,
    )
    ax.set_title(
        f"MAE by county — hindcast_start={test_year} "
        f"(yield year {test_year + 1})\n(grey = no test data, n={len(has_data)} counties)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "mae_adm2.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  mae_adm2.png")


# ---------------------------------------------------------------------------
# Admin1 (state-level) maps  — uses global shapefile states
# ---------------------------------------------------------------------------

def plot_adm1_rmse(meta, adm1_rmse, states_gdf, save_dir):
    test_year  = meta["test_year"]
    merged     = states_gdf.merge(adm1_rmse, on="adm1", how="left")

    fig, ax = plt.subplots(figsize=(14, 8))
    _choropleth(
        ax, merged, merged, "rmse", "YlOrRd",
        "RMSE (yield units)", county_level=False,
    )
    ax.set_title(
        f"RMSE by state — hindcast_start={test_year} (yield year {test_year + 1})\n"
        "(grey = no test data)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "rmse_map_adm1.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  rmse_map_adm1.png")


def plot_adm1_mae(meta, errors, states_gdf, save_dir):
    test_year = meta["test_year"]
    state_mae = (
        errors
        .assign(state_fips=errors["adm2_id"].str[:2].astype(int))
        .groupby("state_fips")["abs_error"].mean()
        .reset_index()
        .rename(columns={"abs_error": "mae"})
    )
    merged = states_gdf.merge(state_mae, on="state_fips", how="left")

    fig, ax = plt.subplots(figsize=(14, 8))
    _choropleth(
        ax, merged, merged, "mae", "YlOrRd",
        "MAE (yield units)", county_level=False,
    )
    ax.set_title(
        f"MAE by state — hindcast_start={test_year} "
        f"(yield year {test_year + 1})\n(grey = no test data)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "mae_adm1.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  mae_adm1.png")


def plot_adm1_observed_yield(meta, errors, states_gdf, save_dir):
    """State-level fallback when --county_shp is not provided."""
    test_year   = meta["test_year"]
    state_yield = (
        errors
        .assign(state_fips=errors["adm2_id"].str[:2].astype(int))
        .groupby("state_fips")["y_true"].mean()
        .reset_index()
        .rename(columns={"y_true": "yield"})
    )
    merged = states_gdf.merge(state_yield, on="state_fips", how="left")

    fig, ax = plt.subplots(figsize=(14, 8))
    _choropleth(
        ax, merged, merged, "yield", "YlGn",
        "Yield (kg/ha)", county_level=False,
    )
    ax.set_title(
        f"Observed yield (state mean) — hindcast_start={test_year} "
        f"(yield year={test_year + 1})\n(grey = no data)",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "observed_yield_adm1.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  observed_yield_adm1.png")


# ---------------------------------------------------------------------------
# Bin analysis
# ---------------------------------------------------------------------------

def compute_bin_stats(errors: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = errors.copy()
    df["bin"] = pd.cut(df["y_true"], bins=n_bins)
    stats = (
        df.groupby("bin", observed=True)
        .apply(
            lambda g: pd.Series({
                "n":          len(g),
                "mean_yield": g["y_true"].mean(),
                "rmse":       np.sqrt(((g["y_pred"] - g["y_true"]) ** 2).mean()),
                "mae":        (g["y_pred"] - g["y_true"]).abs().mean(),
                "bias":       (g["y_pred"] - g["y_true"]).mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    stats["bin_label"] = stats["bin"].apply(
        lambda b: f"{b.mid:,.0f}" if pd.notna(b) else ""
    )
    return stats


def plot_rmse_by_bin(stats: pd.DataFrame, save_dir: str, n_bins: int) -> None:
    """Two-panel figure: RMSE per bin (top) and mean bias per bin (bottom)."""
    fig, (ax_rmse, ax_bias) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    x      = np.arange(len(stats))
    labels = stats["bin_label"].tolist()

    bars = ax_rmse.bar(x, stats["rmse"], color="steelblue",
                       edgecolor="white", linewidth=0.5)
    for bar, n in zip(bars, stats["n"]):
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stats["rmse"].max() * 0.015,
            f"n={int(n)}", ha="center", va="bottom", fontsize=7,
        )
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title(f"RMSE by observed yield bin  (n_bins={n_bins})")
    ax_rmse.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    colours = ["#d62728" if b > 0 else "#1f77b4" for b in stats["bias"]]
    ax_bias.bar(x, stats["bias"], color=colours, edgecolor="white", linewidth=0.5)
    ax_bias.axhline(0, color="black", linewidth=0.8)
    ax_bias.set_ylabel("Mean bias\n(pred − obs)")
    ax_bias.set_xlabel("Observed yield bin midpoint")
    ax_bias.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_bias.set_xticks(x)
    ax_bias.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_bias.legend(
        handles=[
            Patch(color="#d62728", label="Over-predict"),
            Patch(color="#1f77b4", label="Under-predict"),
        ],
        fontsize=8, loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rmse_by_bin.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  rmse_by_bin.png")


def plot_yield_density(errors: pd.DataFrame, save_dir: str) -> None:
    """KDE of observed and predicted yield on the same axes."""
    y_true = errors["y_true"].values
    y_pred = errors["y_pred"].values

    # gaussian_kde needs a spread of points; thin folds (e.g. a BF year with a
    # single reporting adm2) have nothing to estimate a density from.
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        print(f"  yield_density.png — skipped (only {len(y_true)} test point(s))")
        return

    xs        = np.linspace(min(y_true.min(), y_pred.min()),
                            max(y_true.max(), y_pred.max()), 600)
    dens_true = gaussian_kde(y_true)(xs)
    dens_pred = gaussian_kde(y_pred)(xs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, dens_true, label="Observed",  color="steelblue",  linewidth=2)
    ax.plot(xs, dens_pred, label="Predicted", color="darkorange", linewidth=2, linestyle="--")
    ax.fill_between(xs, dens_true, dens_pred, alpha=0.15, color="grey", label="Difference")

    for vals, colour, label in [
        (y_true, "steelblue",  "Observed"),
        (y_pred, "darkorange", "Predicted"),
    ]:
        mu = vals.mean()
        ax.axvline(mu, color=colour, linewidth=1, linestyle=":")
        ax.text(mu, ax.get_ylim()[1] * 0.02, f"{label} mean\n{mu:,.0f}",
                color=colour, fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Yield")
    ax.set_ylabel("Density")
    ax.set_title("Yield distribution: observed vs predicted (KDE)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "yield_density.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  yield_density.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_fold(save_dir, states_gdf, counties_gdf=None, n_bins=10, aoi="CONUS"):
    """Generate the full plot suite for one fold/run into save_dir.

    Takes pre-loaded shapefiles so a batch caller (plot_loo_all.py) can load
    them once and reuse across many folds.

    Parameters
    ----------
    save_dir : str
        A directory with run_meta.json, adm1_rmse.csv, adm2_errors.csv.
    states_gdf : GeoDataFrame
        Base state geometries from load_states_geo (WITHOUT adm1 names attached;
        names are attached per-fold from that fold's errors).
    counties_gdf : GeoDataFrame or None
        County geometries from load_counties_geo, or None to skip admin2 maps.
    n_bins : int
        Observed-yield bins for the bin analysis.
    aoi : str
        AOI key from AOI_GEO. Controls adm2_id padding and whether the
        admin1 (state) maps run — those are keyed off US FIPS codes and are
        skipped for the cy-bench AOIs.
    """
    meta, adm1_rmse, errors = load_run(save_dir, aoi=aoi)

    if aoi == "CONUS":
        # Drop Hawaii (state FIPS 15) from all outputs
        errors    = errors[errors["adm2_id"].str[:2] != "15"].copy()
        adm1_rmse = adm1_rmse[adm1_rmse["adm1"] != "Hawaii"].copy()
    else:
        # The admin1 maps derive state_fips from the first two digits of
        # adm2_id, which only exists for US FIPS ids.
        states_gdf = None

    # State geometries are optional: skip admin1 maps / state borders when the
    # global shapefile isn't available (e.g. county-only CONUS runs on an HPC).
    states_named = (attach_state_names(states_gdf, errors)
                    if states_gdf is not None else None)

    print(
        f"\nGenerating plots for {meta['model_type'].upper()} "
        f"test_year={meta['test_year']}:"
    )

    # Non-geographic (always)
    plot_scatter(meta, errors, save_dir)
    plot_feature_importance(meta, save_dir)

    # State-level (only when the global shapefile was loaded)
    if states_named is not None:
        plot_adm1_rmse(meta, adm1_rmse, states_named, save_dir)
        plot_adm1_mae(meta, errors, states_named, save_dir)

    # County-level (requires counties_gdf; state borders are an optional overlay)
    if counties_gdf is not None:
        # Shared colorbar limits so observed and predicted are directly comparable
        yield_vmin = min(errors["y_true"].min(), errors["y_pred"].min())
        yield_vmax = max(errors["y_true"].max(), errors["y_pred"].max())

        plot_adm2_yield(
            meta, errors, counties_gdf, states_named,
            value_col="y_true", title_prefix="Observed maize yield",
            cmap="YlGn", fname="observed_yield_adm2.png",
            save_dir=save_dir, vmin=yield_vmin, vmax=yield_vmax,
        )
        plot_adm2_yield(
            meta, errors, counties_gdf, states_named,
            value_col="y_pred", title_prefix="Predicted maize yield",
            cmap="YlGn", fname="predicted_yield_adm2.png",
            save_dir=save_dir, vmin=yield_vmin, vmax=yield_vmax,
        )
        plot_adm2_mae(meta, errors, counties_gdf, states_named, save_dir)
    elif states_named is not None:
        plot_adm1_observed_yield(meta, errors, states_named, save_dir)

    # Bin analysis
    bin_errors = errors.dropna(subset=["y_true", "y_pred"]).copy()
    bin_stats  = compute_bin_stats(bin_errors, n_bins)
    print(f"\nBin stats (n_bins={n_bins}, n_total={len(bin_errors):,}):")
    print(
        bin_stats[["bin", "n", "mean_yield", "rmse", "mae", "bias"]]
        .to_string(index=False, float_format="{:,.1f}".format)
    )
    bin_stats.to_csv(os.path.join(save_dir, "bin_stats.csv"), index=False)
    print("  bin_stats.csv")
    plot_rmse_by_bin(bin_stats, save_dir, n_bins)
    plot_yield_density(bin_errors, save_dir)

    print(f"\nAll plots saved to {save_dir}/")


def load_plot_geometries(shp_path, country, county_shp, aoi="CONUS"):
    """Load the state (required) and optional county shapefiles for plotting.

    Returns (states_gdf, counties_gdf); either may be None. The state shapefile
    is optional — if shp_path is missing or doesn't exist, state-level (admin1)
    maps are skipped rather than raising (e.g. county-only CONUS runs on an HPC
    that doesn't have the global shapefile). The admin1 maps are US-only, so
    they are skipped outright for the non-CONUS AOIs. county_shp falls back to
    the known shapefile for the AOI when it exists on disk. Shared by the
    single-fold CLI and the batch runner so both load the shapefiles the same
    way — and, for the batch runner, exactly once.
    """
    states_gdf = None
    if aoi != "CONUS":
        print(f"AOI {aoi}: skipping state-level (admin1) maps "
              "(they are keyed off US FIPS codes).")
    elif shp_path and os.path.exists(shp_path):
        print(f"Loading global shapefile: {shp_path}")
        states_gdf = load_states_geo(shp_path, country)
        print(f"States loaded: {len(states_gdf)}")
    else:
        print(f"State shapefile not found ({shp_path!r}); "
              "skipping state-level (admin1) maps. Pass --shp_path to enable them.")

    counties_gdf = None
    county_shp = county_shp or AOI_DEFAULT_SHP.get(aoi)
    if county_shp and os.path.exists(county_shp):
        print(f"Loading county shapefile: {county_shp}")
        counties_gdf = load_counties_geo(county_shp, aoi=aoi)
        print(f"Admin-2 units loaded: {len(counties_gdf)}")
    else:
        print(f"No adm2 shapefile for AOI {aoi} ({county_shp!r}); "
              "skipping county (admin2) maps. Pass --county_shp to enable them.")
    return states_gdf, counties_gdf


def main():
    args = parse_args()
    states_gdf, counties_gdf = load_plot_geometries(
        args.shp_path, args.country, args.county_shp, aoi=args.aoi)
    plot_fold(args.save_dir, states_gdf, counties_gdf,
              n_bins=args.n_bins, aoi=args.aoi)


if __name__ == "__main__":
    main()
