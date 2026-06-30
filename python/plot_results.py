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
        "--county_shp", type=str, default=None,
        help=(
            "Census-style county shapefile with a GEOID column (e.g. "
            "cb_2020_us_county_20m.shp from tigris). Required for admin2 maps."
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

def load_run(save_dir):
    meta_path = os.path.join(save_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"run_meta.json not found in {save_dir}. Run train_eval.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    adm1_rmse = pd.read_csv(os.path.join(save_dir, "adm1_rmse.csv"))
    errors    = pd.read_csv(os.path.join(save_dir, "adm2_errors.csv"))
    # Ensure adm2_id is a zero-padded 5-char string to match GEOID
    errors["adm2_id"] = errors["adm2_id"].astype(str).str.zfill(5)
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


def load_counties_geo(county_shp):
    """
    Load a Census-style county shapefile, filter to CONUS, reproject.
    Expects a GEOID column (5-char FIPS string).
    """
    gdf = gpd.read_file(county_shp)
    return _clip_to_conus(gdf).to_crs("EPSG:5070")


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

def main():
    args = parse_args()

    meta, adm1_rmse, errors = load_run(args.save_dir)

    # Drop Hawaii (state FIPS 15) from all outputs
    errors    = errors[errors["adm2_id"].str[:2] != "15"].copy()
    adm1_rmse = adm1_rmse[adm1_rmse["adm1"] != "Hawaii"].copy()

    print(f"Loading global shapefile: {args.shp_path}")
    states_gdf = load_states_geo(args.shp_path, args.country)
    states_gdf = attach_state_names(states_gdf, errors)
    print(f"States loaded: {len(states_gdf)}")

    counties_gdf = None
    if args.county_shp:
        print(f"Loading county shapefile: {args.county_shp}")
        counties_gdf = load_counties_geo(args.county_shp)
        print(f"Counties loaded: {len(counties_gdf)}")
    else:
        print(
            "No --county_shp provided. "
            "Admin2 yield maps will be skipped; using state-level fallback."
        )

    print(
        f"\nGenerating plots for {meta['model_type'].upper()} "
        f"test_year={meta['test_year']}:"
    )

    # Non-geographic
    plot_scatter(meta, errors, args.save_dir)
    plot_feature_importance(meta, args.save_dir)

    # State-level (always available)
    plot_adm1_rmse(meta, adm1_rmse, states_gdf, args.save_dir)
    plot_adm1_mae(meta, errors, states_gdf, args.save_dir)

    # County-level (requires --county_shp)
    if counties_gdf is not None:
        # Shared colorbar limits so observed and predicted are directly comparable
        yield_vmin = min(errors["y_true"].min(), errors["y_pred"].min())
        yield_vmax = max(errors["y_true"].max(), errors["y_pred"].max())

        plot_adm2_yield(
            meta, errors, counties_gdf, states_gdf,
            value_col="y_true", title_prefix="Observed maize yield",
            cmap="YlGn", fname="observed_yield_adm2.png",
            save_dir=args.save_dir, vmin=yield_vmin, vmax=yield_vmax,
        )
        plot_adm2_yield(
            meta, errors, counties_gdf, states_gdf,
            value_col="y_pred", title_prefix="Predicted maize yield",
            cmap="YlGn", fname="predicted_yield_adm2.png",
            save_dir=args.save_dir, vmin=yield_vmin, vmax=yield_vmax,
        )
        plot_adm2_mae(meta, errors, counties_gdf, states_gdf, args.save_dir)
    else:
        plot_adm1_observed_yield(meta, errors, states_gdf, args.save_dir)

    # Bin analysis
    bin_errors = errors.dropna(subset=["y_true", "y_pred"]).copy()
    bin_stats  = compute_bin_stats(bin_errors, args.n_bins)
    print(f"\nBin stats (n_bins={args.n_bins}, n_total={len(bin_errors):,}):")
    print(
        bin_stats[["bin", "n", "mean_yield", "rmse", "mae", "bias"]]
        .to_string(index=False, float_format="{:,.1f}".format)
    )
    bin_stats.to_csv(os.path.join(args.save_dir, "bin_stats.csv"), index=False)
    print("  bin_stats.csv")
    plot_rmse_by_bin(bin_stats, args.save_dir, args.n_bins)
    plot_yield_density(bin_errors, args.save_dir)

    print(f"\nAll plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
