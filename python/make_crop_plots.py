"""
Generate per-crop summary plots from a completed PatchTST LOO experiment dir.

Reads from <exp_dir>:
    loo_all_errors.csv   — all county-year predictions across folds
    loo_county_rmse.csv  — RMSE per county across folds
    loo_meta.json        — per-fold + overall metrics

Writes to <exp_dir>/plots/:
    scatter_all.png      — observed vs predicted (all folds), with overall RMSE/R2
    skill_by_year.png    — RMSE% and R2 per LOO test year
    yield_density.png    — KDE of observed vs predicted yield
    mae_county.png       — per-county mean MAE choropleth (best-effort; needs shapefile)
And writes <exp_dir>/crop_summary.json with the headline numbers used by the
cross-crop summary step.

Usage:
    python make_crop_plots.py <exp_dir>
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COUNTY_SHP = "/Users/kdoerkse/Documents/UCT/AgImpacts/us_county_2020_shapefile/cb_2020_us_county_20m.shp"
STATE_SHP  = "/Users/kdoerkse/Documents/UCT/AgImpacts/ir_shapefile/agglomerated-world-new.shp"


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse   = float(np.sqrt(ss_res / len(y_true)))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    mean_y = float(y_true.mean())
    return dict(rmse=rmse, r2=r2, mae=mae, mean_yield=mean_y,
                rmse_pct=100 * rmse / mean_y if mean_y else float("nan"),
                bias=float(np.mean(y_pred - y_true)))


def plot_scatter(errors, crop, m, out):
    yt, yp = errors["y_true"].values, errors["y_pred"].values
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=8, alpha=0.3, edgecolor="none")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel("Observed yield"); ax.set_ylabel("Predicted yield")
    ax.set_title(f"{crop} — PatchTST LOO (all folds)\n"
                 f"RMSE={m['rmse']:,.0f}  R²={m['r2']:.3f}  "
                 f"RMSE%={m['rmse_pct']:.1f}  n={len(yt):,}")
    ax.legend(); ax.set_aspect("equal", "box"); fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  {os.path.basename(out)}")


def plot_skill_by_year(meta, crop, out):
    folds = sorted(meta["folds"], key=lambda f: f["test_year"])
    years = [f["test_year"] for f in folds]
    rmse_pct = [100 * f["rmse_overall"] / f["mean_yield"] if f["mean_yield"] else np.nan
                for f in folds]
    r2 = [f["r2_overall"] for f in folds]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    c1 = "tab:blue"
    ax1.plot(years, rmse_pct, "-o", color=c1, label="RMSE%")
    ax1.set_xlabel("LOO test year (hindcast_start)")
    ax1.set_ylabel("RMSE % of mean yield", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    c2 = "tab:red"
    ax2.plot(years, r2, "-s", color=c2, label="R²")
    ax2.axhline(0, color=c2, ls=":", lw=0.8)
    ax2.set_ylabel("R²", color=c2); ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title(f"{crop} — skill by LOO year")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  {os.path.basename(out)}")


def plot_density(errors, crop, out):
    from scipy.stats import gaussian_kde
    yt, yp = errors["y_true"].values, errors["y_pred"].values
    grid = np.linspace(min(yt.min(), yp.min()), max(yt.max(), yp.max()), 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for vals, lab, col in [(yt, "observed", "tab:green"), (yp, "predicted", "tab:orange")]:
        try:
            ax.plot(grid, gaussian_kde(vals)(grid), color=col, label=lab)
        except Exception:
            ax.hist(vals, bins=40, density=True, alpha=0.4, color=col, label=lab)
    ax.set_xlabel("Yield"); ax.set_ylabel("density")
    ax.set_title(f"{crop} — observed vs predicted yield distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  {os.path.basename(out)}")


def plot_county_mae(county_rmse, crop, out):
    """Per-county mean MAE choropleth. Best-effort: needs the county shapefile."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_results import load_counties_geo, load_states_geo, _choropleth
        if not os.path.exists(COUNTY_SHP):
            print("  mae_county.png — skipped (county shapefile not found)")
            return
        counties = load_counties_geo(COUNTY_SHP)
        states   = load_states_geo(STATE_SHP, "USA") if os.path.exists(STATE_SHP) else None
        df = county_rmse.copy()
        df["adm2_id"] = df["adm2_id"].astype(str).str.zfill(5)
        merged = counties.merge(df[["adm2_id", "mae"]], left_on="GEOID",
                                right_on="adm2_id", how="left")
        has = merged[merged["mae"].notna()]
        fig, ax = plt.subplots(figsize=(11, 7))
        _choropleth(ax, counties, has, "mae", "viridis_r",
                    f"{crop} — mean MAE by county (PatchTST LOO, all years)",
                    border_gdf=states, county_level=True)
        ax.set_axis_off(); fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  {os.path.basename(out)}  ({len(has)} counties)")
    except Exception as e:
        print(f"  mae_county.png — skipped ({type(e).__name__}: {e})")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python make_crop_plots.py <exp_dir>")
    exp_dir = sys.argv[1]

    errors = pd.read_csv(os.path.join(exp_dir, "loo_all_errors.csv"))
    county_rmse = pd.read_csv(os.path.join(exp_dir, "loo_county_rmse.csv"))
    with open(os.path.join(exp_dir, "loo_meta.json")) as f:
        meta = json.load(f)
    crop = meta.get("crop", "crop")

    m = _metrics(errors["y_true"], errors["y_pred"])
    plots_dir = os.path.join(exp_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    print(f"[plots] {crop}: overall RMSE={m['rmse']:,.0f} R²={m['r2']:.3f} "
          f"RMSE%={m['rmse_pct']:.1f} bias={m['bias']:+,.0f} n={len(errors):,}")

    plot_scatter(errors, crop, m, os.path.join(plots_dir, "scatter_all.png"))
    plot_skill_by_year(meta, crop, os.path.join(plots_dir, "skill_by_year.png"))
    plot_density(errors, crop, os.path.join(plots_dir, "yield_density.png"))
    plot_county_mae(county_rmse, crop, os.path.join(plots_dir, "mae_county.png"))

    folds = sorted(meta["folds"], key=lambda f: f["test_year"])
    summary = {
        "crop": crop,
        "exp_id": meta.get("exp_id"),
        "study_start": meta.get("study_start"),
        "study_end": meta.get("study_end"),
        "n_obs": int(len(errors)),
        "n_counties": int(errors["adm2_id"].nunique()),
        "overall": m,
        "by_year": [
            {"year": f["test_year"], "rmse": f["rmse_overall"], "r2": f["r2_overall"],
             "mean_yield": f["mean_yield"],
             "rmse_pct": 100 * f["rmse_overall"] / f["mean_yield"] if f["mean_yield"] else None,
             "n_test": f["n_test"]}
            for f in folds
        ],
    }
    with open(os.path.join(exp_dir, "crop_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[plots] wrote crop_summary.json + 4 figures to {plots_dir}")


if __name__ == "__main__":
    main()
