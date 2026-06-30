"""
Binned yield error analysis and distribution comparison.

Reads an errors CSV (adm2_errors.csv from train_eval.py, or
loo_all_errors.csv from run_loo_experiment.py) and produces:

    rmse_by_bin.png   — RMSE and bias per observed-yield bin
    yield_density.png — KDE of observed vs predicted yield
    bin_stats.csv     — per-bin n, mean_yield, rmse, mae, bias

Example:
    python bin_analysis.py \\
        --errors_csv /path/to/loo_all_errors.csv \\
        --save_dir   /path/to/results \\
        --n_bins 10
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binned RMSE analysis and yield distribution comparison."
    )
    parser.add_argument(
        "--errors_csv", type=str, required=True,
        help="Path to adm2_errors.csv or loo_all_errors.csv.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True,
        help="Directory to save output plots and CSV.",
    )
    parser.add_argument(
        "--n_bins", type=int, default=10,
        help="Number of equal-width observed-yield bins. Default: 10.",
    )
    return parser.parse_args()


def load_errors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"y_true", "y_pred"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path!r}: {missing}")
    return df.dropna(subset=["y_true", "y_pred"]).copy()


def compute_bin_stats(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = df.copy()
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
    # Human-readable label: midpoint of each bin interval
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

    # --- RMSE bars ---
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
    ax_rmse.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))

    # --- Bias bars (positive = over-predict, negative = under-predict) ---
    colours = ["#d62728" if b > 0 else "#1f77b4" for b in stats["bias"]]
    ax_bias.bar(x, stats["bias"], color=colours, edgecolor="white", linewidth=0.5)
    ax_bias.axhline(0, color="black", linewidth=0.8)
    ax_bias.set_ylabel("Mean bias\n(pred − obs)")
    ax_bias.set_xlabel("Observed yield bin midpoint")
    ax_bias.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))

    ax_bias.set_xticks(x)
    ax_bias.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Legend for bias colours
    from matplotlib.patches import Patch
    ax_bias.legend(
        handles=[
            Patch(color="#d62728", label="Over-predict"),
            Patch(color="#1f77b4", label="Under-predict"),
        ],
        fontsize=8, loc="upper left",
    )

    plt.tight_layout()
    out = os.path.join(save_dir, "rmse_by_bin.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  rmse_by_bin.png")


def plot_yield_density(df: pd.DataFrame, save_dir: str) -> None:
    """KDE of observed and predicted yield on the same axes."""
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    x_min = min(y_true.min(), y_pred.min())
    x_max = max(y_true.max(), y_pred.max())
    xs    = np.linspace(x_min, x_max, 600)

    kde_true = gaussian_kde(y_true)
    kde_pred = gaussian_kde(y_pred)
    dens_true = kde_true(xs)
    dens_pred = kde_pred(xs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, dens_true, label="Observed", color="steelblue", linewidth=2)
    ax.plot(xs, dens_pred, label="Predicted", color="darkorange",
            linewidth=2, linestyle="--")
    ax.fill_between(xs, dens_true, dens_pred, alpha=0.15, color="grey",
                    label="Difference")

    # Annotate means
    for vals, colour, label in [
        (y_true, "steelblue", "Observed"),
        (y_pred, "darkorange", "Predicted"),
    ]:
        mu = vals.mean()
        ax.axvline(mu, color=colour, linewidth=1, linestyle=":")
        ax.text(mu, ax.get_ylim()[1] * 0.02, f"{label} mean\n{mu:,.0f}",
                color=colour, fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Yield")
    ax.set_ylabel("Density")
    ax.set_title("Yield distribution: observed vs predicted (KDE)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(save_dir, "yield_density.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  yield_density.png")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    df    = load_errors(args.errors_csv)
    stats = compute_bin_stats(df, args.n_bins)

    print(f"\nBin stats (n_bins={args.n_bins}, n_total={len(df):,}):")
    print(
        stats[["bin", "n", "mean_yield", "rmse", "mae", "bias"]]
        .to_string(index=False, float_format="{:,.1f}".format)
    )

    stats.to_csv(os.path.join(args.save_dir, "bin_stats.csv"), index=False)
    print(f"\nOutputs:")
    print(f"  bin_stats.csv")
    plot_rmse_by_bin(stats, args.save_dir, args.n_bins)
    plot_yield_density(df, args.save_dir)
    print(f"\nAll outputs saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
