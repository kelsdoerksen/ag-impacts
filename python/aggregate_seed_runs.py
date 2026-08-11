"""Pool several LOO runs of the SAME configuration that differ only in seed.

run_loo_experiment.py draws a fresh training seed per run and stamps the folder
with a config fingerprint (cfg-xxxxxxxx) plus the seed (seed<N>). Runs sharing a
config hash are replicates of one experiment arm — same crop, AOI, feature
space, CV scheme, model settings, study window — so their results can be
averaged to report a mean and a spread instead of one arbitrary seed.

Typical use
-----------
    # 5 replicates of one arm
    for s in 1 2 3 4 5; do
        python run_loo_experiment.py --seed $s \\
            --study_years 2002 2018 --model_type tst --use_phenology ndvi \\
            --data_dir /path/data --results_dir /path/results
    done

    # pool them (auto-detects the group when the folder holds only one arm)
    python aggregate_seed_runs.py --runs_dir /path/results

    # or pick the arm explicitly when several arms live side by side
    python aggregate_seed_runs.py --runs_dir /path/results --config_hash 3f9a12c7

    # then plot the pooled result with the existing scripts
    python plot_loo_all.py --exp_dir /path/results/agg_cfg-3f9a12c7 \\
        --county_shp /path/to/counties.shp

Two aggregations are produced, and they answer different questions
------------------------------------------------------------------
1. Spread across seeds (what you report): each seed's metric is computed
   independently, then averaged — agg_fold_summary.csv and agg_summary.json
   carry mean, std, min and max over seeds. loo_county_rmse.csv holds the
   seed-mean per-county RMSE with its std.
2. Seed-ensemble prediction (a bonus, for maps): per county-year the
   predictions are averaged across seeds and the errors recomputed. Averaging
   predictions cancels noise, so the ensemble RMSE is usually LOWER than the
   mean-of-seeds RMSE — do not quote it as the model's error. It is written
   into fold_<year>/ subfolders so plot_loo_all.py and plot_results.py run on
   the aggregate folder unchanged.

Outputs (in --out_dir, default {runs_dir}/agg_cfg-{hash})
    seed_runs.csv          one row per pooled run: seed, overall rmse/mae, path
    agg_fold_metrics.csv   long form: fold x seed -> rmse, r2, n_test
    agg_fold_summary.csv   per fold: mean/std/min/max over seeds
    loo_county_rmse.csv    per county: seed-mean rmse/mae + std (plot-compatible)
    loo_all_errors.csv     seed-ensemble predictions for every county-year
    fold_<label>/          seed-ensemble per-fold outputs (plot-compatible)
    agg_summary.json       headline numbers, config, and the runs that went in
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Columns of a run's loo_all_errors.csv that identify one prediction.
SAMPLE_KEYS = ["adm2_id", "hindcast_start"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--runs_dir",
        help="Directory to scan (recursively) for completed run folders. Runs "
             "are grouped by config hash; pass --config_hash when more than one "
             "group is present.",
    )
    src.add_argument(
        "--run_dirs", nargs="+",
        help="Explicit list of run folders to pool, bypassing the scan. Their "
             "config hashes must match unless --allow_mixed_config is set.",
    )
    p.add_argument(
        "--config_hash",
        help="Pool only runs with this config hash (the cfg-xxxxxxxx tag in the "
             "folder name). Required when --runs_dir holds several arms.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Where to write the pooled outputs. "
             "Default: {runs_dir}/agg_cfg-{hash}.",
    )
    p.add_argument(
        "--min_seeds", type=int, default=2,
        help="Fail if fewer than this many runs are found. Default: 2.",
    )
    p.add_argument(
        "--allow_mixed_config", action="store_true",
        help="Permit pooling runs whose config hashes differ. Off by default — "
             "averaging different feature spaces or study windows together "
             "silently produces a meaningless number.",
    )
    p.add_argument(
        "--allow_duplicate_seeds", action="store_true",
        help="Permit two runs with the same seed in one group. Off by default: "
             "a repeated seed is usually an accidentally re-run replicate, and "
             "it would be double-counted in the spread.",
    )
    p.add_argument(
        "--no_fold_dirs", action="store_true",
        help="Skip writing the seed-ensemble fold_<label>/ subfolders.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List the config-hash groups found under --runs_dir and exit "
             "without aggregating.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_run_dirs(runs_dir):
    """Recursively locate completed run folders under runs_dir.

    A run folder is one holding loo_meta.json. Descent stops there so per-fold
    subfolders are never mistaken for runs.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(runs_dir):
        if "loo_meta.json" in filenames:
            found.append(dirpath)
            dirnames[:] = []          # don't descend into fold_*/
    return sorted(found)


def read_run(run_dir):
    """Load one run's identity and metrics. Returns None if it never finished.

    config_hash and seed come from run_diagnostics.json when present, else from
    loo_meta.json (runs written before seed support have neither, and are
    reported as unknown so they can't be silently pooled).
    """
    meta_path = os.path.join(run_dir, "loo_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    diag = None
    diag_path = os.path.join(run_dir, "run_diagnostics.json")
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diag = json.load(f)

    src = diag or meta
    return {
        "run_dir":      run_dir,
        "exp_id":       meta.get("exp_id", os.path.basename(run_dir)),
        "config_hash":  src.get("config_hash"),
        "seed":         src.get("seed"),
        "cv_scheme":    meta.get("cv_scheme", "year"),
        "overall_rmse": meta.get("overall_rmse"),
        "overall_mae":  meta.get("overall_mae"),
        "n_folds":      meta.get("n_folds"),
        "folds":        meta.get("folds", []),
        "config":       (diag or {}).get("config"),
        "meta":         meta,
    }


def group_by_config(runs):
    """Bucket runs by config hash, with unknown hashes kept separate."""
    groups = {}
    for run in runs:
        groups.setdefault(run["config_hash"] or "unknown", []).append(run)
    return groups


def select_runs(args):
    """Resolve the CLI arguments to the list of runs to pool."""
    if args.run_dirs:
        runs = [read_run(d) for d in args.run_dirs]
        missing = [d for d, r in zip(args.run_dirs, runs) if r is None]
        if missing:
            raise FileNotFoundError(
                f"No loo_meta.json in: {missing}. Point --run_dirs at completed "
                "run_loo_experiment.py output folders."
            )
        hashes = {r["config_hash"] for r in runs}
        if len(hashes) > 1 and not args.allow_mixed_config:
            raise ValueError(
                f"--run_dirs span {len(hashes)} different configurations "
                f"({sorted(map(str, hashes))}). These are not seed replicates of "
                "one arm. Pass --allow_mixed_config only if you are sure."
            )
        return runs, (hashes.pop() if len(hashes) == 1 else "mixed")

    runs = [r for r in (read_run(d) for d in find_run_dirs(args.runs_dir))
            if r is not None]
    if not runs:
        raise FileNotFoundError(
            f"No completed runs (loo_meta.json) found under {args.runs_dir!r}."
        )
    groups = group_by_config(runs)

    if args.config_hash:
        if args.config_hash not in groups:
            raise ValueError(
                f"No runs with config hash {args.config_hash!r} under "
                f"{args.runs_dir!r}. Found: {sorted(groups)}"
            )
        return groups[args.config_hash], args.config_hash

    if len(groups) > 1:
        summary = "\n".join(
            f"  {h}  ({len(g)} run(s))  e.g. {os.path.basename(g[0]['run_dir'])}"
            for h, g in sorted(groups.items())
        )
        raise ValueError(
            f"{len(groups)} different configurations found under "
            f"{args.runs_dir!r} — pass --config_hash to pick one:\n{summary}"
        )

    only = next(iter(groups))
    return groups[only], only


def describe_groups(runs_dir):
    """Print the config-hash groups under runs_dir (for --list)."""
    runs = [r for r in (read_run(d) for d in find_run_dirs(runs_dir))
            if r is not None]
    groups = group_by_config(runs)
    print(f"\n{len(runs)} run(s) in {len(groups)} configuration group(s) "
          f"under {runs_dir}\n")
    for h, g in sorted(groups.items()):
        seeds = [r["seed"] for r in g]
        rmses = [r["overall_rmse"] for r in g if r["overall_rmse"] is not None]
        spread = (f"  overall RMSE {np.mean(rmses):,.0f} ± {np.std(rmses, ddof=1):,.0f}"
                  if len(rmses) > 1 else
                  (f"  overall RMSE {rmses[0]:,.0f}" if rmses else ""))
        print(f"{h}  {len(g)} run(s)  seeds={seeds}{spread}")
        for r in sorted(g, key=lambda r: r["exp_id"]):
            print(f"    {os.path.basename(r['run_dir'])}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _std(series):
    """Sample std (ddof=1); NaN for a single value rather than 0."""
    return series.std(ddof=1) if len(series) > 1 else np.nan


def fold_tables(runs):
    """Long per-(fold, seed) metrics table and its across-seed summary."""
    rows = []
    for run in runs:
        for fold in run["folds"]:
            rows.append({
                "fold_label": fold.get("fold_label"),
                "seed":       run["seed"],
                "exp_id":     run["exp_id"],
                "rmse":       fold.get("rmse_overall"),
                "r2":         fold.get("r2_overall"),
                "mean_yield": fold.get("mean_yield"),
                "n_test":     fold.get("n_test"),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long

    summary = (
        long.groupby("fold_label")
        .agg(n_seeds  =("seed",   "nunique"),
             rmse_mean=("rmse",   "mean"),
             rmse_std =("rmse",   _std),
             rmse_min =("rmse",   "min"),
             rmse_max =("rmse",   "max"),
             r2_mean  =("r2",     "mean"),
             r2_std   =("r2",     _std),
             n_test   =("n_test", "max"))
        .reset_index()
    )
    # Standard error of the mean over seeds — the bar to draw when plotting the
    # fold means, as distinct from rmse_std (the spread of individual seeds).
    summary["rmse_sem"] = summary["rmse_std"] / np.sqrt(summary["n_seeds"])
    summary["rmse_cv_pct"] = 100 * summary["rmse_std"] / summary["rmse_mean"]
    # Sort numerically for year folds, lexically otherwise.
    numeric = pd.to_numeric(summary["fold_label"], errors="coerce")
    summary = (summary.assign(_k=numeric)
               .sort_values(["_k", "fold_label"]).drop(columns="_k")
               .reset_index(drop=True))
    return long, summary


def county_table(runs):
    """Seed-mean per-county RMSE/MAE with across-seed spread.

    Reads each run's loo_county_rmse.csv (already aggregated over folds within
    that run) and averages the per-county metrics over seeds. adm2_id is read as
    a string so FIPS codes keep their leading zeros.
    """
    frames = []
    for run in runs:
        path = os.path.join(run["run_dir"], "loo_county_rmse.csv")
        if not os.path.exists(path):
            print(f"  [warn] no loo_county_rmse.csv in {run['exp_id']} — skipped")
            continue
        df = pd.read_csv(path, dtype={"adm2_id": str})
        df["seed"] = run["seed"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    allc = pd.concat(frames, ignore_index=True)
    out = (
        allc.groupby(["adm2_id", "adm2", "adm1"])
        .agg(rmse    =("rmse",  "mean"),
             rmse_std=("rmse",  _std),
             rmse_min=("rmse",  "min"),
             rmse_max=("rmse",  "max"),
             mae     =("mae",   "mean"),
             mae_std =("mae",   _std),
             n_obs   =("n_obs", "max"),
             n_seeds =("seed",  "nunique"))
        .reset_index()
        .sort_values("rmse", ascending=False)
    )
    return out


def ensemble_errors(runs):
    """Average each county-year's prediction across seeds and recompute errors.

    y_true is identical across replicates (same data, same folds), so the first
    value is kept; y_pred is averaged and the residual/abs/sq errors are derived
    from that mean. y_pred_std records the across-seed disagreement per sample,
    which is a per-county-year measure of model instability.
    """
    frames = []
    for run in runs:
        path = os.path.join(run["run_dir"], "loo_all_errors.csv")
        if not os.path.exists(path):
            print(f"  [warn] no loo_all_errors.csv in {run['exp_id']} — skipped")
            continue
        df = pd.read_csv(path, dtype={"adm2_id": str})
        df["seed"] = run["seed"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    all_err = pd.concat(frames, ignore_index=True)
    agg = (
        all_err.groupby(SAMPLE_KEYS)
        .agg(adm2      =("adm2",   "first"),
             adm1      =("adm1",   "first"),
             y_true    =("y_true", "first"),
             y_pred    =("y_pred", "mean"),
             y_pred_std=("y_pred", _std),
             n_seeds   =("seed",   "nunique"))
        .reset_index()
    )
    agg["residual"]  = agg["y_pred"] - agg["y_true"]
    agg["abs_error"] = agg["residual"].abs()
    agg["sq_error"]  = agg["residual"] ** 2

    n_max = agg["n_seeds"].max()
    partial = int((agg["n_seeds"] < n_max).sum())
    if partial:
        print(f"  [warn] {partial:,} county-year sample(s) are absent from some "
              f"seeds (fewer than {n_max} runs cover them); they are averaged "
              "over the seeds that do have them.")

    cols = (SAMPLE_KEYS[:1] + ["adm2", "adm1", "hindcast_start", "y_true",
                               "y_pred", "y_pred_std", "residual", "abs_error",
                               "sq_error", "n_seeds"])
    return agg[cols]


def write_fold_dirs(out_dir, ens_errors, runs, config_hash):
    """Write per-fold ensemble outputs so the existing plot scripts just work.

    Only the 'year' CV scheme is reconstructable here: loo_all_errors.csv
    carries hindcast_start but not climate_cell_id, so cell folds cannot be
    rebuilt from it.
    """
    schemes = {r["cv_scheme"] for r in runs}
    if schemes != {"year"}:
        print(f"  [skip] fold folders: cv_scheme(s) {sorted(schemes)} — only "
              "'year' folds can be rebuilt from loo_all_errors.csv.")
        return 0

    template = runs[0]["folds"]
    by_label = {str(f.get("fold_label")): f for f in template}
    n_written = 0

    for year, grp in ens_errors.groupby("hindcast_start"):
        label = str(year)
        fold_dir = os.path.join(out_dir, f"fold_{label}")
        os.makedirs(fold_dir, exist_ok=True)
        grp = grp.reset_index(drop=True)

        mean_yield = float(grp["y_true"].mean())
        rmse = float(np.sqrt(grp["sq_error"].mean()))
        if len(grp) > 1:
            ss_res = float(((grp["y_true"] - grp["y_pred"]) ** 2).sum())
            ss_tot = float(((grp["y_true"] - grp["y_true"].mean()) ** 2).sum())
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")

        grp.to_csv(os.path.join(fold_dir, "adm2_errors.csv"), index=False)
        (grp.groupby("adm1")["sq_error"].mean().pipe(np.sqrt)
            .rename("rmse").reset_index()
            .assign(rmse_pct=lambda d: 100 * d["rmse"] / mean_yield)
            .sort_values("rmse", ascending=False)
            .to_csv(os.path.join(fold_dir, "adm1_rmse.csv"), index=False))

        src = by_label.get(label, {})
        fold_meta = {
            "run_id":       f"ensemble_fold_{label}",
            "fold_label":   label,
            "test_year":    int(year),
            "cv_scheme":    "year",
            "model_type":   src.get("model_type", runs[0]["meta"].get("model_type")),
            "aggregation":  "seed_ensemble_mean_prediction",
            "config_hash":  config_hash,
            "seeds":        [r["seed"] for r in runs],
            "n_seeds":      len(runs),
            "rmse_overall": rmse,
            "r2_overall":   r2,
            "pct_error":    None,
            "mean_yield":   mean_yield,
            "n_train":      src.get("n_train"),
            "n_test":       int(len(grp)),
            # Averaging predictions has no single feature-importance vector.
            "feature_importances": None,
        }
        with open(os.path.join(fold_dir, "run_meta.json"), "w") as f:
            json.dump(fold_meta, f, indent=2)
        n_written += 1

    return n_written


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list:
        if not args.runs_dir:
            raise ValueError("--list requires --runs_dir.")
        describe_groups(args.runs_dir)
        return

    runs, cfg_hash = select_runs(args)
    runs = sorted(runs, key=lambda r: (r["seed"] is None, r["seed"], r["exp_id"]))

    if len(runs) < args.min_seeds:
        raise ValueError(
            f"Only {len(runs)} run(s) found for config {cfg_hash!r}, "
            f"--min_seeds is {args.min_seeds}. Run more replicates "
            "(same arguments, different --seed) or lower --min_seeds."
        )

    seeds = [r["seed"] for r in runs]
    if None in seeds:
        print("  [warn] some runs record no seed (written before --seed "
              "support). They are pooled, but were probably all trained with "
              "the old hardcoded 42 — i.e. duplicates, not replicates.")
    known = [s for s in seeds if s is not None]
    dupes = {s for s in known if known.count(s) > 1}
    if dupes and not args.allow_duplicate_seeds:
        raise ValueError(
            f"Seed(s) {sorted(dupes)} appear in more than one run of config "
            f"{cfg_hash!r}. Identical seeds are duplicates, not replicates, and "
            "would skew the spread. Drop the extra folder(s) or pass "
            "--allow_duplicate_seeds."
        )

    out_dir = args.out_dir or os.path.join(
        args.runs_dir or os.path.dirname(os.path.abspath(runs[0]["run_dir"])),
        f"agg_cfg-{cfg_hash}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nConfig hash : {cfg_hash}")
    print(f"Runs pooled : {len(runs)}  (seeds: {seeds})")
    for r in runs:
        rmse = r["overall_rmse"]
        print(f"  seed {str(r['seed']):>10}  RMSE {rmse:>12,.1f}  "
              f"{os.path.basename(r['run_dir'])}")
    print(f"Output dir  : {out_dir}\n")

    # ── one row per run ──
    seed_runs = pd.DataFrame([{
        "seed":         r["seed"],
        "exp_id":       r["exp_id"],
        "overall_rmse": r["overall_rmse"],
        "overall_mae":  r["overall_mae"],
        "n_folds":      r["n_folds"],
        "run_dir":      r["run_dir"],
    } for r in runs])
    seed_runs.to_csv(os.path.join(out_dir, "seed_runs.csv"), index=False)

    # ── per-fold spread across seeds ──
    long, summary = fold_tables(runs)
    if not long.empty:
        long.to_csv(os.path.join(out_dir, "agg_fold_metrics.csv"), index=False)
        summary.to_csv(os.path.join(out_dir, "agg_fold_summary.csv"), index=False)

    # ── per-county spread across seeds (plot_loo_county_rmse.py compatible) ──
    counties = county_table(runs)
    if not counties.empty:
        counties.to_csv(os.path.join(out_dir, "loo_county_rmse.csv"), index=False)

    # ── seed-ensemble predictions ──
    ens = ensemble_errors(runs)
    n_fold_dirs = 0
    if not ens.empty:
        ens.to_csv(os.path.join(out_dir, "loo_all_errors.csv"), index=False)
        if not args.no_fold_dirs:
            n_fold_dirs = write_fold_dirs(out_dir, ens, runs, cfg_hash)

    # ── headline numbers ──
    run_rmse = np.array([r["overall_rmse"] for r in runs], dtype=float)
    run_mae  = np.array([r["overall_mae"] for r in runs], dtype=float)
    ens_rmse = float(np.sqrt(ens["sq_error"].mean())) if not ens.empty else None
    ens_mae  = float(ens["abs_error"].mean()) if not ens.empty else None

    agg_summary = {
        "config_hash":   cfg_hash,
        "n_seeds":       len(runs),
        "seeds":         seeds,
        "config":        runs[0]["config"],
        "cv_scheme":     runs[0]["cv_scheme"],
        # The numbers to report: each seed scored on its own, then averaged.
        "overall_rmse_mean": float(run_rmse.mean()),
        "overall_rmse_std":  float(run_rmse.std(ddof=1)) if len(runs) > 1 else None,
        "overall_rmse_min":  float(run_rmse.min()),
        "overall_rmse_max":  float(run_rmse.max()),
        "overall_mae_mean":  float(run_mae.mean()),
        "overall_mae_std":   float(run_mae.std(ddof=1)) if len(runs) > 1 else None,
        # Averaged predictions, not an average of metrics — lower by
        # construction, so keep it separate from the reported figure.
        "ensemble_rmse":     ens_rmse,
        "ensemble_mae":      ens_mae,
        "runs": [{"seed": r["seed"], "exp_id": r["exp_id"],
                  "run_dir": r["run_dir"], "overall_rmse": r["overall_rmse"]}
                 for r in runs],
    }
    with open(os.path.join(out_dir, "agg_summary.json"), "w") as f:
        json.dump(agg_summary, f, indent=2)

    # ── console report ──
    std_str = ("" if agg_summary["overall_rmse_std"] is None
               else f" ± {agg_summary['overall_rmse_std']:,.1f}")
    print(f"{'=' * 68}")
    print(f"Pooled over {len(runs)} seeds — config {cfg_hash}")
    print(f"{'=' * 68}")
    print(f"  Overall RMSE (mean of seeds) : "
          f"{agg_summary['overall_rmse_mean']:,.1f}{std_str}")
    print(f"  Overall MAE  (mean of seeds) : {agg_summary['overall_mae_mean']:,.1f}")
    if ens_rmse is not None:
        print(f"  Seed-ensemble RMSE           : {ens_rmse:,.1f}  "
              "(averaged predictions — not the reported error)")
    if not summary.empty:
        print(f"\n  {'fold':>10}  {'RMSE mean':>12}  {'std':>10}  "
              f"{'cv%':>6}  {'seeds':>5}")
        for _, row in summary.iterrows():
            std = "        —" if np.isnan(row["rmse_std"]) else f"{row['rmse_std']:>10,.1f}"
            cv  = "     —" if np.isnan(row["rmse_cv_pct"]) else f"{row['rmse_cv_pct']:>6.1f}"
            print(f"  {row['fold_label']:>10}  {row['rmse_mean']:>12,.1f}  "
                  f"{std}  {cv}  {int(row['n_seeds']):>5}")

    print(f"\nWrote to {out_dir}:")
    print("  seed_runs.csv  agg_fold_metrics.csv  agg_fold_summary.csv")
    print("  loo_county_rmse.csv  loo_all_errors.csv  agg_summary.json")
    if n_fold_dirs:
        print(f"  {n_fold_dirs} seed-ensemble fold_<year>/ folder(s)")
    print(f"\nTo plot the pooled result:\n"
          f"  python plot_loo_all.py --exp_dir {out_dir} [--county_shp ...]")


if __name__ == "__main__":
    main()
