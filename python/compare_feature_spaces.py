"""Compare completed LOO experiments that differ only in their feature space.

Point --runs at two or more experiment folders (the run_loo_experiment.py output
roots, or aggregate_seed_runs.py agg_cfg-* folders — both write
loo_all_errors.csv) and this scores them against each other on the samples they
all share.

Why paired, and why bootstrap over years
----------------------------------------
Every arm predicts the same (county, year) test points, so the arms can be
compared pair-by-pair on identical samples instead of by their headline RMSEs.
That removes the sample-composition differences that otherwise dominate small
effects — arms drop slightly different rows depending on which inputs are
available (e.g. a county-year with no NDVI).

The ~28k county-year predictions are NOT independent: a bad year is bad
everywhere at once, and a county's errors persist across years. Treating rows as
independent would report absurdly tight confidence intervals. The honest unit of
resampling is the LOO fold (the year), so the CIs here come from a block
bootstrap that resamples whole years with replacement. With ~17 folds that is a
small sample, which is the point: it keeps the reported uncertainty honest.

Usage
-----
    python compare_feature_spaces.py \\
        --runs climate=/path/loo_corn_tst_..._clim-tasmax-tasmin-pr_2002_2018_<ts> \\
               +ndvi=/path/loo_corn_tst_..._pheno-ndvi_2002_2018_<ts> \\
        --baseline climate --out_dir /path/results/featurespace_compare

A label may be given as LABEL=DIR; a bare DIR gets a label derived from the
folder name. The first --runs entry is the baseline unless --baseline names one.

Outputs (in --out_dir)
    featurespace_overall.csv    one row per arm: rmse, mae, r2, bias, skill
    featurespace_per_fold.csv   fold x arm RMSE, plus which arm won each fold
    featurespace_pairwise.csv   every arm pair: ΔRMSE, bootstrap CI, P(better)
"""

import argparse
import os

import numpy as np
import pandas as pd

# Columns identifying one prediction, shared with aggregate_seed_runs.py.
KEY_COLS = ["adm2_id", "hindcast_start"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Experiment folders to compare, as LABEL=DIR or bare DIR. "
             "Each must contain loo_all_errors.csv.",
    )
    p.add_argument(
        "--baseline", default=None,
        help="Label to score skill against. Default: the first --runs entry.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Where to write the CSVs. Default: alongside the baseline run.",
    )
    p.add_argument(
        "--no_climatology", dest="climatology", action="store_false",
        help="Omit the county-climatology reference arm. It is included by "
             "default because an arm that does not beat it is not yet a useful "
             "yield model, and absolute-yield RMSE hides that.",
    )
    p.add_argument(
        "--n_boot", type=int, default=10000,
        help="Block-bootstrap resamples. Default: 10000.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Bootstrap RNG seed, so a rerun reproduces the CIs. Default: 0.",
    )
    return p.parse_args()


def label_from_dir(path):
    """Derive a readable arm label from an exp_id folder name.

    Keeps the feature-space tags run_loo_experiment.py stamps into the name
    (_clim-…, _yhist, _pheno-…) and drops the crop/model/window/timestamp parts
    that are identical across a well-formed ablation.
    """
    name = os.path.basename(os.path.normpath(path))
    tags = [t for t in name.split("_")
            if t.startswith(("clim-", "yhist", "pheno-"))]
    return "+".join(tags) if tags else name


def load_runs(specs):
    """Read each arm's loo_all_errors.csv. Returns {label: DataFrame keyed by sample}."""
    runs = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:                      # bare DIR, no explicit label
            path, label = label, label_from_dir(label)
        csv = os.path.join(path, "loo_all_errors.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(
                f"loo_all_errors.csv not found in {path!r}. Point --runs at a "
                "completed run_loo_experiment.py or aggregate_seed_runs.py folder."
            )
        if label in runs:
            raise ValueError(f"Duplicate arm label {label!r} — give explicit LABEL=DIR.")
        df = pd.read_csv(csv)
        df["adm2_id"] = df["adm2_id"].astype(str)
        runs[label] = df.set_index([c for c in KEY_COLS])
        print(f"  {label:24} {len(df):>7,} predictions   {path}")
    if len(runs) < 2:
        raise ValueError("Need at least two runs to compare.")
    return runs


def common_index(runs):
    """Intersect the arms on (adm2_id, hindcast_start) and report what it costs."""
    idx = set.intersection(*[set(df.index) for df in runs.values()])
    idx = sorted(idx)
    print(f"\nPaired on {len(idx):,} samples common to all {len(runs)} arms.")
    for label, df in runs.items():
        dropped = len(df) - len(idx)
        if dropped:
            print(f"  {label:24} drops {dropped:,} of its {len(df):,} "
                  f"({100 * dropped / len(df):.2f}%) to pair")
    return idx


CLIMATOLOGY = "county climatology"


def climatology_predictions(idx, y_true):
    """Leave-one-year-out county climatology: what each county normally yields.

    For sample (county c, year t) the prediction is the mean of c's observed
    yields in every OTHER year of the paired set — the same information the LOO
    models are trained on, so it is a fair reference rather than an oracle.
    Counties seen in only one year fall back to the global mean.

    This is the null a yield forecast has to beat. Most of the variance in
    absolute yield is between counties (structural: soil, irrigation,
    management), not between years, so a model can post a respectable R² purely
    by learning county levels while carrying no weather skill at all. Putting
    climatology in the table makes that visible.
    """
    df = pd.DataFrame({
        "adm2_id": [k[KEY_COLS.index("adm2_id")] for k in idx],
        "y": y_true,
    })
    g = df.groupby("adm2_id")["y"]
    total, count = g.transform("sum"), g.transform("count")
    loo = (total - df["y"]) / (count - 1).where(count > 1)
    return loo.fillna(df["y"].mean()).to_numpy()


def score(y_true, y_pred):
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        r2=float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        bias=float(np.mean(err)),
    )


def overall_table(preds, y, baseline):
    """Per-arm metrics on the common samples, plus skill vs baseline and climatology.

    Two skill columns because they answer different questions: skill_vs_baseline
    is the ablation contrast (what did this input buy?), skill_vs_climatology is
    the absolute one (is this a useful model at all?).
    """
    rows = [dict(arm=label, n=len(y), **score(y, p)) for label, p in preds.items()]
    base_rmse = next(r["rmse"] for r in rows if r["arm"] == baseline)
    clim_rmse = next((r["rmse"] for r in rows if r["arm"] == CLIMATOLOGY), None)
    for r in rows:
        r["skill_vs_baseline_pct"] = 100 * (1 - r["rmse"] / base_rmse)
        if clim_rmse:
            is_clim = r["arm"] == CLIMATOLOGY
            r["skill_vs_climatology_pct"] = (None if is_clim
                                             else 100 * (1 - r["rmse"] / clim_rmse))
            r["beats_climatology"] = None if is_clim else r["rmse"] < clim_rmse
    return pd.DataFrame(rows)


def per_fold_table(preds, y, years):
    """RMSE per LOO fold per arm, and the winning arm in each fold."""
    sq = {lab: (p - y) ** 2 for lab, p in preds.items()}
    rows = []
    for yr in np.unique(years):
        m = years == yr
        row = {"fold": int(yr), "n_test": int(m.sum())}
        row.update({lab: float(np.sqrt(sq[lab][m].mean())) for lab in preds})
        row["best"] = min(preds, key=lambda lab: row[lab])
        rows.append(row)
    return pd.DataFrame(rows), sq


def pairwise_table(sq, years, n_boot, seed):
    """Paired ΔRMSE for every arm pair, with a year-block bootstrap CI.

    Resampling whole folds (not rows) respects the within-year correlation of
    yield errors; see the module docstring.
    """
    uyears = np.unique(years)
    by_year = {yr: np.where(years == yr)[0] for yr in uyears}
    rng = np.random.default_rng(seed)

    # One shared set of fold resamples so every pair sees identical bootstrap
    # worlds — the comparisons stay consistent with each other.
    draws = [np.concatenate([by_year[y] for y in rng.choice(uyears, len(uyears), True)])
             for _ in range(n_boot)]

    labels = list(sq)
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            obs = float(np.sqrt(sq[a].mean()) - np.sqrt(sq[b].mean()))
            diffs = np.array([np.sqrt(sq[a][d].mean()) - np.sqrt(sq[b][d].mean())
                              for d in draws])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append(dict(
                arm_a=a, arm_b=b,
                delta_rmse=obs,               # >0 means b is the better arm
                ci_lo=float(lo), ci_hi=float(hi),
                prob_a_better=float(np.mean(diffs < 0)),
                folds_a_better=int(sum(
                    np.sqrt(sq[a][by_year[y]].mean()) < np.sqrt(sq[b][by_year[y]].mean())
                    for y in uyears)),
                n_folds=len(uyears),
                significant=bool(lo > 0 or hi < 0),
            ))
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    print("Loading runs:")
    runs = load_runs(args.runs)
    baseline = args.baseline or next(iter(runs))
    if baseline not in runs:
        raise ValueError(f"--baseline {baseline!r} is not one of {list(runs)}.")

    idx = common_index(runs)
    # hindcast_start is an index level once the frames are keyed, not a column.
    years = np.array([k[KEY_COLS.index("hindcast_start")] for k in idx], dtype=int)
    y = runs[baseline].loc[idx, "y_true"].to_numpy()

    preds = {label: df.loc[idx, "y_pred"].to_numpy() for label, df in runs.items()}
    if args.climatology:
        preds[CLIMATOLOGY] = climatology_predictions(idx, y)

    overall = overall_table(preds, y, baseline)
    per_fold, sq = per_fold_table(preds, y, years)
    pairwise = pairwise_table(sq, years, args.n_boot, args.seed)

    out_dir = args.out_dir or os.path.dirname(
        os.path.normpath(args.runs[0].partition("=")[2] or args.runs[0]))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 74}\nOverall (paired on {len(idx):,} samples, baseline = {baseline})")
    print(overall.to_string(index=False, float_format="{:,.3f}".format))

    print(f"\n{'=' * 74}\nPer-fold RMSE")
    print(per_fold.to_string(index=False, float_format="{:,.0f}".format))
    wins = per_fold["best"].value_counts()
    print("\nFolds won: " + "  ".join(f"{lab}={wins.get(lab, 0)}" for lab in preds))

    if args.climatology:
        beaten = overall[(overall.arm != CLIMATOLOGY) & overall.beats_climatology]
        if beaten.empty:
            print(f"\n!! No arm beats {CLIMATOLOGY} — none of these feature spaces "
                  "is yet a useful yield model, whatever their ranking against "
                  "each other.")

    print(f"\n{'=' * 74}\nPairwise ΔRMSE (positive ⇒ arm_b better), "
          f"{args.n_boot:,}-resample year-block bootstrap")
    print(pairwise.to_string(index=False, float_format="{:,.1f}".format))

    for name, df in [("overall", overall), ("per_fold", per_fold),
                     ("pairwise", pairwise)]:
        path = os.path.join(out_dir, f"featurespace_{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
