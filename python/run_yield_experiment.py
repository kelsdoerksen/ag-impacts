#!/usr/bin/env python
"""
Train and score a tabular yield model against a named feature source.

One command answers one question: how much of next season's yield can be
predicted from THIS set of engineered features. Swap --source era5 for
--source dcpp:ens and nothing else changes, so the difference in score is the
cost of forecasting the weather rather than observing it.

    source      era5 | dcpp:<member> | dcpp:ens      (yield_features.py)
    scope       one model per AOI, or --combine for a single pooled model
    split       --test-year Y (one held-out year) or --loo (every year in turn)
    model       rf | gbm | xgb | ridge | enet         (models.py)
    extras      --add ndvi,fpar,soil_moisture,yield_history

FOLDS ARE WHOLE YEARS, ALWAYS

The districts of a country share their weather, so their residuals are
correlated within a year: for Zambian maize r ~ 0.28. Holding out random ROWS
leaves ~70 near-duplicates of each test row in the training set and reports a
score that has nothing to do with forecasting. Every split here removes a
calendar year entirely, which is also what a real forecast faces.

WHAT THE METRICS MEAN

Absolute yield is dominated by which district a row is from -- a productive
county out-yields a marginal one by more than weather ever moves either. A
model can therefore post a healthy R2 having learned only the map. Three
numbers separate that from forecast skill:

    r2            absolute, on kg/ha. Includes the between-unit spread.
    rmse_clim     the per-unit climatology null: predict each unit's own mean
                  (or its fitted trend, with --detrend), fit on TRAINING years
                  only. This is the bar a forecast has to clear.
    skill         1 - rmse/rmse_clim. Positive means the features beat
                  climatology; negative means they are worse than knowing
                  nothing but the district.
    r_anom        correlation after each unit's own baseline is removed from
                  both sides -- the interannual signal, and the number that
                  corresponds to r_anom in compare_dcpp_era5.py.
    r_space       within one year, across units. Reported per fold: it answers
                  "did the model get the geography of this year's anomaly
                  right", which r_anom pooled over folds can hide.

The baseline is re-fit inside every fold on that fold's training rows. Fitting
it once on all years would let the held-out year set its own reference, which
inflates every anomaly metric above.

Examples
--------
    # ERA5 features, US only, leave-one-year-out, random forest
    python python/run_yield_experiment.py --aois US --source era5 --loo

    # the same experiment on the FY1 hindcast ensemble mean
    python python/run_yield_experiment.py --aois US --source dcpp:ens \
        --forecast-year 1 --loo

    # pool the small AOIs to get the sample count up, hold out 2018
    python python/run_yield_experiment.py --aois HU,RO,FR --combine \
        --source era5 --test-year 2018 --add ndvi,yield_history
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import CountyBaseline
from models import SUPPORTED_MODELS, YieldModel
from yield_features import COVARIATES, DEFAULT_SHARE, Roots, build, parse_source

# A fold with a handful of rows produces a correlation that is mostly noise.
MIN_TEST_ROWS = 5


def parse_years(spec: str) -> list[int]:
    """'2002-2022', '2004,2006', or a mixture."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out |= set(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _r(a, b) -> float:
    """Pearson r, NaN rather than an exception when it is undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def score(df: pd.DataFrame) -> dict:
    """Metrics for a set of predictions already carrying baseline columns.

    `df` needs y_true, y_pred, y_clim, unit, harvest_year -- all in kg/ha, so
    an anomaly-target run is inverted back before it gets here and every number
    below is comparable across targets.
    """
    y, p, c = df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["y_clim"].to_numpy()
    rmse, rmse_c = _rmse(y, p), _rmse(y, c)
    return {
        "n": int(len(df)),
        "n_units": int(df["unit"].nunique()),
        "mean_yield": float(y.mean()),
        "rmse": rmse,
        "rmse_pct": 100.0 * rmse / float(y.mean()) if y.mean() else float("nan"),
        "mae": float(np.mean(np.abs(y - p))),
        "r2": _r2(y, p),
        "rmse_clim": rmse_c,
        "skill": 1.0 - rmse / rmse_c if rmse_c else float("nan"),
        "r_raw": _r(y, p),
        "r_anom": _r(y - c, p - c),
        "r2_anom": _r2(y - c, p - c),
        "bias": float(np.mean(p - y)),
        "sd_ratio_anom": (float(np.std(p - c) / np.std(y - c))
                          if np.std(y - c) else float("nan")),
    }


def fit_fold(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
             model_type: str, target: str, seed: int,
             detrend: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Fit on `train`, predict `test`, return predictions in kg/ha.

    The per-unit baseline is fit here, on training rows only, and is used for
    two separate purposes: as the reference every anomaly metric is taken
    against, and -- when --target anomaly -- as the offset the model actually
    learns on. Both have to come from the same training-only fit.
    """
    base = CountyBaseline(detrend=detrend).fit(
        train["unit"].to_numpy(), train["yield"].to_numpy(),
        years=train["harvest_year"].to_numpy() if detrend else None)

    yr_tr = train["harvest_year"].to_numpy() if detrend else None
    yr_te = test["harvest_year"].to_numpy() if detrend else None
    clim_te = base.inverse(test["unit"].to_numpy(), np.zeros(len(test)),
                           standardize=False, years=yr_te)

    y_tr = train["yield"].to_numpy(float)
    if target == "anomaly":
        y_tr = base.transform(train["unit"].to_numpy(), y_tr,
                              standardize=False, years=yr_tr)

    model = YieldModel(model_type, random_state=seed)
    # groups = the training rows' years, so a CV-tuned linear model picks its
    # penalty by holding out whole years too.
    model.fit(train[feats], y_tr, groups=train["harvest_year"].to_numpy())

    pred = np.asarray(model.predict(test[feats]), float)
    if target == "anomaly":
        pred = base.inverse(test["unit"].to_numpy(), pred,
                            standardize=False, years=yr_te)

    out = test[["aoi", "unit", "adm2_id", "harvest_year", "yield"]].copy()
    out = out.rename(columns={"yield": "y_true"})
    out["y_pred"] = pred
    out["y_clim"] = clim_te
    out["residual"] = out["y_pred"] - out["y_true"]
    imp = model.feature_importances_
    return out, (np.asarray(imp, float) if imp is not None else None)


def run_scope(df: pd.DataFrame, feats: list[str], folds: list[int], label: str,
              a) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every fold for one model scope (one AOI, or the pooled set)."""
    preds, per_fold, imps = [], [], []
    for yr in folds:
        test = df[df["harvest_year"] == yr]
        train = df[df["harvest_year"] != yr]
        if len(test) < MIN_TEST_ROWS or train["harvest_year"].nunique() < 3:
            print(f"    {yr}: SKIPPED -- {len(test)} test row(s), "
                  f"{train['harvest_year'].nunique()} training year(s)")
            continue
        p, imp = fit_fold(train, test, feats, a.model, a.target, a.seed, a.detrend)
        m = score(p)
        # Within this one year, across units: did the model place the anomaly
        # correctly in space.
        m["r_space"] = _r(p["y_true"] - p["y_clim"], p["y_pred"] - p["y_clim"])
        per_fold.append({"scope": label, "test_year": yr,
                         "n_train": int(len(train)), **m})
        preds.append(p.assign(scope=label, test_year=yr))
        if imp is not None and len(imp) == len(feats):
            imps.append(imp)
        print(f"    {yr}: n={m['n']:>5}  rmse={m['rmse']:>8,.0f}  "
              f"skill={m['skill']:+.3f}  r_space={m['r_space']:+.3f}")

    if not preds:
        raise SystemExit(f"[{label}] no fold produced predictions")

    allp = pd.concat(preds, ignore_index=True)
    overall = {"scope": label, "test_year": "ALL",
               "n_folds": len(per_fold), **score(allp)}
    # Mean of the per-fold spatial correlations: the pooled r_anom above mixes
    # the spatial and interannual parts, this one is purely spatial.
    overall["r_space_mean"] = float(np.nanmean([f["r_space"] for f in per_fold]))

    imp_df = pd.DataFrame(columns=["scope", "feature", "importance"])
    if imps:
        mean_imp = np.mean(np.vstack(imps), axis=0)
        imp_df = (pd.DataFrame({"scope": label, "feature": feats,
                                "importance": mean_imp})
                  .sort_values("importance", ascending=False))
    return allp, pd.DataFrame(per_fold + [overall]), imp_df


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("what to read")
    g.add_argument("--aois", required=True, help="comma-separated, e.g. US,HU,ZM")
    g.add_argument("--source", default="era5",
                   help="era5 | dcpp:<member> | dcpp:ens")
    g.add_argument("--forecast-year", type=int, default=None,
                   help="DCPP lead time; default 0 for era5, 1 for dcpp")
    g.add_argument("--crop", default="corn")
    g.add_argument("--reduce", choices=["season", "blocks"], default="season",
                   help="whole-season aggregates, or season-relative windows")
    g.add_argument("--years", default="2002-2022", help="harvest years")
    g.add_argument("--add", default="", help=f"covariates: {','.join(COVARIATES)}")
    g.add_argument("--phenology-lag", type=int, default=1,
                   help="seasons back for ndvi/fpar/soil_moisture (>=1)")
    g.add_argument("--phenology-suffix", default=None)
    g.add_argument("--yield-lags", type=int, default=3)
    g.add_argument("--share", default=DEFAULT_SHARE)
    g.add_argument("--era5-root"), g.add_argument("--dcpp-root")
    g.add_argument("--targets-root", help="holds {AOI}_processed_data")

    g = p.add_argument_group("how to fit")
    g.add_argument("--combine", action="store_true",
                   help="one pooled model over all AOIs (default: one each)")
    g.add_argument("--test-year", type=int, help="hold out this year only")
    g.add_argument("--loo", action="store_true",
                   help="leave-one-year-out over every available year")
    g.add_argument("--model", choices=SUPPORTED_MODELS, default="rf")
    g.add_argument("--target", choices=["yield", "anomaly"], default="yield",
                   help="'anomaly' subtracts each unit's training-only baseline "
                        "before fitting, so the model spends its capacity on "
                        "interannual variation instead of the map")
    g.add_argument("--detrend", action="store_true",
                   help="per-unit baseline is a line in year, not a flat mean. "
                        "Use it where a technology trend dominates (US maize)")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default=None,
                   help="default {share}/experiments")
    g.add_argument("--tag", default=None, help="appended to the run directory name")
    g.add_argument("--go", action="store_true",
                   help="without this, print the plan and exit")
    a = p.parse_args(argv)

    if bool(a.test_year) == bool(a.loo):
        raise SystemExit("choose exactly one of --test-year Y or --loo")
    try:
        kind, member = parse_source(a.source)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if a.forecast_year is None:
        a.forecast_year = 0 if kind == "era5" else 1
    if kind == "era5" and a.forecast_year != 0:
        raise SystemExit("ERA5 is observed: --forecast-year must be 0")
    if kind == "dcpp" and a.forecast_year < 1:
        raise SystemExit("a hindcast needs --forecast-year >= 1")

    aois = [x.strip() for x in a.aois.split(",") if x.strip()]
    covs = [x.strip() for x in a.add.split(",") if x.strip()]
    bad = [c for c in covs if c not in COVARIATES]
    if bad:
        raise SystemExit(f"unknown --add {bad}; choose from {COVARIATES}")
    years = parse_years(a.years)

    roots = Roots.resolve(a.share, a.era5_root, a.dcpp_root, a.targets_root)
    out_root = Path(a.out_dir) if a.out_dir else roots.share / "experiments"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = f"test{a.test_year}" if a.test_year else "loo"
    scope_tag = "pooled" if a.combine else "peraoi"
    run_id = "_".join(filter(None, [
        a.source.replace(":", "-"), f"fy{a.forecast_year}", a.model, a.target,
        scope_tag, split, "-".join(aois) if len(aois) <= 4 else f"{len(aois)}aois",
        a.tag, stamp]))
    run_dir = out_root / run_id

    print("=" * 78)
    print("yield experiment")
    print("=" * 78)
    print(f"  AOIs          : {', '.join(aois)}"
          f"{'  (pooled into one model)' if a.combine else '  (one model each)'}")
    print(f"  source        : {a.source}   forecast year {a.forecast_year}")
    print(f"  features      : {a.reduce}"
          f"{'  + ' + ', '.join(covs) if covs else ''}")
    print(f"  harvest years : {years[0]}-{years[-1]}")
    print(f"  split         : {'leave-one-year-out' if a.loo else f'hold out {a.test_year}'}")
    print(f"  model         : {a.model}   target {a.target}"
          f"{'   detrended baseline' if a.detrend else ''}")
    print(f"  out           : {run_dir}")
    if not a.go:
        print("\nPlan only. Re-run with --go.")
        return 0

    print("\nassembling:")
    try:
        df, feats, infos = build(roots, aois, a.source, a.crop, a.forecast_year,
                                 a.reduce, years, covs, a.phenology_lag,
                                 a.yield_lags, a.phenology_suffix)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"\ncannot assemble the design matrix:\n{exc}")

    # Pooled absolute-yield fits need some way to tell the AOIs apart, or the
    # model attributes a national yield level to whichever feature happens to
    # correlate with it. An anomaly target removes the level already.
    if a.combine and len(aois) > 1 and a.target == "yield":
        dummies = pd.get_dummies(df["aoi"], prefix="aoi").astype(float)
        df = pd.concat([df, dummies], axis=1)
        feats = feats + list(dummies.columns)
        print(f"  + {len(dummies.columns)} AOI indicator(s) for the pooled "
              f"absolute-yield fit")

    folds = [a.test_year] if a.test_year else sorted(df["harvest_year"].unique())
    if a.test_year and a.test_year not in set(df["harvest_year"]):
        raise SystemExit(
            f"--test-year {a.test_year} is not in the assembled data "
            f"({df.harvest_year.min()}-{df.harvest_year.max()})")

    run_dir.mkdir(parents=True, exist_ok=True)
    scopes = [("pooled", df)] if a.combine else \
             [(aoi, d) for aoi, d in df.groupby("aoi", sort=True)]

    preds, folds_tbl, imps = [], [], []
    for label, sub in scopes:
        print(f"\n{label}: {len(sub):,} samples, {sub.unit.nunique()} units, "
              f"{len(feats)} features")
        pr, ft, im = run_scope(sub.reset_index(drop=True), feats,
                               [y for y in folds if y in set(sub.harvest_year)],
                               label, a)
        preds.append(pr), folds_tbl.append(ft), imps.append(im)

    pred_df = pd.concat(preds, ignore_index=True)
    fold_df = pd.concat(folds_tbl, ignore_index=True)
    imp_df = pd.concat(imps, ignore_index=True)

    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    fold_df.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    fold_df[fold_df.test_year == "ALL"].to_csv(run_dir / "metrics_overall.csv",
                                               index=False)
    imp_df.to_csv(run_dir / "importances.csv", index=False)

    unit = (pred_df.assign(sq=lambda d: d.residual ** 2,
                           sq_clim=lambda d: (d.y_clim - d.y_true) ** 2)
            .groupby(["scope", "aoi", "adm2_id"])
            .agg(n=("y_true", "size"), mean_yield=("y_true", "mean"),
                 rmse=("sq", lambda s: float(np.sqrt(s.mean()))),
                 rmse_clim=("sq_clim", lambda s: float(np.sqrt(s.mean()))))
            .assign(skill=lambda d: 1 - d.rmse / d.rmse_clim)
            .reset_index().sort_values("skill", ascending=False))
    unit.to_csv(run_dir / "metrics_by_unit.csv", index=False)

    meta = {"run_id": run_id, "created": stamp, "argv": sys.argv[1:],
            "aois": aois, "combine": bool(a.combine), "source": a.source,
            "forecast_year": a.forecast_year, "crop": a.crop, "reduce": a.reduce,
            "covariates": covs, "model": a.model, "target": a.target,
            "detrend": bool(a.detrend), "seed": a.seed,
            "years": [years[0], years[-1]], "n_features": len(feats),
            "features": feats, "sources": infos,
            "overall": fold_df[fold_df.test_year == "ALL"].to_dict("records")}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    cols = ["scope", "n", "n_units", "n_folds", "rmse", "rmse_pct", "rmse_clim",
            "skill", "r2", "r_anom", "r_space_mean", "bias", "sd_ratio_anom"]
    summary = fold_df[fold_df.test_year == "ALL"][cols].copy()
    for c in ("n", "n_units", "n_folds"):
        summary[c] = summary[c].astype(int)
    print("\n" + "=" * 78)
    print(f"{a.source} fy{a.forecast_year} / {a.model} / target {a.target}")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n  skill > 0 means the features beat the per-unit climatology null.")
    print(f"\nwrote {len(list(run_dir.glob('*')))} file(s) to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
