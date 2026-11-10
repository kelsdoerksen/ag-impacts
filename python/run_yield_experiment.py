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
from models import CLASSIFIERS, REGRESSORS, SUPPORTED_MODELS, YieldModel
from era5_features import DEFAULT_SLICE_POLICY, SLICE_POLICIES, lead_tag
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


# ======================== the direction target ========================

DIRECTION_CLASSES = ["down", "flat", "up"]


class DirectionLabeller:
    """sign(y[u,t] - y[u,t-k]) with a dead band, fit on training rows only.

    THE BAND. "About the same" needs a width, and it cannot be a fixed number
    of kg/ha: a district whose yield swings 40% year to year and one that
    swings 5% would get bands that mean completely different things. So the
    width is tau * sigma(u,k), where sigma(u,k) is that unit's own standard
    deviation of its k-year differences. tau is in units of the unit's typical
    year-to-year movement, which makes the three classes comparable across
    districts and across AOIs.

    SIGMA IS k-SPECIFIC. The spread of a five-year difference is much larger
    than that of a one-year difference -- for a random walk it grows like
    sqrt(k) -- so a band fitted at k=1 and reused at k=5 would put almost
    everything in "up" or "down". Each lag gets its own sigma.

    FIT ON TRAINING ROWS ONLY. sigma is a property of the distribution the
    model is being asked to predict, so estimating it with the held-out year
    included leaks that year into its own class boundary. A unit with fewer
    than `min_years` differences falls back to the pooled training sigma.

    tau = 0 removes the flat class, giving a plain binary up/down.
    """

    def __init__(self, lag: int, tau: float = 0.5, min_years: int = 4):
        self.lag = int(lag)
        self.tau = float(tau)
        self.min_years = max(int(min_years), 2)
        self.sigma: dict = {}
        self.sigma_global = 1.0
        self.n_fallback = 0

    @property
    def ref_col(self) -> str:
        return f"yref{self.lag}"

    def diffs(self, df: pd.DataFrame) -> np.ndarray:
        return df["yield"].to_numpy(float) - df[self.ref_col].to_numpy(float)

    def fit(self, train: pd.DataFrame):
        d = self.diffs(train)
        units = train["unit"].to_numpy()
        ok = np.isfinite(d)
        self.sigma_global = float(np.std(d[ok], ddof=1)) if ok.sum() > 1 else 1.0
        if not np.isfinite(self.sigma_global) or self.sigma_global <= 0:
            self.sigma_global = 1.0
        self.n_fallback = 0
        for u in np.unique(units):
            sel = (units == u) & ok
            if sel.sum() < self.min_years:
                self.n_fallback += 1
                continue
            sd = float(np.std(d[sel], ddof=1))
            self.sigma[u] = sd if np.isfinite(sd) and sd > 0 else self.sigma_global
        return self

    def label(self, df: pd.DataFrame) -> np.ndarray:
        """Class per row; NaN where the reference year is missing."""
        d = self.diffs(df)
        sig = np.array([self.sigma.get(u, self.sigma_global)
                        for u in df["unit"].to_numpy()])
        band = self.tau * sig
        out = np.full(len(df), "up", dtype=object)
        out[d < -band] = "down"
        if self.tau > 0:
            out[np.abs(d) <= band] = "flat"
        out[~np.isfinite(d)] = None
        return out

    def classes(self) -> list[str]:
        return DIRECTION_CLASSES if self.tau > 0 else ["down", "up"]


def heidke(y_true, y_pred, classes) -> float:
    """Heidke skill score: accuracy corrected for the agreement expected by
    chance from the two marginal distributions. 0 means no better than a
    forecast that reproduces the observed class frequencies at random, 1 is
    perfect. The standard summary for a categorical forecast, and the reason
    raw accuracy is not quoted alone -- with an imbalanced flat band, always
    predicting the common class can score 0.6 at HSS 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    acc = float((y_true == y_pred).mean())
    exp = sum((y_true == c).sum() * (y_pred == c).sum() for c in classes) / n ** 2
    return float("nan") if exp >= 1 else (acc - exp) / (1 - exp)


def score_direction(df: pd.DataFrame, classes, major: str) -> dict:
    """Metrics for one set of direction predictions.

    `major` is the TRAINING majority class, so acc_major measures the honest
    null -- what a forecast that had learned nothing but the base rate would
    have scored on this same test set.
    """
    import warnings
    from sklearn.metrics import balanced_accuracy_score, f1_score
    y, p = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    # macro F1 is taken over the full class set on purpose, so that a fold in
    # which a class never occurred still contributes a 0 rather than being
    # quietly averaged out. sklearn warns that y_pred holds a label y_true
    # does not; that is the case being measured, not a mistake.
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    out = {
        "n": int(len(df)),
        "n_units": int(df["unit"].nunique()),
        "acc": float((y == p).mean()),
        "bacc": float(balanced_accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro",
                                   labels=list(classes), zero_division=0)),
        "acc_major": float((y == major).mean()),
        "hss": heidke(y, p, classes),
    }
    out["skill"] = ((out["acc"] - out["acc_major"]) / (1 - out["acc_major"])
                    if out["acc_major"] < 1 else float("nan"))
    for c in classes:
        obs, pred = (y == c), (p == c)
        out[f"n_{c}"] = int(obs.sum())
        out[f"recall_{c}"] = float((pred & obs).sum() / obs.sum()) if obs.sum() else float("nan")
        out[f"prec_{c}"] = float((pred & obs).sum() / pred.sum()) if pred.sum() else float("nan")
    if len(classes) == 2 and "p_up" in df.columns:
        from sklearn.metrics import roc_auc_score
        try:
            out["auc"] = float(roc_auc_score((y == "up").astype(int),
                                             df["p_up"].to_numpy()))
        except ValueError:
            out["auc"] = float("nan")
    return out


def fit_fold_direction(train, test, feats, lab: DirectionLabeller,
                       model_type: str, seed: int,
                       mlp_hidden=None, mlp_alpha=None):
    """Fit on `train`, predict the direction class for `test`.

    The labeller is fit here, inside the fold, for the same reason the
    regression baseline is: the band is estimated from the training years only.
    """
    lab.fit(train)
    y_tr = lab.label(train)
    y_te = lab.label(test)
    keep_tr, keep_te = pd.notna(y_tr), pd.notna(y_te)
    if keep_te.sum() < MIN_TEST_ROWS or keep_tr.sum() < 20:
        return None, None, None
    tr, te = train[keep_tr], test[keep_te]
    y_tr, y_te = y_tr[keep_tr].astype(str), y_te[keep_te].astype(str)
    if len(np.unique(y_tr)) < 2:
        return None, None, None

    model = YieldModel(model_type, random_state=seed, task="classify",
                       mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha)
    model.fit(tr[feats], y_tr, groups=tr["harvest_year"].to_numpy())
    pred = np.asarray(model.predict(te[feats])).astype(str)

    out = te[["aoi", "unit", "adm2_id", "harvest_year", "yield"]].copy()
    out["y_ref"] = te[lab.ref_col].to_numpy()
    out["delta"] = lab.diffs(te)
    out["band"] = [lab.tau * lab.sigma.get(u, lab.sigma_global)
                   for u in te["unit"]]
    out["y_true"] = y_te
    out["y_pred"] = pred
    out["correct"] = (y_te == pred).astype(int)
    proba = model.predict_proba(te[feats])
    if proba is not None and model.classes_ is not None:
        for i, c in enumerate(model.classes_):
            out[f"p_{c}"] = proba[:, i]
    vals, counts = np.unique(y_tr, return_counts=True)
    imp = model.feature_importances_
    out.attrs["alpha"] = model.alpha_
    return out, (np.asarray(imp, float) if imp is not None else None), str(vals[counts.argmax()])


def fit_fold(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
             model_type: str, target: str, seed: int, detrend: bool,
             mlp_hidden=None, mlp_alpha=None):
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

    model = YieldModel(model_type, random_state=seed, mlp_hidden=mlp_hidden,
                       mlp_alpha=mlp_alpha)
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
    out.attrs["alpha"] = model.alpha_
    return out, (np.asarray(imp, float) if imp is not None else None)


def run_scope(df: pd.DataFrame, feats: list[str], folds: list[int], label: str,
              a, lab: "DirectionLabeller | None" = None):
    """Every fold for one model scope (one AOI, or the pooled set).

    `lab` present means the direction task: the labeller carries the lag and
    the band, and the metrics are categorical rather than squared-error.
    """
    preds, per_fold, imps = [], [], []
    classes = lab.classes() if lab else None
    hidden = ([int(x) for x in a.mlp_hidden.split(",") if x.strip()]
              if getattr(a, "mlp_hidden", None) else None)
    for yr in folds:
        test = df[df["harvest_year"] == yr]
        train = df[df["harvest_year"] != yr]
        if len(test) < MIN_TEST_ROWS or train["harvest_year"].nunique() < 3:
            print(f"    {yr}: SKIPPED -- {len(test)} test row(s), "
                  f"{train['harvest_year'].nunique()} training year(s)")
            continue

        if lab is not None:
            p, imp, major = fit_fold_direction(train, test, feats, lab,
                                               a.model, a.seed, hidden,
                                               a.mlp_alpha)
            if p is None:
                print(f"    {yr}: SKIPPED -- too few labelled rows (reference "
                      f"year t-{lab.lag} missing, or one class only)")
                continue
            m = score_direction(p, classes, major)
            if p.attrs.get("alpha") is not None:
                m["alpha"] = float(p.attrs["alpha"])
            per_fold.append({"scope": label, "test_year": yr,
                             "n_train": int(len(train)), "major": major, **m})
            preds.append(p.assign(scope=label, test_year=yr))
            print(f"    {yr}: n={m['n']:>5}  acc={m['acc']:.3f}  "
                  f"bacc={m['bacc']:.3f}  hss={m['hss']:+.3f}  "
                  f"(null {m['acc_major']:.3f})")
        else:
            p, imp = fit_fold(train, test, feats, a.model, a.target, a.seed,
                              a.detrend, hidden, a.mlp_alpha)
            m = score(p)
            # The selected penalty is a diagnostic in its own right: pinned at
            # the top of the grid means that fold's CV found no signal and the
            # model was shrunk to the mean on purpose.
            if p.attrs.get("alpha") is not None:
                m["alpha"] = float(p.attrs["alpha"])
            # Within this one year, across units: did the model place the
            # anomaly correctly in space.
            m["r_space"] = _r(p["y_true"] - p["y_clim"], p["y_pred"] - p["y_clim"])
            per_fold.append({"scope": label, "test_year": yr,
                             "n_train": int(len(train)), **m})
            preds.append(p.assign(scope=label, test_year=yr))
            print(f"    {yr}: n={m['n']:>5}  rmse={m['rmse']:>8,.0f}  "
                  f"skill={m['skill']:+.3f}  r_space={m['r_space']:+.3f}")

        if imp is not None and len(imp) == len(feats):
            imps.append(imp)

    if not preds:
        raise SystemExit(f"[{label}] no fold produced predictions")

    allp = pd.concat(preds, ignore_index=True)
    if lab is not None:
        # Pooled over folds, the null is the overall observed base rate of the
        # class that was most often the training majority.
        major = pd.Series([f["major"] for f in per_fold]).mode()[0]
        overall = {"scope": label, "test_year": "ALL", "n_folds": len(per_fold),
                   "major": major, **score_direction(allp, classes, major)}
        if "alpha" in per_fold[0]:
            overall["alpha"] = float(np.median([f["alpha"] for f in per_fold]))
    else:
        overall = {"scope": label, "test_year": "ALL",
                   "n_folds": len(per_fold), **score(allp)}
        if "alpha" in per_fold[0]:
            overall["alpha"] = float(np.median([f["alpha"] for f in per_fold]))
        # Mean of the per-fold spatial correlations: the pooled r_anom above
        # mixes the spatial and interannual parts, this one is purely spatial.
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
    g.add_argument("--slice-policy", choices=list(SLICE_POLICIES),
                   default=DEFAULT_SLICE_POLICY,
                   help="which stage-3 DCPP store to read: fixed-init (fy{F}_) or "
                        "newest-run (fy{F}n_, the season assembled from the newest "
                        "run no newer than s = H - F, so more of it sits at lead 1). "
                        "Build the store with era5_features.py --slice-policy first.")
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
    g.add_argument("--model", choices=SUPPORTED_MODELS, default="rf",
                   help=f"regression: {', '.join(REGRESSORS)}; "
                        f"direction: {', '.join(CLASSIFIERS)}")
    g.add_argument("--target", choices=["yield", "anomaly", "direction"],
                   default="yield",
                   help="'yield' predicts kg/ha. 'anomaly' subtracts each "
                        "unit's training-only baseline first, so the model "
                        "spends its capacity on interannual variation instead "
                        "of the map. 'direction' classifies this year against "
                        "a prior year as up / flat / down")
    g.add_argument("--target-lags", default="1",
                   help="direction only: how many years back the reference "
                        "year is. Comma-separated fits one model per lag, e.g. "
                        "1,2,3,5. Each must be >= --forecast-year")
    g.add_argument("--flat-band", type=float, default=0.5,
                   help="direction only: half-width of the 'about the same' "
                        "class, in units of each unit's own standard deviation "
                        "of its k-year yield differences (training years only). "
                        "0 removes the flat class, giving binary up/down")
    g.add_argument("--detrend", action="store_true",
                   help="per-unit baseline is a line in year, not a flat mean. "
                        "Use it where a technology trend dominates (US maize)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--mlp-hidden", default=None,
                   help="MLP layer sizes, e.g. 64,32 (default) or 128,64,32")
    g.add_argument("--mlp-alpha", type=float, default=None,
                   help="pin the MLP's weight decay instead of selecting it by "
                        "grouped CV. The search costs n_splits extra network "
                        "fits per fold, so pin it once you know the decade")

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
    if kind == "era5" and a.slice_policy != DEFAULT_SLICE_POLICY:
        raise SystemExit("--slice-policy applies to a hindcast only; ERA5 is "
                         "observed and has no initialisation")

    allowed = CLASSIFIERS if a.target == "direction" else REGRESSORS
    if a.model not in allowed:
        raise SystemExit(
            f"--model {a.model} cannot do --target {a.target}; "
            f"choose from {', '.join(allowed)}")

    tlags = []
    if a.target == "direction":
        tlags = parse_years(a.target_lags)
        if not tlags or min(tlags) < 1:
            raise SystemExit(f"--target-lags must be >= 1, got {a.target_lags!r}")
        # The reference year has to be observable when the forecast is issued.
        # A lead-F forecast is made before harvest year H-F+1 exists, so
        # comparing H against H-1 at F=2 asks the model to beat a number
        # nobody has yet.
        bad = [k for k in tlags if k < a.forecast_year]
        if bad:
            raise SystemExit(
                f"--target-lags {bad} are shorter than the lead time "
                f"({a.forecast_year}): at forecast year {a.forecast_year} the "
                f"most recent observable harvest is t-{a.forecast_year}, so the "
                f"reference year must be at least that far back.")
        if a.flat_band < 0:
            raise SystemExit("--flat-band must be >= 0")

    aois = [x.strip() for x in a.aois.split(",") if x.strip()]
    covs = [x.strip() for x in a.add.split(",") if x.strip()]
    bad = [c for c in covs if c not in COVARIATES]
    if bad:
        raise SystemExit(f"unknown --add {bad}; choose from {COVARIATES}")
    years = parse_years(a.years)

    # A lead-F forecast is issued on 1 Nov (H-F), when the latest complete
    # season is H-F, so --phenology-lag below F hands the model NDVI that did
    # not exist at the issue date. That is allowed on purpose: holding the
    # phenology lag fixed across leads (e.g. lag 1 everywhere, "last season's
    # NDVI is always in hand") keeps the covariates identical between runs, so
    # the only thing that differs is the climate source. It is noted rather
    # than refused, and recorded in run_meta.json.
    phen = [c for c in covs if c != "yield_history"]
    phen_note = ""
    if phen and a.phenology_lag < max(1, a.forecast_year):
        phen_note = (f"{', '.join(phen)} at lag {a.phenology_lag} postdates the "
                     f"FY{a.forecast_year} issue date (latest observable season "
                     f"H-{a.forecast_year}); covariates are held fixed across "
                     f"leads, not issue-date-realistic")

    roots = Roots.resolve(a.share, a.era5_root, a.dcpp_root, a.targets_root)
    out_root = Path(a.out_dir) if a.out_dir else roots.share / "experiments"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = f"test{a.test_year}" if a.test_year else "loo"
    scope_tag = "pooled" if a.combine else "peraoi"
    tgt = a.target
    if a.target == "direction":
        tgt = (f"dir-lag{'-'.join(str(k) for k in tlags)}"
               f"-b{a.flat_band:g}".replace(".", "p"))
    run_id = "_".join(filter(None, [
        a.source.replace(":", "-"),
        lead_tag(a.forecast_year, a.slice_policy).rstrip("_") or "fy0",
        a.model, tgt,
        scope_tag, split, "-".join(aois) if len(aois) <= 4 else f"{len(aois)}aois",
        a.tag, stamp]))
    run_dir = out_root / run_id

    print("=" * 78)
    print("yield experiment")
    print("=" * 78)
    print(f"  AOIs          : {', '.join(aois)}"
          f"{'  (pooled into one model)' if a.combine else '  (one model each)'}")
    print(f"  source        : {a.source}   forecast year {a.forecast_year}"
          + (f"   slice policy {a.slice_policy}" if kind == "dcpp" else ""))
    print(f"  features      : {a.reduce}"
          f"{'  + ' + ', '.join(covs) if covs else ''}")
    if phen:
        print(f"  phenology lag : {a.phenology_lag} season(s)")
    if phen_note:
        print(f"  note          : {phen_note}")
    print(f"  harvest years : {years[0]}-{years[-1]}")
    print(f"  split         : {'leave-one-year-out' if a.loo else f'hold out {a.test_year}'}")
    if a.target == "direction":
        band = ("binary up/down" if a.flat_band == 0
                else f"flat band +/-{a.flat_band:g} sd of each unit's "
                     f"k-year differences")
        print(f"  model         : {a.model}   target direction vs t-"
              f"{', t-'.join(str(k) for k in tlags)}")
        print(f"  classes       : {band}")
    else:
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
                                 a.yield_lags, a.phenology_suffix,
                                 ref_lags=tlags, slice_policy=a.slice_policy)
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

    # One model per (scope, target lag). Separate models rather than a
    # multi-output one: the five "is it above t-k" questions have very
    # different base rates and very different amounts of signal, and a shared
    # output head would hide that behind one averaged loss. Running them in one
    # invocation gives the same comparison without the coupling.
    jobs = [(lbl, sub, None) for lbl, sub in scopes] if a.target != "direction" \
        else [(f"{lbl} t-{k}", sub, DirectionLabeller(k, a.flat_band))
              for lbl, sub in scopes for k in tlags]

    preds, folds_tbl, imps = [], [], []
    for label, sub, lab in jobs:
        sub = sub.reset_index(drop=True)
        if lab is not None:
            n_ref = int(sub[lab.ref_col].notna().sum())
            print(f"\n{label}: {len(sub):,} samples ({n_ref:,} with a t-{lab.lag} "
                  f"reference), {sub.unit.nunique()} units, {len(feats)} features")
        else:
            print(f"\n{label}: {len(sub):,} samples, {sub.unit.nunique()} units, "
                  f"{len(feats)} features")
        pr, ft, im = run_scope(sub, feats,
                               [y for y in folds if y in set(sub.harvest_year)],
                               label, a, lab)
        preds.append(pr), folds_tbl.append(ft), imps.append(im)

    pred_df = pd.concat(preds, ignore_index=True)
    fold_df = pd.concat(folds_tbl, ignore_index=True)
    imp_df = pd.concat(imps, ignore_index=True)

    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    fold_df.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    fold_df[fold_df.test_year == "ALL"].to_csv(run_dir / "metrics_overall.csv",
                                               index=False)
    imp_df.to_csv(run_dir / "importances.csv", index=False)

    if a.target == "direction":
        unit = (pred_df.groupby(["scope", "aoi", "adm2_id"])
                .agg(n=("correct", "size"), acc=("correct", "mean"),
                     mean_delta=("delta", "mean"), mean_band=("band", "mean"))
                .reset_index().sort_values("acc", ascending=False))
    else:
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
            "forecast_year": a.forecast_year, "slice_policy": a.slice_policy,
            "crop": a.crop, "reduce": a.reduce,
            "covariates": covs, "phenology_lag": a.phenology_lag if phen else None,
            "yield_min_lag": max(1, a.forecast_year) if "yield_history" in covs else None,
            "model": a.model, "target": a.target,
            "detrend": bool(a.detrend), "seed": a.seed,
            "target_lags": tlags, "flat_band": a.flat_band,
            "years": [years[0], years[-1]], "n_features": len(feats),
            "features": feats, "sources": infos,
            "overall": fold_df[fold_df.test_year == "ALL"].to_dict("records")}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if a.target == "direction":
        cols = ["scope", "n", "n_units", "n_folds", "acc", "acc_major", "skill",
                "bacc", "hss", "f1_macro", "alpha"] + \
               [c for c in fold_df.columns if c.startswith(("n_down", "n_flat",
                                                            "n_up", "recall_"))] + \
               (["auc"] if "auc" in fold_df.columns else [])
    else:
        cols = ["scope", "n", "n_units", "n_folds", "rmse", "rmse_pct", "rmse_clim",
                "skill", "r2", "r_anom", "r_space_mean", "bias", "sd_ratio_anom",
                "alpha"]
    cols = [c for c in cols if c in fold_df.columns]
    summary = fold_df[fold_df.test_year == "ALL"][cols].copy()
    for c in ("n", "n_units", "n_folds"):
        summary[c] = summary[c].astype(int)
    print("\n" + "=" * 78)
    print(f"{a.source} fy{a.forecast_year} / {a.model} / target {a.target}")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if a.target == "direction":
        print("\n  acc_major is the base-rate null: always predict the training "
              "majority class.")
        print("  skill = (acc - acc_major)/(1 - acc_major); hss corrects "
              "accuracy for chance agreement.")
        print("  bacc averages recall over classes, so a model that ignores the "
              "rare class cannot hide in it.")
    else:
        print("\n  skill > 0 means the features beat the per-unit climatology null.")
    print(f"\nwrote {len(list(run_dir.glob('*')))} file(s) to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
