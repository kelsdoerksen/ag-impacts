#!/usr/bin/env python3
"""How much does climate tell us about yield change?

This is not a forecasting script. run_loo_experiment.py answers "can a model
predict a year it has never seen".
A genuine climate-yield relationship can sit inside a model that
forecasts worse than a constant, because estimation noise swamps it. Measuring
the relationship and its precision directly is a different job from scoring a
forecast, and it is the one this script does.

WHAT IT REPORTS

1. Variance decomposition of yield. District yield anomalies split into a
   spatially coherent part (a common year effect -- what weather can move) and
   district-specific idiosyncrasy (management, measurement error, reporting).
   The coherent share is a hard CEILING on what any gridded climate product can
   explain, and it is worth knowing before attributing a low R2 to the climate
   data.

2. Panel regression with district fixed effects, the standard design in the
   crop-climate literature (Schlenker & Roberts 2009; Lobell et al. 2011).
   Coefficients are the yield response to a unit of each climate variable,
   holding district constant. Standard errors are clustered BY YEAR, because
   districts within a year are strongly correlated (r ~ 0.28 in Zambia) and
   treating 2,300 rows as independent would understate the errors several-fold.

3. The same regression run on national annual means, where the signal actually
   lives. Few points, but no illusion of a large sample.

4. Leave-one-year-out on the LINEAR model, for an out-of-sample number at a
   capacity the sample size supports -- comparable to the tree runs but not
   handicapped by them.

Usage
-----
  python python/climate_yield_information.py \\
      --data_dir $AGIMPACTS_ROOT/ZM_processed_data \\
      --pixel_features_dir /home/crl/kdoerksen/data/era5_zm \\
      --aoi ZM --crop corn --climate_model ERA5 --forecast_year 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_pixel_season_features  # noqa: E402


# ------------------------------------------------------------------ regression


def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS with cluster-robust (CR0) standard errors.

    Implemented here rather than pulled from statsmodels/linearmodels: it is
    twenty lines, and the lab machine should not need another dependency to run
    a diagnostic.

    Clustering matters more than usual here. Districts within a year share the
    same weather, so their residuals are correlated (r ~ 0.28); the classical
    OLS error, which assumes 2,300 independent rows, would be optimistic by
    roughly sqrt(1 + (n_per_year - 1) * r) -- a factor of ~4.5 for Zambia. The
    honest sample size is the number of YEAR clusters.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = np.zeros((k, k))
    for c in np.unique(clusters):
        m = clusters == c
        Xc, uc = X[m], resid[m]
        s = Xc.T @ uc
        meat += np.outer(s, s)

    g = len(np.unique(clusters))
    # Small-cluster correction: with ~33 year clusters this is not negligible.
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * scale
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, r2, resid


def t_to_p(t: np.ndarray, dof: int) -> np.ndarray:
    """Two-sided p-value from a t statistic, via the normal approximation when
    scipy is absent. dof here is the number of clusters minus one, not n."""
    try:
        from scipy import stats
        return 2 * (1 - stats.t.cdf(np.abs(t), dof))
    except ImportError:
        from math import erfc, sqrt
        return np.array([erfc(abs(v) / sqrt(2)) for v in np.atleast_1d(t)])


def demean(df: pd.DataFrame, cols: list, by: str) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df[cols] - df.groupby(by)[cols].transform("mean")
    return out


# ------------------------------------------------------------------- reporting


def rule(title: str, ch: str = "=", w: int = 78):
    print("\n" + ch * w)
    print(title)
    print(ch * w)


def variance_decomposition(df: pd.DataFrame) -> float:
    """Split yield variance and return the coherent (year-effect) share.

    Returned value is the ceiling on R2-in-anomaly-space for any predictor built
    from a spatially smooth field, which every gridded climate product is.
    """
    rule("1. WHERE IS THE VARIANCE?")
    y = df["yield"].to_numpy(dtype=float)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    dmean = df.groupby("adm2_id")["yield"].transform("mean").to_numpy()
    ss_between = float(((dmean - y.mean()) ** 2).sum())
    anom = y - dmean
    ss_within = float((anom ** 2).sum())

    a = df.assign(_a=anom)
    ymean = a.groupby("target_year")["_a"].transform("mean").to_numpy()
    ss_year = float((ymean ** 2).sum())
    ss_idio = ss_within - ss_year

    print(f"  total yield variance                      {ss_tot:14,.0f}")
    print(f"    between districts (level, not change)   {ss_between:14,.0f}"
          f"   {100*ss_between/ss_tot:5.1f}%")
    print(f"    within district  (the change we model)  {ss_within:14,.0f}"
          f"   {100*ss_within/ss_tot:5.1f}%")
    print(f"      of which common to all districts      {ss_year:14,.0f}"
          f"   {100*ss_year/ss_within:5.1f}% of within")
    print(f"      of which district-specific            {ss_idio:14,.0f}"
          f"   {100*ss_idio/ss_within:5.1f}% of within")

    share = ss_year / ss_within
    print(f"\n  The spatially coherent share is {100*share:.1f}% of the within-district")
    print( "  variance. Weather is smooth in space, so a climate predictor can only")
    print( "  reach that part: district-specific idiosyncrasy (management, measurement")
    print( "  error, reporting) is out of range for ANY gridded product.")
    print(f"\n  => CEILING on R2anom from a perfect climate predictor: ~{share:.3f}")
    print( "     (higher only insofar as climate itself varies district-to-district")
    print( "      within a year, which the panel regression below tests directly)")
    return share


def coef_table(names, beta, se, dof, sd_x, sd_y):
    t = np.where(se > 0, beta / np.where(se > 0, se, 1), np.nan)
    p = t_to_p(t, dof)
    print(f"\n  {'variable':<12}{'coef':>12}{'se(year)':>11}{'t':>7}{'p':>8}"
          f"{'beta*':>9}")
    print("  " + "-" * 59)
    for i, nm in enumerate(names):
        # Standardised coefficient: yield sd per 1 sd of the feature, which is
        # the only way to compare GDD (degC-day) against precipitation (mm).
        bstar = beta[i] * sd_x[i] / sd_y if sd_y > 0 else np.nan
        star = "***" if p[i] < 0.01 else "**" if p[i] < 0.05 else "*" if p[i] < 0.10 else ""
        print(f"  {nm:<12}{beta[i]:>12.3f}{se[i]:>11.3f}{t[i]:>7.2f}"
              f"{p[i]:>8.3f}{bstar:>9.3f} {star}")
    print("  " + "-" * 59)
    print("  beta* = yield sd per 1 sd of the feature. *** p<.01  ** p<.05  * p<.10")
    print("  Standard errors clustered by year; p-values use "
          f"{dof} degrees of freedom (year clusters - 1), not the row count.")


def effective_sample_size(df: pd.DataFrame, feats: list):
    """How many independent observations does this panel really contain?

    Not the row count. Districts within a year share their weather, so their
    yield anomalies are correlated (ICC ~ 0.25 for Zambian maize) and the rows
    are partly repeats of each other. Kish's design effect discounts for that:

        n_eff = n / (1 + (m - 1) * rho)

    with m the districts per year and rho the intra-year correlation of the
    residual being predicted. For Zambia that is ~2,368 / 18 ~ 130, not 2,368.

    It also differs per FEATURE, which is the part that actually guides model
    choice. A predictor that is essentially one national number per year carries
    only as many independent values as there are years, however many rows it is
    copied across; one that varies district-to-district within a year carries
    close to the full row count. The between-year variance share below places
    each feature on that scale, and the implied n_eff is roughly how many points
    its coefficient is really estimated from.
    """
    rule("2. HOW MANY INDEPENDENT OBSERVATIONS?")
    n = len(df)
    ny = df["target_year"].nunique()
    m = n / ny

    anom = df["yield"] - df.groupby("adm2_id")["yield"].transform("mean")
    a = df.assign(_a=anom)
    rho = float((a.groupby("target_year")["_a"].transform("mean") ** 2).sum()
                / (anom ** 2).sum())
    deff = 1 + (m - 1) * rho
    n_eff = n / deff

    print(f"  rows                       {n:>10,}")
    print(f"  years                      {ny:>10}")
    print(f"  districts per year         {m:>10.1f}")
    print(f"  intra-year ICC (yield)     {rho:>10.3f}")
    print(f"  design effect 1+(m-1)*rho  {deff:>10.1f}")
    print(f"  effective sample size      {n_eff:>10,.0f}")

    print("\n  Per feature -- a predictor that is one national number per year is")
    print("  worth only as many points as there are years, however many rows it")
    print("  is copied across:")
    print(f"\n    {'feature':<12}{'between-year':>14}{'implied n_eff':>15}")
    print("    " + "-" * 41)
    rows = []
    for f in feats:
        x = df[f].astype(float)
        # Between-year share of the WITHIN-DISTRICT variation: the district mean
        # is absorbed by the fixed effects, so it is not what identifies the
        # coefficient.
        xw = x - df.groupby("adm2_id")[f].transform("mean")
        ss_t = float((xw ** 2).sum())
        if ss_t <= 0:
            continue
        b = float((df.assign(_x=xw).groupby("target_year")["_x"]
                   .transform("mean") ** 2).sum()) / ss_t
        rows.append((f, b, b * ny + (1 - b) * n_eff))
    for f, b, e in sorted(rows, key=lambda t: -t[1]):
        print(f"    {f:<12}{100*b:>13.1f}%{e:>15,.0f}")
    print("    " + "-" * 41)
    print("  A feature high in this column is close to a national index: its")
    print("  coefficient rests on a few dozen points no matter how many rows the")
    print("  table has, and a flexible model will overfit it. One low in the")
    print("  column carries genuine district-level information and supports more.")
    return n_eff


def panel_regression(df: pd.DataFrame, feats: list, share: float):
    rule("3. PANEL REGRESSION, DISTRICT FIXED EFFECTS")
    print("  yield ~ climate + district FE, SEs clustered by year.")
    print("  Within transformation: every variable is a deviation from its own")
    print("  district's mean, so this is a pure within-district (year-to-year)")
    print("  relationship -- no cross-sectional 'hot districts yield less'.")

    n_clu = df["target_year"].nunique()
    if len(feats) + 1 >= n_clu:
        # The cluster-robust "meat" matrix is a sum of n_clu outer products, so
        # its rank is at most n_clu. With more parameters than year clusters it
        # is singular: the standard errors below are not identified and the
        # coefficients are fitting year effects one-for-one. Say so rather than
        # printing a table of numbers that look like inference.
        print(f"\n  ! {len(feats)} features against {n_clu} year clusters.")
        print( "    Cluster-robust standard errors need MORE clusters than")
        print( "    parameters -- the ones below are not identified, and the")
        print( "    coefficients are absorbing year effects one-for-one.")
        print( "    Re-run with --features to subset, e.g. --features 'pr_*'")
        print( "    to ask which part of the season the rainfall signal sits in.")

    d = demean(df, feats + ["yield"], "adm2_id")
    X = np.column_stack([np.ones(len(d))] + [d[f].to_numpy(dtype=float) for f in feats])
    y = d["yield"].to_numpy(dtype=float)
    yrs = df["target_year"].to_numpy()

    beta, se, r2, resid = ols_cluster(X, y, yrs)
    n_clu = len(np.unique(yrs))
    sd_x = np.r_[1.0, [df[f].std() for f in feats]]
    coef_table(["(const)"] + feats, beta, se, n_clu - 1, sd_x, float(d["yield"].std()))

    print(f"\n  within-R2 (share of year-to-year yield variance explained): {r2:.3f}")
    print(f"  ceiling from the decomposition above:                       {share:.3f}")
    if share > 0:
        print(f"  => climate explains {100*r2/share:.0f}% of the reachable signal")

    # An F-test on all climate terms jointly: does the block carry information
    # at all, regardless of which individual coefficient is significant?
    X0 = np.ones((len(d), 1))
    _, _, _, r0 = ols_cluster(X0, y, yrs)
    ss_r, ss_u = float(r0 @ r0), float(resid @ resid)
    q, dof_d = len(feats), n_clu - len(feats) - 1
    if dof_d > 0 and ss_u > 0:
        F = ((ss_r - ss_u) / q) / (ss_u / dof_d)
        print(f"\n  joint F on all {q} climate terms: F({q},{dof_d}) = {F:.2f}")
        print( "  (computed on year-clustered dof, so it is a conservative test)")
    return r2


def national_regression(df: pd.DataFrame, feats: list):
    rule("4. NATIONAL ANNUAL MEANS")
    print("  The coherent signal is national, so this is the same question asked")
    print("  at the scale it lives, with no illusion of a large sample.")

    nat = df.groupby("target_year").agg(
        {**{f: "mean" for f in feats}, "yield": "mean"}).reset_index()
    nat["yield_anom"] = nat["yield"] - nat["yield"].mean()
    for f in feats:
        nat[f] = nat[f] - nat[f].mean()

    n = len(nat)
    print(f"\n  n = {n} years, {len(feats)} predictors "
          f"({n - len(feats) - 1} residual dof)")
    if n <= len(feats) + 2:
        print("  Too few years to fit; skipping.")
        return

    X = np.column_stack([np.ones(n)] + [nat[f].to_numpy(dtype=float) for f in feats])
    y = nat["yield_anom"].to_numpy(dtype=float)
    beta, se, r2, _ = ols_cluster(X, y, nat["target_year"].to_numpy())
    sd_x = np.r_[1.0, [nat[f].std() for f in feats]]
    coef_table(["(const)"] + feats, beta, se, n - 1, sd_x, float(nat["yield_anom"].std()))

    adj = 1 - (1 - r2) * (n - 1) / max(n - len(feats) - 1, 1)
    print(f"\n  R2 = {r2:.3f}   adjusted R2 = {adj:.3f}")
    print( "  Adjusted R2 is the one to read: with this many predictors per year,")
    print( "  raw R2 is inflated by construction.")

    # Univariate screen: which single variable carries the most, on its own?
    print("\n  Single-variable fits (same n, 1 predictor each):")
    rows = []
    for f in feats:
        Xi = np.column_stack([np.ones(n), nat[f].to_numpy(dtype=float)])
        b, s, r, _ = ols_cluster(Xi, y, nat["target_year"].to_numpy())
        rr = float(np.corrcoef(nat[f], y)[0, 1])
        rows.append((f, r, rr, b[1] / s[1] if s[1] > 0 else np.nan))
    rows.sort(key=lambda t: -t[1])
    print(f"    {'variable':<12}{'R2':>8}{'r':>8}{'t':>8}")
    for f, r, rr, t in rows:
        print(f"    {f:<12}{r:>8.3f}{rr:>+8.3f}{t:>8.2f}")


def loo_linear(df: pd.DataFrame, feats: list, share: float):
    rule("5. LEAVE-ONE-YEAR-OUT, LINEAR MODEL")
    print("  Out-of-sample, but at a capacity the sample size supports. Compare")
    print("  this to the tree runs: if it is better, the trees were overfitting,")
    print("  not the features failing.")

    yrs = np.sort(df["target_year"].unique())
    y_true, y_pred, a_true, a_pred = [], [], [], []

    for yr in yrs:
        tr, te = df[df.target_year != yr], df[df.target_year == yr]
        if len(te) == 0 or tr["target_year"].nunique() < 5:
            continue
        # District means from TRAINING years only -- the held-out year must not
        # contribute to the normal it is measured against.
        mu_d = tr.groupby("adm2_id")["yield"].mean()
        mu_g = tr["yield"].mean()
        fx = {f: tr.groupby("adm2_id")[f].mean() for f in feats}
        fg = {f: tr[f].mean() for f in feats}

        def prep(d):
            X = np.column_stack(
                [np.ones(len(d))] +
                [d[f].to_numpy(float) - d["adm2_id"].map(fx[f]).fillna(fg[f]).to_numpy(float)
                 for f in feats])
            base = d["adm2_id"].map(mu_d).fillna(mu_g).to_numpy(float)
            return X, base

        Xtr, btr = prep(tr)
        Xte, bte = prep(te)
        beta, *_ = ols_cluster(Xtr, tr["yield"].to_numpy(float) - btr,
                               tr["target_year"].to_numpy())
        pred_anom = Xte @ beta
        y_pred.append(pred_anom + bte)
        y_true.append(te["yield"].to_numpy(float))
        a_pred.append(pred_anom)
        a_true.append(te["yield"].to_numpy(float) - bte)

    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    at, ap = np.concatenate(a_true), np.concatenate(a_pred)
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    base_rmse = float(np.sqrt((at ** 2).mean()))
    r2a = 1 - float(((at - ap) ** 2).sum()) / float((at ** 2).sum())
    corr = float(np.corrcoef(at, ap)[0, 1])

    print(f"\n  pooled RMSE              {rmse:10,.0f} kg/ha")
    print(f"  county-climatology RMSE  {base_rmse:10,.0f} kg/ha")
    print(f"  skill vs climatology     {100*(1-rmse/base_rmse):+10.1f} %")
    print(f"  R2anom                   {r2a:10.3f}   (ceiling ~{share:.3f})")
    print(f"  corr(anom)               {corr:+10.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pixel_features_dir", default=None)
    p.add_argument("--aoi", default="ZM")
    p.add_argument("--crop", default="corn")
    p.add_argument("--climate_model", default="ERA5")
    p.add_argument("--forecast_year", type=int, default=0)
    p.add_argument("--study_years", type=int, nargs=2, metavar=("START", "END"))
    p.add_argument("--kind", choices=["season", "blocks"], default="season",
                   help="Which stage-3 store to read. Default: season.")
    p.add_argument("--features", default=None, metavar="LIST",
                   help="Comma-separated subset of feature columns, or a prefix "
                        "with a trailing '*' (e.g. 'pr_*,edd_*'). With --kind "
                        "blocks the full 42-column set exceeds the 33 year "
                        "clusters, which makes cluster-robust inference "
                        "unidentified -- subset to one or two families instead.")
    a = p.parse_args()

    df, feats = load_pixel_season_features(
        crop=a.crop, data_dir=a.data_dir, aoi=a.aoi,
        climate_model=a.climate_model, forecast_year=a.forecast_year,
        target="yield", features_dir=a.pixel_features_dir, kind=a.kind)

    if a.features:
        want, keep = [w.strip() for w in a.features.split(",") if w.strip()], []
        for f in feats:
            if any(f == w or (w.endswith("*") and f.startswith(w[:-1])) for w in want):
                keep.append(f)
        if not keep:
            raise SystemExit(f"--features {a.features!r} matched none of: {feats}")
        feats = keep

    if a.study_years:
        lo, hi = a.study_years
        df = df[df.target_year.between(lo, hi)]

    rule(f"CLIMATE INFORMATION ABOUT YIELD -- {a.aoi} {a.crop}, "
         f"{a.climate_model} FY{a.forecast_year}")
    print(f"  {len(df):,} district-years, {df.adm2_id.nunique()} districts, "
          f"{df.target_year.nunique()} years "
          f"({df.target_year.min()}-{df.target_year.max()})")
    print(f"  features: {', '.join(feats)}")

    share = variance_decomposition(df)
    effective_sample_size(df, feats)
    panel_regression(df, feats, share)
    national_regression(df, feats)
    loo_linear(df, feats, share)

    rule("HOW TO READ THIS", "=")
    print("""  The panel regression's within-R2 and the joint F test answer the
  question directly: is there a systematic year-to-year climate-yield
  relationship, and how precisely is it measured? A significant coefficient
  block with a small R2 is a real but modest signal -- which is the normal
  finding for seasonal aggregates, and is NOT the same result as no signal.

  Compare within-R2 to the ceiling. If climate explains most of the reachable
  signal, the remaining error is target noise and better features will not
  help. If it explains little of it, timing information is the usual missing
  piece: season totals cannot separate a dry spell at flowering from the same
  deficit at maturity.

  Section 5 vs the tree runs separates a model problem from a feature problem.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
