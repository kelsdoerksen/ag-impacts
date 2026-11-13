#!/usr/bin/env python
"""
Turn one or more run_yield_experiment.py output directories into one HTML page.

Answers three questions that the CSVs answer only with effort:

    1. Which configuration predicts yield best, and by how much.
    2. Where does it work -- which districts, mapped.
    3. Where does one configuration beat another, mapped as a difference.

(3) is the point of running more than one configuration. An ERA5 run and a
DCPP run differ only in whether the climate was observed or forecast, so a
map of (dcpp skill - era5 skill) per district is a map of what the forecast
costs you, spatially. Two runs that differ only in --add show what the
covariates buy, and where.

EVERY NUMBER IS AGAINST THE CLIMATOLOGY NULL

`skill` = 1 - rmse/rmse_clim, where rmse_clim is each district's own
training-only mean (or fitted trend with --detrend). Zero means "no better
than knowing which district this is", negative means worse. Absolute R2 is
reported too but it is dominated by between-district differences in yield
level and is near 1 for almost any model, which is why it is not the headline
and not what the maps are coloured by.

For a --target direction run the per-district quantity is accuracy instead,
and the honest reference is the base rate rather than zero.

Reads, per run directory:
    run_meta.json        the configuration, for labelling
    metrics_by_fold.csv  one row per held-out year, plus an ALL row
    metrics_by_unit.csv  per district -- what the maps are built from
    importances.csv      mean over folds

The map machinery is imported from report_dcpp_era5.py rather than copied, so
both reports project, trim and shade identically and a fix lands in one place.
Projection is an equal-area Albers fitted to each AOI's own extent.

    python python/report_yield_experiment.py \\
        --dir /mnt/crl_storage/personal_folders/kdoerksen/experiments \\
        --polygons /mnt/crl_storage/personal_folders/kdoerksen/cy-bench/polygons \\
        --out ~/yield_experiments.html
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from report_dcpp_era5 import (CSS, line_chart, load_polygons, map_legend,
                              path_defs, svg_map)

# Regression metrics worth a column, in reading order. skill first: it is the
# only one that is zero when the model has learned nothing useful.
REG_COLS = ["n", "n_units", "n_folds", "skill", "rmse", "rmse_pct", "rmse_clim",
            "r2", "r_anom", "r_space_mean", "sd_ratio_anom", "bias", "alpha"]
# delta / zscore runs add their native-unit metrics (see RefTarget in
# run_yield_experiment.py); the kg/ha columns stay comparable across targets.
REF_COLS = REG_COLS + ["t_skill", "t_r", "t_skill_zero", "t_skill_clim_vs_zero"]
DIR_COLS = ["n", "n_units", "n_folds", "acc", "acc_major", "skill", "bacc",
            "hss", "f1_macro", "alpha"]


def label_for(meta: dict) -> str:
    """A short run label: the things that actually differ between runs."""
    bits = [str(meta.get("source", "?"))]
    if meta.get("forecast_year"):
        bits.append(f"fy{meta['forecast_year']}")
    bits.append(str(meta.get("model", "?")))
    tgt = meta.get("target", "?")
    if tgt == "direction":
        lags = meta.get("target_lags") or []
        tgt = f"dir t-{','.join(str(k) for k in lags)}"
        if meta.get("flat_band") == 0:
            tgt += " binary"
    elif tgt in ("delta", "zscore"):
        rt = meta.get("ref_target") or {}
        k = rt.get("end", "?")
        tgt = (f"Δ vs t-{k}{' %' if rt.get('pct') else ''}" if tgt == "delta"
               else f"z{rt.get('window', '?')} to t-{k}")
    bits.append(tgt)
    if meta.get("detrend"):
        bits.append("detrended")
    covs = meta.get("covariates") or []
    bits.append("+" + ",".join(covs) if covs else "climate only")
    return " · ".join(bits)


def load_runs(root: Path, patterns=None) -> list[dict]:
    """Every run directory under `root`, newest first, optionally filtered."""
    runs = []
    for meta_path in sorted(root.glob("*/run_meta.json")):
        d = meta_path.parent
        if patterns and not any(fnmatch.fnmatch(d.name, p) for p in patterns):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"  skipping {d.name}: unreadable run_meta.json ({exc})")
            continue
        fold = _read(d / "metrics_by_fold.csv")
        unit = _read(d / "metrics_by_unit.csv")
        imp = _read(d / "importances.csv")
        pred = _read(d / "predictions.csv")
        if fold.empty:
            print(f"  skipping {d.name}: no metrics_by_fold.csv")
            continue
        runs.append({"dir": d, "name": d.name, "meta": meta,
                     "label": label_for(meta), "fold": fold, "unit": unit,
                     "imp": imp, "pred": pred,
                     "target": meta.get("target", "yield")})
    runs.sort(key=lambda r: r["meta"].get("created", ""))
    return runs


def _read(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def overall(runs) -> pd.DataFrame:
    """The ALL row of every run x scope, tagged with the run label."""
    rows = []
    for r in runs:
        f = r["fold"]
        allrow = f[f.test_year.astype(str) == "ALL"]
        for _, x in allrow.iterrows():
            rows.append({"run": r["label"], "run_dir": r["name"],
                         "target": r["target"], **x.to_dict()})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, cols, ref: str | None = None) -> str:
    """Scope-major table, one row per run, with a delta against `ref`."""
    cols = [c for c in cols if c in df.columns]
    out = []
    for scope, g in df.groupby("scope", sort=True):
        g = g.copy()
        base = None
        if ref is not None:
            m = g[g.run == ref]
            if not m.empty and "skill" in g.columns:
                base = float(m.iloc[0]["skill"])
        head = ("<tr><th>run</th>"
                + "".join(f'<th class="r">{c}</th>' for c in cols)
                + ('<th class="r delta">&Delta; skill</th>' if base is not None else "")
                + "</tr>")
        body = []
        for _, x in g.iterrows():
            cells = []
            for c in cols:
                v = x[c]
                if pd.isna(v):
                    cells.append('<td class="r dim">&mdash;</td>')
                elif c in ("n", "n_units", "n_folds"):
                    cells.append(f'<td class="r">{int(v)}</td>')
                elif c in ("rmse", "rmse_clim", "bias"):
                    cells.append(f'<td class="r">{v:,.0f}</td>')
                else:
                    cells.append(f'<td class="r">{v:+.3f}</td>')
            if base is not None:
                d = float(x["skill"]) - base
                cells.append(f'<td class="r delta">'
                             f'{"&mdash;" if x["run"] == ref else f"{d:+.3f}"}</td>')
            body.append(f'<tr><td class="sm">{html.escape(str(x["run"]))}</td>'
                        + "".join(cells) + "</tr>")
        out.append(f'<h3 class="tight">{html.escape(str(scope))}</h3>'
                   f'<table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>')
    return "".join(out) or '<p class="na">nothing to tabulate</p>'


def year_charts(runs, metric: str, ylab: str, ylim, w=760, h=300) -> str:
    """One chart per scope; one line per run, so runs overlay for comparison."""
    scopes: dict = {}
    for r in runs:
        f = r["fold"]
        f = f[f.test_year.astype(str) != "ALL"]
        if metric not in f.columns:
            continue
        for scope, g in f.groupby("scope"):
            scopes.setdefault(str(scope), {})[r["label"]] = dict(
                zip(g.test_year.astype(int), g[metric]))
    figs = []
    for scope, series in sorted(scopes.items()):
        years = sorted({y for s in series.values() for y in s})
        if len(years) < 2:
            continue
        figs.append(line_chart(series, years, scope, ylab, ylim, w=w, h=h))
    return (f'<div class="grid">{"".join(figs)}</div>' if figs
            else '<p class="na">no per-year rows</p>')


# ------------------------------------------------------------------- maps --

# How each mappable quantity behaves. "diverging" means zero is the meaningful
# midpoint and the scale is symmetric; "magnitude" means it is a positive
# quantity with no natural centre, scaled from the data. `good` says which
# direction is better, so blue can mean "good" on every map rather than
# "large" on some and "small" on others.
METRIC_KINDS = {
    "skill":      ("diverging", "up",   "skill vs each district's own climatology"),
    "acc":        ("magnitude", "up",   "direction accuracy"),
    "rmse":       ("magnitude", "down", "RMSE (kg/ha)"),
    "rmse_clim":  ("magnitude", "down", "climatology RMSE (kg/ha)"),
    "rmse_pct":   ("magnitude", "down", "RMSE as % of the district's mean yield"),
    "mean_yield": ("magnitude", "up",   "mean yield (kg/ha)"),
    "n":          ("magnitude", "up",   "held-out years scored"),
}


def derive(u: pd.DataFrame) -> pd.DataFrame:
    """Add the columns worth mapping that the runner does not write directly.

    rmse_pct is the one to reach for: raw RMSE is dominated by yield LEVEL --
    a 10 t/ha county in Iowa will out-error a 2 t/ha county in Zambia whatever
    the model does -- so an RMSE map mostly redraws the yield map. Dividing by
    the district's own mean makes the errors comparable across the panel.
    """
    u = u.copy()
    if {"rmse", "mean_yield"} <= set(u.columns):
        u["rmse_pct"] = 100.0 * u["rmse"] / u["mean_yield"].replace(0, pd.NA)
    return u


def metric_scale(metric: str, vals, diff: bool = False) -> tuple:
    """(lo, hi, label) for the colour ramp.

    A difference is always diverging and symmetric, whatever the underlying
    metric. For a magnitude the range comes from the 5th-95th percentile across
    every run so the maps are comparable, trimmed because a handful of extreme
    districts would otherwise flatten the rest to one colour. When lower is
    better the range is handed over INVERTED, which maps small values onto the
    blue end -- so blue means "good" on every map in the report.
    """
    kind, good, label = METRIC_KINDS.get(metric, ("magnitude", "up", metric))
    s = pd.Series(list(vals), dtype=float).dropna()
    if s.empty:
        return -1.0, 1.0, label
    if diff:
        m = float(s.abs().quantile(0.95)) or float(s.abs().max()) or 1.0
        return (m, -m, label) if good == "down" else (-m, m, label)
    if kind == "diverging":
        m = max(float(s.abs().quantile(0.95)), 1e-9)
        return -m, m, label
    lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1e-9)
    return (hi, lo, label) if good == "down" else (lo, hi, label)


def unit_values(run, aoi: str, metric: str) -> dict:
    """{adm2_id: value} for one run and one AOI."""
    u = run["unit"]
    if u.empty or "aoi" not in u.columns:
        return {}
    u = derive(u)
    if metric not in u.columns:
        return {}
    g = u[u.aoi == aoi]
    if g.empty:
        return {}
    # A pooled run has one scope; a direction run has one per target lag. Taking
    # the mean over scopes keeps one map per (run, AOI) rather than a grid that
    # is mostly duplicates.
    return g.groupby("adm2_id")[metric].mean().to_dict()


def fig(ids, box, vals: dict, title: str, sub: str, fid: str, back: str,
        lo: float, hi: float) -> str:
    if not vals:
        return ""
    s = pd.Series(vals, dtype=float)
    return (f'<figure class="fig" id="{fid}">'
            f'<figcaption>'
            f'<a class="exp" href="#{fid}" title="expand">{html.escape(title)} '
            f'<span class="zi">&#9974;</span></a>'
            f'<span class="ylab">{html.escape(sub)}'
            f'<a class="col" href="#{back}" title="collapse">&times;</a></span>'
            f'</figcaption>{svg_map(ids, vals, title, box, lo, hi)}</figure>')


def _fmt(v, signed: bool = False) -> str:
    if pd.isna(v):
        return "n/a"
    if abs(v) >= 100:
        return f"{v:+,.0f}" if signed else f"{v:,.0f}"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def _slug(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", "-".join(str(p) for p in parts)).strip("-")


def map_section(runs, poly_dir: Path, metrics, ref: str | None,
                trim: bool, k: float, crs: str | None = None,
                scale: float | None = None,
                diff_scale: float | None = None) -> str:
    """Per-district maps for every metric and run, plus differences vs `ref`.

    AOI-major, with the metrics nested inside: the geometry is emitted ONCE per
    AOI into a hidden <defs> and every map of that AOI -- all metrics, all runs,
    absolute and difference -- references the same paths. Looping metric-major
    would re-serialise 2000-odd district outlines per metric.
    """
    aois = sorted({a for r in runs for a in
                   (derive(r["unit"]).aoi.unique() if not r["unit"].empty
                    and "aoi" in r["unit"].columns else [])})
    if not aois:
        return '<p class="na">no per-unit metrics in these runs</p>'

    blocks = []
    for aoi in aois:
        per_metric = {m: [(r, v) for r, v in
                          ((r, unit_values(r, aoi, m)) for r in runs) if v]
                      for m in metrics}
        per_metric = {m: h for m, h in per_metric.items() if h}
        if not per_metric:
            continue
        drawn_ids = {a for h in per_metric.values() for _, v in h for a in v}
        polys, err = load_polygons(poly_dir, aoi, crs=crs, keep=drawn_ids)
        if polys is None:
            blocks.append(f'<h3>{html.escape(aoi)}</h3>'
                          f'<p class="na">{html.escape(err)}</p>')
            continue
        drawn = polys[polys.adm_id.isin(drawn_ids)]
        if drawn.empty:
            blocks.append(
                f'<h3>{html.escape(aoi)}</h3><p class="na">no district id in '
                f'the results matched {aoi}.shp &mdash; check adm_id spelling</p>')
            continue
        defs, ids, box, outside = path_defs(drawn, aoi, trim=trim, k=k)
        back = f"maps-{_slug(aoi)}"

        notes = [f"{len(drawn)} districts"]
        if outside:
            notes.append(f"{outside} outside view")
        body = [f'<h3 id="{back}">{html.escape(aoi)} '
                f'<span class="dim">&middot; {" &middot; ".join(notes)}</span>'
                f'</h3>{defs}']

        for m, have in per_metric.items():
            allv = [x for _, v in have for x in v.values()]
            lo, hi, label = metric_scale(m, allv)
            if scale is not None and METRIC_KINDS.get(m, ("", "", ""))[0] == "diverging":
                lo, hi = -scale, scale
            body.append(f'<h4 class="tight"><span class="mono">{html.escape(m)}'
                        f'</span> &mdash; {html.escape(label)}</h4>')
            body.append(map_legend(lo, hi, f"{label} · blue is better",
                                   _slug("lg", aoi, m)))
            figs = []
            for r, vals in have:
                sv = pd.Series(vals, dtype=float)
                figs.append(fig(ids, box, vals, r["label"],
                                f"median {_fmt(sv.median())} · {len(vals)} districts",
                                _slug("m", aoi, m, r["name"]), back, lo, hi))
            body.append(f'<div class="grid maps">{"".join(x for x in figs if x)}</div>')

            refrun = next((r for r, _ in have if r["label"] == ref), None)
            if refrun is None or len(have) < 2:
                continue
            base = unit_values(refrun, aoi, m)
            dvs = {}
            for r, vals in have:
                if r["label"] == ref:
                    continue
                dv = {a: vals[a] - base[a] for a in vals if a in base}
                if dv:
                    dvs[r["name"]] = (r, dv)
            if not dvs:
                continue
            dlo, dhi, _ = metric_scale(
                m, [x for _, d in dvs.values() for x in d.values()], diff=True)
            if diff_scale is not None and METRIC_KINDS.get(m, ("", "", ""))[0] == "diverging":
                good = METRIC_KINDS.get(m, ("", "up", ""))[1]
                dlo, dhi = ((diff_scale, -diff_scale) if good == "down"
                            else (-diff_scale, diff_scale))
            dfigs = []
            for name, (r, dv) in dvs.items():
                sv = pd.Series(dv, dtype=float)
                good = METRIC_KINDS.get(m, ("", "up", ""))[1]
                better = 100.0 * ((sv < 0) if good == "down" else (sv > 0)).mean()
                dfigs.append(fig(
                    ids, box, dv, f"{r['label']}  −  {ref}",
                    f"median {_fmt(sv.median(), signed=True)} · "
                    f"{better:.0f}% of districts better",
                    _slug("d", aoi, m, name), back, dlo, dhi))
            body.append(f'<h4 class="tight">difference in <span class="mono">'
                        f'{html.escape(m)}</span> against <span class="mono">'
                        f'{html.escape(ref)}</span></h4>')
            body.append(map_legend(dlo, dhi,
                                   "blue = this run beats the reference district "
                                   "by district", _slug("lgd", aoi, m)))
            body.append(f'<div class="grid maps">'
                        f'{"".join(x for x in dfigs if x)}</div>')
        blocks.append("".join(body))
    return "".join(blocks) or '<p class="na">nothing mappable</p>'


def _r(a, b) -> float:
    """Pearson r, NaN when undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _year_frame(run, aoi: str, year: int) -> pd.DataFrame:
    """One run's held-out predictions for one AOI and harvest year, indexed by
    adm2_id. Empty for a direction run or a year the run never held out."""
    p = run.get("pred")
    if p is None or p.empty or not {"y_pred", "y_clim", "harvest_year"} <= set(p.columns):
        return pd.DataFrame()
    g = p[(p.aoi == aoi) & (p.harvest_year == year)]
    if g.empty:
        return pd.DataFrame()
    # A pooled run has one scope per row already; mean is a no-op guard.
    return g.groupby("adm2_id")[["y_true", "y_pred", "y_clim"]].mean()


def year_section(runs, poly_dir: Path, years, ref: str, trim: bool, k: float,
                 crs: str | None = None) -> str:
    """Observed and predicted yield in chosen harvest years, side by side.

    Two rows per (AOI, year), each on one colour scale shared by every map in
    the row so the panels can be compared by eye:

      yield     kg/ha. Observed first, then each run's prediction. This is
                mostly the map of which districts are productive -- every run
                gets that right, because the between-district level is in the
                training data -- so read it for the large-scale picture only.
      anomaly   departure from the district's own training-only climatology,
                in % of that climatology: 100 (y - y_clim) / y_clim. This is
                the row that shows whether a run caught THIS season. Percent
                rather than kg/ha so a 2 t/ha and a 10 t/ha district are on the
                same footing.

    The observed anomaly uses the reference run's y_clim. Every run's
    climatology is fit on the same training years when the runs share --years,
    so the choice only matters if they do not.
    """
    runs = [r for r in runs if r["target"] != "direction"]
    aois = sorted({a for r in runs if r.get("pred") is not None
                   and not r["pred"].empty and "aoi" in r["pred"].columns
                   for a in r["pred"].aoi.unique()})
    if not runs or not aois:
        return '<p class="na">no regression predictions in these runs</p>'
    refrun = next((r for r in runs if r["label"] == ref), runs[0])

    blocks = []
    for aoi in aois:
        frames = {yr: [(r, _year_frame(r, aoi, yr)) for r in runs] for yr in years}
        drawn_ids = {a for fr in frames.values() for _, f in fr for a in f.index}
        if not drawn_ids:
            blocks.append(f'<h3>{html.escape(aoi)}</h3><p class="na">none of '
                          f'{", ".join(map(str, years))} was held out in these runs</p>')
            continue
        polys, err = load_polygons(poly_dir, aoi, crs=crs, keep=drawn_ids)
        if polys is None:
            blocks.append(f'<h3>{html.escape(aoi)}</h3><p class="na">{html.escape(err)}</p>')
            continue
        drawn = polys[polys.adm_id.isin(drawn_ids)]
        if drawn.empty:
            blocks.append(f'<h3>{html.escape(aoi)}</h3><p class="na">no district id '
                          f'matched {aoi}.shp</p>')
            continue
        defs, ids, box, _ = path_defs(drawn, aoi, trim=trim, k=k)
        body = [defs]

        for yr in years:
            have = [(r, f) for r, f in frames[yr] if not f.empty]
            back = f"yr-{_slug(aoi, yr)}"
            body.append(f'<h3 id="{back}">{html.escape(aoi)} &middot; harvest {yr}</h3>')
            if not have:
                body.append(f'<p class="na">{yr} was not held out by any run</p>')
                continue
            obs_src = next((f for r, f in have if r is refrun), have[0][1])
            obs = obs_src["y_true"]
            clim = obs_src["y_clim"]

            # ---- absolute yield ----
            allv = pd.concat([obs] + [f["y_pred"] for _, f in have]).dropna()
            lo, hi = float(allv.quantile(0.05)), float(allv.quantile(0.95))
            if hi <= lo:
                hi = lo + 1.0
            body.append('<h4 class="tight">yield (kg/ha)</h4>')
            body.append(map_legend(lo, hi, "kg/ha · same scale in every panel",
                                   _slug("lgy", aoi, yr)))
            figs = [fig(ids, box, obs.to_dict(), f"observed {yr}",
                        f"median {_fmt(obs.median())} · {obs.notna().sum()} districts",
                        _slug("yo", aoi, yr), back, lo, hi)]
            for r, f in have:
                ape = (100 * (f.y_pred - f.y_true).abs() / f.y_true).median()
                figs.append(fig(ids, box, f["y_pred"].to_dict(), r["label"],
                                f"median {_fmt(f.y_pred.median())} · "
                                f"median |error| {ape:.1f}%",
                                _slug("yp", aoi, yr, r["name"]), back, lo, hi))
            body.append(f'<div class="grid maps">{"".join(x for x in figs if x)}</div>')

            # ---- anomaly vs climatology ----
            oa = (100 * (obs - clim) / clim).dropna()
            pa = {r["name"]: (r, (100 * (f.y_pred - f.y_clim) / f.y_clim).dropna())
                  for r, f in have}
            m = float(pd.concat([oa] + [v for _, v in pa.values()]).abs().quantile(0.95)) or 1.0
            body.append('<h4 class="tight">anomaly vs district climatology '
                        '(% of climatology)</h4>')
            body.append(map_legend(-m, m, "% · red below the district's normal, "
                                   "blue above", _slug("lga", aoi, yr)))
            figs = [fig(ids, box, oa.to_dict(), f"observed {yr}",
                        f"median {_fmt(oa.median(), signed=True)}% · "
                        f"{100 * (oa < 0).mean():.0f}% of districts below normal",
                        _slug("ao", aoi, yr), back, -m, m)]
            for name, (r, v) in pa.items():
                both = pd.concat([oa.rename("o"), v.rename("p")], axis=1).dropna()
                rs = _r(both.o, both.p) if len(both) >= 3 else float("nan")
                sign = 100 * (np.sign(both.o) == np.sign(both.p)).mean() if len(both) else float("nan")
                figs.append(fig(ids, box, v.to_dict(), r["label"],
                                f"r vs observed {_fmt(rs, signed=True)} · "
                                f"sign right in {sign:.0f}%",
                                _slug("ap", aoi, yr, name), back, -m, m))
            body.append(f'<div class="grid maps">{"".join(x for x in figs if x)}</div>')
        blocks.append("".join(body))
    return "".join(blocks)


def importance_table(runs, top: int = 12) -> str:
    figs = []
    for r in runs:
        imp = r["imp"]
        if imp.empty or "feature" not in imp.columns:
            continue
        g = (imp.groupby("feature")["importance"].mean()
             .sort_values(ascending=False).head(top))
        tot = float(g.sum()) or 1.0
        rows = "".join(
            f'<tr><td class="mono sm">{html.escape(str(i))}</td>'
            f'<td class="r">{v:.3f}</td>'
            f'<td class="r dim">{100*v/tot:.0f}%</td></tr>'
            for i, v in g.items())
        figs.append(f'<figure class="fig"><figcaption>{html.escape(r["label"])}'
                    f'</figcaption><table><thead><tr><th>feature</th>'
                    f'<th class="r">importance</th><th class="r">share</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></figure>')
    return (f'<div class="grid">{"".join(figs)}</div>' if figs
            else '<p class="na">no importances recorded</p>')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="experiments root; reads {dir}/{run_id}/")
    p.add_argument("--runs", default=None,
                   help="comma-separated run directory names or globs "
                        "(e.g. 'era5_*,dcpp-ens_*'); default: all")
    p.add_argument("--metrics", default=None,
                   help="comma-separated per-district quantities to map. "
                        f"Available: {', '.join(METRIC_KINDS)}. Default "
                        "skill,rmse_pct,rmse for a regression run, acc for "
                        "a direction run")
    p.add_argument("--metric", default=None,
                   help="single-metric alias for --metrics")
    p.add_argument("--reference", default=None,
                   help="run label, directory name, or a glob matching exactly "
                        "one directory (e.g. 'era5_*') to difference against; "
                        "default: the oldest run")
    p.add_argument("--scale", type=float, default=1.0,
                   help="colour range for the absolute maps (default +/-1)")
    p.add_argument("--diff-scale", type=float, default=0.3,
                   help="colour range for the difference maps (default +/-0.3)")
    p.add_argument("--polygons", default=None, help="cy-bench polygons dir")
    p.add_argument("--year", default=None,
                   help="harvest year(s) to map observed vs predicted yield for, "
                        "e.g. 2022 or 2012,2022. Each must be a held-out year of "
                        "the runs: any year under --loo, only Y under --test-year Y")
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--map-full-extent", action="store_true")
    p.add_argument("--map-trim-k", type=float, default=6.0)
    p.add_argument("--map-crs", default=None,
                   help="override the per-AOI fitted Albers projection, e.g. "
                        "EPSG:5070 for CONUS or EPSG:3035 for Europe")
    p.add_argument("--out", default=None,
                   help="default {dir}/yield_experiments.html, i.e. beside the runs")
    p.add_argument("--map-min", type=int, default=620,
                   help="minimum width in px of each map/chart cell. Bigger means "
                        "fewer per row and larger figures (default 620: two per "
                        "row on a wide screen). 900+ gives one per row")
    p.add_argument("--page-width", type=int, default=1560,
                   help="max page width in px (default 1560)")
    p.add_argument("--title", default="Yield experiments — skill and geography")
    a = p.parse_args(argv)

    root = Path(a.dir)
    pats = [x.strip() for x in a.runs.split(",") if x.strip()] if a.runs else None
    runs = load_runs(root, pats)
    if not runs:
        raise SystemExit(f"no run directories with a run_meta.json under {root}")
    print(f"{len(runs)} run(s):")
    for r in runs:
        print(f"  {r['name']}\n    {r['label']}")

    targets = {r["target"] for r in runs}
    if len(targets) > 1:
        print(f"note: mixing targets {sorted(targets)} -- the metric columns "
              f"differ, so the tables are split by target")
    is_dir = targets == {"direction"}
    want = a.metrics or a.metric
    if want:
        metrics = [x.strip() for x in want.split(",") if x.strip()]
        bad = [m for m in metrics if m not in METRIC_KINDS]
        if bad:
            raise SystemExit(f"unknown --metrics {bad}; "
                             f"choose from {list(METRIC_KINDS)}")
    else:
        metrics = ["acc"] if is_dir else ["skill", "rmse_pct", "rmse"]

    ref = a.reference
    if ref is not None:
        match = [r for r in runs if ref in (r["label"], r["name"])] or \
                [r for r in runs if fnmatch.fnmatch(r["name"], ref)]
        if len(match) > 1 and ref not in (match[0]["label"], match[0]["name"]):
            raise SystemExit(f"--reference {ref!r} matched {len(match)} runs: "
                             + ", ".join(r["name"] for r in match))
        if not match:
            raise SystemExit(f"--reference {ref!r} matched no run")
        ref = match[0]["label"]
    else:
        ref = runs[0]["label"]

    ov = overall(runs)
    B = []

    B.append('<h2>1 &middot; Runs compared</h2>')
    B.append('<p class="sub">Every row is one invocation. Only the fields that '
             'differ between runs matter &mdash; the rest of the configuration is '
             'in each run\'s <span class="mono">run_meta.json</span>.</p>')
    cfg = "".join(
        f'<tr><td class="sm">{html.escape(r["label"])}</td>'
        f'<td class="mono sm">{html.escape(r["meta"].get("source","?"))}</td>'
        f'<td class="r">{r["meta"].get("forecast_year","")}</td>'
        f'<td class="mono sm">{html.escape(str(r["meta"].get("model","")))}</td>'
        f'<td class="mono sm">{html.escape(str(r["meta"].get("target","")))}</td>'
        f'<td class="sm">{html.escape(", ".join(r["meta"].get("covariates") or []) or "—")}</td>'
        f'<td class="r">{r["meta"].get("n_features","")}</td>'
        f'<td class="sm">{html.escape(", ".join(r["meta"].get("aois") or []))}</td>'
        f'<td class="mono sm">{html.escape(r["name"])}</td></tr>'
        for r in runs)
    B.append('<table><thead><tr><th>run</th><th>source</th><th class="r">FY</th>'
             '<th>model</th><th>target</th><th>covariates</th>'
             '<th class="r">features</th><th>AOIs</th><th>directory</th>'
             f'</tr></thead><tbody>{cfg}</tbody></table>')

    B.append('<h2>2 &middot; Overall skill</h2>')
    B.append('<p class="sub"><span class="mono">skill</span> = '
             '1 &minus; rmse/rmse_clim against each district\'s own training-only '
             'climatology. Zero means no better than knowing which district this '
             'is; negative means worse. <span class="mono">r2</span> is beside it '
             'for reference and is near 1 for almost any model, because absolute '
             'yield is dominated by which district a row came from.</p>')
    for tgt in sorted(targets):
        sub = ov[ov.target == tgt]
        if sub.empty:
            continue
        if len(targets) > 1:
            B.append(f'<h3>target: <span class="mono">{html.escape(tgt)}</span></h3>')
        cols = (DIR_COLS if tgt == "direction"
                else REF_COLS if tgt in ("delta", "zscore") else REG_COLS)
        B.append(table(sub, [c for c in cols if c in sub.columns], ref))

    B.append('<h2>3 &middot; Skill year by year</h2>')
    B.append('<p class="sub">One held-out year per point, one line per run. A run '
             'that is good on average but swings between folds is carried by a few '
             'years &mdash; with ~20 years that is worth knowing before quoting the '
             'average.</p>')
    B.append(year_charts(runs, "skill", "skill", (-1, 1),
                         w=max(a.map_min + 140, 700), h=330))

    if not is_dir:
        B.append('<h2>4 &middot; Predicted vs observed amplitude</h2>')
        B.append('<p class="sub"><span class="mono">sd_ratio_anom</span> is the sd of '
                 'the predicted anomaly over the sd of the observed one. For a '
                 'calibrated squared-error predictor it should land near '
                 '<span class="mono">r_anom</span>, not near 1 &mdash; a conditional '
                 'mean genuinely varies less than the thing it predicts. Well above '
                 '<span class="mono">r_anom</span> means the model is inheriting the '
                 'steeper between-district slope and over-amplifying.</p>')
        comp = ov[["run", "scope", "r_anom", "sd_ratio_anom"]].copy() \
            if {"r_anom", "sd_ratio_anom"} <= set(ov.columns) else pd.DataFrame()
        if not comp.empty:
            comp["ratio_over_r"] = comp.sd_ratio_anom / comp.r_anom
            rows = "".join(
                f'<tr><td class="sm">{html.escape(str(x["run"]))}</td>'
                f'<td class="sm">{html.escape(str(x["scope"]))}</td>'
                f'<td class="r">{x["r_anom"]:+.3f}</td>'
                f'<td class="r">{x["sd_ratio_anom"]:+.3f}</td>'
                f'<td class="r delta">{x["ratio_over_r"]:.2f}</td></tr>'
                for _, x in comp.iterrows())
            B.append('<table><thead><tr><th>run</th><th>scope</th>'
                     '<th class="r">r_anom</th><th class="r">sd_ratio_anom</th>'
                     '<th class="r delta">ratio / r</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table>')
            B.append('<p class="sub">ratio / r near 1 is calibrated; above 1 is '
                     'over-dispersed; below 1 is shrunk further than the '
                     'correlation justifies.</p>')

    poly_dir = Path(a.polygons) if a.polygons else None
    B.append('<h2>5 &middot; Where it works</h2>')
    if a.no_maps:
        B.append('<p class="na">maps skipped (--no-maps)</p>')
    elif poly_dir is None:
        B.append('<p class="na">Pass <span class="mono">--polygons '
                 '/path/to/cy-bench/polygons</span> for the maps.</p>')
    else:
        B.append(f'<p class="sub">Each district shaded by its own value across '
                 f'the held-out years, one block per metric: '
                 f'<span class="mono">{html.escape(", ".join(metrics))}</span>. '
                 f'<b>Click a caption to expand a map</b>; hover a district for its '
                 f'value. After each metric, the same quantity is differenced '
                 f'against <span class="mono">{html.escape(ref)}</span> district by '
                 f'district &mdash; that is the spatial comparison. '
                 f'<b>Blue is always better</b>, including for RMSE, where the '
                 f'scale is reversed so low error reads blue. Albers equal-area '
                 f'projection fitted to each AOI; colour ranges are the 5th&ndash;'
                 f'95th percentile across runs, so maps of one metric are '
                 f'comparable with each other.</p>')
        B.append('<div class="note"><b>On RMSE maps.</b> Raw '
                 '<span class="mono">rmse</span> is dominated by yield LEVEL: a '
                 '10 t/ha county out-errors a 2 t/ha one whatever the model does, '
                 'so the map largely redraws the yield map. '
                 '<span class="mono">rmse_pct</span> divides by each district\'s own '
                 'mean and is the one to compare across the panel; '
                 '<span class="mono">skill</span> divides by that district\'s '
                 'climatology error and is the one that says whether the model '
                 'helped at all.</div>')
        B.append(map_section(runs, poly_dir, metrics, ref,
                             not a.map_full_extent, a.map_trim_k, a.map_crs,
                             a.scale if a.scale != 1.0 else None,
                             a.diff_scale if a.diff_scale != 0.3 else None))

    if a.year and poly_dir is not None and not a.no_maps:
        yrs = [int(x) for x in a.year.split(",") if x.strip()]
        B.append(f'<h2>5b &middot; Observed vs predicted yield in '
                 f'{", ".join(map(str, yrs))}</h2>')
        B.append('<p class="sub">Each run\'s held-out prediction for the year, '
                 'beside what was observed. The <b>yield</b> row is dominated by '
                 'which districts are productive, which every run learns from the '
                 'training years, so it looks alike across runs by construction. '
                 'The <b>anomaly</b> row removes each district\'s training-only '
                 'climatology and is the one that shows whether a run caught this '
                 'season: compare its pattern with the observed panel. '
                 '<span class="mono">r vs observed</span> is the spatial '
                 'correlation of the two anomaly fields this year.</p>')
        B.append(year_section(runs, poly_dir, yrs, ref,
                              not a.map_full_extent, a.map_trim_k, a.map_crs))

    B.append('<h2>6 &middot; What the models used</h2>')
    B.append('<p class="sub">Mean over folds. Tree importance is impurity '
             'reduction, the linear models report |standardised coefficient| '
             '&mdash; not the same quantity, but both answer which columns the '
             'model is actually using.</p>')
    B.append(importance_table(runs))

    out = Path(a.out) if a.out else root / "yield_experiments.html"

    # The figures are the point of this page, so they get the width. Overridden
    # here rather than in report_dcpp_era5.CSS so the feature-skill report keeps
    # its own proportions.
    #
    # The hover readout is sized in viewBox units, so its on-screen size depends
    # on how far the SVG is scaled -- a bigger thumbnail needs a SMALLER number
    # here, and an expanded map smaller still. These are tuned for a
    # {a.map_min}px cell against the 330-unit viewBox.
    # rendered_px = font_size_in_viewBox_units * (rendered_width / 330), so
    # font_size = desired_px * 330 / rendered_width. A collapsed map is drawn at
    # roughly one cell wide; an expanded one spans the page.
    lbl = 12.0 * 330.0 / max(a.map_min, 120)
    lbl_x = 14.0 * 330.0 / max(a.page_width, 400)
    big = f"""
h4.tight{{font-size:14px;margin:26px 0 6px;color:var(--mut);font-weight:600}}
.wrap{{max-width:{a.page_width}px}}
.grid{{grid-template-columns:repeat(auto-fill,minmax({a.map_min}px,1fr));gap:22px}}
.maps{{grid-template-columns:repeat(auto-fill,minmax({a.map_min}px,1fr));gap:22px}}
.maps .fig{{padding:14px 12px 8px}}
figcaption{{font-size:14px;padding:0 6px 8px}}
.ylab{{font-size:12.5px}}
.u .lbl{{font-size:{lbl:.2f}px;stroke-width:{lbl/4.5:.2f}px}}
.maps .fig:target .u .lbl{{font-size:{lbl_x:.2f}px;stroke-width:{lbl_x/4.5:.2f}px}}
.maps .fig:target .map{{max-height:88vh}}
.map{{min-height:0}}
.maplegend .lg{{width:300px;height:32px}}
h2{{font-size:20px;margin:48px 0 10px}}
@media(max-width:760px){{.grid,.maps{{grid-template-columns:1fr}}}}
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yield Experiments</title><style>{CSS}{big}</style></head>
<body><div class="wrap">
<h1>{html.escape(a.title)}</h1>
<p class="meta">{len(runs)} run(s) &middot; reference
<span class="mono">{html.escape(ref)}</span> &middot; mapped
<span class="mono">{html.escape(", ".join(metrics))}</span></p>
{''.join(B)}
<div class="note"><b>Reading order.</b> Section 2 says which configuration wins
overall; section 5 says whether it wins everywhere or in one region. A positive
average with a difference map that is half red means the gain is regional, not
general &mdash; and with 20 or so years per district, a single district's value
carries a wide confidence interval. Treat the maps as a pattern, not as 3000
independent tests.</div>
</div></body></html>"""
    out.write_text(page, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    flat = out.with_suffix(".csv")
    ov.drop(columns=[c for c in ("target",) if c in ov.columns]).to_csv(flat, index=False)
    print(f"wrote {flat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
