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

import pandas as pd

from report_dcpp_era5 import (CSS, line_chart, load_polygons, map_legend,
                              path_defs, svg_map)

# Regression metrics worth a column, in reading order. skill first: it is the
# only one that is zero when the model has learned nothing useful.
REG_COLS = ["n", "n_units", "n_folds", "skill", "rmse", "rmse_pct", "rmse_clim",
            "r2", "r_anom", "r_space_mean", "sd_ratio_anom", "bias", "alpha"]
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
        if fold.empty:
            print(f"  skipping {d.name}: no metrics_by_fold.csv")
            continue
        runs.append({"dir": d, "name": d.name, "meta": meta,
                     "label": label_for(meta), "fold": fold, "unit": unit,
                     "imp": imp,
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


def year_charts(runs, metric: str, ylab: str, ylim) -> str:
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
        figs.append(line_chart(series, years, scope, ylab, ylim))
    return (f'<div class="grid">{"".join(figs)}</div>' if figs
            else '<p class="na">no per-year rows</p>')


# ------------------------------------------------------------------- maps --

def unit_values(run, aoi: str, metric: str) -> dict:
    """{adm2_id: value} for one run and one AOI."""
    u = run["unit"]
    if u.empty or "aoi" not in u.columns or metric not in u.columns:
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


def _slug(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", "-".join(str(p) for p in parts)).strip("-")


def map_section(runs, poly_dir: Path, metric: str, lo: float, hi: float,
                ref: str | None, diff_scale: float, trim: bool, k: float) -> str:
    """Per-district skill for every run, plus difference maps against `ref`.

    Both the absolute and the difference map for one AOI reference the same
    geometry, emitted once into a hidden <defs>, so adding runs costs fills
    rather than coordinates.
    """
    aois = sorted({a for r in runs for a in
                   (r["unit"].aoi.unique() if not r["unit"].empty
                    and "aoi" in r["unit"].columns else [])})
    if not aois:
        return '<p class="na">no per-unit metrics in these runs</p>'

    blocks = []
    for aoi in aois:
        have = [(r, unit_values(r, aoi, metric)) for r in runs]
        have = [(r, v) for r, v in have if v]
        if not have:
            continue
        polys, err = load_polygons(poly_dir, aoi)
        if polys is None:
            blocks.append(f'<h3>{html.escape(aoi)}</h3>'
                          f'<p class="na">{html.escape(err)}</p>')
            continue
        drawn = polys[polys.adm_id.isin({a for _, v in have for a in v})]
        if drawn.empty:
            blocks.append(
                f'<h3>{html.escape(aoi)}</h3><p class="na">no district id in '
                f'the results matched {aoi}.shp &mdash; check adm_id spelling</p>')
            continue
        defs, ids, box, outside = path_defs(drawn, aoi, trim=trim, k=k)
        back = f"maps-{_slug(aoi)}"

        figs = []
        for r, vals in have:
            s = pd.Series(vals, dtype=float)
            figs.append(fig(ids, box, vals, r["label"],
                            f"median {s.median():+.3f} · {len(vals)} districts",
                            _slug("m", aoi, r["name"]), back, lo, hi))
        absolute = "".join(x for x in figs if x)

        diffs = []
        refrun = next((r for r, _ in have if r["label"] == ref), None)
        if refrun is not None and len(have) > 1:
            base = unit_values(refrun, aoi, metric)
            for r, vals in have:
                if r["label"] == ref:
                    continue
                dv = {a: vals[a] - base[a] for a in vals if a in base}
                if not dv:
                    continue
                s = pd.Series(dv, dtype=float)
                better = 100.0 * (s > 0).mean()
                diffs.append(fig(
                    ids, box, dv, f"{r['label']}  −  {ref}",
                    f"median {s.median():+.3f} · {better:.0f}% of districts better",
                    _slug("d", aoi, r["name"]), back, -diff_scale, diff_scale))

        notes = [f"{len(drawn)} districts"]
        if outside:
            notes.append(f"{outside} outside view")
        blocks.append(
            f'<h3 id="{back}">{html.escape(aoi)} '
            f'<span class="dim">&middot; {" &middot; ".join(notes)}</span></h3>'
            f'{defs}<div class="grid maps">{absolute}</div>'
            + (f'<h4 class="tight">difference against <span class="mono">'
               f'{html.escape(ref)}</span></h4>'
               f'{map_legend(-diff_scale, diff_scale, "blue = this run beats the reference district by district", _slug("lgd", aoi))}'
               f'<div class="grid maps">{"".join(diffs)}</div>' if diffs else ""))
    return "".join(blocks) or '<p class="na">nothing mappable</p>'


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
    p.add_argument("--metric", default=None,
                   help="per-district quantity to map; default skill "
                        "(regression) or acc (direction)")
    p.add_argument("--reference", default=None,
                   help="run label or directory name to difference against; "
                        "default: the oldest run")
    p.add_argument("--scale", type=float, default=1.0,
                   help="colour range for the absolute maps (default +/-1)")
    p.add_argument("--diff-scale", type=float, default=0.3,
                   help="colour range for the difference maps (default +/-0.3)")
    p.add_argument("--polygons", default=None, help="cy-bench polygons dir")
    p.add_argument("--no-maps", action="store_true")
    p.add_argument("--map-full-extent", action="store_true")
    p.add_argument("--map-trim-k", type=float, default=6.0)
    p.add_argument("--out", default=None)
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
    metric = a.metric or ("acc" if is_dir else "skill")

    ref = a.reference
    if ref is not None:
        match = [r for r in runs if ref in (r["label"], r["name"])]
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
        B.append(table(sub, DIR_COLS if tgt == "direction" else REG_COLS, ref))

    B.append('<h2>3 &middot; Skill year by year</h2>')
    B.append('<p class="sub">One held-out year per point, one line per run. A run '
             'that is good on average but swings between folds is carried by a few '
             'years &mdash; with ~20 years that is worth knowing before quoting the '
             'average.</p>')
    B.append(year_charts(runs, "skill", "skill", (-1, 1)))

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
        B.append(f'<p class="sub">Each district shaded by its own '
                 f'<span class="mono">{html.escape(metric)}</span> across the '
                 f'held-out years. <b>Click a caption to expand a map</b>; hover a '
                 f'district for its value. The second block differences each run '
                 f'against <span class="mono">{html.escape(ref)}</span> district by '
                 f'district &mdash; that is the spatial comparison: blue where this '
                 f'configuration beats the reference, red where it loses. '
                 f'Equal-area projection (EPSG:6933).</p>')
        B.append(map_legend(-a.scale, a.scale,
                            f"per-district {metric}", "lgabs"))
        B.append(map_section(runs, poly_dir, metric, -a.scale, a.scale, ref,
                             a.diff_scale, not a.map_full_extent, a.map_trim_k))

    B.append('<h2>6 &middot; What the models used</h2>')
    B.append('<p class="sub">Mean over folds. Tree importance is impurity '
             'reduction, the linear models report |standardised coefficient| '
             '&mdash; not the same quantity, but both answer which columns the '
             'model is actually using.</p>')
    B.append(importance_table(runs))

    out = Path(a.out) if a.out else root / "yield_experiments.html"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yield Experiments</title><style>{CSS}
h4.tight{{font-size:13px;margin:18px 0 4px;color:var(--mut);font-weight:600}}
</style></head>
<body><div class="wrap">
<h1>{html.escape(a.title)}</h1>
<p class="meta">{len(runs)} run(s) &middot; reference
<span class="mono">{html.escape(ref)}</span> &middot; mapped metric
<span class="mono">{html.escape(metric)}</span></p>
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
