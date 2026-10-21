#!/usr/bin/env python3
"""Report on compare_dcpp_era5.py output: overall skill and per-year skill.

Reads, for each AOI folder under --dir:
    dcpp_era5_corr_*.csv   one row per (member, feature)             -> overall
    dcpp_era5_year_*.csv   one row per (member, feature, year)       -> per-year

and writes one self-contained HTML plus a tidy CSV.

    python python/report_dcpp_era5.py --dir .../comparisons --aois AR,ZM

The five correlations are NOT interchangeable. The report states each one's
definition next to its table rather than leaving the reader to infer it, because
r_pooled and r_space are both near 1 on any realistic panel for reasons that
have nothing to do with forecast skill.

No plotting dependency: tables are plain numbers, charts are hand-built SVG.
"""

from __future__ import annotations

import argparse
import glob
import html
import os
import re
from pathlib import Path

import pandas as pd

# The metrics, in the order a reader should meet them.
METRICS = {
    "r_anom": ("Interannual, pooled",
               "corr over all (u,t) of [x(u,t) - x̄(u)] against [y(u,t) - ȳ(u)]",
               "Does the hindcast track year-to-year variation? Variance-weighted "
               "across units, so high-amplitude units dominate."),
    "r_time": ("Interannual, per unit",
               "mean over u of corr over t of x(u,·) against y(u,·)",
               "Same question, each unit weighted equally. Its sd shows whether "
               "skill is uniform or carried by a few units."),
    "r_acc": ("Spatial anomaly (ACC)",
              "mean over t of corr over u of [x(u,t) - x̄(u)] against [y(u,t) - ȳ(u)]",
              "Within a single season, does the hindcast get the spatial pattern "
              "of the anomaly right? Independent of r_anom."),
    "r_space": ("Spatial, raw",
                "mean over t of corr over u of x(u,t) against y(u,t)",
                "Dominated by the shared climatological field. Near 1 whenever "
                "both sources know which units are hot. Not a skill measure."),
    "r_pooled": ("Pooled, raw",
                 "corr over all (u,t) of x(u,t) against y(u,t)",
                 "Mixes between-unit climatology with interannual signal. The "
                 "between-unit part usually dominates. Reported to be dismissed."),
}
SKILL_ORDER = ["r_anom", "r_time", "r_acc", "r_space", "r_pooled"]
LEAD_COLOURS = {1: "#2563eb", 2: "#059669", 3: "#d97706", 4: "#dc2626", 5: "#7c3aed"}


def colour_for(r):
    if pd.isna(r):
        return "var(--mut)"
    if r >= 0.6:
        return "#15803d"
    if r >= 0.4:
        return "#4d7c0f"
    if r >= 0.2:
        return "#a16207"
    if r >= 0.0:
        return "#b45309"
    return "#b91c1c"


def num(v, dp=3, signed=True):
    if pd.isna(v):
        return '<span class="na">&mdash;</span>'
    fmt = f"{{:+.{dp}f}}" if signed else f"{{:.{dp}f}}"
    return f'<span style="color:{colour_for(v)}">{fmt.format(v)}</span>'


# ------------------------------------------------------------------ loading --

def find_files(d: Path, pattern: str, aois):
    if aois:
        files, empty = [], []
        for aoi in aois:
            hit = sorted(glob.glob(str(d / aoi / pattern)))
            files += hit
            if not hit:
                empty.append(aoi)
        if empty and pattern.startswith("dcpp_era5_corr"):
            avail = sorted(x.name for x in d.iterdir()
                           if x.is_dir() and glob.glob(str(x / pattern)))
            raise SystemExit(
                f"no results for {', '.join(empty)} under {d}\n"
                f"  available: {', '.join(avail) or '(none)'}\n"
                f"  run: python python/compare_dcpp_era5.py --aois {','.join(empty)} --go")
        return files
    nested = sorted(glob.glob(str(d / "*" / pattern)))
    return nested or sorted(glob.glob(str(d / pattern)))


def load(d: Path, pattern: str, aois, required=True) -> pd.DataFrame:
    files = find_files(d, pattern, aois)
    if not files:
        if required:
            raise SystemExit(f"no files matching {pattern} under {d}/<AOI>/")
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "forecast_year" not in df.columns:
            m = re.search(r"_fy(\d+)_", os.path.basename(f))
            df["forecast_year"] = int(m.group(1)) if m else 0
        parent = os.path.basename(os.path.dirname(f))
        if parent != d.name:
            df["aoi"] = parent
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["forecast_year"] = out["forecast_year"].astype(int)
    return out


# ------------------------------------------------------------------- tables --

def metric_table(ens: pd.DataFrame, fy: int) -> str:
    """One row per AOI/feature, one column per correlation, at a single lead."""
    sub = ens[ens.forecast_year == fy]
    cols = [c for c in SKILL_ORDER if c in sub.columns]
    head = "".join(f'<th class="r">{c}</th>' for c in cols)
    rows = []
    for (aoi, feat), g in sub.groupby(["aoi", "feature"], observed=True):
        r = g.iloc[0]
        cells = "".join(f'<td class="r">{num(r.get(c))}</td>' for c in cols)
        sd = (f' <span class="sd">&plusmn;{r["r_time_sd"]:.2f}</span>'
              if "r_time_sd" in r and not pd.isna(r["r_time_sd"]) else "")
        rows.append(
            f'<tr><td>{html.escape(str(aoi))}</td><td class="mono">{html.escape(str(feat))}</td>'
            f'{cells}<td class="r dim">{int(r["n_units"])}</td>'
            f'<td class="r dim">{int(r["n_years"])}</td></tr>'.replace(
                f'<td class="r">{num(r.get("r_time"))}</td>',
                f'<td class="r">{num(r.get("r_time"))}{sd}</td>'))
    return (f'<table><thead><tr><th>AOI</th><th>feature</th>{head}'
            f'<th class="r">units</th><th class="r">yrs</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def lead_compare_table(ens: pd.DataFrame, metric: str, leads) -> str:
    """Rows AOI/feature, one column per lead, plus the change across leads.

    This is the layout for answering "how much skill is lost going from FY1 to
    FY2": the two numbers sit adjacent, and the delta is computed rather than
    left to the reader. The delta is deliberately NOT colour-coded by sign --
    decay with lead is the expected physics, not a defect.
    """
    p = ens.pivot_table(index=["aoi", "feature"], columns="forecast_year",
                        values=metric, aggfunc="mean", observed=True)
    if p.empty:
        return '<p class="na">no data</p>'
    cols = list(p.columns)
    head = "".join(f'<th class="r">FY{c}</th>' for c in cols)
    delta_head = (f'<th class="r delta">&Delta; FY{cols[-1]}&minus;FY{cols[0]}</th>'
                  if len(cols) > 1 else "")
    rows = []
    for (aoi, feat), row in p.iterrows():
        cells = "".join(f'<td class="r">{num(v)}</td>' for v in row)
        d = ""
        if len(cols) > 1:
            dv = row[cols[-1]] - row[cols[0]]
            d = ('<td class="r delta">&mdash;</td>' if pd.isna(dv)
                 else f'<td class="r delta">{dv:+.3f}</td>')
        rows.append(f'<tr><td>{html.escape(str(aoi))}</td>'
                    f'<td class="mono">{html.escape(str(feat))}</td>{cells}{d}</tr>')
    return (f'<table><thead><tr><th>AOI</th><th>feature</th>{head}{delta_head}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def lead_table(ens: pd.DataFrame, metric: str) -> str:
    p = ens.pivot_table(index=["aoi", "feature"], columns="forecast_year",
                        values=metric, aggfunc="mean", observed=True)
    head = "".join(f'<th class="r">FY{c}</th>' for c in p.columns)
    rows = []
    for (aoi, feat), row in p.iterrows():
        cells = "".join(f'<td class="r">{num(v)}</td>' for v in row)
        rows.append(f'<tr><td>{html.escape(str(aoi))}</td>'
                    f'<td class="mono">{html.escape(str(feat))}</td>{cells}</tr>')
    return (f'<table><thead><tr><th>AOI</th><th>feature</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def simple_table(df, index, columns, values, label, dp=3) -> str:
    p = df.pivot_table(index=index, columns=columns, values=values,
                       aggfunc="mean", observed=True)
    if p.empty:
        return '<p class="na">no data</p>'
    head = "".join(f'<th class="r">{html.escape(str(c))}</th>' for c in p.columns)
    rows = []
    for idx, row in p.iterrows():
        name = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        cells = "".join(f'<td class="r">{num(v, dp)}</td>' for v in row)
        rows.append(f'<tr><td>{html.escape(name)}</td>{cells}</tr>')
    return (f'<table><thead><tr><th>{html.escape(label)}</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ------------------------------------------------------------------- charts --

def line_chart(series: dict, years, title, ylab, ylim=(-1, 1), zero=True,
               w=440, h=150):
    """series: {label: {year: value}}. Hand-built SVG, no dependency."""
    pad_l, pad_r, pad_t, pad_b = 42, 8, 10, 24
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    y0, y1 = ylim
    if y1 == y0:
        y1 = y0 + 1
    xs = sorted(years)
    if len(xs) < 2:
        return '<p class="na">need at least two years</p>'

    def px(yr):
        return pad_l + (xs.index(yr) / (len(xs) - 1)) * iw

    def py(v):
        return pad_t + (1 - (v - y0) / (y1 - y0)) * ih

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
             f'aria-label="{html.escape(title)}">']
    # y gridlines
    ticks = [t for t in (-1, -0.5, 0, 0.5, 1) if y0 <= t <= y1] or [y0, (y0+y1)/2, y1]
    for t in ticks:
        strong = (t == 0 and zero)
        parts.append(
            f'<line x1="{pad_l}" y1="{py(t):.1f}" x2="{w-pad_r}" y2="{py(t):.1f}" '
            f'class="{"zero" if strong else "grid"}"/>')
        parts.append(f'<text x="{pad_l-6}" y="{py(t)+3.5:.1f}" class="tick ta-end">{t:g}</text>')
    # x labels: first, last, and a middle one
    for yr in {xs[0], xs[-1], xs[len(xs)//2]}:
        parts.append(f'<text x="{px(yr):.1f}" y="{h-7}" class="tick ta-mid">{yr}</text>')
    for label, data in series.items():
        pts, colour = [], LEAD_COLOURS.get(label if isinstance(label, int) else 0, "#64748b")
        for yr in xs:
            v = data.get(yr)
            if v is None or pd.isna(v):
                continue
            pts.append(f"{px(yr):.1f},{py(max(min(v, y1), y0)):.1f}")
        if not pts:
            continue
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{colour}" stroke-width="1.8"/>')
        for p in pts:
            x, y = p.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.1" fill="{colour}"/>')
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:'
        f'{LEAD_COLOURS.get(l if isinstance(l,int) else 0, "#64748b")}"></i>FY{l}</span>'
        for l in series)
    return (f'<figure class="fig"><figcaption>{html.escape(title)}'
            f'<span class="ylab">{html.escape(ylab)}</span></figcaption>'
            f'{"".join(parts)}<div class="legend">{legend}</div></figure>')


def chart_grid(yearly: pd.DataFrame, metric: str, ylab, ylim=(-1, 1)) -> str:
    """One chart per (AOI, feature); one line per lead."""
    figs = []
    for (aoi, feat), g in yearly.groupby(["aoi", "feature"], observed=True):
        series = {}
        for fy, gg in g.groupby("forecast_year"):
            series[int(fy)] = dict(zip(gg.harvest_year, gg[metric]))
        years = sorted(g.harvest_year.unique())
        figs.append(line_chart(series, years, f"{aoi} · {feat}", ylab, ylim))
    return f'<div class="grid">{"".join(figs)}</div>' if figs else '<p class="na">no per-year data</p>'


# --------------------------------------------------------------------- main --

CSS = """
:root{--bg:#fff;--fg:#17181a;--mut:#6b7280;--line:#e5e7eb;--head:#f9fafb;--card:#fff}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#121417;--fg:#e7e9ea;--mut:#9ba1a6;--line:#2a2e34;--head:#191c20;--card:#171a1e}}
:root[data-theme=dark]{--bg:#121417;--fg:#e7e9ea;--mut:#9ba1a6;--line:#2a2e34;--head:#191c20;--card:#171a1e}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:34px 18px;
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:40px 0 8px;padding-top:16px;border-top:2px solid var(--line)}
h3{font-size:14px;margin:20px 0 6px;color:var(--mut);font-weight:600;
 text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);font-size:13.5px;margin:0 0 14px;max-width:80ch}
.meta{color:var(--mut);font-size:13px;margin:8px 0 0}
table{border-collapse:collapse;width:100%;margin:10px 0 6px;font-size:13.5px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}
thead th{background:var(--head);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.03em;color:var(--mut)}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.sm{font-size:12.5px;white-space:normal;color:var(--mut)}
.defs td{white-space:normal;vertical-align:top}
.dim{color:var(--mut)} .na{color:var(--mut)} .sd{color:var(--mut);font-size:11.5px}
td.delta,th.delta{border-left:2px solid var(--line);font-variant-numeric:tabular-nums;
 color:var(--fg);font-weight:600}
h3.tight{margin:12px 0 2px;text-transform:none;letter-spacing:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;margin:12px 0}
.fig{margin:0;background:var(--card);border:1px solid var(--line);border-radius:7px;padding:10px 8px 4px}
figcaption{font-size:12.5px;font-weight:600;padding:0 6px 4px;display:flex;
 justify-content:space-between;align-items:baseline}
.ylab{color:var(--mut);font-weight:400;font-size:11.5px}
.chart{width:100%;height:auto;display:block}
.grid line.grid{stroke:var(--line);stroke-width:1}
line.grid{stroke:var(--line);stroke-width:1}
line.zero{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 2}
text.tick{fill:var(--mut);font-size:9.5px}
.ta-end{text-anchor:end}.ta-mid{text-anchor:middle}
.legend{padding:2px 6px 6px;font-size:11.5px;color:var(--mut)}
.key{margin-right:10px}.key i{display:inline-block;width:9px;height:9px;border-radius:2px;
 margin-right:4px;vertical-align:-1px}
.note{background:var(--head);border-left:3px solid var(--mut);padding:12px 15px;
 margin:20px 0;font-size:13.5px;border-radius:0 5px 5px 0}
@media(max-width:700px){body{padding:20px 14px}table{display:block;overflow-x:auto}
 .grid{grid-template-columns:1fr}}
"""

def render_page(a, ens, mem, ey, leads, order, css):
    """Assemble the whole HTML document for one set of leads."""
    defs = "".join(
        f'<tr><td class="mono">{k}</td><td>{html.escape(v[0])}</td>'
        f'<td class="mono sm">{html.escape(v[1])}</td><td class="sm">{html.escape(v[2])}</td></tr>'
        for k, v in METRICS.items())

    B = []
    B.append('<h2>1 &middot; What each correlation is</h2>')
    B.append('<p class="sub">x = ERA5, y = DCPP. u indexes admin-2 units, t indexes '
             'harvest years. x̄(u) is unit u\'s mean over the compared years, computed '
             'separately for each source so a constant bias cannot enter any anomaly '
             'correlation.</p>')
    B.append(f'<table class="defs"><thead><tr><th>column</th><th>name</th>'
             f'<th>definition</th><th>what it answers</th></tr></thead>'
             f'<tbody>{defs}</tbody></table>')
    B.append('<p class="sub">Note that <span class="mono">r_time</span> needs no raw/anomaly '
             'distinction: within one unit, removing that unit\'s mean subtracts a constant, '
             'which leaves a correlation unchanged. Within one <em>year</em> the unit means '
             'differ across units, so <span class="mono">r_space</span> and '
             '<span class="mono">r_acc</span> are genuinely different quantities.</p>')

    B.append('<h2>2 &middot; Overall skill by lead, ensemble mean</h2>')
    if len(leads) > 1:
        B.append('<p class="sub">One table per metric so the leads sit side by side. '
                 'The &Delta; column is the change from the shortest to the longest '
                 'lead shown; decay is expected, since a longer lead means the '
                 'initialisation is further from the target season.</p>')
    for m in ("r_anom", "r_time", "r_acc"):
        if m not in ens.columns:
            continue
        name, formula, _ = METRICS[m]
        B.append(f'<h3>{html.escape(name)} &mdash; <span class="mono">{m}</span></h3>')
        B.append(f'<p class="sub mono sm">{html.escape(formula)}</p>')
        B.append(lead_compare_table(ens, m, leads))
    diag = [m for m in ("r_space", "r_pooled") if m in ens.columns]
    if diag:
        B.append('<h3>Diagnostics &mdash; not skill</h3>')
        B.append('<p class="sub">Both are climatology-dominated and sit near 1 on any '
                 'real panel. A value materially below ~0.95 for a temperature-like '
                 'feature points at the regridding or the weight join, not the forecast.</p>')
        for m in diag:
            B.append(f'<h3 class="tight"><span class="mono">{m}</span></h3>')
            B.append(lead_compare_table(ens, m, leads))

    B.append('<h2>2b &middot; All metrics at one lead</h2>')
    B.append('<p class="sub">The same numbers arranged the other way, one lead at a '
             'time, for reading a single lead in isolation.</p>')
    for fy in leads:
        B.append(f'<h3>Lead FY{fy}</h3>')
        B.append(metric_table(ens, fy))

    if not ey.empty and "r_acc" in ey.columns:
        B.append('<h2>3 &middot; Per-year spatial anomaly correlation, ACC(t)</h2>')
        B.append('<p class="sub">Each point is one harvest year: the correlation across '
                 'admin units between the two anomaly fields in that year. This is the '
                 'quantity averaged into <span class="mono">r_acc</span>. Scatter between '
                 'years is expected &mdash; with N units the sampling sd of a single year\'s '
                 'ACC is roughly 1/&radic;(N&minus;1).</p>')
        B.append(chart_grid(ey, "r_acc", "ACC", (-1, 1)))

        B.append('<h2>4 &middot; Per-year bias</h2>')
        B.append('<p class="sub">mean(DCPP &minus; ERA5) across units, in each feature\'s '
                 'own units. A flat non-zero line is a calibration offset; a trending line '
                 'is drift.</p>')
        lo = float(ey.bias.min()) if "bias" in ey else -1.0
        hi = float(ey.bias.max()) if "bias" in ey else 1.0
        span = max(abs(lo), abs(hi)) * 1.15 or 1.0
        B.append(chart_grid(ey, "bias", "bias", (-span, span)))

    if len(leads) > 1:
        B.append(f'<h2>5 &middot; Skill vs lead &mdash; <span class="mono">{a.metric}</span></h2>')
        B.append('<p class="sub">FY1 is initialised in the November its own target season '
                 'begins, so it carries almost no lead and should be read as an upper bound.</p>')
        B.append(lead_table(ens, a.metric))

    if not mem.empty:
        k = ["aoi", "feature", "forecast_year"]
        gain = (ens.set_index(k)[a.metric] - mem.groupby(k, observed=True)[a.metric].mean())
        gain = gain.reset_index(name="gain")
        B.append('<h2>6 &middot; Ensemble mean minus mean single member</h2>')
        B.append('<p class="sub">Positive means averaging members recovers signal that '
                 'individual realisations bury in internal variability.</p>')
        B.append(simple_table(gain, "aoi", ["feature", "forecast_year"], "gain", "AOI"))

    if "sd_ratio_anom" in ens.columns:
        B.append('<h2>7 &middot; Anomaly dispersion</h2>')
        B.append('<p class="sub">sd of DCPP anomalies over sd of ERA5 anomalies. Below 1 the '
                 'hindcast under-varies. The ensemble mean is structurally low here: averaging '
                 'members damps variance, so this is not by itself a model defect.</p>')
        B.append(simple_table(ens, "aoi", ["feature", "forecast_year"],
                              "sd_ratio_anom", "AOI", dp=2))

    cov = ens.groupby("aoi", observed=True).agg(
        features=("feature", "nunique"), units=("n_units", "max"),
        years=("n_years", "max"), pairs=("n_pairs", "max"))
    if "n_members" in ens.columns:
        cov["members"] = ens.groupby("aoi", observed=True)["n_members"].max()
    cov_rows = "".join(
        "<tr><td>" + html.escape(str(i)) + "</td>"
        + "".join(f'<td class="r">{"" if pd.isna(v) else int(v)}</td>' for v in r)
        + "</tr>" for i, r in cov.iterrows())
    B.append('<h2>8 &middot; Coverage</h2>')
    B.append('<table><thead><tr><th>AOI</th>'
             + "".join(f'<th class="r">{c}</th>' for c in cov.columns)
             + f'</tr></thead><tbody>{cov_rows}</tbody></table>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCPP Skill Report</title><style>{css}</style></head>
<body><div class="wrap">
<h1>{html.escape(a.title)}</h1>
<p class="meta">{len(order)} AOI(s) &middot; {ens.feature.nunique()} feature(s) &middot;
leads {', '.join(f'FY{l}' for l in leads)} &middot; cropland-area-weighted admin-2,
crop-calendar season</p>
{''.join(B)}
<div class="note"><b>Reading order.</b> <span class="mono">r_anom</span> and
<span class="mono">r_time</span> answer the temporal question, <span class="mono">r_acc</span>
the spatial one, and they are independent: a model can reproduce a basin-wide
year-to-year signal with no spatial anomaly skill, or the reverse.
<span class="mono">r_space</span> and <span class="mono">r_pooled</span> are included
so that nobody quotes them by accident.</div>
</div></body></html>"""
    return page


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="comparisons root; reads {dir}/{AOI}/")
    p.add_argument("--aois", help="comma-separated AOIs; default: all found")
    p.add_argument("--metric", default="r_anom", choices=SKILL_ORDER,
                   help="headline metric for the skill-vs-lead table")
    p.add_argument("--out", default=None)
    p.add_argument("--per-lead-pages", action="store_true",
                   help="also write one standalone HTML per lead, for slides")
    p.add_argument("--title", default="EC-Earth3 DCPP vs ERA5 — feature skill")
    a = p.parse_args(argv)

    d = Path(a.dir)
    aois = [x.strip() for x in a.aois.split(",") if x.strip()] if a.aois else None
    summ = load(d, "dcpp_era5_corr_*.csv", aois)
    yearly = load(d, "dcpp_era5_year_*.csv", aois, required=False)
    print(f"summary: {len(summ)} rows, AOIs {sorted(summ.aoi.unique())}, "
          f"leads {sorted(summ.forecast_year.unique())}")
    print(f"per-year: {len(yearly)} rows" if not yearly.empty
          else "per-year: none found (older run; rerun compare to get them)")

    if a.out:
        out = Path(a.out)
    else:
        out = d / ("dcpp_skill_report" + (f"_{'-'.join(aois)}" if aois else "") + ".html")

    ens = summ[summ.member == "ENSMEAN"].copy()
    mem = summ[summ.member != "ENSMEAN"].copy()
    if ens.empty:
        raise SystemExit("no ENSMEAN rows found")
    order = ens.groupby("aoi")[a.metric].mean().sort_values(ascending=False).index.tolist()
    for f in (ens, mem):
        f["aoi"] = pd.Categorical(f.aoi, categories=order, ordered=True)
    leads = sorted(summ.forecast_year.unique())
    ey = yearly[yearly.member == "ENSMEAN"].copy() if not yearly.empty else pd.DataFrame()
    if not ey.empty:
        ey["aoi"] = pd.Categorical(ey.aoi, categories=order, ordered=True)

    page = render_page(a, ens, mem, ey, leads, order, CSS)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    # Optional standalone page per lead: same document, filtered to one lead, so
    # a slide can show FY2 alone without the FY1 columns beside it.
    if a.per_lead_pages and len(leads) > 1:
        for fy in leads:
            e1 = ens[ens.forecast_year == fy]
            m1 = mem[mem.forecast_year == fy] if not mem.empty else mem
            y1 = ey[ey.forecast_year == fy] if not ey.empty else ey
            sub = render_page(a, e1, m1, y1, [fy], order, CSS)
            q = out.with_name(f"{out.stem}_FY{fy}.html")
            q.write_text(sub, encoding="utf-8")
            print(f"wrote {q}  ({q.stat().st_size/1024:.0f} KB)")

    flat = out.with_suffix(".csv")
    cols = [c for c in ["aoi", "feature", "forecast_year", "member", *SKILL_ORDER,
                        "r_time_sd", "r_acc_sd", "bias", "rmse", "sd_ratio_anom",
                        "n_pairs", "n_units", "n_years"] if c in summ.columns]
    summ.sort_values(["aoi", "forecast_year", "feature", "member"])[cols].to_csv(flat, index=False)
    print(f"wrote {flat}")
    if not yearly.empty:
        yflat = out.with_name(out.stem + "_per_year.csv")
        yearly.sort_values(["aoi", "forecast_year", "feature", "member",
                            "harvest_year"]).to_csv(yflat, index=False)
        print(f"wrote {yflat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
