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
import json
import os
import re
from pathlib import Path

import pandas as pd

# The metrics, in the order a reader should meet them.
METRICS = {
    "r_anom": ("Interannual, pooled",
               "corr over all (u,t) of [x(u,t) - x̄(u)] against [y(u,t) - ȳ(u)]",
               "Does DCPP track year-to-year variation? Variance-weighted "
               "across units, so high-amplitude units dominate."),
    "r_time": ("Interannual, per unit",
               "mean over u of corr over t of x(u,·) against y(u,·)",
               "Same question, each unit weighted equally. Its sd shows whether "
               "skill is uniform or carried by a few units."),
    "r_acc": ("Spatial anomaly (ACC)",
              "mean over t of corr over u of [x(u,t) - x̄(u)] against [y(u,t) - ȳ(u)]",
              "Within a single season, does DCPP get the spatial pattern "
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


# Written by compare_dcpp_era5.py since the --slice-policy flag existed; absent
# from anything older, which was all fixed-init by definition.
DEFAULT_SLICE_POLICY = "fixed-init"


def policies_in(df: pd.DataFrame) -> list[str]:
    if df.empty or "slice_policy" not in df.columns:
        return [DEFAULT_SLICE_POLICY] if not df.empty else []
    return sorted(df["slice_policy"].fillna(DEFAULT_SLICE_POLICY).unique())


def pick_policy(df: pd.DataFrame, want: str) -> pd.DataFrame:
    """Keep the rows of one slice policy.

    Two policies at the same lead are two different forecasts of the same
    harvest year. Every table here pivots on forecast_year with a mean
    aggregation, so leaving both in would average arm A with arm D and label
    the result FY1 -- which is why main() refuses to continue without a choice
    rather than picking one.
    """
    if df.empty:
        return df
    if "slice_policy" not in df.columns:
        return df if want == DEFAULT_SLICE_POLICY else df.iloc[0:0]
    pol = df["slice_policy"].fillna(DEFAULT_SLICE_POLICY)
    return df[pol == want].copy()


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


# --------------------------------------------------------------------- maps --

# Diverging ramp for a correlation: red through neutral to blue. Anchored at 0
# so the sign is readable at a glance, which a sequential ramp would hide.
_RAMP = [(-1.0, (178, 24, 43)), (-0.5, (214, 96, 77)), (0.0, (240, 240, 240)),
         (0.5, (67, 147, 195)), (1.0, (33, 102, 172))]


def ramp(v, lo=-1.0, hi=1.0):
    if v is None or pd.isna(v):
        return "#d9d9d9"
    v = max(min(float(v), hi), lo)
    for i in range(len(_RAMP) - 1):
        a, ca = _RAMP[i]
        b, cb = _RAMP[i + 1]
        if a <= v <= b:
            f = 0 if b == a else (v - a) / (b - a)
            c = tuple(round(ca[k] + f * (cb[k] - ca[k])) for k in range(3))
            return "#%02x%02x%02x" % c
    return "#d9d9d9"


def _rings(geom):
    polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        yield list(poly.exterior.coords)
        for h in poly.interiors:
            yield list(h.coords)


def _ring_centroid(ring):
    """Area-weighted centroid of one closed ring, or its bbox centre if
    degenerate. Used to place the hover readout inside the unit."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    if abs(a) < 1e-12:
        xs = [x for x, _ in ring]
        ys = [y for _, y in ring]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return cx / (3 * a), cy / (3 * a)


def _ring_area(ring):
    return abs(sum(x0 * y1 - x1 * y0
                   for (x0, y0), (x1, y1) in zip(ring, ring[1:]))) / 2


def view_extent(gdf, trim: bool = True, k: float = 6.0,
                keep_frac: float = 0.75) -> tuple[tuple, int]:
    """Bounding box to draw, and how many units fall outside it.

    The cy-bench shapefiles carry everything a country administers, which for
    some AOIs is most of a hemisphere: FR ships Guadeloupe, Guyane, Reunion and
    Mayotte (lon -63 to +56, so metropolitan France is a tenth of the width),
    and US ships American Samoa, the Northern Marianas and the Aleutians across
    the antimeridian (lon -179 to +180, so CONUS is a sixth of the width).
    Scaling to that extent draws the actual crop area as a postage stamp.

    Outliers are found by median absolute deviation rather than by a quantile.
    A quantile cut has to guess what fraction is remote -- 2% is too little for
    FR, where five of 101 units are overseas, and too much for a compact AOI
    like HU, where dropping one of twenty units was enough to trip a
    span-ratio test. MAD asks the scale-free question instead: is this unit far
    from the bulk in units of how spread the bulk is. For a compact AOI the
    largest deviation is about 2 MAD, so k=6 never fires; the remote
    territories sit at 8 to 25 MAD.

    The trim is then applied only if it actually changes the picture -- if the
    kept units still span `keep_frac` of the full extent there was nothing
    pathological, and the full box is used so nothing is cropped for free. The
    count of excluded units is returned so the caption can say so rather than
    quietly dropping them.
    """
    x0, y0, x1, y1 = gdf.total_bounds
    full = (x0, y0, x1, y1)
    if not trim or len(gdf) < 8:
        return full, 0

    cs = [_ring_centroid(max(_rings(g), key=_ring_area, default=[(0, 0), (0, 0)]))
          for g in gdf.geometry]
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]

    def keep_mask(v):
        med = sorted(v)[len(v) // 2]
        dev = sorted(abs(t - med) for t in v)
        mad = dev[len(dev) // 2]
        if mad <= 0:                 # degenerate: every centroid on one line
            return [True] * len(v)
        return [abs(t - med) <= k * mad for t in v]

    kx, ky = keep_mask(xs), keep_mask(ys)
    keep = [a and b for a, b in zip(kx, ky)]
    if all(keep):
        return full, 0

    # Box over the kept units' GEOMETRY, not their centroids, so a kept unit is
    # never clipped at the edge of its own view.
    boxes = [g.bounds if hasattr(g, "bounds") else None
             for g, kp in zip(gdf.geometry, keep) if kp]
    if any(b is None for b in boxes) or not boxes:
        bx0 = min(x for x, kp in zip(xs, keep) if kp)
        bx1 = max(x for x, kp in zip(xs, keep) if kp)
        by0 = min(y for y, kp in zip(ys, keep) if kp)
        by1 = max(y for y, kp in zip(ys, keep) if kp)
    else:
        bx0 = min(b[0] for b in boxes); by0 = min(b[1] for b in boxes)
        bx1 = max(b[2] for b in boxes); by1 = max(b[3] for b in boxes)

    wf, hf = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    if (bx1 - bx0) / wf >= keep_frac and (by1 - by0) / hf >= keep_frac:
        return full, 0
    return (bx0, by0, bx1, by1), sum(1 for kp in keep if not kp)


def path_defs(gdf, aoi: str, w=330, pad=6, trim: bool = True,
              k: float = 6.0):
    """Serialise this AOI's geometry ONCE, as a hidden <defs> of <path> ids.

    Every map of one AOI then references the same paths through <use>, which is
    the difference between a readable page and an unusable one: mapping six
    features at two leads would otherwise write each of US maize's 3000-odd
    county outlines twelve times -- tens of megabytes of identical coordinates.

    `gdf` must already be restricted to the units that will be mapped, so the
    extent comes from the crop area rather than from every island the country
    administers. All of an AOI's maps then share one transform; side-by-side
    maps drawn at different zooms would invite exactly the wrong comparison.

    The viewBox height follows the data's aspect ratio instead of being fixed,
    so a wide AOI like CONUS fills its box rather than being letterboxed into
    the middle third of it.

    Geometry arrives already projected to an equal-area Albers fitted to this
    AOI (see local_equal_area), so unit areas are preserved and the country
    keeps its familiar shape.
    """
    (x0, y0, x1, y1), outside = view_extent(gdf, trim, k)
    span_x, span_y = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    inner = w - 2 * pad
    # Clamped so one very tall or very wide AOI cannot wreck the grid rhythm.
    h = min(max(inner * span_y / span_x, 150.0), 430.0) + 2 * pad
    sc = min(inner / span_x, (h - 2 * pad) / span_y)
    ox = pad + (inner - span_x * sc) / 2
    oy = pad + ((h - 2 * pad) - span_y * sc) / 2
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(aoi)).strip("-")

    parts, ids = [], {}
    for i, row in enumerate(gdf.itertuples()):
        d, rings = [], []
        for ring in _rings(row.geometry):
            pts = " L".join(f"{ox + (x - x0) * sc:.1f},{oy + (y1 - y) * sc:.1f}"
                            for x, y in ring)
            d.append("M" + pts + "Z")
            rings.append([(ox + (x - x0) * sc, oy + (y1 - y) * sc) for x, y in ring])
        if not d:
            continue
        pid = f"p-{slug}-{i}"
        lx, ly = _ring_centroid(max(rings, key=_ring_area))
        ids[row.adm_id] = (pid, lx, ly)
        parts.append(f'<path id="{pid}" d="{" ".join(d)}"/>')
    defs = (f'<svg class="defs" aria-hidden="true" width="0" height="0">'
            f'<defs>{"".join(parts)}</defs></svg>')
    return defs, ids, (w, round(h, 1)), outside


def svg_map(ids: dict, values: dict, title: str, box,
            lo: float = -1.0, hi: float = 1.0):
    """One choropleth as <use> references into this AOI's shared <defs>.

    Each unit is a <g> holding a <title> and a hidden <text>. The <title> alone
    was not enough: a tooltip declared inside a <use> is resolved
    inconsistently, because the hover target is the shadow instance of the
    referenced <path> rather than the <use> itself. On a <g> it is an ordinary
    element and works everywhere. The <text> is the real readout -- it appears
    immediately on hover instead of after the browser's tooltip delay, and it
    is legible on a projector, which a native tooltip is not.
    """
    w, h = box
    parts = [f'<svg viewBox="0 0 {w} {h}" class="map" role="img" '
             f'aria-label="{html.escape(title)}">',
             '<g fill-rule="evenodd" stroke="#888" stroke-width="0.25">']
    for aid, (pid, lx, ly) in ids.items():
        if aid not in values:
            continue
        v = values[aid]
        lab = "n/a" if v is None or pd.isna(v) else f"{v:+.2f}"
        parts.append(
            f'<g class="u"><title>{html.escape(str(aid))}: {lab}</title>'
            f'<use href="#{pid}" fill="{ramp(v, lo, hi)}"/>'
            f'<text class="lbl" x="{lx:.1f}" y="{ly:.1f}">{lab}</text></g>')
    parts.append("</g></svg>")
    return "".join(parts)


def _tick(v: float) -> str:
    """Legend tick: signed for a diverging scale around zero, plain and
    thousands-separated for a magnitude like kg/ha."""
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:+g}" if abs(v) <= 1.0 else f"{v:g}"


def map_legend(lo: float = -1.0, hi: float = 1.0, label: str | None = None,
               gid: str = "g"):
    stops = "".join(
        f'<stop offset="{(v + 1) / 2:.2%}" stop-color="{ramp(v)}"/>'
        for v, _ in _RAMP)
    txt = label or "per-unit temporal correlation; grey = too few years"
    return (f'<div class="maplegend"><svg viewBox="0 0 220 26" class="lg">'
            f'<defs><linearGradient id="{gid}">{stops}</linearGradient></defs>'
            f'<rect x="10" y="2" width="200" height="9" fill="url(#{gid})" '
            f'stroke="var(--line)"/>'
            f'<text x="10" y="22" class="tick">{lo:+g}</text>'
            f'<text x="110" y="22" class="tick ta-mid">0</text>'
            f'<text x="210" y="22" class="tick ta-end">{hi:+g}</text>'
            f'</svg><span class="sm">{html.escape(txt)}</span></div>')


def local_equal_area(bounds) -> str:
    """An Albers equal-area conic fitted to one AOI, as a proj string.

    Area has to be preserved -- every feature in this pipeline is a
    cropland-AREA-weighted mean, so a map that distorts area misrepresents what
    was averaged. But EPSG:6933, the obvious global equal-area choice, is
    CYLINDRICAL with its standard parallel at 30 degrees: it preserves area
    everywhere and shape nowhere, and the error grows with distance from 30.
    Measured on the 2153 CONUS counties in these runs it renders the country at
    aspect 2.17 against Albers' 1.63 -- a third too wide, visibly wrong to
    anyone who knows the shape of the US.

    An Albers conic with its two standard parallels set at 1/6 and 5/6 of the
    AOI's own latitude span is the textbook fit, minimising distortion across
    the middle of the region while staying exactly equal-area. Deriving it from
    the data means every AOI gets a sensible projection with no lookup table --
    for CONUS it lands essentially on EPSG:5070, and for ZM or BR it adapts
    rather than inheriting a parallel chosen for somewhere else.
    """
    minx, miny, maxx, maxy = bounds
    lon0, lat0 = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    span = maxy - miny
    if span < 1.0:                      # a single-latitude AOI: widen so the
        lat1, lat2 = lat0 - 1.0, lat0 + 1.0   # two parallels stay distinct
    else:
        lat1, lat2 = miny + span / 6.0, maxy - span / 6.0
    return (f"+proj=aea +lat_1={lat1:.4f} +lat_2={lat2:.4f} +lat_0={lat0:.4f} "
            f"+lon_0={lon0:.4f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def load_polygons(poly_dir: Path, aoi: str, crs: str | None = None,
                  keep: set | None = None, simplify_div: float = 2500.0):
    """cy-bench AOI polygons, projected equal-area and lightly simplified.

    `keep` restricts to the units that will actually be drawn BEFORE the
    projection is fitted, so the parallels follow the crop area rather than
    every island the country administers. `crs` overrides the fitted projection
    (e.g. "EPSG:5070").

    The simplify tolerance is a fraction of the AOI's extent. At the previous
    extent/900 a CONUS county -- about 50 km across -- was being simplified at
    4.8 km, which visibly chunks its borders; the maps are drawn up to 1560 px
    wide, where one pixel is under 3 km. Geometry is now emitted once per AOI
    into a shared <defs>, so the finer tolerance costs one copy, not one per
    map.
    """
    import geopandas as gpd          # lazy: only the map path needs it
    shp = poly_dir / aoi / f"{aoi}.shp"
    if not shp.exists():
        return None, f"no shapefile at {shp}"
    g = gpd.read_file(shp)
    if g.crs is None:                # Brazil ships without a .prj
        g = g.set_crs("EPSG:4326")
    idcol = "adm_id" if "adm_id" in g.columns else g.columns[0]
    g = g.rename(columns={idcol: "adm_id"})[["adm_id", "geometry"]]
    if keep:
        sub = g[g.adm_id.isin(keep)]
        if not sub.empty:
            g = sub
    g = g.to_crs(crs or local_equal_area(g.to_crs("EPSG:4326").total_bounds))
    x0, y0, x1, y1 = g.total_bounds
    tol = max(x1 - x0, y1 - y0) / float(simplify_div)
    g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    return g, None


def _fig_id(aoi, feature, fy) -> str:
    """A DOM id safe to put in an href. Feature names are plain tokens, but
    adm/feature spellings vary enough that anything outside [A-Za-z0-9] is
    folded rather than trusted."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{aoi}-{feature}-fy{fy}")
    return f"m-{slug.strip('-')}"


def map_figure(ids: dict, box, sub, aoi, feature, fy, back: str) -> str:
    """One choropleth, collapsed by default and expandable in place.

    Expansion is a :target rule, not script: clicking the caption navigates to
    this figure's id and the CSS gives the targeted figure the full grid width.
    Nothing is overlaid on the map, so the hover readout still works expanded.
    """
    vals = {a: v for a, v in zip(sub.adm2_id, sub.r_time) if a in ids}
    if not vals:
        return (f'<figure class="fig"><figcaption>{html.escape(feature)} '
                f'&middot; FY{fy}</figcaption>'
                f'<p class="na sm">no unit ids matched the shapefile</p></figure>')
    fid = _fig_id(aoi, feature, fy)
    med = sub.r_time.median()
    n_na = int(pd.isna(pd.Series(list(vals.values()))).sum())
    note = f" &middot; {n_na} n/a" if n_na else ""
    return (f'<figure class="fig" id="{fid}">'
            f'<figcaption>'
            f'<a class="exp" href="#{fid}" title="expand">'
            f'<span class="mono">{html.escape(feature)}</span> FY{fy} '
            f'<span class="zi">&#9974;</span></a>'
            f'<span class="ylab">median {med:+.2f} &middot; {len(vals)} units{note}'
            f'<a class="col" href="#{back}" title="collapse">&times;</a></span>'
            f'</figcaption>'
            f'{svg_map(ids, vals, f"{aoi} {feature} FY{fy}", box)}</figure>')


def map_section(byunit: pd.DataFrame, poly_dir: Path, features, leads,
                trim: bool = True, k: float = 6.0,
                crs: str | None = None) -> str:
    """Every feature mapped, side by side, one block per AOI.

    The polygons are cut down to the units that actually appear in the data
    BEFORE the extent is computed. Drawing to the shapefile's own extent is
    what made FR and US unreadable: those files carry overseas departments and
    Pacific territories, and the crop area is a small corner of the result.

    Ordered feature-major so the leads of one feature sit next to each other:
    the comparison these maps exist to support is FY1 against FY2 for the same
    quantity, not two quantities at the same lead.
    """
    blocks = []
    for aoi, g in byunit.groupby("aoi", observed=True):
        aoi = str(aoi)
        shown = set(g[g.feature.isin(features)].adm2_id)
        polys, err = load_polygons(poly_dir, aoi, crs=crs, keep=shown)
        if polys is None:
            blocks.append(f'<h3>{html.escape(aoi)}</h3>'
                          f'<p class="na">{html.escape(err)}</p>')
            continue
        drawn = polys[polys.adm_id.isin(shown)]
        if drawn.empty:
            blocks.append(
                f'<h3>{html.escape(aoi)}</h3><p class="na">none of the '
                f'{len(shown)} unit id(s) in the data matched the {len(polys)} '
                f'id(s) in {aoi}.shp &mdash; check the adm_id spelling</p>')
            continue
        defs, ids, box, outside = path_defs(drawn, aoi, trim=trim, k=k)
        back = f"maps-{re.sub(r'[^A-Za-z0-9]+', '-', aoi).strip('-')}"
        figs = []
        for feat in features:
            gg = g[g.feature == feat]
            if gg.empty:
                continue
            for fy in leads:
                sub = gg[gg.forecast_year == fy]
                if not sub.empty:
                    figs.append(map_figure(ids, box, sub, aoi, feat, fy, back))
        if figs:
            miss = len(shown) - len(drawn)
            notes = [f"{len(figs)} map(s)", f"{len(drawn)} units"]
            if outside:
                notes.append(f"{outside} outside view")
            if miss:
                notes.append(f"{miss} not in shapefile")
            blocks.append(
                f'<h3 id="{back}">{html.escape(aoi)} '
                f'<span class="dim">&middot; {" &middot; ".join(notes)}</span>'
                f'</h3>{defs}'
                f'<div class="grid maps">{"".join(figs)}</div>')
    if not blocks:
        return '<p class="na">no per-unit data to map</p>'
    return map_legend() + "".join(blocks)


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
.maps{grid-template-columns:repeat(auto-fill,minmax(250px,1fr));align-items:start}
/* The readout is sized in viewBox units, so its on-screen size depends on how
   far the SVG is being scaled: a collapsed thumbnail is drawn at ~0.75x of the
   330-unit viewBox and an expanded one at ~3x, hence the much smaller value
   when targeted. paint-order puts the background-coloured stroke behind the
   glyphs so the number stays legible over any fill in the ramp. */
.u .lbl{display:none;pointer-events:none;text-anchor:middle;font-weight:700;
 font-size:13px;fill:var(--fg);stroke:var(--bg);stroke-width:3px;paint-order:stroke;
 font-variant-numeric:tabular-nums}
.u:hover .lbl{display:block}
.u:hover use{stroke:var(--fg);stroke-width:0.9}
.maps .fig:target .u .lbl{font-size:5px;stroke-width:1.4px}
.map{width:100%;height:auto;display:block;background:transparent}
/* Click-to-expand without script: the caption link targets the figure, and a
   targeted figure takes the whole grid row. :target survives printing and
   works with the file opened straight off disk. */
.maps .fig{scroll-margin-top:14px}
svg.defs{position:absolute;width:0;height:0;overflow:hidden}
.maps .fig:target{grid-column:1/-1;border-color:var(--mut);box-shadow:0 2px 14px rgba(0,0,0,.12)}
.maps .fig:target .map{max-height:76vh;margin:0 auto}
.maps .fig:target .col{display:inline}
.maps .fig:target .zi{display:none}
a.exp{color:inherit;text-decoration:none;display:inline-flex;gap:5px;align-items:baseline}
a.exp:hover{text-decoration:underline}
.zi{color:var(--mut);font-size:11px}
a.col{display:none;color:var(--mut);text-decoration:none;font-size:15px;
 line-height:1;padding-left:8px}
a.col:hover{color:var(--fg)}
.maplegend{display:flex;align-items:center;gap:10px;margin:4px 0 10px}
.lg{width:220px;height:26px;flex:0 0 auto}
.note{background:var(--head);border-left:3px solid var(--mut);padding:12px 15px;
 margin:20px 0;font-size:13.5px;border-radius:0 5px 5px 0}
@media(max-width:700px){body{padding:20px 14px}table{display:block;overflow-x:auto}
 .grid{grid-template-columns:1fr}}
"""

def render_page(a, ens, mem, ey, eu, poly_dir, leads, order, css):
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
                 'DCPP under-varies. The ensemble mean is structurally low here: averaging '
                 'members damps variance, so this is not by itself a model defect.</p>')
        B.append(simple_table(ens, "aoi", ["feature", "forecast_year"],
                              "sd_ratio_anom", "AOI", dp=2))

    if not eu.empty and poly_dir is not None and not a.no_maps:
        feats_avail = list(dict.fromkeys(eu.feature))
        # Default is every feature. --map-features narrows it, for a run with
        # twenty features where a slide only needs three.
        want = a.map_features or a.map_feature
        if want:
            asked = [x.strip() for x in want.split(",") if x.strip()]
            feats = [f for f in feats_avail if f in asked]
            missing = [f for f in asked if f not in feats_avail]
        else:
            feats, missing = feats_avail, []
        B.append('<h2>7b &middot; Where the skill is</h2>')
        B.append('<p class="sub">Every feature, mapped per lead. Each unit is '
                 'shaded by its own temporal correlation across years &mdash; the '
                 'per-unit values that <span class="mono">r_time</span> averages. '
                 'Maps sit side by side for comparison; <b>click a caption to '
                 'expand one to full width</b>, and the &times; to collapse it. '
                 'Hover a unit for its id and value. Equal-area projection '
                 '(Albers equal-area, fitted per AOI); grey units have fewer '
                 'than four paired years. '
                 'Remote units that would shrink the crop area to a corner of the frame '
                 '&mdash; overseas departments, Pacific territories &mdash; are left out of '
                 'the view and counted in each AOI heading; <span class="mono">--map-full-extent</span> keeps them.</p>')
        if missing:
            B.append(f'<p class="na">not in this run: '
                     f'<span class="mono">{html.escape(", ".join(missing))}</span> '
                     f'&mdash; available: '
                     f'<span class="mono">{html.escape(", ".join(feats_avail))}</span></p>')
        if not feats:
            B.append('<p class="na">no features left to map after filtering</p>')
        else:
            B.append(map_section(eu, poly_dir, feats, leads,
                                 trim=not a.map_full_extent, k=a.map_trim_k,
                                 crs=a.map_crs))
    elif not a.no_maps and not eu.empty:
        B.append('<h2>7b &middot; Maps</h2>')
        B.append('<p class="na">No polygon directory found. Pass '
                 '<span class="mono">--polygons /path/to/cy-bench/polygons</span>.</p>')

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
    p.add_argument("--slice-policy", default=None,
                   help="which stage-3 stitching to report when the directory "
                        "holds both: fixed-init or newest-run. Required only if "
                        "both are present, since the tables pivot on the lead "
                        "and would otherwise average the two.")
    p.add_argument("--metric", default="r_anom", choices=SKILL_ORDER,
                   help="headline metric for the skill-vs-lead table")
    p.add_argument("--out", default=None)
    p.add_argument("--polygons", default=None,
                   help="cy-bench polygons dir for the maps "
                        "(default: {data_root}/cy-bench/polygons from the run JSON)")
    p.add_argument("--map-features", default=None,
                   help="comma-separated features to map; default: all of them")
    p.add_argument("--map-feature", default=None,
                   help="deprecated single-feature alias for --map-features")
    p.add_argument("--no-maps", action="store_true", help="skip the map section")
    p.add_argument("--map-full-extent", action="store_true",
                   help="draw maps to the shapefile's full extent instead of "
                        "trimming far-flung units (FR's overseas departments, "
                        "US territories and the Aleutians) out of the view")
    p.add_argument("--map-crs", default=None,
                   help="force one CRS for every map (e.g. EPSG:5070); the "
                        "default fits an equal-area Albers per AOI")
    p.add_argument("--map-trim-k", type=float, default=6.0,
                   help="how remote a unit must be to leave the map view, in "
                        "median absolute deviations (default 6). Lower trims "
                        "harder; at 6 only FR and US are affected at all")
    p.add_argument("--per-lead-pages", action="store_true",
                   help="also write one standalone HTML per lead, for slides")
    p.add_argument("--title", default="EC-Earth3 DCPP vs ERA5 — feature skill")
    a = p.parse_args(argv)

    d = Path(a.dir)
    aois = [x.strip() for x in a.aois.split(",") if x.strip()] if a.aois else None
    summ = load(d, "dcpp_era5_corr_*.csv", aois)
    yearly = load(d, "dcpp_era5_year_*.csv", aois, required=False)
    byunit = load(d, "dcpp_era5_unit_*.csv", aois, required=False)
    pols = policies_in(summ)
    if a.slice_policy:
        if a.slice_policy not in pols:
            raise SystemExit(f"--slice-policy {a.slice_policy} is not in these "
                             f"tables; they hold {', '.join(pols)}")
        policy = a.slice_policy
    elif len(pols) > 1:
        raise SystemExit(
            f"{d} holds more than one slice policy ({', '.join(pols)}), and the "
            f"lead-indexed tables would average them into one FY column.\n"
            f"  pick one: --slice-policy {pols[0]}")
    else:
        policy = pols[0] if pols else DEFAULT_SLICE_POLICY
    summ, yearly, byunit = (pick_policy(x, policy) for x in (summ, yearly, byunit))
    print(f"summary: {len(summ)} rows, AOIs {sorted(summ.aoi.unique())}, "
          f"leads {sorted(summ.forecast_year.unique())}, "
          f"slice policy {policy}")
    print(f"per-year: {len(yearly)} rows" if not yearly.empty
          else "per-year: none found (older run; rerun compare to get them)")
    print(f"per-unit: {len(byunit)} rows" if not byunit.empty
          else "per-unit: none found (older run; rerun compare to get them)")

    if a.out:
        out = Path(a.out)
    else:
        # The policy is in the name when it is not the default, so building both
        # arms into one directory does not have the second overwrite the first.
        out = d / ("dcpp_skill_report"
                   + (f"_{'-'.join(aois)}" if aois else "")
                   + ("" if policy == DEFAULT_SLICE_POLICY else f"_{policy}")
                   + ".html")

    poly_dir = Path(a.polygons) if a.polygons else None
    if poly_dir is None and not a.no_maps:
        # The run JSON records the data root stage 3 used, which is where
        # cy-bench lives -- so the maps need no extra flag in the normal case.
        for j in sorted(glob.glob(str(d / "*" / "dcpp_era5_corr_*.json"))):
            try:
                root = json.load(open(j)).get("run", {}).get("data_root")
            except Exception:
                continue
            if root and (Path(root) / "cy-bench" / "polygons").is_dir():
                poly_dir = Path(root) / "cy-bench" / "polygons"
                break

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

    eu = byunit[byunit.member == "ENSMEAN"].copy() if not byunit.empty else byunit
    if not eu.empty:
        eu["aoi"] = pd.Categorical(eu.aoi, categories=order, ordered=True)
    page = render_page(a, ens, mem, ey, eu, poly_dir, leads, order, CSS)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    # Optional standalone page per lead: same document, filtered to one lead, so
    # a slide can show FY2 alone without the FY1 columns beside it.
    if a.per_lead_pages and len(leads) > 1:
        for fy in leads:
            e1 = ens[ens.forecast_year == fy]
            m1 = mem[mem.forecast_year == fy] if not mem.empty else mem
            y1 = ey[ey.forecast_year == fy] if not ey.empty else ey
            u1 = eu[eu.forecast_year == fy] if not eu.empty else eu
            sub = render_page(a, e1, m1, y1, u1, poly_dir, [fy], order, CSS)
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
    if not byunit.empty:
        # Joins to the cy-bench polygons on adm2_id -> adm_id, for mapping in
        # QGIS or R rather than in the page.
        uflat = out.with_name(out.stem + "_per_unit.csv")
        byunit.sort_values(["aoi", "forecast_year", "feature", "member",
                            "adm2_id"]).to_csv(uflat, index=False)
        print(f"wrote {uflat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
