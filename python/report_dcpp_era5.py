#!/usr/bin/env python3
"""Turn compare_dcpp_era5.py output into one self-contained HTML report.

Reads every dcpp_era5_corr_*.csv in a comparisons directory -- one per lead
time -- and writes a single page: headline skill per AOI, how skill decays with
lead, what the ensemble mean buys over one member, and the calibration terms.

    python python/report_dcpp_era5.py --dir /mnt/crl_storage/.../comparisons
    python python/report_dcpp_era5.py --dir ... --out ~/meeting/dcpp_skill.html

No plotting dependency: the bars are inline CSS, so the file opens anywhere and
survives being emailed.
"""

from __future__ import annotations

import argparse
import glob
import html
import os
import re
from pathlib import Path

import pandas as pd

# Skill bands used to colour the headline cells. Deliberately coarse -- the
# point in a meeting is "which AOIs are worth modelling", not the third decimal.
BANDS = [(0.6, "#1a7f37", "strong"), (0.4, "#4a9c5d", "moderate"),
         (0.2, "#b8860b", "weak"), (-1.0, "#c0392b", "none")]


def band(r):
    if pd.isna(r):
        return "#999", "n/a"
    for lo, colour, label in BANDS:
        if r >= lo:
            return colour, label
    return "#c0392b", "none"


def load(d: Path, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(d / pattern)))
    if not files:
        raise SystemExit(f"no files matching {pattern} in {d}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # Lead comes from the filename when the column is absent (older runs).
        if "forecast_year" not in df.columns:
            m = re.search(r"_fy(\d+)_", os.path.basename(f))
            df["forecast_year"] = int(m.group(1)) if m else 0
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(files)} file(s), {len(out)} rows, "
          f"leads {sorted(out.forecast_year.unique())}")
    return out


def bar(r, width=90):
    """A signed bar: negative skill is as informative as positive."""
    if pd.isna(r):
        return '<span class="na">n/a</span>'
    colour, _ = band(r)
    frac = max(min(r, 1.0), -1.0)
    w = abs(frac) * width
    off = width if frac >= 0 else width - w
    return (f'<span class="bar"><span class="fill" style="left:{off}px;'
            f'width:{w:.0f}px;background:{colour}"></span></span>'
            f'<span class="num" style="color:{colour}">{r:+.2f}</span>')


def table(df, index, columns, values, label):
    p = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if p.empty:
        return f"<p class='na'>no data for {label}</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in p.columns)
    rows = []
    for idx, row in p.iterrows():
        cells = "".join(f"<td>{bar(v)}</td>" for v in row)
        name = " / ".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        rows.append(f"<tr><th class='rh'>{html.escape(name)}</th>{cells}</tr>")
    return (f"<table><thead><tr><th class='rh'>{html.escape(label)}</th>{head}</tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="directory holding dcpp_era5_corr_*.csv")
    p.add_argument("--pattern", default="dcpp_era5_corr_*.csv")
    p.add_argument("--out", default=None, help="output .html (default: {dir}/dcpp_skill_report.html)")
    p.add_argument("--metric", default="r_anom",
                   help="skill metric to headline (r_anom, r_time, r_space, r_pooled)")
    p.add_argument("--title", default="EC-Earth3 DCPP vs ERA5 -- feature skill")
    a = p.parse_args(argv)

    d = Path(a.dir)
    df = load(d, a.pattern)
    out = Path(a.out) if a.out else d / "dcpp_skill_report.html"
    M = a.metric

    ens = df[df.member == "ENSMEAN"].copy()
    mem = df[df.member != "ENSMEAN"].copy()
    leads = sorted(df.forecast_year.unique())

    # AOIs ordered best-first: lead with the ones worth modelling.
    order = (ens.groupby("aoi")[M].mean().sort_values(ascending=False).index.tolist())
    ens["aoi"] = pd.Categorical(ens.aoi, categories=order, ordered=True)
    mem["aoi"] = pd.Categorical(mem.aoi, categories=order, ordered=True)

    # Ensemble gain: ENSMEAN minus the average single member, same (aoi, feature, lead).
    k = ["aoi", "feature", "forecast_year"]
    gain = (ens.set_index(k)[M] - mem.groupby(k)[M].mean()).reset_index(name="gain")

    blocks = []
    blocks.append(f"<h2>Headline skill &mdash; ensemble mean, {html.escape(M)}</h2>")
    blocks.append("<p class='sub'>AOIs ordered best first. Each cell is one feature.</p>")
    for fy in leads:
        sub = ens[ens.forecast_year == fy]
        blocks.append(f"<h3>Lead time FY{fy}</h3>")
        blocks.append(table(sub, "aoi", "feature", M, "AOI"))

    if len(leads) > 1:
        blocks.append("<h2>Skill vs lead time</h2>")
        blocks.append("<p class='sub'>FY1 is initialised at the start of its own "
                      "target season, so it is close to an upper bound. The "
                      "question is how fast this decays.</p>")
        blocks.append(table(ens, ["aoi", "feature"], "forecast_year", M, "AOI / feature"))

    blocks.append("<h2>What the ensemble mean buys</h2>")
    blocks.append("<p class='sub'>Ensemble mean minus the average single member. "
                  "Positive means averaging members recovers signal.</p>")
    blocks.append(table(gain, "aoi", "feature", "gain", "AOI"))

    if "sd_ratio_anom" in ens.columns:
        blocks.append("<h2>Calibration</h2>")
        blocks.append("<p class='sub'>sd(DCPP anomalies) / sd(ERA5 anomalies). "
                      "Below 1 means the hindcast under-varies; the ensemble mean "
                      "is always low here because averaging damps variance.</p>")
        blocks.append(table(ens, "aoi", "feature", "sd_ratio_anom", "AOI"))

    n = ens.groupby("aoi", observed=True).agg(
        features=("feature", "nunique"), pairs=("n_pairs", "max"),
        units=("n_units", "max"), years=("n_years", "max"),
        members=("n_members", "max") if "n_members" in ens.columns else ("n_pairs", "size"))
    cov = "".join(
        f"<tr><td>{html.escape(str(i))}</td><td>{int(r.features)}</td>"
        f"<td>{int(r.units)}</td><td>{int(r.years)}</td><td>{int(r.pairs)}</td>"
        f"<td>{'' if pd.isna(r.members) else int(r.members)}</td></tr>"
        for i, r in n.iterrows())

    css = """
    :root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--head:#f6f6f6}
    @media(prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#15171a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2c2f34;--head:#1d2024}}
    :root[data-theme=dark]{--bg:#15171a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2c2f34;--head:#1d2024}
    *{box-sizing:border-box}
    body{background:var(--bg);color:var(--fg);margin:0;padding:32px 16px;
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
    .wrap{max-width:1100px;margin:0 auto}
    h1{font-size:25px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 6px;
      padding-top:14px;border-top:1px solid var(--line)}
    h3{font-size:15px;margin:18px 0 6px;color:var(--mut);font-weight:600}
    .sub{color:var(--mut);margin:0 0 12px;font-size:13.5px}
    table{border-collapse:collapse;width:100%;margin:8px 0 4px;font-size:13.5px}
    th,td{padding:5px 9px;text-align:left;border-bottom:1px solid var(--line);
      white-space:nowrap}
    thead th{background:var(--head);font-weight:600;font-size:12.5px}
    .rh{font-weight:600}
    .bar{position:relative;display:inline-block;width:180px;height:11px;
      background:linear-gradient(90deg,transparent 89px,var(--line) 89px,
      var(--line) 91px,transparent 91px);vertical-align:middle;border-radius:2px}
    .fill{position:absolute;top:0;height:11px;border-radius:2px;opacity:.85}
    .num{margin-left:9px;font-variant-numeric:tabular-nums;font-weight:600;font-size:12.5px}
    .na{color:var(--mut)}
    .meta{color:var(--mut);font-size:13px;margin:10px 0 0}
    .note{background:var(--head);border-left:3px solid var(--line);
      padding:11px 14px;margin:18px 0;font-size:13.5px;border-radius:0 4px 4px 0}
    .note b{font-weight:600}
    code{background:var(--head);padding:1px 5px;border-radius:3px;font-size:12.5px}
    @media(max-width:700px){body{padding:20px 16px}
      table{display:block;overflow-x:auto}.bar{width:120px;
      background:linear-gradient(90deg,transparent 59px,var(--line) 59px,
      var(--line) 61px,transparent 61px)}}
    """

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCPP Skill Report</title><style>{css}</style></head><body><div class="wrap">
<h1>{html.escape(a.title)}</h1>
<p class="meta">Leads {', '.join(f'FY{l}' for l in leads)} &middot;
{len(order)} AOI(s) &middot; {ens.feature.nunique()} feature(s) &middot;
metric <code>{html.escape(M)}</code></p>
<div class="note"><b>How to read this.</b> <code>r_anom</code> is the correlation
between the hindcast and ERA5 after removing each admin unit's own mean from both
sides &mdash; it measures whether the hindcast tracks <em>interannual</em> variation.
The raw pooled correlation is far higher but mostly reflects units differing in
mean climate, so it is not used here. Values are cropland-area-weighted admin-2
features over each unit's crop-calendar season.</div>
{''.join(blocks)}
<h2>Coverage</h2>
<table><thead><tr><th>AOI</th><th>features</th><th>units</th><th>years</th>
<th>pairs</th><th>members</th></tr></thead><tbody>{cov}</tbody></table>
<div class="note"><b>Caveat for discussion.</b> FY1 is initialised in the November
its target season begins, so it has almost no lead and should be read as an upper
bound rather than a forecast result. Skill at FY2&ndash;FY5 is the decadal-prediction
question.</div>
</div></body></html>"""

    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    flat = out.with_suffix(".csv")
    cols = [c for c in ["aoi", "feature", "forecast_year", "member", M, "r_time",
                        "bias", "sd_ratio_anom", "n_pairs", "n_units", "n_years"]
            if c in df.columns]
    df.sort_values(["aoi", "forecast_year", "feature", "member"])[cols].to_csv(flat, index=False)
    print(f"wrote {flat}  (tidy table for slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
