# Brazil yield `harvest_year` is one year late in cy-bench

Checked 2026-11-12. Correct with `run_yield_experiment.py --yield-shift BR=-1`.

## Finding

cy-bench's `yield_maize_BR.csv` records IBGE's reference year Y as
`planting_year = Y`, `harvest_year = Y + 1`, for every one of the 4,699
municipalities. IBGE's year is the harvest year, so every Brazil yield is
labelled one year late. The R script `R/crop-data-gen-cybench.R` passes the
label through unchanged (`hindcast_start = harvest_year - 1`, and
`yield_features.load_yield` adds the 1 back), so the error is in the cy-bench
file, not in this pipeline.

This breaks cy-bench's own convention. The paper (Kallenberg et al. 2026, ESSD
18, 3997–4018, §2.2.3) assigns a crop season to the calendar year in which it
ends: "We define the boundaries of the crop season as 90 d before the start of
season … to the end of season in a particular calendar year." The paper names
IBGE SIDRA table 1612 as the Brazil source and does not mention year
assignment in its limitations.

## Evidence

**1. IBGE vs cy-bench, direct comparison** (SIDRA table 1612, maize, yield in
kg/ha):

| | IBGE 2005 | IBGE 2012 | IBGE 2020 |
|---|---|---|---|
| Passo Fundo (RS), IBGE | 3,750 | 3,600 | 3,900 |
| Passo Fundo, cy-bench | 3,750 at 2006 | 3,600 at 2013 | 3,900 at 2021 |
| Alegrete (RS), IBGE | 960 | 900 | — |
| Alegrete, cy-bench | 960 at 2006 | 900 at 2013 | — |

IBGE puts the Rio Grande do Sul drought losses in the harvest years of the
2004/05, 2011/12 and 2019/20 droughts. The shift applies even in states whose
crop calendar does not cross New Year (Pará, Maranhão: sos ≈ January,
eos ≈ August), so it is uniform. It is not tied to the calendar.

**2. Known droughts, yield table alone.** `python/check_harvest_offset.py`
puts the state-median yield loss at label H+1 for Rio Grande do Sul (5/5
events), Ceará (4/4) and Pernambuco (4/4). The event years are well-known
droughts entered by hand in `EVENTS`, not taken from a cited dataset. Cite a
source (CONAB, EMBRAPA) before relying on them in a paper.

**3. Weather vs yield.** The yield anomaly labelled Y correlates with ERA5
season features of Y−1 (e.g. vpd r ≈ −0.5 in CE, PI and RS) and hardly at
all with those of Y. This holds in 18 of 23 states. ERA5 itself is labelled
correctly: its driest seasons in Rio Grande do Sul are 2012 (rainfall −36%)
and 2020 (−32%). With `--shift -1` the strongest match moves to k = 0 in
18/23 states, and the event losses land at H.

## Effect on the yield experiments

Brazil, RF, LOO 2003–2022, `--target anomaly --detrend`, previous-season
NDVI and fPAR:

| source | skill, unshifted → shifted | r_anom, unshifted → shifted |
|---|---|---|
| ERA5 | +0.014 → +0.057 | 0.17 → 0.33 |
| DCPP FY1 (newest-run) | +0.019 → +0.010 | 0.19 → 0.17 |
| DCPP FY2 | +0.014 → −0.015 | 0.17 → 0.00 |
| DCPP FY3 (newest-run) | +0.019 → −0.005 | 0.20 → 0.06 |

Discard the unshifted runs. Each unshifted row joined climate to the wrong
crop. Its "previous-season" NDVI was the NDVI of the crop being predicted,
which is leakage, and it explains why every source scored about the same
regardless of lead.

## Implementation

- `yield_features.load_yield(..., shift=0)` adds `shift` to `harvest_year`;
  `build_aoi` and `build` pass a per-AOI value through.
- `run_yield_experiment.py --yield-shift BR=-1` prints the shift in the plan
  and records `yield_shift` in `run_meta.json`.
- `check_harvest_offset.py --shift -1` reruns both tests with the correction
  applied.
- Phenology (NDVI, fPAR) needs no shift: its seasons come from observation
  dates (Nov–Oct), not from cy-bench's year label.

## Open points

- Paraná and Mato Grosso do Sul give mixed event results, probably because
  second-season maize (safrinha) dominates there. Table 1612 is total maize,
  and the static WorldCereal calendar ends in June–July, which is later than
  the summer-crop harvest.
- Other AOIs may have the same kind of error. Before using one, compare
  `planting_year` with `harvest_year` in its cy-bench CSV, and check one known
  event against the national source (for the US, 2012).
