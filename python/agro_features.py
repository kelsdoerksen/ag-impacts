"""
Agronomic (crop-calendar) features for the tabular yield pipeline.

These are physically motivated summaries of the daily climate series over a
county's own crop calendar, rather than plain mean/std over the whole loaded
window. They exist because a tree model given only window means cannot express
"heat accumulated fast enough to finish grain fill" or "the rain came in July,
not April" — both of which drive maize yield far more than a seasonal average.

Two windows are used, and the distinction matters:

  season window  [plant_doy, harvest_end_doy + postharvest_buffer]
      The crop is actually in the ground. Growing degree days, extreme degree
      days and cumulative season precipitation are accumulated here — GDD from
      the day of planting, extended `postharvest_buffer` days past the end of
      the harvest window (14 by default, matching the pipeline's existing
      post-harvest buffer).

  buffer window  [plant_start_doy - preseason_buffer,
                  harvest_end_doy + postharvest_buffer]
      The window the rest of the pipeline already aggregates over (loaded daily
      data is trimmed to exactly this by data_loader.load_crop_climate_data).
      The per-month precipitation statistics are computed here, so the
      pre-season months — which recharge soil moisture before planting — are
      not thrown away.

Features produced
-----------------
    gdd_sum              Σ max(0, (tasmax + tasmin)/2 - gdd_base)  over the
                         season window, in °C-days. gdd_base defaults to 10 °C
                         (50 °F), the maize lower threshold.
                         With method="capped" the textbook maize modification is
                         applied first: tasmax clipped down to 30 °C and tasmin
                         raised up to 10 °C before averaging.
    edd_days_gt{T}       Number of days in the season window with
                         tasmax > T °C (default T = 30).
    edd_sum_gt{T}        Σ max(0, tasmax - T) over the season window, in °C-days
                         — the same heat exposure weighted by how far over the
                         threshold each day went.
    pr_season_sum        Total precipitation (mm) over the season window.
                         (The pipeline's existing pr_mm_sum is the same total
                         over the wider buffer window; both are kept.)
    pr_m{MM}_{stat}      Precipitation per calendar month MM of the yield year
                         over the buffer window, for stat in
                         sum / mean / min / max / var.
    gdd_per_day          gdd_sum / season_days_obs.

season_days_obs and season_days_missing are carried on the returned frame as
diagnostics but are deliberately NOT features — see build_agro_features.

Truncated seasons
-----------------
The EC-Earth3 hindcast for start year s covers Nov 1 of year s to Oct 31 of
year s+1, so on CONUS maize the daily series stops around Oct 31 of the yield
year while the median county's harvest_end + 14 days lands on Dec 1. About 91%
of county-years are therefore short of their requested season window, by a
median of ~33 days.

That matters because degree days and season precipitation are *cumulative*: an
unobserved tail biases them low, and by an amount that tracks the county's
harvest date rather than its weather. This module reports the shortfall on
every run (see `describe`) rather than accumulating silently. In practice the
lost tail is early-winter, when maize GDD is near zero over most of the corn
belt, but it is not zero in the South and it does carry real November rainfall.

Note that the shortfall is a per-county constant here, not a per-year one: the
crop calendar is time-invariant in this dataset and the daily series always ends
on the same date. That is why season_days_obs / season_days_missing are
diagnostics rather than features.

Ragged month coverage
---------------------
Crop calendars differ by county, so no single set of calendar months is fully
inside every county's window (for CONUS maize, only April-September is). A
month is emitted for a county-year only when at least `monthly_min_days` of it
fall inside the window; otherwise its statistics are NaN. Which months are
emitted at all is chosen from the data: a month is kept when at least
`monthly_min_coverage` of county-years clear that day threshold. The resulting
coverage table is returned in the info dict and printed by the loader, so it is
always visible which months a run actually used.
"""

import numpy as np
import pandas as pd

GDD_BASE_C      = 10.0     # maize lower threshold, 50 °F
GDD_CAP_MAX_C   = 30.0     # capped-method upper clip on tasmax, 86 °F
GDD_CAP_MIN_C   = 10.0     # capped-method lower clip on tasmin, 50 °F
EDD_THRESHOLD_C = 30.0     # "extreme" daily maximum temperature

GDD_METHODS      = ["plain", "capped"]
MONTHLY_STATS    = ["sum", "mean", "min", "max", "var"]

# Daily climate columns each part of the block needs, in --climate_vars terms.
REQUIRED_VARS = {
    "gdd":     ["tasmax_C", "tasmin_C"],
    "edd":     ["tasmax_C"],
    "precip":  ["pr_mm"],
}


def edd_column_names(threshold: float = EDD_THRESHOLD_C) -> tuple[str, str]:
    """Column names for the extreme-degree-day pair at a given threshold."""
    tag = f"{threshold:g}".replace(".", "p").replace("-", "m")
    return f"edd_days_gt{tag}", f"edd_sum_gt{tag}"


def monthly_column_names(months, stats=MONTHLY_STATS) -> list:
    """Per-month precipitation column names, month-major then stat order."""
    return [f"pr_m{m:02d}_{s}" for m in months for s in stats]


def _day_index(daily_df: pd.DataFrame) -> pd.Series:
    """Day-of-year of the *yield* year for each row (<= 0 = previous year).

    Matches the arithmetic data_loader.load_crop_climate_data uses to build the
    window (Jan 1 of the yield year + doy - 1), so the season window carved here
    lines up exactly with the window the daily frame was trimmed to.
    """
    jan1 = pd.to_datetime(daily_df["yield_year"].astype(str) + "-01-01")
    return (pd.to_datetime(daily_df["date"]) - jan1).dt.days + 1


def _season_mask(daily_df: pd.DataFrame, day_idx: pd.Series,
                 postharvest_buffer: int) -> pd.Series:
    """Rows inside [plant_doy, harvest_end_doy + postharvest_buffer]."""
    return (day_idx >= daily_df["plant_doy"]) & \
           (day_idx <= daily_df["harvest_end_doy"] + postharvest_buffer)


def _degree_days(daily_df: pd.DataFrame, base: float, method: str) -> pd.Series:
    """Daily growing degree days, °C-days (never negative)."""
    tmax = daily_df["tasmax_C"]
    tmin = daily_df["tasmin_C"]
    if method == "capped":
        tmax = tmax.clip(upper=GDD_CAP_MAX_C)
        tmin = tmin.clip(lower=GDD_CAP_MIN_C)
    return ((tmax + tmin) / 2.0 - base).clip(lower=0.0)


def _drop_gapped_county_years(daily_df, want_temp, want_precip):
    """Remove county-years with any missing day in the variables being summed.

    Every feature here is an accumulation, and pandas' sum() treats NaN as zero:
    a county-year whose daily series is absent would otherwise emerge as
    gdd_sum = 0, pr_season_sum = 0 — a fabricated extreme rather than a gap, and
    one no downstream NaN filter can catch. Partial gaps are just as wrong, only
    less visibly, so any gap disqualifies the county-year.

    Returns (filtered_df, day_index_for_filtered_df, n_dropped).
    """
    cols = []
    if want_temp:
        cols += REQUIRED_VARS["gdd"] + REQUIRED_VARS["edd"]
    if want_precip:
        cols += REQUIRED_VARS["precip"]
    cols = sorted(set(cols))

    bad = daily_df[cols].isna().any(axis=1)
    if not bad.any():
        return daily_df, _day_index(daily_df), 0

    keys    = ["adm2_id", "hindcast_start"]
    gapped  = daily_df.loc[bad, keys].drop_duplicates()
    keep    = ~daily_df.set_index(keys).index.isin(
        pd.MultiIndex.from_frame(gapped))
    filtered = daily_df[keep].copy()
    if filtered.empty:
        raise ValueError(
            f"Every county-year has a gap in {cols}; no agronomic features can "
            f"be computed. Check the climate parquets for this AOI."
        )
    print(f"[agro] Dropped {len(gapped)} county-year(s) with gaps in {cols} — "
          f"cumulative features cannot be formed from a partial series.")
    return filtered, _day_index(filtered), len(gapped)


def _season_block(daily_df, day_idx, postharvest_buffer, gdd_base, gdd_method,
                  edd_threshold, want_temp, want_precip):
    """GDD / EDD / cumulative precipitation over the season window.

    Also returns the observed-vs-requested day counts. The seasonal quantities
    are cumulative, so a season window the daily series does not fully cover
    biases them low by an amount that varies with the county's harvest date —
    see the season_days_missing discussion in build_agro_features.
    """
    season = daily_df[_season_mask(daily_df, day_idx, postharvest_buffer)].copy()
    if season.empty:
        raise ValueError(
            "No daily rows fall inside the season window "
            "[plant_doy, harvest_end_doy + postharvest_buffer]. Check that the "
            "crop calendar columns are populated for this crop/AOI."
        )

    agg = {}
    if want_temp:
        season["_gdd"] = _degree_days(season, gdd_base, gdd_method)
        season["_edd_deg"] = (season["tasmax_C"] - edd_threshold).clip(lower=0.0)
        season["_edd_day"] = (season["tasmax_C"] > edd_threshold).astype(float)
        edd_days_col, edd_sum_col = edd_column_names(edd_threshold)
        agg["gdd_sum"]    = ("_gdd",     "sum")
        agg[edd_days_col] = ("_edd_day", "sum")
        agg[edd_sum_col]  = ("_edd_deg", "sum")
    if want_precip:
        agg["pr_season_sum"] = ("pr_mm", "sum")
    agg["season_days_obs"] = ("date",            "count")
    agg["_plant_doy"]      = ("plant_doy",       "first")
    agg["_harvest_end"]    = ("harvest_end_doy", "first")

    out = season.groupby(["adm2_id", "hindcast_start"]).agg(**agg).reset_index()

    requested = (out.pop("_harvest_end") + postharvest_buffer
                 - out.pop("_plant_doy") + 1)
    out["season_days_missing"] = (requested - out["season_days_obs"]).clip(lower=0)
    if want_temp:
        # Heat accumulated per observed day: the same signal with the length of
        # the observed window divided out, so it is not confounded by how much
        # of the season the daily series happened to cover.
        out["gdd_per_day"] = out["gdd_sum"] / out["season_days_obs"]
    return out, requested


def resolve_monthly_stats(stats: list | None) -> list:
    """Validate a requested monthly statistic subset, preserving canonical order.

    Note that `min` is near-degenerate for daily precipitation — on CONUS maize
    it is exactly 0 for 78-98% of county-years, because almost every month
    contains at least one dry day — so a model can read little from it beyond
    noise. It is included by default because it is a natural thing to ask for,
    but dropping it is a reasonable first move if the block looks overfit.
    """
    if not stats:
        return list(MONTHLY_STATS)
    unknown = [s for s in stats if s not in MONTHLY_STATS]
    if unknown:
        raise ValueError(
            f"Unknown monthly statistic(s) {unknown}. Choices: {MONTHLY_STATS}")
    return [s for s in MONTHLY_STATS if s in stats]


def _monthly_precip_block(daily_df, day_idx, months, min_coverage, min_days, stats):
    """Per-calendar-month precipitation statistics over the buffer window.

    Only days inside the yield calendar year are binned (day_idx >= 1). A
    handful of counties plant early enough that the pre-season buffer reaches
    into the previous December; those days still count toward pr_mm_sum and the
    seasonal totals, but binning them by month number would collide with the
    yield year's own December.

    Returns (wide_df, kept_months, coverage) where coverage maps every candidate
    month to the fraction of county-years with at least min_days of it in window.
    """
    df = daily_df.loc[day_idx >= 1,
                      ["adm2_id", "hindcast_start", "date", "pr_mm"]].copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    grouped = (
        df.groupby(["adm2_id", "hindcast_start", "month"])["pr_mm"]
        .agg(sum="sum", mean="mean", min="min", max="max", var="var", n_days="count")
    )

    n_pairs  = df.groupby(["adm2_id", "hindcast_start"]).ngroups
    n_ok     = grouped["n_days"].ge(min_days).groupby(level="month").sum()
    coverage = {m: float(n_ok.get(m, 0)) / n_pairs for m in range(1, 13)}

    if months is None:
        kept = [m for m in range(1, 13) if coverage[m] >= min_coverage]
    else:
        kept = [int(m) for m in months]
        absent = [m for m in kept if coverage[m] == 0.0]
        if absent:
            raise ValueError(
                f"--monthly_precip_months requested month(s) {absent} that no "
                f"county-year covers. Observed coverage (fraction of county-years "
                f"with >= {min_days} days in window): "
                + ", ".join(f"{m}:{coverage[m]:.2f}" for m in range(1, 13))
            )
    if not kept:
        raise ValueError(
            f"No calendar month reaches the --monthly_min_coverage threshold of "
            f"{min_coverage:.2f} at --monthly_min_days {min_days}. Observed: "
            + ", ".join(f"{m}:{coverage[m]:.2f}" for m in range(1, 13))
        )

    # A month with only a sliver of days inside the window would otherwise
    # report a "monthly mean" computed from two days; blank those instead.
    values = grouped[stats].mask(grouped["n_days"] < min_days)
    wide   = values.loc[values.index.get_level_values("month").isin(kept)].unstack("month")
    wide.columns = [f"pr_m{int(month):02d}_{stat}" for stat, month in wide.columns]
    wide = wide.reindex(columns=monthly_column_names(kept, stats)).reset_index()
    return wide, kept, coverage


def build_agro_features(
    daily_df: pd.DataFrame,
    postharvest_buffer: int = 14,
    gdd_base: float = GDD_BASE_C,
    gdd_method: str = "plain",
    edd_threshold: float = EDD_THRESHOLD_C,
    monthly_precip: bool = True,
    monthly_precip_months: list | None = None,
    monthly_precip_stats: list | None = None,
    monthly_min_coverage: float = 0.95,
    monthly_min_days: int = 20,
    climate_vars: list | None = None,
) -> tuple:
    """Build the agronomic feature block from windowed daily county data.

    Parameters
    ----------
    daily_df : the merged daily frame from data_loader.load_crop_climate_data
        in "aggregated" feature mode — one row per county-day, already trimmed
        to the buffer window, carrying plant_doy / harvest_end_doy / yield_year.
    climate_vars : the run's selected daily variables. Used only to refuse
        combinations that would smuggle an excluded variable back into the
        feature space via a derived feature.

    Returns (agro_df, feature_cols, info): agro_df is one row per
    (adm2_id, hindcast_start), info records the settings and month coverage.
    """
    if gdd_method not in GDD_METHODS:
        raise ValueError(f"gdd_method must be one of {GDD_METHODS}, got {gdd_method!r}")

    selected = set(climate_vars) if climate_vars else None

    def _have(part):
        return selected is None or set(REQUIRED_VARS[part]).issubset(selected)

    want_temp   = _have("gdd") and _have("edd")
    want_precip = _have("precip")
    if not want_temp and not want_precip:
        raise ValueError(
            "Agronomic features need tasmax/tasmin (GDD, EDD) or pr (precip), "
            f"but --climate_vars selected only {sorted(selected)}."
        )
    if not want_temp:
        print("[agro] tasmax/tasmin not in --climate_vars — skipping GDD/EDD.")
    if not want_precip:
        print("[agro] pr not in --climate_vars — skipping precipitation features.")

    missing = [c for c in ["plant_doy", "harvest_end_doy", "yield_year", "date"]
               if c not in daily_df.columns]
    if missing:
        raise ValueError(f"daily_df is missing required column(s) {missing}.")

    daily_df, day_idx, n_gapped = _drop_gapped_county_years(
        daily_df, want_temp, want_precip)

    out, requested = _season_block(daily_df, day_idx, postharvest_buffer,
                                   gdd_base, gdd_method, edd_threshold,
                                   want_temp, want_precip)

    # season_days_obs / season_days_missing are kept on the frame as diagnostics
    # but are NOT features: this dataset's crop calendar is time-invariant per
    # county and the daily series always ends on the same date, so both are
    # constants per county — deterministic functions of plant_doy and
    # harvest_end_doy, which are already in the feature set. Measured on
    # 1990-2004 CONUS maize, 100% of their variance is between counties, i.e.
    # they carry county identity and no weather at all. Against an anomaly
    # target, where the county baseline has already been removed, such a column
    # can only supply overfitting surface.
    diagnostic_cols = ["season_days_obs", "season_days_missing"]
    feature_cols = [c for c in out.columns
                    if c not in ("adm2_id", "hindcast_start", *diagnostic_cols)]

    short = out["season_days_missing"]
    info = {
        "postharvest_buffer": postharvest_buffer,
        "gdd_base_C":         gdd_base,
        "gdd_method":         gdd_method,
        "edd_threshold_C":    edd_threshold,
        "season_days_requested_mean": float(requested.mean()),
        "season_days_obs_mean": float(out["season_days_obs"].mean()),
        "season_days_obs_min":  int(out["season_days_obs"].min()),
        "truncated_frac":     float((short > 0).mean()),
        "truncated_days_median": float(short[short > 0].median()) if (short > 0).any() else 0.0,
        "truncated_days_max": int(short.max()),
        "gapped_county_years": n_gapped,
        "monthly_precip":     bool(monthly_precip and want_precip),
        "monthly_months":     None,
        "monthly_coverage":   None,
        "monthly_min_days":   monthly_min_days,
        "monthly_min_coverage": monthly_min_coverage,
        "monthly_cols":       [],
    }

    if monthly_precip and want_precip:
        stats = resolve_monthly_stats(monthly_precip_stats)
        wide, kept, coverage = _monthly_precip_block(
            daily_df, day_idx, monthly_precip_months,
            monthly_min_coverage, monthly_min_days, stats)
        out = out.merge(wide, on=["adm2_id", "hindcast_start"], how="left")
        monthly_cols = monthly_column_names(kept, stats)
        feature_cols += monthly_cols
        info["monthly_months"]   = kept
        info["monthly_stats"]    = stats
        info["monthly_coverage"] = coverage
        info["monthly_cols"]     = monthly_cols

    return out, feature_cols, info


def describe(info: dict) -> str:
    """One-block human summary of an info dict, for run logs."""
    lines = [
        f"Agronomic feats : season window = [plant_doy, harvest_end_doy + "
        f"{info['postharvest_buffer']}d]  "
        f"({info['season_days_requested_mean']:.0f} days requested, "
        f"{info['season_days_obs_mean']:.0f} observed on average)",
        f"                  GDD base {info['gdd_base_C']:g}°C ({info['gdd_method']}), "
        f"EDD threshold {info['edd_threshold_C']:g}°C",
    ]
    if info["truncated_frac"] > 0:
        lines.append(
            f"                  WARNING: {info['truncated_frac']:.0%} of "
            f"county-years end before harvest_end + "
            f"{info['postharvest_buffer']}d (median "
            f"{info['truncated_days_median']:.0f} days short, max "
            f"{info['truncated_days_max']}) — the daily climate series stops "
            f"there, so cumulative gdd_sum / pr_season_sum are truncated by "
            f"that much. The shortfall is a per-county constant, so it shifts "
            f"each county's level rather than varying year to year."
        )
    if info.get("monthly_precip"):
        cov  = info["monthly_coverage"]
        kept = info["monthly_months"]
        lines.append(
            f"                  monthly precip months {kept} x "
            f"{info['monthly_stats']} (>= {info['monthly_min_days']}d in window "
            f"for >= {info['monthly_min_coverage']:.0%} of county-years)"
        )
        lines.append(
            "                  month coverage: "
            + " ".join(f"{m}:{cov[m]:.2f}" for m in range(1, 13))
        )
    return "\n".join(lines)
