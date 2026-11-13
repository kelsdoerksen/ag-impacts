# Yield targets in `run_yield_experiment.py`

Choose a target with `--target`. Every regression target is converted back to
kg/ha before scoring, so the kg/ha columns (`rmse`, `rmse_pct`, `skill`,
`r_anom`, maps) can be compared across targets. Folds are always whole years
(leave-one-year-out with `--loo`). Every reference value used to define or
judge a target is computed from training years only, or from past yields
already known when the forecast is issued.

Notation: `y[u,t]` is the yield of unit u in harvest year t. `c[u,t]` is the
unit's baseline, fitted on the training years of each fold: its mean, or its
linear trend with `--detrend`. `F` is the forecast year (`--forecast-year`;
0 for ERA5).

## The five targets

| `--target` | model learns | what it means | null it must beat |
|---|---|---|---|
| `yield` | `y` (kg/ha) | absolute yield | `c` |
| `anomaly` | `y − c` | deviation from the unit's normal or trend, in kg/ha | `c` (i.e. 0 anomaly) |
| `delta` | `y − y[t−k]`, or `100·(y − y[t−k])/y[t−k]` with `--pct` | change from k years earlier | `c`, written as a change: `c − y[t−k]` |
| `zscore` | `(y − m)/s`, where m and s are the mean and sd of `y[t−k−w+1 … t−k]` | how many recent standard deviations yield moves, against the last w years | `c`, written as a z-score: `(c − m)/s` |
| `direction` | class: down / flat / up compared with `y[t−k]` | sign of the change, with a "flat" band of ±`--flat-band` × the unit's own sd of k-year changes | the training majority class |

## Your three targets

- **Absolute yield:** `--target yield --detrend`. Without `--detrend` the
  null is a flat mean, and a tree model cannot extrapolate the technology
  trend under LOO. The error is then mostly trend, not weather.
- **Standard deviations from the last 5 years:**
  `--target zscore --z-window 5`. The window ends at year t−k, with
  k = `--target-lags` (default `max(1, F)`), so at FY1 it is t−5 … t−1.
  - The sd is floored at `--z-floor` × mean (default 0.05). Five values give
    a noisy sd, and repeated yields (19.5% of Brazil's municipality-years
    equal the previous year's) give exactly 0.
  - A row needs at least 3 of the 5 years.
- **Change from the previous year:** `--target delta` gives kg/ha;
  `--target delta --pct` gives %. The reference is y[t−k] with
  k = `--target-lags` (default `max(1, F)`). For the change as up / flat /
  down instead of an amount, use `--target direction`.

## Rules the code enforces

- **Reference years must already be observed.** A lead-F forecast is issued
  on 1 Nov of year t−F, so the most recent known harvest is t−F. A reference
  that includes a later year is refused: `--target-lags` must be ≥ F.
  - At FY2, "change from last year" doesn't exist as a forecast. The
    nearest is change from t−2.
- **One reference per delta / zscore run.** Run once per `--target-lags`
  value; `direction` accepts a comma-separated list.
- **Rows without a usable reference are dropped.** That happens when y[t−k]
  is missing, fewer than 3 window years exist, or the mean is ≤ 0. The plan
  prints how many rows remain. A delta or zscore run therefore covers
  slightly fewer rows than a yield run.

## Why delta and zscore need a different null

Year-to-year changes revert. After a drought year, the next change is almost
certainly "up" because the bad year is over, not because of anything the
model knows. Scored against "no change" (T = 0), a model would collect that
reversion as skill without any weather information.

So both targets are judged against the same climatology null as `yield` and
`anomaly`: the unit's baseline `c`, written in target units as
`T_clim = (c − R)/S`. That null already contains the reversion.

The weather features can't see last year's yield, so `T_clim` is also given
to the model as a feature called `ref_clim`. The model then has the null in
hand and can only beat it through the weather. Turn this off with
`--no-ref-feature`, for example to see how much the weather does on its own.

| column | meaning |
|---|---|
| `skill`, `r_anom`, `rmse_pct` | kg/ha, against `c`. Comparable with every other target |
| `t_skill` | 1 − RMSE/RMSE_null in target units, against `T_clim`. **The headline for delta / zscore** |
| `t_r` | correlation of predicted and observed target |
| `t_skill_zero` | against T = 0 ("no change"). Includes free reversion credit; don't quote it as skill |
| `t_skill_clim_vs_zero` | how much the null alone beats T = 0, i.e. the size of the reversion credit |

Synthetic check (60 units × 20 years, yield = level + trend + weather +
noise):
- **No useful feature:** `t_skill` ≈ −0.1, but `t_skill_zero` = +0.22. The
  reversion credit appears even with nothing to learn from.
- **With the weather feature:** delta and zscore score the same kg/ha skill
  as `anomaly` (≈ +0.47 against +0.49).

For delta in kg/ha, the target error equals the kg/ha error, so `t_skill` =
`skill`. Predicting the change well is the same problem as predicting the
anomaly well. The targets differ only in what the model is asked to output
and in the weighting across units: `--pct` and `zscore` rescale each row, so
units with small or volatile yields count more.

## Comparing forecast years

With the default `--target-lags max(1, F)`, the target itself changes with
the lead:
- delta: change from t−1 at FY1, from t−3 at FY3.
- zscore: the window shifts back with the lead.

That mixes "harder target" with "worse forecast". To compare ERA5, FY1, FY2
and FY3 on the same target, fix the reference at the longest lead,
`--target-lags 3`, for all four runs. Use the default only when each lead
should answer its own most useful question.

## Running (Brazil, all four sources, XGBoost)

`--yield-shift BR=-1` is required for Brazil; see
`docs/brazil-harvest-year-offset.md`.

```sh
cd ~/code/ag-impacts
PY=/opt/anaconda3/envs/ag/bin/python; D=~/Desktop/AgImpacts
COMMON="--aois BR --model xgb --add ndvi,fpar --phenology-lag 1 --detrend \
  --years 2003-2022 --loo --share $D --out-dir $D/experiments --yield-shift BR=-1"
SOURCES=("--source era5"
         "--source dcpp:ens --forecast-year 1 --slice-policy newest-run"
         "--source dcpp:ens --forecast-year 2"
         "--source dcpp:ens --forecast-year 3 --slice-policy newest-run")

# one target per block; a fixed reference (--target-lags 3) keeps leads comparable
for T in "yield:--target yield" \
         "zscore:--target zscore --z-window 5 --target-lags 3" \
         "delta:--target delta --pct --target-lags 3"; do
  name=${T%%:*}; args=${T#*:}
  for SRC in $SOURCES; do
    $PY python/run_yield_experiment.py ${=COMMON} ${=SRC} ${=args} \
      --tag br-xgb-$name --go
  done
  $PY python/report_yield_experiment.py --dir $D/experiments \
    --runs "*br-xgb-${name}_*" --reference "era5_*" \
    --polygons $D/cy-bench/polygons --metrics skill,rmse_pct --year 2021,2022 \
    --out $D/experiments/br_xgb_${name}.html
done
```

Leave out `--go` for a plan-only check. Each plan prints the target, its
reference years and the null.

## Model choice

`--model xgb` uses XGBoost, configured against overfitting
(`XGB_REG_PARAMS` and `XGB_CLF_PARAMS` in `models.py`).

The underlying problem: rows within a year share their weather, so about
90k rows hold only about as many independent weather samples as there are
years (about 20). A flexible booster will happily fit those year
idiosyncrasies.

| setting | value | why |
|---|---|---|
| `max_depth` | 3 (was 6) | interactions of at most three features |
| `min_child_weight` | 50 (classifier 10) | each leaf rests on ≥ 50 rows, so no leaf is carved around a handful of municipalities |
| `subsample`, `colsample_bytree` | 0.7 (was 0.8) | more randomness per tree, less memorisation |
| `reg_lambda` | 10 (was 1) | shrinks leaf weights; unit-free, so it acts the same on kg/ha, % and z targets |
| `learning_rate` | 0.03 (was 0.05) | smaller steps, so early stopping has a finer choice |
| `n_estimators` | ceiling 1000 | the actual count is chosen by early stopping |

**Early stopping on held-out years** (`YieldModel._fit_xgb_early_stopped`):
- **What it does:** inside each fold, 20% of the *training* years (at
  least 2) are held out. Trees are added until the held-out error hasn't
  improved for 50 rounds, then the model is refit on all training years with
  that many trees.
- **Why years, not random rows:** a random-row split would share years with
  training and never stop.
- **No leakage:** the outer test year is never involved.
- **When it is skipped:** with fewer than 6 training years it falls back to
  the 1000-tree ceiling.
- **Where it shows:** the number of trees kept is recorded per fold as
  `n_trees` (fold lines print `trees=`; the summary shows the median). A
  low count means the held-out years found little to learn beyond the null.
  That is informative about DCPP at long leads.

**Changes for the direction classifier:**
- **Label encoding fixed:** XGBoost needs integer class labels, and the
  direction task passes "down" / "flat" / "up". Previously `--target
  direction --model xgb` crashed; labels are now encoded and decoded inside
  `YieldModel`.
- **Same early stopping:** it applies to the classifier as well.

**Check for overfitting:** compare `sd_ratio_anom` with `r_anom` in the
report. A calibrated model's ratio sits near `r_anom`; well above it means
over-amplification.

**Changing one thing at a time:** the earlier Brazil runs used `rf`. Before
comparing an xgb run on a new target with them, rerun the existing anomaly
setup with xgb (`--target anomaly --model xgb`), so model choice and target
choice are not confounded.

## Where it lives in the code

- `run_yield_experiment.py`:
  - `RefTarget` defines R and S for delta / zscore.
  - `fit_fold` does the transform, adds the `ref_clim` feature and maps the
    prediction back to kg/ha.
  - `score_target` computes the `t_*` metrics.
  - `DirectionLabeller` handles direction.
- `yield_features.reference_yields` attaches the `yref{k}` columns from the
  full yield table, so a year without climate features still serves as a
  reference.
- `report_yield_experiment.py` labels the runs ("Δ vs t-3 %", "z5 to t-3")
  and adds the `t_*` columns to its tables.
