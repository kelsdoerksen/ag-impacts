"""
PatchTST Crop Yield Predictor
=================================
Predicts a single scalar crop yield (e.g. soy kg/ha) from a county-level
daily climate time series using a PatchTST Transformer regression model.

Problem framing
---------------
Each sample represents one (county, hindcast_year) pair:

  Input  (past_values):  shape (T=366, C=4)
         T = days in the Nov-1 → Oct-31 hindcast season
         C = climate channels: [tasmax_C, tasmin_C, hursmin, pr_mm]

  Target (target_values): shape (1,)
         Scalar crop yield for hindcast_year + 1  (the "yield year")

PatchTST segments the T-length series into overlapping patches of length P,
encodes each patch independently per channel with a Transformer encoder, then
pools the patch representations and passes them through a linear head to
predict the scalar yield.

Reference: https://huggingface.co/blog/patchtst

Data files expected under --data_dir
--------------------------------------
  {data_dir}/{climate_model}/climate_{climate_model}_s{year}-{ens_member}.parquet
      Daily climate (one file per hindcast year, 366 rows per county)
  {data_dir}/crop_{crop}_CONUS.parquet
      Annual yield per county (one row per (county, hindcast_start))

Usage modes
-----------
Single year (county split for validation — quick local test):
  python tst_implementation.py \\
      --data_dir /path/to/processed_data \\
      --hindcast_year 1983 \\
      --crop soy

Multi-year with temporal hold-out (leave-one-year-out):
  python tst_implementation.py \\
      --data_dir /path/to/processed_data \\
      --study_years 1980 2009 \\
      --test_year 2007 \\
      --crop corn

See --help for all hyperparameters.

Environment: tested with transformers==4.57, Python 3.13.
Parquet files are read with fastparquet (pyarrow has a format incompatibility
with the files produced by parquet-cpp-arrow 23.x).
"""

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    PatchTSTConfig,
    PatchTSTForRegression,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Climate channels used as input features (order determines channel index).
CLIMATE_CHANNELS = ["tasmax_C", "tasmin_C", "hursmin", "pr_mm"]

# Optional CY-Bench phenology channels, appended AFTER the climate channels when
# requested via --use_phenology (off by default). Each token maps to a parquet
# file stem and its value column(s); files live in
#   {data_dir}/phenology/phenology_{stem}_{suffix}.parquet
# Phenology has shorter/irregular coverage than climate (NDVI/FPAR 2001+, soil
# moisture 2003+) and a native non-daily cadence; build_sequence_array forward/
# back-fills each channel onto the daily grid and drops county-years that have no
# observations at all (so enabling phenology restricts the sample set).
PHENOLOGY_CHANNELS = {
    "ndvi":          {"stem": "ndvi",         "cols": ["ndvi"]},
    "fpar":          {"stem": "fpar",         "cols": ["fpar"]},
    "soil_moisture": {"stem": "soilmoisture", "cols": ["ssm", "rsm"]},
}


def phenology_channel_cols(use_phenology) -> list[str]:
    """Flatten requested phenology var tokens into ordered channel column names."""
    cols: list[str] = []
    for var in (use_phenology or []):
        if var not in PHENOLOGY_CHANNELS:
            raise ValueError(
                f"Unknown phenology variable {var!r}. "
                f"Choices: {list(PHENOLOGY_CHANNELS)}"
            )
        cols += PHENOLOGY_CHANNELS[var]["cols"]
    return cols


def parse_phenology_arg(s: str) -> list[str]:
    """Parse a comma-separated --use_phenology value (e.g. 'ndvi,fpar') to a list."""
    import argparse
    vars_ = [v.strip() for v in s.split(",") if v.strip()]
    for v in vars_:
        if v not in PHENOLOGY_CHANNELS:
            raise argparse.ArgumentTypeError(
                f"Unknown phenology variable {v!r}. Choices: {list(PHENOLOGY_CHANNELS)}"
            )
    return vars_

# Expected number of days per county-year (Nov 1 → Oct 31 inclusive).
# 366 covers leap years; non-leap seasons (365 days) are padded by bfill.
SEQ_LEN = 366

# Default PatchTST architecture and training settings. Used as the baseline by
# both the standalone CLI and train_predict_patchtst(); callers override
# individual keys as needed (e.g. the LOO experiment passes its own CLI values).
DEFAULT_HPARAMS = dict(
    patch_length=16,
    patch_stride=8,
    d_model=128,
    num_layers=3,
    num_heads=8,
    ffn_dim=256,
    attention_dropout=0.1,
    head_dropout=0.1,
)
DEFAULT_TRAIN_CFG = dict(
    epochs=50,
    batch_size=64,
    lr=1e-4,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_climate_file(
    data_dir: str, climate_model: str, year: int, ens_member: str
) -> str:
    """Locate the climate parquet for one hindcast year, tolerating AOI naming.

    CONUS files are named climate_{climate_model}_s{year}-{ens_member}.parquet;
    other AOIs (e.g. Germany) use climate_{aoi}_{climate_model}_s{year}.parquet
    with no ensemble suffix. Try the exact CONUS name first, then fall back to
    any parquet in the climate_model dir whose name carries the year token.
    """
    cdir  = os.path.join(data_dir, climate_model)
    exact = os.path.join(cdir, f"climate_{climate_model}_s{year}-{ens_member}.parquet")
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(cdir, f"*s{year}*.parquet")))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No climate parquet for year {year} in {cdir!r} "
        f"(tried exact name {os.path.basename(exact)!r} then *s{year}*.parquet). "
        "Check --data_dir, --climate_model, and year range."
    )


def _merge_phenology(
    merged: pd.DataFrame,
    data_dir: str,
    use_phenology: list[str] | None,
    suffix: str,
    phenology_dir: str | None = None,
) -> pd.DataFrame:
    """Left-merge native-cadence CY-Bench phenology onto the daily climate frame.

    Joins by (adm2_id, date), adding one column per phenology channel. Days with
    no observation become NaN; build_sequence_array fills each channel onto the
    daily grid (ffill/bfill) and skips county-years with no data at all. No-op
    when use_phenology is falsy.
    """
    if not use_phenology:
        return merged
    phenology_dir = phenology_dir or os.path.join(data_dir, "phenology")
    for var in use_phenology:
        cfg  = PHENOLOGY_CHANNELS[var]
        path = os.path.join(phenology_dir, f"phenology_{cfg['stem']}_{suffix}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Phenology parquet not found: {path!r}. "
                "Generate it with phenology-sample-gen.R or check --phenology_suffix."
            )
        ph = pd.read_parquet(path, columns=["adm2_id", "date"] + cfg["cols"])
        ph["date"] = pd.to_datetime(ph["date"])
        merged = merged.merge(ph, on=["adm2_id", "date"], how="left")
        print(
            f"[data] Phenology '{var}': merged {cfg['cols']}  "
            f"({ph['adm2_id'].nunique()} counties, {len(ph):,} obs)"
        )
    return merged


def load_climate_crop_data(
    data_dir: str,
    hindcast_years: list[int],
    crop: str,
    climate_model: str = "EC-Earth3",
    ens_member: str = "r1i4p1f1",
    aoi: str = "CONUS",
    use_phenology: list[str] | None = None,
    phenology_suffix: str | None = None,
) -> pd.DataFrame:
    """Load one or more hindcast years of climate data and merge with crop yield.

    Climate parquet schema (per file):
        adm2_id, adm2, adm1, hindcast_start, date, doy,
        year_month, tasmax_C, tasmin_C, hursmin, pr_mm

    One file per hindcast year covers Nov 1 of that year through Oct 31 of
    (year+1), with one row per (county, day).

    Parameters
    ----------
    data_dir : str
        Root directory of processed data.
    hindcast_years : list[int]
        One or more forecast start years (e.g. [1983] or list(range(1980, 2010))).
        Target yield for each year = hindcast_year + 1.
    crop : str
        Crop name matching crop_{crop}_CONUS.parquet (e.g. 'soy', 'corn').
    climate_model : str
        Climate model name used in the subdirectory and file prefix.
    ens_member : str
        Ensemble member tag in the filename (e.g. 'r1i4p1f1').

    Returns
    -------
    pd.DataFrame
        Long-format daily data with columns:
            adm2_id, hindcast_start, date, doy, tasmax_C, tasmin_C,
            hursmin, pr_mm, yield
        One row per (county, day, year).  Counties without yield data are excluded.

    Raises
    ------
    FileNotFoundError
        If any climate parquet or the crop parquet cannot be located.
    """
    # ── climate ──────────────────────────────────────────────────────────────
    climate_frames = []
    for year in sorted(hindcast_years):
        path = _find_climate_file(data_dir, climate_model, year, ens_member)
        climate_frames.append(pd.read_parquet(path, engine="fastparquet"))

    print(f"[data] Loaded {len(climate_frames)} climate parquet(s) for years {sorted(hindcast_years)}")
    climate_df = pd.concat(climate_frames, ignore_index=True)
    climate_df["date"] = pd.to_datetime(climate_df["date"])
    print(
        f"         {len(climate_df):,} total rows  |  "
        f"{climate_df['adm2_id'].nunique()} unique counties"
    )

    # ── crop ─────────────────────────────────────────────────────────────────
    crop_path = os.path.join(data_dir, f"crop_{crop}_{aoi}.parquet")
    if not os.path.exists(crop_path):
        raise FileNotFoundError(
            f"Crop parquet not found: {crop_path!r}. Check --crop, --aoi, and --data_dir."
        )

    print(f"[data] Crop    : {crop_path}")
    crop_df = pd.read_parquet(crop_path, engine="fastparquet")
    crop_subset = (
        crop_df[crop_df["hindcast_start"].isin(hindcast_years)][
            ["adm2_id", "hindcast_start", "yield"]
        ]
        .dropna(subset=["yield"])
        .reset_index(drop=True)
    )
    if crop_subset.empty:
        raise ValueError(
            f"No yield data found for years {sorted(hindcast_years)} in {crop_path!r}."
        )
    print(f"         {len(crop_subset)} county-year yield pairs across {len(hindcast_years)} year(s)")

    # ── merge ─────────────────────────────────────────────────────────────────
    # Join on both adm2_id and hindcast_start so each county-year is matched correctly.
    merged = climate_df.merge(crop_subset, on=["adm2_id", "hindcast_start"], how="inner")
    merged = merged.reset_index(drop=True)
    print(
        f"[data] Merged  : {len(merged):,} rows  |  "
        f"{merged.groupby(['adm2_id','hindcast_start']).ngroups} county-year pairs"
    )

    # ── optional phenology channels (off by default) ───────────────────────────
    merged = _merge_phenology(
        merged, data_dir, use_phenology,
        suffix=(phenology_suffix or aoi),
    )
    return merged


def build_sequence_array(
    daily_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Convert long-format daily data into stacked (N, T, C) arrays.

    For each (county, hindcast_year) pair, constructs a (seq_len, n_channels)
    matrix of daily climate observations ordered by day offset from Nov 1.
    Gaps (missing days) are forward-filled then backward-filled within each
    county-year.  Pairs with unfillable gaps or missing yield are skipped.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Long-format daily data (output of load_climate_crop_data).
    seq_len : int
        Expected sequence length.  Default 366 to accommodate leap years.

    Returns
    -------
    sequences : np.ndarray, shape (N, seq_len, n_channels), dtype float32
        Climate time series, one row per (county, year).
    targets : np.ndarray, shape (N, 1), dtype float32
        Crop yield value for each (county, year).
    sample_meta : pd.DataFrame, shape (N, 4)
        Per-sample metadata with columns adm2_id, adm2, adm1, hindcast_start.
        Row order matches sequences and targets exactly.
        Used for train/test splitting by year and for writing output CSVs.
    """
    channels = channels or CLIMATE_CHANNELS

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Day offset from Nov 1 of the hindcast start year (1-indexed so d1 = Nov 1)
    season_start = pd.to_datetime(df["hindcast_start"].astype(str) + "-11-01")
    df["day_offset"] = (df["date"] - season_start).dt.days + 1

    # Clip to the expected window; any date outside [1, seq_len] is noise
    df = df[(df["day_offset"] >= 1) & (df["day_offset"] <= seq_len)]

    # Per-(county, year) constants — extract before the reshape loop
    meta_map = (
        df.groupby(["adm2_id", "hindcast_start"])[["adm2", "adm1", "yield"]]
        .first()
    )

    sequences, targets, meta_rows = [], [], []
    n_skipped = 0

    n_pairs = len(meta_map)
    print(f"[data] Building sequences for {n_pairs} county-year pairs…")

    for (adm2_id, hindcast_start), group in df.groupby(["adm2_id", "hindcast_start"]):
        # Collapse any duplicate days to one row per day_offset, then align to a
        # complete [1, seq_len] index so all days are present. Duplicates arise
        # when a crop has multiple sub-types sharing a (county, year) — e.g.
        # winter and spring wheat — which duplicates the merged climate rows.
        # Averaging matches the timeseries featurizer's pivot aggfunc="mean".
        group = (
            group.groupby("day_offset")[channels]
                 .mean()
                 .reindex(range(1, seq_len + 1))
        )

        # Fill calendar gaps within each channel independently (no cross-channel
        # bleed). A phenology channel with zero observations in the window stays
        # all-NaN and the sample is skipped below.
        for ch in channels:
            group[ch] = group[ch].ffill().bfill()

        seq = group[channels].values.astype(np.float32)  # (T, C)
        row = meta_map.loc[(adm2_id, hindcast_start)]
        county_yield = float(row["yield"])

        if np.isnan(seq).any() or np.isnan(county_yield):
            n_skipped += 1
            continue

        sequences.append(seq)
        targets.append([county_yield])
        meta_rows.append({
            "adm2_id":        adm2_id,
            "adm2":           row["adm2"],
            "adm1":           row["adm1"],
            "hindcast_start": hindcast_start,
        })

    if n_skipped:
        print(f"[data] Skipped {n_skipped} county-year pairs (unfillable gaps or missing yield).")

    n = len(sequences)
    print(f"[data] {n} sequences ready — shape per sample ({seq_len}, {len(channels)})  "
          f"channels: {channels}")
    return (
        np.array(sequences, dtype=np.float32),            # (N, T, C)
        np.array(targets,   dtype=np.float32),            # (N, 1)
        pd.DataFrame(meta_rows).reset_index(drop=True),  # (N, 3)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class YieldSequenceDataset(Dataset):
    """PyTorch Dataset wrapping per-county climate sequences and yield targets.

    Each __getitem__ returns a dict matching the keyword arguments accepted by
    PatchTSTForRegression.forward():

        past_values   : FloatTensor (seq_len, n_channels)
                        Z-score normalised daily climate time series.
        target_values : FloatTensor (1,)
                        Raw (unnormalised) crop yield in original units.

    The model computes MSE loss internally when target_values is provided,
    which the HuggingFace Trainer uses for the backward pass.
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray, log_target: bool = False):
        """
        Parameters
        ----------
        sequences  : np.ndarray, shape (N, T, C) — normalised climate data
        targets    : np.ndarray, shape (N, 1)    — raw yield values (not normalised)
        log_target : bool — if True, model learns to predict log(yield)
        """
        self.log_target = log_target
        self.sequences  = torch.tensor(sequences, dtype=torch.float32)
        t = torch.tensor(targets, dtype=torch.float32)
        self.targets = torch.log(t) if log_target else t

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return {
            "past_values":   self.sequences[idx],  # (T, C)
            "target_values": self.targets[idx],    # (1,)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def fit_normalizer(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel Z-score statistics from training sequences only.

    Statistics are computed over all counties and all timesteps so each
    climate channel is independently centred and scaled.  Yield targets are
    kept in their original units — normalising the target is not appropriate
    here because we want the model to predict actual yield values.

    Always fit on training data only; apply to both train and validation to
    prevent data leakage from the validation set.

    Parameters
    ----------
    sequences : np.ndarray, shape (N, T, C) — training climate sequences

    Returns
    -------
    ch_mean : ndarray (C,) — per-channel mean
    ch_std  : ndarray (C,) — per-channel std  (floored at 1.0 to avoid ÷0)
    """
    flat = sequences.reshape(-1, sequences.shape[-1])  # (N*T, C)
    ch_mean = flat.mean(axis=0)
    ch_std  = np.where(flat.std(axis=0) == 0, 1.0, flat.std(axis=0))
    return ch_mean, ch_std


def apply_normalizer(
    sequences: np.ndarray,
    ch_mean: np.ndarray,
    ch_std: np.ndarray,
) -> np.ndarray:
    """Z-score normalise climate sequences using pre-fitted channel statistics."""
    return ((sequences - ch_mean) / ch_std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_patchtst_model(
    context_length: int = SEQ_LEN,
    n_channels: int = len(CLIMATE_CHANNELS),
    patch_length: int = 16,
    patch_stride: int = 8,
    d_model: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    ffn_dim: int = 256,
    attention_dropout: float = 0.1,
    head_dropout: float = 0.1,
) -> PatchTSTForRegression:
    """Construct a PatchTSTForRegression model for scalar yield prediction.

    Architecture overview
    ---------------------
    1. The input (T, C) series is divided into overlapping patches of length P
       with step S.  Number of patches = (T - P) // S + 2.
    2. Each patch is linearly projected to a d_model-dimensional embedding.
    3. A stack of Transformer encoder layers (channel-independent: each of the
       C channels is encoded with shared weights but separately) produces
       contextualised patch embeddings.
    4. Patch embeddings are mean-pooled across the sequence dimension.
    5. A linear regression head maps the pooled representation to a scalar.

    Parameters
    ----------
    context_length : int
        Input sequence length T (days).  Default 366.
    n_channels : int
        Number of climate channels C.  Default 4.
    patch_length : int
        Patch size in timesteps.  Default 16 (≈ 2 weeks of daily data).
    patch_stride : int
        Step between consecutive patches.  Default 8 (50 % overlap).
    d_model : int
        Transformer hidden dimension.  Default 128 (small for local testing).
    num_layers : int
        Number of Transformer encoder layers.  Default 3.
    num_heads : int
        Number of self-attention heads (d_model must be divisible by num_heads).
    ffn_dim : int
        Feed-forward network inner dimension.  Default 256.
    attention_dropout : float
        Dropout applied inside attention.  Default 0.1.
    head_dropout : float
        Dropout in the regression head.  Default 0.1.

    Returns
    -------
    PatchTSTForRegression
    """
    n_patches = (context_length - patch_length) // patch_stride + 2
    n_params_estimate = (
        # rough: patch proj + pos embed + layers + head
        patch_length * d_model           # patch projection (per channel)
        + n_patches * d_model            # positional embedding
        + num_layers * 4 * d_model ** 2  # attention + ffn (very rough)
        + d_model                        # regression head
    ) * n_channels
    print(
        f"[model] PatchTST config:"
        f"  T={context_length}  P={patch_length}  S={patch_stride}"
        f"  → {n_patches} patches/channel\n"
        f"         d_model={d_model}  layers={num_layers}  heads={num_heads}"
        f"  ffn={ffn_dim}  (rough param estimate: ~{n_params_estimate:,})"
    )

    config = PatchTSTConfig(
        # Input dimensions
        num_input_channels=n_channels,
        context_length=context_length,
        # Patching
        patch_length=patch_length,
        patch_stride=patch_stride,
        # Transformer encoder
        d_model=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        ffn_dim=ffn_dim,
        attention_dropout=attention_dropout,
        # Regression head
        num_targets=1,          # single scalar yield prediction
        head_dropout=head_dropout,
        # Architecture choices
        pooling_type="mean",    # average patch embeddings before head
        channel_attention=False,  # enable cross-channel attention
        scaling=None,           # we normalise inputs ourselves; skip internal scaling
        loss="mse",
    )
    return PatchTSTForRegression(config)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute RMSE and R² directly in original yield units.

    Parameters
    ----------
    preds   : ndarray (N,) — model predictions in original yield units
    targets : ndarray (N,) — true yield values in original yield units

    Returns
    -------
    dict with keys: rmse, r2, mean_yield, rmse_pct
    """
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    rmse   = float(np.sqrt(ss_res / len(targets)))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mean_y = float(targets.mean())

    return {
        "rmse":       rmse,
        "r2":         r2,
        "mean_yield": mean_y,
        "rmse_pct":   100 * rmse / mean_y if mean_y != 0 else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Train + predict (reusable)
# ──────────────────────────────────────────────────────────────────────────────

def train_predict_patchtst(
    train_seqs: np.ndarray,
    train_tgts: np.ndarray,
    val_seqs: np.ndarray,
    val_tgts: np.ndarray,
    *,
    run_dir: str,
    hparams: dict | None = None,
    train_cfg: dict | None = None,
    log_target: bool = False,
) -> np.ndarray:
    """Train a PatchTST model on one train/val split and return val predictions.

    Encapsulates normalisation, dataset construction, model building, training,
    and inference so the same code path serves both the standalone CLI and the
    leave-one-out experiment runner.

    Parameters
    ----------
    train_seqs, val_seqs : np.ndarray, shape (N, T, C) — raw (unnormalised) climate
        sequences. Channel statistics are fit on train_seqs only to avoid leakage.
    train_tgts, val_tgts : np.ndarray, shape (N, 1) — raw yield values.
    run_dir : str
        Directory for the HuggingFace Trainer's checkpoints and logs.
    hparams : dict, optional
        Architecture overrides merged over DEFAULT_HPARAMS (patch_length,
        patch_stride, d_model, num_layers, num_heads, ffn_dim, attention_dropout,
        head_dropout).
    train_cfg : dict, optional
        Training overrides merged over DEFAULT_TRAIN_CFG (epochs, batch_size, lr).
    log_target : bool
        If True the model learns log(yield); predictions are exp'd back so the
        returned array is always in original yield units.

    Returns
    -------
    np.ndarray, shape (N,)
        Validation predictions in original yield units.
    """
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    tc = {**DEFAULT_TRAIN_CFG, **(train_cfg or {})}

    # ── Normalise climate inputs (fit on training data only to avoid leakage) ──
    ch_mean, ch_std = fit_normalizer(train_seqs)
    print(
        f"[norm]  Channel means : {ch_mean.round(2).tolist()}\n"
        f"[norm]  Channel stds  : {ch_std.round(2).tolist()}"
    )
    train_seqs_n = apply_normalizer(train_seqs, ch_mean, ch_std)
    val_seqs_n   = apply_normalizer(val_seqs,   ch_mean, ch_std)

    train_dataset = YieldSequenceDataset(train_seqs_n, train_tgts, log_target=log_target)
    val_dataset   = YieldSequenceDataset(val_seqs_n,   val_tgts,   log_target=log_target)

    # ── Build model ──
    # Derive channel count from the data so optional phenology channels flow
    # through automatically (climate-only = 4, + phenology = more).
    model = build_patchtst_model(
        context_length=SEQ_LEN,
        n_channels=train_seqs.shape[-1],
        patch_length=hp["patch_length"],
        patch_stride=hp["patch_stride"],
        d_model=hp["d_model"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        ffn_dim=hp["ffn_dim"],
        attention_dropout=hp["attention_dropout"],
        head_dropout=hp["head_dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Actual trainable parameters: {n_params:,}\n")

    # ── Configure and run training ──
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        learning_rate=tc["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        label_names=["target_values"],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(f"[train] Starting training ({tc['epochs']} epochs)…\n")
    trainer.train()

    print("\n[eval]  Running final inference on validation set…")
    pred_output = trainer.predict(val_dataset, metric_key_prefix="val")
    preds = pred_output.predictions.squeeze()  # (N,)
    if log_target:
        preds = np.exp(preds)
    return np.asarray(preds, dtype=float).reshape(-1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PatchTST for crop yield regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── data ──
    p.add_argument("--data_dir",      type=str, required=True,
                   help="Root directory containing processed parquet files.")
    p.add_argument("--crop",          type=str, default="soy",
                   help="Crop name matching crop_<name>_CONUS.parquet.")
    p.add_argument("--climate_model", type=str, default="EC-Earth3")
    p.add_argument("--ens_member",    type=str, default="r1i4p1f1")
    p.add_argument("--aoi",           type=str, default="CONUS",
                   help="Area-of-interest tag in the crop filename "
                        "(crop_<crop>_<aoi>.parquet). E.g. 'CONUS' or 'germany'.")
    p.add_argument("--use_phenology", type=parse_phenology_arg, default=None,
                   metavar="VARS",
                   help="Comma-separated CY-Bench phenology channels to append to "
                        "the climate sequence (off by default). "
                        f"Choices: {','.join(PHENOLOGY_CHANNELS)}. E.g. "
                        "--use_phenology ndvi,fpar. Restricts samples to county-years "
                        "with that data (NDVI/FPAR 2001+, soil moisture 2003+).")
    p.add_argument("--phenology_suffix", type=str, default=None,
                   help="AOI suffix in phenology filenames "
                        "(phenology_<var>_<suffix>.parquet). Defaults to --aoi.")

    # ── year selection (two mutually exclusive modes) ──
    year_group = p.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        "--hindcast_year", type=int, metavar="YEAR",
        help="Single hindcast year (quick test; val = random 20%% county split).",
    )
    year_group.add_argument(
        "--study_years", type=int, nargs=2, metavar=("START", "END"),
        help=(
            "Inclusive start and end of the study period (e.g. --study_years 1980 2009). "
            "Requires --test_year. Train = all years except test_year; "
            "test = test_year counties."
        ),
    )
    p.add_argument(
        "--test_year", type=int, default=None,
        help="Hindcast year to hold out as the test set. Required with --study_years.",
    )
    p.add_argument("--val_frac",      type=float, default=0.2,
                   help="Val fraction for county split (single --hindcast_year mode only).")
    p.add_argument("--target",        type=str,   default="yield",
                   choices=["yield", "log_yield"],
                   help="Target variable: 'yield' for raw values, 'log_yield' to predict log(yield)."
                        " Metrics and outputs are always reported in original yield units.")
    # ── model ──
    p.add_argument("--patch_length",  type=int,   default=16,
                   help="Patch size in days (e.g. 16 ≈ 2 weeks).")
    p.add_argument("--patch_stride",  type=int,   default=8,
                   help="Step between consecutive patches (50%% overlap at default).")
    p.add_argument("--d_model",       type=int,   default=128,
                   help="Transformer hidden dimension.")
    p.add_argument("--num_layers",    type=int,   default=3,
                   help="Number of Transformer encoder layers.")
    p.add_argument("--num_heads",     type=int,   default=8,
                   help="Self-attention heads (d_model must be divisible by this).")
    p.add_argument("--ffn_dim",       type=int,   default=256,
                   help="Feed-forward network inner dimension.")
    p.add_argument("--attn_dropout",  type=float, default=0.1,
                   help="Attention dropout probability.")
    p.add_argument("--head_dropout",  type=float, default=0.1,
                   help="Regression head dropout probability.")
    # ── training ──
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="AdamW learning rate.")
    p.add_argument("--output_dir",    type=str,   default="./patchtst_output",
                   help="Directory for Trainer checkpoints and logs.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── resolve year lists and validate args ──────────────────────────────────
    if args.study_years is not None:
        if args.test_year is None:
            raise ValueError("--test_year is required when using --study_years.")
        study_start, study_end = args.study_years
        if not (study_start <= args.test_year <= study_end):
            raise ValueError(
                f"--test_year {args.test_year} must lie within "
                f"--study_years {study_start}–{study_end}."
            )
        all_years   = list(range(study_start, study_end + 1))
        train_years = [y for y in all_years if y != args.test_year]
        test_year   = args.test_year
        mode        = "temporal"
    else:
        all_years   = [args.hindcast_year]
        train_years = all_years
        test_year   = None
        mode        = "county_split"

    # ── create a unique run directory for outputs ─────────────────────────────
    run_id  = f"patchtst_{args.crop}_test{test_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PatchTST Yield Predictor")
    print(f"  Run ID : {run_id}")
    print(f"  Crop   : {args.crop}  |  mode: {mode}")
    print(f"  Inputs : climate{(' + phenology ' + ','.join(args.use_phenology)) if args.use_phenology else ' only'}")
    if mode == "temporal":
        print(f"  Train years: {train_years[0]}–{train_years[-1]} (excl. {test_year})")
        print(f"  Test year  : {test_year}  (yield year {test_year + 1})")
    else:
        print(f"  Hindcast year: {all_years[0]}  (yield year {all_years[0] + 1})")
    print(f"  Output : {run_dir}")
    print(f"{'='*60}\n")

    # ── 1. Load and merge climate + crop data ─────────────────────────────────
    daily_df = load_climate_crop_data(
        data_dir=args.data_dir,
        hindcast_years=all_years,
        crop=args.crop,
        climate_model=args.climate_model,
        ens_member=args.ens_member,
        aoi=args.aoi,
        use_phenology=args.use_phenology,
        phenology_suffix=args.phenology_suffix,
    )

    # ── 2. Build per-(county, year) sequence arrays ───────────────────────────
    # sample_meta: DataFrame(adm2_id, adm1, hindcast_start) — one row per sample
    channels = CLIMATE_CHANNELS + phenology_channel_cols(args.use_phenology)
    sequences, targets, sample_meta = build_sequence_array(daily_df, channels=channels)

    hindcast_years_arr = sample_meta["hindcast_start"].values

    # ── 3. Split into train and val/test sets ─────────────────────────────────
    if mode == "temporal":
        # Temporal hold-out: train on all years except test_year
        train_mask = hindcast_years_arr != test_year
        val_mask   = hindcast_years_arr == test_year
        print(
            f"\n[split] Train: {train_mask.sum()} county-year samples  |  "
            f"Test: {val_mask.sum()} county-year samples (hindcast_start={test_year})"
        )
    else:
        # Single year: random county split for a quick local sanity check
        idx = np.arange(len(sequences))
        train_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=42)
        train_mask = np.zeros(len(sequences), dtype=bool)
        val_mask   = np.zeros(len(sequences), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        print(
            f"\n[split] Train: {train_mask.sum()} counties  |  "
            f"Val: {val_mask.sum()} counties  ({args.val_frac:.0%} held out)"
        )

    train_seqs, train_tgts = sequences[train_mask], targets[train_mask]
    val_seqs,   val_tgts   = sequences[val_mask],   targets[val_mask]

    # ── 4–7. Normalise, build, train, and run inference ──────────────────────
    # Yield targets are kept in original units — no target normalisation.
    log_target = args.target == "log_yield"
    if log_target:
        print("[target] Predicting log(yield); outputs will be inverse-transformed to original units.")

    preds = train_predict_patchtst(
        train_seqs, train_tgts, val_seqs, val_tgts,
        run_dir=run_dir,
        hparams=dict(
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            attention_dropout=args.attn_dropout,
            head_dropout=args.head_dropout,
        ),
        train_cfg=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr),
        log_target=log_target,
    )
    targets = val_tgts.squeeze()  # always original yield units

    metrics = compute_regression_metrics(preds, targets)

    # ── 8. Save outputs compatible with plot_results.py ──────────────────────

    # adm2_errors.csv — one row per test county with predictions and error terms
    val_meta = sample_meta[val_mask].reset_index(drop=True)
    errors_df = val_meta.copy()
    errors_df["y_true"]    = val_tgts.squeeze()
    errors_df["y_pred"]    = preds
    errors_df["residual"]  = preds - errors_df["y_true"]
    errors_df["abs_error"] = errors_df["residual"].abs()
    errors_df["sq_error"]  = errors_df["residual"] ** 2
    errors_df.to_csv(os.path.join(run_dir, "adm2_errors.csv"), index=False)

    # adm1_rmse.csv — RMSE aggregated per state
    mean_yield = metrics["mean_yield"]
    adm1_rmse = (
        errors_df.groupby("adm1")["sq_error"]
        .mean()
        .pipe(np.sqrt)
        .rename("rmse")
        .reset_index()
        .assign(rmse_pct=lambda d: 100 * d["rmse"] / mean_yield)
        .sort_values("rmse", ascending=False)
    )
    adm1_rmse.to_csv(os.path.join(run_dir, "adm1_rmse.csv"), index=False)

    # run_meta.json — overall metrics and run parameters (read by plot_results.py)
    run_meta = {
        "run_id":             run_id,
        "crop":               args.crop,
        "model_type":         "patchtst",
        "test_year":          test_year if mode == "temporal" else all_years[0],
        "train_years":        train_years if mode == "temporal" else [],
        "rmse_overall":       metrics["rmse"],
        "r2_overall":         metrics["r2"],
        "mean_yield":         metrics["mean_yield"],
        "n_train":            int(train_mask.sum()),
        "n_test":             int(val_mask.sum()),
        "target":              args.target,
        "feature_importances": None,  # PatchTST has no feature importances
        # hyperparameters for reproducibility
        "patch_length":       args.patch_length,
        "patch_stride":       args.patch_stride,
        "d_model":            args.d_model,
        "num_layers":         args.num_layers,
        "num_heads":          args.num_heads,
        "ffn_dim":            args.ffn_dim,
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "lr":                 args.lr,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    year_label = (
        f"test year {test_year} → yield year {test_year + 1}"
        if mode == "temporal"
        else f"hindcast {all_years[0]} → yield year {all_years[0] + 1}"
    )
    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.crop.upper()}  {year_label}")
    print(f"  Val RMSE   : {metrics['rmse']:,.1f}")
    print(f"  Val R²     : {metrics['r2']:.3f}")
    print(f"  Mean yield : {metrics['mean_yield']:,.1f}  (RMSE%: {metrics['rmse_pct']:.1f}%)")
    print(f"\n  Outputs saved to: {run_dir}/")
    print(f"    adm2_errors.csv  adm1_rmse.csv  run_meta.json")
    print(f"\n  To plot results:")
    print(f"    python ../plot_results.py --save_dir {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
