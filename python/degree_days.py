"""Single-sine degree days, shared by the two feature pipelines.

Both `era5_features.py` (per-pixel ERA5 features) and `data_loader.py`
(`--feature_mode agro`) accumulate growing and extreme degree days. They used to
carry separate implementations — one integrating the diurnal cycle, one working
off the daily mean — which meant `gdd` and `edd` named different quantities
depending on which path produced them. One definition lives here now.

Why a sine and not the daily mean
---------------------------------
Degree days above a threshold are an integral over the day,

    DD(theta) = (1/24) * integral over the day of max(T(t) - theta, 0) dt

so they need an assumption about T(t). The naive
`max((tasmax + tasmin) / 2 - theta, 0)` assumes T is CONSTANT at the daily mean.
Because max(., 0) is nonlinear, that is not the same integral: the two agree
exactly when theta falls outside [tasmin, tasmax], and diverge on every day the
threshold crosses, which for a heat threshold is the entire population of
interest. A day running 20 -> 40 C spends hours above 30 C but has a daily mean
of exactly 30, so the naive form scores its extreme heat as zero.

Taking T(t) to be a sine between tasmin and tasmax instead gives the standard
construction of the crop-impacts literature (Baskerville & Emin 1969; Allen
1976; Schlenker & Roberts 2009).
"""

from __future__ import annotations

import numpy as np


def degree_days_sine(tmin, tmax, threshold):
    """Degree days above `threshold` for one day, single-sine method.

    Integrates max(T - threshold, 0) over the day, taking the diurnal cycle to
    be a sine wave running between tmin and tmax. Baskerville & Emin's closed
    form, verified here against numerical integration of the same sine.

    Accepts scalars or arrays; returns a numpy array of the same shape.
    """
    tmin = np.asarray(tmin, dtype="float64")
    tmax = np.asarray(tmax, dtype="float64")
    mid = (tmax + tmin) / 2.0
    amp = (tmax - tmin) / 2.0

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip((threshold - mid) / np.where(amp > 0, amp, np.nan), -1.0, 1.0)
        theta = np.arcsin(ratio)
        # Area of the sine above the threshold, for a threshold strictly inside
        # the day's range.
        partial = ((mid - threshold) * (np.pi / 2.0 - theta) + amp * np.cos(theta)) / np.pi

    out = np.where(
        threshold <= tmin, mid - threshold, np.where(threshold >= tmax, 0.0, partial)
    )
    # A day with no diurnal range degenerates to a point, not a sine.
    return np.where(amp <= 0, np.maximum(mid - threshold, 0.0), out)


def capped_degree_days(tmin, tmax, base, cap):
    """Degree days accumulated only between `base` and `cap`.

    DD(base) - DD(cap) is an algebraic identity for the degree days of
    clamp(T, base, cap) - base: heat above `cap` stops adding to the growth term
    instead of being credited as if it were beneficial. It is counted separately
    as extreme degree days -- the piecewise-linear yield response of Schlenker &
    Roberts (2009), where maize gains up to ~29-30 C and loses above it.
    """
    return degree_days_sine(tmin, tmax, base) - degree_days_sine(tmin, tmax, cap)
