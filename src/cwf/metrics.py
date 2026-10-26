"""Forecast accuracy metrics.

Conventions
-----------
* All metrics take ``y_true`` and ``y_pred`` as equally shaped 2-D arrays
  ``(n_origins, horizon)`` unless stated otherwise, so a metric can be computed
  per origin and then averaged.
* ``mase`` needs a training series to build its scale; the scale is the
  in-sample one-step naive MAE of **training data only**.  That is what makes
  MASE honest here: a constant series has ``scale == 0`` and MASE becomes
  undefined, which we surface rather than disguise (documented in the report).
* Percentage errors (MAPE, sMAPE) are provided but are *not* used to select a
  model: the utilisation target is a ratio whose denominator can approach zero,
  which makes percentage errors explode.  ``mape`` refuses to divide by values
  below an epsilon and reports the number of excluded points.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def errors(y_true, y_pred) -> np.ndarray:
    """Elementwise ``y_true - y_pred`` (positive = under-forecast)."""
    a, b = _as_2d(y_true), _as_2d(y_pred)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a - b


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(errors(y_true, y_pred))))


def rmse(y_true, y_pred) -> float:
    e = errors(y_true, y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def bias(y_true, y_pred) -> float:
    return float(np.mean(errors(y_true, y_pred)))


def naive_scale(train_values) -> float:
    """In-sample one-step naive MAE: ``mean(|y_t - y_{t-1}|)`` on training data."""
    v = np.asarray(train_values, dtype=float).ravel()
    if v.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(v))))


def mase(y_true, y_pred, scale: float) -> float:
    """Mean absolute scaled error; ``nan`` when the scale is degenerate."""
    if not np.isfinite(scale) or scale < EPS:
        return float("nan")
    return mae(y_true, y_pred) / scale


def mape(y_true, y_pred, eps: float = 1e-6, min_abs_denominator: float | None = None):
    """Mean absolute percentage error, plus the number of excluded points.

    ``min_abs_denominator`` defaults to ``eps``; points whose ``|y_true|`` is
    below it are excluded and counted, never silently divided.
    """
    a, b = _as_2d(y_true), _as_2d(y_pred)
    threshold = eps if min_abs_denominator is None else min_abs_denominator
    mask = np.abs(a) > threshold
    excluded = int((~mask).sum())
    if not mask.any():
        return float("nan"), excluded
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0), excluded


def smape(y_true, y_pred) -> float:
    a, b = _as_2d(y_true), _as_2d(y_pred)
    denom = np.abs(a) + np.abs(b)
    mask = denom > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(a[mask] - b[mask]) / denom[mask]) * 100.0)


def skill_score(mse_model: float, mse_reference: float) -> float:
    """``1 - MSE_model / MSE_reference``; positive means the model wins."""
    if not np.isfinite(mse_reference) or mse_reference <= EPS:
        return float("nan")
    return float(1.0 - mse_model / mse_reference)


def per_horizon(errors_2d) -> np.ndarray:
    """RMSE for each of the H horizons, given ``(n_origins, H)`` errors."""
    e = _as_2d(errors_2d)
    return np.sqrt(np.mean(e ** 2, axis=0))


def coverage(y_true, lower, upper) -> float:
    """Fraction of observations inside the interval."""
    a, lo, hi = _as_2d(y_true), _as_2d(lower), _as_2d(upper)
    return float(np.mean((a >= lo) & (a <= hi)))


def pinball_loss(y_true, q_pred, q: float) -> float:
    """Quantile (pinball) loss for a single quantile level ``q``."""
    a, p = _as_2d(y_true), _as_2d(q_pred)
    d = a - p
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def interval_width(lower, upper) -> float:
    lo, hi = _as_2d(lower), _as_2d(upper)
    return float(np.mean(hi - lo))


def summarize(y_true, y_pred, scale: float | None = None) -> dict:
    """A dict of all headline metrics for one candidate on one target."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if scale is not None:
        out["mase"] = mase(y_true, y_pred, scale)
        out["scale_naive_mae"] = float(scale)
    m, excluded = mape(y_true, y_pred)
    out["mape_pct"] = m
    out["mape_excluded_points"] = excluded
    return out
