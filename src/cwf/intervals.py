"""Empirical prediction intervals from rolling-origin residuals.

Method
------
For each horizon step ``h`` we take the rolling-origin residuals
``e_h = y_true - y_pred`` of the *selected* model over the validation folds and
use their empirical quantiles.  The band is then symmetric in quantile terms:
``[y_hat + q_low(h), y_hat + q_high(h)]`` with ``q`` fixed per horizon step.

This is the simplest honest option at this data size:

* It needs no distributional assumption beyond exchangeability of residuals.
* Fixed-per-horizon quantiles capture the widening of uncertainty with the
  horizon (a `h`-step-ahead forecast is worse than a 1-step-ahead one), which a
  single global quantile would hide.
* With only ~5 origins it is noisy.  We therefore also report *leave-one-origin-out*
  coverage on the validation folds so the report can show the bands are honest,
  not merely wide.

A conformal-style variant (pool residuals, use the ``ceil((n+1)(1-alpha))/n``
empirical quantile) is available via ``method="conformal"``; it is nearly
identical here because the residual pool is small, and the run reports both.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)


def residuals_by_horizon(y_true, y_pred) -> np.ndarray:
    """``(n_origins, horizon)`` residuals."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a - b


def quantiles_by_horizon(residuals: np.ndarray, levels=DEFAULT_LEVELS, method: str = "empirical") -> dict:
    """Residual quantile per horizon step, keyed by level."""
    res = np.asarray(residuals, dtype=float)
    if res.ndim == 1:
        res = res.reshape(-1, 1)
    out = {}
    n = res.shape[0]
    for level in levels:
        if method == "conformal":
            # conformal-style: use the ceil((n+1)*level)-th smallest residual
            k = int(np.ceil((n + 1) * level))
            k = min(max(k, 1), n)
            out[level] = np.sort(res, axis=0)[k - 1]
        elif method == "empirical":
            out[level] = np.quantile(res, level, axis=0)
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def band(forecast: np.ndarray, quantiles: dict, level: float, method: str = "symmetric") -> tuple[np.ndarray, np.ndarray]:
    """Build ``(lower, upper)`` for a central interval of the given nominal level.

    ``symmetric`` uses the ``level/2`` and ``1 - level/2`` residual quantiles;
    this is the default because the residuals here are close to centred.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    alpha = 1.0 - float(level)
    keys = sorted(quantiles)
    lo_q = min(keys, key=lambda k: abs(k - alpha / 2))
    hi_q = min(keys, key=lambda k: abs(k - (1 - alpha / 2)))
    # Quantiles are per horizon step; tile them across origins when the caller
    # passes a flattened (n_origins * horizon) forecast vector.
    lo = np.resize(np.asarray(quantiles[lo_q], dtype=float), f.size)
    hi = np.resize(np.asarray(quantiles[hi_q], dtype=float), f.size)
    return f + lo, f + hi


def leave_one_origin_out_coverage(y_true, y_pred, level: float, levels=DEFAULT_LEVELS) -> float:
    """Coverage of bands whose quantiles exclude the fold being scored."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.ndim == 1:
        a, b = a.reshape(1, -1), b.reshape(1, -1)
    n = a.shape[0]
    if n < 3:
        return float("nan")
    hits = []
    for i in range(n):
        mask = np.arange(n) != i
        q = quantiles_by_horizon(residuals_by_horizon(a[mask], b[mask]), levels=levels)
        lo, hi = band(b[i], q, level)
        hits.extend(((a[i] >= lo) & (a[i] <= hi)).tolist())
    return float(np.mean(hits))


def widen(lower: np.ndarray, upper: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale the band around its midpoint (used by the policy knob)."""
    mid = 0.5 * (np.asarray(lower, float) + np.asarray(upper, float))
    half = 0.5 * (np.asarray(upper, float) - np.asarray(lower, float)) * float(factor)
    return mid - half, mid + half
