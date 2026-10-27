"""Baseline forecasters.

Every baseline implements the same tiny interface so the rolling-origin
harness treats them identically to the learned models::

    forecast(context: np.ndarray, horizon: int) -> np.ndarray

``context`` is the history available at the forecast origin (oldest first) and
contains no future information by construction, so a baseline cannot leak by
accident.  Baselines are deliberately strong: on a 9-hour, low-variance series a
persistence forecast is hard to beat, and saying so is part of the story.
"""

from __future__ import annotations

import numpy as np


def persistence(context: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value.  Exact on a constant series."""
    ctx = np.asarray(context, dtype=float).ravel()
    if ctx.size == 0:
        raise ValueError("persistence needs at least one observation")
    return np.full(horizon, ctx[-1], dtype=float)


def moving_average(window: int):
    """Factory: repeat the mean of the last ``window`` observations."""

    def _forecast(context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=float).ravel()
        if ctx.size == 0:
            raise ValueError("moving_average needs at least one observation")
        w = min(int(window), ctx.size)
        return np.full(horizon, float(np.mean(ctx[-w:])), dtype=float)

    _forecast.__name__ = f"ma{int(window)}"
    return _forecast


def drift(context: np.ndarray, horizon: int) -> np.ndarray:
    """Extend the straight line from the first to the last context point.

    With only a handful of context points a drift term is mostly noise, which
    is exactly why it is in the candidate set: it shows whether extrapolating
    the local slope helps or hurts.
    """
    ctx = np.asarray(context, dtype=float).ravel()
    if ctx.size < 2:
        return persistence(ctx, horizon)
    slope = (ctx[-1] - ctx[0]) / (ctx.size - 1)
    return ctx[-1] + slope * np.arange(1, horizon + 1, dtype=float)


def seasonal_naive(period: int):
    """Repeat the value ``period`` steps earlier.  Diagnostic only (see D011)."""

    def _forecast(context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=float).ravel()
        p = int(period)
        if p <= 0 or ctx.size < p:
            return persistence(ctx, horizon)
        tail = ctx[-p:]
        reps = int(np.ceil(horizon / p))
        return np.tile(tail, reps)[:horizon]

    _forecast.__name__ = f"seasonal_naive{int(period)}"
    return _forecast


#: The frozen candidate list from section 6 of docs/protocol.md.
BASELINES = {
    "persistence": persistence,
    "ma6": moving_average(6),
    "ma12": moving_average(12),
    "ma24": moving_average(24),
    "drift": drift,
}

#: Kept out of selection; evaluated to document why it is not selected.
DIAGNOSTIC_BASELINES = {
    "seasonal_naive12": seasonal_naive(12),
}
