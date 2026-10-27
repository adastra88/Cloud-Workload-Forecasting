"""Origin-safe feature construction for the direct multi-horizon models.

The causality contract (section 4 of docs/protocol.md) is enforced structurally: every
public function takes the **context array** (values at or before the origin) and
never the full series, so a feature cannot see the future even by mistake.  The
property test in ``tests/test_features.py`` perturbs values after the origin and
asserts the feature matrix is unchanged.

Only univariate, self-contained features are used.  No metadata columns are
features anywhere (see ``docs/feature_audit.md``): ``timestamp_vm_deleted`` and
the lifetime CPU statistics ``max_cpu`` / ``avg_cpu`` / ``p95_max_cpu`` are
banned outright because they summarise the whole life of a VM, holdout included.
"""

from __future__ import annotations

import numpy as np

#: Lag offsets (in observations of the series) exposed to the models.
LAGS = (1, 2, 3, 6, 12, 24)
#: Rolling-window lengths for mean / std / min / max features.
WINDOWS = (6, 12, 24)


def _tail(ctx: np.ndarray, k: int) -> np.ndarray:
    v = np.asarray(ctx, dtype=float).ravel()
    return v[-k:] if v.size >= k else v


def lag_features(ctx: np.ndarray) -> dict[str, float]:
    """Value ``k`` steps before the origin.  Uses the available history only."""
    v = np.asarray(ctx, dtype=float).ravel()
    out = {}
    for k in LAGS:
        out[f"lag_{k}"] = float(v[-k]) if v.size >= k else float("nan")
    return out


def rolling_features(ctx: np.ndarray) -> dict[str, float]:
    """Rolling statistics over the tail of the context (origin-safe by design)."""
    v = np.asarray(ctx, dtype=float).ravel()
    out = {}
    for w in WINDOWS:
        tail = _tail(v, w)
        if tail.size == 0:
            out[f"mean_{w}"] = out[f"std_{w}"] = float("nan")
            out[f"min_{w}"] = out[f"max_{w}"] = float("nan")
            continue
        out[f"mean_{w}"] = float(np.mean(tail))
        out[f"std_{w}"] = float(np.std(tail, ddof=0)) if tail.size > 1 else 0.0
        out[f"min_{w}"] = float(np.min(tail))
        out[f"max_{w}"] = float(np.max(tail))
    return out


def shape_features(ctx: np.ndarray) -> dict[str, float]:
    """Growth / momentum / range / level descriptors at the origin."""
    v = np.asarray(ctx, dtype=float).ravel()
    out = {}
    if v.size < 2:
        for key in ("diff_1", "slope_6", "slope_24", "range_24", "last_vs_mean_24", "level"):
            out[key] = float("nan") if v.size else 0.0
        return out
    out["diff_1"] = float(v[-1] - v[-2])
    for k in (6, 24):
        if v.size >= k:
            x = np.arange(k, dtype=float)
            y = v[-k:]
            out[f"slope_{k}"] = float(np.polyfit(x, y, 1)[0]) if np.ptp(y) > 0 else 0.0
        else:
            out[f"slope_{k}"] = float("nan")
    tail = _tail(v, 24)
    out["range_24"] = float(np.ptp(tail)) if tail.size else float("nan")
    out["last_vs_mean_24"] = float(v[-1] - np.mean(tail)) if tail.size else float("nan")
    out["level"] = float(v[-1])
    return out


def baseline_features(ctx: np.ndarray) -> dict[str, float]:
    """Compact feature set: the last value plus its recent mean and momentum.

    Kept small on purpose.  With 5 rolling origins there are very few training
    rows per horizon, and a wide feature matrix would be pure overfitting.
    """
    v = np.asarray(ctx, dtype=float).ravel()
    return {
        "level": float(v[-1]) if v.size else 0.0,
        "mean_6": float(np.mean(_tail(v, 6))) if v.size else 0.0,
        "mean_24": float(np.mean(_tail(v, 24))) if v.size else 0.0,
        "slope_24": shape_features(v).get("slope_24", 0.0),
    }


FEATURE_SETS = {
    "full": lambda ctx: {**lag_features(ctx), **rolling_features(ctx), **shape_features(ctx)},
    "compact": baseline_features,
}


def build_matrix(
    values: np.ndarray,
    origins,
    set_name: str = "compact",
    horizon: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for a batch of origins; ``horizon`` shifts the target only.

    The horizon is passed by the caller when it forms the target, so the same
    matrix can be reused across horizons.  Nothing here depends on values after
    ``origin``.
    """
    builder = FEATURE_SETS[set_name]
    rows = []
    for o in origins:
        ctx = np.asarray(values[: o + 1], dtype=float)
        rows.append(builder(ctx))
    names = list(rows[0].keys()) if rows else []
    return np.array([[r[n] for n in names] for r in rows], dtype=float), names
