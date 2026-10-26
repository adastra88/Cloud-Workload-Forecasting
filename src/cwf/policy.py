"""Capacity policy: turn a point forecast and a band into a provisioning decision.

The newsvendor rule
-------------------
Provision for a quantile ``q`` of the demand distribution, where

    q* = c_under / (c_under + c_over)

``c_under`` is the cost of one unit of shortfall (a request that cannot be
served, an SLO breach, an on-call escalation) and ``c_over`` is the cost of one
unit of idle capacity (wasted instance-hours).  With ``c_under = 8`` and
``c_over = 1`` that gives ``q* = 0.889``: provision near the 89th percentile.

Why the mean is the wrong target
--------------------------------
The aggregate mean is a weak proxy for the risk the fleet actually carries.
In this release the mean is ~9.4 pp while the cross-VM P95 is ~37.5 pp, and
1.3% of readings are above 80% (`reports/artifacts/mean_hides_tails.csv`).
A policy that provisions the mean will look "efficient" and starve hot VMs, so
the recommendation is expressed at a quantile of the aggregate band *and* cross
checked against the hot-VM share.

The margin is chosen on rolling-origin folds only (never the holdout), by
minimising expected net cost.  The holdout is used exactly once, afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_C_UNDER = 8.0
DEFAULT_C_OVER = 1.0
DEFAULT_LEVELS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def newsvendor_quantile(c_under: float, c_over: float) -> float:
    """Cost-optimal service level; guards against degenerate inputs."""
    c_under = float(c_under)
    c_over = float(c_over)
    if c_under <= 0 and c_over <= 0:
        raise ValueError("at least one cost must be positive")
    return float(c_under / (c_under + c_over))


@dataclass
class PolicyResult:
    level: float
    provisioned: np.ndarray
    realised: np.ndarray
    shortfall_units: np.ndarray
    overprovision_units: np.ndarray
    c_under: float
    c_over: float

    @property
    def shortfall_events(self) -> int:
        return int((self.shortfall_units > 0).sum())

    @property
    def mean_shortfall(self) -> float:
        return float(np.mean(self.shortfall_units))

    @property
    def mean_overprovision(self) -> float:
        return float(np.mean(self.overprovision_units))

    @property
    def net_cost(self) -> float:
        return float(self.c_under * self.mean_shortfall + self.c_over * self.mean_overprovision)

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "c_under": self.c_under,
            "c_over": self.c_over,
            "shortfall_events": self.shortfall_events,
            "n_steps": int(self.realised.size),
            "mean_shortfall_units": self.mean_shortfall,
            "mean_overprovision_units": self.mean_overprovision,
            "max_shortfall_units": float(np.max(self.shortfall_units)) if self.realised.size else 0.0,
            "net_cost_per_step": self.net_cost,
        }


def evaluate_policy(
    provisioned,
    realised,
    c_under: float = DEFAULT_C_UNDER,
    c_over: float = DEFAULT_C_OVER,
    level: float = float("nan"),
) -> PolicyResult:
    """Score one provisioning vector against realised demand."""
    p = np.asarray(provisioned, dtype=float).ravel()
    y = np.asarray(realised, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    shortfall = np.maximum(y - p, 0.0)
    over = np.maximum(p - y, 0.0)
    return PolicyResult(level=level, provisioned=p, realised=y, shortfall_units=shortfall,
                        overprovision_units=over, c_under=float(c_under), c_over=float(c_over))


def level_to_band(forecast, lower, upper, level: float) -> np.ndarray:
    """Pick a provisioning level by interpolating the band at ``level``.

    ``lower``/``upper`` are the 5%/95% residual bounds; the band is assumed to
    grow linearly between them centred on the forecast, which is enough for a
    two-parameter decision and avoids fitting a full distribution.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    half = 0.5 * (hi - lo)
    z = _normal_quantile(level)
    z95 = _normal_quantile(0.95)
    scale = half / z95 if z95 else 0.0
    return f + z * scale


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF without a scipy import at module load."""
    from statistics import NormalDist

    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(NormalDist().inv_cdf(p))


def choose_level(
    folds: list[dict],
    c_under: float = DEFAULT_C_UNDER,
    c_over: float = DEFAULT_C_OVER,
    levels=DEFAULT_LEVELS,
) -> tuple[float, list[dict]]:
    """Pick the cost-minimising provisioning level on rolling-origin folds.

    ``folds`` is a list of dicts with ``forecast``, ``lower``, ``upper`` and
    ``realised`` arrays.  Returns ``(best_level, table_rows)``.
    """
    rows = []
    for level in levels:
        prov = np.concatenate([level_to_band(f["forecast"], f["lower"], f["upper"], level) for f in folds])
        real = np.concatenate([np.asarray(f["realised"], dtype=float) for f in folds])
        res = evaluate_policy(prov, real, c_under=c_under, c_over=c_over, level=level)
        rows.append(res.as_dict())
    table = sorted(rows, key=lambda r: r["net_cost_per_step"])
    return float(table[0]["level"]), table


def sensitivity(
    provisioned_by_level: dict,
    realised,
    ratios=(1, 2, 4, 8, 16, 32),
    c_over: float = DEFAULT_C_OVER,
) -> list[dict]:
    """Net cost for each cost ratio at each provisioning level."""
    out = []
    for ratio in ratios:
        for level in sorted(provisioned_by_level, key=str):
            prov = provisioned_by_level[level]
            res = evaluate_policy(prov, realised, c_under=float(ratio) * c_over, c_over=c_over,
                                  level=level if isinstance(level, float) else float("nan"))
            row = res.as_dict()
            row["cost_ratio"] = float(ratio)
            row["policy"] = str(level)
            out.append(row)
    return out


def peak_risk_trigger(forecast, lower, upper, peak_threshold: float, prob_threshold: float = 0.25,
                      bench_width: float | None = None) -> dict:
    """Cheap alert rule: is there a material chance of exceeding the peak threshold?

    Two independent triggers, both derivable at the origin:

    * the upper band crosses the training-window peak threshold, implying a
      non-trivial probability of a peak;
    * the band is wide relative to normal step-to-step movement, meaning the
      model does not know enough to provision tightly.
    """
    f = np.asarray(forecast, dtype=float)
    hi = np.asarray(upper, dtype=float)
    lo = np.asarray(lower, dtype=float)
    z95 = _normal_quantile(0.95)
    half = 0.5 * (hi - lo)
    scale = np.where(half > 0, half / z95, 0.0)
    prob_peak = np.array([1.0 - _normal_cdf((peak_threshold - fi) / si) if si > 0 else float(fi >= peak_threshold)
                          for fi, si in zip(f, scale)])
    wide = bool(bench_width is not None and half.max() > 2.0 * bench_width)
    return {
        "max_probability_of_peak": float(np.max(prob_peak)) if prob_peak.size else 0.0,
        "steps_with_peak_probability_above_threshold": int((prob_peak > prob_threshold).sum()),
        "band_unusually_wide": wide,
        "action": "hold extra headroom" if (prob_peak.max(initial=0.0) > prob_threshold or wide) else "normal",
    }


def _normal_cdf(x: float) -> float:
    from statistics import NormalDist

    return float(NormalDist().cdf(float(x)))
