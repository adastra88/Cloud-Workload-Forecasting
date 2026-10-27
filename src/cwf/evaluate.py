"""Rolling-origin validation harness (expanding window, overlapping horizons).

Why not a random split?  A random split lets the model see the neighbours of the
point it is asked to predict.  On a series with lag-1 autocorrelation of 0.66
that is close to handing the model the answer.  Rolling-origin evaluation always
trains on the past and tests on the future, which is the only ordering that
exists at deployment time.

Fold construction (frozen in section 5 of docs/protocol.md)::

    origins = range(min_train, n - horizon + 1, step)

so every fold satisfies ``origin + horizon <= n``, asserted, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import metrics as M

#: Frozen defaults from section 5 of docs/protocol.md.
DEFAULT_HORIZON = 24
DEFAULT_MIN_TRAIN = 60
DEFAULT_STEP = 6


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold, described purely by indices."""

    origin: int  # index of the last observation available for training
    horizon: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.origin + 1)

    @property
    def test_slice(self) -> slice:
        return slice(self.origin + 1, self.origin + 1 + self.horizon)


def make_folds(
    n_obs: int,
    horizon: int = DEFAULT_HORIZON,
    min_train: int = DEFAULT_MIN_TRAIN,
    step: int = DEFAULT_STEP,
) -> list[Fold]:
    """Enumerate rolling-origin folds for a series of length ``n_obs``.

    Returns an empty list when the series is too short: callers must fall back
    to something honest rather than crash (smoke datasets do this).
    """
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be positive")
    last_origin = n_obs - horizon
    if last_origin < min_train:
        return []
    return [Fold(origin=o, horizon=horizon) for o in range(min_train, last_origin + 1, step)]


def assert_folds_valid(folds: Iterable[Fold], n_obs: int) -> None:
    """Every fold must keep its test window inside the observed series."""
    for f in folds:
        if f.origin + f.horizon > n_obs:
            raise AssertionError(
                f"fold origin={f.origin} horizon={f.horizon} exceeds series length {n_obs}"
            )
        if f.origin < 1:
            raise AssertionError(f"fold origin={f.origin} leaves no training data")


@dataclass
class FoldResult:
    fold: Fold
    y_true: np.ndarray
    y_pred: np.ndarray
    train_values: np.ndarray

    @property
    def abs_errors(self) -> np.ndarray:
        return np.abs(self.y_true - self.y_pred)


@dataclass
class CandidateResult:
    """Every fold's outcome for one candidate on one target."""

    name: str
    target: str
    results: list[FoldResult]

    @property
    def y_true(self) -> np.ndarray:
        return np.vstack([r.y_true for r in self.results])

    @property
    def y_pred(self) -> np.ndarray:
        return np.vstack([r.y_pred for r in self.results])

    @property
    def residuals(self) -> np.ndarray:
        return self.y_true - self.y_pred

    def per_origin_rmse(self) -> np.ndarray:
        return np.array([M.rmse(r.y_true, r.y_pred) for r in self.results], dtype=float)

    def per_origin_mae(self) -> np.ndarray:
        return np.array([M.mae(r.y_true, r.y_pred) for r in self.results], dtype=float)

    def per_horizon_rmse(self) -> np.ndarray:
        return M.per_horizon(self.residuals)

    @property
    def scale(self) -> float:
        """MASE scale: pooled in-sample naive MAE over each fold's train window."""
        vals = []
        for r in self.results:
            s = M.naive_scale(r.train_values)
            if np.isfinite(s):
                vals.append(s)
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> dict:
        scale = self.scale
        per_origin = self.per_origin_rmse()
        se = float(np.std(per_origin, ddof=1) / np.sqrt(len(per_origin))) if len(per_origin) > 1 else float("nan")
        out = M.summarize(self.y_true, self.y_pred, scale=scale)
        out.update({
            "candidate": self.name,
            "target": self.target,
            "n_folds": len(self.results),
            "rmse_mean_per_origin": float(np.mean(per_origin)),
            "rmse_se_across_origins": se,
            "rmse_min_per_origin": float(np.min(per_origin)),
            "rmse_max_per_origin": float(np.max(per_origin)),
        })
        return out


def forecast_fn_context(
    fn: Callable[[np.ndarray, int], np.ndarray],
    values: np.ndarray,
    folds: Sequence[Fold],
    name: str,
    target: str,
) -> CandidateResult:
    """Evaluate a ``context -> forecast`` callable (baselines) on every fold."""
    results = []
    for f in folds:
        ctx = np.asarray(values[f.train_slice], dtype=float)
        y_true = np.asarray(values[f.test_slice], dtype=float)
        y_pred = np.asarray(fn(ctx, f.horizon), dtype=float)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"{name} produced {y_pred.shape}, expected {y_true.shape}")
        results.append(FoldResult(fold=f, y_true=y_true, y_pred=y_pred, train_values=ctx))
    return CandidateResult(name=name, target=target, results=results)


def evaluate_candidate(
    candidate,
    values: np.ndarray,
    folds: Sequence[Fold],
    target: str,
    name: str | None = None,
) -> CandidateResult:
    """Evaluate anything exposing ``forecast(context, horizon)`` or a plain callable."""
    fn = candidate.forecast if hasattr(candidate, "forecast") else candidate
    label = name or getattr(candidate, "name", getattr(fn, "__name__", "candidate"))
    return forecast_fn_context(fn, values, folds, label, target)


def results_table(results: Iterable[CandidateResult]) -> pd.DataFrame:
    """One row per candidate with the headline metrics plus fold spread."""
    return pd.DataFrame([r.summary() for r in results])


def per_horizon_table(results: Iterable[CandidateResult]) -> pd.DataFrame:
    """Wide table of RMSE by horizon step for every candidate."""
    rows = {}
    for r in results:
        rows[r.name] = r.per_horizon_rmse()
    frame = pd.DataFrame(rows)
    frame.insert(0, "horizon_step", np.arange(1, len(frame) + 1))
    return frame


def per_origin_table(results: Iterable[CandidateResult]) -> pd.DataFrame:
    """Long table of per-origin RMSE, useful for the spread plot."""
    rows = []
    for r in results:
        for f_res, rmse in zip(r.results, r.per_origin_rmse()):
            rows.append({
                "candidate": r.name,
                "origin": f_res.fold.origin,
                "rmse": float(rmse),
            })
    return pd.DataFrame(rows)


def dev_holdout_fold(n_obs: int, horizon: int = DEFAULT_HORIZON) -> Fold:
    """The single fold that mirrors the real protocol: train 0..n-h-1, test last h."""
    return Fold(origin=n_obs - horizon - 1, horizon=horizon)


def select_by_rule(
    table: pd.DataFrame,
    complexity_order: Sequence[str],
) -> tuple[str, pd.DataFrame]:
    """Apply the pre-registered 1-SE simplicity rule to a results table.

    Returns ``(winner_name, annotated_table)`` where the table gains a
    ``within_1se_of_best`` column and a boolean ``selected`` column.
    """
    if table.empty:
        raise ValueError("empty results table")
    order = {name: i for i, name in enumerate(complexity_order)}

    def rank(name: str) -> int:
        return order.get(name, len(order) + hash(name) % 1000)

    table = table.copy()
    best_idx = table["rmse_mean_per_origin"].idxmin()
    best = table.loc[best_idx]
    threshold = float(best["rmse_mean_per_origin"]) + float(
        0.0 if not np.isfinite(best["rmse_se_across_origins"]) else best["rmse_se_across_origins"]
    )
    table["within_1se_of_best"] = table["rmse_mean_per_origin"] <= threshold + 1e-12
    tied = table[table["within_1se_of_best"]].copy()
    tied["_rank"] = tied["candidate"].map(rank)
    winner = str(tied.sort_values(["_rank", "rmse_mean_per_origin"]).iloc[0]["candidate"])
    table["selected"] = table["candidate"] == winner
    table["complexity_rank"] = table["candidate"].map(rank)
    return winner, table.drop(columns=[c for c in table.columns if c == "_rank"])
