"""Candidate models beyond the baselines.

All models share the ``forecast(context, horizon)`` interface from
``cwf.baselines``.  Two families:

* Statistical: ``ets_holt`` and ``ets_damped`` via statsmodels' ETS.  On a
  constant or near-constant series the optimiser can fail to converge; we catch
  that and fall back to persistence rather than emitting NaN.
* Direct multi-horizon regression: one model per horizon step trained on
  (feature matrix at origin) -> (value ``origin + h``).  Direct, not recursive,
  because recursive multi-step compounds the one-step error 24 times with no
  evidence it helps at this data size.  The number of training rows per horizon
  is tiny, so the models are deliberately small: heavy L2 for the ridge, shallow
  trees with a strong penalty for the booster.

Nothing here is fitted on data after the forecast origin: the regressors receive
the context array and only build features from it.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import features as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# statistical models
# --------------------------------------------------------------------------- #
def _ets_forecast(context: np.ndarray, horizon: int, damped: bool | None) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ctx = np.asarray(context, dtype=float).ravel()
    if ctx.size < 4 or np.ptp(ctx) < 1e-12:
        return np.full(horizon, ctx[-1] if ctx.size else 0.0, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                ctx,
                trend="add",
                damped_trend=bool(damped),
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=True)
        out = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:  # noqa: BLE001 - convergence failures are expected on short series
        return np.full(horizon, ctx[-1], dtype=float)
    if not np.all(np.isfinite(out)):
        return np.full(horizon, ctx[-1], dtype=float)
    return out


class ETSModel:
    """Additive-error Holt (optionally damped) trend, no seasonality."""

    def __init__(self, damped: bool | None = False, name: str | None = None):
        self.damped = damped
        self.name = name or ("ets_damped" if damped else "ets_holt")

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return _ets_forecast(context, horizon, self.damped)


# --------------------------------------------------------------------------- #
# direct multi-horizon regressors
# --------------------------------------------------------------------------- #
class DirectRegressor:
    """Direct multi-horizon regressor with one small model per horizon step.

    ``make_model`` must return a fresh scikit-learn estimator.  Features come
    from :mod:`cwf.features` and are built from the context only.
    """

    def __init__(
        self,
        make_model,
        name: str,
        set_name: str = "compact",
        min_rows: int = 25,
        min_train_obs: int = 30,
    ):
        self.make_model = make_model
        self.name = name
        self.set_name = set_name
        self.min_rows = min_rows
        self.min_train_obs = min_train_obs

    def _training_rows(self, ctx: np.ndarray, horizon: int):
        """Build (X, y) pairs entirely from ``ctx``; return (None, None) when too few."""
        n = ctx.size
        last_origin = n - horizon
        if last_origin < self.min_train_obs:
            return None, None
        origins = list(range(self.min_train_obs - 1, last_origin))
        if len(origins) < self.min_rows:
            return None, None
        X, _ = F.build_matrix(ctx, origins, set_name=self.set_name)
        y = np.array([ctx[o + horizon] for o in origins], dtype=float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if ok.sum() < self.min_rows:
            return None, None
        return X[ok], y[ok]

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=float).ravel()
        X_origin, _ = F.build_matrix(ctx, [ctx.size - 1], set_name=self.set_name)
        x_row = X_origin[0]
        out = np.empty(horizon, dtype=float)
        for h in range(1, horizon + 1):
            X, y = self._training_rows(ctx, h)
            if X is None or not np.isfinite(x_row).all():
                out[h - 1] = ctx[-1]
                continue
            model = self.make_model()
            try:
                model.fit(X, y)
                pred = float(model.predict(x_row.reshape(1, -1))[0])
            except Exception:  # noqa: BLE001
                pred = float(np.mean(y)) if y.size else ctx[-1]
            out[h - 1] = pred if np.isfinite(pred) else ctx[-1]
        return out


class ConstantModel:
    """Forecast the training-window mean; used when a regressor cannot fit."""

    name = "constant_mean"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=float).ravel()
        return np.full(horizon, float(np.mean(ctx)) if ctx.size else 0.0)


def ridge_direct(alpha: float = 10.0) -> DirectRegressor:
    from sklearn.linear_model import Ridge

    return DirectRegressor(
        make_model=lambda: Ridge(alpha=alpha, solver="svd"),
        name="ridge_direct",
        set_name="compact",
    )


def gbm_direct(max_depth: int = 2, learning_rate: float = 0.05, max_iter: int = 120) -> DirectRegressor:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return DirectRegressor(
        make_model=lambda: HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=8,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        ),
        name="gbm_direct",
        set_name="compact",
        min_rows=40,
    )


def candidate_models(alpha: float = 10.0) -> dict:
    """The frozen candidate set from docs/protocol.md §6 (models only)."""
    return {
        "ets_holt": ETSModel(damped=False),
        "ets_damped": ETSModel(damped=True),
        "ridge_direct": ridge_direct(alpha=alpha),
        "gbm_direct": gbm_direct(),
    }
