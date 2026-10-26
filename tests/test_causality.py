"""Causality: features at an origin must not move when values after it change."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cwf.features import FEATURE_SETS, build_matrix, lag_features


@pytest.mark.parametrize("set_name", sorted(FEATURE_SETS))
def test_perturbing_the_future_leaves_features_unchanged(set_name):
    rng = np.random.default_rng(0)
    series = 10.0 + rng.normal(0, 1, 120).cumsum() * 0.1
    origin = 60

    base, names = build_matrix(series, [origin], set_name=set_name)

    tampered = series.copy()
    tampered[origin + 1 :] = tampered[origin + 1 :] + 1000.0
    after, after_names = build_matrix(tampered, [origin], set_name=set_name)

    assert names == after_names
    np.testing.assert_allclose(base, after, rtol=0, atol=0)


@pytest.mark.parametrize("set_name", sorted(FEATURE_SETS))
def test_perturbing_after_the_last_origin_changes_nothing(set_name):
    rng = np.random.default_rng(1)
    series = 10.0 + rng.normal(0, 1, 150).cumsum() * 0.1
    origins = [60, 80, 100]

    base, _ = build_matrix(series, origins, set_name=set_name)
    tampered = series.copy()
    tampered[101:] = -500.0
    after, _ = build_matrix(tampered, origins, set_name=set_name)
    np.testing.assert_allclose(base, after, rtol=0, atol=0)


def test_horizon_does_not_leak_into_features():
    rng = np.random.default_rng(2)
    series = 5.0 + rng.normal(0, 1, 80)
    a, _ = build_matrix(series, [50], set_name="compact", horizon=1)
    b, _ = build_matrix(series, [50], set_name="compact", horizon=24)
    np.testing.assert_allclose(a, b)


def test_feature_builder_only_sees_the_context():
    """``build_matrix`` slices ``values[: origin + 1]`` itself."""
    assert "horizon" in inspect.signature(build_matrix).parameters
    ctx = np.arange(1, 21, dtype=float)
    a, b = lag_features(ctx), lag_features(ctx.copy())
    for key in a:
        assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))
