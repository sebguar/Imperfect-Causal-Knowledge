"""Tests for the group-blind LR classifier + labeling.

TRIANGLE only, linear family, additive + effect-modifying regimes. Exercises the
Classifier decisions:
  accuracy-calibrated non-trivial labels; per-group negative pool;
  group-blind features (A excluded); no scaling (raw units);
  near-unregularized LR; full-dataset recourse pool; determinism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from icknowledge.classifier import build_classifier
from icknowledge.classifier.labeling import compute_signal, make_labels
from icknowledge.scm import make_linear_triangle
from icknowledge.utils.config import load_config

_CONFIG_PATH = "configs/classifier_triangle.yaml"
_REGIMES = ("additive", "effect_modifying")


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CONFIG_PATH)


# build_classifier runs a bisection over ~50 LR fits, so cache one result per
# regime for the whole module rather than rebuilding per test.
_RESULT_CACHE: dict = {}


def _result(cfg, regime):
    if regime not in _RESULT_CACHE:
        scm = make_linear_triangle(regime)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
            _RESULT_CACHE[regime] = build_classifier(scm, cfg)
    return _RESULT_CACHE[regime]


# --------------------------------------------------------------------------- #
# Accuracy lands in band (additive AND effect-modifying) — the PS-2 gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_accuracy_in_band(cfg, regime):
    lo, hi = tuple(cfg.classifier.accuracy_band)
    res = _result(cfg, regime)
    assert lo <= res.accuracy_overall <= hi


@pytest.mark.parametrize("regime", _REGIMES)
def test_both_classes_predicted(cfg, regime):
    res = _result(cfg, regime)
    # h is non-degenerate: it predicts BOTH classes over the full dataset. The
    # negative pool (h-negatives) is a strict, non-empty subset of all samples.
    assert 0 < len(res.negative_pool_indices) < res.n_samples


@pytest.mark.parametrize("regime", _REGIMES)
def test_per_group_negative_pool_meets_minimum(cfg, regime):
    min_neg = int(cfg.classifier.min_neg_per_group)
    res = _result(cfg, regime)
    # BOTH groups present and each has a non-trivial negative (recourse) pool.
    assert set(res.neg_counts_by_group) == {-1.0, 1.0}
    for group, count in res.neg_counts_by_group.items():
        assert count >= min_neg, f"group A={group} has only {count} negatives"


@pytest.mark.parametrize("regime", _REGIMES)
def test_base_rate_within_tolerance(cfg, regime):
    target = float(cfg.classifier.base_rate_target)
    tol = float(cfg.classifier.base_rate_tol)
    res = _result(cfg, regime)
    assert abs(res.realized_base_rate - target) <= tol


# --------------------------------------------------------------------------- #
# Group-blind invariant + no scaling + near-unregularized
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_group_blind_A_absent_from_features(cfg, regime):
    res = _result(cfg, regime)
    # A (Gender) must NOT be a feature of h; only descendants {X1, X2} are.
    assert res.protected_attr == "A"
    assert "A" not in res.feature_names
    assert res.feature_names == ["X1", "X2"]
    # the fitted model itself saw exactly the two group-blind features.
    assert res.model.n_features_in_ == 2


@pytest.mark.parametrize("regime", _REGIMES)
def test_no_scaler_raw_units_preserved(cfg, regime):
    res = _result(cfg, regime)
    # h is a bare LogisticRegression on RAW features — NOT a Pipeline with a
    # StandardScaler. Cost units == classifier units == raw feature units.
    assert isinstance(res.model, LogisticRegression)
    assert not hasattr(res.model, "steps")  # not a sklearn Pipeline
    # coefficients live in raw feature units: decision on a RAW feature row matches
    # the model's own prediction (no hidden transform between features and h).
    scm = make_linear_triangle(regime)
    raw_row = scm.sample(1, seed=123)[res.feature_names].to_numpy()
    logit = float(raw_row[0] @ res.model.coef_[0] + res.model.intercept_[0])
    assert int(logit > 0) == int(res.model.predict(raw_row)[0])


@pytest.mark.parametrize("regime", _REGIMES)
def test_near_unregularized_lr(cfg, regime):
    res = _result(cfg, regime)
    # penalty=None (or the large-C fallback) — negligible regularization.
    assert res.model.penalty is None or res.model.C >= 1e6


# --------------------------------------------------------------------------- #
# Determinism (fixed master seed)
# --------------------------------------------------------------------------- #


def test_determinism_under_fixed_master_seed(cfg):
    scm1 = make_linear_triangle("additive")
    scm2 = make_linear_triangle("additive")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = build_classifier(scm1, cfg)
        r2 = build_classifier(scm2, cfg)
    assert r1.label_spec.eps_scale == r2.label_spec.eps_scale
    assert r1.label_spec.tau == r2.label_spec.tau
    assert r1.accuracy_overall == r2.accuracy_overall
    assert r1.realized_base_rate == r2.realized_base_rate
    np.testing.assert_array_equal(r1.negative_pool_indices, r2.negative_pool_indices)
    np.testing.assert_allclose(r1.model.coef_, r2.model.coef_)
    np.testing.assert_allclose(r1.model.intercept_, r2.model.intercept_)


# --------------------------------------------------------------------------- #
# Labeling unit tests
# --------------------------------------------------------------------------- #


def test_make_labels_hits_target_base_rate():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=5000)
    z = rng.standard_normal(5000)
    y, tau = make_labels(signal, z, eps_scale=1.0, base_rate_target=0.35)
    # tau is the (1 - base_rate) quantile of signal+eps => base rate ~ target.
    assert abs(y.mean() - 0.35) < 0.01
    assert set(np.unique(y)) == {0, 1}


def test_compute_signal_rejects_mismatched_beta():
    import pandas as pd

    features = pd.DataFrame({"X1": [1.0, 2.0], "X2": [3.0, 4.0]})
    with pytest.raises(ValueError, match="beta keys"):
        compute_signal(features, {"X1": 1.0})  # missing X2


def test_compute_signal_linear_combination():
    import pandas as pd

    features = pd.DataFrame({"X1": [1.0, 2.0], "X2": [3.0, 4.0]})
    signal = compute_signal(features, {"X1": 2.0, "X2": 0.5})
    np.testing.assert_allclose(signal, [2.0 * 1 + 0.5 * 3, 2.0 * 2 + 0.5 * 4])
