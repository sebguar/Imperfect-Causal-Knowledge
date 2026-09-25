"""Tests for the model-agnostic recourse harness.

TRIANGLE only, linear family, L0 + L2. Exercises the pilot decisions:
  ℓ₂ intervention cost; immutable A / grid-includes-0 / subset coverage;
  raw units; acted-only cost + believed==realized cost; the model-agnostic
  contract; L0 J=I no-propagation vs L2 downstream propagation; the G1 anchor smoke
  check (additive linear cell); L0 self-dual ℓ₂ collapse; determinism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from icknowledge.classifier import build_classifier
from icknowledge.recourse import (
    L0AssociationalModel,
    L2TrueSCMModel,
    build_action_space,
    generate_recourse,
    intervention_cost,
    raw_euclidean_reach,
)
from icknowledge.recourse.action_space import _IMMUTABLE
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import make_linear_triangle
from icknowledge.utils.config import load_config

_CONFIG_PATH = "configs/recourse_triangle.yaml"
_REGIMES = ("additive", "effect_modifying")


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CONFIG_PATH)


# The full L0+L2 pipeline (classifier bisection + grid over ~2600 negatives) is
# expensive; run each regime once for the whole module.
_RUN_CACHE: dict = {}


def _run(cfg, regime):
    if regime not in _RUN_CACHE:
        # The run_regime default is the full ladder; this
        # legacy module asserts the two-condition world it was written for, so
        # it passes its old tuple EXPLICITLY rather than relying on the default.
        _RUN_CACHE[regime] = run_regime(cfg, regime, conditions=("L0", "L2"))
    return _RUN_CACHE[regime]


# --------------------------------------------------------------------------- #
# Action space: immutability, grid-includes-0, subset reachability
# --------------------------------------------------------------------------- #


def test_A_is_immutable_and_rejected_from_action_space():
    assert "A" in _IMMUTABLE
    data = make_linear_triangle("additive").sample(200, seed=1)
    with pytest.raises(ValueError, match="immutable"):
        build_action_space(data, ["X1", "A"])


def test_grid_includes_no_change_per_axis_and_subsets_reachable():
    data = make_linear_triangle("additive").sample(500, seed=2)
    space = build_action_space(data, ["X1", "X2"], resolution=21)
    # each axis grid includes the no-change value 0 exactly.
    for var in space.acted:
        assert np.any(space.axis_grids[var] == 0.0)
    matrix = space.candidate_matrix()
    # no-op (all-zero) row is present.
    assert np.any(np.all(matrix == 0.0, axis=1))
    # single-variable interventions are present (exactly one axis nonzero).
    nonzero_counts = np.count_nonzero(matrix, axis=1)
    assert np.any(nonzero_counts == 1)
    # joint interventions are present (both axes nonzero).
    assert np.any(nonzero_counts == 2)


def test_A_never_appears_in_any_delta(cfg):
    run = _run(cfg, "additive")
    # every scored row's delta columns are over descendants only; there is no delta_A.
    assert "delta_A" not in run.table.columns
    assert set(c for c in run.table.columns if c.startswith("delta_")) == {"delta_X1", "delta_X2"}


# --------------------------------------------------------------------------- #
# Model-agnostic contract + L0 (J=I) vs L2 (propagation)
# --------------------------------------------------------------------------- #


def test_procedure_runs_with_L0_and_L2_interchangeably(cfg):
    # ONE procedure, two causal models, same call signature (model-agnostic).
    scm = make_linear_triangle("additive")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = build_classifier(scm, cfg)
    data = clf.dataset
    space = build_action_space(data, list(cfg.recourse.actionable), resolution=41)
    l2 = L2TrueSCMModel(scm, clf.feature_names, space.acted)
    l0 = L0AssociationalModel(clf.feature_names, space.acted)
    sol_l2 = generate_recourse(clf, l2, space, data)
    sol_l0 = generate_recourse(clf, l0, space, data)
    assert len(sol_l2) == len(sol_l0) == len(clf.negative_pool_indices)


def test_L0_leaves_non_acted_features_unchanged_but_L2_propagates():
    # X1 -> X2 with slope g in the additive linear triangle: acting on X1 alone must
    # move X2 under L2 (propagation) but NOT under L0 (J=I).
    scm = make_linear_triangle("additive", g=0.5)
    factual = scm.sample(1, seed=7).iloc[0]
    features = ["X1", "X2"]
    l0 = L0AssociationalModel(features, ["X1", "X2"])
    l2 = L2TrueSCMModel(scm, features, ["X1", "X2"])
    delta = {"X1": 1.0, "X2": 0.0}  # act on X1 only, X2 free
    cf0 = l0.predict(factual, delta)
    cf2 = l2.predict(factual, delta)
    # X1 moves by 1 under both.
    np.testing.assert_allclose(cf0[0], factual["X1"] + 1.0)
    np.testing.assert_allclose(cf2[0], factual["X1"] + 1.0)
    # X2: unchanged under L0 (J=I); moved by g=0.5 under L2 (propagation).
    np.testing.assert_allclose(cf0[1], factual["X2"])
    np.testing.assert_allclose(cf2[1], factual["X2"] + 0.5)


def test_L2_batch_matches_single_predict():
    scm = make_linear_triangle("effect_modifying")
    factual = scm.sample(1, seed=11).iloc[0]
    l2 = L2TrueSCMModel(scm, ["X1", "X2"], ["X1", "X2"])
    deltas = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.5, -0.5]])
    batch = l2.predict_batch(factual, deltas)
    for i, d in enumerate(deltas):
        single = l2.predict(factual, {"X1": d[0], "X2": d[1]})
        np.testing.assert_allclose(batch[i], single, atol=1e-12)
    # no-op row reproduces the factual features (identity counterfactual).
    np.testing.assert_allclose(batch[0], factual[["X1", "X2"]].to_numpy(), atol=1e-12)


# --------------------------------------------------------------------------- #
# Cost: ℓ₂ on acted vars, raw units, believed == realized
# --------------------------------------------------------------------------- #


def test_intervention_cost_is_l2_on_acted_vars():
    delta = {"X1": 3.0, "X2": 4.0}
    assert intervention_cost(delta, ["X1", "X2"]) == pytest.approx(5.0)
    # acted set is only X1 => X2's move is not charged.
    assert intervention_cost(delta, ["X1"]) == pytest.approx(3.0)
    # no-op costs nothing.
    assert intervention_cost({"X1": 0.0, "X2": 0.0}, ["X1", "X2"]) == pytest.approx(0.0)


def test_believed_cost_equals_realized_cost_per_individual(cfg):
    # EXPECTED BEHAVIOUR: same delta on same acted vars costs the same whichever
    # SCM enacts it; the believed-vs-realized signal lives in VALIDITY, not cost.
    run = _run(cfg, "additive")
    found = run.table[run.table["found"]]
    np.testing.assert_allclose(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy(), atol=1e-12
    )


def test_raw_units_no_scaler_in_cost_or_classifier(cfg):
    run = _run(cfg, "additive")
    # h is a bare LogisticRegression on raw features (no pipeline/scaler); cost lives
    # in the same raw units. A hand ℓ₂ over the stored raw deltas matches realized_cost.
    found = run.table[run.table["found"]]
    hand = np.hypot(found["delta_X1"].to_numpy(), found["delta_X2"].to_numpy())
    np.testing.assert_allclose(hand, found["realized_cost"].to_numpy(), atol=1e-9)


# --------------------------------------------------------------------------- #
# Scoring skeleton: records + no-∞ imputation
# --------------------------------------------------------------------------- #


def test_scoring_table_has_required_columns(cfg):
    run = _run(cfg, "additive")
    required = {
        "condition", "index", "A", "acted_set", "believed_cost", "realized_cost",
        "believed_validity", "realized_validity", "found",
        "delta_X1", "delta_X2", "factual_X1", "factual_X2",
        "realized_cf_X1", "realized_cf_X2",
    }
    assert required <= set(run.table.columns)
    assert set(run.table["condition"].unique()) == {"L0", "L2"}


def test_no_infinite_cost_imputed_for_failed_actions(cfg):
    run = _run(cfg, "additive")
    # costs are finite-or-NaN, never +/-inf (never impute ∞ into means).
    for col in ("believed_cost", "realized_cost"):
        assert not np.isinf(run.table[col].to_numpy()).any()
    # rows with no found action carry NaN cost and validity 0 on both sides.
    not_found = run.table[~run.table["found"]]
    if len(not_found):
        assert not_found["believed_cost"].isna().all()
        assert (not_found["realized_validity"] == 0).all()
        assert (not_found["believed_validity"] == 0).all()


def test_L2_realized_equals_believed_validity(cfg):
    # L2 believed == realized (same SCM): a believed-valid action is realized-valid.
    run = _run(cfg, "additive")
    l2 = run.table[(run.table["condition"] == "L2") & (run.table["found"])]
    assert (l2["believed_validity"] == 1).all()
    np.testing.assert_array_equal(
        l2["realized_validity"].to_numpy(), np.ones(len(l2), dtype=int)
    )


# --------------------------------------------------------------------------- #
# G1 anchor smoke check (additive linear cell) + L0 self-dual collapse
# --------------------------------------------------------------------------- #


def test_anchor_smoke_check_passes_additive(cfg):
    run = _run(cfg, "additive")
    assert run.anchor.passed, run.anchor.format()


def test_L0_believed_optimum_equals_raw_euclidean(cfg):
    # Self-dual ℓ₂ collapse: L0 (J=I) believed optimum == |h(x)|/||w||_2 on
    # resolvable individuals (this is exactly the L0 arm of the anchor check).
    run = _run(cfg, "additive")
    assert run.anchor.l0_passed
    # independent spot-check: the stored L0-believed cost tracks the closed form on a
    # near-boundary individual (smallest cost => reachable inside the grid span),
    # within one grid step.
    clf = run.classifier
    model = clf.model
    l0_found = run.table[(run.table["condition"] == "L0") & (run.table["found"])]
    l0 = l0_found.loc[l0_found["believed_cost"].idxmin()]
    factual_feats = np.array([l0["factual_X1"], l0["factual_X2"]])
    raw = raw_euclidean_reach(model, factual_feats)
    assert abs(l0["believed_cost"] - raw) <= run.anchor.grid_diag_step + 1e-9


# --------------------------------------------------------------------------- #
# Both regimes produce the end-to-end table; determinism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_end_to_end_table_produced(cfg, regime):
    run = _run(cfg, regime)
    assert len(run.table) == 2 * len(run.classifier.negative_pool_indices)
    assert set(run.summary["condition"].unique()) == {"L0", "L2"}
    # validity-by-group reported for both groups in both conditions.
    assert set(run.summary["A"].unique()) == {-1.0, 1.0}


def test_determinism_under_fixed_master_seed(cfg):
    r1 = _run(cfg, "additive")  # cached first run
    # fresh run under the same master seed; old tuple passed explicitly
    r2 = run_regime(cfg, "additive", conditions=("L0", "L2"))
    pd.testing.assert_frame_equal(r1.table, r2.table)
    assert r1.anchor.l2_max_abs_err == r2.anchor.l2_max_abs_err
