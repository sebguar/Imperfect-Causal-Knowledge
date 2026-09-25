"""Plumbing tests for the PS-9 EXPLORATORY mechanism diagnostic.

What is asserted here is that the three signals COMPUTE correctly on synthetic
cases with known answers, and that the separation read classifies known-disjoint
and known-overlapping configurations correctly.

What is deliberately NOT asserted is any scientific conclusion about the grid.
PS-9 is exploratory and post-hoc; pinning "signal X separates in cell Y" into the
test suite would convert a descriptive read into a de-facto pre-commitment and
would break the moment N goes 6 -> 20 (PS-1), which is the whole point of the
staging. The diagnostic's plumbing is tested; its verdict is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icknowledge.analysis.loading import (
    Cell,
    estimated_load_bearing,
    load_classifier_params,
    tanh_input_features,
    true_load_bearing,
)
from icknowledge.analysis.mechanism import (
    NOT_APPLICABLE,
    boundary_crosses_pool_mode,
    boundary_distance,
    cell_verdict,
    chance_separation_p,
    coefficient_errors,
    l0_projection_diagnostics,
    mean_boundary_distance,
    mean_boundary_distance_proxy,
    mean_saturation_positions,
    separation_read,
)

# --------------------------------------------------------------------------- #
# Signal 1 — boundary distance
# --------------------------------------------------------------------------- #


def test_boundary_distance_known_3_4_5_triangle():
    """w=(3,4), b=0, x=(1,1) -> |7| / 5 = 1.4, by hand."""
    assert boundary_distance(np.array([3.0, 4.0]), 0.0, np.array([1.0, 1.0])) == (
        pytest.approx(1.4)
    )


def test_boundary_distance_is_signless_and_scale_invariant_in_w():
    """Doubling (w, b) is the same boundary, hence the same distance."""
    w, b = np.array([2.0, -1.0]), 0.5
    x = np.array([[3.0, 1.0], [-4.0, 2.0]])
    base = boundary_distance(w, b, x)
    assert base == pytest.approx(boundary_distance(2 * w, 2 * b, x))
    assert np.all(base >= 0.0)


def test_boundary_distance_zero_on_the_boundary():
    w = np.array([1.0, 1.0])
    assert boundary_distance(w, -2.0, np.array([1.0, 1.0]))[0] == pytest.approx(0.0)


def test_boundary_distance_rejects_degenerate_classifier():
    with pytest.raises(ValueError, match="degenerate"):
        boundary_distance(np.zeros(2), 1.0, np.ones(2))


def test_l0_min_flipping_action_norm_equals_boundary_distance():
    """The identity that licenses reading Signal 1 off the persisted L0 rung.

    At L0 the belief is J = I, so the believed counterfactual is x + delta and the
    minimum-l_2 flipping action is the orthogonal projection onto the boundary,
    whose norm IS |w'x + b| / ||w||_2 (the self-dual l_2 collapse the G1 anchor
    asserts). Verified here by brute force against a fine search, which is also
    what the harness does — on a grid, hence the >= and the tolerance.
    """
    rng = np.random.default_rng(20260710)
    w, b = np.array([1.3, -0.7]), 0.4
    for _ in range(20):
        x = rng.normal(size=2)
        if x @ w + b >= 0:  # only negatives need recourse
            continue
        analytic = float(boundary_distance(w, b, x)[0])

        # Fine grid over the 2-D action space, mirroring the harness's search.
        axis = np.linspace(-6.0, 6.0, 601)
        d1, d2 = np.meshgrid(axis, axis, indexing="ij")
        deltas = np.stack([d1.ravel(), d2.ravel()], axis=1)
        flips = (x + deltas) @ w + b > 0
        best = float(np.min(np.linalg.norm(deltas[flips], axis=1)))

        # A grid solution can only OVERSHOOT the continuous optimum, never beat it.
        assert best >= analytic - 1e-9
        assert best == pytest.approx(analytic, abs=0.05)


def test_mean_boundary_distance_proxy_reads_l0_believed_cost_of_the_right_group():
    scoring = pd.DataFrame(
        {
            "A": [-1.0, -1.0, 1.0, 1.0],
            "believed_cost": [1.0, 3.0, 100.0, 200.0],
        }
    )
    assert mean_boundary_distance_proxy(scoring, -1.0) == pytest.approx(2.0)
    assert mean_boundary_distance_proxy(scoring, 1.0) == pytest.approx(150.0)


def test_mean_boundary_distance_proxy_rejects_empty_pool():
    scoring = pd.DataFrame({"A": [1.0], "believed_cost": [1.0]})
    with pytest.raises(ValueError, match="empty"):
        mean_boundary_distance_proxy(scoring, -1.0)


# --- Exact Signal 1 from persisted classifier params -------------------- #


def test_mean_boundary_distance_exact_uses_factual_columns_and_the_right_group():
    """The exact reading: |w'x + b| / ||w||_2 on the pool's own factual vectors.

    w = (3, 4), ||w|| = 5, b = 0. The A=-1 pool sits at (1,2) and (3,4), giving
    distances |3+8|/5 = 2.2 and |9+16|/5 = 5.0 -> mean 3.6. The A=+1 rows carry
    wildly different values, so a group-filter bug cannot pass.
    """
    scoring = pd.DataFrame(
        {
            "A": [-1.0, -1.0, 1.0],
            "factual_X1": [1.0, 3.0, 100.0],
            "factual_X2": [2.0, 4.0, 100.0],
        }
    )
    value = mean_boundary_distance(
        scoring, np.array([3.0, 4.0]), 0.0, ["X1", "X2"], -1.0
    )
    assert value == pytest.approx(3.6)


def test_mean_boundary_distance_exact_has_no_grid_overshoot_by_construction():
    """The exact reading equals the analytic distance; the proxy exceeds it.

    This is the whole point of the persisted-params path: the proxy is the boundary distance
    PLUS the finite-grid overshoot, so it can only ever be larger. Asserted on a
    synthetic where the analytic answer is known, so the relation is a property of
    the two estimators rather than an artifact of one results tree.
    """
    scoring = pd.DataFrame(
        {"A": [-1.0], "factual_X1": [3.0], "factual_X2": [4.0], "believed_cost": [5.4]}
    )
    exact = mean_boundary_distance(
        scoring, np.array([1.0, 0.0]), 0.0, ["X1", "X2"], -1.0
    )
    assert exact == pytest.approx(3.0)  # |3| / 1
    assert mean_boundary_distance_proxy(scoring, -1.0) > exact


def test_boundary_margin_at_pool_mode_is_signed_and_uses_the_median():
    """Signal 4: NEGATIVE when the modal individual is on the rejected side.

    The median, not the mean, defines the modal point — asserted here by giving
    the pool one extreme outlier that would drag a mean across the boundary while
    leaving the median (and hence the reported sign) where it belongs.
    """
    scoring = pd.DataFrame(
        {
            "A": [-1.0] * 4,
            "factual_X1": [-2.0, -1.0, -1.0, 400.0],
            "factual_X2": [0.0, 0.0, 0.0, 0.0],
        }
    )
    margin = boundary_crosses_pool_mode(
        scoring, np.array([1.0, 0.0]), 0.0, ["X1", "X2"], -1.0
    )
    assert margin < 0  # median X1 = -1.0 -> rejected side
    assert margin == pytest.approx(-1.0)

    # Shift the boundary past the bulk and the sign flips.
    flipped = boundary_crosses_pool_mode(
        scoring, np.array([1.0, 0.0]), 5.0, ["X1", "X2"], -1.0
    )
    assert flipped > 0


def test_load_classifier_params_refuses_to_fall_back_on_a_pre_persistence_tree(tmp_path):
    """Absent params RAISE — they never silently degrade to the L0 proxy.

    The N=6 tree legitimately has no classifier_params.json. A fallback would
    make one column name mean two different quantities depending on which tree was
    read, which is exactly the ambiguity persistence was added to remove.
    """
    cell = Cell("triangle", "linear", "additive")
    cell.seed_dir(0, tmp_path).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="does not silently substitute"):
        load_classifier_params(cell, 0, tmp_path)


def test_projection_audit_is_perfect_on_exactly_parallel_actions():
    """Exact projections: every action parallel to w, no grid overshoot."""
    scoring = pd.DataFrame(
        {
            "A": [-1.0, -1.0, -1.0],
            "delta_X1": [0.6, 1.2, 3.0],  # all along (0.6, 0.8)
            "delta_X2": [0.8, 1.6, 4.0],
        }
    )
    audit = l0_projection_diagnostics(scoring, ["X1", "X2"], -1.0)
    assert audit["mean_cosine"] == pytest.approx(1.0)
    assert audit["overshoot"] == pytest.approx(0.0, abs=1e-12)


def test_projection_audit_detects_off_axis_actions_as_overshoot():
    """A 45-degree-scattered action set cannot be a set of exact projections."""
    scoring = pd.DataFrame(
        {
            "A": [-1.0, -1.0],
            "delta_X1": [1.0, 0.0],
            "delta_X2": [0.0, 1.0],
        }
    )
    audit = l0_projection_diagnostics(scoring, ["X1", "X2"], -1.0)
    assert audit["mean_cosine"] == pytest.approx(np.cos(np.pi / 4))
    assert audit["overshoot"] > 0.4  # ||d|| = 1 but the along-w component is ~0.707


def test_projection_audit_rejects_an_all_zero_action_set():
    scoring = pd.DataFrame({"A": [-1.0], "delta_X1": [0.0], "delta_X2": [0.0]})
    with pytest.raises(ValueError, match="no nonzero"):
        l0_projection_diagnostics(scoring, ["X1", "X2"], -1.0)


# --------------------------------------------------------------------------- #
# Signal 2 — estimation-error sign
# --------------------------------------------------------------------------- #


def test_true_load_bearing_matches_the_builder_defaults():
    """Ground truth is read from the builder signature; pin the known values."""
    tri_add = Cell("triangle", "linear", "additive")
    assert true_load_bearing(tri_add, -1.0)["X1->X2"] == pytest.approx(0.5)

    tri_em = Cell("triangle", "linear", "effect_modifying")
    assert true_load_bearing(tri_em, -1.0)["X1->X2"] == pytest.approx(0.3)  # g_neg
    assert true_load_bearing(tri_em, 1.0)["X1->X2"] == pytest.approx(0.9)  # g_pos

    col_em = Cell("collider", "linear", "effect_modifying")
    truth = true_load_bearing(col_em, -1.0)
    assert truth["X1->X3"] == pytest.approx(0.2)  # beta - gamma = 0.5 - 0.3
    assert truth["X2->X3"] == pytest.approx(0.5)  # eta, unmodified
    assert true_load_bearing(col_em, 1.0)["X1->X3"] == pytest.approx(0.8)


def test_estimated_load_bearing_folds_the_interaction_at_the_group_s_own_A():
    cell = Cell("collider", "linear", "effect_modifying")
    coefficients = {"X3": {"X1": 0.55, "A*X1": 0.25, "X2": 0.44, "A": 9.9}}
    neg = estimated_load_bearing(cell, coefficients, -1.0)
    assert neg["X1->X3"] == pytest.approx(0.55 - 0.25)
    assert neg["X2->X3"] == pytest.approx(0.44)  # no A interaction on this edge
    assert estimated_load_bearing(cell, coefficients, 1.0)["X1->X3"] == pytest.approx(
        0.80
    )


def test_coefficient_error_sign_is_estimated_minus_true():
    """Known error: +0.10 over-estimate on X1->X3, -0.05 under-estimate on X2->X3."""
    cell = Cell("collider", "linear", "effect_modifying")
    coefficients = {"X3": {"X1": 0.6, "A*X1": 0.3, "X2": 0.45, "A": 1.0}}
    # A=-1 -> estimated X1->X3 = 0.6 - 0.3 = 0.30 vs true 0.20 -> +0.10.
    errors = coefficient_errors(cell, coefficients, -1.0)
    assert errors["X1->X3"] == pytest.approx(+0.10)
    assert errors["X2->X3"] == pytest.approx(-0.05)


def test_coefficient_error_is_exactly_zero_when_the_fit_is_the_truth():
    cell = Cell("triangle", "nlg", "effect_modifying")
    truth = true_load_bearing(cell, -1.0)["X1->X2"]
    # tanh(X1) main effect + A*tanh(X1) interaction reproducing g_neg at A=-1.
    coefficients = {"X2": {"tanh(X1)": truth, "A*tanh(X1)": 0.0}}
    assert coefficient_errors(cell, coefficients, -1.0)["X1->X2"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Signal 3 — saturation position
# --------------------------------------------------------------------------- #


def test_saturation_is_mean_abs_tanh_input_over_the_pool():
    cell = Cell("collider", "nlg", "additive")
    scoring = pd.DataFrame(
        {
            "A": [-1.0, -1.0, 1.0],
            "factual_X1": [-2.0, 4.0, 999.0],
            "factual_X2": [1.0, -3.0, 999.0],
        }
    )
    positions = mean_saturation_positions(scoring, cell, -1.0)
    assert positions == {"X1": pytest.approx(3.0), "X2": pytest.approx(2.0)}


def test_saturation_is_na_on_linear_cells_and_x1_only_on_the_nlg_triangle():
    assert tanh_input_features(Cell("collider", "linear", "additive")) == []
    assert tanh_input_features(Cell("triangle", "nlg", "additive")) == ["X1"]
    assert tanh_input_features(Cell("collider", "nlg", "additive")) == ["X1", "X2"]

    scoring = pd.DataFrame({"A": [-1.0], "factual_X1": [1.0], "factual_X2": [1.0]})
    assert mean_saturation_positions(
        scoring, Cell("collider", "linear", "additive"), -1.0
    ) == {}


# --------------------------------------------------------------------------- #
# The separation read
# --------------------------------------------------------------------------- #


def _table(validity: list[float], **signals: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"validity_A_neg": validity})
    frame["seed_idx"] = range(len(validity))
    frame["seed"] = [1000 + i for i in range(len(validity))]
    frame["n_pool"] = 100
    for name, values in signals.items():
        frame[name] = values
    return frame


def test_disjoint_ranges_are_a_clean_separator_and_overlapping_ranges_are_not():
    table = _table(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        clean=[0.1, 0.2, 0.3, 0.8, 0.9, 1.0],  # strictly disjoint
        muddy=[0.1, 0.9, 0.3, 0.2, 0.8, 1.0],  # interleaved
    )
    read = separation_read(table).set_index("signal")
    assert read.loc["clean", "clean_separator"]
    assert read.loc["clean", "separation_gap"] == pytest.approx(0.5)
    assert not read.loc["muddy", "clean_separator"]
    assert read.loc["muddy", "separation_gap"] == pytest.approx(0.0)


def test_a_constant_signal_never_separates_and_yields_no_rank_correlation():
    table = _table([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], flat=[0.5] * 6)
    read = separation_read(table).iloc[0]
    assert not read["clean_separator"]
    assert np.isnan(read["spearman_rho"])


def test_single_stratum_is_reported_not_applicable_rather_than_separated():
    """All seeds high: there is no contrast, and the read must say so."""
    table = _table([1.0, 0.9, 0.8, 0.95, 0.99, 0.85], anything=[1, 2, 3, 4, 5, 6])
    read = separation_read(table)
    assert read["verdict"].eq(NOT_APPLICABLE).all()
    assert not read["clean_separator"].any()
    assert np.isnan(read["chance_separation_p"].iloc[0])
    assert "NOT APPLICABLE" in cell_verdict(read, table)


def test_rank_correlation_survives_a_degenerate_split():
    """A graded (non-bimodal) cell still yields a monotone read, not an empty row."""
    table = _table(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.55], monotone=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    read = separation_read(table)
    assert read["verdict"].eq(NOT_APPLICABLE).all()  # no strata to separate...
    assert read["spearman_rho"].iloc[0] == pytest.approx(-1.0)  # ...but rho survives
    assert "rho=-1.000" in cell_verdict(read, table)


def test_chance_separation_probability_is_the_random_labelling_null():
    assert chance_separation_p(3, 3) == pytest.approx(2 / 20)  # 2 / C(6,3)
    assert chance_separation_p(5, 1) == pytest.approx(2 / 6)  # 2 / C(6,1)
    assert np.isnan(chance_separation_p(6, 0))


def test_two_separators_are_reported_as_confounded_not_picked_between():
    table = _table(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        a=[0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        b=[5.0, 6.0, 7.0, 1.0, 2.0, 3.0],
    )
    verdict = cell_verdict(separation_read(table), table)
    assert "CONFOUNDED" in verdict
    assert "a" in verdict and "b" in verdict


def test_no_separator_is_a_reportable_outcome_not_an_error():
    table = _table(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        a=[0.1, 0.9, 0.3, 0.2, 0.8, 1.0],
        b=[5.0, 1.0, 7.0, 6.0, 2.0, 3.0],
    )
    verdict = cell_verdict(separation_read(table), table)
    assert "NO CLEAN SEPARATOR" in verdict
    assert "Multiplicity" in verdict


def test_verdict_states_multiplicity_so_a_lone_separator_is_read_against_chance():
    table = _table(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        a=[0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        b=[0.1, 0.9, 0.3, 0.2, 0.8, 1.0],
    )
    verdict = cell_verdict(separation_read(table), table)
    assert "single clean separator: a" in verdict
    assert "p=0.100" in verdict
