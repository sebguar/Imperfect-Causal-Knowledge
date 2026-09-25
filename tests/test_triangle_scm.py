"""Tests for the concrete TRIANGLE SCMs.

Covers both functional families (linear, nonlinear-Gaussian) × both attribute
regimes (additive, effect_modifying). Three groups:

  Group 1 — hand-derived linear counterfactual GATE (the gate criterion).
  Group 2 — regime contrast: the mechanical core of prediction H4 (asymmetry
            under effect-modification, symmetry under additivity).
  Group 3 — structural / sanity (regime storage, graph, sampling, ANM round-trip,
            nonlinear-Gaussian smoke test).

Exercises scm-specification conventions 1 (Gender root, nonzero influence),
2 (regime declared not inferred), 5/6 (ANM, mechanism-level precondition).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icknowledge.scm import (
    make_linear_triangle,
    make_nonlinear_gaussian_triangle,
)

# ===========================================================================
# Group 1 — HAND-DERIVED linear counterfactual gate (THE gate criterion)
# ===========================================================================
#
# Build make_linear_triangle("additive", a=2.0, b=1.0, g=0.5, sigma=1.0).
# Factual applicant: A=1, X1=4, X2=6.
#   Abduct: U_A = 1
#           U1 = X1 - a*A       = 4 - 2*1        = 2
#           U2 = X2 - b*A - g*X1 = 6 - 1*1 - 0.5*4 = 6 - 1 - 2 = 3
#   Action do(X1=10). Predict:
#           A  = 1   (unchanged)
#           X1 = 10  (intervened)
#           X2 = b*A + g*X1 + U2 = 1 + 0.5*10 + 3 = 1 + 5 + 3 = 9
#   Expected counterfactual: (A=1, X1=10, X2=9).


def _linear_additive_gate_scm():
    return make_linear_triangle("additive", a=2.0, b=1.0, g=0.5, sigma=1.0)


def test_gate_abduct_recovers_hand_derived_noise():
    scm = _linear_additive_gate_scm()
    factual = pd.DataFrame({"A": [1.0], "X1": [4.0], "X2": [6.0]})
    u = scm.abduct(factual)
    np.testing.assert_allclose(u["A"].to_numpy(), [1.0])
    np.testing.assert_allclose(u["X1"].to_numpy(), [2.0])
    np.testing.assert_allclose(u["X2"].to_numpy(), [3.0])


def test_gate_counterfactual_hand_derived():
    # THE gate: hand-derived structural counterfactual under do(X1=10).
    scm = _linear_additive_gate_scm()
    factual = pd.DataFrame({"A": [1.0], "X1": [4.0], "X2": [6.0]})
    cf = scm.counterfactual(factual, {"X1": 10.0})
    np.testing.assert_allclose(cf["A"].to_numpy(), [1.0])
    np.testing.assert_allclose(cf["X1"].to_numpy(), [10.0])
    np.testing.assert_allclose(cf["X2"].to_numpy(), [9.0])


# ===========================================================================
# Group 2 — regime contrast (the H4 mechanism, hand-checkable)
# ===========================================================================
#
# Two applicants identical in (X1, X2) but differing in A:
#   A = [+1, -1], X1 = [4.0, 4.0], X2 = [6.0, 6.0].
# Same action do(X1=10) applied to both.


def _two_applicants():
    return pd.DataFrame({"A": [1.0, -1.0], "X1": [4.0, 4.0], "X2": [6.0, 6.0]})


def test_regime_contrast_effect_modifying_is_asymmetric():
    # EFFECT-MODIFYING: X2 := g_of_A*X1 + U2, g_of_A = 0.9 (A>0) or 0.3 (A<0).
    # Per-row abduction of U2 = X2 - g_of_A*X1:
    #   A=+1: U2 = 6 - 0.9*4 = 6 - 3.6 = 2.4 ; X2_cf = 0.9*10 + 2.4 = 11.4 ; ΔX2 = 5.4
    #   A=-1: U2 = 6 - 0.3*4 = 6 - 1.2 = 4.8 ; X2_cf = 0.3*10 + 4.8 =  7.8 ; ΔX2 = 1.8
    # ΔX2 differs between groups (asymmetry); A=+1 change exceeds A=-1 change.
    scm = make_linear_triangle(
        "effect_modifying", a=2.0, g_pos=0.9, g_neg=0.3, sigma=1.0
    )
    factual = _two_applicants()
    cf = scm.counterfactual(factual, {"X1": 10.0})
    delta = cf["X2"].to_numpy() - factual["X2"].to_numpy()
    np.testing.assert_allclose(delta, [5.4, 1.8])
    # asymmetric: the two ΔX2 values are NOT equal
    assert not np.isclose(delta[0], delta[1])
    # direction: A=+1 (higher slope) responds more strongly than A=-1
    assert delta[0] > delta[1]


def test_regime_contrast_additive_is_symmetric():
    # ADDITIVE control: X2 := b*A + g*X1 + U2, slope g SHARED.
    #   ΔX2 = g*(10 - X1_factual) = 0.5*(10 - 4) = 3.0 for BOTH rows, independent of A.
    # This symmetry is the control that makes the effect-modifying test meaningful.
    scm = make_linear_triangle("additive", a=2.0, b=1.0, g=0.5, sigma=1.0)
    factual = _two_applicants()
    cf = scm.counterfactual(factual, {"X1": 10.0})
    delta = cf["X2"].to_numpy() - factual["X2"].to_numpy()
    np.testing.assert_allclose(delta, [3.0, 3.0])
    # symmetric: equal across the two applicants regardless of A
    np.testing.assert_allclose(delta[0], delta[1])


# ===========================================================================
# Group 3 — structural / sanity
# ===========================================================================


def test_regime_is_stored_and_invalid_regime_raises():
    for regime in ("additive", "effect_modifying"):
        assert make_linear_triangle(regime).regime == regime
        assert make_nonlinear_gaussian_triangle(regime).regime == regime
    with pytest.raises(ValueError):
        make_linear_triangle("bogus")
    with pytest.raises(ValueError):
        make_nonlinear_gaussian_triangle("bogus")


def test_graph_structure_matches_triangle_topology():
    scm = make_linear_triangle("additive")
    graph = scm.graph
    assert set(graph.nodes()) == {"A", "X1", "X2"}
    assert set(graph.edges()) == {("A", "X1"), ("A", "X2"), ("X1", "X2")}
    # A is a root (no parents), the protected Gender node (convention 1).
    assert scm.parents("A") == []
    # topological order: A before X1 before X2.
    order = scm.nodes
    assert order.index("A") < order.index("X1") < order.index("X2")


def test_sample_produces_binary_A_and_expected_columns():
    scm = make_linear_triangle("additive")
    data = scm.sample(20000, seed=0)
    assert set(data.columns) == {"A", "X1", "X2"}
    # A ∈ {-1, +1} only (centered binary group indicator).
    assert set(np.unique(data["A"].to_numpy())) == {-1.0, 1.0}


def test_is_anm_round_trip_recovers_sampled_noise_both_families():
    # abduct(sample) allclose sample_noise: confirms the abduction noise contract
    # holds for the concrete triangle in both core families (additive regime).
    for builder in (make_linear_triangle, make_nonlinear_gaussian_triangle):
        scm = builder("additive")
        x = scm.sample(3000, seed=5)
        u_abd = scm.abduct(x)
        u_direct = scm.sample_noise(3000, seed=5)
        for node in ("A", "X1", "X2"):
            np.testing.assert_allclose(
                u_abd[node].to_numpy(), u_direct[node].to_numpy()
            )


def test_nonlinear_gaussian_builds_and_counterfactual_runs():
    scm = make_nonlinear_gaussian_triangle("effect_modifying")
    x = scm.sample(5, seed=3)
    cf = scm.counterfactual(x, {"X1": 1.5})
    assert list(cf.columns) == ["A", "X1", "X2"]
    assert cf.shape == x.shape
    # intervened node takes the action value exactly.
    np.testing.assert_allclose(cf["X1"].to_numpy(), 1.5)
