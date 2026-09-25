"""Unit tests for the SCM counterfactual engine (abduction-action-prediction).

Exercises `abduct` / `counterfactual` through trivial
inline linear ANM fixtures, NOT project SCMs. Two fixtures:

  Helper A (2-var chain):  X -> Y;      X := U_X;  Y := 2X + U_Y
  Helper B (3-var chain):  X -> Y -> Z;  X := U_X;  Y := 2X + U_Y;  Z := 0.5Y + U_Z

All exogenous noise is standard normal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from icknowledge.scm import SCM

# -- inline fixtures --------------------------------------------------------


def _std_normal(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n)


def _eq_root(parents: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    return u  # X := U_X


def _eq_y(parents: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    return 2 * parents["X"] + u  # Y := 2X + U_Y


def _eq_z(parents: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    return 0.5 * parents["Y"] + u  # Z := 0.5Y + U_Z


def _make_helper_a(is_anm: bool = True) -> SCM:
    """Helper A -- 2-var linear ANM chain X -> Y."""
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    return SCM(
        graph=graph,
        equations={"X": _eq_root, "Y": _eq_y},
        noise_samplers={"X": _std_normal, "Y": _std_normal},
        name="helper-a",
        is_anm=is_anm,
    )


def _make_helper_b(is_anm: bool = True) -> SCM:
    """Helper B -- 3-var linear ANM chain X -> Y -> Z."""
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    return SCM(
        graph=graph,
        equations={"X": _eq_root, "Y": _eq_y, "Z": _eq_z},
        noise_samplers={"X": _std_normal, "Y": _std_normal, "Z": _std_normal},
        name="helper-b",
        is_anm=is_anm,
    )


# -- 1. is_anm gate ---------------------------------------------------------


def test_is_anm_gate_refuses_counterfactual_and_abduct():
    scm = _make_helper_a(is_anm=False)
    x = pd.DataFrame({"X": [1.0], "Y": [5.0]})
    with pytest.raises(NotImplementedError):
        scm.counterfactual(x, {"X": 1.0})
    with pytest.raises(NotImplementedError):
        scm.abduct(x)


# -- 2. empty-action identity -----------------------------------------------


def test_empty_action_is_identity():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0, 2.0], "Y": [5.0, 3.0]})
    cf = scm.counterfactual(x, {})
    pd.testing.assert_frame_equal(cf, x, check_dtype=False, atol=1e-12)


# -- 3. intervened node is exact --------------------------------------------


def test_intervened_node_exact():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0, 2.0], "Y": [5.0, 3.0]})
    cf = scm.counterfactual(x, {"X": 10.0})
    np.testing.assert_allclose(cf["X"].to_numpy(), 10.0)


# -- 4. absolute-value semantics (not delta) --------------------------------


def test_action_is_absolute_value_not_delta():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0, 2.0], "Y": [5.0, 3.0]})
    cf = scm.counterfactual(x, {"X": 5.0})
    # absolute: both rows become 5.0, NOT 1+5=6 and 2+5=7
    np.testing.assert_allclose(cf["X"].to_numpy(), [5.0, 5.0])


# -- 5. downstream propagation, hand-computed -------------------------------


def test_downstream_propagation_hand_computed():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0, 2.0], "Y": [5.0, 3.0]})
    # U_Y = Y - 2X = [5-2, 3-4] = [3.0, -1.0]
    cf = scm.counterfactual(x, {"X": 10.0})
    np.testing.assert_allclose(cf["X"].to_numpy(), [10.0, 10.0])
    # cf Y = 2*10 + U_Y = [23.0, 19.0]
    np.testing.assert_allclose(cf["Y"].to_numpy(), [23.0, 19.0])


# -- 6. upstream ancestors unaffected ---------------------------------------


def test_upstream_ancestor_unaffected():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0, 2.0], "Y": [5.0, 3.0]})
    cf = scm.counterfactual(x, {"Y": 100.0})
    np.testing.assert_allclose(cf["Y"].to_numpy(), [100.0, 100.0])
    # X is a non-intervened ancestor -> retains factual values
    np.testing.assert_allclose(cf["X"].to_numpy(), [1.0, 2.0])


# -- 7. middle-node intervention, hand-computed (Helper B) ------------------


def test_middle_node_intervention_hand_computed():
    scm = _make_helper_b()
    x = pd.DataFrame({"X": [2.0], "Y": [5.0], "Z": [4.0]})
    # Abduct: U_X = 2.0, U_Y = 5 - 2*2 = 1.0, U_Z = 4 - 0.5*5 = 1.5
    cf = scm.counterfactual(x, {"Y": 10.0})
    np.testing.assert_allclose(cf["X"].to_numpy(), [2.0])   # root, unchanged
    np.testing.assert_allclose(cf["Y"].to_numpy(), [10.0])  # intervened
    # cf Z = 0.5*10 + U_Z = 5 + 1.5 = 6.5
    np.testing.assert_allclose(cf["Z"].to_numpy(), [6.5])


# -- 8. round-trip abduction (consistency with shared-noise design) ---------


def test_round_trip_abduction_recovers_sampled_noise():
    scm = _make_helper_a()
    x = scm.sample(5000, seed=7)
    u_direct = scm.sample_noise(5000, seed=7)
    u_abd = scm.abduct(x)
    np.testing.assert_allclose(u_abd["X"].to_numpy(), u_direct["X"].to_numpy())
    np.testing.assert_allclose(u_abd["Y"].to_numpy(), u_direct["Y"].to_numpy())


# -- 9. unknown action key --------------------------------------------------


def test_unknown_action_key_raises_and_names_it():
    scm = _make_helper_a()
    x = pd.DataFrame({"X": [1.0], "Y": [5.0]})
    with pytest.raises(ValueError, match="NotANode"):
        scm.counterfactual(x, {"NotANode": 1.0})


# -- 10. missing factual column ---------------------------------------------


def test_missing_factual_column_raises_and_names_it():
    scm = _make_helper_a()
    x_missing_y = pd.DataFrame({"X": [1.0]})  # no Y column
    with pytest.raises(ValueError, match="Y"):
        scm.abduct(x_missing_y)
