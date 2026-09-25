"""Unit tests for the SCM base engine.

The engine is exercised through a trivial inline linear 2-variable SCM
(X -> Y, Y := 2X + noise). This is a test fixture, NOT a project SCM.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from icknowledge.scm import SCM

# -- trivial linear 2-variable test SCM (inline fixture) --------------------


def _std_normal(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n)


def _eq_x(parents: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    return u  # X := U_X


def _eq_y(parents: dict[str, np.ndarray], u: np.ndarray) -> np.ndarray:
    return 2 * parents["X"] + u  # Y := 2X + U_Y


def _make_linear_scm(regime: str | None = None) -> SCM:
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    return SCM(
        graph=graph,
        equations={"X": _eq_x, "Y": _eq_y},
        noise_samplers={"X": _std_normal, "Y": _std_normal},
        regime=regime,
        name="linear-test",
    )


# -- construction validation ------------------------------------------------


def test_cyclic_graph_raises():
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "X")
    with pytest.raises(ValueError):
        SCM(
            graph=graph,
            equations={"X": _eq_x, "Y": _eq_y},
            noise_samplers={"X": _std_normal, "Y": _std_normal},
        )


def test_missing_equation_raises_and_names_node():
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    with pytest.raises(ValueError, match="Y"):
        SCM(
            graph=graph,
            equations={"X": _eq_x},  # missing Y
            noise_samplers={"X": _std_normal, "Y": _std_normal},
        )


def test_missing_noise_sampler_raises():
    graph = nx.DiGraph()
    graph.add_edge("X", "Y")
    with pytest.raises(ValueError, match="Y"):
        SCM(
            graph=graph,
            equations={"X": _eq_x, "Y": _eq_y},
            noise_samplers={"X": _std_normal},  # missing Y
        )


def test_invalid_regime_raises_and_valid_regimes_stored():
    with pytest.raises(ValueError, match="regime"):
        _make_linear_scm(regime="foo")

    assert _make_linear_scm(regime="additive").regime == "additive"
    assert _make_linear_scm(regime="effect_modifying").regime == "effect_modifying"
    assert _make_linear_scm().regime is None


# -- sampling ---------------------------------------------------------------


def test_sample_shape_and_columns():
    scm = _make_linear_scm()
    df = scm.sample(100, 0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert set(df.columns) == {"X", "Y"}


def test_determinism():
    scm = _make_linear_scm()
    pd.testing.assert_frame_equal(scm.sample(100, 0), scm.sample(100, 0))
    # different seed -> different data
    assert not scm.sample(100, 0).equals(scm.sample(100, 1))


def test_ancestral_correctness_and_noise_consistency():
    scm = _make_linear_scm()
    noise = scm.sample_noise(5000, 7)
    data = scm.sample(5000, 7)
    # X := U_X exactly
    np.testing.assert_allclose(data["X"].to_numpy(), noise["X"].to_numpy())
    # Y := 2X + U_Y, using the SAME seeded noise
    np.testing.assert_allclose(
        data["Y"].to_numpy(),
        2 * noise["X"].to_numpy() + noise["Y"].to_numpy(),
    )


def test_marginal_statistics():
    scm = _make_linear_scm()
    df = scm.sample(50000, 123)
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    assert abs(x.mean()) < 0.05
    assert abs(y.mean()) < 0.05
    assert abs(x.var() - 1.0) < 0.1
    assert abs(y.var() - 5.0) < 0.2  # Var(2X + U) = 4*1 + 1 = 5
    cov = np.cov(x, y)[0, 1]
    assert abs(cov - 2.0) < 0.1  # Cov(X, 2X + U) = 2*Var(X) = 2


def test_topological_order_valid():
    scm = _make_linear_scm()
    order = scm.nodes
    assert order.index("X") < order.index("Y")


# -- interventions ----------------------------------------------------------


def test_intervene_constant_and_downstream():
    scm = _make_linear_scm()
    intervened = scm.intervene("X", 3.0)
    df = intervened.sample(10000, 0)
    # X is held constant at 3.0
    np.testing.assert_allclose(df["X"].to_numpy(), 3.0)
    # Y = 2*3 + U_Y -> mean ~ 6.0
    assert abs(df["Y"].mean() - 6.0) < 0.1
    # incoming edges to X removed
    assert intervened.parents("X") == []
