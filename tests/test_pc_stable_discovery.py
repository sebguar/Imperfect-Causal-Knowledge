"""PC-Stable discovery end to end: BK, determinism, order-invariance, record.

BEHAVIOUR-FREEZING, NOT DISCOVERY-VALIDATION. Nothing in this file asserts that
PC recovered the true graph, and nothing here is tuned on what PC returned —
[the method-switch authority, amended] forbids calibrating the discoverer on production SCMs, and
[PS-7] makes non-oracle-matching a property of the call site, not of a
test that keeps re-running until the answer looks good. The one production-SCM
test below asserts SHAPE ONLY (a DAG over the true node set, every edge
channelled, the record serializable).

The toy fixture is HAND-MADE and lives outside the icknowledge SCM builders, in
the same spirit as tests/test_discovery_env.py: it keeps the constraint tests
(background knowledge, determinism, column-order invariance) away from anything
that could turn this file into a discovery benchmark.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from icknowledge.discovery import (
    CHANNELS,
    LIBRARY,
    PROTECTED,
    discover_and_synthesize,
    discover_graph,
)
from icknowledge.estimation.base import dataset_identity
from icknowledge.scm.triangles import make_linear_triangle

TOY_N = 800
TOY_SEED = 20260809


def toy_frame() -> pd.DataFrame:
    """Hand-drawn A -> X1 -> X2 with A -> X2, A centered ±1. NOT an icknowledge SCM.

    Extends the estimation toy chain with one A-like column so the PS-7
    background-knowledge path has something to constrain.
    """
    rng = np.random.default_rng(TOY_SEED)
    a = rng.choice([-1.0, 1.0], size=TOY_N)
    x1 = 2.0 * a + rng.normal(0.0, 1.0, TOY_N)
    x2 = 1.5 * x1 + 1.0 * a + rng.normal(0.0, 1.0, TOY_N)
    return pd.DataFrame({"A": a, "X1": x1, "X2": x2})


# -- background knowledge ----------------------------------------------------


def test_no_edge_points_into_A_and_A_edges_carry_channel_bk():
    """[PS-7] 'A is a root' is supplied as DOMAIN background knowledge."""
    graph, record = discover_graph(toy_frame())

    assert list(graph.predecessors(PROTECTED)) == []
    a_edges = [e for e in record.edges if e.source == PROTECTED]
    assert a_edges, "the toy DGP has A -> X edges; the search found none"
    assert {e.channel for e in a_edges} == {"bk"}
    assert all(e.channel in CHANNELS for e in record.edges)


def test_the_forbidden_list_in_the_record_names_every_non_A_column():
    _, record = discover_graph(toy_frame())

    forbidden = record.resolved_args["background_knowledge"]["forbidden"]
    assert sorted(forbidden) == [["X1", "A"], ["X2", "A"]]


def test_missing_protected_column_raises_rather_than_discovering_a_truncated_node_set():
    frame = toy_frame().drop(columns=[PROTECTED])
    with pytest.raises(ValueError, match="PS-7"):
        discover_graph(frame)


# -- the pinned call site ----------------------------------------------------


def test_every_pinned_argument_is_recorded_not_inherited():
    """[PS-7] Defaults are passed explicitly so the manifest records them."""
    _, record = discover_graph(toy_frame(), alpha=0.05)

    args = record.resolved_args
    assert args["alpha"] == 0.05
    assert args["indep_test"] == "fisherz"
    assert args["stable"] is True
    assert args["uc_rule"] == 0
    assert args["uc_priority"] == 2
    assert args["mvpc"] is False
    assert args["node_names"] == ["A", "X1", "X2"]
    assert args["verbose"] is False
    assert args["show_progress"] is False


def test_library_and_version_are_recorded():
    _, record = discover_graph(toy_frame())

    assert record.library == LIBRARY == "causal-learn"
    assert record.library_version.startswith("0.1.4")


def test_the_A_encoding_is_recorded_and_the_returned_graph_keeps_the_original_names():
    """The ±1 -> 0/1 map is for the CI test only; it never leaves `discover_graph`."""
    frame = toy_frame()
    graph, record = discover_graph(frame)

    assert "0/1" in record.encoding_note and "affine" in record.encoding_note
    assert set(graph.nodes()) == set(frame.columns)
    assert set(frame["A"].unique()) == {-1.0, 1.0}  # caller's frame untouched


def test_canonical_order_is_the_lexical_node_order():
    """[PS-7 note (g)]"""
    _, record = discover_graph(toy_frame())
    assert record.canonical_order == ("A", "X1", "X2")


# -- determinism -------------------------------------------------------------


def test_same_data_twice_gives_an_identical_graph_and_an_identical_record():
    graph_a, record_a = discover_graph(toy_frame())
    graph_b, record_b = discover_graph(toy_frame())

    assert set(graph_a.edges()) == set(graph_b.edges())
    assert set(graph_a.nodes()) == set(graph_b.nodes())
    assert record_a.as_dict() == record_b.as_dict()


# -- column-order invariance -------------------------------------------------


def test_permuting_the_input_columns_changes_nothing():
    """The Velev order-independence property PC-Stable + canonical reindexing buys.

    [PS-7] Vanilla PC is order-dependent: rearranging the variables in the
    passively observed dataset could yield a different estimated structure — a
    reproducibility hazard incompatible with this project's seed/manifest
    discipline. `stable=True` buys order-independence of the SKELETON; reindexing
    to the canonical order (note (g)) before the call is what makes the whole
    record, tie-break included, a function of the data alone.
    """
    frame = toy_frame()
    graph_a, record_a = discover_graph(frame)
    graph_b, record_b = discover_graph(frame[["X2", "A", "X1"]])

    assert set(graph_a.edges()) == set(graph_b.edges())
    assert record_a.as_dict() == record_b.as_dict()


# -- the record --------------------------------------------------------------


def test_the_record_is_json_serializable_and_internally_consistent():
    graph, record = discover_graph(toy_frame())
    payload = record.as_dict()

    json.loads(json.dumps(payload))  # no numpy scalars, no BackgroundKnowledge object

    order = payload["canonical_order"]
    index = {name: k for k, name in enumerate(order)}
    from_edges = {(e["source"], e["target"]) for e in payload["edges"]}
    from_matrix = {
        (order[i], order[j])
        for i, row in enumerate(payload["adjacency"])
        for j, cell in enumerate(row)
        if cell
    }
    assert from_edges == from_matrix == set(graph.edges())
    assert all(index[s] != index[t] for s, t in from_edges)
    assert payload["n_bidirected"] == sum(e["was_bidirected"] for e in payload["edges"])


def test_the_dataset_block_matches_the_estimation_blocks_shape_and_hash():
    """[the L1-oracle form-template-known semantics] Discovery and estimation must run on
    the SAME sample.

    Recording the identity in the same {n_rows, columns, sha256} shape is what
    lets the wiring layer assert discovery-hash == estimation-hash instead of trusting the
    wiring. Taken on the canonically-ordered frame, which is why the assertion
    below survives the column permutation.
    """
    frame = toy_frame()
    _, record = discover_graph(frame)
    _, permuted = discover_graph(frame[["X2", "X1", "A"]])

    expected = dataset_identity(frame.reindex(columns=sorted(frame.columns)))
    assert record.dataset == expected
    assert permuted.dataset == expected
    assert set(record.dataset) == {"n_rows", "columns", "sha256"}
    assert record.dataset["n_rows"] == TOY_N


# -- one production sample, MECHANICS ONLY -----------------------------------


def test_mechanics_on_one_production_sample_linear_triangle_one_seed():
    """Discovery half of the never-fired stop-loss: PC-Stable runs end to end on
    the linear triangle, CPDAG -> Meek -> tie-break -> oriented DAG.

    SHAPE ASSERTIONS ONLY. Recovery of the true graph is deliberately NOT
    asserted — the test target must stay truth-free and this test must stay
    behaviour-freezing — and nothing here is tuned on the outcome [the method-switch authority,
    amended]. The estimator/scoring half of the stop-loss lands in the wiring layer.
    """
    scm = make_linear_triangle(regime="additive")
    data = scm.sample(n=2000, seed=20260809)

    graph, record = discover_graph(data)

    assert set(graph.nodes()) == {"A", "X1", "X2"}
    assert nx.is_directed_acyclic_graph(graph)
    assert list(graph.predecessors("A")) == []
    assert all(e.channel in CHANNELS for e in record.edges)
    assert record.canonical_order == ("A", "X1", "X2")
    assert record.dataset == dataset_identity(data.reindex(columns=["A", "X1", "X2"]))
    json.loads(json.dumps(record.as_dict()))


# -- round trip into the synthesis layer --------------------------------------------------


@pytest.mark.parametrize("family", ["linear", "nlg"])
def test_discover_and_synthesize_round_trips_through_the_task2_structural_guard(family):
    """[PS-7] Every emitted term is single-factor: no A·X column, ever."""
    graph, form, record = discover_and_synthesize(toy_frame(), family)

    assert isinstance(graph, nx.DiGraph)
    assert record.canonical_order == ("A", "X1", "X2")
    assert set(form) <= set(graph.nodes())
    for node, node_form in form.items():
        assert set(node_form.parents) == set(graph.predecessors(node))
        assert node_form.interactions == ()
        assert all(len(term.factors) == 1 for term in node_form.terms)
    # Roots carry no fitted equation; A is a root by PS-7 background knowledge.
    assert PROTECTED not in form


def test_discover_and_synthesize_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown functional family"):
        discover_and_synthesize(toy_frame(), "quadratic")
