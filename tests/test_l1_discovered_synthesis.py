"""L1-discovered form synthesis: form-preserving basis, interaction-blindness, guards.

Unit-level only — graphs are HAND-DRAWN `nx.DiGraph`s, never PC output. This
file tests the TRANSLATION from a graph to a form template; running the
search that produces the graph is the discovery layer, and nothing here calibrates or judges a
discoverer.

Covers:
  1. synthesis on a WRONG-ORIENTATION graph (form follows the discovered edges,
     not the truth);
  2. synthesis on a DISCOVERED-ROOT graph (a parentless X is absent from the form);
  3. synthesis on a MATCHES-TRUTH graph (linear and NLG bases, per-parent);
  4. the interaction-blindness guard — no multi-factor term, no A·X column, no
     `interactions`, for every graph × family;
  5. the input guards (DiGraph, DAG, "A" present, A parentless, known family).
"""

import networkx as nx
import pytest

from icknowledge.discovery import synthesize_form_spec
from icknowledge.estimation.forms import IDENTITY, TANH, term_names

FAMILIES = ["linear", "nlg"]


def _graph(edges: list[tuple[str, str]], nodes: list[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def _true_triangle() -> nx.DiGraph:
    """A->X1, A->X2, X1->X2 — the ground-truth triangle (scm/triangles.py)."""
    return _graph([("A", "X1"), ("A", "X2"), ("X1", "X2")], ["A", "X1", "X2"])


def _reversed_triangle() -> nx.DiGraph:
    """Triangle with the X1->X2 edge REVERSED: a plausible tie-break error."""
    return _graph([("A", "X1"), ("A", "X2"), ("X2", "X1")], ["A", "X1", "X2"])


def _orphaned_x1() -> nx.DiGraph:
    """X1 left PARENTLESS by the search; X2 keeps both its parents."""
    return _graph([("A", "X2"), ("X1", "X2")], ["A", "X1", "X2"])


ALL_GRAPHS = {
    "true-triangle": _true_triangle,
    "reversed": _reversed_triangle,
    "orphaned-x1": _orphaned_x1,
}


# -- 1. wrong orientation ----------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_synthesis_follows_the_discovered_orientation_not_the_truth(family):
    """On a reversed X1<->X2 edge the form is built for the DISCOVERED parent sets.

    This is the whole point of the rung: L1-discovered inherits the search's
    mistakes. X1 gains X2 as a parent, X2 loses X1 — the exact mirror of the truth.
    """
    form = synthesize_form_spec(_reversed_triangle(), family)

    assert set(form) == {"X1", "X2"}
    assert form["X1"].parents == ("A", "X2")
    assert form["X2"].parents == ("A",)
    # And the basis follows the parent's identity, per node (PS-7 note (a)).
    expected_x1 = ["A", "X2"] if family == "linear" else ["A", "tanh(X2)"]
    assert term_names(form["X1"]) == expected_x1
    assert term_names(form["X2"]) == ["A"]


# -- 2. discovered roots -----------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_parentless_x_node_is_absent_from_the_form(family):
    """A node the search left parentless carries NO fitted equation.

    Mirrors the oracle form-spec contract (roots absent). Its mechanism is the
    PS-7-note-(b) marginal, built at assembly — not a zero-column form here.
    """
    form = synthesize_form_spec(_orphaned_x1(), family)

    assert set(form) == {"X2"}, "only X2 has discovered parents"
    assert "X1" not in form
    assert "A" not in form, "A is parentless by PS-7 and is never fitted"
    assert form["X2"].parents == ("A", "X1")


# -- 3. matches-truth graph, both bases --------------------------------------


def test_matches_truth_graph_linear_basis():
    """Linear: every discovered parent enters as an identity main effect."""
    form = synthesize_form_spec(_true_triangle(), "linear")

    assert set(form) == {"X1", "X2"}
    assert form["X1"].parents == ("A",)
    assert form["X2"].parents == ("A", "X1")
    assert term_names(form["X1"]) == ["A"]
    assert term_names(form["X2"]) == ["A", "X1"]
    for node_form in form.values():
        for term in node_form.terms:
            assert all(f.transform == IDENTITY for f in term.factors)


def test_matches_truth_graph_nlg_basis_is_per_parent():
    """NLG: A enters as identity (oracle `a_main`), every X-parent under tanh.

    This is the form-preserving estimation basis applied PER PARENT — the same column
    the oracle NLG templates carry on the same edge (`nlg_triangle_form_spec`).
    """
    form = synthesize_form_spec(_true_triangle(), "nlg")

    assert term_names(form["X1"]) == ["A"]
    assert term_names(form["X2"]) == ["A", "tanh(X1)"]
    (a_factor,) = form["X2"].terms[0].factors
    (x1_factor,) = form["X2"].terms[1].factors
    assert (a_factor.parent, a_factor.transform) == ("A", IDENTITY)
    assert (x1_factor.parent, x1_factor.transform) == ("X1", TANH)


def test_parent_order_is_sorted_and_deterministic():
    """Column order is a function of the graph's node NAMES, not of nx insertion order."""
    forward = _graph([("A", "X2"), ("X1", "X2")], ["A", "X1", "X2"])
    shuffled = _graph([("X1", "X2"), ("A", "X2")], ["X2", "X1", "A"])

    assert synthesize_form_spec(forward, "nlg")["X2"].parents == ("A", "X1")
    assert synthesize_form_spec(shuffled, "nlg")["X2"].parents == ("A", "X1")
    assert (
        term_names(synthesize_form_spec(forward, "nlg")["X2"])
        == term_names(synthesize_form_spec(shuffled, "nlg")["X2"])
    )


# -- 4. Interaction-blindness: no A·X column, ever ---------------------------


@pytest.mark.parametrize("graph_name", sorted(ALL_GRAPHS))
@pytest.mark.parametrize("family", FAMILIES)
def test_no_interaction_column_for_any_graph_or_family(graph_name, family):
    """[PS-7] The synthesized form is interaction-free, unconditionally.

    Regime never enters this function, so there is no arm in which an A·X column
    could appear: every emitted term is SINGLE-FACTOR and `interactions` is empty.
    An A·X column here would oracle-inject the effect-modification hypothesis into
    the rung that is supposed to have discovered its own structure.
    """
    form = synthesize_form_spec(ALL_GRAPHS[graph_name](), family)

    for node, node_form in form.items():
        assert node_form.interactions == (), f"{node}: interactions must stay empty"
        assert node_form.terms is not None
        for term in node_form.terms:
            assert len(term.factors) == 1, f"{node}: multi-factor term {term}"
        names = term_names(node_form)
        assert not any("*" in name for name in names), f"{node}: product column in {names}"
        assert len(set(names)) == len(names), f"{node}: duplicate column in {names}"


def test_synthesis_takes_no_regime_argument():
    """Interaction-blindness is structural: there is no regime knob to turn it on."""
    import inspect

    params = inspect.signature(synthesize_form_spec).parameters
    assert list(params) == ["discovered_graph", "family"]


# -- 5. input guards ---------------------------------------------------------


def test_rejects_a_non_digraph():
    with pytest.raises(TypeError, match="nx.DiGraph"):
        synthesize_form_spec(nx.Graph([("A", "X1")]), "linear")


def test_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown functional family"):
        synthesize_form_spec(_true_triangle(), "mlp")


def test_rejects_a_cyclic_graph_pointing_at_task_3():
    """An unresolved orientation from CPDAG -> Meek -> tie-break surfaces as a cycle."""
    cyclic = _graph([("A", "X1"), ("X1", "X2"), ("X2", "X1")], ["A", "X1", "X2"])
    with pytest.raises(ValueError, match="not a DAG"):
        synthesize_form_spec(cyclic, "linear")


def test_rejects_a_graph_without_the_protected_node():
    with pytest.raises(ValueError, match="no 'A' node"):
        synthesize_form_spec(_graph([("X1", "X2")], ["X1", "X2"]), "linear")


@pytest.mark.parametrize("family", FAMILIES)
def test_rejects_a_graph_in_which_a_has_parents(family):
    """[PS-7] A's exogeneity is BACKGROUND KNOWLEDGE, so this is a discovery-stage bug."""
    bad = _graph([("X1", "A"), ("A", "X2")], ["A", "X1", "X2"])
    with pytest.raises(ValueError, match="PS-7"):
        synthesize_form_spec(bad, family)
