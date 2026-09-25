"""Tie-break CPDAG → DAG completion, ISOLATED from PC.

Every endpoint matrix below is HAND-CRAFTED and no search runs in this file, so
the completion rule — channel attribution, bidirected demotion, single-pass
semantics, the acyclicity assertion — is tested as a deterministic function.

Matrix encoding, verified against the installed causal-learn 0.1.4.8 source (see
`pc_stable`'s module docstring):
    graph[i,j] = -1, graph[j,i] =  1  ->  i -> j
    graph[i,j] = -1, graph[j,i] = -1  ->  i -- j
    graph[i,j] =  1, graph[j,i] =  1  ->  i <-> j
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from icknowledge.discovery.pc_stable import _classify_adjacency, _orient_cpdag


def _matrix(names: list[str]) -> np.ndarray:
    return np.zeros((len(names), len(names)), dtype=int)


def _directed(m: np.ndarray, names: list[str], src: str, tgt: str) -> None:
    i, j = names.index(src), names.index(tgt)
    m[i, j], m[j, i] = -1, 1


def _undirected(m: np.ndarray, names: list[str], a: str, b: str) -> None:
    i, j = names.index(a), names.index(b)
    m[i, j], m[j, i] = -1, -1


def _bidirected(m: np.ndarray, names: list[str], a: str, b: str) -> None:
    i, j = names.index(a), names.index(b)
    m[i, j], m[j, i] = 1, 1


def _channels(edges) -> dict[tuple[str, str], str]:
    return {(e.source, e.target): e.channel for e in edges}


def _flags(edges) -> dict[tuple[str, str], bool]:
    return {(e.source, e.target): e.was_bidirected for e in edges}


# -- classification ----------------------------------------------------------


@pytest.mark.parametrize(
    ("e_ij", "e_ji", "expected"),
    [
        (0, 0, "none"),
        (-1, 1, "i->j"),
        (1, -1, "j->i"),
        (-1, -1, "undirected"),
        (1, 1, "bidirected"),
    ],
)
def test_endpoint_pairs_classify_as_the_verified_encoding_says(e_ij, e_ji, expected):
    assert _classify_adjacency(e_ij, e_ji, "X1", "X2") == expected


@pytest.mark.parametrize(("e_ij", "e_ji"), [(2, 2), (2, 1), (4, 4), (5, 5), (-1, 0)])
def test_endpoint_pairs_outside_the_cpdag_alphabet_raise(e_ij, e_ji):
    """CIRCLE / TAIL_AND_ARROW / ARROW_AND_ARROW are FCI shapes, not guessed at."""
    with pytest.raises(ValueError, match="unrecognized endpoint pair"):
        _classify_adjacency(e_ij, e_ji, "X1", "X2")


# -- channel attribution -----------------------------------------------------


def test_ci_directed_x_edges_carry_channel_ci():
    """A directed X -> X edge came from a v-structure or a Meek propagation."""
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    _directed(m, names, "X1", "X3")
    _directed(m, names, "X2", "X3")

    graph, edges, n_bidirected = _orient_cpdag(m, names)

    assert n_bidirected == 0
    assert _channels(edges) == {("X1", "X3"): "ci", ("X2", "X3"): "ci"}
    assert set(graph.edges()) == {("X1", "X3"), ("X2", "X3")}


def test_every_directed_edge_out_of_A_is_attributed_to_bk_not_ci():
    """[PS-7 note (e)] BK makes the reverse impossible, so the direction is
    BK-attributable by construction; we do not try to tell 'PC oriented it' from
    'BK forced it' internally."""
    names = ["A", "X1", "X2"]
    m = _matrix(names)
    _directed(m, names, "A", "X1")
    _directed(m, names, "A", "X2")
    _undirected(m, names, "X1", "X2")

    _, edges, _ = _orient_cpdag(m, names)

    channels = _channels(edges)
    assert channels[("A", "X1")] == "bk"
    assert channels[("A", "X2")] == "bk"
    # ...and the tie-break carries ONLY the X — X edge (note (e)).
    assert channels[("X1", "X2")] == "tiebreak"
    assert sum(c == "tiebreak" for c in channels.values()) == 1


def test_an_undirected_A_edge_is_bk_oriented_never_tiebroken():
    """Defensive BK layer: should never fire in practice, must never mislabel."""
    names = ["A", "X1"]
    m = _matrix(names)
    _undirected(m, names, "A", "X1")

    graph, edges, _ = _orient_cpdag(m, names)

    assert set(graph.edges()) == {("A", "X1")}
    assert _channels(edges) == {("A", "X1"): "bk"}


def test_an_edge_pointing_into_A_raises_as_a_background_knowledge_bug():
    names = ["A", "X1"]
    m = _matrix(names)
    _directed(m, names, "X1", "A")

    with pytest.raises(ValueError, match="background-knowledge"):
        _orient_cpdag(m, names)


# -- bidirected demotion (note (d)) ------------------------------------------


def test_bidirected_edge_falls_through_to_the_tiebreak_with_the_flag_set():
    """[PS-7 note (d)] i <-> j is finite-sample ambiguity, not a
    latent confounder: demoted to undirected, tie-broken, flagged, counted."""
    names = ["A", "X1", "X2"]
    m = _matrix(names)
    _directed(m, names, "A", "X1")
    _bidirected(m, names, "X1", "X2")

    graph, edges, n_bidirected = _orient_cpdag(m, names)

    assert n_bidirected == 1
    assert set(graph.edges()) == {("A", "X1"), ("X1", "X2")}
    assert _channels(edges)[("X1", "X2")] == "tiebreak"
    assert _flags(edges)[("X1", "X2")] is True
    assert _flags(edges)[("A", "X1")] is False


def test_bidirected_count_is_per_adjacency():
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    _bidirected(m, names, "X1", "X2")
    _bidirected(m, names, "X2", "X3")

    _, edges, n_bidirected = _orient_cpdag(m, names)

    assert n_bidirected == 2
    assert sum(e.was_bidirected for e in edges) == 2


# -- single-pass semantics ---------------------------------------------------


def test_tiebreak_is_single_pass_an_iterative_rule_would_differ_here():
    """A case where re-running Meek BETWEEN tie-break orientations changes the answer.

    Skeleton: X1 — X3 and X2 — X3, with X1 and X2 NON-adjacent (an unshielded
    triple), nothing oriented. The pre-committed SINGLE-PASS rule sweeps the
    canonical pairs and orients X1 -> X3 and X2 -> X3, manufacturing a collider.
    An ITERATIVE rule would instead orient X1 -> X3, then let Meek R1 fire on the
    unshielded triple (X1, X3, X2) and force X3 -> X2 — the opposite direction on
    the second edge. [PS-7] The single-pass answer is the registered
    one; adopting the iterative rule now would be an unregistered rule change.
    """
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    _undirected(m, names, "X1", "X3")
    _undirected(m, names, "X2", "X3")

    graph, edges, _ = _orient_cpdag(m, names)

    assert set(graph.edges()) == {("X1", "X3"), ("X2", "X3")}
    assert ("X3", "X2") not in graph.edges()  # what an iterative rule would give
    assert all(e.channel == "tiebreak" for e in edges)


def test_tiebreak_always_runs_lower_to_higher_in_canonical_order():
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    _undirected(m, names, "X3", "X1")
    _undirected(m, names, "X3", "X2")
    _undirected(m, names, "X2", "X1")

    graph, _, _ = _orient_cpdag(m, names)

    assert set(graph.edges()) == {("X1", "X3"), ("X2", "X3"), ("X1", "X2")}


def test_note_f_the_tiebreak_reproduces_the_true_chain_and_triangle_orientations():
    """[PS-7 note (f)] Pre-stated structural property of the rule-graph pair.

    On a CORRECTLY-discovered chain (X1 -> X2 -> X3) or triangle (X1 -> X2)
    skeleton, the canonical-order tie-break returns the TRUE X — X orientations.
    Asserted here so the property is a frozen behavioural fact of the harness
    rather than an observation made later, next to a clean linear control arm.
    """
    chain_names = ["A", "X1", "X2", "X3"]
    chain = _matrix(chain_names)
    for x in ("X1", "X2", "X3"):
        _directed(chain, chain_names, "A", x)
    _undirected(chain, chain_names, "X1", "X2")
    _undirected(chain, chain_names, "X2", "X3")
    chain_graph, _, _ = _orient_cpdag(chain, chain_names)
    assert ("X1", "X2") in chain_graph.edges()
    assert ("X2", "X3") in chain_graph.edges()

    tri_names = ["A", "X1", "X2"]
    tri = _matrix(tri_names)
    _directed(tri, tri_names, "A", "X1")
    _directed(tri, tri_names, "A", "X2")
    _undirected(tri, tri_names, "X1", "X2")
    tri_graph, _, _ = _orient_cpdag(tri, tri_names)
    assert ("X1", "X2") in tri_graph.edges()


# -- acyclicity assertion ----------------------------------------------------


def test_a_cycle_raises_the_task3_message_naming_the_cycle_and_the_tiebreak_edges():
    """CI/BK edge running higher -> lower, closed into a loop by the tie-break.

    X3 -> X1 arrives directed; X1 — X2 and X2 — X3 are tie-broken lower -> higher,
    closing X1 -> X2 -> X3 -> X1. The failure must be named HERE, with PS-7
    cited, not left to surface as a synthesis error.
    """
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    _directed(m, names, "X3", "X1")
    _undirected(m, names, "X1", "X2")
    _undirected(m, names, "X2", "X3")

    with pytest.raises(ValueError) as excinfo:
        _orient_cpdag(m, names)

    message = str(excinfo.value)
    assert "CPDAG → Meek → tie-break" in message
    assert "PS-7" in message
    assert "X1" in message and "X2" in message and "X3" in message
    assert "('X1', 'X2')" in message or "('X2', 'X3')" in message


def test_a_pure_tiebreak_graph_can_never_be_cyclic():
    """Sanity: lower -> higher on every edge is a topological order by construction."""
    names = ["A", "X1", "X2", "X3"]
    m = _matrix(names)
    for a, b in [("X1", "X2"), ("X2", "X3"), ("X1", "X3")]:
        _undirected(m, names, a, b)

    graph, _, _ = _orient_cpdag(m, names)

    assert nx.is_directed_acyclic_graph(graph)


# -- shape guards ------------------------------------------------------------


def test_endpoint_matrix_shape_must_match_the_canonical_order():
    with pytest.raises(ValueError, match="does not match the canonical order"):
        _orient_cpdag(np.zeros((2, 2), dtype=int), ["A", "X1", "X2"])


def test_isolated_nodes_are_kept_in_the_graph():
    """A node PC left with no adjacency is still a node — the synthesis layer gives it the
    PS-7-note-(b) truth-free marginal, and it cannot do that for a node that
    silently vanished."""
    names = ["A", "X1", "X2"]
    m = _matrix(names)
    _directed(m, names, "A", "X1")

    graph, _, _ = _orient_cpdag(m, names)

    assert set(graph.nodes()) == {"A", "X1", "X2"}
    assert list(graph.predecessors("X2")) == []
