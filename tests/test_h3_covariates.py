"""H3 covariate module: SID, skeleton-SHD, the orientation/provenance split.

The adjacency + provenance fixtures are hand-crafted, NOT read off the grid, so
each covariate is asserted against a case whose right answer was worked out on
paper before the code ran.
"""

from __future__ import annotations

import networkx as nx
import pytest

from icknowledge.analysis.h3_covariates import (
    D_EFF,
    covariates,
    discovered_from_manifest,
    sid,
    true_dag,
)


def _graph(nodes, edges) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


# --------------------------------------------------------------------------- #
# SID
# --------------------------------------------------------------------------- #


def test_sid_identical_graphs_is_zero():
    """The parent adjustment of the TRUE graph is valid for every ordered pair."""
    for graph in (
        _graph(["1", "2"], [("1", "2")]),
        _graph(["1", "2", "3"], [("1", "2"), ("2", "3")]),
        _graph(["1", "2", "3"], [("1", "3"), ("2", "3")]),
    ):
        assert sid(graph, graph) == 0


def test_sid_two_node_reversal_is_two():
    """G: 1->2 vs H: 2->1, both ordered pairs wrong (hand-computed).

    (1,2): H makes 2 a parent of 1, asserting no effect of 1 on 2 — but 1 IS an
    ancestor of 2 in G, so the pair is a mistake.
    (2,1): H gives 2 an empty adjustment set; in G the single edge 1->2 is an
    unblockable back-door path from 2 to 1, so the empty set is invalid.
    """
    true_graph = _graph(["1", "2"], [("1", "2")])
    reversed_graph = _graph(["1", "2"], [("2", "1")])
    assert sid(true_graph, reversed_graph) == 2


def test_sid_three_node_chain_vs_reversed_chain():
    """G: 1->2->3 vs H: 3->2->1 — all six ordered pairs wrong (hand-computed).

    (1,2) and (2,3): H puts the outcome in the adjustment set while the true
    graph has a directed path, so both are mistakes.
    (1,3): H adjusts for {2}, a descendant of 1 in G — the mediator is blocked.
    (2,1): H adjusts for {3}, a descendant of 2 in G.
    (3,1) and (3,2): H gives 3 an empty set; in G the paths 3<-2<-1 and 3<-2 are
    unblockable back-door paths.
    """
    true_graph = _graph(["1", "2", "3"], [("1", "2"), ("2", "3")])
    reversed_graph = _graph(["1", "2", "3"], [("3", "2"), ("2", "1")])
    assert sid(true_graph, reversed_graph) == 6


def test_sid_is_asymmetric_on_a_spurious_edge():
    """A spurious 1->3 on top of the true chain costs nothing; dropping one does.

    est = chain + 1->3, true = chain: every parent set of `est` still satisfies
    the back-door criterion in the chain, so SID = 0 — SID scores adjustment
    validity, not edge equality.
    Reversing the roles, `est` = chain misses the true 1->3 edge, leaving node 3
    with the adjustment set {2}; the pair (3,1) then has the unblockable
    back-door path 3<-1, one mistake.
    """
    chain = _graph(["1", "2", "3"], [("1", "2"), ("2", "3")])
    chain_plus = _graph(["1", "2", "3"], [("1", "2"), ("2", "3"), ("1", "3")])
    assert sid(chain, chain_plus) == 0
    assert sid(chain_plus, chain) == 1


def test_sid_rejects_a_mismatched_node_set():
    with pytest.raises(ValueError, match="shared node set"):
        sid(_graph(["1", "2"], []), _graph(["1", "2", "3"], []))


# --------------------------------------------------------------------------- #
# True graphs and the endogenous node set
# --------------------------------------------------------------------------- #


def test_true_dags_match_the_frozen_builders():
    assert set(true_dag("triangle").edges()) == {("A", "X1"), ("A", "X2"), ("X1", "X2")}
    assert set(true_dag("chain").edges()) == {
        ("A", "X1"),
        ("A", "X2"),
        ("A", "X3"),
        ("X1", "X2"),
        ("X2", "X3"),
    }
    assert set(true_dag("collider").edges()) == {
        ("A", "X1"),
        ("A", "X2"),
        ("A", "X3"),
        ("X1", "X3"),
        ("X2", "X3"),
    }


def test_d_eff_excludes_the_protected_root():
    """[PS-8] SID is normalized on the ENDOGENOUS sub-graph, A excluded."""
    assert D_EFF == {"triangle": 2, "collider": 3, "chain": 3}


# --------------------------------------------------------------------------- #
# Covariate fixtures — hand-crafted adjacency + provenance
# --------------------------------------------------------------------------- #

_COLLIDER_ORDER = ["A", "X1", "X2", "X3"]


def _manifest(order, edges):
    """A minimal manifest discovery block with a consistent adjacency matrix."""
    index = {name: i for i, name in enumerate(order)}
    adjacency = [[0] * len(order) for _ in order]
    for edge in edges:
        adjacency[index[edge["source"]]][index[edge["target"]]] = 1
    return {
        "git_commit": "deadbeef",
        "discovery": {
            "additive": {
                "canonical_order": order,
                "edges": edges,
                "adjacency": adjacency,
                "library": "causal-learn",
                "library_version": "0.1.4.8",
                "n_bidirected": sum(e["was_bidirected"] for e in edges),
                "resolved_args": {"indep_test": "fisherz", "alpha": 0.05, "stable": True},
            }
        },
    }


def _edge(source, target, channel, was_bidirected=False):
    return {
        "source": source,
        "target": target,
        "channel": channel,
        "was_bidirected": was_bidirected,
    }


def _collider_covariates(edges):
    manifest = _manifest(_COLLIDER_ORDER, edges)
    return covariates("collider", discovered_from_manifest(manifest, "additive"))


_PERFECT_COLLIDER = [
    _edge("A", "X1", "bk"),
    _edge("A", "X2", "bk"),
    _edge("A", "X3", "bk"),
    _edge("X1", "X3", "ci"),
    _edge("X2", "X3", "ci"),
]


def test_perfect_recovery():
    out = _collider_covariates(_PERFECT_COLLIDER)
    assert out["skeleton_shd"] == 0
    assert out["orientation_wrong_count"] == 0
    assert out["tiebreak_resolved_count"] == 0
    assert out["wrong_and_tiebreak_count"] == 0
    assert out["was_bidirected_count"] == 0
    assert out["sid"] == 0
    assert out["n_sid"] == 0.0


def test_missing_edge_counts_one_skeleton_error():
    edges = [e for e in _PERFECT_COLLIDER if (e["source"], e["target"]) != ("X2", "X3")]
    out = _collider_covariates(edges)
    assert out["skeleton_shd"] == 1
    assert out["orientation_wrong_count"] == 0


def test_spurious_edge_counts_one_skeleton_error_not_an_orientation_error():
    """A spurious X1–X2 adjacency is an ADJACENCY error, never an R-term error."""
    out = _collider_covariates([*_PERFECT_COLLIDER, _edge("X1", "X2", "tiebreak")])
    assert out["skeleton_shd"] == 1
    assert out["orientation_wrong_count"] == 0
    assert out["tiebreak_resolved_count"] == 1
    assert out["wrong_and_tiebreak_count"] == 0


def test_reversed_ci_determined_edge():
    """Correct adjacency, wrong direction, CI channel -> R term, not tie-break."""
    edges = [
        e for e in _PERFECT_COLLIDER if (e["source"], e["target"]) != ("X2", "X3")
    ] + [_edge("X3", "X2", "ci")]
    out = _collider_covariates(edges)
    assert out["skeleton_shd"] == 0
    assert out["orientation_wrong_count"] == 1
    assert out["wrong_ci_orientation_count"] == 1
    assert out["tiebreak_resolved_count"] == 0
    assert out["wrong_and_tiebreak_count"] == 0


def test_wrong_tiebreak_edge_populates_the_class_b_cell():
    edges = [
        e for e in _PERFECT_COLLIDER if (e["source"], e["target"]) != ("X2", "X3")
    ] + [_edge("X3", "X2", "tiebreak")]
    out = _collider_covariates(edges)
    assert out["skeleton_shd"] == 0
    assert out["orientation_wrong_count"] == 1
    assert out["tiebreak_resolved_count"] == 1
    assert out["wrong_and_tiebreak_count"] == 1
    assert out["wrong_ci_orientation_count"] == 0


def test_bidirected_then_tiebreak_counts_in_both_columns():
    """[PS-7 note (d)] bidirected edges are tie-break-resolved."""
    edges = [
        e for e in _PERFECT_COLLIDER if (e["source"], e["target"]) != ("X1", "X3")
    ] + [_edge("X1", "X3", "tiebreak", was_bidirected=True)]
    out = _collider_covariates(edges)
    assert out["skeleton_shd"] == 0
    assert out["orientation_wrong_count"] == 0
    assert out["tiebreak_resolved_count"] == 1
    assert out["was_bidirected_count"] == 1


def test_a_x_adjacency_error_counts_in_skeleton_but_not_in_orientation():
    """BK constrains A-edge ORIENTATION only — a missing A–X2 is a real error."""
    edges = [e for e in _PERFECT_COLLIDER if (e["source"], e["target"]) != ("A", "X2")]
    out = _collider_covariates(edges)
    assert out["skeleton_shd"] == 1
    assert out["orientation_wrong_count"] == 0
    assert out["tiebreak_resolved_count"] == 0
    # The endogenous sub-graph is untouched, so SID sees nothing.
    assert out["sid"] == 0


def test_structural_expectation_flags_a_chain_tiebreak_shortfall():
    """[PS-8] shd = 0 on a chain but an X–X edge NOT tie-break-resolved."""
    order = ["A", "X1", "X2", "X3"]
    clean = [
        _edge("A", "X1", "bk"),
        _edge("A", "X2", "bk"),
        _edge("A", "X3", "bk"),
        _edge("X1", "X2", "tiebreak"),
        _edge("X2", "X3", "tiebreak"),
    ]
    out = covariates(
        "chain", discovered_from_manifest(_manifest(order, clean), "additive")
    )
    assert out["structural_expectation_applies"] is True
    assert out["structural_expectation_violated"] is False

    ci_resolved = [*clean[:4], _edge("X2", "X3", "ci")]
    out = covariates(
        "chain", discovered_from_manifest(_manifest(order, ci_resolved), "additive")
    )
    assert out["tiebreak_resolved_count"] == 1
    assert out["n_xx_true_edges"] == 2
    assert out["structural_expectation_violated"] is True


def test_structural_expectation_does_not_apply_when_the_skeleton_is_wrong():
    order = ["A", "X1", "X2", "X3"]
    edges = [
        _edge("A", "X1", "bk"),
        _edge("A", "X3", "bk"),
        _edge("X1", "X2", "tiebreak"),
        _edge("X2", "X3", "tiebreak"),
    ]
    out = covariates(
        "chain", discovered_from_manifest(_manifest(order, edges), "additive")
    )
    assert out["skeleton_shd"] == 1
    assert out["structural_expectation_applies"] is False
    assert out["structural_expectation_violated"] is False


def test_manifest_adjacency_edge_list_disagreement_is_fatal():
    manifest = _manifest(_COLLIDER_ORDER, _PERFECT_COLLIDER)
    manifest["discovery"]["additive"]["adjacency"][1][2] = 1  # X1 -> X2, unlisted
    with pytest.raises(AssertionError, match="adjacency disagrees"):
        discovered_from_manifest(manifest, "additive")
