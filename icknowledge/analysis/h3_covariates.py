"""H3 diagnostic covariates: discovered graph vs. truth, read off frozen manifests.

Analysis-side only: every number is derived from the persisted
``results/cross_seed_L1d_N20/<cell>_seed_<nn>/manifest.json`` provenance block
(``discovery/<regime>/{edges, adjacency, canonical_order}``) against the true DAG
obtained programmatically from the frozen builders. The covariates are PS-8's:

``skeleton_shd`` — extra plus missing adjacencies on the FULL node set including
A. BK constrains the ORIENTATION of A-edges only (PS-7 note (e)), so a wrong A–X
adjacency is a real discovery error.
``orientation_wrong_count`` — correct adjacency, discovered direction != truth,
X–X edges only, marginalized over channel: the strict orientation-error term. A→X edges are
BK-oriented and excluded.
``tiebreak_resolved_count`` — edges carrying ``channel == "tiebreak"`` (PS-7 note
(d)), marginalized over correctness: a provenance count, not an error count.
``wrong_and_tiebreak_count`` — their intersection, the class-(b) cell.
``was_bidirected_count`` — edges the CPDAG returned bidirected before resolution.
``sid`` / ``n_sid`` — SID on the ENDOGENOUS sub-graph {X_i} EXCLUDING A;
``n_sid = sid / (d_eff * (d_eff - 1))``, d_eff = 3 (collider, chain), 2
(triangle). Descriptive: enters no gate, band or halt.

Pre-stated structural expectation (recorded before any covariate was computed): on chain/triangle
cells with skeleton_shd == 0, tiebreak_resolved_count = true X–X edges, orientation_wrong_count = 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from icknowledge.analysis.loading import (
    _BUILDERS,
    L1D_ROOT,
    N_SEEDS_FULL,
    grid_cells,
)

#: The protected root. Excluded from the orientation and SID node sets (its edges
#: are BK-oriented, never discovered), included in the skeleton node set.
PROTECTED = "A"

#: [PS-8] Endogenous node count per topology, used as the
#: SID normalizer. Derived from the frozen builders at import time rather than
#: restated, so a builder change cannot leave a stale constant here.
D_EFF: dict[str, int] = {}

#: Channel value carried by manifest edges resolved by the deterministic tie-break.
TIEBREAK_CHANNEL = "tiebreak"


def true_dag(topology: str) -> nx.DiGraph:
    """The frozen ground-truth DAG for ``topology``, read off the builders.

    The graph is family- and regime-invariant by construction; this asserts it
    rather than trusting it, so a builder edit that made the truth depend on the
    functional family would fail here instead of silently splitting the covariate.
    """
    graphs = [
        _BUILDERS[(topology, family)](regime).graph
        for family in ("linear", "nlg")
        for regime in ("additive", "effect_modifying")
    ]
    reference = graphs[0]
    for other in graphs[1:]:
        if set(other.edges()) != set(reference.edges()):
            raise AssertionError(
                f"true DAG for {topology} is not family/regime-invariant: "
                f"{sorted(reference.edges())} vs {sorted(other.edges())}"
            )
    return nx.DiGraph(sorted(reference.edges()))


for _topology in ("triangle", "collider", "chain"):
    D_EFF[_topology] = len([n for n in true_dag(_topology).nodes() if n != PROTECTED])
del _topology


# --------------------------------------------------------------------------- #
# SID
# --------------------------------------------------------------------------- #


def _backdoor_valid(graph: nx.DiGraph, i: str, j: str, adjust: set[str]) -> bool:
    """Is ``adjust`` a valid back-door adjustment set for (i -> j) in ``graph``?

    The textbook criterion: no member of ``adjust`` is a descendant of ``i``, and
    ``adjust`` blocks every path from ``i`` to ``j`` carrying an arrow INTO ``i``
    (equivalently: d-separation in the graph with ``i``'s out-edges removed).
    """
    descendants = nx.descendants(graph, i) | {i}
    if adjust & descendants:
        return False
    lower = graph.copy()
    lower.remove_edges_from(list(graph.out_edges(i)))
    return nx.is_d_separator(lower, {i}, {j}, set(adjust))


def sid(true_graph: nx.DiGraph, est_graph: nx.DiGraph) -> int:
    """Structural Intervention Distance of ``est_graph`` w.r.t. ``true_graph``.

    Counts the ordered pairs (i, j), i != j, for which the parent-adjustment set
    of ``i`` taken from the ESTIMATED graph fails to deliver p(x_j | do(x_i)) in
    the TRUE graph. Two cases:

    * ``j`` in ``PA_i^H`` — the adjustment formula is not applicable; the
      estimate asserts "no effect of i on j", a mistake exactly when
      ``j`` in ``DE_i^G`` (j is a descendant of i in the true graph).
    * otherwise — a mistake unless ``PA_i^H`` satisfies the back-door-type
      condition (*) for (G, i, j).

    Both limbs are needed for ``SID(G, G) == 0``: the parents of ``i`` in the
    TRUE graph block every back-door path, and a true parent ``j`` of ``i`` is
    never a descendant of ``i``.

    Exact enumeration over all ordered pairs; d_eff <= 3 here, so the cost is
    trivial and no approximation is taken.
    """
    nodes = sorted(true_graph.nodes())
    if set(nodes) != set(est_graph.nodes()):
        raise ValueError(
            f"SID needs a shared node set; got {nodes} vs {sorted(est_graph.nodes())}"
        )
    mistakes = 0
    for i in nodes:
        adjust = set(est_graph.predecessors(i))
        true_descendants = nx.descendants(true_graph, i)
        for j in nodes:
            if j == i:
                continue
            if j in adjust:
                mistakes += int(j in true_descendants)
            else:
                mistakes += int(not _backdoor_valid(true_graph, i, j, adjust))
    return mistakes


# --------------------------------------------------------------------------- #
# Manifest provenance -> discovered graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscoveredGraph:
    """One (cell, seed)'s discovered graph plus the per-edge provenance."""

    graph: nx.DiGraph
    #: (source, target) -> channel, verbatim from the manifest.
    channel: dict[tuple[str, str], str]
    #: (source, target) -> was_bidirected, verbatim from the manifest.
    was_bidirected: dict[tuple[str, str], bool]
    canonical_order: tuple[str, ...]
    provenance: dict[str, object]


def discovered_from_manifest(manifest: dict, regime: str) -> DiscoveredGraph:
    """Read the discovered DAG + provenance out of a frozen manifest block."""
    block = manifest["discovery"][regime]
    order = tuple(block["canonical_order"])
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    channel: dict[tuple[str, str], str] = {}
    bidirected: dict[tuple[str, str], bool] = {}
    for edge in block["edges"]:
        key = (edge["source"], edge["target"])
        graph.add_edge(*key)
        channel[key] = edge["channel"]
        bidirected[key] = bool(edge["was_bidirected"])

    # The adjacency matrix and the edge list are two views of one graph; a
    # disagreement means the manifest is internally inconsistent and no covariate
    # derived from it would be trustworthy.
    adjacency = block["adjacency"]
    from_matrix = {
        (order[r], order[c])
        for r, row in enumerate(adjacency)
        for c, value in enumerate(row)
        if value
    }
    if from_matrix != set(graph.edges()):
        raise AssertionError(
            f"manifest adjacency disagrees with its edge list: "
            f"{sorted(from_matrix)} vs {sorted(graph.edges())}"
        )

    return DiscoveredGraph(
        graph=graph,
        channel=channel,
        was_bidirected=bidirected,
        canonical_order=order,
        provenance={
            "git_commit": manifest.get("git_commit"),
            "discovery_library": block.get("library"),
            "discovery_library_version": block.get("library_version"),
            "indep_test": block.get("resolved_args", {}).get("indep_test"),
            "alpha": block.get("resolved_args", {}).get("alpha"),
            "stable": block.get("resolved_args", {}).get("stable"),
            "n_bidirected": block.get("n_bidirected"),
            "canonical_order": "|".join(order),
        },
    )


# --------------------------------------------------------------------------- #
# Covariates
# --------------------------------------------------------------------------- #


def _skeleton(graph: nx.DiGraph) -> set[frozenset[str]]:
    return {frozenset(edge) for edge in graph.edges()}


def _endogenous(graph: nx.DiGraph) -> nx.DiGraph:
    sub = graph.subgraph([n for n in graph.nodes() if n != PROTECTED])
    return nx.DiGraph(sub)


def covariates(
    topology: str, discovered: DiscoveredGraph
) -> dict[str, object]:
    """All PS-8 covariates for one (cell, seed), against ``topology``'s truth."""
    truth = true_dag(topology)
    est = discovered.graph

    true_skeleton = _skeleton(truth)
    est_skeleton = _skeleton(est)
    skeleton_shd = len(true_skeleton ^ est_skeleton)

    # X-X edges only: A-edges are BK-oriented, so their direction was never the
    # discoverer's to get wrong (PS-8 / PS-7 note (e)).
    xx_true = {e for e in true_skeleton if PROTECTED not in e}
    orientation_wrong = {
        (u, v)
        for u, v in est.edges()
        if PROTECTED not in (u, v)
        and frozenset((u, v)) in xx_true
        and not truth.has_edge(u, v)
    }
    tiebreak = {e for e, c in discovered.channel.items() if c == TIEBREAK_CHANNEL}
    bidirected = {e for e, flag in discovered.was_bidirected.items() if flag}

    d_eff = D_EFF[topology]
    sid_value = sid(_endogenous(truth), _endogenous(est))

    # Pre-stated structural expectation (PS-8): on a correctly-recovered
    # chain/triangle skeleton every X-X edge is tie-break-resolved AND correct.
    expectation_applies = topology in ("chain", "triangle") and skeleton_shd == 0
    expectation_violated = expectation_applies and (
        len(tiebreak) != len(xx_true) or len(orientation_wrong) != 0
    )

    return {
        "skeleton_shd": skeleton_shd,
        "orientation_wrong_count": len(orientation_wrong),
        "tiebreak_resolved_count": len(tiebreak),
        "wrong_and_tiebreak_count": len(orientation_wrong & tiebreak),
        "was_bidirected_count": len(bidirected),
        "sid": sid_value,
        "n_sid": sid_value / (d_eff * (d_eff - 1)),
        "d_eff": d_eff,
        "n_xx_true_edges": len(xx_true),
        "wrong_ci_orientation_count": len(
            {e for e in orientation_wrong if discovered.channel.get(e) == "ci"}
        ),
        "structural_expectation_applies": expectation_applies,
        "structural_expectation_violated": expectation_violated,
    }


def build_covariate_frame(
    root: Path = L1D_ROOT, n_seeds: int = N_SEEDS_FULL
) -> pd.DataFrame:
    """One row per (topology, family, regime, seed) over the four-rung tree."""
    rows: list[dict[str, object]] = []
    for cell in grid_cells():
        for seed_idx in range(n_seeds):
            path = cell.seed_dir(seed_idx, root) / "manifest.json"
            with path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            discovered = discovered_from_manifest(manifest, cell.regime)
            row: dict[str, object] = {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "seed_idx": seed_idx,
                "seed": manifest["config"]["seed"],
            }
            row.update(covariates(cell.topology, discovered))
            row.update(discovered.provenance)
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["topology", "family", "regime", "seed_idx"]).reset_index(
        drop=True
    )


def write_covariates(root: Path = L1D_ROOT, n_seeds: int = N_SEEDS_FULL) -> Path:
    frame = build_covariate_frame(root=root, n_seeds=n_seeds)
    out = root / "summary" / "h3_covariates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8")
    return out
