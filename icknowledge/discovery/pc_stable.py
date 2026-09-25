"""PC-Stable search + CPDAG → DAG completion for the L1-discovered rung.

PS-7 (method, CI test, encoding, background knowledge, variable set), Meek
closure and the lexical tie-break live here, every pinned choice passed
explicitly and echoed into a `DiscoveryRecord`.

TRUTH-FREE BY SIGNATURE. No function takes a true SCM, a true graph, a family or
a regime; `discover_and_synthesize`'s family key is a DECLARED property of the
experimental cell, not an inference from the data.

VERIFIED ENDPOINT ENCODING, read off the installed causal-learn 0.1.4.8 source
(`search/ConstraintBased/PC.py:98-100`, `graph/Endpoint.py:10-13`,
`graph/GraphClass.py:96-102`, `graph/GeneralGraph.py:122-167`) — NOT from
convention. `cg.G.graph` is a d × d int array over `node_names`; for (i, j):

    graph[i, j] == -1 and graph[j, i] ==  1   →  i → j      (directed)
    graph[i, j] == -1 and graph[j, i] == -1   →  i — j      (undirected)
    graph[i, j] ==  1 and graph[j, i] ==  1   →  i ↔ j      (bidirected)
    graph[i, j] ==  0 and graph[j, i] ==  0   →  no adjacency

MEEK CLOSURE IS ALREADY DONE BY THE LIBRARY — verified, not assumed:
`PC.py:113-118` (the `uc_rule == 0` branch PS-7 pins) runs `Meek.meek` to a
fixed point, so the tie-break here operates on Markov-ambiguous edges only.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version as _dist_version

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from icknowledge.discovery.provenance import DiscoveredEdge, DiscoveryRecord
from icknowledge.discovery.synthesis import PROTECTED, synthesize_form_spec
from icknowledge.estimation.base import dataset_identity
from icknowledge.estimation.forms import NodeForm

__all__ = ["LIBRARY", "discover_and_synthesize", "discover_graph"]

#: Distribution name for `importlib.metadata.version`. VERIFIED:
#: `causallearn` exposes NO `__version__` attribute, so the metadata accessor is
#: the only one available and is what PS-7's "library + version" is read from.
LIBRARY = "causal-learn"

# Endpoint values, named rather than inlined (see the module docstring for the
# source lines each is verified against).
_TAIL = -1
_NULL = 0
_ARROW = 1

_ENCODING_NOTE = (
    "A is passed to the CI test as 0/1 via (a + 1) / 2, on a COPY of the "
    "estimation sample; the SCM's centered ±1 encoding is what leaves this "
    "function and what every downstream rung sees. [PS-7] The "
    "transform is INERT to the search: Fisher-z is a partial-correlation test "
    "and partial correlations are invariant under an affine rescaling of any "
    "variable, so every CI decision is bit-identical under either encoding. It "
    "is applied identically on both functional families and both regimes, so it "
    "cannot generate a linear-vs-NLG difference; it is recorded here so no "
    "reviewer has to imagine it did work."
)


def _library_version() -> str:
    """Installed causal-learn version, from distribution metadata.

    `import causallearn; causallearn.__version__` does NOT exist at 0.1.4.8
    (checked against the installed package); `importlib.metadata.version` does.
    """
    return _dist_version(LIBRARY)


def _classify_adjacency(e_ij: int, e_ji: int, name_i: str, name_j: str) -> str:
    """One (i, j) endpoint pair → one of 'none' / 'i->j' / 'j->i' / 'undirected' / 'bidirected'."""
    if e_ij == _NULL and e_ji == _NULL:
        return "none"
    if e_ij == _TAIL and e_ji == _ARROW:
        return "i->j"
    if e_ij == _ARROW and e_ji == _TAIL:
        return "j->i"
    if e_ij == _TAIL and e_ji == _TAIL:
        return "undirected"
    if e_ij == _ARROW and e_ji == _ARROW:
        return "bidirected"
    raise ValueError(
        f"unrecognized endpoint pair (graph[{name_i},{name_j}]={e_ij}, "
        f"graph[{name_j},{name_i}]={e_ji}) for the adjacency {name_i} — {name_j}. "
        "A PC CPDAG at causal-learn 0.1.4.8 emits only the four pairs verified in "
        "this module's docstring (directed / undirected / bidirected / absent); "
        "CIRCLE, TAIL_AND_ARROW and ARROW_AND_ARROW endpoints belong to FCI-family "
        "output and are not silently reinterpreted here."
    )


def _orient_cpdag(
    endpoints: np.ndarray, canonical_order: Sequence[str]
) -> tuple[nx.DiGraph, tuple[DiscoveredEdge, ...], int]:
    """CPDAG endpoint matrix → oriented DAG + per-edge orientation channels.

    Deliberately takes the raw endpoint matrix rather than a `CausalGraph`, so
    the tie-break completion rule is testable on hand-crafted matrices with no PC
    run in the loop (tests/test_pc_stable_orientation.py).

    Order of operations, exactly as pre-committed:
      1. classify every adjacency; bidirected is demoted to undirected with
         ``was_bidirected=True`` (note (d));
      2. defensive background-knowledge layer — nothing may point into A, and an
         A — X_i left undirected is oriented A → X_i on channel ``"bk"``;
      3. SINGLE-PASS tie-break over whatever is still undirected;
      4. acyclicity assertion.

    Returns ``(graph, edges, n_bidirected)``.
    """
    names = list(canonical_order)
    endpoints = np.asarray(endpoints)
    if endpoints.shape != (len(names), len(names)):
        raise ValueError(
            f"endpoint matrix {endpoints.shape} does not match the canonical order "
            f"{names} ({len(names)} nodes)."
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    edges: list[DiscoveredEdge] = []
    pending: list[tuple[int, int, bool]] = []  # (i, j, was_bidirected), i < j
    n_bidirected = 0

    def _emit(src: int, tgt: int, channel: str, was_bidirected: bool) -> None:
        graph.add_edge(names[src], names[tgt])
        edges.append(
            DiscoveredEdge(
                source=names[src],
                target=names[tgt],
                channel=channel,  # type: ignore[arg-type]
                was_bidirected=was_bidirected,
            )
        )

    # -- 1. classify -------------------------------------------------------
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            kind = _classify_adjacency(
                int(endpoints[i, j]), int(endpoints[j, i]), names[i], names[j]
            )
            if kind == "none":
                continue
            if kind in ("i->j", "j->i"):
                src, tgt = (i, j) if kind == "i->j" else (j, i)
                if names[tgt] == PROTECTED:
                    raise ValueError(
                        f"the search returned {names[src]!r} → {PROTECTED!r}, but "
                        f"[PS-7] supplies 'A is a root' as BACKGROUND KNOWLEDGE "
                        "(add_forbidden_by_node) before the search runs, so no edge can "
                        "legitimately point into A. This is a background-knowledge "
                        "wiring bug, not an admissible discovery result."
                    )
                # A-EDGE CHANNEL ATTRIBUTION RULE. Any directed edge OUT of A is
                # labelled "bk" regardless of which causal-learn code path wrote
                # it: the background knowledge makes the reverse orientation
                # impossible, so the direction is BK-attributable BY
                # CONSTRUCTION, and "PC oriented it" vs. "BK forced it" is not a
                # distinction this module tries to make internally. That costs
                # nothing, because [PS-7 note (e)] the substantive
                # CI-orientation claim — the collider being the only cell of the
                # three where CI testing supplies orientation — lives entirely on the
                # non-A edges.
                _emit(src, tgt, "bk" if names[src] == PROTECTED else "ci", False)
                continue
            if kind == "bidirected":
                # [PS-7 note (d)] Under causal sufficiency by
                # construction, i ↔ j cannot mean latent confounding; it is a
                # finite-sample collider-orientation conflict. Demote to
                # undirected, let the tie-break resolve it, keep the flag.
                n_bidirected += 1
                pending.append((i, j, True))
            else:
                pending.append((i, j, False))

    # -- 2. defensive BK layer, BEFORE the tie-break -----------------------
    # An A — X_i should never survive: `orient_by_background_knowledge` runs
    # inside pc() (PC.py:110-111) and Meek is BK-aware (Meek.py). If one does,
    # PS-7's "A is a root" still decides it, and it is NEVER attributed to the
    # tie-break — the tie-break's arbitrariness is what PS-8 reads on the X — X
    # edges, and folding a BK-determined A-edge into that count would inflate it.
    still_undirected: list[tuple[int, int, bool]] = []
    for i, j, was_bidirected in pending:
        if names[i] == PROTECTED:
            _emit(i, j, "bk", was_bidirected)
        elif names[j] == PROTECTED:
            _emit(j, i, "bk", was_bidirected)
        else:
            still_undirected.append((i, j, was_bidirected))

    # -- 3. SINGLE-PASS tie-break [PS-7] --------------------------
    # Every remaining undirected edge is oriented lower → higher in the canonical
    # order, in ONE sweep. Meek is deliberately NOT re-run between orientations:
    # an iterative rule (orient one edge, re-propagate, orient the next) is a
    # DIFFERENT completion rule from the one pre-committed, and
    # adopting it now would be an unregistered rule change. The arbitrariness of
    # the single-pass rule is not a defect to be engineered away — it is part of
    # what the linear control arm measures (PS-7 rationale; a tie-break
    # mis-orientation on a Markov-ambiguous edge is the identifiability-limited
    # case PS-8 classifies as genuine, not as a bug).
    for i, j, was_bidirected in still_undirected:
        _emit(i, j, "tiebreak", was_bidirected)

    # -- 4. acyclicity, asserted AT THE ORIENTATION SITE -------------------
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        cycle_edges = [(u, v) for u, v, *_ in cycle]
        tiebreak_in_cycle = [
            (e.source, e.target)
            for e in edges
            if e.channel == "tiebreak" and (e.source, e.target) in set(cycle_edges)
        ]
        raise ValueError(
            "The orientation stage (CPDAG → Meek → tie-break) produced a CYCLIC graph: "
            f"{' → '.join([u for u, _ in cycle_edges] + [cycle_edges[-1][1]])}. "
            f"Tie-break edges inside the cycle: {tiebreak_in_cycle or 'none'}. "
            "[PS-7] The completion rule must return a DAG; the lower → "
            "higher tie-break cannot close a cycle on its own, so a cycle here means "
            "a CI/BK-oriented edge runs higher → lower and the tie-break closed the "
            "loop around it. Fix the orientation stage — this must NOT surface "
            "downstream as a synthesis or assembly error."
        )
    return graph, tuple(edges), n_bidirected


def discover_graph(
    data: pd.DataFrame, *, alpha: float = 0.05
) -> tuple[nx.DiGraph, DiscoveryRecord]:
    """Run PC-Stable on ``data`` and complete its CPDAG to an oriented DAG.

    Parameters
    ----------
    data:
        The estimation sample — the classifier's training sample, per the L1-oracle
        form-template-known semantics's
        shared-sample constraint. Must carry the full node set
        {A, X₁ … X_k} (PS-7 variable set): excluding A would make a
        discovered graph structurally incapable of representing the A → X_i
        edges every one of the three topologies carries.
    alpha:
        CI-test significance level. Library default 0.05, passed explicitly.

    Returns
    -------
    (graph, record)
        ``graph`` is an `nx.DiGraph` over the ORIGINAL node names, guaranteed
        acyclic; ``record`` is the PS-7 provenance block.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
    if PROTECTED not in data.columns:
        raise ValueError(
            f"data has no {PROTECTED!r} column. [PS-7] PC-Stable runs on the "
            "FULL node set including the protected attribute; dropping it would convert "
            "graph error into variable-set truncation and break comparability with "
            "L1-oracle's parent sets."
        )

    # [PS-7 note (g)] CANONICAL ORDER = the lexical order of the
    # node identifiers (A < X1 < X2 < X3), fixed from the names alone. Four legs,
    # each closing a channel by which the ordering could carry information it
    # must not:
    #   * family-agnostic — the names are identical across linear and NLG cells,
    #     so the rule cannot differ between the two arms of PS-8's interaction;
    #   * data-independent — it reads no value, variance or marginal, so no
    #     varsortability channel opens; decisive under the raw-units convention, which forbids
    #     standardization and leaves every run on original-scale data;
    #   * discoverer-blind — it is a POST-discovery completion policy applied to
    #     the returned CPDAG; PC never sees it, so it cannot steer a skeleton or
    #     collider decision;
    #   * inert to the contrast under test — the linear-vs-NLG difference H3
    #     predicts is driven by family-dependent SKELETON error under a
    #     misspecified Fisher-z test (PS-8 mechanism note), not by the
    #     family-agnostic orientation rule.
    # Its coincidence with the true topological order on all three topologies' graphs
    # is a known, PRE-STATED property (note (f)), not a discovered convenience.
    canonical_order = sorted(data.columns)

    # Reindex BEFORE the call so `node_names[k]` and column k of the array are
    # the same variable — causal-learn zips them positionally.
    frame = data.reindex(columns=canonical_order).copy()

    # [PS-7] A-ENCODING, on this copy and for the PC call only. See
    # `_ENCODING_NOTE`: Fisher-z is a partial-correlation test and partial
    # correlations are affine-invariant, so the ±1 → 0/1 map leaves every CI
    # decision bit-identical; it is applied identically on both families, and it
    # is recorded in the provenance block so nobody has to imagine it did work.
    frame[PROTECTED] = (frame[PROTECTED].to_numpy(dtype=float) + 1.0) / 2.0
    arr = frame.to_numpy(dtype=float)

    # [PS-7] BACKGROUND KNOWLEDGE: "A is a root" — no X_i → A. DOMAIN
    # knowledge (gender is not caused by a credit feature), not DGP-family
    # knowledge: it says nothing about functional form, coefficients, noise or
    # which X → X edges exist, so it does not oracle-match the discoverer
    # (skill:recourse-harness). Built from FRESH GraphNode objects —
    # GraphNode equality is by name (verified in tests/test_discovery_env.py), so
    # these match the nodes pc() constructs from `node_names`.
    forbidden = [(name, PROTECTED) for name in canonical_order if name != PROTECTED]
    bk = BackgroundKnowledge()
    for source, target in forbidden:
        bk.add_forbidden_by_node(GraphNode(source), GraphNode(target))

    # [PS-7] Every argument explicit, including the ones that are already
    # library defaults, so the manifest RECORDS them rather than inheriting them.
    # On uc_rule=0 / uc_priority=2 specifically: these govern unshielded-collider
    # orientation and conflict resolution, and therefore how often bidirected
    # edges arise at all (note (d)). They are PINNED AND FROZEN HERE — revisiting
    # them after seeing ΔCost_ld would be a rule change, not a bug fix.
    resolved_args = {
        "alpha": float(alpha),
        "indep_test": str(fisherz),  # the string "fisherz" (pinned by tests/test_discovery_env.py)
        "stable": True,  # PC-Stable, not vanilla PC
        "uc_rule": 0,
        "uc_priority": 2,
        "mvpc": False,
        "background_knowledge": {"forbidden": [list(pair) for pair in forbidden]},
        "node_names": list(canonical_order),
        "verbose": False,
        "show_progress": False,
    }
    cg = pc(
        data=arr,
        alpha=float(alpha),
        indep_test=fisherz,
        stable=True,
        uc_rule=0,
        uc_priority=2,
        mvpc=False,
        background_knowledge=bk,
        node_names=list(canonical_order),
        verbose=False,
        show_progress=False,
    )

    # The CPDAG is already Meek-closed inside pc() (PC.py:118) — see the module
    # docstring; nothing is re-run or hand-rolled here.
    graph, edges, n_bidirected = _orient_cpdag(cg.G.graph, canonical_order)

    index = {name: k for k, name in enumerate(canonical_order)}
    adjacency = [[0] * len(canonical_order) for _ in canonical_order]
    for edge in edges:
        adjacency[index[edge.source]][index[edge.target]] = 1

    record = DiscoveryRecord(
        library=LIBRARY,
        library_version=_library_version(),
        resolved_args=resolved_args,
        canonical_order=tuple(canonical_order),
        encoding_note=_ENCODING_NOTE,
        edges=edges,
        n_bidirected=n_bidirected,
        adjacency=tuple(tuple(row) for row in adjacency),
        # [the L1-oracle form-template-known semantics] Identity of the sample the search
        # actually ran on,
        # in the same shape the estimation block records, so the provenance seam can assert
        # discovery-hash == estimation-hash. Taken on the CANONICAL frame (before
        # the A re-encoding, which never leaves this function), which is what
        # makes the record invariant to the caller's column order.
        dataset=dataset_identity(data.reindex(columns=canonical_order)),
    )
    return graph, record


def discover_and_synthesize(
    data: pd.DataFrame, family: str
) -> tuple[nx.DiGraph, dict[str, NodeForm], DiscoveryRecord]:
    """`discover_graph` → `synthesize_form_spec`: the pipeline call target.

    Composition only. No coefficient is estimated and no SCM is assembled here —
    that is `estimation.linear` + `estimation.discovered`, joined by the L1-discovered path.
    ``family`` is a DECLARED cell property and never reaches the
    search; only the form synthesizer sees it, exactly as at L1-oracle.
    """
    graph, record = discover_graph(data)
    return graph, synthesize_form_spec(graph, family), record
