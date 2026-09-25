"""Truth-free form-template synthesis for the L1-discovered rung.

Turns a DISCOVERED DiGraph into the supplied-form contract the estimation layer
already speaks (`NodeForm` / `TermSpec` / `Factor`, estimation/forms.py), so the
L1-discovered rung reuses the L1-oracle estimator verbatim and differs from
L1-oracle ONLY in the graph it is handed.

[PS-7 — the form-preserving estimation basis; note (a)] The per-family
functional basis of the L1-oracle templates is carried onto the DISCOVERED
graph, applied PER PARENT VARIABLE:

  family="linear" — every discovered parent enters as an identity main effect;
  family="nlg"    — a discovered A → X_i edge enters as identity (the oracle's
                    `a_main` column), a discovered X_j → X_i edge as tanh(X_j).

[PS-7] NO A·X INTERACTION COLUMN, IN EITHER REGIME: every emitted term is
SINGLE-FACTOR and `interactions` is always empty. The function takes NO `regime`
argument — the discovered form is regime-free by construction, and the regime
enters only through the DATA the estimator is later fitted on. Nothing here
reads a true SCM, a true form spec or a regime label.
"""

from __future__ import annotations

import networkx as nx

from icknowledge.estimation.forms import TANH, Factor, NodeForm, TermSpec

__all__ = ["PROTECTED", "synthesize_form_spec"]

#: The protected attribute's node name, literal across every SCM builder
#: (scm/triangles.py, scm/chains.py, scm/colliders.py all name it "A").
PROTECTED = "A"

#: Functional families with a settled form-preserving basis. Keyed exactly as
#: `pipeline._ESTIMATORS` / the topology drivers' `_FAMILIES`, so one config
#: field selects estimator, oracle template and discovered basis alike.
_FAMILIES = ("linear", "nlg")


def _factor(parent: str, family: str) -> Factor:
    """The form-preserving basis element for ONE discovered parent edge.

    Linear: identity, always. NLG: identity for the protected root A, tanh for
    any other parent — read off the oracle NLG templates, whose X-parents always
    enter as tanh(X_j) and whose A-parent, where present, always enters as the
    untransformed `a_main` column (scm/triangles.py `nlg_triangle_form_spec`,
    scm/chains.py `nlg_chain_form_spec`, scm/colliders.py `nlg_collider_form_spec`).
    """
    if family == "linear":
        return Factor(parent)
    return Factor(parent) if parent == PROTECTED else Factor(parent, TANH)


def synthesize_form_spec(
    discovered_graph: nx.DiGraph, family: str
) -> dict[str, NodeForm]:
    """Supplied form template for the DISCOVERED graph — main effects only.

    Mirrors the oracle form-spec contract: keyed by node, ROOTS ABSENT (no
    equation is fitted for a parentless node — the assembly step gives those the
    PS-7-note (b) truth-free marginal instead). Parents are taken in SORTED order
    so the emitted design-column order is a deterministic function of the
    discovered graph alone, never of `nx` insertion order.

    Every node's terms are emitted EXPLICITLY (rather than relying on the
    parents-plus-interactions default of `resolve_terms`) in both families, one
    uniform code path: PS-7's "no A·X column" is then a structural property of
    the object this function returns — every `TermSpec` carries exactly one
    `Factor` — rather than a property of a reconstruction rule applied later.

    Raises
    ------
    TypeError
        ``discovered_graph`` is not an `nx.DiGraph`.
    ValueError
        Unknown ``family``; the graph is cyclic (an UNRESOLVED ORIENTATION left
        by the CPDAG → Meek → tie-break stage surfaces here, as a cycle,
        rather than silently downstream); "A" is absent; or A has parents
        (PS-7 puts A's exogeneity in as BACKGROUND KNOWLEDGE before the search,
        so an A-with-parents graph is an orientation-stage bug, not a discovery result).

    Deliberately NOT validated: that the node set matches the true SCM's. That
    comparison needs the truth and belongs at the wiring seam, not inside
    a truth-free synthesizer.
    """
    if not isinstance(discovered_graph, nx.DiGraph):
        raise TypeError(
            f"discovered_graph must be an nx.DiGraph, got {type(discovered_graph).__name__}."
        )
    if family not in _FAMILIES:
        raise ValueError(
            f"unknown functional family {family!r}; settled bases (PS-7): {list(_FAMILIES)}."
        )
    if not nx.is_directed_acyclic_graph(discovered_graph):
        raise ValueError(
            "discovered_graph is not a DAG — a cycle here means the orientation stage "
            "(CPDAG → Meek → tie-break) left an orientation unresolved or resolved it "
            "inconsistently. Fix the orientation upstream; the estimator has no "
            "meaningful parent set for a cyclic node."
        )
    if PROTECTED not in discovered_graph:
        raise ValueError(
            f"discovered_graph has no {PROTECTED!r} node; the protected attribute is a "
            "variable of every cell's SCM and must be carried through discovery."
        )
    a_parents = sorted(discovered_graph.predecessors(PROTECTED))
    if a_parents:
        raise ValueError(
            f"{PROTECTED!r} has discovered parents {a_parents} — PS-7 supplies A's "
            "exogeneity as BACKGROUND KNOWLEDGE to the search, so this is a discovery-stage "
            "(discovery / orientation) bug, not an admissible discovery result."
        )

    form: dict[str, NodeForm] = {}
    for node in discovered_graph.nodes():
        parents = tuple(sorted(discovered_graph.predecessors(node)))
        if not parents:
            continue  # roots carry no fitted equation (assembly handles them, PS-7 note b)
        form[node] = NodeForm(
            node=node,
            parents=parents,
            terms=tuple(TermSpec((_factor(p, family),)) for p in parents),
        )
    return form
