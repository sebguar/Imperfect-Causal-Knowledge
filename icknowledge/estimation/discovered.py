"""Estimated-SCM assembly for the L1-discovered rung.

Counterpart of `base.build_estimated_scm`, for the rung whose graph came out of
the search instead of out of the ground truth: DISCOVERED graph +
estimated structural equations, assembled into the SAME `SCM` class so `abduct`
and `counterfactual` work unchanged.

# DUPLICATED, NOT SHARED. [PS-7; deliberate] The assembly below is a
# near-copy of `base.build_estimated_scm`, and that is the point: the two differ
# in the graph they build against, in the guard they run, and in how a parentless
# node is handled, and factoring the common lines into a shared helper would put
# the FROZEN L1-oracle path one refactor away from a
# behavioural change. Duplication keeps `build_estimated_scm` byte-for-byte as
# audited and makes the three differences visible side by side rather than hidden
# behind branch flags.
#
# TRUTH-FREE BY SIGNATURE. [PS-7 note (b)] This function does
# not accept a `true_scm` and cannot reach one. The L1-discovered rung sees the
# discovered graph and the estimation sample, nothing else; anything needing the
# truth belongs at the scoring layer (the realizer), exactly as for L1-oracle.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from icknowledge.estimation.base import FittedEquations, equation_from_coefficients
from icknowledge.estimation.forms import INTERCEPT, NodeForm
from icknowledge.scm.base import SCM, NoiseSampler, StructuralEquation

__all__ = ["build_estimated_scm_from_discovery"]


def _unestimated_noise_sampler(node: str) -> NoiseSampler:
    """Guard sampler: sampling is not part of any estimated rung (mean function only).

    A local twin of `base._unestimated_noise_sampler` — same mean-function-only
    (no σ̂²) semantics, own error text, no import across the frozen path (see the module note on
    duplication). The `SCM` constructor requires a noise sampler for EVERY node,
    including the parentless ones, so this stub is what a discovered root carries:
    its noise is supplied by ABDUCTION at counterfactual time and is never drawn.
    """

    def _raise(rng: np.random.Generator, n: int) -> np.ndarray:
        raise RuntimeError(
            f"discovered-graph estimated SCM has no noise distribution for {node!r}: only "
            "the mean function is fitted (no σ̂² estimation), so ancestral sampling is "
            "undefined. Use abduct/counterfactual, or sample from the true SCM."
        )

    return _raise


def _parentless_equation(node: str, data: pd.DataFrame) -> tuple[StructuralEquation, float]:
    """The PS-7-note-(b) truth-free marginal for a node the search left parentless.

    # [PS-7 note (b)] An intercept-only fitted equation,
    # x = ȳ + u, with ȳ the node's sample mean over the ESTIMATION SAMPLE — the
    # unregularized-OLS solution for a node with no design columns, i.e. the same
    # L1-oracle estimator, not a second one. Behaviourally INERT: abduction
    # recovers u = x − ȳ, so a non-acted parentless node reproduces its factual
    # value exactly whatever ȳ is, and it is never forward-evaluated (no ancestral
    # sampling from an estimated SCM).
    #
    # A IS NOT SPECIAL-CASED. It reaches this function by exactly the route an
    # X_i whose incoming edges the search failed to recover reaches it — that
    # uniformity is what keeps the builder truth-free. (The L1-oracle builder, by
    # contrast, copies A's TRUE root mechanism verbatim; it may, because it holds
    # the true SCM. This is the one intended numerical difference between the two
    # assemblies and is excluded from the drift guard, per PS-7 note (c).)
    """
    if node not in data.columns:
        raise ValueError(
            f"parentless discovered node {node!r} has no column in the estimation sample; "
            "its marginal intercept (PS-7 note b) cannot be fitted."
        )
    intercept = float(data[node].to_numpy(dtype=float).mean())
    equation = equation_from_coefficients(
        NodeForm(node=node, parents=()), {INTERCEPT: intercept}
    )
    return equation, intercept


def build_estimated_scm_from_discovery(
    discovered_graph: nx.DiGraph,
    fitted: FittedEquations,
    data: pd.DataFrame | None = None,
) -> SCM:
    """Assemble the L1-discovered causal model: DISCOVERED graph + fitted equations.

    Parameters
    ----------
    discovered_graph:
        The oriented DAG from the orientation stage (CPDAG → Meek → tie-break). Used VERBATIM —
        it is the definition of the rung.
    fitted:
        Output of the form-template-known estimator run on a `synthesize_form_spec` template for
        THIS graph. Must cover every non-root node of ``discovered_graph`` and
        nothing outside it.
    data:
        The estimation sample. Required whenever the graph has a parentless node
        (it always does — A is parentless by PS-7 background knowledge), since
        that is where the PS-7-note-(b) marginal intercept ȳ comes from.

    Raises on: a non-DAG graph, a fitted node absent from the graph, a missing
    equation or form for a non-root, a form whose declared parents disagree with
    the DISCOVERED parent set, a parentless node that nonetheless carries a form,
    and a missing ``data`` when one is needed.
    """
    if not isinstance(discovered_graph, nx.DiGraph):
        raise TypeError(
            f"discovered_graph must be an nx.DiGraph, got {type(discovered_graph).__name__}."
        )
    if not nx.is_directed_acyclic_graph(discovered_graph):
        raise ValueError(
            "discovered_graph is not a DAG — the orientation stage must return an "
            "acyclic graph before any equation is fitted against it."
        )

    nodes = list(discovered_graph.nodes())
    roots = [n for n in nodes if not list(discovered_graph.predecessors(n))]
    non_roots = [n for n in nodes if list(discovered_graph.predecessors(n))]

    outside = sorted(set(fitted.form) - set(nodes))
    if outside:
        raise ValueError(
            f"fitted form covers node(s) {outside} that are not in the discovered graph — "
            "the template and the graph it was synthesized from have drifted apart."
        )
    missing = [n for n in non_roots if n not in fitted.equations]
    if missing:
        raise ValueError(f"fitted equations missing for non-root node(s): {missing}.")

    # The DISCOVERED-flavoured guard. Same shape as base.py's L1-oracle guard, but
    # the reference parent set is the DISCOVERED graph's, never the truth's: this
    # rung is DEFINED by the search's parent sets, right or wrong, and the guard's
    # only job is that the columns fitted for a node are the columns its
    # discovered edges call for.
    for node in non_roots:
        node_form = fitted.form.get(node)
        if node_form is None:
            raise ValueError(f"form spec missing for non-root node {node!r}.")
        if set(node_form.parents) != set(discovered_graph.predecessors(node)):
            raise ValueError(
                f"form parents for {node!r} ({sorted(node_form.parents)}) do not match the "
                f"DISCOVERED graph's parent set "
                f"({sorted(discovered_graph.predecessors(node))}) — L1-discovered requires "
                "the template to be synthesized from the graph it is assembled against."
            )
    fitted_roots = sorted(set(roots) & set(fitted.form))
    if fitted_roots:
        raise ValueError(
            f"parentless discovered node(s) {fitted_roots} carry a form spec; "
            "`synthesize_form_spec` emits forms for non-root nodes only, and a parentless "
            "node's mechanism is the PS-7-note-(b) marginal built here, not a fitted "
            "equation supplied from outside."
        )
    if roots and data is None:
        raise ValueError(
            f"discovered graph has parentless node(s) {sorted(roots)}; `data` (the "
            "estimation sample) is required to fit their PS-7-note-(b) marginal intercepts."
        )

    equations: dict[str, StructuralEquation] = {}
    noise_samplers: dict[str, NoiseSampler] = {}
    intercepts: dict[str, float] = {}
    for node in roots:
        equations[node], intercepts[node] = _parentless_equation(node, data)
        noise_samplers[node] = _unestimated_noise_sampler(node)  # mean function only
    for node in non_roots:
        equations[node] = fitted.equations[node]
        noise_samplers[node] = _unestimated_noise_sampler(node)  # mean function only

    scm = SCM(
        graph=discovered_graph,  # defensive-copied by SCM; the DISCOVERED graph, verbatim
        equations=equations,
        noise_samplers=noise_samplers,
        regime=fitted.metadata.get("regime"),
        is_anm=True,  # fitted equations are additive in u by construction
        name="estimated[discovered-graph]",
    )
    # Provenance hook for PS-4: the marginal intercepts are the only
    # numbers this assembly creates that are not already in `fitted`. Attached as
    # an attribute rather than threaded through the SCM constructor, mirroring the
    # `jacobian_fn` discovery pattern (the NLG family specification, scm/triangles.py).
    scm.parentless_intercepts = intercepts
    return scm
