"""Equation-estimation layer: supplied form templates + estimated-SCM assembly (L1-oracle).

Produces the causal model handed to the generator under the L1-ORACLE knowledge
condition: TRUE graph + ESTIMATED structural equations (the "Causal structure"
specification; recourse-harness). The estimated SCM is an
instance of the SAME `SCM` class as the ground truth, so `abduct` and
`counterfactual` work unchanged and wiring L1-oracle into the recourse pipeline
is a model swap, not a new code path.

# FORM-TEMPLATE-KNOWN L1-oracle. [the L1-oracle form-template-known
# semantics]
# The estimator is SUPPLIED the parametric form template of each structural
# equation (which parents enter, and whether an A×parent interaction term is
# present) and estimates ONLY the coefficients. The true graph alone does not
# determine functional form — it says A→X₂ but not whether A enters as a main
# effect or as an interaction — so form must be supplied explicitly. This makes
# L1-oracle a pure FINITE-SAMPLE ESTIMATION-ERROR condition, which is what the
# L2→L1-oracle contrast in H1 is defined to isolate. The form template is
# SUPPLIED, NOT INFERRED — a documented scope choice, not an inference the
# harness makes.

# The estimator interface is PLUGGABLE PER FUNCTIONAL FAMILY. The linear and
# nonlinear-Gaussian estimators ship (linear.py, nonlinear_gaussian.py); a new
# family must slot in behind the same `EquationEstimator` protocol WITHOUT touching
# procedure.py / scoring.py / model_conditions.py — mirroring the way the shared
# predict(factual, intervention) contract already isolates L0/L2 from the
# pipeline. [functional families; the scope ladder]
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# NodeForm / FormSpec / term naming live in forms.py (dependency-free) so SCM
# builder modules can host their form specs without a circular import; they are
# re-exported here as part of the estimation interface.
from icknowledge.estimation.forms import (
    INTERCEPT,
    Factor,
    FormSpec,
    NodeForm,
    TermSpec,
    evaluate_term,
    interaction_term_name,
    resolve_terms,
    term_names,
    term_spec_name,
)
from icknowledge.scm.base import SCM, NoiseSampler, StructuralEquation

__all__ = [
    "INTERCEPT",
    "EquationEstimator",
    "Factor",
    "FittedEquations",
    "FormSpec",
    "NodeForm",
    "TermSpec",
    "build_estimated_scm",
    "dataset_identity",
    "equation_from_coefficients",
    "estimation_provenance",
    "evaluate_term",
    "interaction_term_name",
    "resolve_terms",
    "term_names",
    "term_spec_name",
]


class EquationEstimator(Protocol):
    """One family's coefficient estimator behind the pluggable interface."""

    def fit(self, data: pd.DataFrame, form: FormSpec) -> FittedEquations: ...


@dataclass
class FittedEquations:
    """The output of one estimator run: coefficients + assembled equations + provenance."""

    coefficients: dict[str, dict[str, float]]  # node -> term name -> fitted value
    equations: dict[str, StructuralEquation]  # node -> callable matching the SCM's
    # equation signature f(parents, noise) & additive-noise convention (u passed as arg)
    form: FormSpec
    metadata: dict  # estimator id, dataset identity, regime, family


def equation_from_coefficients(
    node_form: NodeForm, coefficients: Mapping[str, float]
) -> StructuralEquation:
    """Assemble x = f̂(pa) + u from a coefficient dict, in the SCM's equation convention.

    Preserves the additive-noise convention (noise passed as an equation argument,
    scm-specification), so `abduct` recovers û = x − f̂(pa) exactly and
    `counterfactual` runs unchanged on the estimated SCM.

    # [the NLG family specification] The reconstructed mean function is assembled from the SAME
    # resolved term list (and the same `evaluate_term`) that built the design
    # matrix at fit time, so a template's fixed transform — tanh(·) for the NLG
    # family — propagates through `counterfactual` exactly as it was fitted. The
    # equation stays ADDITIVE in the noise argument whatever the transform, which
    # is what keeps abduction exact (û = x − f̂(pa)) on the NLG rung too: the
    # nonlinearity is in the parents, never in the noise (the L1-oracle form-template-known
    # semantics mean-function-only).
    """
    intercept = float(coefficients[INTERCEPT])
    term_coefs = [
        (term, float(coefficients[term_spec_name(term)]))
        for term in resolve_terms(node_form)
    ]

    def _f(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        out = np.full(np.shape(noise), intercept, dtype=float)
        for term, coef in term_coefs:
            out = out + coef * evaluate_term(term, parents)
        return out + noise

    return _f


def dataset_identity(data: pd.DataFrame) -> dict:
    """Stable identity record of the estimation sample (shared-sample auditability).

    # Estimation sample = the classifier's training sample; no held-out
    # split. The L1-discovered condition
    # must later run its PC step on THIS SAME sample; recording the sample's
    # content hash in the manifest makes the shared-sample constraint auditable.
    """
    row_hashes = pd.util.hash_pandas_object(data, index=False).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return {
        "n_rows": int(len(data)),
        "columns": [str(c) for c in data.columns],
        "sha256": digest,
    }


def _unestimated_noise_sampler(node: str) -> NoiseSampler:
    """Guard sampler for estimated non-root nodes: sampling is not part of L1-oracle.

    # Noise variance is NOT estimated. [skill:scm-specification
    # noise convention] Only the mean function f̂(pa) is fitted: under the
    # additive-noise convention abduction recovers û = x − f̂(pa) exactly regardless
    # of any variance estimate, so σ̂² never enters a point counterfactual and is
    # not estimated or STORED as a structural parameter (this holds even where the
    # true SCM is group-heteroskedastic). Copying the TRUE noise samplers instead
    # would silently smuggle ground-truth noise distributions into the estimated
    # object; raising loudly keeps the estimated SCM honest about what was fitted.
    """

    def _raise(rng: np.random.Generator, n: int) -> np.ndarray:
        raise RuntimeError(
            f"estimated SCM has no noise distribution for {node!r}: L1-oracle fits the "
            "mean function only (no σ̂² estimation), so ancestral sampling from the "
            "estimated SCM is undefined. Use abduct/counterfactual, or sample from the "
            "true SCM."
        )

    return _raise


def build_estimated_scm(true_scm: SCM, fitted: FittedEquations) -> SCM:
    """Assemble the L1-oracle causal model: TRUE graph + fitted equations, same `SCM` class.

    The true graph is reused VERBATIM — L1-oracle = true graph + estimated
    equations, by definition of the rung (the "Causal structure" specification;
    recourse-harness). Returning the same `SCM` class keeps `abduct` /
    `counterfactual` working unchanged, so the pipeline wiring is a swap, not a new
    code path.
    """
    roots = [n for n in true_scm.nodes if not true_scm.parents(n)]
    non_roots = [n for n in true_scm.nodes if true_scm.parents(n)]

    missing = [n for n in non_roots if n not in fitted.equations]
    if missing:
        raise ValueError(f"fitted equations missing for non-root node(s): {missing}.")
    for node in non_roots:
        node_form = fitted.form.get(node)
        if node_form is None:
            raise ValueError(f"form spec missing for non-root node {node!r}.")
        if set(node_form.parents) != set(true_scm.parents(node)):
            raise ValueError(
                f"form parents for {node!r} ({sorted(node_form.parents)}) do not match the "
                f"true graph's parent set ({sorted(true_scm.parents(node))}) — L1-oracle "
                "requires the TRUE parent sets."
            )

    equations: dict[str, StructuralEquation] = {}
    noise_samplers: dict[str, NoiseSampler] = {}
    for node in roots:
        # Root A: the TRUE root mechanism is copied VERBATIM. This is
        # non-load-bearing — A is never sampled during recourse (it is immutable
        # and its factual value is given), so A's marginal is never consulted;
        # estimating it would be a DISTRIBUTIONAL estimate, not a structural
        # equation between endogenous variables, and is outside what the estimation step
        # asks for. If this ever becomes load-bearing, it needs a follow-up
        # adjudication.
        equations[node] = true_scm.equations[node]
        noise_samplers[node] = true_scm.noise_samplers[node]
    for node in non_roots:
        equations[node] = fitted.equations[node]
        noise_samplers[node] = _unestimated_noise_sampler(node)  # mean function only

    return SCM(
        graph=true_scm.graph,  # defensive copy of the TRUE graph, verbatim
        equations=equations,
        noise_samplers=noise_samplers,
        regime=true_scm.regime,
        is_anm=True,  # fitted equations are additive in u by construction (see above)
        name=f"estimated[{true_scm.name}]",
    )


def estimation_provenance(fitted: FittedEquations) -> dict:
    """JSON-serializable provenance record for the run manifest.

    Version-controlled evidence that the form template was SUPPLIED as specified
    and that L1-oracle and L1-discovered share one estimation sample:
    estimator id, serialized FormSpec, all fitted coefficients, dataset identity
    (seed/hash), regime, family — all inside ``metadata`` as recorded at fit time.
    """
    return {
        "form_spec": {
            node: {
                "node": nf.node,
                "parents": list(nf.parents),
                "interactions": [list(pair) for pair in nf.interactions],
                # The RESOLVED design columns: for linear/collider
                # templates these are just parents + interactions, but for an
                # explicit-`terms` template (NLG, the NLG family specification) they are the only
                # record of
                # which transformed columns were actually fitted — so the auditor
                # can read the supplied form off the manifest for EVERY family.
                "terms": term_names(nf),
            }
            for node, nf in fitted.form.items()
        },
        "coefficients": {
            node: {term: float(v) for term, v in coefs.items()}
            for node, coefs in fitted.coefficients.items()
        },
        "metadata": dict(fitted.metadata),
    }
