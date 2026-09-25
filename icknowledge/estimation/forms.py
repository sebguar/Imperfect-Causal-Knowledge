"""Form-template types for equation estimation — dependency-free by design.

Kept free of any `icknowledge.scm` import so the SCM builders can place their
form specs NEXT TO the true equations (e.g. `triangle_form_spec` in
scm/triangles.py) without a circular import. The estimation machinery proper
lives in base.py / linear.py; see base.py for the form-template-known decision comments.

# A design column is a PRODUCT OF TRANSFORMED PARENTS. [the NLG family
# specification] The nonlinear-Gaussian family is LINEAR IN PARAMETERS after a FIXED,
# supplied transform of the parents (X₂ := b·A + g·tanh(X₁) + U₂), so the only
# generalization the NLG L1-oracle needs over the linear one is the ability for a
# design column to carry a fixed elementwise transform. The NLG family specification is explicit
# that
# this needs NO form-template-known extension: the estimator stays unregularized OLS, the
# transform is part of the SUPPLIED form template, never fitted or selected.
# A nonlinear-least-squares estimator would be a different estimator and is
# out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

INTERCEPT = "intercept"

#: Transform names admissible inside a form template. IDENTITY reproduces the
#: linear family's plain parent column; TANH is the fixed nonlinearity the NLG
#: builders use (scm/triangles.py `make_nonlinear_gaussian_triangle`). The map is
#: CLOSED: a template naming an unknown transform raises rather than silently
#: falling back to identity, so a typo cannot quietly refit a different model.
IDENTITY = "identity"
TANH = "tanh"
_TRANSFORMS: Mapping[str, Callable[[np.ndarray], np.ndarray]] = {
    IDENTITY: lambda values: values,
    TANH: np.tanh,
}


@dataclass(frozen=True)
class Factor:
    """One factor of a design column: a parent, optionally under a fixed transform."""

    parent: str
    transform: str = IDENTITY

    def __post_init__(self) -> None:
        if self.transform not in _TRANSFORMS:
            raise ValueError(
                f"unknown transform {self.transform!r} for parent {self.parent!r}; "
                f"admissible: {sorted(_TRANSFORMS)}."
            )


@dataclass(frozen=True)
class TermSpec:
    """One design-matrix column: the elementwise PRODUCT of its factors.

    A single identity factor is a plain main effect (``A``); two identity factors
    are the linear family's interaction column (``A*X1``); a transformed factor
    carries the fixed nonlinearity (``tanh(X1)``, ``A*tanh(X1)``).
    """

    factors: tuple[Factor, ...]


@dataclass(frozen=True)
class NodeForm:
    """Supplied parametric form template for ONE endogenous node's equation.

    ``parents`` is the node's TRUE parent set — it is what `build_estimated_scm`
    checks against the true graph, and (when ``terms`` is None) it also fixes the
    main-effect columns and their deterministic order. ``interactions`` lists
    named parent pairs whose elementwise product enters as an explicit
    interaction column, e.g. ``(("A", "X1"),)``. Omission from ``interactions``
    means the term is HARD-ZERO by exclusion from the design matrix, never
    fit-then-discarded.

    ``terms`` is the explicit-column escape hatch: when supplied it
    fully determines the design matrix, and ``parents`` degrades to the
    true-parent-set DECLARATION alone. It exists because the NLG effect-modifying
    builder's terms are {tanh(X₁), A·tanh(X₁)} with NO A main effect — a column
    set that "parents + interactions" cannot express, since A must still be
    declared a parent for the L1-oracle true-graph check. Templates supply either
    ``interactions`` or ``terms``, never both.
    """

    node: str
    parents: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...] = ()
    terms: tuple[TermSpec, ...] | None = None

    def __post_init__(self) -> None:
        if self.terms is not None:
            if self.interactions:
                raise ValueError(
                    f"{self.node!r}: supply either `interactions` or explicit `terms`, "
                    "not both — two column sources would make the design ambiguous."
                )
            declared = set(self.parents)
            unknown = sorted(
                {f.parent for t in self.terms for f in t.factors} - declared
            )
            if unknown:
                raise ValueError(
                    f"{self.node!r}: term factor(s) {unknown} are not declared parents "
                    f"{sorted(declared)} — the template must stay inside the true parent set."
                )


FormSpec = Mapping[str, NodeForm]  # keyed by node; roots absent (no equation is fitted for roots)


def factor_name(factor: Factor) -> str:
    """Canonical column-name fragment: ``A`` for identity, ``tanh(X1)`` otherwise."""
    return factor.parent if factor.transform == IDENTITY else f"{factor.transform}({factor.parent})"


def term_spec_name(term: TermSpec) -> str:
    """Canonical coefficient-dict key for a design column (factors joined by ``*``)."""
    return "*".join(factor_name(f) for f in term.factors)


def interaction_term_name(a: str, b: str) -> str:
    """Canonical coefficient-dict key for the elementwise-product column a·b."""
    return f"{a}*{b}"


def resolve_terms(node_form: NodeForm) -> tuple[TermSpec, ...]:
    """The node's design columns, in deterministic order.

    Explicit ``terms`` win verbatim; otherwise the linear-family default is
    reconstructed — parents as identity main effects, then each interaction as a
    two-identity-factor product. The reconstruction reproduces the linear-family
    column set and naming EXACTLY (``A``, ``X1``, ``A*X1``), so the linear and
    collider templates are unaffected by the generalization.
    """
    if node_form.terms is not None:
        return tuple(node_form.terms)
    return (
        *(TermSpec((Factor(p),)) for p in node_form.parents),
        *(TermSpec((Factor(a), Factor(b))) for a, b in node_form.interactions),
    )


def term_names(node_form: NodeForm) -> list[str]:
    """Deterministic design-matrix column order, as coefficient-dict keys."""
    return [term_spec_name(t) for t in resolve_terms(node_form)]


def evaluate_term(term: TermSpec, parent_values: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate one design column: product over factors of transform(parent).

    The single site where a form template's transform is APPLIED — shared by the
    design matrix (fit time, linear.py) and by the reconstructed structural
    equation (counterfactual time, base.py), so the column the estimator fits and
    the term the estimated SCM propagates can never disagree.
    """
    column: np.ndarray | float = 1.0
    for factor in term.factors:
        values = np.asarray(parent_values[factor.parent], dtype=float)
        column = column * _TRANSFORMS[factor.transform](values)
    return np.asarray(column, dtype=float)
