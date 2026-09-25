"""Unregularized-OLS coefficient estimator for the L1-oracle rung.

One OLS fit per non-root endogenous node, on the supplied form template
in RAW feature units (the raw-units convention). See base.py for the
form-template / mean-function-only decision comments; the unregularized-OLS and
shared-sample comments bind here, at the fit itself.

Named for the LINEAR family it shipped for, but the fit is valid for any family
that is LINEAR IN PARAMETERS on its supplied design columns — which, per the
NLG family specification, includes the nonlinear-Gaussian family after its fixed tanh(·)
transform.
The NLG estimator therefore adds a template and an estimator id, not new fitting
math (see nonlinear_gaussian.py).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from icknowledge.estimation.base import (
    FittedEquations,
    FormSpec,
    NodeForm,
    dataset_identity,
    equation_from_coefficients,
    evaluate_term,
    resolve_terms,
    term_names,
)
from icknowledge.scm.base import StructuralEquation

ESTIMATOR_ID = "LinearOLSEstimator[sklearn.LinearRegression(fit_intercept=True), unregularized]"


def design_matrix(data: pd.DataFrame, node_form: NodeForm) -> tuple[np.ndarray, list[str]]:
    """Design matrix for one node, SHARED between fit and tests: (X, column names).

    One column per RESOLVED term of the template, in its deterministic order: for
    a linear template that is the parents followed by each interaction's
    elementwise product; for a transformed template (NLG, the NLG family specification) it is the
    product
    of the transformed factors, e.g. tanh(X₁) and A·tanh(X₁). The intercept is NOT
    a column here — it enters via ``fit_intercept=True``. NO standardization,
    NO centering, NO scaling: raw feature units throughout, so cost units ==
    classifier units == closed-form units (the raw-units convention — a correctness invariant, not a
    preference). A term absent from the template is HARD-ZERO by OMISSION from
    this matrix, never fit-then-discarded.
    """
    parent_values = {p: data[p].to_numpy(dtype=float) for p in node_form.parents}
    columns = [evaluate_term(term, parent_values) for term in resolve_terms(node_form)]
    return np.column_stack(columns), term_names(node_form)


class LinearOLSEstimator:
    """Unregularized OLS per node. Design matrix in raw units (the raw-units convention).

    # OLS via sklearn.linear_model.LinearRegression, fit_intercept=True,
    # NO regularization. [the form-template-known semantics; H1]
    # Ridge/Lasso/ElasticNet are PROHIBITED here: regularization injects bias into
    # the coefficient estimates and would contradict the "L1-oracle isolates
    # finite-sample estimation error" semantics — a penalized fit is a DIFFERENT
    # (biased) estimator, not a noisier version of the true one. Any future swap of
    # the estimator class must be justified against this line (the lstsq guard in
    # tests/test_l1_oracle_estimation.py fails loudly if it drifts).
    """

    name = ESTIMATOR_ID

    def fit(
        self,
        data: pd.DataFrame,
        form: FormSpec,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> FittedEquations:
        """Fit one OLS per templated node on ``data``; return the fitted bundle.

        # Estimation sample = the classifier's training sample; NO held-out
        # split. No leakage risk: the
        # generator sees only features, never labels. ``data`` is the dataset
        # OBJECT produced upstream (e.g. ClassifierResult.dataset) — deliberately
        # not a seed or a re-sampling call, so the L1-discovered PC step can later
        # run on THIS SAME sample; a different or re-drawn sample would introduce a
        # sample-composition confound between L1-oracle and L1-discovered that
        # corresponds to no hypothesis in H1–H4. The sample's identity (hash) is
        # recorded in the metadata for the manifest.
        """
        coefficients: dict[str, dict[str, float]] = {}
        equations: dict[str, StructuralEquation] = {}
        for node, node_form in form.items():
            if node_form.node != node:
                raise ValueError(
                    f"form spec key {node!r} disagrees with NodeForm.node {node_form.node!r}."
                )
            y = data[node].to_numpy(dtype=float)
            if not term_names(node_form):
                # ZERO-TERM FORM — an intercept-only node. [PS-7 note (b)]
                # Unreachable from any L0/L1-oracle/L2 template (every
                # oracle NodeForm has at least one column), so the frozen rungs
                # are numerically untouched; it exists for the L1-discovered
                # path, where a node the search left parentless has no design
                # column at all. The unregularized-OLS solution with an intercept
                # and no regressors IS the sample mean, so this is the SAME
                # estimator, not a second one — computed directly because
                # `design_matrix` would np.column_stack([]) and `LinearRegression`
                # would have to be trusted to fit an (n, 0) array identically
                # across sklearn versions.
                node_coefs = {"intercept": float(y.mean())}
            else:
                x, terms = design_matrix(data, node_form)
                # The unregularized rule binds here: plain OLS, intercept via
                # fit_intercept, no penalty.
                reg = LinearRegression(fit_intercept=True).fit(x, y)
                node_coefs = {"intercept": float(reg.intercept_)}
                node_coefs.update(
                    {term: float(c) for term, c in zip(terms, reg.coef_, strict=True)}
                )
            coefficients[node] = node_coefs
            # See base.py: mean function only — the equation carries f̂(pa)
            # plus the pass-through additive noise argument; no σ̂² is estimated.
            equations[node] = equation_from_coefficients(node_form, node_coefs)

        metadata: dict = {
            "estimator": self.name,
            "dataset": dataset_identity(data),  # auditable sample identity
        }
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        return FittedEquations(
            coefficients=coefficients, equations=equations, form=form, metadata=metadata
        )
