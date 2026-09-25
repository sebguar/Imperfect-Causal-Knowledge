"""Nonlinear-Gaussian (NLG) coefficient estimator for the L1-oracle rung.

# [the L1-oracle form-template-known semantics] The NLG family is LINEAR IN PARAMETERS after the
# fixed
# tanh(·) transform its builders apply to the relevant parent
# (X₂ := b·A + g·tanh(X₁) + U₂ additive; X₂ := g_of_A·tanh(X₁) + U₂
# effect-modifying — verified against `make_nonlinear_gaussian_triangle` in
# scm/triangles.py, not from a summary). The NLG family specification therefore carries the
# L1-oracle form-template-known semantics's
# unregularized-OLS semantics over to this family with NO form-template-known extension row: the
# nonlinearity lives entirely in the SUPPLIED design columns (form template
# supplied, never inferred or selected), so the fit itself is the same
# unregularized `LinearRegression(fit_intercept=True)` least-squares solve,
# on the same estimation sample (the classifier's training sample, no held-out
# split), fitting the mean function only (no σ̂²).
#
# What this module deliberately does NOT do: nonlinear least squares. If an NLG
# builder were ever written so that it is NOT linear-in-parameters after a fixed
# transform, an NLS estimator is out of scope — the estimator class would then no longer be
# the form-template-known one.
#
# The class exists (rather than the pipeline just reusing LinearOLSEstimator with
# an NLG template) for two reasons: it satisfies the pluggable-per-family
# `EquationEstimator` interface the estimation layer is built around, and it
# stamps a DISTINCT estimator id into the fit metadata, so the run manifest
# records which family's estimator produced an L1-oracle rung. It inherits the
# fit verbatim so the two families can never drift to different fitting
# semantics — the NLG-family claim is that there is nothing to differ.
"""

from __future__ import annotations

from icknowledge.estimation.linear import LinearOLSEstimator

ESTIMATOR_ID = (
    "NLGOLSEstimator[sklearn.LinearRegression(fit_intercept=True), unregularized; "
    "linear-in-parameters after the supplied fixed tanh transform, the NLG family specification]"
)


class NLGOLSEstimator(LinearOLSEstimator):
    """Unregularized OLS on the NLG form template's transformed design columns.

    Behaviourally identical to `LinearOLSEstimator` by design (the NLG family specification): only
    the
    SUPPLIED form template differs, and templates are the estimator's input, not
    its implementation. Overriding anything else here would break the NLG-family claim.
    """

    name = ESTIMATOR_ID
