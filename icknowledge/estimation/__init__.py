"""Equation estimation for the L1 knowledge conditions."""

from icknowledge.estimation.base import (
    EquationEstimator,
    Factor,
    FittedEquations,
    FormSpec,
    NodeForm,
    TermSpec,
    build_estimated_scm,
    dataset_identity,
    equation_from_coefficients,
    estimation_provenance,
    evaluate_term,
    interaction_term_name,
    resolve_terms,
    term_names,
    term_spec_name,
)
from icknowledge.estimation.discovered import build_estimated_scm_from_discovery
from icknowledge.estimation.linear import LinearOLSEstimator, design_matrix
from icknowledge.estimation.nonlinear_gaussian import NLGOLSEstimator

__all__ = [
    "EquationEstimator",
    "Factor",
    "FittedEquations",
    "FormSpec",
    "LinearOLSEstimator",
    "NLGOLSEstimator",
    "NodeForm",
    "TermSpec",
    "build_estimated_scm",
    # [PS-4] The L1-discovered counterpart of `build_estimated_scm`, promoted
    # from a leading-underscore name so the pipeline imports it like any other
    # public assembly entry point. Deliberately a SIBLING, not a branch of
    # `build_estimated_scm` — see the module note in estimation/discovered.py on
    # why the two assemblies stay duplicated rather than shared.
    "build_estimated_scm_from_discovery",
    "dataset_identity",
    "design_matrix",
    "equation_from_coefficients",
    "estimation_provenance",
    "evaluate_term",
    "interaction_term_name",
    "resolve_terms",
    "term_names",
    "term_spec_name",
]
