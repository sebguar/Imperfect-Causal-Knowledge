"""Model-agnostic recourse harness.

A single brute-force procedure over (classifier, causal_model, cost_fn,
action_space), the four knowledge conditions (L0 associational, L1-oracle
estimated SCM, L1-discovered estimated-on-discovered SCM, L2 true SCM), the ℓ₂
intervention cost, the per-individual scoring skeleton, and the G1 pulled-back
closed-form check. Three topologies (triangle, collider, chain) × the linear and
nonlinear-Gaussian families. L1-discovered is wired lazily in
`recourse/pipeline.py` and scored only when requested by name (PS-7, PS-4).
The G1 anchor applies to the LINEAR family only — its
closed form assumes a linear SCM and a linear classifier (the ℓ₂ intervention-cost
convention).
"""

from icknowledge.recourse.action_space import ActionSpace, build_action_space
from icknowledge.recourse.anchor import (
    AnchorReport,
    anchor_smoke_check,
    closed_form_min_cost,
    grid_diagonal_step,
    raw_euclidean_reach,
)
from icknowledge.recourse.cost import intervention_cost, intervention_cost_batch
from icknowledge.recourse.model_conditions import (
    L0AssociationalModel,
    L1DiscoveredEstimatedSCMModel,
    L1OracleEstimatedSCMModel,
    L2TrueSCMModel,
)
from icknowledge.recourse.procedure import RecourseSolution, generate_recourse
from icknowledge.recourse.scoring import score_condition, summarize

__all__ = [
    "ActionSpace",
    "build_action_space",
    "intervention_cost",
    "intervention_cost_batch",
    "L0AssociationalModel",
    "L1DiscoveredEstimatedSCMModel",
    "L1OracleEstimatedSCMModel",
    "L2TrueSCMModel",
    "RecourseSolution",
    "generate_recourse",
    "score_condition",
    "summarize",
    "AnchorReport",
    "anchor_smoke_check",
    "closed_form_min_cost",
    "raw_euclidean_reach",
    "grid_diagonal_step",
]
