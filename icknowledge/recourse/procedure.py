"""The single standardized recourse procedure.

ONE brute-force / grid search, model-agnostic over (classifier, causal_model,
cost_fn, action_space). The SAME procedure runs for every knowledge condition so
that cross-condition differences are attributable to the causal MODEL alone, not
to the optimizer.

    # ONE procedure across all conditions so differences are
    # attributable to the causal model alone. Model-agnostic + gradient-free (admits
    # non-linear h later).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from icknowledge.classifier.training import ClassifierResult
from icknowledge.recourse.action_space import ActionSpace
from icknowledge.recourse.cost import intervention_cost_batch


@dataclass
class RecourseSolution:
    """The BELIEVED recourse solution for one individual under one causal model.

    The realized side (true-SCM push-through) is added later by scoring.py — this
    dataclass carries only what the procedure itself decides.
    """

    index: int  # full-dataset row index of the individual (from the negative pool)
    group: float  # protected attribute A (for group-wise scoring; never acted on)
    acted: tuple[str, ...]  # actionable variables' column order
    delta: dict[str, float] | None  # chosen min-cost believed-valid delta; None if none found
    believed_cost: float  # ℓ₂ intervention cost of the chosen delta (nan if none found)
    believed_validity: int  # 1 iff a grid delta flips h under the causal model, else 0


def _h_positive(model: LogisticRegression, feature_matrix: np.ndarray) -> np.ndarray:
    """Boolean mask: does h classify each feature row as positive?

    Uses the decision function > 0, which is exactly LogisticRegression.predict == 1
    for the binary group-blind h (raw feature units — no scaler in between).
    """
    return model.decision_function(feature_matrix) > 0.0


def generate_recourse(
    classifier_result: ClassifierResult,
    causal_model,
    action_space: ActionSpace,
    data: pd.DataFrame,
) -> list[RecourseSolution]:
    """Run the brute-force grid search for every negatively-classified individual.

    # classifier h held constant across all conditions — h is the fixed
    # group-blind LR, ONE per SCM instance, reused across L0/L1-oracle/L2.
    # Raw-units decision: operate in RAW feature units throughout (no scaling); cost units ==
    # classifier units == raw feature units (the Ehyaei anchor).

    For each individual in the full-dataset negative pool, sweep the joint
    grid; for each candidate delta, form the BELIEVED counterfactual features via
    ``causal_model.predict``; keep the minimum-cost delta whose believed features
    flip h (believed validity).
    """
    model = classifier_result.model
    protected = classifier_result.protected_attr
    acted = action_space.acted

    # Grid deltas and their ℓ₂ costs are dataset-level (deltas relative to factual),
    # so they are identical across individuals — precompute once.
    candidates = action_space.candidate_matrix()  # (K, |acted|)
    costs = intervention_cost_batch(candidates)  # (K,)

    solutions: list[RecourseSolution] = []
    for idx in classifier_result.negative_pool_indices:
        factual = data.iloc[int(idx)]
        # BELIEVED counterfactual features for every candidate under this causal model.
        believed_cf = causal_model.predict_batch(factual, candidates)  # (K, F)
        valid = _h_positive(model, believed_cf)  # (K,) believed validity per candidate

        if not valid.any():
            # No valid action in the grid — flag, believed_validity=0, do NOT
            # impute an infinite cost (that would poison downstream means).
            solutions.append(
                RecourseSolution(
                    index=int(idx),
                    group=float(factual[protected]),
                    acted=acted,
                    delta=None,
                    believed_cost=float("nan"),
                    believed_validity=0,
                )
            )
            continue

        # Minimum-cost delta among the believed-valid candidates.
        masked_costs = np.where(valid, costs, np.inf)
        best = int(np.argmin(masked_costs))
        best_delta = {var: float(candidates[best, i]) for i, var in enumerate(acted)}
        solutions.append(
            RecourseSolution(
                index=int(idx),
                group=float(factual[protected]),
                acted=acted,
                delta=best_delta,
                believed_cost=float(costs[best]),
                believed_validity=1,
            )
        )
    return solutions
