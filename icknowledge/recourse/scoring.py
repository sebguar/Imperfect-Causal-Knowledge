"""Per-individual scoring skeleton.

Turns believed recourse solutions (procedure.py) into a per-individual × condition
scoring table by pushing each chosen action through the TRUE SCM (realized side)
and through the condition-held model (believed side). This module does
NOT aggregate fairness metrics — it only records the raw ingredients so Δ_cost
(primary), Δ_dist (foil) and Gap_c are derived in the aggregation layer.

    # fairness-scoring: realized = ALWAYS through the true SCM, whatever
    # condition generated the action. The true SCM is the fixed measuring ruler.
"""

from __future__ import annotations

import pandas as pd

from icknowledge.classifier.training import ClassifierResult
from icknowledge.recourse.cost import intervention_cost
from icknowledge.recourse.model_conditions import L2TrueSCMModel
from icknowledge.recourse.procedure import RecourseSolution


def score_condition(
    classifier_result: ClassifierResult,
    solutions: list[RecourseSolution],
    true_model: L2TrueSCMModel,
    believed_model,
    data: pd.DataFrame,
    condition_label: str,
) -> pd.DataFrame:
    """Score one condition's solutions against the true SCM, one row per individual.

    Records per individual: acted set + delta; believed/realized cost;
    believed/realized validity; believed/realized/factual vectors; group label A.

    ``true_model`` (the L2 true-SCM wrapper) realizes and scores EVERY condition,
    including L0. ``believed_model`` is the condition-held model that generated
    ``solutions``, used ONLY to record the believed counterfactual vector; scoring
    is the single site with access to both SCMs. ``believed_cf`` is ALWAYS routed
    through it, never hard-coded equal to ``realized_cf`` — a zero delta
    coordinate is NOT intervened, so a partial-acted L0 row diverges.

    [H2, the ℓ₂ intervention-cost convention] `realized_cf` is the canonical
    scoring target — validity, cost and burden are computed against it;
    `believed_cf` is diagnostic and drives the believed side of H2's Gap_c. At
    L1-oracle the two genuinely differ: `believed_cf = v + f̂_L1(δ)`,
    `realized_cf = v + f_true(δ)`.
    """
    model = classifier_result.model
    features = classifier_result.feature_names
    protected = classifier_result.protected_attr
    actionable = true_model.acted

    records: list[dict] = []
    for sol in solutions:
        factual = data.iloc[sol.index]
        factual_vec = factual[features].to_numpy(dtype=float)

        rec: dict = {
            "condition": condition_label,
            "index": sol.index,
            protected: float(factual[protected]),
            "acted_set": "",  # nonzero-delta intervened set (cost charged on S)
            "believed_cost": sol.believed_cost,
            "realized_cost": float("nan"),
            "believed_validity": sol.believed_validity,
            "realized_validity": 0,
            "found": sol.delta is not None,
        }
        for f in features:
            rec[f"delta_{f}"] = float("nan")
            rec[f"factual_{f}"] = float(factual_vec[features.index(f)])
            rec[f"realized_cf_{f}"] = float("nan")
            rec[f"believed_cf_{f}"] = float("nan")

        if sol.delta is None:
            # No believed-valid action found — validity 0 on both sides, costs
            # left as NaN (never imputed to ∞). Realized CF undefined (no action).
            records.append(rec)
            continue

        # realized_cf = true_SCM.predict(factual, chosen delta) — ALWAYS the true SCM.
        realized_cf = true_model.predict(factual, sol.delta)
        # believed_cf = condition_model.predict(factual, chosen delta) — the SAME
        # object the procedure searched with. [the ℓ₂ intervention-cost
        # convention; fairness metrics]
        # Believed cost == realized cost by construction for the chosen δ at EVERY
        # condition (cost is a property of the ACTION, not of the SCM that enacts
        # it), so the believed-vs-realized signal does NOT live in the cost column;
        # it lives in (i) which δ the generator selects, (ii) realized validity,
        # and (iii) these CF VECTORS — at L1-oracle the estimated SCM maps δ to a
        # different downstream point than the true SCM does.
        believed_cf = believed_model.predict(factual, sol.delta)
        # realized_validity = 1{ h(realized_cf) positive }.
        realized_validity = int(model.decision_function(realized_cf.reshape(1, -1))[0] > 0.0)
        # realized_cost = cost(factual, chosen delta). Charged on ACTED vars only;
        # by construction == believed_cost (same delta, same acted set, same norm).
        realized_cost = intervention_cost(sol.delta, actionable)
        # EXPECTED BEHAVIOUR (not a bug): believed_cost == realized_cost per
        # individual — the same delta on the same acted vars costs the same whichever
        # SCM enacts it. The believed-vs-realized signal lives in VALIDITY (L0's edit
        # may not flip on the true SCM) and in action CHOICE across conditions.
        acted_set = tuple(v for v in actionable if abs(sol.delta.get(v, 0.0)) > 0.0)

        rec["acted_set"] = ",".join(acted_set)
        rec["realized_cost"] = realized_cost
        rec["realized_validity"] = realized_validity
        for f in features:
            rec[f"delta_{f}"] = float(sol.delta.get(f, 0.0))
            rec[f"realized_cf_{f}"] = float(realized_cf[features.index(f)])
            rec[f"believed_cf_{f}"] = float(believed_cf[features.index(f)])
        records.append(rec)

    return pd.DataFrame.from_records(records)


def summarize(table: pd.DataFrame, protected: str = "A") -> pd.DataFrame:
    """Compact summary: validity-rate and realized-cost by group × condition.

    # fairness-scoring: validity-rate-by-group is reported by group
    # ALONGSIDE every cost metric. Realized-cost summaries are over VALID actions
    # only (never impute ∞ into means); the validity rate carries the rest.
    """
    rows: list[dict] = []
    for (condition, group), sub in table.groupby(["condition", protected], sort=True):
        valid = sub[sub["realized_validity"] == 1]
        cost_mean = float(valid["realized_cost"].mean()) if len(valid) else float("nan")
        cost_std = float(valid["realized_cost"].std(ddof=0)) if len(valid) else float("nan")
        rows.append(
            {
                "condition": condition,
                protected: group,
                "n": len(sub),
                "believed_validity_rate": float(sub["believed_validity"].mean()),
                "realized_validity_rate": float(sub["realized_validity"].mean()),
                "n_realized_valid": len(valid),
                "realized_cost_mean": cost_mean,
                "realized_cost_std": cost_std,
            }
        )
    return pd.DataFrame.from_records(rows)
