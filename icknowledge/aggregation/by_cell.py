"""aggregate_by_cell.csv: per (cell × condition) fairness roll-up.

Derives the cell-level disparities from the by-group frame (so the two artifacts
can never disagree on a mean) plus the per-individual tables (only for Δ_dist at
L0, which needs the realized CF vectors). The per-group ValidityGap columns live
HERE, not on aggregate_by_group.csv: they are the H2 monotone signal at the cell
level, so the H2 story reads off one CSV without cross-table joins.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from icknowledge.aggregation.by_group import cf_feature_names, found_rows
from icknowledge.aggregation.schema import CellKeys, by_cell_columns, common_found_suffix


def _l0_delta_dist(table: pd.DataFrame, protected: str) -> float:
    """Δ_dist at L0 over the eligible population.

    # Δ_dist at L0 only, in the aggregation layer. This is the Δ_dist FOIL: the true-SCM
    # displacement ‖v − realized_cf‖ — the naive auditor's measurement of how
    # far individuals actually moved in feature space. It differs from Δ_cost
    # (= ‖δ‖, the action cost) exactly on the partial-acted exception set, where
    # the true SCM propagates an unacted axis that the action cost doesn't count
    # (the L0 believed map J = I would give
    # ‖v − believed_cf‖ ≡ ‖δ‖ identically, carrying no foil signal at all).
    # Δ_dist at L2/L1-oracle is an H2/H3 story and is stored as NaN, not zero.
    """
    # FOUND rows: a not-found individual has no realized CF vector to displace
    # (it is NaN — never imputed), so it cannot enter a displacement mean.
    eligible = found_rows(table)
    features = cf_feature_names(table)
    factual = eligible[[f"factual_{f}" for f in features]].to_numpy()
    realized_cf = eligible[[f"realized_cf_{f}" for f in features]].to_numpy()
    displacement = np.sqrt(np.sum((factual - realized_cf) ** 2, axis=1))
    a = eligible[protected].to_numpy()
    return float(np.mean(displacement[a == -1.0]) - np.mean(displacement[a == 1.0]))


def build_by_cell_frame(
    tables: dict[str, pd.DataFrame],
    by_group: pd.DataFrame,
    cell: CellKeys,
    protected: str = "A",
) -> pd.DataFrame:
    """Build the by-cell aggregate frame from the by-group frame + raw tables."""
    # [PS-4 note (1)] Same arity-derived denominator label the
    # by-group frame was written with — read off ``tables`` (the run's scored
    # condition set), which is exactly what `common_found_populations` intersected
    # over, so the two artifacts cannot disagree about their denominator.
    suffix = common_found_suffix(len(tables))
    records: list[dict] = []
    for condition in by_group["condition"].unique():
        rows = by_group[by_group["condition"] == condition]
        neg = rows[rows["group"] == -1].iloc[0]
        pos = rows[rows["group"] == 1].iloc[0]

        # Δ_cost sign convention (a priori). [PS-3 extension]
        # Δ_cost := mean(cost | A=−1) − mean(cost | A=+1);
        # positive = A=−1 more burdened. Grounded in the strong-root convention
        # that assigns A=−1 the disadvantaged position by construction — NOT
        # selected from pilot numbers. A cell where A=+1 is more burdened shows a
        # negative Δ_cost: a finding, not a sign flip.
        delta_cost = float(neg["realized_mean_cost"] - pos["realized_mean_cost"])

        if condition == "L0":
            delta_dist = _l0_delta_dist(tables[condition], protected)
            # [PS-6; H2 amendment] At L0 the RP switches the believed
            # side to Δ_dist (no SCM is available), so Gap_cost = |Δ_dist − Δ_cost|.
            delta_believed = delta_dist
        else:
            delta_dist = math.nan
            # Gap_c on cost collapses under the ℓ₂ intervention-cost convention.
            # [PS-6; the ℓ₂ intervention-cost convention] At every SCM-carrying condition
            # believed cost ==
            # realized cost per row for the chosen δ, so this difference is
            # IDENTICALLY zero — computed from the stored believed means rather than
            # hard-coded, so the collapse is auditable per row instead of assumed.
            # The monotone H2 signal lives in the ValidityGap columns below, not here.
            delta_believed = float(neg["believed_mean_cost"] - pos["believed_mean_cost"])

        records.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "seed": cell.seed,
                "condition": condition,
                "N_eligible_total": int(neg["N_eligible"] + pos["N_eligible"]),
                "Δ_cost": delta_cost,
                "Δ_dist": delta_dist,
                # No sign: this is the H2 gap, Gap_c := |Δ_believed − Δ_realized|.
                "Gap_cost": abs(delta_believed - delta_cost),
                # ValidityGap_g(c) := believed_validity_rate_g(c) −
                # realized_validity_rate_g(c), the per-group believed-vs-realized
                # validity gap: the direct "auditor overpromises" quantity that
                # preserves H2's monotone story under PS-6. Signed: positive =
                # believed exceeds realized. Zero at L2 by construction; grows at
                # L1-oracle and L0.
                "Gap_validity_A_neg": float(
                    neg["believed_validity_rate"] - neg["realized_validity_rate"]
                ),
                "Gap_validity_A_pos": float(
                    pos["believed_validity_rate"] - pos["realized_validity_rate"]
                ),
                # ValidityDisp(c) := realized_validity_rate(A=−1, c) −
                # realized_validity_rate(A=+1, c): CROSS-group realized-validity
                # disparity, a fairness diagnostic parallel in shape to Δ_cost —
                # reported alongside, not as a H2 signal. Signed to parallel the
                # Δ_cost orientation: negative = A=−1 realizes lower validity.
                "ValidityDisp": float(
                    neg["realized_validity_rate"] - pos["realized_validity_rate"]
                ),
                # -- [the common-found population] cell-level view of the population triple. -----
                f"N_common_found_{suffix}_total": int(
                    neg[f"N_common_found_{suffix}"] + pos[f"N_common_found_{suffix}"]
                ),
                # EARLY WARNING: the thin-primary-population signal. The primary
                # population keys on `found`, not `valid`, so this column is what
                # has to be watched; it is REPORTED, never gated on. The analysis
                # layer decides how to treat a thin cell (common-found is a
                # population definition, not a sample-size rule).
                f"N_common_found_{suffix}_min_group": int(
                    min(neg[f"N_common_found_{suffix}"], pos[f"N_common_found_{suffix}"])
                ),
                # Δ_cost on the common-found PRIMARY population. Same orientation
                # as Δ_cost above — the existing metric on the common-found population's primary
                # population, NOT a new metric. This is the number H1's
                # monotonicity series reads in the analysis pass.
                f"Δ_cost_common_found_{suffix}": float(
                    neg[f"realized_cost_common_found_{suffix}"]
                    - pos[f"realized_cost_common_found_{suffix}"]
                ),
                # Δ_cost on the common-found population SECONDARY (valid-subset) population.
                # Reported
                # alongside, never in place of, the two above: it is
                # outcome-conditioned, so on a low-validity cell it is a mean over
                # a selected minority (the selection artifact) and must be
                # read next to validity-rate-by-group.
                "Δ_cost_valid_subset": float(
                    neg["realized_cost_valid_subset"] - pos["realized_cost_valid_subset"]
                ),
            }
        )

    return pd.DataFrame.from_records(records, columns=list(by_cell_columns(len(tables))))
