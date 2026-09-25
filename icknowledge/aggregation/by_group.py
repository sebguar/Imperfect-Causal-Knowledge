"""aggregate_by_group.csv: per (cell × condition × group) roll-up.

Consumes the per-individual scoring CSVs (recourse/scoring.py schema) — the
source of truth — and rolls them up into one long-format row per
(cell × condition × group). All means are plain numpy means over the eligible
set; no pandas group-by, so the layer boundary stays auditable against the
manual-recompute acceptance gate (tests/test_aggregation_layer.py, test 1).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from icknowledge.aggregation.common_found import common_found_populations
from icknowledge.aggregation.schema import (
    PAIRWISE_COLUMNS,
    CellKeys,
    by_group_columns,
    common_found_suffix,
)

_GROUPS = (-1, 1)  # centered A encoding; A=−1 is the disadvantaged root position


def _mean_or_nan(values: np.ndarray) -> float:
    """Mean of ``values``, or NaN when empty — never 0, never an imputed number.

    # [the NaN convention] A mean over an empty population is UNDEFINED, and every
    # column in this layer that can be undefined-by-design carries NaN rather
    # than the algebraic-looking 0 that a downstream mean or no-effect test
    # would silently read as a genuine observation. Also suppresses numpy's
    # empty-slice RuntimeWarning, which would otherwise fire once per thin
    # group across a 48-run grid and bury real warnings.
    """
    return float(np.mean(values)) if values.size else math.nan


def eligible_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The ELIGIBLE population: every negatively-classified individual scored.

    # [the common-found population] The validity DENOMINATOR, and it includes the not-found. An
    # eligible individual for whom no feasible action was returned is a recourse
    # FAILURE, not an absent observation, so it must sit in the denominator of
    # realized_validity_rate — otherwise a condition that fails to find an action
    # for half a group would report the validity of the lucky half and read as
    # healthy. The split is made explicit now so the common-found convention
    # holds the first time a run does fail to find.
    """
    return table


def found_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The COST population for one condition: individuals with a feasible action.

    # aggregation population. Mean realized cost is computed over ALL eligible individuals
    # who received a recommended action (``found``), regardless of whether the
    # action was realized-valid. Conditioning on realized-valid only would be a
    # selection artifact (PS-5) that would silently break H1 monotonicity. That
    # valid-conditioned mean is still emitted, as the common-found population's SECONDARY column,
    # so the
    # outcome-conditioned read is available without displacing the primary one.
    # Not-found rows are excluded here: their realized cost is NaN
    # (never imputes ∞), so they cannot enter a cost mean at all.
    """
    return table[table["found"]]


def cf_feature_names(table: pd.DataFrame) -> list[str]:
    """Feature names carried by the believed_cf_* / realized_cf_* vector columns."""
    return [c.removeprefix("believed_cf_") for c in table.columns if c.startswith("believed_cf_")]


def build_by_group_frame(
    tables: dict[str, pd.DataFrame],
    cell: CellKeys,
    protected: str = "A",
) -> pd.DataFrame:
    """Build the by-group aggregate frame for one cell across ``tables``' conditions.

    ``tables`` maps condition label -> per-individual scoring table. The L2
    condition MUST be present: it is the ΔB_g reference.

    Implemented as the two-pass write the spec mandates: pass 1 computes every
    per-(condition × group) aggregate except ΔB_g_vs_L2; pass 2 joins each row
    to its (cell, L2, group) sibling and fills ΔB_g in. A missing L2 sibling is
    a pipeline bug and raises loudly rather than silently emitting NaN.
    """
    if "L2" not in tables:
        raise ValueError(
            f"cell {cell}: no L2 condition among {sorted(tables)} — the ΔB_g_vs_L2 "
            "reference requires L2; a partial run is a pipeline bug here."
        )

    # [the common-found population] Both comparison-set populations for this cell, computed ONCE
    # from all the run's tables — they are cross-condition objects, so they
    # cannot be derived inside the per-condition loop below.
    populations = common_found_populations(tables)
    # [PS-4 note (1)] The denominator LABEL is derived from how many
    # conditions were actually scored, so the primary-population columns can never
    # carry a name that misstates their denominator. At three conditions this is
    # the literal "threeway" and every emitted column name is byte-identical to the
    # three-rung schema; at four it is "fourway".
    suffix = common_found_suffix(len(populations.conditions))

    # -- Pass 1: per-(condition × group) aggregates, ΔB_g left unfilled. --------
    records: list[dict] = []
    for condition, table in tables.items():
        eligible = eligible_rows(table)
        features = cf_feature_names(table)
        # The common-found population masks, aligned to THIS condition's row order by individual id.
        threeway_mask = populations.mask(table, populations.threeway)
        pairwise_mask = (
            populations.mask(table, populations.pairwise_vs_L2[condition])
            if condition != "L2"
            else None
        )
        for group in _GROUPS:
            in_group = (table[protected] == float(group)).to_numpy()
            sub_eligible = eligible[eligible[protected] == float(group)]
            sub = found_rows(table)
            sub = sub[sub[protected] == float(group)]
            if len(sub_eligible) == 0:
                # min_neg_per_group guarantees both groups have a pool; an
                # empty group here means the wrong table was handed in.
                raise ValueError(
                    f"cell {cell}, condition {condition}: no eligible rows for "
                    f"group {group} — aggregation input is broken."
                )
            # [the common-found population] N_eligible is the validity denominator and counts the
            # not-found; N_found is the cost-mean denominator. Identical on every
            # scored run (found = 100% everywhere), stored separately so
            # the first genuine recourse failure is visible rather than absorbed.
            n_eligible = int(len(sub_eligible))
            n_found = int(len(sub))
            n_valid = int(sub_eligible["realized_validity"].to_numpy().sum())
            # Means over the FULL found set, not the realized-valid subset.
            realized_mean_cost = float(np.mean(sub["realized_cost"].to_numpy()))
            # [the ℓ₂ intervention-cost convention; PS-6] believed_mean_cost == realized_mean_cost
            # at every
            # condition by construction: r^CAU = ‖δ‖ is a property of the ACTION,
            # so per-row believed cost equals realized cost for the chosen δ.
            # Stored anyway so the equality is auditable per row rather than assumed.
            believed_mean_cost = float(np.mean(sub["believed_cost"].to_numpy()))
            # believed_validity_rate is 1.0 at every condition by construction (the
            # generator only returns δ that flips its OWN model). Stored anyway so
            # ValidityGap_g(c) is a difference of stored columns, not
            # an implicit 1 − realized rate — if the believed flag ever becomes
            # non-trivial (partial-lever generators) the column is already there.
            # [the common-found population] Over the ELIGIBLE set, matching realized_validity_rate's
            # denominator: ValidityGap_g(c) is their difference, so a
            # not-found individual must be a believed-failure and a realized-failure
            # on the same footing or the gap would manufacture itself. scoring.py
            # writes believed_validity = 0 on not-found rows, so this is exact.
            believed_validity_rate = float(
                np.mean(sub_eligible["believed_validity"].to_numpy())
            )
            # [PS-6] believed-vs-realized CF displacement diagnostic: zero at
            # L2 and on L0 full-acted rows; nonzero on L0 partial-acted rows and at
            # L1-oracle (f̂_L1 maps δ to a different downstream point than f_true).
            # Over FOUND rows only — both CF vectors are NaN without an action.
            believed = sub[[f"believed_cf_{f}" for f in features]].to_numpy()
            realized = sub[[f"realized_cf_{f}" for f in features]].to_numpy()
            displacement = float(
                np.mean(np.sqrt(np.sum((believed - realized) ** 2, axis=1)))
            )

            # -- [common-found] the population triple. ---------------------------
            # PRIMARY: three-way common-found ∩ this group. Every member is found
            # in this condition by the mask's own definition, so no extra `found`
            # conjunct is needed — but it is written explicitly anyway so the
            # invariant is visible at the point of use rather than inferred.
            threeway_sel = threeway_mask & in_group & table["found"].to_numpy()
            n_common_threeway = int(threeway_sel.sum())
            cost_threeway = _mean_or_nan(table["realized_cost"].to_numpy()[threeway_sel])
            # PAIRWISE: NaN on L2 rows, whose value is partner-indexed and
            # lives in the pairwise artifact (see schema.PairwiseRow).
            if pairwise_mask is None:
                n_common_pairwise: float = math.nan
                cost_pairwise = math.nan
            else:
                pairwise_sel = pairwise_mask & in_group & table["found"].to_numpy()
                n_common_pairwise = float(pairwise_sel.sum())
                cost_pairwise = _mean_or_nan(
                    table["realized_cost"].to_numpy()[pairwise_sel]
                )
            # SECONDARY: valid subset, outcome-conditioned. NaN (never 0) when a
            # group realizes no valid action at all — a mean over an empty set is
            # undefined, and 0 would read as "free recourse" (never impute).
            valid_sel = in_group & (table["realized_validity"].to_numpy() == 1)
            cost_valid = _mean_or_nan(table["realized_cost"].to_numpy()[valid_sel])

            records.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "seed": cell.seed,
                    "condition": condition,
                    "group": group,
                    "N_eligible": n_eligible,
                    "N_valid": n_valid,
                    "realized_mean_cost": realized_mean_cost,
                    "believed_mean_cost": believed_mean_cost,
                    "realized_validity_rate": n_valid / n_eligible,
                    "believed_validity_rate": believed_validity_rate,
                    "mean_believed_realized_cf_displacement": displacement,
                    "N_found": n_found,
                    f"N_common_found_{suffix}": n_common_threeway,
                    f"realized_cost_common_found_{suffix}": cost_threeway,
                    "N_common_found_pairwise_vs_L2": n_common_pairwise,
                    "realized_cost_common_found_pairwise_vs_L2": cost_pairwise,
                    "N_valid_subset": n_valid,
                    "realized_cost_valid_subset": cost_valid,
                }
            )

    # -- Pass 2: fill ΔB_g_vs_L2 via the (topology, family, regime, seed, group)
    #    join key against the L2 sibling. ------------------------------------------
    l2_reference = {
        (r["topology"], r["family"], r["regime"], r["seed"], r["group"]): r[
            "realized_mean_cost"
        ]
        for r in records
        if r["condition"] == "L2"
    }
    for r in records:
        # ΔB_g(c) := realized_mean_cost(g, c) − realized_mean_cost(g, L2),
        # BOTH means over the same aggregation population (all eligible),
        # so ΔB_g is a pure condition effect, not a mixed condition-and-population
        # effect. [H4]
        # NaN at c=L2 by convention: the L2 row is the self-reference
        # (ΔB_g(L2) = mean(g,L2) − mean(g,L2)), not a comparison. NaN is the honest
        # signal that the row is not a computed zero-effect observation — storing
        # the algebraic 0 would let a downstream mean or no-effect test read the
        # reference cell as a genuine zero. NaN, not 0.
        if r["condition"] == "L2":
            r["ΔB_g_vs_L2"] = math.nan
            continue
        key = (r["topology"], r["family"], r["regime"], r["seed"], r["group"])
        if key not in l2_reference:
            raise ValueError(
                f"missing L2 sibling for join key {key} — a missing ΔB_g reference "
                "is a pipeline bug, not a data-availability condition."
            )
        r["ΔB_g_vs_L2"] = r["realized_mean_cost"] - l2_reference[key]

    return pd.DataFrame.from_records(
        records, columns=list(by_group_columns(len(populations.conditions)))
    )


def build_pairwise_frame(
    tables: dict[str, pd.DataFrame],
    cell: CellKeys,
    protected: str = "A",
) -> pd.DataFrame:
    """Build the pairwise-vs-L2 common-found frame: one row per (c ≠ L2) × group.

    # [the common-found population] ΔB_g(c) on the IDENTICAL population. Both means are
    # taken over found(c) ∩ found(L2) ∩ group — the SAME individuals on both
    # sides of the subtraction — so ΔB_g is a pure condition effect and cannot
    # absorb a population difference. This is a strictly different population
    # from the three-way common-found mask that H1's monotonicity series uses
    # (it is a superset: it does not require the individual to be found in the
    # third condition too), which is exactly why the common-found population's two comparison sets
    # get
    # two artifacts instead of one shared column.
    """
    populations = common_found_populations(tables)
    l2_table = tables["L2"]

    records: list[dict] = []
    for condition, population in populations.pairwise_vs_L2.items():
        table = tables[condition]
        mask_c = populations.mask(table, population)
        mask_l2 = populations.mask(l2_table, population)
        for group in _GROUPS:
            sel_c = mask_c & (table[protected] == float(group)).to_numpy()
            sel_l2 = mask_l2 & (l2_table[protected] == float(group)).to_numpy()
            n_c, n_l2 = int(sel_c.sum()), int(sel_l2.sum())
            if n_c != n_l2:
                # The two selections index the SAME individual set through two
                # tables; a count mismatch means the group labels disagree
                # across conditions for some individual — a pipeline bug, not a
                # data condition, so it raises rather than reporting a mismatched
                # subtraction as a finding.
                raise ValueError(
                    f"cell {cell}, pairing {condition} vs L2, group {group}: matched "
                    f"population sizes disagree ({n_c} vs {n_l2}) — the protected "
                    "attribute is not stable across conditions for some individual."
                )
            cost_c = _mean_or_nan(table["realized_cost"].to_numpy()[sel_c])
            cost_l2 = _mean_or_nan(l2_table["realized_cost"].to_numpy()[sel_l2])
            records.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "seed": cell.seed,
                    "condition": condition,
                    "group": group,
                    "N_common_found_pairwise_vs_L2": n_c,
                    "realized_cost_c_common_found": cost_c,
                    "realized_cost_L2_common_found": cost_l2,
                    "ΔB_g_vs_L2_common_found": cost_c - cost_l2,
                }
            )

    return pd.DataFrame.from_records(records, columns=list(PAIRWISE_COLUMNS))
