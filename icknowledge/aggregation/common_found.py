"""Common-found population filter.

PS-5 pins H1's realized-cost aggregate population as a triple: primary =
realized cost over the COMMON-FOUND population (individuals for whom a feasible
action is returned in every condition of the comparison); secondary = the VALID
SUBSET, outcome-conditioned; always alongside = validity-rate-by-group.

This module owns the first leg: it turns a cell's per-condition scoring tables
into the two population masks the H1 monotonicity series and the ΔB_g(c)
contrasts are averaged over. They are DIFFERENT populations and both are emitted:

  (a) THREE-WAY, for the H1 monotonicity series: intersection of `found` across
      ALL conditions scored in the run — "all conditions present", never a
      hardcoded triple.
  (b) PAIRWISE-vs-L2, per non-L2 condition c, for the ΔB_g(c) contrasts:
      intersection of `found` in c AND `found` in L2.

(b) is a superset of (a) for every c. Conflating them would let an H1 reading and
a ΔB_g reading silently disagree about who was averaged. Both masks are built on
`found`, never on `valid`: an individual with no feasible action is EXCLUDED from
that condition's cost mean and COUNTED as a failure in its validity denominator,
while a found-but-invalid action keeps a well-defined realized cost. Both live on
the ELIGIBLE population. `by_group` / `by_cell` consume the masks; nothing here
computes a mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Column carrying the individual's identity in the source dataframe — the join
#: key across conditions (scoring.py writes it per row).
INDIVIDUAL_KEY = "index"

#: The ΔB_g reference condition: every pairwise contrast is against it.
L2 = "L2"

#: The common-found population sanity floor for the primary population: a warning
#: threshold only. NOTHING in the aggregation gates on it — how to handle a thin cell is an
#: analysis-layer decision, so this constant is read by the reporting path and never by a
#: control-flow branch that drops or rewrites a row.
COMMON_FOUND_MIN_GROUP_WARN = 30


@dataclass(frozen=True)
class CommonFoundPopulations:
    """The common-found comparison-set populations for one cell, as individual-id sets.

    Stored as id sets rather than per-table boolean arrays so a consumer cannot
    accidentally apply a mask built from condition A's row order to condition
    B's table. `mask` does the alignment explicitly.
    """

    #: Condition labels the run actually scored. Its LENGTH is what
    #: `schema.common_found_suffix` turns into the emitted column suffix
    #: (PS-4 note (1)), so this tuple is the single source of truth
    #: for both the population and its denominator label.
    conditions: tuple[str, ...]
    #: Individuals found in EVERY scored condition — the common-found primary population.
    #
    # THE FIELD NAME IS AN INTERNAL ALIAS, NOT A DENOMINATOR CLAIM: the
    # intersection is over whatever was actually scored, so at four rungs this
    # holds the FOUR-way population. The emitted column suffix carries the true
    # arity (PS-4 note (1)); an in-process attribute is read next to the
    # `conditions` tuple that defines it, whereas a column name travels alone.
    threeway: frozenset[int]
    #: c -> individuals found in BOTH c and L2 (one entry per non-L2 condition).
    pairwise_vs_L2: dict[str, frozenset[int]]

    def mask(self, table: pd.DataFrame, population: frozenset[int]) -> np.ndarray:
        """Boolean array over ``table``'s rows selecting ``population``'s members."""
        return table[INDIVIDUAL_KEY].isin(population).to_numpy()


def _found_ids(table: pd.DataFrame) -> frozenset[int]:
    """Individual ids with a feasible action in this condition (`found` true)."""
    return frozenset(int(i) for i in table.loc[table["found"], INDIVIDUAL_KEY])


def common_found_populations(tables: dict[str, pd.DataFrame]) -> CommonFoundPopulations:
    """Build both the common-found population comparison-set populations from one cell's scoring
    tables.

    ``tables`` maps condition label -> per-individual scoring table (the same
    mapping `build_by_group_frame` consumes). Every table must cover the SAME
    eligible individuals — that is what makes the populations "matched across
    conditions" in the common-found population's sense; a mismatch is a pipeline bug and raises here
    rather than silently shrinking the primary population.
    """
    if L2 not in tables:
        raise ValueError(
            f"no {L2} condition among {sorted(tables)} — the common-found pairwise "
            f"comparison set is defined against {L2}; a run without it "
            "cannot produce the ΔB_g population."
        )

    conditions = tuple(tables)
    id_sets = {c: frozenset(int(i) for i in t[INDIVIDUAL_KEY]) for c, t in tables.items()}
    reference = id_sets[conditions[0]]
    for condition, ids in id_sets.items():
        if ids != reference:
            # Matched-across-conditions is the common-found population's whole point: if the
            # eligible
            # sets differ, "common-found" would be silently confounded with
            # "commonly eligible" and the H1 series would compare populations.
            raise ValueError(
                f"condition {condition!r} scores {len(ids)} individuals but "
                f"{conditions[0]!r} scores {len(reference)} — the common-found "
                "population requires an identical eligible set across conditions."
            )

    found = {c: _found_ids(t) for c, t in tables.items()}

    # (a) three-way: intersection over ALL scored conditions (not a hardcoded triple).
    threeway = frozenset.intersection(*found.values())

    # (b) pairwise-vs-L2: one population per non-L2 condition.
    pairwise = {c: found[c] & found[L2] for c in conditions if c != L2}

    return CommonFoundPopulations(
        conditions=conditions, threeway=threeway, pairwise_vs_L2=pairwise
    )
