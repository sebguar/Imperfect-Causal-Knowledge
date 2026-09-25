"""Aggregate-CSV schemas: the exact column sets, version-controllable.

Two frozen dataclasses document the exact column set (and dtype) of each
aggregate CSV, so the schema is testable independent of the production of an
actual CSV. Column names here ARE the CSV headers — any drift between
this module and a written file is a schema bug, caught by the conformance test
(tests/test_aggregation_layer.py, test 2).

    # two-artifact output schema, per-seed rows preserved. [seeds and
    # bands] aggregate_by_group.csv: long format, one row per
    # (cell × condition × group). aggregate_by_cell.csv: one row per
    # (cell × condition). Per-seed rows are preserved to support the
    # effect-size-with-bands reporting (the inference stance); cross-seed roll-up is deferred to
    # analysis time.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CellKeys:
    """One experimental cell = (topology × family × regime × seed).

    Cross-seed aggregation is deferred to analysis time; the seed is a cell key
    so per-seed rows survive into the aggregate CSVs (the inference stance).
    """

    topology: str
    family: str
    regime: str
    seed: int


@dataclass(frozen=True)
class ByGroupRow:
    """One row of aggregate_by_group.csv: per (cell × condition × group).

    Field order == CSV column order. ``group`` ∈ {−1, +1} (the centered A
    encoding). Float columns may carry NaN where a quantity is undefined by
    design (ΔB_g_vs_L2 at the L2 reference row) — NaN, never 0.

    ΔB_g_vs_L2 is NaN at c=L2 by convention (self-reference, not a computed
    zero); downstream code must handle. The row is the L2 reference against
    which every ΔB_g is measured, so ΔB_g(L2) = mean(g,L2) − mean(g,L2) is a
    self-difference that is not a comparison; storing NaN (rather than the
    algebraic 0) keeps a downstream mean or "no-effect" test from silently
    reading the reference cell as a genuine zero-effect observation. Locked as
    the NaN convention.
    """

    topology: str
    family: str
    regime: str
    seed: int
    condition: str
    group: int
    N_eligible: int
    N_valid: int
    realized_mean_cost: float
    believed_mean_cost: float
    realized_validity_rate: float
    believed_validity_rate: float
    mean_believed_realized_cf_displacement: float
    ΔB_g_vs_L2: float
    # ---- The common-found population triple. APPENDED, never interleaved: the
    # columns above keep their pre-existing positions and values so every single-seed reference-run
    # number reproduces bit-for-bit from the same per-individual CSVs. --------
    #: Individuals with a feasible action in THIS condition. Equals N_eligible
    #: wherever recourse never fails; the two diverge the moment it does, and
    #: N_eligible (not N_found) stays the validity denominator (the common-found population).
    N_found: int
    #: common-found PRIMARY population size: found in EVERY scored condition ∩ this group.
    N_common_found_threeway: int
    #: The common-found population PRIMARY cost: mean realized cost over the three-way common-found
    #: mask.
    realized_cost_common_found_threeway: float
    #: pairwise population size: found in this condition AND in L2.
    #: NaN on the L2 rows — the pairwise population is indexed by the PAIRING
    #: PARTNER c, so L2 carries one value per partner, not one value. Those live
    #: in the third artifact (PairwiseRow); NaN here follows the NaN convention
    #: (undefined-by-design is NaN, never a computed-looking number).
    N_common_found_pairwise_vs_L2: float
    #: pairwise cost over that population. NaN on L2 rows, same reason.
    realized_cost_common_found_pairwise_vs_L2: float
    #: common-found SECONDARY population size: realized-valid ∩ this group (== N_valid,
    #: stored under its common-found name so the secondary read needs no cross-reference).
    N_valid_subset: int
    #: The common-found population SECONDARY cost: outcome-conditioned mean over the valid subset.
    realized_cost_valid_subset: float


@dataclass(frozen=True)
class ByCellRow:
    """One row of aggregate_by_cell.csv: per (cell × condition).

    Field order == CSV column order. Δ_dist is populated at L0 only; at
    SCM-carrying conditions it is NaN — zero would falsely
    suggest computed-and-vanishing.
    """

    topology: str
    family: str
    regime: str
    seed: int
    condition: str
    N_eligible_total: int
    Δ_cost: float
    Δ_dist: float
    Gap_cost: float
    Gap_validity_A_neg: float
    Gap_validity_A_pos: float
    ValidityDisp: float
    # ---- The common-found population. Appended; the columns above are untouched. ------
    #: Three-way common-found members across both groups (the common-found population primary, cell
    #: total).
    N_common_found_threeway_total: int
    #: min over groups of N_common_found_threeway — the thin-cell EARLY WARNING.
    #: Reported and warned on below COMMON_FOUND_MIN_GROUP_WARN;
    #: never gated on here. How to handle a thin cell is an analysis-layer decision.
    N_common_found_threeway_min_group: int
    #: Δ_cost recomputed on the common-found PRIMARY population. Same orientation
    #: (mean(A=−1) − mean(A=+1), positive = A=−1 more burdened) — this is the
    #: existing metric on the common-found population's primary population, not a new metric.
    Δ_cost_common_found_threeway: float
    #: Δ_cost on the common-found population SECONDARY (valid-subset) population,
    #: outcome-conditioned.
    Δ_cost_valid_subset: float


@dataclass(frozen=True)
class PairwiseRow:
    """One row of aggregate_pairwise_{regime}.csv: per (cell × c × group), c ≠ L2.

    The common-found population. The pairwise-vs-L2 common-found population
    depends on WHICH condition c is being contrasted with L2, so the L2 side of
    the contrast carries one mean per pairing partner. That cannot live on
    aggregate_by_group.csv without breaking its one-row-per-(condition × group)
    contract, so the pairwise numbers get their own long-format artifact keyed
    by ``condition`` = the pairing partner c. The payoff is that ΔB_g(c) under
    the IDENTICAL-POPULATION rule is a single column read here, with both
    sides of the subtraction stored next to it and auditable.
    """

    topology: str
    family: str
    regime: str
    seed: int
    #: The pairing partner c (never "L2" — L2 is the reference side of every row).
    condition: str
    group: int
    #: |found(c) ∩ found(L2)| ∩ this group — identical on both sides by construction.
    N_common_found_pairwise_vs_L2: int
    #: Mean realized cost in condition c over the matched population.
    realized_cost_c_common_found: float
    #: Mean realized cost in L2 over the SAME matched population.
    realized_cost_L2_common_found: float
    #: ΔB_g(c) on the matched population = the two columns above, subtracted.
    #: No NaN-at-L2 case arises here (L2 is never a row key), so unlike
    #: ByGroupRow.ΔB_g_vs_L2 this column is finite everywhere by construction.
    ΔB_g_vs_L2_common_found: float


def _columns(row_cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(row_cls))


#: Exact CSV column order for each artifact (the testable schema surface).
#: These are the CANONICAL-ARITY (three-condition) names — the exact strings every
#: artifact carries. Do not rename; a four-condition run gets its column
#: names from `by_group_columns(4)` / `by_cell_columns(4)` instead (see below).
BY_GROUP_COLUMNS: tuple[str, ...] = _columns(ByGroupRow)
BY_CELL_COLUMNS: tuple[str, ...] = _columns(ByCellRow)
PAIRWISE_COLUMNS: tuple[str, ...] = _columns(PairwiseRow)


# --------------------------------------------------------------------------- #
# Dynamic-arity common-found denominator suffix
# --------------------------------------------------------------------------- #

# [PS-4 note (1)] The common-found column suffix is derived
# from the NUMBER OF SCORED CONDITIONS through this pinned two-entry map — a
# LOOKUP, never a computed word. Two properties follow, and both are the point:
#
#   * a THREE-condition run emits the existing `_threeway` names byte-for-byte, so
#     every aggregate reproduces at HEAD with no migration and no rename;
#   * a FOUR-condition run emits `_fourway`, so a four-way number can NEVER sit
#     under a `threeway` label. That forecloses PS-4's own JOINT TABLES defect
#     ("an unlabelled mixed-denominator table is a defect") in the SCHEMA rather
#     than in reviewer vigilance: the denominator is legible from the header.
#
# ARITY 2 IS IN THE MAP because the codebase already produces two-condition cells:
# the collider and chain smoke cells both aggregate an L0+L2 run. No confirmatory
# number is read off a two-condition cell.
#
# An arity outside {2, 3, 4} RAISES. There is no five-rung ladder in this thesis,
# and a silent fallback would be exactly the unlabelled denominator the rule forbids.
_ARITY_SUFFIX: dict[int, str] = {2: "twoway", 3: "threeway", 4: "fourway"}

#: The arity the frozen dataclass field names above are written at.
CANONICAL_ARITY = 3
_CANONICAL_SUFFIX = _ARITY_SUFFIX[CANONICAL_ARITY]


def common_found_suffix(n_conditions: int) -> str:
    """Common-found column suffix for a run that scored ``n_conditions`` conditions."""
    suffix = _ARITY_SUFFIX.get(int(n_conditions))
    if suffix is None:
        raise ValueError(
            f"no common-found denominator label for {n_conditions} scored "
            f"condition(s); [PS-4 note (1)] the suffix map is pinned to "
            f"{_ARITY_SUFFIX} and an unmapped arity raises rather than emitting an "
            "unlabelled denominator."
        )
    return suffix


def _retarget(columns: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    """Rewrite the canonical `_threeway` segment of each column name to ``suffix``."""
    if suffix == _CANONICAL_SUFFIX:
        return columns  # identity at arity 3 — the byte-identity guarantee
    return tuple(c.replace(_CANONICAL_SUFFIX, suffix) for c in columns)


def by_group_columns(n_conditions: int = CANONICAL_ARITY) -> tuple[str, ...]:
    """aggregate_by_group.csv column order at the given condition arity."""
    return _retarget(BY_GROUP_COLUMNS, common_found_suffix(n_conditions))


def by_cell_columns(n_conditions: int = CANONICAL_ARITY) -> tuple[str, ...]:
    """aggregate_by_cell.csv column order at the given condition arity."""
    return _retarget(BY_CELL_COLUMNS, common_found_suffix(n_conditions))

#: Integer-dtype columns per artifact (everything else is str keys or float64).
#: N_common_found_pairwise_vs_L2 is deliberately ABSENT from the by-group int set:
#: it is NaN on L2 rows (see ByGroupRow), and NaN has no int64 representation.
BY_GROUP_INT_COLUMNS: tuple[str, ...] = (
    "seed",
    "group",
    "N_eligible",
    "N_valid",
    "N_found",
    "N_common_found_threeway",
    "N_valid_subset",
)
BY_CELL_INT_COLUMNS: tuple[str, ...] = (
    "seed",
    "N_eligible_total",
    "N_common_found_threeway_total",
    "N_common_found_threeway_min_group",
)


def by_group_int_columns(n_conditions: int = CANONICAL_ARITY) -> tuple[str, ...]:
    """Integer-dtype by-group columns at the given condition arity."""
    return _retarget(BY_GROUP_INT_COLUMNS, common_found_suffix(n_conditions))


def by_cell_int_columns(n_conditions: int = CANONICAL_ARITY) -> tuple[str, ...]:
    """Integer-dtype by-cell columns at the given condition arity."""
    return _retarget(BY_CELL_INT_COLUMNS, common_found_suffix(n_conditions))
PAIRWISE_INT_COLUMNS: tuple[str, ...] = (
    "seed",
    "group",
    "N_common_found_pairwise_vs_L2",
)
