"""The confirmatory analysis pass over the cross-seed grid: H1 monotonicity,
H2 validity gap, detectability gate.

Every function is a pure transform of the grid summary frames, so the numbers can
be regression-tested on synthetic frames without a results tree present.

Conventions:
  * SD is the SAMPLE SD over seeds (ddof=1), matching the grid runner's own summarizer.
  * The H1 verdict quantity is realized cost over the PS-5 COMMON-FOUND
    three-way population; the all-eligible legacy column is carried alongside
    only to compute the no-op equality check the H1 monotonicity read discloses
    as a finding.
  * Nothing here writes thesis prose — the analysis layer produces numbers and verdict flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icknowledge.analysis.loading import (
    CONDITIONS,
    N_SEEDS,
    VALID_SUBSET_MIN_N,
    Cell,
    cell_frame,
    grid_cells,
)

#: The common-found population primary DV column and its all-eligible
#: counterpart.
PRIMARY_COST = "realized_cost_common_found_threeway"
LEGACY_COST = "realized_mean_cost"

GROUPS = (-1.0, 1.0)

#: [PS-1 / PS-2] The detectability gate is defined on the FIRST-6-SEED early
#: look and is never re-evaluated on a growing seed count. This is a CONSTANT, not a
#: default derived from the tree being read: on the N=20 tree the gate is computed
#: over the seed-0..5 prefix (the seed-generation prefix property makes that reproduce
#: the frozen
#: 6-seed record exactly), so a 20-seed roll-up still carries a 6-seed gate.
GATE_SEEDS = 6

#: [PS-5] The H1 criterion, QUOTED. Held as a constant so a report cannot
#: paraphrase it into something easier to satisfy.
G2_CRITERION = (
    "a monotone ordering L2 ≤ L1-oracle ≤ L0 with non-overlapping variability on at "
    "least one of the three topologies"
)

#: [PS-6] The H2 criterion, QUOTED. Same discipline as :data:`G2_CRITERION`.
G3_CRITERION = (
    "the gap grows as knowledge decreases, largest at L0, extending [VonKugelgen2022] "
    "to a continuum; a flat gap disconfirms H2"
)

#: The believed−realized cost gap (PS-6's Gap_c).
GAP_COST = "Gap_cost"


def seed_prefix(frame: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """The first ``n_seeds`` seed indices of a summary frame (the seed-generation prefix property).

    Used to evaluate the PS-1-frozen gate over the seed-0..5 prefix of the N=20 tree
    WITHOUT re-evaluating it on 20 seeds. Raises rather than truncating silently when
    the prefix is short: a gate computed over 4 seeds because 2 were missing would
    still render as a 6-seed record.
    """
    out = frame[frame["seed_idx"] < n_seeds]
    present = sorted(int(v) for v in out["seed_idx"].unique())
    if present != list(range(n_seeds)):
        raise ValueError(
            f"seed prefix 0..{n_seeds - 1} is incomplete — found {present}. The PS-1 "
            "gate is defined on a COMPLETE first-6 look; a short prefix is not a gate."
        )
    return out


def _sd(values: np.ndarray) -> float:
    """Sample SD over seeds; NaN for a single observation (never 0 by accident).

    NaN-aware: a collapse seed with an EMPTY valid subset has no cost at all, and
    that missing value must drop out of the dispersion the same way it drops out
    of the mean — not poison the whole column.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _series(
    frame: pd.DataFrame, cell: Cell, condition: str, group: float, column: str
) -> np.ndarray:
    """One (cell, condition, group) column as a seed-ordered vector."""
    sub = cell_frame(frame, cell)
    sub = sub[(sub["condition"] == condition) & (sub["group"] == group)]
    sub = sub.sort_values("seed_idx")
    return sub[column].to_numpy(dtype=float)


def _cell_series(
    frame: pd.DataFrame, cell: Cell, condition: str, column: str
) -> np.ndarray:
    sub = cell_frame(frame, cell)
    sub = sub[sub["condition"] == condition].sort_values("seed_idx")
    return sub[column].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# H1 — the common-found population no-op disclosure
# --------------------------------------------------------------------------- #


def no_op_equality(by_group: pd.DataFrame) -> pd.DataFrame:
    """Per-row evidence that common-found == all-eligible on THIS grid.

    Disclosed as a DGP finding, not a caveat: if ``N_common_found_threeway ==
    N_eligible`` everywhere, the brute-force procedure always returns a feasible
    action, so recourse failure in this design is always VALIDITY failure and
    never FINDABILITY failure. The common-found population's machinery stays correct and
    binding in principle; it merely resolves to the identity here.
    """
    out = by_group.copy()
    out["n_gap_found_vs_eligible"] = out["N_found"] - out["N_eligible"]
    out["n_gap_common_vs_eligible"] = (
        out["N_common_found_threeway"] - out["N_eligible"]
    )
    out["cost_gap_common_vs_legacy"] = (out[PRIMARY_COST] - out[LEGACY_COST]).abs()
    return out[
        [
            "topology",
            "family",
            "regime",
            "seed_idx",
            "condition",
            "group",
            "N_eligible",
            "N_found",
            "N_common_found_threeway",
            "n_gap_found_vs_eligible",
            "n_gap_common_vs_eligible",
            "cost_gap_common_vs_legacy",
        ]
    ]


def no_op_summary(by_group: pd.DataFrame) -> dict[str, float | bool | int]:
    """Scalar roll-up of :func:`no_op_equality` across every row of the grid."""
    detail = no_op_equality(by_group)
    max_cost_gap = float(detail["cost_gap_common_vs_legacy"].max())
    max_n_gap = int(detail["n_gap_common_vs_eligible"].abs().max())
    max_found_gap = int(detail["n_gap_found_vs_eligible"].abs().max())
    return {
        "n_rows": int(len(detail)),
        "max_abs_cost_diff_common_vs_all_eligible": max_cost_gap,
        "max_abs_N_diff_common_vs_eligible": max_n_gap,
        "max_abs_N_diff_found_vs_eligible": max_found_gap,
        "exact_equality": bool(max_n_gap == 0 and max_cost_gap == 0.0),
    }


# --------------------------------------------------------------------------- #
# H1 — realized cost + signed monotonicity
# --------------------------------------------------------------------------- #


def cost_table(by_group: pd.DataFrame) -> pd.DataFrame:
    """Common-found realized cost, mean +- SD over seeds, per cell/group/rung."""
    rows = []
    for cell in grid_cells():
        for group in GROUPS:
            for condition in CONDITIONS:
                values = _series(by_group, cell, condition, group, PRIMARY_COST)
                legacy = _series(by_group, cell, condition, group, LEGACY_COST)
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "group": group,
                        "condition": condition,
                        "n_seeds": int(values.size),
                        "cost_mean": float(np.mean(values)),
                        "cost_sd": _sd(values),
                        "cost_min": float(np.min(values)),
                        "cost_max": float(np.max(values)),
                        "legacy_all_eligible_mean": float(np.mean(legacy)),
                    }
                )
    return pd.DataFrame(rows)


def _rung_record(
    label: str, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float | bool | str]:
    """One monotonicity rung: is ``lower`` <= ``upper``, and how stable is it?

    ``diff = lower - upper``; the rung HOLDS on the mean when mean(diff) <= 0.
    ``abs_mean_over_sd`` separates a stable ordering from one inside seed noise
    (the H1 monotonicity read asks for exactly this discrimination on the
    L2 -> L1-oracle rung).
    """
    diff = lower - upper
    mean = float(np.mean(diff))
    sd = _sd(diff)
    signs = np.sign(diff)
    return {
        "rung": label,
        "diff_mean": mean,
        "diff_sd": sd,
        "diff_sign": "-" if mean < 0 else ("+" if mean > 0 else "0"),
        "holds_on_mean": bool(mean <= 0.0),
        "sign_consistent_across_seeds": bool(np.all(signs == signs[0])),
        "n_seeds_holding": int(np.sum(diff <= 0.0)),
        "abs_mean_over_sd": float(abs(mean) / sd) if sd > 0 else float("inf"),
        "within_seed_noise": bool(sd > 0 and abs(mean) <= sd),
    }


def monotonicity_by_group(by_group: pd.DataFrame) -> pd.DataFrame:
    """H1 signed monotonicity per cell/group on the common-found cost ladder.

    Expected ordering (signed primary): cost(L2) <= cost(L1-oracle) <=
    cost(L0). The L2 -> L1-oracle rung is the one tracked as REVERSED
    (L1-oracle buys cheaper actions that fail the true classifier more often);
    its sign and seed SD are reported so a reversal inside seed noise is
    never read as a stable one.
    """
    rows = []
    for cell in grid_cells():
        for group in GROUPS:
            costs = {
                c: _series(by_group, cell, c, group, PRIMARY_COST)
                for c in CONDITIONS
            }
            for label, lower, upper in (
                ("L1-oracle<=L0", costs["L1-oracle"], costs["L0"]),
                ("L2<=L1-oracle", costs["L2"], costs["L1-oracle"]),
            ):
                record = _rung_record(label, lower, upper)
                record.update(
                    topology=cell.topology,
                    family=cell.family,
                    regime=cell.regime,
                    group=group,
                )
                rows.append(record)
    frame = pd.DataFrame(rows)
    return frame[
        ["topology", "family", "regime", "group", "rung", *(
            c for c in frame.columns
            if c not in {"topology", "family", "regime", "group", "rung"}
        )]
    ]


def monotonicity_by_cell(by_cell: pd.DataFrame) -> pd.DataFrame:
    """Same ladder on the cell-level gap Delta_cost (common-found population)."""
    column = "Δ_cost_common_found_threeway"
    rows = []
    for cell in grid_cells():
        deltas = {c: _cell_series(by_cell, cell, c, column) for c in CONDITIONS}
        for label, lower, upper in (
            ("L1-oracle<=L0", deltas["L1-oracle"], deltas["L0"]),
            ("L2<=L1-oracle", deltas["L2"], deltas["L1-oracle"]),
        ):
            record = _rung_record(label, lower, upper)
            record.update(
                topology=cell.topology, family=cell.family, regime=cell.regime
            )
            rows.append(record)
    return pd.DataFrame(rows)


def reversal_table(by_cell: pd.DataFrame) -> pd.DataFrame:
    """The L2 -> L1-oracle reversal on Delta_cost, per cell, with its seed SD.

    Positive ``delta_mean`` == Delta_cost(L1-oracle) > Delta_cost(L2) == the
    ladder ordering holds at cell level; negative == the tracked reversal.
    """
    column = "Δ_cost_common_found_threeway"
    rows = []
    for cell in grid_cells():
        l1 = _cell_series(by_cell, cell, "L1-oracle", column)
        l2 = _cell_series(by_cell, cell, "L2", column)
        diff = l1 - l2
        sd = _sd(diff)
        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "per_seed": [round(float(v), 6) for v in diff],
                "delta_mean": float(np.mean(diff)),
                "delta_sd": sd,
                "sign": "+" if np.mean(diff) > 0 else "-",
                "sign_consistent": bool(np.all(np.sign(diff) == np.sign(diff[0]))),
                "abs_mean_over_sd": float(abs(np.mean(diff)) / sd)
                if sd > 0
                else float("inf"),
                "within_seed_noise": bool(sd > 0 and abs(np.mean(diff)) <= sd),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H1 (PS-5) — the seed-dispersion bands and the pinned criterion
# --------------------------------------------------------------------------- #

#: The two rungs of the H1 ladder, lower-condition first (signed primary).
RUNGS = (("L1-oracle<=L0", "L1-oracle", "L0"), ("L2<=L1-oracle", "L2", "L1-oracle"))


def h1_rung_bands(by_group: pd.DataFrame, cells: list[Cell] | None = None) -> pd.DataFrame:
    """Per (cell, group, rung): the ±1 SD seed bands and how far apart they are.

    The H1 monotonicity criterion (PS-5) asks for "non-overlapping variability",
    which admits two readings. Both are
    computed and reported side by side rather than one being chosen silently:

    * ``band_gap`` — the MARGINAL reading. ``(mean_upper − sd_upper) − (mean_lower +
      sd_lower)``, in raw cost units. Positive means the two ±1 SD bands are DISJOINT
      and the value is the clear space between them; negative means they overlap and
      ``|value|`` is the overlap width.
    * ``abs_mean_over_sd`` — the PAIRED reading, from the per-seed difference (carried
      over from :func:`_rung_record`). Seeds are shared across conditions, so the
      paired band is the tighter and more powerful of the two, and it is the reading
      the H1 monotonicity read reports.

    A rung SEPARATES under a reading when the ordering also holds on the mean: a
    disjoint band on the wrong side of the ordering is a violation, not a pass.

    ``cells`` narrows the enumeration — needed when reading the N=6 tree, which
    has 8 cells, not 12 (no chain cells).
    """
    rows = []
    for cell in cells if cells is not None else grid_cells():
        for group in GROUPS:
            costs = {
                c: _series(by_group, cell, c, group, PRIMARY_COST) for c in CONDITIONS
            }
            for label, lower, upper in RUNGS:
                record = _rung_record(label, costs[lower], costs[upper])
                lo_mean, lo_sd = float(np.mean(costs[lower])), _sd(costs[lower])
                up_mean, up_sd = float(np.mean(costs[upper])), _sd(costs[upper])
                band_gap = (up_mean - up_sd) - (lo_mean + lo_sd)
                paired_separates = bool(
                    record["holds_on_mean"] and not record["within_seed_noise"]
                )
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "group": group,
                        "rung": label,
                        "lower_mean": lo_mean,
                        "lower_sd": lo_sd,
                        "upper_mean": up_mean,
                        "upper_sd": up_sd,
                        "diff_mean": record["diff_mean"],
                        "diff_sd": record["diff_sd"],
                        "holds_on_mean": record["holds_on_mean"],
                        "n_seeds_holding": record["n_seeds_holding"],
                        "abs_mean_over_sd": record["abs_mean_over_sd"],
                        "band_gap": float(band_gap),
                        "bands_overlap": bool(band_gap <= 0.0),
                        "band_overlap_width": float(max(0.0, -band_gap)),
                        "paired_separates": paired_separates,
                        "marginal_separates": bool(
                            record["holds_on_mean"] and band_gap > 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def h1_topology_read(bands: pd.DataFrame) -> pd.DataFrame:
    """Roll :func:`h1_rung_bands` up to the topology — the unit the criterion names.

    A (cell, group) SERIES satisfies the full ordering when BOTH rungs hold on the
    mean, and satisfies the criterion's variability clause when both rungs also separate. The
    per-topology counts are reported for both the paired and the marginal reading and
    at both aggregation strengths (ANY series / ALL series), because the criterion's text fixes
    the topology as the unit without fixing how the cells inside one are combined.
    """
    rows = []
    for topology, sub in bands.groupby("topology", sort=False):
        series = sub.groupby(["family", "regime", "group"], sort=False)
        n_series = series.ngroups
        ordered = series["holds_on_mean"].all()
        paired = series["paired_separates"].all()
        marginal = series["marginal_separates"].all()
        # Per series, the WEAKER of its two rungs — the one that decides whether the
        # whole ordering separates. Reported as a max over series: if even the best
        # series has a binding ratio below 1, no series in the topology separates.
        binding = sub.loc[
            sub.groupby(["family", "regime", "group"], sort=False)["abs_mean_over_sd"].idxmin()
        ]
        by_rung = sub.groupby("rung", sort=False)["abs_mean_over_sd"].max()
        rows.append(
            {
                "topology": topology,
                "n_series": int(n_series),
                "n_series_ordered": int(ordered.sum()),
                "n_series_paired_separated": int(paired.sum()),
                "n_series_marginal_separated": int(marginal.sum()),
                "any_series_meets_h1_criterion": bool(paired.any()),
                "all_series_meet_h1_criterion": bool(paired.all()),
                "any_series_meets_h1_criterion_marginal": bool(marginal.any()),
                "max_abs_mean_over_sd_on_binding_rung": float(
                    binding["abs_mean_over_sd"].max()
                ),
                "binding_rung": str(by_rung.idxmin()),
                "min_band_gap": float(sub["band_gap"].min()),
                "max_band_overlap_width": float(sub["band_overlap_width"].max()),
            }
        )
    return pd.DataFrame(rows)


def h1_g2_verdict(topology_read: pd.DataFrame) -> dict[str, object]:
    """Apply :data:`G2_CRITERION` mechanically. No narration, no judgement.

    "on at least one of the three topologies" makes the topology the unit and an
    existential quantifier the combiner ACROSS topologies. WITHIN a topology the
    criterion does not say how the cells combine, so the verdict is computed under
    both the weak (any series) and the strong (all series) reading and reported as
    AMBIGUOUS if the two ever disagree — the report never silently picks the one that
    passes.
    """
    weak = topology_read[topology_read["any_series_meets_h1_criterion"]]["topology"].tolist()
    strong = topology_read[topology_read["all_series_meet_h1_criterion"]]["topology"].tolist()
    if bool(weak) == bool(strong):
        label = "MET" if weak else "NOT MET"
    else:
        label = "AMBIGUOUS — the two within-topology readings disagree"
    return {
        "criterion": G2_CRITERION,
        "label": label,
        "topologies_meeting_weak_reading": weak,
        "topologies_meeting_strong_reading": strong,
        "reading_invariant": bool(bool(weak) == bool(strong)),
    }


# --------------------------------------------------------------------------- #
# H2 — ValidityDisp, Gap_c, valid-subset small-N guard
# --------------------------------------------------------------------------- #


def validity_disp(by_group: pd.DataFrame, condition: str = "L1-oracle") -> pd.DataFrame:
    """ValidityDisp(c) = realized validity(A=-1) − realized validity(A=+1).

    Cross-group, no group subscript; signed, never absolute.

    The per-seed lists are the finding, not decoration: on the collider the A=-1
    series switches near-binary between ~1.0 and ~0.0, so the moments alone
    misdescribe the distribution (PS-9's motivation).
    """
    rows = []
    for cell in grid_cells():
        neg = _series(by_group, cell, condition, -1.0, "realized_validity_rate")
        pos = _series(by_group, cell, condition, 1.0, "realized_validity_rate")
        disp = neg - pos
        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "condition": condition,
                "validity_neg_per_seed": [round(float(v), 6) for v in neg],
                "validity_pos_per_seed": [round(float(v), 6) for v in pos],
                "validity_neg_mean": float(np.mean(neg)),
                "validity_neg_sd": _sd(neg),
                "validity_pos_mean": float(np.mean(pos)),
                "validity_pos_sd": _sd(pos),
                "disp_mean": float(np.mean(disp)),
                "disp_sd": _sd(disp),
                # SD >= |mean| is the seed-fragility signature PS-9 was opened
                # on. It is reported on BOTH series because they do not agree:
                # PS-9's sentence holds on the DISPARITY, not on the raw A=-1
                # validity level. Reporting only one would overstate or understate it.
                "sd_ge_abs_mean_disp": bool(_sd(disp) >= abs(float(np.mean(disp)))),
                "sd_ge_abs_mean_validity_neg": bool(
                    _sd(neg) >= abs(float(np.mean(neg)))
                ),
                "n_high_validity_seeds": int(np.sum(neg > 0.5)),
                "n_collapse_seeds": int(np.sum(neg < 0.5)),
            }
        )
    return pd.DataFrame(rows)


def gap_cost_check(by_cell: pd.DataFrame) -> pd.DataFrame:
    """PS-6: Gap_c == 0 at L1-oracle and L2, nonzero only at L0. One row per cell."""
    rows = []
    for cell in grid_cells():
        record: dict[str, object] = {
            "topology": cell.topology,
            "family": cell.family,
            "regime": cell.regime,
        }
        for condition in CONDITIONS:
            values = np.abs(_cell_series(by_cell, cell, condition, "Gap_cost"))
            record[f"max_abs_Gap_cost_{condition}"] = float(np.max(values))
            if condition == "L0":
                # Reported so a cell whose L0 gap is merely TINY is not confused
                # with one where it is exactly zero on some seed.
                record["min_abs_Gap_cost_L0"] = float(np.min(values))
        record["zero_at_L1_and_L2"] = bool(
            record["max_abs_Gap_cost_L1-oracle"] == 0.0
            and record["max_abs_Gap_cost_L2"] == 0.0
        )
        record["nonzero_at_L0_all_seeds"] = bool(
            np.all(np.abs(_cell_series(by_cell, cell, "L0", "Gap_cost")) > 0.0)
        )
        rows.append(record)
    return pd.DataFrame(rows)


def h2_gap_ladder(by_cell: pd.DataFrame, cells: list[Cell] | None = None) -> pd.DataFrame:
    """Gap_c: the per-condition believed−realized cost gap, per cell.

    The Gap_c criterion names "the per-condition gap |believed − realized|", which is PS-6's
    ``Gap_cost``. One row per (cell, condition) with the seed moments of |Gap_cost|,
    so the ladder L2 → L1-oracle → L0 can be read directly off the table.

    ``cells`` narrows the enumeration, as in :func:`h1_rung_bands`.
    """
    rows = []
    for cell in cells if cells is not None else grid_cells():
        for condition in CONDITIONS:
            values = np.abs(_cell_series(by_cell, cell, condition, GAP_COST))
            rows.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "condition": condition,
                    "n_seeds": int(values.size),
                    "abs_gap_mean": float(np.mean(values)),
                    "abs_gap_sd": _sd(values),
                    "abs_gap_min": float(np.min(values)),
                    "abs_gap_max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def h2_topology_read(ladder: pd.DataFrame) -> pd.DataFrame:
    """Roll the gap ladder up to the topology and flag each clause separately.

    The clauses are kept apart because they can disagree. ``non_decreasing`` and
    ``largest_at_L0`` are what the criterion's "grows as knowledge decreases, largest at L0"
    asserts; ``strictly_increasing`` is the stronger reading in which EVERY rung
    grows; ``flat`` is the disconfirming case. PS-6 makes the gap exactly zero at
    L1-oracle and L2 by construction, so a satisfied ladder here is expected to be
    0 = 0 < positive — non-decreasing and largest at L0, but not strictly increasing.
    That structure is reported, not smoothed into "grows".
    """
    rows = []
    order = list(CONDITIONS[::-1])  # L2 -> L1-oracle -> L0, knowledge decreasing
    for topology, sub in ladder.groupby("topology", sort=False):
        cells = sub.groupby(["family", "regime"], sort=False)
        non_decreasing, largest, strict, flat = [], [], [], []
        for _, cell_rows in cells:
            values = [
                float(cell_rows[cell_rows["condition"] == c]["abs_gap_mean"].iloc[0])
                for c in order
            ]
            rungs = list(zip(values[:-1], values[1:], strict=True))
            non_decreasing.append(all(b >= a for a, b in rungs))
            largest.append(values[-1] > max(values[:-1]))
            strict.append(all(b > a for a, b in rungs))
            flat.append(max(values) == min(values))
        rows.append(
            {
                "topology": topology,
                "n_cells": int(cells.ngroups),
                "n_non_decreasing": int(sum(non_decreasing)),
                "n_largest_at_L0": int(sum(largest)),
                "n_strictly_increasing": int(sum(strict)),
                "n_flat": int(sum(flat)),
                "meets_h2_criterion": bool(all(non_decreasing) and all(largest) and not any(flat)),
            }
        )
    return pd.DataFrame(rows)


def h2_g3_verdict(topology_read: pd.DataFrame) -> dict[str, object]:
    """Apply :data:`G3_CRITERION` mechanically. No narration, no judgement."""
    met = topology_read[topology_read["meets_h2_criterion"]]["topology"].tolist()
    total = int(len(topology_read))
    return {
        "criterion": G3_CRITERION,
        "label": "MET" if len(met) == total else ("PARTIAL" if met else "NOT MET"),
        "topologies_meeting": met,
        "n_topologies": total,
        "n_flat_cells": int(topology_read["n_flat"].sum()),
        "n_strictly_increasing_cells": int(topology_read["n_strictly_increasing"].sum()),
    }


def valid_subset_guard(
    by_group: pd.DataFrame, condition: str = "L1-oracle"
) -> pd.DataFrame:
    """Small-N guard on the common-found population SECONDARY (valid-subset) cost contrast.

    A collapse seed leaves a handful of valid individuals, so its valid-subset
    cost is a near-empty average that can dominate an unweighted cross-seed
    mean. The cost is therefore reported TWICE — over all seeds, and over only
    the seeds clearing ``VALID_SUBSET_MIN_N`` — so small-N contamination is
    separable from genuine cost signal.
    """
    rows = []
    for cell in grid_cells():
        for group in GROUPS:
            n_valid = _series(by_group, cell, condition, group, "N_valid_subset")
            cost = _series(
                by_group, cell, condition, group, "realized_cost_valid_subset"
            )
            validity = _series(
                by_group, cell, condition, group, "realized_validity_rate"
            )
            keep = n_valid >= VALID_SUBSET_MIN_N
            guarded = cost[keep]
            # The guard is an absolute-COUNT threshold, so a collapse seed with a
            # large eligible pool can clear it (e.g. 51 valid of 732 = 7% validity
            # at N=51 >= 30). Those seeds are named so the guarded mean is not
            # mistaken for a collapse-free one.
            leaked = [int(i) for i in np.flatnonzero(keep & (validity < 0.5))]
            rows.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "condition": condition,
                    "group": group,
                    "N_valid_subset_per_seed": [int(v) for v in n_valid],
                    "n_seeds_flagged_small": int(np.sum(~keep)),
                    "flagged_seed_idx": [int(i) for i in np.flatnonzero(~keep)],
                    "collapse_seeds_passing_guard": leaked,
                    "cost_mean_all_seeds": float(np.nanmean(cost)),
                    "cost_sd_all_seeds": _sd(cost),
                    "cost_mean_guarded": float(np.nanmean(guarded))
                    if guarded.size
                    else float("nan"),
                    "cost_sd_guarded": _sd(guarded) if guarded.size > 1 else float("nan"),
                    "n_seeds_guarded": int(guarded.size),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# detectability gate (PS-2) formal record
# --------------------------------------------------------------------------- #


def detectability_gate(
    by_cell: pd.DataFrame,
    s_of_g: pd.DataFrame,
    cells: list[Cell] | None = None,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    """Formal pass/fail record of the PS-2 gate on the N=6 early look.

    Condition (i)  sign(Delta_cost(L2)) consistent across all 6 seeds AND
                   |mean(Delta_cost(L2))| > per-seed SD.
    Condition (ii) sign(Delta S) unambiguous under the PRODUCTION S(g) in the
                   effect-modifying regime. S(g) is fit-independent by
                   construction (PS-3), so it must also be bit-identical
                   across seeds — that invariance is restated and re-checked
                   here rather than assumed from the grid run.

    ``n_seeds`` DEFAULTS TO 6 AND SHOULD STAY THERE. Both the SCM specification's
    amendment and PS-2 define the gate on the FIRST-6-SEED early look, and PS-1
    forecloses re-evaluating it on a growing seed count — re-running this at
    n_seeds=20 would reintroduce exactly the stopping-rule freedom that
    discipline exists to prevent. The parameter exists only so the assertion
    below can name the count it is enforcing; passing 20 is a register decision,
    not a code one.

    ``cells`` narrows the evaluation to one topology's cells, so a gate can be
    discharged on a topology while the rest of the grid is still materializing.

    A failing cell is NOT silently retuned: the PS-2 fallback-lever protocol
    is a register-logged decision, so the caller must stop and escalate.
    """
    rows = []
    for cell in cells if cells is not None else grid_cells():
        delta = _cell_series(by_cell, cell, "L2", "Δ_cost")
        if delta.size != n_seeds:
            raise ValueError(
                f"{cell.label}: expected {n_seeds} seeds for the PS-1 early look, "
                f"got {delta.size} — the gate is defined on the first-6 look."
            )
        mean, sd = float(np.mean(delta)), _sd(delta)
        signs = np.sign(delta)
        sign_ok = bool(np.all(signs == signs[0]) and signs[0] != 0)
        magnitude_ok = bool(abs(mean) > sd)

        s_sub = cell_frame(s_of_g, cell).sort_values("seed_idx")
        ds = s_sub["ΔS"].to_numpy(dtype=float)
        s_identical = bool(np.all(ds == ds[0]))
        # Convention 4: the additive control's Delta S is flat BY CONSTRUCTION,
        # so an unambiguous sign is required only of the effect-modifying regime.
        if cell.regime == "effect_modifying":
            ds_ok = bool(s_identical and ds[0] != 0.0)
            ds_note = "sign unambiguous" if ds_ok else "SIGN AMBIGUOUS"
        else:
            ds_ok = s_identical
            ds_note = "N/A (additive control: Delta S flat by construction)"

        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "delta_cost_L2_per_seed": [round(float(v), 6) for v in delta],
                "mean": mean,
                "sd": sd,
                "abs_mean_over_sd": float(abs(mean) / sd) if sd > 0 else float("inf"),
                "cond_i_sign_consistent": sign_ok,
                "cond_i_magnitude": magnitude_ok,
                "cond_i_pass": bool(sign_ok and magnitude_ok),
                "delta_S": float(ds[0]),
                "delta_S_identical_across_seeds": s_identical,
                "cond_ii_pass": ds_ok,
                "cond_ii_note": ds_note,
                "gate_pass": bool(sign_ok and magnitude_ok and ds_ok),
                # Discharge provenance, carried in the artifact rather than left to
                # the reader: the gate is a FIRST-6-SEED read even when it appears
                # in an N=20 roll-up, so a row that says 6 next to a 20-seed grid is
                # the discipline working, not a stale number (PS-1; the chain SCM specification).
                "n_seeds_evaluated_for_gate": int(n_seeds),
            }
        )
    return pd.DataFrame(rows)


#: [PS-2] Which frozen record each cell's gate was originally discharged
#: in. The provenance artifact names it per row so a reader never has to guess which
#: of the three records a topology's gate lives in.
GATE_SOURCE_RECORDS = {
    "triangle": "results/cross_seed_N6/summary/t4c_detectability_gate.md",
    "collider": "results/cross_seed_N6/summary/t4c_detectability_gate.md",
    "chain": "results/cross_seed_N20/summary/chain_detectability_gate.md",
}


def gate_provenance(
    gate: pd.DataFrame, source_records: dict[str, str] | None = None
) -> pd.DataFrame:
    """The consolidated all-three-topologies gate record (PS-2).

    Machine-readable counterpart to the three markdown records, which between them
    cover the grid but individually cover only part of it: the triangle and collider
    gates live in the N=6 detectability-gate record and the chain gate in
    `chain_detectability_gate.md`.
    This is ONE row per cell, carrying the discharge provenance alongside the numbers
    — including ``n_seeds_evaluated_for_gate``, which PS-2 requires the gate artifact
    to carry and which the markdown records do not carry.

    Carries FROZEN FACTS ONLY. It re-states the PS-1 first-6 gate; it does not
    re-evaluate it, and it establishes no display convention for any figure.
    """
    records = source_records or GATE_SOURCE_RECORDS
    out = gate[
        [
            "topology",
            "family",
            "regime",
            "n_seeds_evaluated_for_gate",
            "mean",
            "sd",
            "abs_mean_over_sd",
            "cond_i_sign_consistent",
            "cond_i_magnitude",
            "cond_i_pass",
            "delta_S",
            "delta_S_identical_across_seeds",
            "cond_ii_pass",
            "cond_ii_note",
            "gate_pass",
        ]
    ].copy()
    missing = sorted(set(out["topology"]) - set(records))
    if missing:
        raise KeyError(
            f"no frozen source record registered for {missing} — the provenance "
            "artifact names the record every row came from and will not emit a blank."
        )
    out["source_record"] = out["topology"].map(records)
    return out.reset_index(drop=True)


#: [PS-1 / PS-2] The gate as it stands in the three frozen records, transcribed
#: VERBATIM as rendered strings — (mean, sd, |mean|/sd, ΔS). Strings, not floats,
#: because the records are markdown: the check can only be as exact as the record is,
#: and a string carries its own precision (the chain record renders the ratio at 2 dp,
#: the N=6 detectability-gate record at 6). Triangle + collider are from
#: ``results/cross_seed_N6/summary/t4c_detectability_gate.md`` (PS-2);
#: chain from ``results/cross_seed_N20/summary/chain_detectability_gate.md``.
#: Embedded here rather than parsed from results/, which is local-only.
FROZEN_GATE: dict[tuple[str, str, str], tuple[str, str, str, str]] = {
    ("triangle", "linear", "additive"): ("3.727707", "0.058849", "63.343897", "0.000000"),
    ("triangle", "linear", "effect_modifying"): (
        "2.957301", "0.097124", "30.448666", "-0.600000",
    ),
    ("triangle", "nlg", "additive"): ("3.340271", "0.056630", "58.984023", "0.001983"),
    ("triangle", "nlg", "effect_modifying"): (
        "2.239280", "0.039787", "56.281365", "-0.185335",
    ),
    ("collider", "linear", "additive"): ("4.788193", "0.191368", "25.020910", "0.000000"),
    ("collider", "linear", "effect_modifying"): (
        "5.272316", "0.308842", "17.071250", "-0.404882",
    ),
    ("collider", "nlg", "additive"): ("1.811379", "0.082451", "21.969051", "0.000497"),
    ("collider", "nlg", "effect_modifying"): (
        "2.000189", "0.114999", "17.393109", "-0.299159",
    ),
    ("chain", "linear", "additive"): ("2.302231", "0.079133", "29.09", "0.000000"),
    ("chain", "linear", "effect_modifying"): ("2.309343", "0.100232", "23.04", "-0.476973"),
    ("chain", "nlg", "additive"): ("1.423975", "0.039646", "35.92", "0.001771"),
    ("chain", "nlg", "effect_modifying"): ("1.449895", "0.041994", "34.53", "-0.342228"),
}

_FROZEN_FIELDS = ("mean", "sd", "abs_mean_over_sd", "delta_S")


def assert_gate_matches_frozen(
    gate: pd.DataFrame,
    frozen: dict[tuple[str, str, str], tuple[str, str, str, str]] | None = None,
) -> pd.DataFrame:
    """Check the first-6 gate against the frozen records at each record's own precision.

    RAISES on any drift. A drifting cell means the seed-0..5 prefix no longer
    reproduces the frozen gate, which is a HALT condition: PS-1 forecloses
    re-evaluating the gate on a changed seed set, so a mismatch must be REPORTED,
    never reconciled by moving the new number onto the old one.
    """
    frozen = FROZEN_GATE if frozen is None else frozen
    rows, drifted = [], []
    for _, row in gate.iterrows():
        key = (row["topology"], row["family"], row["regime"])
        if key not in frozen:
            raise KeyError(f"{key}: no frozen gate value to check against")
        record = dict(zip(["topology", "family", "regime"], key, strict=True))
        matches = []
        for field, want in zip(_FROZEN_FIELDS, frozen[key], strict=True):
            decimals = len(want.partition(".")[2])
            got = f"{float(row[field]):.{decimals}f}"
            # Compared as NUMBERS after rendering, so "-0.000000" and "0.000000" —
            # the same ΔS on an additive control — do not read as a drift.
            matches.append(float(got) == float(want))
            record[f"{field}_frozen"] = want
            record[f"{field}_first6"] = got
            record[f"{field}_matches"] = matches[-1]
        rows.append(record)
        if not all(matches):
            got = tuple(f"{float(row[f]):.6f}" for f in _FROZEN_FIELDS)
            drifted.append((key, frozen[key], got))
    if drifted:
        detail = "\n".join(f"  {k}: frozen {e} vs first-6 {a}" for k, e, a in drifted)
        raise ValueError(
            "the seed-0..5 prefix no longer reproduces the frozen detectability gate — "
            "HALT and report; do NOT reconcile (PS-1 forecloses re-evaluating the gate "
            f"on a changed seed set):\n{detail}"
        )
    return pd.DataFrame(rows)
