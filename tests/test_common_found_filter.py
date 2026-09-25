"""Common-found population filter gates (the common-found population).

Permanent validation tests, parallel in style to the aggregation gates: a
failure here is a HARNESS BUG, not a finding. Covers, per the common-found spec:

  (a) mask cardinality on a synthetic {L0: all found, L1-oracle: 80/100 found,
      L2: all found} cell — three-way = 80, pairwise-vs-L2(L1-oracle) = 80,
      pairwise-vs-L2(L0) = 100;
  (b) realized_cost_common_found_threeway equals a HAND-COMPUTED mean over the
      intersection (not a recompute through the same code path);
  (c) backward compatibility — the pre-existing columns of a shipped aggregate CSV
      regenerate to identical values under the new aggregation;
  (d) the common-found validity-denominator convention — N_valid / N_eligible, with
      not-found individuals counted as recourse failures in the denominator.

Plus the invariants that make the common-found triple safe to read downstream:
  (e) the two comparison sets are genuinely different populations (pairwise ⊇
      three-way), which is why the aggregation emits both;
  (f) the pairwise artifact's ΔB_g is a matched-population subtraction;
  (g) the not-found individual is excluded from every cost mean.

The synthetic cell is built here rather than read from results/ (which is
local-only and gitignored) so (a), (b), (d)-(g) run on a fresh clone; only (c)
needs the local artifacts and skips cleanly without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icknowledge.aggregation import (
    CellKeys,
    aggregate_run,
    build_by_group_frame,
    build_pairwise_frame,
    common_found_populations,
    pairwise_path,
)

_FEATURES = ("X1", "X2")
_CELL = CellKeys(topology="triangle", family="linear", regime="additive", seed=20260710)

#: The spec's synthetic scenario: 100 eligible, L1-oracle fails to find an action
#: for 20 of them, L0 and L2 find one for everybody.
_N = 100
_NOT_FOUND_AT_L1 = tuple(range(80, 100))  # the 20 L1-oracle recourse failures


def _synthetic_table(
    condition: str, not_found: tuple[int, ...] = (), invalid: tuple[int, ...] = ()
) -> pd.DataFrame:
    """One condition's per-individual scoring table for the synthetic cell.

    Group assignment is deliberately INTERLEAVED with the not-found block (even
    index -> A=−1) so the not-found individuals span both groups: a filter bug
    that dropped a whole group would otherwise hide behind a clean split.
    Realized cost is ``1 + index/100``, a distinct value per individual, so any
    mean over any subset identifies exactly which individuals were averaged.
    """
    records = []
    for i in range(_N):
        found = i not in not_found
        cost = 1.0 + i / 100.0 if found else float("nan")
        rec = {
            "condition": condition,
            "index": i,
            "A": -1.0 if i % 2 == 0 else 1.0,
            "acted_set": "X1" if found else "",
            "believed_cost": cost,
            "realized_cost": cost,
            # The common-found population: a not-found individual is invalid on BOTH sides — it
            # is a
            # recourse failure, not a missing observation.
            "believed_validity": int(found),
            "realized_validity": int(found and i not in invalid),
            "found": found,
        }
        for f in _FEATURES:
            rec[f"delta_{f}"] = 0.5 if found else float("nan")
            rec[f"factual_{f}"] = float(i)
            rec[f"realized_cf_{f}"] = float(i) + 0.5 if found else float("nan")
            rec[f"believed_cf_{f}"] = float(i) + 0.5 if found else float("nan")
        records.append(rec)
    return pd.DataFrame.from_records(records)


@pytest.fixture(scope="module")
def synthetic_tables() -> dict[str, pd.DataFrame]:
    return {
        "L0": _synthetic_table("L0"),
        "L1-oracle": _synthetic_table("L1-oracle", not_found=_NOT_FOUND_AT_L1),
        "L2": _synthetic_table("L2"),
    }


def _row(frame: pd.DataFrame, condition: str, group: int) -> pd.Series:
    rows = frame[(frame["condition"] == condition) & (frame["group"] == group)]
    assert len(rows) == 1, f"expected exactly one ({condition}, {group}) row"
    return rows.iloc[0]


# --------------------------------------------------------------------------- #
# (a) mask cardinality — the two the common-found population comparison sets
# --------------------------------------------------------------------------- #


def test_mask_cardinality(synthetic_tables):
    """Three-way = 80; pairwise-vs-L2 is 80 for L1-oracle and 100 for L0."""
    pops = common_found_populations(synthetic_tables)

    assert len(pops.threeway) == 80
    assert pops.threeway == frozenset(range(80))

    # L1-oracle's pairwise set is its own found set (L2 finds everybody).
    assert len(pops.pairwise_vs_L2["L1-oracle"]) == 80
    # L0's pairwise set is the FULL eligible population — L0's 20 extra members
    # are exactly what makes the pairwise and three-way sets different objects.
    assert len(pops.pairwise_vs_L2["L0"]) == 100
    # L2 is the reference side, never a pairing key of itself.
    assert "L2" not in pops.pairwise_vs_L2


# --------------------------------------------------------------------------- #
# (e) the two comparison sets are DIFFERENT populations (why the aggregation emits both)
# --------------------------------------------------------------------------- #


def test_pairwise_is_superset_of_threeway(synthetic_tables):
    """pairwise-vs-L2(c) ⊇ three-way for every c, strictly so for L0 here.

    The three-way set additionally requires the individual to be found in the
    THIRD condition; that is the whole difference between H1's monotonicity
    population and ΔB_g's pairwise population, and reading one for the other
    would silently change which individuals a H1 claim is about.
    """
    pops = common_found_populations(synthetic_tables)
    for condition, population in pops.pairwise_vs_L2.items():
        assert pops.threeway <= population, condition
    assert pops.threeway < pops.pairwise_vs_L2["L0"]  # strict: the 20 L1 failures


# --------------------------------------------------------------------------- #
# (b) hand-computed mean over the intersection
# --------------------------------------------------------------------------- #


def test_common_found_cost_matches_hand_computed_mean(synthetic_tables):
    """realized_cost_common_found_threeway == mean(1 + i/100) over the mask ∩ group.

    Expected values are built from the closed-form cost rule by an INDEPENDENT
    list comprehension — not by re-running the filter — so this gate fails if
    the mask and the mean ever disagree about who was averaged.
    """
    by_group = build_by_group_frame(synthetic_tables, _CELL)
    for group, parity in ((-1, 0), (1, 1)):
        expected_members = [i for i in range(80) if i % 2 == parity]
        expected_mean = float(np.mean([1.0 + i / 100.0 for i in expected_members]))
        for condition in ("L0", "L1-oracle", "L2"):
            row = _row(by_group, condition, group)
            assert row["N_common_found_threeway"] == len(expected_members)
            np.testing.assert_allclose(
                row["realized_cost_common_found_threeway"],
                expected_mean,
                rtol=0,
                atol=1e-12,
            )
    # ... and the all-eligible mean at L0 differs from it, so the two columns are
    # demonstrably measuring different populations on this cell (at L0 all 100
    # are found, but only 80 are in the three-way mask).
    l0 = _row(by_group, "L0", -1)
    assert l0["realized_mean_cost"] != l0["realized_cost_common_found_threeway"]


# --------------------------------------------------------------------------- #
# (g) not-found individuals never enter a cost mean
# --------------------------------------------------------------------------- #


def test_not_found_excluded_from_cost_means(synthetic_tables):
    """At L1-oracle every cost column averages the 80 found only (the common-found population)."""
    by_group = build_by_group_frame(synthetic_tables, _CELL)
    for group, parity in ((-1, 0), (1, 1)):
        row = _row(by_group, "L1-oracle", group)
        found_members = [i for i in range(80) if i % 2 == parity]
        expected = float(np.mean([1.0 + i / 100.0 for i in found_members]))
        assert row["N_found"] == len(found_members)
        # all-eligible cost mean == found-only mean: the 20 failures carry NaN
        # cost (never imputes ∞) and are excluded, not averaged in as zero.
        np.testing.assert_allclose(
            row["realized_mean_cost"], expected, rtol=0, atol=1e-12
        )
        assert np.isfinite(row["realized_mean_cost"])


# --------------------------------------------------------------------------- #
# (d) validity denominator: N_valid / N_eligible, not-found counted as failures
# --------------------------------------------------------------------------- #


def test_validity_denominator_counts_not_found_as_failure(synthetic_tables):
    """N_eligible includes the not-found; the validity rate is N_valid/N_eligible.

    This is the common-found clause that makes a recourse failure visible: with the
    original found-only denominator, L1-oracle here would report validity 1.00 on
    a condition that failed to serve 20% of the population.
    """
    by_group = build_by_group_frame(synthetic_tables, _CELL)
    for group, parity in ((-1, 0), (1, 1)):
        row = _row(by_group, "L1-oracle", group)
        n_eligible = len([i for i in range(_N) if i % 2 == parity])
        n_found = len([i for i in range(80) if i % 2 == parity])
        assert row["N_eligible"] == n_eligible == 50
        assert row["N_found"] == n_found == 40
        assert row["N_valid"] == n_found  # every found action is valid here
        # The denominator is N_eligible: 40/50 = 0.8, NOT the found-only 40/40.
        np.testing.assert_allclose(
            row["realized_validity_rate"], n_found / n_eligible, rtol=0, atol=1e-12
        )
        assert row["realized_validity_rate"] == 0.8
        # believed_validity_rate shares the denominator, so ValidityGap_g
        # stays a like-for-like difference rather than manufacturing
        # a gap out of two different denominators.
        assert row["believed_validity_rate"] == 0.8
    # L0 and L2 find everybody, so their denominators are untouched.
    for condition in ("L0", "L2"):
        for group in (-1, 1):
            assert _row(by_group, condition, group)["realized_validity_rate"] == 1.0


def test_valid_subset_is_outcome_conditioned(synthetic_tables):
    """the common-found population SECONDARY: the valid-subset mean averages only realized-valid "
    "rows."""
    tables = dict(synthetic_tables)
    # Invalidate the four cheapest even-index (A=−1) actions.
    tables["L2"] = _synthetic_table("L2", invalid=(0, 2, 4, 6))
    by_group = build_by_group_frame(tables, _CELL)
    row = _row(by_group, "L2", -1)
    expected_members = [i for i in range(_N) if i % 2 == 0 and i not in (0, 2, 4, 6)]
    assert row["N_valid_subset"] == len(expected_members)
    np.testing.assert_allclose(
        row["realized_cost_valid_subset"],
        float(np.mean([1.0 + i / 100.0 for i in expected_members])),
        rtol=0,
        atol=1e-12,
    )
    # The primary (common-found) mean is NOT outcome-conditioned and still
    # covers all 40 even-index members of the three-way mask — the
    # selection artifact the common-found primary column exists to avoid.
    assert row["N_common_found_threeway"] == 40


# --------------------------------------------------------------------------- #
# (f) the pairwise artifact: matched-population ΔB_g
# --------------------------------------------------------------------------- #


def test_pairwise_frame_matched_population_delta_b(synthetic_tables):
    """Both sides of ΔB_g(c) average the SAME individuals."""
    pairwise = build_pairwise_frame(synthetic_tables, _CELL)
    assert set(pairwise["condition"]) == {"L0", "L1-oracle"}  # never L2 itself
    for _, row in pairwise.iterrows():
        np.testing.assert_allclose(
            row["ΔB_g_vs_L2_common_found"],
            row["realized_cost_c_common_found"] - row["realized_cost_L2_common_found"],
            rtol=0,
            atol=1e-12,
        )
        # Costs are identical across conditions in this synthetic cell, so a
        # matched-population subtraction must be EXACTLY zero. An unmatched one
        # would not be: L0 would average 100 individuals against L2's 80.
        assert row["ΔB_g_vs_L2_common_found"] == 0.0
    # L1-oracle's pairwise population is the 80 found; L0's is all 100.
    assert set(pairwise[pairwise["condition"] == "L1-oracle"][
        "N_common_found_pairwise_vs_L2"
    ]) == {40}
    assert set(pairwise[pairwise["condition"] == "L0"][
        "N_common_found_pairwise_vs_L2"
    ]) == {50}


def test_pairwise_columns_are_nan_on_l2_rows(synthetic_tables):
    """by_group's pairwise columns are NaN at L2 — the value is partner-indexed.

    Not zero, not the L2 found-count: the NaN convention's convention is that a quantity
    undefined by design carries NaN so a downstream reader cannot mistake it for
    a computed observation. The partner-indexed values live in the pairwise
    artifact, which this asserts is non-empty.
    """
    by_group = build_by_group_frame(synthetic_tables, _CELL)
    for group in (-1, 1):
        row = _row(by_group, "L2", group)
        assert np.isnan(row["N_common_found_pairwise_vs_L2"])
        assert np.isnan(row["realized_cost_common_found_pairwise_vs_L2"])
        # ... while every non-L2 row carries a finite pairwise value.
        for condition in ("L0", "L1-oracle"):
            assert np.isfinite(
                _row(by_group, condition, group)["N_common_found_pairwise_vs_L2"]
            )
    assert not build_pairwise_frame(synthetic_tables, _CELL).empty


def test_mismatched_eligible_sets_raise(synthetic_tables):
    """A cell whose conditions score different individuals is a pipeline bug.

    The common-found population is "matched across conditions"; if the eligible sets
    differ, an intersection would silently conflate common-FOUND with
    commonly-ELIGIBLE, so the filter refuses rather than returning a number.
    """
    tables = dict(synthetic_tables)
    tables["L0"] = tables["L0"].iloc[:-1]
    with pytest.raises(ValueError, match="identical eligible set"):
        common_found_populations(tables)


def test_missing_l2_raises(synthetic_tables):
    """No L2 -> no pairwise comparison set (reference)."""
    with pytest.raises(ValueError, match="pairwise comparison set"):
        common_found_populations(
            {k: v for k, v in synthetic_tables.items() if k != "L2"}
        )


def test_threeway_is_over_conditions_present_not_a_hardcoded_triple(synthetic_tables):
    """A two-condition run intersects over the two conditions it actually scored.

    Pins the spec's "all conditions in the run" wording against a future
    L1-discovered rung: the filter must not be silently hardcoded to {L0,
    L1-oracle, L2}.
    """
    pops = common_found_populations(
        {k: synthetic_tables[k] for k in ("L1-oracle", "L2")}
    )
    assert pops.conditions == ("L1-oracle", "L2")
    assert len(pops.threeway) == 80  # L0's 20 extra members are not required here


# --------------------------------------------------------------------------- #
# (c) backward compatibility against a shipped aggregate CSV
# --------------------------------------------------------------------------- #

#: (results dir, topology, family, regime) for the shipped single-seed cells the
#: new aggregation must reproduce. results/ is local-only (gitignored), so this
#: gate skips cleanly on a fresh clone.
_SHIPPED_CELLS = [
    ("results/recourse_triangle", "triangle", "linear", "additive"),
    ("results/recourse_triangle", "triangle", "linear", "effect_modifying"),
    ("results/recourse_triangle_nlg", "triangle", "nlg", "additive"),
    ("results/recourse_triangle_nlg", "triangle", "nlg", "effect_modifying"),
    ("results/recourse_collider", "collider", "linear", "additive"),
    ("results/recourse_collider", "collider", "linear", "effect_modifying"),
    ("results/recourse_collider_nlg", "collider", "nlg", "additive"),
    ("results/recourse_collider_nlg", "collider", "nlg", "effect_modifying"),
]


@pytest.mark.parametrize(("results_dir", "topology", "family", "regime"), _SHIPPED_CELLS)
def test_backward_compat_shipped_aggregates(
    tmp_path, results_dir, topology, family, regime
):
    """Every pre-existing column of a shipped aggregate CSV regenerates IDENTICALLY.

    This is the load-bearing gate of the whole common-found aggregation change: the
    common-found columns are ADDITIVE, so the single-seed reference-run numbers must
    survive the aggregation rewrite bit-for-bit. Compared with ``DataFrame.equals``
    (exact), not ``allclose``: nothing in this change touches an arithmetic path that
    could legitimately move a last-ulp digit, so any difference at all is a regression.
    """
    source = Path(results_dir)
    shipped_dir = source / "aggregates"
    shipped_group = shipped_dir / f"aggregate_by_group_{regime}.csv"
    shipped_cell = shipped_dir / f"aggregate_by_cell_{regime}.csv"
    if not (shipped_group.exists() and shipped_cell.exists()):
        pytest.skip(
            "single-seed reference aggregates not present (results/ is not tracked in "
            f"the repository): {shipped_dir}"
        )

    cell = CellKeys(topology=topology, family=family, regime=regime, seed=20260710)
    new_group_path, new_cell_path = aggregate_run(source, tmp_path, cell)

    for shipped_path, new_path in (
        (shipped_group, new_group_path),
        (shipped_cell, new_cell_path),
    ):
        old = pd.read_csv(shipped_path, float_precision="round_trip")
        new = pd.read_csv(new_path, float_precision="round_trip")
        assert set(old.columns) <= set(new.columns), (
            f"{shipped_path.name}: a pre-existing column disappeared — the common-found change "
            "must be additive."
        )
        overlapping = list(old.columns)
        assert old[overlapping].equals(new[overlapping]), (
            f"{shipped_path.name}: pre-existing values moved under the new aggregation."
        )
    # The new pairwise artifact is written alongside and is well-formed.
    assert not pd.read_csv(
        pairwise_path(tmp_path, regime), float_precision="round_trip"
    ).empty
