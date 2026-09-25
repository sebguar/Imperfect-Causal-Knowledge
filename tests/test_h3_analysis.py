"""H3 gates, the corrected region partition, and the halt paths.

The region tests assert the SEMANTIC outcomes stated in the PS-8 region-partition
correction, not the shape of the implementation: each case names the situation
("worse than both baselines", "between the baselines under reversal") and asserts
the region that situation must carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icknowledge.analysis import h3
from icknowledge.analysis.h3 import (
    LINEAR_HOLDS_MIN,
    N_VALID_REQUIRED,
    SIGN_GATE_K,
    band_label,
    classify_region,
    orientation_label,
)

# --------------------------------------------------------------------------- #
# Region partition (PS-8) — mutually exclusive
# --------------------------------------------------------------------------- #


class TestRegionPartitionForwardCase:
    """dcost_l0 > 0: the expected case, where the original glosses were exact."""

    L0 = 1.0

    def test_between_the_baselines_holds(self):
        assert classify_region(0.4, self.L0) == 1

    def test_worse_than_both_baselines_is_the_predicted_violation(self):
        assert classify_region(1.5, self.L0) == 2

    def test_better_than_both_baselines_is_better_than_oracle(self):
        assert classify_region(-0.3, self.L0) == 3

    def test_both_boundaries_are_closed_and_read_as_holds(self):
        assert classify_region(0.0, self.L0) == 1
        assert classify_region(self.L0, self.L0) == 1


class TestRegionPartitionReversalCase:
    """dcost_l0 < 0: the case the correction exists for.

    Under the ORIGINAL glosses a point with dcost_l0 < dcost_ld < 0 satisfied
    "dcost_ld > dcost_l0" (violation), "dcost_ld < 0" (better-than-oracle) and
    the ordered-interval "holds" simultaneously. The corrected partition assigns
    it exactly one region, and semantically it is a PARTIAL degradation relative
    to the reversed L0 baseline — it reads "holds", not worse-than-no-graph.
    """

    L0 = -1.0

    def test_between_the_baselines_holds_not_violation(self):
        assert classify_region(-0.4, self.L0) == 1

    def test_worse_than_both_baselines_is_the_predicted_violation(self):
        # Above zero, hence above BOTH 0 and the negative L0 baseline.
        assert classify_region(0.25, self.L0) == 2

    def test_better_than_both_baselines_is_better_than_oracle(self):
        assert classify_region(-1.5, self.L0) == 3

    def test_both_boundaries_are_closed_and_read_as_holds(self):
        assert classify_region(0.0, self.L0) == 1
        assert classify_region(self.L0, self.L0) == 1


class TestRegionPartitionDegenerateCase:
    """dcost_l0 == 0: the interval collapses to the single point 0."""

    def test_exact_zero_holds(self):
        assert classify_region(0.0, 0.0) == 1

    def test_any_positive_is_the_predicted_violation(self):
        assert classify_region(1e-12, 0.0) == 2

    def test_any_negative_is_better_than_oracle(self):
        assert classify_region(-1e-12, 0.0) == 3


def test_regions_are_mutually_exclusive_and_exhaustive():
    """No point may fall in two regions, and every point falls in one."""
    rng = np.random.default_rng(0)
    for dcost_l0 in (*rng.normal(size=200), 0.0, -1.0, 1.0):
        for dcost_ld in (*rng.normal(size=20), 0.0, dcost_l0):
            assert classify_region(dcost_ld, dcost_l0) in (1, 2, 3)


# --------------------------------------------------------------------------- #
# Gate arithmetic
# --------------------------------------------------------------------------- #


def test_gate_and_band_thresholds_are_the_carried_integers():
    assert (SIGN_GATE_K, N_VALID_REQUIRED, LINEAR_HOLDS_MIN) == (16, 20, 16)


def test_band_label_boundaries():
    assert band_label(6, 6) == "Supported"
    assert band_label(5, 6) == "Supported"
    assert band_label(4, 6) == "Mixed"
    assert band_label(3, 6) == "Null"
    assert band_label(0, 6) == "Null"


def test_orientation_label_is_the_two_pair_band():
    assert orientation_label(2) == "Supported"
    assert orientation_label(1) == "Mixed"
    assert orientation_label(0) == "Null"


# --------------------------------------------------------------------------- #
# Synthetic frames
# --------------------------------------------------------------------------- #


def _costs(rows):
    """rows: (topology, family, regime, seed_idx, {condition: cost, ...}, n)."""
    records = []
    for topology, family, regime, seed_idx, costs, n_common in rows:
        for condition, cost in costs.items():
            records.append(
                {
                    "topology": topology,
                    "family": family,
                    "regime": regime,
                    "seed": 1000 + seed_idx,
                    "seed_idx": seed_idx,
                    "condition": condition,
                    "cost": cost,
                    "n_common_found": n_common,
                }
            )
    return pd.DataFrame(records)


def _four(l0, oracle, discovered):
    return {"L0": l0, "L1-oracle": oracle, "L1-discovered": discovered, "L2": 0.0}


def _covariates(rows, **overrides):
    records = []
    for topology, family, regime, seed_idx in rows:
        record = {
            "topology": topology,
            "family": family,
            "regime": regime,
            "seed_idx": seed_idx,
            "skeleton_shd": 0,
            "orientation_wrong_count": 0,
            "tiebreak_resolved_count": 2,
            "wrong_and_tiebreak_count": 0,
            "wrong_ci_orientation_count": 0,
            "was_bidirected_count": 0,
            "sid": 0,
            "n_sid": 0.0,
            "d_eff": 3,
            "n_xx_true_edges": 2,
            "structural_expectation_applies": False,
            "structural_expectation_violated": False,
        }
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records)


def test_per_seed_quantities_and_strict_interaction_event():
    """A zero cross-family difference is NON-firing (PS-8, strict >)."""
    rows = []
    for seed_idx in range(2):
        # NLG pays more for discovery than linear on seed 0; ties on seed 1.
        rows.append(("collider", "linear", "additive", seed_idx, _four(1.0, 0.0, 0.1), 100))
        nlg_ld = 0.5 if seed_idx == 0 else 0.1
        rows.append(("collider", "nlg", "additive", seed_idx, _four(1.0, 0.0, nlg_ld), 100))
    per_seed = h3.per_seed_quantities(_costs(rows))
    assert set(per_seed["dcost_l0"]) == {1.0}
    events = per_seed[per_seed["family"] == "nlg"].set_index("seed_idx")[
        "interaction_event"
    ]
    assert bool(events.loc[0]) is True
    assert bool(events.loc[1]) is False  # exact tie -> non-firing


def test_t_is_nan_safe_at_a_zero_denominator():
    rows = [("chain", "nlg", "additive", 0, _four(0.0, 0.0, 0.0), 100)]
    per_seed = h3.per_seed_quantities(_costs(rows))
    assert np.isnan(per_seed["t"].iloc[0])
    assert per_seed["region"].iloc[0] == 1


def test_n_valid_halt_path_suppresses_the_gate(tmp_path):
    """[PS-8] An instance short of 20 valid seeds gets no verdict label."""
    rows = []
    for seed_idx in range(N_VALID_REQUIRED):
        # Region 2 on every seed: the gate would pass on the numbers alone.
        n_common = 0 if seed_idx == 0 else 100
        rows.append(
            ("collider", "nlg", "additive", seed_idx, _four(1.0, 0.0, 5.0), n_common)
        )
    costs = _costs(rows)
    n_valid = h3.n_valid_table(costs)
    assert int(n_valid["N_valid"].iloc[0]) == N_VALID_REQUIRED - 1
    assert bool(n_valid["halt_n_valid"].iloc[0]) is True

    per_seed = h3.per_seed_quantities(
        costs,
        _covariates(
            [("collider", "nlg", "additive", i) for i in range(N_VALID_REQUIRED)]
        ),
    )
    instances = h3.instance_table(per_seed, n_valid)
    assert int(instances["region_2_count"].iloc[0]) == N_VALID_REQUIRED
    # 20 region-2 seeds and the gate still does NOT pass: the halt precedes the label.
    assert bool(instances["region_2_gate_pass"].iloc[0]) is False
    del tmp_path


def _collider_linear_class_a_fixture():
    """20 collider-linear seeds violating with a PERFECT graph -> class (a)."""
    rows = []
    for seed_idx in range(N_VALID_REQUIRED):
        rows.append(
            ("collider", "linear", "additive", seed_idx, _four(1.0, 0.0, -0.5), 100)
        )
        rows.append(
            ("collider", "nlg", "additive", seed_idx, _four(1.0, 0.0, 0.2), 100)
        )
        rows.append(
            ("collider", "linear", "effect_modifying", seed_idx, _four(1.0, 0.0, 0.2), 100)
        )
        rows.append(
            ("collider", "nlg", "effect_modifying", seed_idx, _four(1.0, 0.0, 0.4), 100)
        )
    costs = _costs(rows)
    keys = [
        (topology, family, regime, seed_idx)
        for topology in ("collider",)
        for family in ("linear", "nlg")
        for regime in ("additive", "effect_modifying")
        for seed_idx in range(N_VALID_REQUIRED)
    ]
    n_valid = h3.n_valid_table(costs)
    per_seed = h3.per_seed_quantities(costs, _covariates(keys))
    return per_seed, h3.instance_table(per_seed, n_valid)


def test_linear_class_a_majority_halts_and_propagates():
    """[PS-8] Perfect graph + still violating -> class (a) -> HALT."""
    per_seed, instances = _collider_linear_class_a_fixture()
    diagnostics = h3.linear_diagnostics(per_seed, instances)
    additive = next(d for d in diagnostics if d.regime == "additive")
    assert additive.triggered is True
    assert additive.region_1_count == 0
    assert len(additive.class_a_seeds) == N_VALID_REQUIRED
    assert additive.class_b_seeds == []
    assert additive.halt is True

    propagation = h3.halt_propagation(diagnostics)
    # A collider-linear halt invalidates BOTH collider interaction pairs — the
    # confirmatory-primary orientation channel — plus the worse-than-no-graph / fallback controls.
    assert propagation["orientation_pairs"] == [
        "collider × additive",
        "collider × effect_modifying",
    ]
    assert propagation["q2_linear_controls"]
    assert propagation["corroborating_pairs"] == []


def test_clean_linear_control_is_a_positive_reportable_outcome():
    rows = []
    for seed_idx in range(N_VALID_REQUIRED):
        rows.append(
            ("triangle", "linear", "additive", seed_idx, _four(1.0, 0.0, 0.3), 100)
        )
    costs = _costs(rows)
    keys = [("triangle", "linear", "additive", i) for i in range(N_VALID_REQUIRED)]
    per_seed = h3.per_seed_quantities(costs, _covariates(keys))
    instances = h3.instance_table(per_seed, h3.n_valid_table(costs))
    diagnostic = h3.linear_diagnostics(per_seed, instances)[0]
    assert diagnostic.triggered is False
    assert diagnostic.halt is False
    assert "linear control clean" in diagnostic.statement
    assert f"{N_VALID_REQUIRED}/{N_VALID_REQUIRED}" in diagnostic.statement


def test_class_b_call_on_a_linear_cell_is_escalated_not_accepted():
    """[PS-7 note (f)/(g)] A benign class (b) is not expected to occur."""
    rows = [
        ("chain", "linear", "additive", seed_idx, _four(1.0, 0.0, -0.5), 100)
        for seed_idx in range(N_VALID_REQUIRED)
    ]
    costs = _costs(rows)
    keys = [("chain", "linear", "additive", i) for i in range(N_VALID_REQUIRED)]
    covariates = _covariates(
        keys, orientation_wrong_count=1, wrong_and_tiebreak_count=1
    )
    per_seed = h3.per_seed_quantities(costs, covariates)
    instances = h3.instance_table(per_seed, h3.n_valid_table(costs))
    diagnostic = h3.linear_diagnostics(per_seed, instances)[0]
    assert len(diagnostic.class_b_seeds) == N_VALID_REQUIRED
    assert diagnostic.halt is False  # class (b) majority: no harness halt
    assert diagnostic.escalations
    assert "escalated for inspection" in diagnostic.escalations[0]
    # A chain/triangle halt touches only its corroborating pairs.
    assert h3.halt_propagation([diagnostic])["orientation_pairs"] == []


def test_region_three_seeds_neither_fire_nor_shrink_the_denominator():
    """[PS-8] 4 region-2 + 16 region-3 seeds must NOT pass a 16/20 gate."""
    rows = []
    for seed_idx in range(N_VALID_REQUIRED):
        discovered = 5.0 if seed_idx < 4 else -5.0
        rows.append(
            ("chain", "nlg", "additive", seed_idx, _four(1.0, 0.0, discovered), 100)
        )
    costs = _costs(rows)
    keys = [("chain", "nlg", "additive", i) for i in range(N_VALID_REQUIRED)]
    per_seed = h3.per_seed_quantities(costs, _covariates(keys))
    instances = h3.instance_table(per_seed, h3.n_valid_table(costs))
    row = instances.iloc[0]
    assert (row["region_2_count"], row["region_3_count"]) == (4, 16)
    assert int(row["n_seeds"]) == N_VALID_REQUIRED
    assert bool(row["region_2_gate_pass"]) is False


def test_load_realized_costs_uses_the_group_level_column_not_the_gap():
    """Guards the Step-0c choice: the cell-level column is a BETWEEN-GROUP gap."""
    assert h3.COST_COLUMN == "realized_cost_common_found_fourway"
    assert h3.COUNT_COLUMN == "N_common_found_fourway"


def test_load_realized_costs_reports_a_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="per_seed_by_group.csv"):
        h3.load_realized_costs(root=tmp_path)
