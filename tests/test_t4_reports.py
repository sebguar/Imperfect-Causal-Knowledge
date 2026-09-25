"""H1 / H2 / detectability-gate report-layer tests on synthetic frames.

The report functions are pure transforms of the grid run's summary frames, so they are
exercised here against hand-built frames with known answers. No results tree is
touched: the suite must stay green on a clean checkout, where results/ is
gitignored and absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icknowledge.analysis.loading import CONDITIONS, N_SEEDS, grid_cells
from icknowledge.analysis.t4_reports import (
    FROZEN_GATE,
    G2_CRITERION,
    G3_CRITERION,
    GATE_SEEDS,
    assert_gate_matches_frozen,
    detectability_gate,
    gap_cost_check,
    gate_provenance,
    h1_g2_verdict,
    h1_rung_bands,
    h1_topology_read,
    h2_g3_verdict,
    h2_gap_ladder,
    h2_topology_read,
    monotonicity_by_group,
    no_op_summary,
    seed_prefix,
    valid_subset_guard,
    validity_disp,
)
from scripts import run_t4_analysis

GROUPS = (-1.0, 1.0)


def _by_group(
    cost: dict[str, float] | None = None,
    validity: dict[float, list[float]] | None = None,
    n_valid_subset: dict[float, list[int]] | None = None,
    cost_valid_subset: dict[float, list[float]] | None = None,
    n_common_offset: int = 0,
    cost_offset: float = 0.0,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    """A full 12-cell x 3-condition x 2-group x n_seeds by-group frame."""
    cost = cost or {"L0": 3.0, "L1-oracle": 2.0, "L2": 1.0}
    rows = []
    for cell in grid_cells():
        for condition in CONDITIONS:
            for group in GROUPS:
                for seed_idx in range(n_seeds):
                    base = cost[condition] * (1.0 if group < 0 else 0.5)
                    rows.append(
                        {
                            "topology": cell.topology,
                            "family": cell.family,
                            "regime": cell.regime,
                            "seed_idx": seed_idx,
                            "seed": 1000 + seed_idx,
                            "condition": condition,
                            "group": group,
                            "N_eligible": 100,
                            "N_found": 100,
                            "N_common_found_threeway": 100 + n_common_offset,
                            "realized_mean_cost": base,
                            "realized_cost_common_found_threeway": base + cost_offset,
                            "realized_validity_rate": (
                                validity[group][seed_idx] if validity else 1.0
                            ),
                            "N_valid_subset": (
                                n_valid_subset[group][seed_idx]
                                if n_valid_subset
                                else 100
                            ),
                            "realized_cost_valid_subset": (
                                cost_valid_subset[group][seed_idx]
                                if cost_valid_subset
                                else base
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _by_cell(
    delta: dict[str, list[float]] | None = None,
    gap_cost: dict[str, float] | None = None,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    delta = delta or {c: [3.0, 2.0, 1.0][i] * np.ones(n_seeds) for i, c in enumerate(CONDITIONS)}
    gap_cost = gap_cost or {"L0": 0.5, "L1-oracle": 0.0, "L2": 0.0}
    rows = []
    for cell in grid_cells():
        for condition in CONDITIONS:
            for seed_idx in range(n_seeds):
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "seed_idx": seed_idx,
                        "condition": condition,
                        "Δ_cost": float(delta[condition][seed_idx]),
                        "Δ_cost_common_found_threeway": float(delta[condition][seed_idx]),
                        "Gap_cost": gap_cost[condition],
                    }
                )
    return pd.DataFrame(rows)


def _s_of_g(
    delta_s: float = -0.6, vary: bool = False, n_seeds: int = N_SEEDS
) -> pd.DataFrame:
    rows = []
    for cell in grid_cells():
        value = 0.0 if cell.regime == "additive" else delta_s
        for seed_idx in range(n_seeds):
            rows.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "seed_idx": seed_idx,
                    "ΔS": value + (0.01 * seed_idx if vary else 0.0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H1
# --------------------------------------------------------------------------- #


def test_no_op_summary_detects_exact_equality():
    summary = no_op_summary(_by_group())
    assert summary["exact_equality"]
    assert summary["max_abs_N_diff_common_vs_eligible"] == 0
    assert summary["max_abs_cost_diff_common_vs_all_eligible"] == 0.0
    # cells x conditions x groups x seeds, with the cell count DERIVED from the
    # enumeration rather than written as a literal: the chain took it from 8 to 12
    # (the chain SCM specification), and a fourth topology must not break this assertion
    # either.
    assert summary["n_rows"] == len(grid_cells()) * len(CONDITIONS) * 2 * N_SEEDS


def test_no_op_summary_detects_a_real_common_found_gap():
    """If common-found ever dropped an individual, the no-op claim must fail."""
    summary = no_op_summary(_by_group(n_common_offset=-7, cost_offset=0.25))
    assert not summary["exact_equality"]
    assert summary["max_abs_N_diff_common_vs_eligible"] == 7
    assert summary["max_abs_cost_diff_common_vs_all_eligible"] == pytest.approx(0.25)


def test_monotonicity_holds_when_the_ladder_is_ordered():
    frame = monotonicity_by_group(_by_group())
    assert frame["holds_on_mean"].all()
    # Zero-variance ladder: |mean| / SD is infinite, and nothing is "noise".
    assert not frame["within_seed_noise"].any()


def test_monotonicity_flags_a_reversed_rung():
    """L1-oracle CHEAPER than L2 is the tracked reversal and must read as NO."""
    frame = monotonicity_by_group(
        _by_group(cost={"L0": 3.0, "L1-oracle": 1.0, "L2": 2.0})
    )
    reversed_rung = frame[frame["rung"] == "L2<=L1-oracle"]
    assert not reversed_rung["holds_on_mean"].any()
    assert (reversed_rung["diff_sign"] == "+").all()
    assert frame[frame["rung"] == "L1-oracle<=L0"]["holds_on_mean"].all()


def test_within_seed_noise_flag_separates_a_stable_from_a_noisy_rung():
    rng = np.random.default_rng(0)
    noisy = {c: 2.0 + rng.normal(0, 1.0, N_SEEDS) for c in CONDITIONS}
    frame = monotonicity_by_group(
        _by_group().assign(
            realized_cost_common_found_threeway=lambda f: [
                noisy[row.condition][row.seed_idx] for row in f.itertuples()
            ]
        )
    )
    assert frame["within_seed_noise"].all()


# --------------------------------------------------------------------------- #
# H2
# --------------------------------------------------------------------------- #


def test_validity_disp_reports_per_seed_values_and_strata_counts():
    validity = {-1.0: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 1.0: [1.0] * N_SEEDS}
    frame = validity_disp(_by_group(validity=validity))
    row = frame.iloc[0]
    assert row["validity_neg_per_seed"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert row["n_high_validity_seeds"] == 3
    assert row["n_collapse_seeds"] == 3
    assert row["disp_mean"] == pytest.approx(-0.5)
    # A perfectly bimodal series has SD (0.548) >= |mean| (0.5).
    assert row["sd_ge_abs_mean_validity_neg"]


def test_gap_cost_check_confirms_pattern():
    frame = gap_cost_check(_by_cell())
    assert frame["zero_at_L1_and_L2"].all()
    assert frame["nonzero_at_L0_all_seeds"].all()


def test_gap_cost_check_fails_when_a_nonzero_gap_survives_to_l1():
    frame = gap_cost_check(
        _by_cell(gap_cost={"L0": 0.5, "L1-oracle": 0.01, "L2": 0.0})
    )
    assert not frame["zero_at_L1_and_L2"].any()


def test_valid_subset_guard_flags_small_n_and_reports_both_means():
    """The collapse seed's wild cost must be separable from the guarded mean."""
    n_valid = {-1.0: [100, 100, 100, 100, 100, 5], 1.0: [100] * N_SEEDS}
    costs = {-1.0: [2.0, 2.0, 2.0, 2.0, 2.0, 62.0], 1.0: [1.0] * N_SEEDS}
    frame = valid_subset_guard(
        _by_group(n_valid_subset=n_valid, cost_valid_subset=costs)
    )
    neg = frame[frame["group"] == -1.0].iloc[0]
    assert neg["n_seeds_flagged_small"] == 1
    assert neg["flagged_seed_idx"] == [5]
    assert neg["cost_mean_all_seeds"] == pytest.approx(12.0)  # contaminated
    assert neg["cost_mean_guarded"] == pytest.approx(2.0)  # clean
    assert neg["n_seeds_guarded"] == 5
    assert frame[frame["group"] == 1.0].iloc[0]["n_seeds_flagged_small"] == 0


def test_guard_names_collapse_seeds_that_clear_the_count_threshold():
    """A count guard cannot catch a low-validity seed with a large pool."""
    n_valid = {-1.0: [100, 100, 100, 100, 100, 51], 1.0: [100] * N_SEEDS}
    validity = {-1.0: [1.0, 1.0, 1.0, 1.0, 1.0, 0.07], 1.0: [1.0] * N_SEEDS}
    frame = valid_subset_guard(
        _by_group(n_valid_subset=n_valid, validity=validity)
    )
    neg = frame[frame["group"] == -1.0].iloc[0]
    assert neg["n_seeds_flagged_small"] == 0  # 51 >= 30, so the count guard passes
    assert neg["collapse_seeds_passing_guard"] == [5]  # ... but validity is 0.07


def test_valid_subset_guard_survives_an_empty_valid_subset():
    """N_valid_subset == 0 leaves NO cost at all; NaN must not poison the column."""
    n_valid = {-1.0: [100, 100, 100, 100, 100, 0], 1.0: [100] * N_SEEDS}
    costs = {-1.0: [2.0, 2.0, 2.0, 2.0, 2.0, float("nan")], 1.0: [1.0] * N_SEEDS}
    neg = valid_subset_guard(
        _by_group(n_valid_subset=n_valid, cost_valid_subset=costs)
    )
    row = neg[neg["group"] == -1.0].iloc[0]
    assert row["cost_mean_all_seeds"] == pytest.approx(2.0)
    assert not np.isnan(row["cost_sd_all_seeds"])
    assert row["n_seeds_guarded"] == 5


# --------------------------------------------------------------------------- #
# detectability gate
# --------------------------------------------------------------------------- #


def test_detectability_gate_passes_a_sign_stable_high_margin_cell():
    frame = detectability_gate(_by_cell(), _s_of_g())
    assert frame["gate_pass"].all()
    assert frame["cond_i_sign_consistent"].all()
    assert len(frame) == len(grid_cells())
    # The gate is a FIRST-6-SEED read by construction (PS-2), and
    # says so in its own artifact — asserted here so an N=20 roll-up that quietly
    # re-evaluated it would fail rather than read as a discharge.
    assert (frame["n_seeds_evaluated_for_gate"] == N_SEEDS).all()


def test_detectability_gate_fails_on_random_sign_seed_noise():
    """The degenerate case condition (i) exists to catch."""
    flipping = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    delta = {"L0": np.full(N_SEEDS, 3.0), "L1-oracle": np.full(N_SEEDS, 2.0), "L2": flipping}
    frame = detectability_gate(_by_cell(delta=delta), _s_of_g())
    assert not frame["cond_i_sign_consistent"].any()
    assert not frame["gate_pass"].any()


def test_detectability_gate_fails_when_the_magnitude_is_inside_seed_sd():
    delta = {
        "L0": np.full(N_SEEDS, 3.0),
        "L1-oracle": np.full(N_SEEDS, 2.0),
        "L2": np.array([0.01, 0.02, 0.03, 2.0, 2.5, 3.0]),  # same sign, SD > |mean|
    }
    frame = detectability_gate(_by_cell(delta=delta), _s_of_g())
    assert frame["cond_i_sign_consistent"].all()
    assert not frame["cond_i_magnitude"].any()
    assert not frame["gate_pass"].any()


def test_detectability_gate_catches_a_seed_varying_S_of_g():
    """S(g) is fit-independent by construction (PS-3) — drift is a leak."""
    frame = detectability_gate(_by_cell(), _s_of_g(vary=True))
    assert not frame["delta_S_identical_across_seeds"].any()
    assert not frame["cond_ii_pass"].any()


def test_detectability_gate_condition_ii_is_na_in_the_additive_control():
    """Delta S is flat by construction there (PS-3 convention 4), not a failure."""
    frame = detectability_gate(_by_cell(), _s_of_g())
    additive = frame[frame["regime"] == "additive"]
    assert (additive["delta_S"] == 0.0).all()
    assert additive["cond_ii_pass"].all()
    assert additive["cond_ii_note"].str.contains("N/A").all()


def test_detectability_gate_refuses_a_wrong_seed_count():
    truncated = _by_cell()
    truncated = truncated[truncated["seed_idx"] < 4]
    with pytest.raises(ValueError, match="early look"):
        detectability_gate(truncated, _s_of_g())


# ===========================================================================
# The N=20 H1/H2 re-emission: seed threading, the PS-5 and Gap_c reads, the frozen guard
# ===========================================================================


def _write_tree(root, n_seeds: int, **kwargs) -> None:
    """A synthetic grid root the analysis driver can be pointed at, in tmp_path.

    No real results tree is touched — the suite must stay green on a clean checkout
    where results/ is gitignored and absent (the house contract this module and
    `test_figure_scripts` share).
    """
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    _by_group(n_seeds=n_seeds, **kwargs).to_csv(summary / "per_seed_by_group.csv", index=False)
    _by_cell(n_seeds=n_seeds).to_csv(summary / "per_seed_by_cell.csv", index=False)
    _s_of_g(n_seeds=n_seeds).to_csv(summary / "s_of_g_by_seed.csv", index=False)


class TestSeedThreading:
    """`--n-seeds` reaches the H1/H2 reads and can never silently fall back to 6."""

    def test_n_seeds_is_required(self, tmp_path):
        _write_tree(tmp_path, 20)
        with pytest.raises(SystemExit):
            run_t4_analysis.main(["--root", str(tmp_path), "--reports", "t4a"])

    def test_a_twenty_seed_tree_is_read_at_twenty_seeds(self, tmp_path):
        _write_tree(tmp_path, 20)
        assert run_t4_analysis.main(
            ["--root", str(tmp_path), "--n-seeds", "20", "--reports", "t4a", "t4b"]
        ) == 0
        text = (tmp_path / "summary" / "t4a_h1_monotonicity.md").read_text(encoding="utf-8")
        assert "N=20" in text
        # The no-op row count is the seed count times the rest of the grid: 6 would
        # give 432, and only 20 gives 1440. This is the assertion that a H1 path
        # cannot have quietly used the PS-1 early-look default.
        assert f"**{len(grid_cells()) * len(CONDITIONS) * 2 * 20}**" in text
        assert "over the 20 seeds of this grid" in text

    def test_a_seed_count_that_disagrees_with_the_tree_is_refused(self, tmp_path):
        _write_tree(tmp_path, 6)
        with pytest.raises(SystemExit, match="never inferred"):
            run_t4_analysis.main(
                ["--root", str(tmp_path), "--n-seeds", "20", "--reports", "t4a"]
            )

    def test_the_gate_stays_at_six_seeds_on_a_twenty_seed_tree(self, tmp_path):
        """PS-1: the gate is a first-6 read even inside an N=20 roll-up."""
        _write_tree(tmp_path, 20)
        by_cell = pd.read_csv(tmp_path / "summary" / "per_seed_by_cell.csv")
        s_of_g = pd.read_csv(tmp_path / "summary" / "s_of_g_by_seed.csv")
        gate = detectability_gate(
            seed_prefix(by_cell, GATE_SEEDS), seed_prefix(s_of_g, GATE_SEEDS),
            n_seeds=GATE_SEEDS,
        )
        assert (gate["n_seeds_evaluated_for_gate"] == 6).all()
        assert (gate["delta_cost_L2_per_seed"].map(len) == 6).all()

    def test_seed_prefix_refuses_an_incomplete_prefix(self, tmp_path):
        frame = _by_cell(n_seeds=20)
        holed = frame[frame["seed_idx"] != 3]
        with pytest.raises(ValueError, match="incomplete"):
            seed_prefix(holed, GATE_SEEDS)


class TestNoOverwrite:
    """The N=6 analysis artifacts are PS-1 audit objects: the driver may not touch them."""

    def test_a_pre_existing_target_refuses_the_whole_run(self, tmp_path):
        _write_tree(tmp_path, 20)
        frozen = tmp_path / "summary" / "t4a_h1_monotonicity.md"
        frozen.write_text("FROZEN EARLY LOOK — do not touch\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="FROZEN audit object"):
            run_t4_analysis.main(
                ["--root", str(tmp_path), "--n-seeds", "20", "--reports", "t4a", "t4b"]
            )
        assert frozen.read_text(encoding="utf-8") == "FROZEN EARLY LOOK — do not touch\n"
        # Refused before the FIRST write, so t4b never appeared either.
        assert not (tmp_path / "summary" / "t4b_validity_gap.md").exists()

    def test_the_refusal_names_the_path_and_the_way_out(self, tmp_path):
        summary = tmp_path / "summary"
        summary.mkdir()
        (summary / "t4b_validity_gap.md").write_text("frozen\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            run_t4_analysis._refuse_if_frozen(summary, ("t4b_validity_gap.md",))
        message = str(excinfo.value)
        assert "t4b_validity_gap.md" in message
        assert "PS-1" in message
        assert "Nothing was written" in message
        assert "remove or rename" in message

    def test_fresh_generation_succeeds_when_the_targets_are_absent(self, tmp_path):
        _write_tree(tmp_path, 20)
        assert run_t4_analysis.main(
            ["--root", str(tmp_path), "--n-seeds", "20", "--reports", "t4a", "t4b"]
        ) == 0
        assert (tmp_path / "summary" / "t4a_h1_monotonicity.md").exists()
        assert (tmp_path / "summary" / "t4b_validity_gap.md").exists()


class TestG2Read:
    """H1's monotonicity criterion (PS-5), applied mechanically."""

    def test_a_clean_ladder_separates_under_both_readings(self):
        bands = h1_rung_bands(_by_group(n_seeds=20))
        assert bands["holds_on_mean"].all()
        # The synthetic ladder is 3/2/1 with zero seed variance, so both readings
        # separate and the criterion is MET — the passing branch of the rule.
        assert bands["paired_separates"].all()
        assert bands["marginal_separates"].all()
        verdict = h1_g2_verdict(h1_topology_read(bands))
        assert verdict["label"] == "MET"
        assert verdict["reading_invariant"]

    def test_a_ladder_inside_seed_noise_does_not_meet_g2(self):
        """Ordering on the mean is not enough: the criterion also demands separation."""
        rng = np.random.default_rng(11)
        noisy = _by_group(n_seeds=20)
        column = "realized_cost_common_found_threeway"
        noisy[column] = noisy[column] + rng.normal(0, 5.0, len(noisy))
        bands = h1_rung_bands(noisy)
        assert not bands["paired_separates"].all()
        verdict = h1_g2_verdict(h1_topology_read(bands))
        assert verdict["label"] == "NOT MET"

    def test_the_criterion_matches_the_frozen_expected_text(self):
        assert G2_CRITERION == (
            "a monotone ordering L2 ≤ L1-oracle ≤ L0 with non-overlapping variability "
            "on at least one of the three topologies"
        )

    def test_band_gap_is_signed_so_overlap_is_a_magnitude_not_a_flag(self):
        bands = h1_rung_bands(_by_group(n_seeds=20))
        separated = bands[~bands["bands_overlap"]]
        assert (separated["band_gap"] > 0).all()
        assert (separated["band_overlap_width"] == 0.0).all()


class TestG3Read:
    """H2's Gap_c criterion (PS-6), applied mechanically."""

    def test_the_ladder_meets_g3(self):
        """Gap_c is 0 at L1-oracle and L2 and positive at L0 — non-flat, largest at L0."""
        read = h2_topology_read(h2_gap_ladder(_by_cell(n_seeds=20)))
        assert read["meets_h2_criterion"].all()
        # Satisfied WITHOUT being strictly increasing at every rung: the two upper
        # rungs are exactly equal by construction, and the report says so.
        assert (read["n_strictly_increasing"] == 0).all()
        assert (read["n_largest_at_L0"] == read["n_cells"]).all()

    def test_a_flat_gap_disconfirms(self):
        flat = _by_cell(gap_cost={"L0": 0.0, "L1-oracle": 0.0, "L2": 0.0}, n_seeds=20)
        read = h2_topology_read(h2_gap_ladder(flat))
        assert not read["meets_h2_criterion"].any()
        assert (read["n_flat"] == read["n_cells"]).all()
        assert h2_g3_verdict(read)["label"] == "NOT MET"

    def test_a_gap_largest_at_l2_does_not_meet_g3(self):
        """Growth in the wrong direction is not growth as knowledge decreases."""
        inverted = _by_cell(gap_cost={"L0": 0.0, "L1-oracle": 0.2, "L2": 0.5}, n_seeds=20)
        read = h2_topology_read(h2_gap_ladder(inverted))
        assert not read["meets_h2_criterion"].any()
        assert (read["n_largest_at_L0"] == 0).all()

    def test_the_criterion_matches_the_frozen_expected_text(self):
        assert G3_CRITERION == (
            "the gap grows as knowledge decreases, largest at L0, extending "
            "[VonKugelgen2022] to a continuum; a flat gap disconfirms H2"
        )


class TestGateProvenance:
    """PS-2: the consolidated frozen-gate record and its drift check."""

    def test_provenance_carries_a_row_per_cell_with_its_source_record(self):
        gate = detectability_gate(_by_cell(), _s_of_g())
        provenance = gate_provenance(gate)
        assert len(provenance) == len(grid_cells())
        assert (provenance["n_seeds_evaluated_for_gate"] == 6).all()
        assert provenance["source_record"].notna().all()
        chain = provenance[provenance["topology"] == "chain"]["source_record"].unique()
        assert list(chain) == ["results/cross_seed_N20/summary/chain_detectability_gate.md"]

    def test_provenance_refuses_a_topology_with_no_registered_record(self):
        gate = detectability_gate(_by_cell(), _s_of_g())
        with pytest.raises(KeyError, match="no frozen source record"):
            gate_provenance(gate, source_records={"triangle": "somewhere.md"})

    def test_the_drift_check_passes_when_the_prefix_reproduces_the_record(self):
        gate = detectability_gate(_by_cell(), _s_of_g())
        frozen = {
            (r.topology, r.family, r.regime): (
                f"{r.mean:.6f}", f"{r.sd:.6f}", f"{r.abs_mean_over_sd:.6f}",
                f"{r.delta_S:.6f}",
            )
            for r in gate.itertuples()
        }
        checked = assert_gate_matches_frozen(gate, frozen)
        assert len(checked) == len(grid_cells())
        assert checked["mean_matches"].all()

    def test_the_drift_check_halts_rather_than_reconciling(self):
        """A moved gate must be REPORTED — PS-1 forbids re-evaluating it."""
        gate = detectability_gate(_by_cell(), _s_of_g())
        frozen = {
            (r.topology, r.family, r.regime): (
                f"{r.mean + 0.5:.6f}", f"{r.sd:.6f}", f"{r.abs_mean_over_sd:.6f}",
                f"{r.delta_S:.6f}",
            )
            for r in gate.itertuples()
        }
        with pytest.raises(ValueError, match="HALT and report"):
            assert_gate_matches_frozen(gate, frozen)

    def test_the_check_honours_each_records_own_precision(self):
        """The chain record renders |mean|/sd at 2 dp; the N=6 detectability-gate
        record at 6."""
        assert len(FROZEN_GATE[("chain", "linear", "additive")][2].split(".")[1]) == 2
        assert len(FROZEN_GATE[("triangle", "linear", "additive")][2].split(".")[1]) == 6

    def test_the_frozen_gate_table_covers_the_whole_grid(self):
        assert set(FROZEN_GATE) == {(c.topology, c.family, c.regime) for c in grid_cells()}

    def test_the_frozen_minimum_margin_is_the_collider_em_cell(self):
        """Pinned: recomputing the gate over 20 seeds would move the minimum margin
        to ≈9.72 at a different cell."""
        ratios = {k: float(v[2]) for k, v in FROZEN_GATE.items()}
        worst = min(ratios, key=ratios.get)
        assert worst == ("collider", "linear", "effect_modifying")
        assert ratios[worst] == 17.071250
