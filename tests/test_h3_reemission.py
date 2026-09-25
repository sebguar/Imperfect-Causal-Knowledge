"""The LD-2 re-emission layer: taxonomy, halt arithmetic, truth-side fork, artifact safety.

Every test names the SITUATION the LD-2 taxonomy describes and asserts the label
situation must carry, so a rewording of the implementation cannot quietly change a
verdict. The fork is exercised on all THREE zones with synthetic correlations —
the two branches this grid does not take are covered as thoroughly as the one it
does, which is the point of a pre-committed fork.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from icknowledge.analysis import h3_d2, h3_v2
from icknowledge.analysis.h3_d2 import (
    ALARM_MIN,
    GENUINE_MAX,
    d2_fork,
    fisher_z_rejection_probability,
    triangle_em_builder_coefficients,
    triangle_em_population_correlations,
)
from icknowledge.analysis.h3_v2 import (
    SUPERSEDED,
    classify_violating_seed,
    instance_table_v2,
    reclassify_linear_instances,
    standing_halt_propagation,
)

N_DISC = 4000  # read from the manifests; hard-coded here only as a test fixture


def seed_row(
    *,
    skeleton_shd: int = 0,
    orientation_wrong: int = 0,
    wrong_ci: int = 0,
    wrong_and_tiebreak: int = 0,
    region: int = 3,
    seed_idx: int = 0,
) -> pd.Series:
    return pd.Series(
        {
            "seed_idx": seed_idx,
            "region": region,
            "skeleton_shd": skeleton_shd,
            "orientation_wrong_count": orientation_wrong,
            "wrong_ci_orientation_count": wrong_ci,
            "wrong_and_tiebreak_count": wrong_and_tiebreak,
        }
    )


# --------------------------------------------------------------------------- #
# LD-2 — class (c), the registered form-blindness channel
# --------------------------------------------------------------------------- #


class TestClassC:
    """"A violating seed on an EM cell with skeleton_shd = 0,
    orientation_wrong_count = 0, and region-3 placement" — LD-2."""

    def test_em_perfect_graph_region_three_is_class_c(self):
        assert (
            classify_violating_seed(
                seed_row(), "effect_modifying", instance_systematic=False
            )
            == "c"
        )

    def test_the_same_seed_on_an_ADDITIVE_cell_stays_class_a(self):
        """Limb a3 survives where PS-7 note (c)'s identity licenses it."""
        assert (
            classify_violating_seed(
                seed_row(), "additive", instance_systematic=False
            )
            == "a"
        )

    def test_region_two_on_an_em_cell_is_not_class_c(self):
        """Region-3 placement is part of the class-(c) definition, not a gloss."""
        assert (
            classify_violating_seed(
                seed_row(region=2), "effect_modifying", instance_systematic=False
            )
            == "unclassified"
        )

    def test_a_wrong_ci_orientation_still_outranks_class_c(self):
        """Limb a2 is untouched by LD-2: a wrong CI orientation is class (a)."""
        assert (
            classify_violating_seed(
                seed_row(wrong_ci=1, orientation_wrong=1),
                "effect_modifying",
                instance_systematic=False,
            )
            == "a"
        )


# --------------------------------------------------------------------------- #
# LD-2 — the a1 fracture: (a) vs sporadic vs (d)
# --------------------------------------------------------------------------- #


class TestA1Fracture:
    """skeleton_shd > 0 "splits three ways" — LD-2."""

    def test_systematic_plus_truth_side_genuine_is_class_d(self):
        assert (
            classify_violating_seed(
                seed_row(skeleton_shd=1),
                "effect_modifying",
                instance_systematic=True,
                d2_zone="GENUINE",
            )
            == "d"
        )

    def test_isolated_seeds_are_sporadic_finite_sample_not_class_a(self):
        assert (
            classify_violating_seed(
                seed_row(skeleton_shd=1),
                "effect_modifying",
                instance_systematic=False,
            )
            == "sporadic"
        )

    def test_systematic_without_a_truth_side_check_stays_class_a(self):
        """"systematic UNEXPLAINED adjacency error — HARNESS alarm"."""
        assert (
            classify_violating_seed(
                seed_row(skeleton_shd=1),
                "effect_modifying",
                instance_systematic=True,
                d2_zone=None,
            )
            == "a"
        )

    def test_truth_side_alarm_forces_class_a_even_when_systematic(self):
        assert (
            classify_violating_seed(
                seed_row(skeleton_shd=1),
                "effect_modifying",
                instance_systematic=True,
                d2_zone="ALARM",
            )
            == "a"
        )

    def test_truth_side_indeterminate_does_not_grant_class_d(self):
        assert (
            classify_violating_seed(
                seed_row(skeleton_shd=1),
                "effect_modifying",
                instance_systematic=True,
                d2_zone="INDETERMINATE",
            )
            == "a"
        )


# --------------------------------------------------------------------------- #
# LD-2 — the halt rule over MIXED violating sets
# --------------------------------------------------------------------------- #


def _instances(topology: str, regime: str, region_1_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topology": topology,
                "family": "linear",
                "regime": regime,
                "region_1_count": region_1_count,
                "N_valid": 20,
            }
        ]
    )


def _per_seed(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["family"] = "linear"
    return frame


def _synthetic_instance(classes: list[str], topology="chain", regime="effect_modifying"):
    """Build a per-seed frame whose violating seeds land in the named classes."""
    recipes = {
        # Identical covariates: the ONLY thing separating (a) from (c) is the
        # regime, which is exactly what LD-2's restriction of limb a3 says.
        "a": dict(skeleton_shd=0, orientation_wrong_count=0, region=3),
        "c": dict(skeleton_shd=0, orientation_wrong_count=0, region=3),
        "sporadic": dict(skeleton_shd=1, orientation_wrong_count=0, region=3),
    }
    rows = []
    for idx, code in enumerate(classes):
        rows.append(
            {
                "topology": topology,
                "regime": regime,
                "seed_idx": idx,
                "wrong_ci_orientation_count": 0,
                "wrong_and_tiebreak_count": 0,
                **recipes[code],
            }
        )
    for idx in range(len(classes), 20):
        rows.append(
            {
                "topology": topology,
                "regime": regime,
                "seed_idx": idx,
                "region": 1,
                "skeleton_shd": 0,
                "orientation_wrong_count": 0,
                "wrong_ci_orientation_count": 0,
                "wrong_and_tiebreak_count": 0,
            }
        )
    return _per_seed(rows)


class TestMajorityHaltArithmetic:
    """"HARNESS HALT iff class (a) is a majority of violating seeds; (c), (d) and
    sporadic-finite-sample are not (a)" — LD-2."""

    def test_all_class_c_does_not_halt(self):
        per_seed = _synthetic_instance(["c"] * 12)
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "effect_modifying", 8)
        )[0]
        assert result.seeds_by_class["c"] and not result.halt

    def test_eleven_c_plus_one_sporadic_does_not_halt(self):
        """The chain/linear/EM shape: 11 class (c) + 1 sporadic, zero class (a)."""
        per_seed = _synthetic_instance(["c"] * 11 + ["sporadic"])
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "effect_modifying", 8)
        )[0]
        assert len(result.seeds_by_class["c"]) == 11
        assert len(result.seeds_by_class["sporadic"]) == 1
        assert result.n_class_a == 0
        assert not result.halt

    def test_class_a_majority_halts(self):
        """On an ADDITIVE cell the same perfect-graph seeds are class (a)."""
        per_seed = _synthetic_instance(["a"] * 12, regime="additive")
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "additive", 8)
        )[0]
        assert result.n_class_a == 12
        assert result.halt

    def test_exactly_half_class_a_is_not_a_majority(self):
        """6 (a) of 12 violating is a tie, not a majority — no halt."""
        per_seed = _synthetic_instance(["a"] * 6 + ["sporadic"] * 6, regime="additive")
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "additive", 8)
        )[0]
        assert result.n_class_a == 6
        assert result.n_violating == 12
        assert not result.halt

    def test_one_over_half_class_a_is_a_majority(self):
        per_seed = _synthetic_instance(["a"] * 7 + ["sporadic"] * 6, regime="additive")
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "additive", 7)
        )[0]
        assert result.n_class_a == 7
        assert result.n_violating == 13
        assert result.halt

    def test_untriggered_instance_never_halts(self):
        """Region-1 count >= 16 means the diagnostic never activates."""
        per_seed = _synthetic_instance(["a"] * 2, regime="additive")
        result = reclassify_linear_instances(
            per_seed, _instances("chain", "additive", 18)
        )[0]
        assert not result.triggered and not result.halt
        assert result.n_violating == 0


class TestPropagation:
    def test_a_lifted_halt_invalidates_nothing(self):
        per_seed = _synthetic_instance(["c"] * 12, topology="collider")
        items = reclassify_linear_instances(
            per_seed, _instances("collider", "effect_modifying", 8)
        )
        assert all(not v for v in standing_halt_propagation(items).values())

    def test_a_standing_collider_halt_invalidates_both_orientation_pairs(self):
        per_seed = _synthetic_instance(["a"] * 12, topology="collider", regime="additive")
        items = reclassify_linear_instances(
            per_seed, _instances("collider", "additive", 8)
        )
        propagation = standing_halt_propagation(items)
        assert propagation["orientation_pairs"] == [
            "collider × additive",
            "collider × effect_modifying",
        ]

    def test_a_standing_triangle_halt_never_touches_the_collider_pairs(self):
        """A standing triangle halt propagates to no collider pair:
        orientation_pairs and q2_linear_controls stay empty.
        """
        per_seed = _synthetic_instance(["a"] * 12, topology="triangle", regime="additive")
        items = reclassify_linear_instances(
            per_seed, _instances("triangle", "additive", 8)
        )
        propagation = standing_halt_propagation(items)
        assert propagation["orientation_pairs"] == []
        assert propagation["q2_linear_controls"] == []
        assert propagation["corroborating_pairs"] == ["triangle × additive"]


# --------------------------------------------------------------------------- #
# LD-2 — the fork, all three zones
# --------------------------------------------------------------------------- #


class TestD2ForkZones:
    """The fork is pre-committed on three zones; all three are exercised."""

    def test_low_rho_is_genuine(self):
        r = fisher_z_rejection_probability(0.0, N_DISC, 1)
        assert r <= GENUINE_MAX
        assert d2_fork(r) == "GENUINE"

    def test_high_rho_is_alarm(self):
        r = fisher_z_rejection_probability(0.3, N_DISC, 1)
        assert r >= ALARM_MIN
        assert d2_fork(r) == "ALARM"

    def test_intermediate_rho_is_indeterminate(self):
        """A rho tuned to land strictly between the two zone edges."""
        rho = math.tanh((1.959964 + 0.5) / math.sqrt(N_DISC - 1 - 3))
        r = fisher_z_rejection_probability(rho, N_DISC, 1)
        assert GENUINE_MAX < r < ALARM_MIN
        assert d2_fork(r) == "INDETERMINATE"

    def test_zone_edges_are_inclusive_exactly_as_committed(self):
        """"iff r_op <= 0.5" and "iff r_op >= 0.95" — closed on both sides."""
        assert d2_fork(GENUINE_MAX) == "GENUINE"
        assert d2_fork(ALARM_MIN) == "ALARM"
        assert d2_fork(GENUINE_MAX + 1e-12) == "INDETERMINATE"
        assert d2_fork(ALARM_MIN - 1e-12) == "INDETERMINATE"

    def test_only_genuine_lifts_the_halt(self):
        for zone, lifts in (("GENUINE", True), ("ALARM", False), ("INDETERMINATE", False)):
            result = h3_d2.D2Result(
                coefficients=triangle_em_builder_coefficients(),
                n_disc=N_DISC,
                alpha=0.05,
                subsets=(),
                operative=h3_d2.SubsetCorrelation((), 0.0, N_DISC, 0.05, 0.0),
                zone=zone,
            )
            assert result.halt_lifts is lifts


class TestOperativeSubsetSelection:
    """"the operative subset is the one with the smallest |rho|" — LD-2."""

    def test_operative_subset_is_the_conditioned_one_on_this_builder(self):
        result = h3_d2.run_d2_check(N_DISC)
        assert result.operative.conditioning == ("X1",)
        assert abs(result.operative.rho) < abs(
            next(s for s in result.subsets if s.conditioning == ()).rho
        )

    def test_both_tested_subsets_are_covered(self):
        result = h3_d2.run_d2_check(N_DISC)
        assert {s.conditioning for s in result.subsets} == {(), ("X1",)}

    def test_selection_is_on_absolute_value_not_signed_value(self):
        """A large NEGATIVE rho must not be selected over a small positive one."""
        subsets = (
            h3_d2.SubsetCorrelation((), -0.9, N_DISC, 0.05, 1.0),
            h3_d2.SubsetCorrelation(("X1",), 0.01, N_DISC, 0.05, 0.3),
        )
        assert min(subsets, key=lambda s: abs(s.rho)).conditioning == ("X1",)

    def test_conditioning_set_size_enters_the_degrees_of_freedom(self):
        assert fisher_z_rejection_probability(
            0.05, N_DISC, 0
        ) != fisher_z_rejection_probability(0.05, N_DISC, 1)

    def test_degenerate_sample_size_raises_rather_than_returning_a_number(self):
        with pytest.raises(ValueError, match="must be positive"):
            fisher_z_rejection_probability(0.5, 4, 1)


class TestPopulationCorrelations:
    """The partial vanishes IDENTICALLY, not just at the frozen coefficients."""

    def test_partial_is_exactly_zero_at_the_frozen_coefficients(self):
        rhos = triangle_em_population_correlations()
        assert rhos[("X1",)] == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize(
        "a,g_pos,g_neg,sigma",
        [
            (2.0, 0.9, 0.3, 1.0),
            (0.5, 1.7, -0.4, 0.25),
            (3.3, 0.05, 0.05, 2.0),  # no interaction at all
            (1.0, -1.2, 0.8, 1.5),
        ],
    )
    def test_partial_is_zero_for_every_admissible_parameter_set(
        self, a, g_pos, g_neg, sigma
    ):
        coefficients = h3_d2.TriangleEMCoefficients(a=a, g_pos=g_pos, g_neg=g_neg, sigma=sigma)
        rhos = triangle_em_population_correlations(coefficients)
        assert rhos[("X1",)] == pytest.approx(0.0, abs=1e-12)

    def test_the_marginal_is_substantial_so_the_subsets_genuinely_differ(self):
        rhos = triangle_em_population_correlations()
        assert abs(rhos[()]) > 0.7

    def test_coefficients_are_read_from_the_builder_not_transcribed(self):
        """A retune of scm/triangles.py must flow through automatically."""
        import inspect

        from icknowledge.scm.triangles import make_linear_triangle

        defaults = inspect.signature(make_linear_triangle).parameters
        coefficients = triangle_em_builder_coefficients()
        assert coefficients.a == defaults["a"].default
        assert coefficients.g_pos == defaults["g_pos"].default
        assert coefficients.g_neg == defaults["g_neg"].default

    def test_no_A_main_effect_under_effect_modification(self):
        """The fact the whole derivation turns on (PS-7 / scm-specification)."""
        assert triangle_em_builder_coefficients().beta_a == 0.0


# --------------------------------------------------------------------------- #
# LD-2 — supersession headers and the no-overwrite guarantee
# --------------------------------------------------------------------------- #


def _artifacts() -> tuple[str, pd.DataFrame, list[h3_v2.InstanceClassification]]:
    per_seed = _synthetic_instance(["c"] * 12, topology="collider")
    instances = _instances("collider", "effect_modifying", 8)
    items = reclassify_linear_instances(per_seed, instances)
    return "", instances, items


class TestSupersessionHeader:
    def test_both_superseded_artifacts_are_named_by_path_and_commit(self):
        paths = {path for path, _ in SUPERSEDED}
        assert "results/cross_seed_L1d_N20/summary/h3_verdict.md" in paths
        assert "results/cross_seed_L1d_N20/summary/h3_instance_table.csv" in paths
        assert all(commit == "d8bdd23" for _, commit in SUPERSEDED)

    def test_the_verdict_header_names_every_superseded_artifact(self):
        header = "\n".join(h3_v2._supersession_header("H3 verdict"))
        assert "SUPERSEDES" in header
        for path, commit in SUPERSEDED:
            assert path in header and commit in header

    def test_the_header_states_the_audit_object_guarantee(self):
        header = "\n".join(h3_v2._supersession_header("H3 verdict"))
        assert "audit objects" in header
        assert "NOT overwritten" in header

    def test_the_instance_table_carries_the_header_as_comment_lines(self, tmp_path):
        _, instances, items = _artifacts()
        summary = tmp_path / "summary"
        summary.mkdir()
        table = instance_table_v2(instances, items)
        _, table_path = h3_v2.write_v2_artifacts("# report\n", table, root=tmp_path)
        text = table_path.read_text(encoding="utf-8")
        assert text.startswith("#")
        assert "SUPERSEDES" in text
        assert "d8bdd23" in text


class TestNoOverwrite:
    """The halted artifacts are audit objects: the writer may not touch them."""

    def test_writer_only_ever_creates_the_two_versioned_names(self, tmp_path):
        _, instances, items = _artifacts()
        (tmp_path / "summary").mkdir()
        table = instance_table_v2(instances, items)
        verdict_path, table_path = h3_v2.write_v2_artifacts("# r\n", table, root=tmp_path)
        assert verdict_path.name == "h3_verdict_v2.md"
        assert table_path.name == "h3_instance_table_v2.csv"

    def test_pre_existing_superseded_artifacts_are_left_byte_identical(self, tmp_path):
        summary = tmp_path / "summary"
        summary.mkdir()
        original = {
            "h3_verdict.md": "HALTED ORIGINAL — do not touch\n",
            "h3_instance_table.csv": "topology,family\nchain,linear\n",
        }
        for name, text in original.items():
            (summary / name).write_text(text, encoding="utf-8")

        _, instances, items = _artifacts()
        h3_v2.write_v2_artifacts("# r\n", instance_table_v2(instances, items), root=tmp_path)

        for name, text in original.items():
            assert (summary / name).read_text(encoding="utf-8") == text

    def test_the_v2_names_are_distinct_from_every_superseded_name(self):
        superseded_names = {Path(path).name for path, _ in SUPERSEDED}
        assert "h3_verdict_v2.md" not in superseded_names
        assert "h3_instance_table_v2.csv" not in superseded_names


# --------------------------------------------------------------------------- #
# The re-emitted instance table carries the frozen numbers through unchanged
# --------------------------------------------------------------------------- #


class TestInstanceTableV2:
    def test_every_pre_existing_column_survives_unchanged(self):
        _, instances, items = _artifacts()
        table = instance_table_v2(instances, items)
        for column in instances.columns:
            pd.testing.assert_series_equal(table[column], instances[column])

    def test_classification_columns_are_added(self):
        _, instances, items = _artifacts()
        table = instance_table_v2(instances, items)
        assert table.loc[0, "halt_class_c"] == 12
        assert table.loc[0, "halt_class_a"] == 0
        assert not table.loc[0, "halt"]

    def test_an_em_channel_is_never_reported_as_clean(self):
        _, instances, items = _artifacts()
        channel = instance_table_v2(instances, items).loc[0, "diagnostic_channel"]
        assert "clean" not in channel
        assert "harness bug" not in channel.lower()
        assert "class (c)" in channel


# --------------------------------------------------------------------------- #
# Framing constraints the diagnostic taxonomy imposes on the emitted prose
# --------------------------------------------------------------------------- #


class TestFraming:
    def test_the_d3_region3_clause_matches_the_frozen_expected_text(self):
        assert h3_v2.D3_REGION3_CLAUSE == (
            "interaction reflects form-blindness magnitude in the region-3 (cheaper) "
            "direction, opposite to H3's predicted discovery-penalty direction; cannot "
            "be read as a discovery-penalty differential."
        )

    def test_the_epistemic_block_never_claims_outcome_independence(self):
        """Both terms may appear ONLY inside an explicit denial (LD-2)."""
        block = "\n".join(h3_v2._epistemic_status_block())
        assert "OUTCOME-EXPOSED" in block
        assert "not* outcome-independent" in block
        assert "NOT outcome-blind" in block
        # Neither term may be asserted of this re-emission anywhere else.
        assert block.count("outcome-independent") == 1
        assert block.count("outcome-blind") == 1

    def test_class_definitions_are_quoted_for_every_defined_class(self):
        for code in ("a", "b", "c", "d", "sporadic"):
            assert h3_v2.CLASS_DEFINITIONS[code].count('"') >= 2

    def test_unclassified_is_documented_as_outside_the_registered_taxonomy(self):
        """It is a defensive guard, not a registered class — and says so."""
        text = h3_v2.CLASS_DEFINITIONS["unclassified"]
        assert "no defined class covers this seed" in text
        assert "never folded into class (a)" in text
