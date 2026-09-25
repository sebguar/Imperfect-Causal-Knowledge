"""H4 analysis-layer gates.

The verdict function is PURE and is tested at every threshold boundary PS-9
defines, on synthetic frames with hand-built ΔΔB values. That is the point of
keeping it pure: the label the thesis reports is produced by a function whose
behaviour at each boundary is pinned by a test, not by a run.

Covers:
  1. Verdict boundaries — 6/6 Supported; 5/6 with a LOW-|ΔS| outlier -> Supported;
     5/6 with a HIGH-|ΔS| outlier -> Mixed (the outlier-discrimination clause);
     4/6 -> Mixed; 3/6 -> Null; contrast failure on 2 topologies -> Null.
  2. Sign-gate arithmetic at k = 16/20, both sides of the boundary.
  3. The NaN convention guard — a non-finite ΔB reaching the difference raises, never nanmeans.
  4. Common-found sourcing — the module reads ΔB_g_vs_L2_common_found, and a
     fixture whose all-eligible column disagrees proves which one was used.
  5. ΔS pairing assertions trip on a sign-flipped EM ΔS and on a non-flat control.
  6. The frozen-artifact guard — the driver refuses to overwrite h4_verdict.md.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from icknowledge.analysis import h4
from icknowledge.analysis.loading import grid_cells
from scripts import run_h4_analysis

_EM_TOPOLOGIES = ("triangle", "collider", "chain")
_FAMILIES = ("linear", "nlg")

#: The production ΔS vector, so the synthetic fixtures exercise the real |ΔS|
#: classes the outlier clause discriminates on (low: nlg triangle 0.185, nlg
#: collider 0.299; high: the other four).
_DELTA_S = {
    ("triangle", "linear"): -0.600000,
    ("triangle", "nlg"): -0.185335,
    ("collider", "linear"): -0.404882,
    ("collider", "nlg"): -0.299159,
    ("chain", "linear"): -0.476973,
    ("chain", "nlg"): -0.342228,
}


def _cells_frame(
    agree: dict[tuple[str, str], int],
    control_abs: float = 0.001,
    treatment_abs: float = 0.05,
    condition: str = "L0",
) -> pd.DataFrame:
    """Synthetic by-cell frame: `agree` gives each EM instance's seed-agree count.

    Treatment |mean ΔΔB| is held above control |mean ΔΔB| by default so the
    contrast passes and the tally is what the verdict turns on; individual tests
    override the magnitudes to exercise the contrast branch.
    """
    rows = []
    for topology in _EM_TOPOLOGIES:
        for family in _FAMILIES:
            ds = _DELTA_S[(topology, family)]
            n = agree[(topology, family)]
            rows.append(
                {
                    "topology": topology, "family": family,
                    "regime": "effect_modifying", "condition": condition,
                    "delta_S": ds, "n_seeds": 20,
                    "mean_ddB": -treatment_abs, "sd_ddB": 0.01,
                    "min_ddB": -treatment_abs, "max_ddB": -treatment_abs,
                    "n_seeds_neg": n, "n_seeds_pos": 20 - n, "n_seeds_zero": 0,
                    "n_seeds_sign_agree": n,
                    "sign_gate_pass": bool(n >= h4.SIGN_GATE_K),
                    "gate_detail": f"{n}/20 (need ≥{h4.SIGN_GATE_K})",
                    "control_read": "",
                }
            )
            rows.append(
                {
                    "topology": topology, "family": family,
                    "regime": "additive", "condition": condition,
                    "delta_S": 0.0, "n_seeds": 20,
                    "mean_ddB": control_abs, "sd_ddB": 0.01,
                    "min_ddB": control_abs, "max_ddB": control_abs,
                    "n_seeds_neg": 10, "n_seeds_pos": 10, "n_seeds_zero": 0,
                    "n_seeds_sign_agree": 10, "sign_gate_pass": None,
                    "gate_detail": "",
                    "control_read": "ABSENCE (flat within seed noise): 10 / 10",
                }
            )
    return pd.DataFrame(rows)


def _verdict(agree: dict[tuple[str, str], int], **kwargs) -> h4.Verdict:
    cells = _cells_frame(agree, **kwargs)
    contrast = h4.regime_contrast(cells)
    return h4.verdict_for_rung(cells, contrast, kwargs.get("condition", "L0"))


def _all(n: int) -> dict[tuple[str, str], int]:
    return {(t, f): n for t in _EM_TOPOLOGIES for f in _FAMILIES}


# ===========================================================================
# 1 — verdict boundaries
# ===========================================================================


def test_six_of_six_is_supported():
    v = _verdict(_all(20))
    assert v.label == "Supported"
    assert (v.n_instances_passing, v.n_instances) == (6, 6)
    assert v.failing_instances == ()


def test_five_of_six_with_a_low_delta_s_outlier_stays_supported():
    """The tolerated failure: PS-9 expects the odd one out to be low-propagation."""
    agree = _all(20)
    agree[("triangle", "nlg")] = 5  # |ΔS| = 0.185, the lowest of the six
    v = _verdict(agree)
    assert v.label == "Supported"
    assert v.n_instances_passing == 5
    assert v.failing_instances == ("triangle/nlg",)
    assert "LOW-propagation" in v.outlier_clause
    assert "tolerated" in v.outlier_clause


def test_five_of_six_with_a_high_delta_s_outlier_is_forced_to_mixed():
    """The anti-tautology teeth: a high-|ΔS| non-conformer costs the verdict.

    Without this branch a 5/6 tally would read Supported regardless of WHICH
    instance failed, which is exactly the reading PS-9's outlier-discrimination
    clause exists to prevent.
    """
    agree = _all(20)
    agree[("triangle", "linear")] = 5  # |ΔS| = 0.600, the highest of the six
    v = _verdict(agree)
    assert v.label == "Mixed"
    assert v.n_instances_passing == 5
    assert "HIGH-propagation" in v.outlier_clause
    assert "counts AGAINST the theory" in v.outlier_clause


def test_four_of_six_is_mixed():
    agree = _all(20)
    agree[("triangle", "nlg")] = 5
    agree[("collider", "nlg")] = 5
    v = _verdict(agree)
    assert v.label == "Mixed"
    assert v.n_instances_passing == 4


def test_three_of_six_is_null():
    agree = _all(20)
    for key in (("triangle", "nlg"), ("collider", "nlg"), ("chain", "nlg")):
        agree[key] = 5
    v = _verdict(agree)
    assert v.label == "Null"
    assert v.n_instances_passing == 3


def test_contrast_failure_on_two_topologies_forces_null_even_at_six_of_six():
    """The contrast is H4's primary evidence — 6/6 signs cannot rescue its absence."""
    v = _verdict(_all(20), treatment_abs=0.05, control_abs=0.9)
    assert v.label == "Null"
    assert v.n_instances_passing == 6
    assert len(v.contrast_failures) == 3
    assert any("forced Null" in r for r in v.reasons)


def test_contrast_failure_on_one_topology_downgrades_supported_to_mixed():
    """PS-9's Mixed clause: 'the contrast holds on some topologies but not others'."""
    cells = _cells_frame(_all(20))
    mask = (cells["topology"] == "collider") & (cells["regime"] == "additive")
    cells.loc[mask, "mean_ddB"] = 0.9  # control now dwarfs treatment on collider only
    contrast = h4.regime_contrast(cells)
    v = h4.verdict_for_rung(cells, contrast, "L0")
    assert v.label == "Mixed"
    assert v.contrast_failures == ("collider",)


def test_saturated_topology_is_exempted_from_the_strict_inequality():
    """PS-9's contrast-on-saturated-topology note: tiny in BOTH arms is not a failure."""
    cells = _cells_frame(_all(20), treatment_abs=0.001, control_abs=0.002)
    contrast = h4.regime_contrast(cells)
    assert contrast[contrast["family"] == "ALL"]["saturated_both_arms"].all()
    v = h4.verdict_for_rung(cells, contrast, "L0")
    assert v.contrast_failures == ()
    assert v.label == "Supported"


# ===========================================================================
# 2 — sign-gate arithmetic
# ===========================================================================


@pytest.mark.parametrize(
    "n_agree,expected", [(20, True), (17, True), (16, True), (15, False), (0, False)]
)
def test_sign_gate_boundary_is_at_sixteen_of_twenty(n_agree, expected):
    agree = _all(20)
    agree[("chain", "linear")] = n_agree
    cells = _cells_frame(agree)
    row = cells[
        (cells["topology"] == "chain")
        & (cells["family"] == "linear")
        & (cells["regime"] == "effect_modifying")
    ].iloc[0]
    assert bool(row["sign_gate_pass"]) is expected


def test_sign_gate_counts_the_product_with_delta_s_not_the_raw_sign():
    """sign(ΔS·ΔΔB) > 0 — with ΔS < 0 that means ΔΔB < 0, and the code must agree.

    Built from the real aggregation path rather than a hand-set column, so a
    sign-convention slip in `by_cell` is caught rather than assumed away.
    """
    per_seed = pd.DataFrame(
        {
            "topology": ["chain"] * 4, "family": ["linear"] * 4,
            "regime": ["effect_modifying"] * 4, "seed_idx": [0, 1, 2, 3],
            "seed": [1, 2, 3, 4], "condition": ["L0"] * 4,
            "ddB": [-1.0, -1.0, -1.0, +1.0],
        }
    )
    delta_s = pd.DataFrame(
        [{"topology": "chain", "family": "linear",
          "regime": "effect_modifying", "delta_S": -0.476973}]
    )
    row = h4.by_cell(per_seed, delta_s).iloc[0]
    assert row["n_seeds_sign_agree"] == 3  # the three ΔΔB < 0 seeds
    assert row["n_seeds_neg"] == 3 and row["n_seeds_pos"] == 1


# ===========================================================================
# 3 — the NaN convention guard
# ===========================================================================


def _write_pairwise(cell, tmp_path, values, extra=None):
    """Write a hand-built pairwise artifact carrying BOTH rungs for one cell.

    Both rungs always, because `load_per_seed_ddb` requires exactly two group rows
    per rung — a fixture with only L0 would fail on a shape check rather than on
    the behaviour under test.
    """
    seed_dir = cell.seed_dir(0, tmp_path)
    seed_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "topology": [cell.topology] * 4,
            "family": [cell.family] * 4,
            "regime": [cell.regime] * 4,
            "seed": [1] * 4,
            "condition": ["L0", "L0", "L1-oracle", "L1-oracle"],
            "group": [-1, 1, -1, 1],
            "N_common_found_pairwise_vs_L2": [10] * 4,
            "realized_cost_c_common_found": [1.0] * 4,
            "realized_cost_L2_common_found": [1.0] * 4,
        }
    )
    if values is not None:
        frame[h4.SOURCE_COLUMN] = values
    for name, column in (extra or {}).items():
        frame[name] = column
    frame.to_csv(seed_dir / f"aggregate_pairwise_{cell.regime}.csv", index=False)
    return frame


def test_non_finite_delta_b_raises_instead_of_being_averaged_over(tmp_path):
    """A NaN reaching the ΔΔB difference is a hard error, never a silent nanmean.

    Under the NaN convention, NaN is a reference-row MARKER (the L2 self-reference is not a
    comparison). Averaging over it would let a downstream mean read the reference
    cell as a genuine zero-effect observation — precisely what the NaN convention forbids.
    """
    cell = grid_cells()[0]
    _write_pairwise(cell, tmp_path, [np.nan, 0.5, 0.5, 0.1])
    with pytest.raises(ValueError, match="non-finite"):
        h4.load_per_seed_ddb(tmp_path, n_seeds=1, cells=[cell])


def test_l2_rows_never_reach_the_difference(tmp_path):
    """An L2 row present in the artifact is dropped, not differenced."""
    cell = grid_cells()[0]
    seed_dir = cell.seed_dir(0, tmp_path)
    seed_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "topology": [cell.topology] * 6,
            "family": [cell.family] * 6,
            "regime": [cell.regime] * 6,
            "seed": [1] * 6,
            "condition": ["L0", "L0", "L1-oracle", "L1-oracle", "L2", "L2"],
            "group": [-1, 1, -1, 1, -1, 1],
            "N_common_found_pairwise_vs_L2": [10] * 6,
            "realized_cost_c_common_found": [1.0] * 6,
            "realized_cost_L2_common_found": [1.0] * 6,
            h4.SOURCE_COLUMN: [0.9, 0.2, 0.5, 0.1, np.nan, np.nan],
        }
    ).to_csv(seed_dir / f"aggregate_pairwise_{cell.regime}.csv", index=False)
    out = h4.load_per_seed_ddb(tmp_path, n_seeds=1, cells=[cell])
    assert set(out["condition"]) == set(h4.RUNGS)
    assert out[out["condition"] == "L0"]["ddB"].iloc[0] == pytest.approx(0.7)


# ===========================================================================
# 4 — common-found sourcing
# ===========================================================================


def test_reads_common_found_not_the_all_eligible_column(tmp_path):
    """Proven by divergence: the two columns disagree and only one answer matches."""
    cell = grid_cells()[0]
    _write_pairwise(
        cell,
        tmp_path,
        [0.9, 0.2, 0.5, 0.1],                      # common-found -> ΔΔB(L0) = 0.7
        extra={"ΔB_g_vs_L2": [99.0, 1.0, 9.0, 1.0]},  # all-eligible -> 98.0
    )
    out = h4.load_per_seed_ddb(tmp_path, n_seeds=1, cells=[cell])
    assert out[out["condition"] == "L0"]["ddB"].iloc[0] == pytest.approx(0.7)


def test_missing_common_found_column_raises_rather_than_falling_back(tmp_path):
    cell = grid_cells()[0]
    _write_pairwise(
        cell, tmp_path, None, extra={"ΔB_g_vs_L2": [0.9, 0.2, 0.5, 0.1]}
    )
    with pytest.raises(KeyError, match="COMMON-FOUND"):
        h4.load_per_seed_ddb(tmp_path, n_seeds=1, cells=[cell])


# ===========================================================================
# 5 — ΔS pairing assertions
# ===========================================================================


def test_delta_s_precondition_trips_on_a_sign_flipped_effect_modifying_instance():
    """Every sign gate multiplies by sign(ΔS) — a flipped ΔS would invert a verdict."""
    frame = pd.DataFrame(
        [
            {"topology": "chain", "family": "linear",
             "regime": "effect_modifying", "delta_S": +0.476973},
        ]
    )
    with pytest.raises(ValueError, match="must be negative"):
        h4.assert_delta_s_preconditions(frame)


def test_delta_s_precondition_trips_on_a_non_flat_control():
    frame = pd.DataFrame(
        [{"topology": "chain", "family": "linear", "regime": "additive", "delta_S": 0.05}]
    )
    with pytest.raises(ValueError, match="flat within"):
        h4.assert_delta_s_preconditions(frame)


def test_delta_s_preconditions_pass_on_the_production_vector():
    rows = [
        {"topology": t, "family": f, "regime": "effect_modifying", "delta_S": v}
        for (t, f), v in _DELTA_S.items()
    ] + [
        {"topology": t, "family": f, "regime": "additive", "delta_S": v}
        for (t, f), v in {
            ("triangle", "linear"): 0.0, ("chain", "linear"): 0.0,
            ("collider", "linear"): 0.0, ("triangle", "nlg"): 0.001983,
            ("chain", "nlg"): 0.001771, ("collider", "nlg"): 0.000497,
        }.items()
    ]
    h4.assert_delta_s_preconditions(pd.DataFrame(rows))


# ===========================================================================
# 6 — the frozen-artifact guard on h4_verdict.md
# ===========================================================================


def _delta_s_for(cell) -> float:
    """EM cells carry the production ΔS; additive controls are flat by construction."""
    if cell.regime == "effect_modifying":
        return _DELTA_S[(cell.topology, cell.family)]
    return 0.0


def _synthetic_grid(root, n_seeds: int = 2) -> None:
    """A hand-built N20-shaped tree: the minimum `run_h4_analysis.main` consumes.

    No real results tree is touched — the suite must stay green on a clean
    checkout where results/ is gitignored and absent (same contract as
    `test_figure_scripts` and `test_t4_reports`). The numbers are arbitrary but
    structurally valid: EM ΔΔB tracks sign(ΔS) and dominates the control arm, so
    the report builder's contrast and correlation branches both have something
    finite to read. The VERDICT these numbers produce is irrelevant here; what is
    under test is whether the driver writes at all.
    """
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    s_rows = []
    for cell in grid_cells():
        ds = _delta_s_for(cell)
        for seed_idx in range(n_seeds):
            seed_dir = cell.seed_dir(seed_idx, root)
            seed_dir.mkdir(parents=True, exist_ok=True)
            # ΔΔB = (group −1) − (group +1), so the +1 rows are held at zero and the
            # −1 rows carry the whole difference. Jittered by seed so SD is finite.
            ddb = (2.0 * ds if ds else 0.001) * (1.0 + 0.01 * seed_idx)
            keys = {
                "topology": [cell.topology] * 4,
                "family": [cell.family] * 4,
                "regime": [cell.regime] * 4,
                "seed": [seed_idx + 1] * 4,
                "condition": ["L0", "L0", "L1-oracle", "L1-oracle"],
                "group": [-1, 1, -1, 1],
            }
            pd.DataFrame(
                {
                    **keys,
                    "N_common_found_pairwise_vs_L2": [10] * 4,
                    h4.SOURCE_COLUMN: [ddb, 0.0, ddb, 0.0],
                }
            ).to_csv(seed_dir / f"aggregate_pairwise_{cell.regime}.csv", index=False)
            pd.DataFrame({**keys, "N_eligible": [10] * 4}).to_csv(
                seed_dir / f"aggregate_by_group_{cell.regime}.csv", index=False
            )
            s_rows.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "seed_idx": seed_idx,
                    "ΔS": ds,
                }
            )
    pd.DataFrame(s_rows).to_csv(summary / "s_of_g_by_seed.csv", index=False)


class TestNoOverwrite:
    """`h4_verdict.md` is a frozen audit object: the driver may not touch it.

    PS-7/PS-4 isolate the H4 verdict from the four-rung tree;
    these tests make that isolation a property of the code.
    """

    def test_fresh_generation_succeeds_when_the_verdict_is_absent(self, tmp_path):
        """A clean-checkout re-run must still work: absent means writable."""
        _synthetic_grid(tmp_path)
        assert run_h4_analysis.main(["--root", str(tmp_path), "--n-seeds", "2"]) == 0
        summary = tmp_path / "summary"
        assert (summary / "h4_verdict.md").read_text(encoding="utf-8").startswith("#")
        for name in ("h4_per_seed_ddb.csv", "h4_ddb_by_cell.csv", "h4_regime_contrast.csv"):
            assert (summary / name).exists()

    def test_a_second_run_refuses_and_leaves_the_verdict_byte_identical(self, tmp_path):
        _synthetic_grid(tmp_path)
        run_h4_analysis.main(["--root", str(tmp_path), "--n-seeds", "2"])
        verdict = tmp_path / "summary" / "h4_verdict.md"
        before = hashlib.sha256(verdict.read_bytes()).hexdigest()

        with pytest.raises(SystemExit, match="FROZEN audit object"):
            run_h4_analysis.main(["--root", str(tmp_path), "--n-seeds", "2"])

        assert hashlib.sha256(verdict.read_bytes()).hexdigest() == before

    def test_the_refusal_precedes_every_write_not_just_the_verdict_write(self, tmp_path):
        """A partial re-run replacing the CSVs but sparing the verdict is worse than none.

        The three CSVs and the verdict are one run's output, so the guard fires
        before the first write: the sibling artifacts must be untouched too.
        """
        _synthetic_grid(tmp_path)
        run_h4_analysis.main(["--root", str(tmp_path), "--n-seeds", "2"])
        summary = tmp_path / "summary"
        siblings = ("h4_per_seed_ddb.csv", "h4_ddb_by_cell.csv", "h4_regime_contrast.csv")
        before = {
            name: hashlib.sha256((summary / name).read_bytes()).hexdigest()
            for name in siblings
        }
        # A marker no re-run could reproduce: if any write happened, it is gone.
        (summary / "h4_per_seed_ddb.csv").write_text("SENTINEL\n", encoding="utf-8")

        with pytest.raises(SystemExit):
            run_h4_analysis.main(["--root", str(tmp_path), "--n-seeds", "2"])

        assert (summary / "h4_per_seed_ddb.csv").read_text(encoding="utf-8") == "SENTINEL\n"
        for name in siblings[1:]:
            assert hashlib.sha256((summary / name).read_bytes()).hexdigest() == before[name]

    def test_the_refusal_names_the_path_and_the_deliberate_way_out(self, tmp_path):
        summary = tmp_path / "summary"
        summary.mkdir()
        (summary / "h4_verdict.md").write_text("FROZEN ORIGINAL — do not touch\n")
        with pytest.raises(SystemExit) as excinfo:
            run_h4_analysis._refuse_if_frozen(summary)
        message = str(excinfo.value)
        assert "h4_verdict.md" in message
        assert "PS-7/PS-4" in message
        assert "remove or rename" in message
        assert "Nothing was written" in message

    def test_the_guard_and_the_writer_name_the_same_file(self):
        """Guard and write path share one constant, so they cannot drift apart."""
        assert run_h4_analysis.FROZEN_VERDICT == "h4_verdict.md"


# ===========================================================================
# 7 — control count read
# ===========================================================================


@pytest.mark.parametrize(
    "n_neg,n_pos,expected",
    [(20, 0, "ATTENUATION"), (16, 4, "ATTENUATION"), (15, 5, "ABSENCE"), (10, 10, "ABSENCE")],
)
def test_control_read_labels_absence_versus_attenuation(n_neg, n_pos, expected):
    assert h4._control_read(n_neg, n_pos).startswith(expected)
