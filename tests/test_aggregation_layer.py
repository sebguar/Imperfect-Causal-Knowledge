"""Aggregation-layer gates: manual recompute, schema, PS-6 invariants.

Permanent validation tests, parallel in style to the G1 anchor and L1-oracle
wiring gates: a failure here is a HARNESS BUG, not a finding. Covers, per the
Aggregation spec:

  1. manual numpy recompute — the acceptance gate (no pandas group-by, no
     aggregation-package call, atol=1e-10);
  2. schema conformance — exact column sets + dtypes of both aggregate CSVs;
  3. Gap_cost collapse (PS-6 anchor) — exactly 0 at SCM-carrying conditions,
     strictly > 0 at L0;
  4. ValidityGap_g monotonicity — the preserved H2 signal;
  5. ΔB_g_vs_L2 two-pass join correctness (NaN at L2; stored-mean subtraction);
  6. N_eligible / N_valid monotonicity;
  7. Δ_dist presence pattern (finite at L0, NaN elsewhere);
  8. run_regime default flip to the full ladder;
  9. regression via the pinned checksums — two-hop continuity back to the pilot.

Reference values (L2 means etc.) are read programmatically from the
per-individual CSVs — no hardcoded pilot numeric constants.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icknowledge.aggregation import (
    BY_CELL_COLUMNS,
    BY_CELL_INT_COLUMNS,
    BY_GROUP_COLUMNS,
    BY_GROUP_INT_COLUMNS,
    CellKeys,
    aggregate_run,
)
from icknowledge.recourse.pipeline import run_regime
from icknowledge.utils.config import load_config

# Continuity anchors: the pinned pilot regression constants are imported,
# not duplicated — one source of truth for the checksum gate (test 9).
from tests.test_l1_oracle_wiring import (
    _PRE_EXISTING_COL_DIGESTS,
    _PRE_EXISTING_COLS,
    _canonical_digest,
)

_CONFIG_PATH = "configs/recourse_triangle.yaml"
_RESULTS_DIR = Path("results/recourse_triangle")
_REGIMES = ("additive", "effect_modifying")
_CONDITIONS = ("L0", "L1-oracle", "L2")
_SEED = 20260710  # the config's master seed — the seed cell key of the pilot CSVs
_SCM_CARRYING = ("L1-oracle", "L2")  # SCM-carrying conditions in the three-rung tree


def _per_individual_path(regime: str, condition: str) -> Path:
    return _RESULTS_DIR / f"scoring_table_{regime}_{condition}.csv"


def _require_task2_csvs() -> None:
    missing = [
        str(p)
        for regime in _REGIMES
        for condition in _CONDITIONS
        if not (p := _per_individual_path(regime, condition)).exists()
    ]
    if missing:  # results/ is local-only (gitignored); skip cleanly on fresh clones
        pytest.skip(f"pilot per-individual CSVs not present locally: {missing}")


# Aggregation is a cheap roll-up; run it once per module into a tmp dir so the
# tests exercise aggregate_run end-to-end (CSV -> CSV) without touching results/.
@pytest.fixture(scope="module")
def aggregates(tmp_path_factory) -> dict[str, dict[str, pd.DataFrame]]:
    _require_task2_csvs()
    out_dir = tmp_path_factory.mktemp("aggregates")
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for regime in _REGIMES:
        cell = CellKeys(topology="triangle", family="linear", regime=regime, seed=_SEED)
        by_group_path, by_cell_path = aggregate_run(_RESULTS_DIR, out_dir, cell)
        frames[regime] = {
            "by_group": pd.read_csv(by_group_path, float_precision="round_trip"),
            "by_cell": pd.read_csv(by_cell_path, float_precision="round_trip"),
        }
    return frames


def _read_columns_numpy(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load selected per-individual CSV columns with stdlib csv + numpy only.

    Deliberately NO pandas anywhere in this loader: the acceptance gate must
    catch any groupby / merge / dtype issue at the layer boundary. (Raw
    np.genfromtxt is unsafe here — full-acted rows quote a comma inside the
    acted_set field — so the stdlib csv parser feeds numpy instead.)
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    out: dict[str, np.ndarray] = {}
    for col in columns:
        if col == "found":
            out[col] = np.array([r[col] == "True" for r in rows])
        else:
            out[col] = np.array([float(r[col]) for r in rows])
    return out


def _group_row(by_group: pd.DataFrame, condition: str, group: int) -> pd.Series:
    rows = by_group[(by_group["condition"] == condition) & (by_group["group"] == group)]
    assert len(rows) == 1, f"expected exactly one ({condition}, {group}) row"
    return rows.iloc[0]


def _cell_row(by_cell: pd.DataFrame, condition: str) -> pd.Series:
    rows = by_cell[by_cell["condition"] == condition]
    assert len(rows) == 1, f"expected exactly one {condition} row"
    return rows.iloc[0]


# --------------------------------------------------------------------------- #
# 1 — manual numpy recompute (THE acceptance gate)
# --------------------------------------------------------------------------- #


def test_manual_numpy_recompute_additive_l1_oracle(aggregates):
    """Recompute the additive × linear × seed 20260710 × L1-oracle means by hand.

    A failure here is a HARNESS bug, not a finding: the per-individual CSV is the
    source of truth, and the aggregate row must equal a hand-computed numpy mean
    over the eligible (found) set to floating-point noise.
    """
    cols = _read_columns_numpy(
        _per_individual_path("additive", "L1-oracle"), ("A", "realized_cost", "found")
    )
    eligible = cols["found"]
    means = {}
    for group in (-1, 1):
        mask = eligible & (cols["A"] == float(group))
        # mean of the ‖δ‖ column by hand — no pandas, no aggregation-package call.
        means[group] = np.mean(cols["realized_cost"][mask])
        stored = _group_row(aggregates["additive"]["by_group"], "L1-oracle", group)
        assert stored["N_eligible"] == int(mask.sum())
        np.testing.assert_allclose(
            stored["realized_mean_cost"], means[group], rtol=0, atol=1e-10
        )
    # cell-level Δ_cost from the same hand-computed means (orientation).
    stored_cell = _cell_row(aggregates["additive"]["by_cell"], "L1-oracle")
    np.testing.assert_allclose(
        stored_cell["Δ_cost"], means[-1] - means[1], rtol=0, atol=1e-10
    )


# --------------------------------------------------------------------------- #
# 2 — schema conformance (exact columns, exact dtypes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_schema_conformance(aggregates, regime):
    """No extra columns (silent downstream drift), no missing columns (broken analysis)."""
    specs = [
        ("by_group", BY_GROUP_COLUMNS, BY_GROUP_INT_COLUMNS),
        ("by_cell", BY_CELL_COLUMNS, BY_CELL_INT_COLUMNS),
    ]
    for artifact, columns, int_columns in specs:
        frame = aggregates[regime][artifact]
        assert list(frame.columns) == list(columns), f"{artifact} column drift"
        for col in columns:
            dtype = frame[col].dtype
            if col in int_columns:
                assert dtype == np.int64, f"{artifact}.{col}: {dtype}"
            elif col in ("topology", "family", "regime", "condition"):
                assert pd.api.types.is_string_dtype(dtype), f"{artifact}.{col}: {dtype}"
            else:
                assert dtype == np.float64, f"{artifact}.{col}: {dtype}"


# --------------------------------------------------------------------------- #
# 3 — Gap_cost collapse (PS-6 anchor, pinned as a code-level invariant)
# --------------------------------------------------------------------------- #


def test_gap_cost_collapse(aggregates):
    """Gap_cost == 0 EXACTLY at every SCM-carrying condition; > 0 strictly at L0.

    Exact zero, not <= tol: under the ℓ₂ intervention-cost convention the per-row believed cost
    equals realized
    cost bitwise for the chosen δ, so the collapse is an algebraic identity, not
    an approximate one. At L0 the gap is strictly positive but SMALL on this SCM
    — it is driven by the partial-acted exception set, so it scales as
    O(N_partial_acted / N_eligible) (34 of ~2400 additive rows ⇒ ≈7.5e-4): assert
    > 0 strictly, never a magnitude.
    """
    for regime in _REGIMES:
        by_cell = aggregates[regime]["by_cell"]
        for condition in _SCM_CARRYING:
            assert _cell_row(by_cell, condition)["Gap_cost"] == 0.0, (
                f"{regime}/{condition}: PS-6 collapse violated — Gap_cost must be "
                "identically zero at an SCM-carrying condition."
            )
    # The partial-acted exception set guarantees Δ_dist ≠ Δ_cost at L0 on
    # at least the additive-triangle cell.
    assert _cell_row(aggregates["additive"]["by_cell"], "L0")["Gap_cost"] > 0.0


# --------------------------------------------------------------------------- #
# 4 — ValidityGap_g: the preserved monotone H2 signal
# --------------------------------------------------------------------------- #


def test_validity_gap_monotone_signal(aggregates):
    """Zero at L2 exactly (true SCM: believed == realized validity); large at L1-oracle.

    The additive A=−1 threshold is > 0.5, not the exact pilot value (~0.957):
    the test pins that the believed-overpromise MECHANISM reads through the
    aggregation, without coupling to one run's fourth decimal.
    """
    for regime in _REGIMES:
        l2 = _cell_row(aggregates[regime]["by_cell"], "L2")
        assert l2["Gap_validity_A_neg"] == 0.0
        assert l2["Gap_validity_A_pos"] == 0.0
    l1 = _cell_row(aggregates["additive"]["by_cell"], "L1-oracle")
    assert l1["Gap_validity_A_neg"] > 0.5


# --------------------------------------------------------------------------- #
# 5 — ΔB_g_vs_L2 correctness under the two-pass write (join, not means)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_delta_b_two_pass_join(aggregates, regime):
    """ΔB_g at L2 is NaN; elsewhere it equals the stored-mean subtraction exactly.

    Asserted by direct subtraction from STORED means (not recomputed from
    per-individual data): this pins the (topology, family, regime, seed, group)
    JOIN correctness, not the mean correctness — test 1 owns the means.
    """
    by_group = aggregates[regime]["by_group"]
    for group in (-1, 1):
        assert np.isnan(_group_row(by_group, "L2", group)["ΔB_g_vs_L2"])
        l2_mean = _group_row(by_group, "L2", group)["realized_mean_cost"]
        for condition in ("L0", "L1-oracle"):
            row = _group_row(by_group, condition, group)
            assert row["ΔB_g_vs_L2"] == row["realized_mean_cost"] - l2_mean


# --------------------------------------------------------------------------- #
# 6 — N_eligible / N_valid monotonicity (guard against population drift)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_population_counts_monotone(aggregates, regime):
    by_group = aggregates[regime]["by_group"]
    assert (by_group["N_valid"] <= by_group["N_eligible"]).all()
    # L2 realized-validity is uniformly 1.00: N_valid == N_eligible.
    l2 = by_group[by_group["condition"] == "L2"]
    assert (l2["N_valid"] == l2["N_eligible"]).all()
    # Wherever the per-individual data show L1-oracle validity < 1 for a group,
    # the aggregate row must show a strict N_valid < N_eligible.
    cols = _read_columns_numpy(
        _per_individual_path(regime, "L1-oracle"), ("A", "realized_validity", "found")
    )
    checked = 0
    for group in (-1, 1):
        mask = cols["found"] & (cols["A"] == float(group))
        if np.mean(cols["realized_validity"][mask]) < 1.0:
            row = _group_row(by_group, "L1-oracle", group)
            assert row["N_valid"] < row["N_eligible"]
            checked += 1
    assert checked >= 1, "no L1-oracle group with validity < 1 — pilot inputs changed?"


# --------------------------------------------------------------------------- #
# 7 — Δ_dist presence pattern (L0 only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_delta_dist_presence_pattern(aggregates, regime):
    """Finite at L0; NaN (not zero) at L2 and L1-oracle — zero would falsely
    suggest computed-and-vanishing; Δ_dist there is an H2/H3 story."""
    by_cell = aggregates[regime]["by_cell"]
    assert np.isfinite(_cell_row(by_cell, "L0")["Δ_dist"])
    for condition in _SCM_CARRYING:
        assert np.isnan(_cell_row(by_cell, condition)["Δ_dist"])


# --------------------------------------------------------------------------- #
# 8 — run_regime default flip: "run" now means the full ladder
# --------------------------------------------------------------------------- #


def test_run_regime_default_is_full_ladder():
    """No-conditions call produces L0 + L1-oracle + L2; the legacy two-condition
    world stays constructible by passing the old tuple explicitly (the pre-existing
    test_recourse_triangle.py module — unedited in its assertions — runs exactly
    that tuple in this same suite)."""
    cfg = load_config(_CONFIG_PATH)
    run_default = run_regime(cfg, "additive")
    assert set(run_default.table["condition"].unique()) == {"L0", "L1-oracle", "L2"}
    run_old = run_regime(cfg, "additive", conditions=("L0", "L2"))
    assert set(run_old.table["condition"].unique()) == {"L0", "L2"}


# --------------------------------------------------------------------------- #
# 9 — regression via checksums: two-hop continuity back to the pilot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_pre_existing_cols_continuity_two_hop(aggregates, regime):
    """Hop 1: the per-individual CSVs still hash to the pinned checksums.
    Hop 2: aggregation reads ONLY those CSVs (test 1 + this module's fixture), so
    the aggregates are anchored back to the pilot transitively."""
    for condition in ("L0", "L2"):
        table = pd.read_csv(
            _per_individual_path(regime, condition), float_precision="round_trip"
        )
        table["acted_set"] = table["acted_set"].fillna("")
        frame = table[_PRE_EXISTING_COLS].reset_index(drop=True)
        assert _canonical_digest(frame) == _PRE_EXISTING_COL_DIGESTS[(regime, condition)], (
            f"{regime}/{condition}: per-individual CSV no longer reproduces the pilot — "
            "the aggregation inputs have drifted from the frozen baseline."
        )
        # numpy group means from the same CSV must be what the aggregate stores.
        cols = _read_columns_numpy(
            _per_individual_path(regime, condition), ("A", "realized_cost", "found")
        )
        for group in (-1, 1):
            mask = cols["found"] & (cols["A"] == float(group))
            stored = _group_row(aggregates[regime]["by_group"], condition, group)
            np.testing.assert_allclose(
                stored["realized_mean_cost"],
                np.mean(cols["realized_cost"][mask]),
                rtol=0,
                atol=1e-10,
            )
