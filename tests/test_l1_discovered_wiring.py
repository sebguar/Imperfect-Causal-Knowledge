"""L1-discovered pipeline wiring gates (PS-7, PS-4, the L1-oracle form-template-known
semantics).

Permanent wiring tests, parallel in style to tests/test_l1_oracle_wiring.py: a
failure here is a WIRING BUG, not a finding.

  1. CENTERPIECE — LAZY-GATE BYTE-IDENTITY. The same config and the same seed,
     run once at three rungs and once at four, must produce FLOAT-EXACT identical
     per-individual scoring tables for the three shared conditions. This is what
     discharges "the frozen L0 / L1-oracle / L2 numbers are untouched by
     construction" (PS-7, PS-4) at the pipeline level: the four-rung tree's
     strict-subset status rests on it, and no amount of code reading substitutes
     for running both and comparing.
  2. Four-rung end-to-end smoke: the rung scores, the record comes back, and a
     written manifest carries the discovery block. (The wiring half of the
     stop-loss; discovery is covered in isolation.)
  3. The L1-oracle form-template-known semantics' shared-sample assertion actually fires on a
  mismatch.
  4. Dynamic-arity denominator suffix (PS-4 note (1)).
  5. Prefix-check parameterization (PS-4 note (2)).
  6. Manifest: the discovery block round-trips; causal-learn is tracked.

EVERY artifact this module writes goes to ``tmp_path``. Nothing here touches
``results/``, and no grid is launched.
"""

from __future__ import annotations

import dataclasses
import json
import warnings

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from icknowledge.aggregation import (
    BY_CELL_COLUMNS,
    BY_GROUP_COLUMNS,
    CellKeys,
    aggregate_run,
    build_by_group_frame,
    by_cell_columns,
    by_group_columns,
    common_found_suffix,
)
from icknowledge.discovery import DiscoveryRecord
from icknowledge.estimation import estimation_provenance
from icknowledge.recourse import pipeline as pipeline_module
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import collider_form_spec, make_linear_collider
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest

_THREE_RUNG = ("L0", "L1-oracle", "L2")
#: [PS-4] canonical four-rung order, matching aggregation._CONDITION_ORDER.
_FOUR_RUNG = ("L0", "L1-oracle", "L1-discovered", "L2")
_REGIME = "additive"


def _small_cfg(path: str, resolution: int):
    """Test-scale config: NOT the grid-resolution production pin (see test_l1_oracle_collider)."""
    cfg = load_config(path)
    cfg.classifier.N = 600
    cfg.classifier.min_neg_per_group = 30
    cfg.recourse.grid.resolution = resolution
    return cfg


def _paired_runs(config_path: str, resolution: int, **kwargs):
    """(three-rung run, four-rung run) on ONE config object and ONE master seed.

    The SAME `cfg` object is handed to both calls — not two loads of the same
    file — so the byte-identity claim cannot be weakened by a config that
    silently differs between them.
    """
    cfg = _small_cfg(config_path, resolution)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        three = run_regime(
            cfg, _REGIME, conditions=_THREE_RUNG, run_anchor=False, **kwargs
        )
        four = run_regime(
            cfg, _REGIME, conditions=_FOUR_RUNG, run_anchor=False, **kwargs
        )
    return cfg, three, four


@pytest.fixture(scope="module")
def triangle_pair():
    return _paired_runs("configs/recourse_triangle.yaml", resolution=21)


@pytest.fixture(scope="module")
def collider_pair():
    """Second topology, kept cheap: k=3 means candidates scale as res³."""
    return _paired_runs(
        "configs/recourse_collider.yaml",
        resolution=11,
        scm_builder=make_linear_collider,
        form_spec_fn=collider_form_spec,
    )


def _condition_table(run, condition: str) -> pd.DataFrame:
    sub = run.table[run.table["condition"] == condition]
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1 — CENTERPIECE: the lazy gate costs the three shared rungs nothing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("condition", _THREE_RUNG)
def test_shared_scoring_tables_are_float_exact_triangle(triangle_pair, condition):
    """[PS-4] Three-rung vs four-rung: the shared tables must be IDENTICAL.

    `check_exact=True` — not a tolerance. A four-way common-found population is a
    strict SUBSET of the three-way one only if the underlying per-individual
    scoring is the very same scoring; a 1-ulp drift would mean the fourth rung
    had perturbed something (an RNG stream, a fit, a grid) and the two trees
    would be measuring subtly different experiments under one name.
    """
    _cfg, three, four = triangle_pair
    assert_frame_equal(
        _condition_table(three, condition),
        _condition_table(four, condition),
        check_exact=True,
    )


@pytest.mark.parametrize("condition", _THREE_RUNG)
def test_shared_scoring_tables_are_float_exact_collider(collider_pair, condition):
    """Same gate on a second topology — the guarantee is not triangle-specific."""
    _cfg, three, four = collider_pair
    assert_frame_equal(
        _condition_table(three, condition),
        _condition_table(four, condition),
        check_exact=True,
    )


def test_three_rung_run_is_untouched_off_the_rung(triangle_pair):
    """The opt-out path allocates no discovery state at all."""
    _cfg, three, four = triangle_pair
    assert three.discovery is None and three.fitted_discovered is None
    assert set(three.table["condition"].unique()) == set(_THREE_RUNG)
    # The L1-ORACLE fit is the same object shape in both runs: `fitted` keeps its
    # singular meaning and is NOT repurposed by the fourth rung.
    assert three.fitted.coefficients == four.fitted.coefficients
    assert three.fitted.metadata == four.fitted.metadata


def test_four_rung_summary_covers_four_conditions(triangle_pair):
    _cfg, _three, four = triangle_pair
    assert set(four.table["condition"].unique()) == set(_FOUR_RUNG)
    assert set(four.summary["condition"].unique()) == set(_FOUR_RUNG)


# --------------------------------------------------------------------------- #
# 2 — four-rung end-to-end smoke (the wiring half of the stop-loss)
# --------------------------------------------------------------------------- #


def test_l1_discovered_scoring_table_exists_and_is_populated(triangle_pair, tmp_path):
    """The rung SCORES: a non-empty per-individual table lands on disk."""
    _cfg, _three, four = triangle_pair
    table = _condition_table(four, "L1-discovered")
    assert not table.empty
    # Same eligible population as every other condition — the common-found population's matched-set
    # rule.
    assert len(table) == len(_condition_table(four, "L2"))
    assert table["found"].any(), "no feasible action at L1-discovered on any individual"
    assert table["realized_cost"][table["found"]].notna().all()

    path = tmp_path / f"scoring_table_{_REGIME}_L1-discovered.csv"
    table.to_csv(path, index=False)
    reread = pd.read_csv(path, float_precision="round_trip")
    assert len(reread) == len(table)


def test_discovery_record_is_populated_and_truth_free(triangle_pair):
    _cfg, _three, four = triangle_pair
    record = four.discovery
    assert isinstance(record, DiscoveryRecord)
    assert record.library == "causal-learn" and record.library_version
    assert record.canonical_order == ("A", "X1", "X2")
    assert record.edges, "PC returned no oriented edge"
    # A is a root by PS-7 background knowledge — every A-edge is BK-attributed.
    assert all(e.channel == "bk" for e in record.edges if e.source == "A")
    assert four.fitted_discovered is not None
    # The discovered fit is a SECOND fit, tagged with its own sample note.
    assert four.fitted_discovered.metadata["sample"] == (
        "same-as-estimation (the L1-oracle form-template-known semantics)"
    )


def test_four_rung_manifest_carries_a_real_discovery_block(triangle_pair, tmp_path):
    """A written manifest carries the block, and it round-trips through JSON."""
    cfg, _three, four = triangle_pair
    path = write_run_manifest(
        tmp_path,
        cfg,
        estimation={_REGIME: estimation_provenance(four.fitted)},
        discovery={_REGIME: four.discovery.as_dict()},
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    block = manifest["discovery"][_REGIME]
    assert block["library"] == "causal-learn"
    assert block["canonical_order"] == ["A", "X1", "X2"]
    assert {"source", "target", "channel", "was_bidirected"} <= set(block["edges"][0])
    assert block["n_bidirected"] == 0
    assert len(block["adjacency"]) == len(block["canonical_order"])
    # [the L1-oracle form-template-known semantics] The sample identity rides in the manifest
    # in the SAME shape
    # the estimation block uses, so the two are comparable by an auditor.
    assert block["dataset"]["sha256"] == (
        manifest["estimation"][_REGIME]["metadata"]["dataset"]["sha256"]
    )
    # The discovery library is tracked like every other dependency.
    assert "causal-learn" in manifest["package_versions"]
    assert manifest["package_versions"]["causal-learn"] != "not installed"


def test_four_rung_aggregation_emits_fourway_columns(triangle_pair, tmp_path):
    """End-to-end through the aggregation layer: four rungs -> _fourway denominators."""
    _cfg, _three, four = triangle_pair
    for condition in _FOUR_RUNG:
        _condition_table(four, condition).to_csv(
            tmp_path / f"scoring_table_{_REGIME}_{condition}.csv", index=False
        )
    cell = CellKeys(topology="triangle", family="linear", regime=_REGIME, seed=20260710)
    by_group_path, by_cell_path = aggregate_run(tmp_path, tmp_path / "agg", cell)

    by_group = pd.read_csv(by_group_path, float_precision="round_trip")
    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")
    assert list(by_group.columns) == list(by_group_columns(4))
    assert list(by_cell.columns) == list(by_cell_columns(4))
    assert set(by_group["condition"]) == set(_FOUR_RUNG)
    # The denominator is legible from the header and no threeway label survives.
    assert "N_common_found_fourway" in by_group.columns
    assert not any("threeway" in c for c in by_group.columns)
    assert not any("threeway" in c for c in by_cell.columns)


# --------------------------------------------------------------------------- #
# 3 — the L1-oracle form-template-known semantics' shared-sample assertion fires
# --------------------------------------------------------------------------- #


def test_d_ii_assertion_fires_on_a_permuted_estimation_sample(monkeypatch):
    """[the L1-oracle form-template-known semantics] A sample the search did not see must HALT
    the run.

    The mismatch is constructed the way it could actually arise: the classifier
    hands back a dataset whose COLUMN ORDER is not the canonical lexical one, so
    the estimation-side hash and the discovery-side hash (taken on the
    canonically-reindexed frame) stop agreeing. That is precisely the
    note (g) coincidence going away, and the assertion is what makes it fail
    loudly at the wiring seam instead of quietly moving a tie-break.
    """
    cfg = _small_cfg("configs/recourse_triangle.yaml", resolution=9)
    real_build = pipeline_module.build_classifier

    def permuted_build_classifier(scm, cfg):
        result = real_build(scm, cfg)
        # Reverse the column order: same rows, same values, different hash.
        shuffled = result.dataset[list(result.dataset.columns)[::-1]]
        return dataclasses.replace(result, dataset=shuffled)

    monkeypatch.setattr(pipeline_module, "build_classifier", permuted_build_classifier)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="the L1-oracle form-template-known semantics"):
            run_regime(cfg, _REGIME, conditions=_FOUR_RUNG, run_anchor=False)


def test_d_ii_mismatch_does_not_affect_a_three_rung_run(monkeypatch):
    """The same permutation is INERT off the rung — no search, nothing to assert."""
    cfg = _small_cfg("configs/recourse_triangle.yaml", resolution=9)
    real_build = pipeline_module.build_classifier

    def permuted_build_classifier(scm, cfg):
        result = real_build(scm, cfg)
        shuffled = result.dataset[list(result.dataset.columns)[::-1]]
        return dataclasses.replace(result, dataset=shuffled)

    monkeypatch.setattr(pipeline_module, "build_classifier", permuted_build_classifier)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_regime(cfg, _REGIME, conditions=_THREE_RUNG, run_anchor=False)
    assert run.discovery is None


# --------------------------------------------------------------------------- #
# 4 — dynamic-arity denominator suffix (PS-4 note (1))
# --------------------------------------------------------------------------- #


def test_arity_suffix_map():
    assert common_found_suffix(3) == "threeway"
    assert common_found_suffix(4) == "fourway"
    # Arity 2 is mapped, not raising: the collider and chain smoke
    # cells aggregate an L0+L2 run, and before the map they wrote that two-way
    # intersection under a `_threeway` header. See the schema.py note.
    assert common_found_suffix(2) == "twoway"
    for arity in (0, 1, 5, 12):
        with pytest.raises(ValueError, match="no common-found denominator label"):
            common_found_suffix(arity)


def test_arity_three_column_names_are_byte_identical():
    """The whole point of the map: three rungs emit EXACTLY the sealed names."""
    assert by_group_columns(3) == BY_GROUP_COLUMNS
    assert by_cell_columns(3) == BY_CELL_COLUMNS
    assert by_group_columns() == BY_GROUP_COLUMNS  # default arity is 3
    assert by_cell_columns() == BY_CELL_COLUMNS


def test_arity_four_renames_only_the_denominator_columns():
    four = by_group_columns(4)
    assert len(four) == len(BY_GROUP_COLUMNS)
    renamed = {
        old: new for old, new in zip(BY_GROUP_COLUMNS, four, strict=True) if old != new
    }
    assert renamed == {
        "N_common_found_threeway": "N_common_found_fourway",
        "realized_cost_common_found_threeway": "realized_cost_common_found_fourway",
    }
    # The pairwise-vs-L2 columns are a DIFFERENT population and keep their names.
    assert "N_common_found_pairwise_vs_L2" in four


_SYNTH_FEATURES = ("X1", "X2")


def _synthetic_table(condition: str, n: int = 40) -> pd.DataFrame:
    """Minimal well-formed scoring table — enough for the aggregation layer."""
    records = []
    for i in range(n):
        rec = {
            "condition": condition,
            "index": i,
            "A": -1.0 if i % 2 == 0 else 1.0,
            "acted_set": "X1",
            "believed_cost": 1.0 + i / 100.0,
            "realized_cost": 1.0 + i / 100.0,
            "believed_validity": 1,
            "realized_validity": 1,
            "found": True,
        }
        for f in _SYNTH_FEATURES:
            rec[f"delta_{f}"] = 0.5
            rec[f"factual_{f}"] = float(i)
            rec[f"realized_cf_{f}"] = float(i) + 0.5
            rec[f"believed_cf_{f}"] = float(i) + 0.5
        records.append(rec)
    return pd.DataFrame.from_records(records)


def test_by_group_frame_labels_its_own_denominator():
    """The frame builder derives the suffix from what it actually intersected."""
    cell = CellKeys(topology="triangle", family="linear", regime=_REGIME, seed=1)
    three = {c: _synthetic_table(c) for c in _THREE_RUNG}
    four = {c: _synthetic_table(c) for c in _FOUR_RUNG}

    assert list(build_by_group_frame(three, cell).columns) == list(BY_GROUP_COLUMNS)
    assert list(build_by_group_frame(four, cell).columns) == list(by_group_columns(4))


def test_five_conditions_raise_rather_than_emit_an_unlabelled_denominator():
    """[PS-4 note (1)] An unmapped arity is a hard stop, not a fallback."""
    cell = CellKeys(topology="triangle", family="linear", regime=_REGIME, seed=1)
    five = {c: _synthetic_table(c) for c in (*_FOUR_RUNG, "L-hypothetical")}
    with pytest.raises(ValueError, match="no common-found denominator label"):
        build_by_group_frame(five, cell)


def test_two_condition_cell_labels_its_denominator_twoway():
    """The pre-existing L0+L2 smoke shape: two-way population, two-way label.

    Before the arity map this cell emitted `N_common_found_threeway` while
    holding a TWO-way intersection — the same unlabelled-denominator defect
    PS-4 note (1) forecloses at four rungs, latent at two. The population was
    always correct (`common_found_populations` intersects over whatever was
    scored); only the header lied.
    """
    cell = CellKeys(topology="collider", family="linear", regime=_REGIME, seed=1)
    two = {c: _synthetic_table(c) for c in ("L0", "L2")}
    frame = build_by_group_frame(two, cell)
    assert list(frame.columns) == list(by_group_columns(2))
    assert "N_common_found_twoway" in frame.columns
    assert not any("threeway" in c for c in frame.columns)


# --------------------------------------------------------------------------- #
# 5 — prefix-check parameterization (PS-4 note (2))
# --------------------------------------------------------------------------- #


def _write_tree(root, topologies, conditions, *, perturb=None):
    """Materialize a minimal grid tree: per-condition scoring tables + aggregates.

    Enumerates exactly what `analysis.loading.grid_cells` enumerates (family ×
    regime under each topology), because `compare` walks that enumeration and
    reports a missing file as a failure — a partial fixture would fail for a
    reason that has nothing to do with what is under test.
    """
    from icknowledge.analysis.loading import grid_cells
    from scripts.run_cross_seed_grid import cell_dir_name

    for cell in grid_cells(tuple(topologies)):
        out = root / cell_dir_name(cell.topology, cell.family, cell.regime, 0)
        out.mkdir(parents=True, exist_ok=True)
        for condition in conditions:
            table = _synthetic_table(condition)
            if perturb is not None and (cell.topology, condition) == perturb:
                table.loc[0, "realized_cost"] += 1e-12
            table.to_csv(
                out / f"scoring_table_{cell.regime}_{condition}.csv", index=False
            )
        aggregate_run(
            out,
            out,
            CellKeys(
                topology=cell.topology,
                family=cell.family,
                regime=cell.regime,
                seed=0,
            ),
        )


_ALL_TOPOLOGIES = ("triangle", "collider", "chain")
_PREFIX_TOPOLOGIES = ("triangle", "collider")


def test_scoring_only_mode_passes_across_all_three_topologies(tmp_path):
    """The PS-4 strict-subset assertion: shared tables identical, aggregates ignored."""
    from scripts.verify_grid_prefix import compare

    ref, new = tmp_path / "ref", tmp_path / "new"
    _write_tree(ref, _ALL_TOPOLOGIES, _THREE_RUNG)
    # The four-rung tree re-materializes the three shared conditions identically
    # and adds a fourth — so its AGGREGATES differ (extra row, _fourway columns)
    # while its shared scoring tables do not. Exactly the PS-4 situation.
    _write_tree(new, _ALL_TOPOLOGIES, _FOUR_RUNG)

    assert (
        compare(
            ref, new, n_seeds=1, topologies=_ALL_TOPOLOGIES, kinds="scoring-only"
        )
        == 0
    )


def test_scoring_only_mode_still_catches_a_perturbed_table(tmp_path):
    """Excluding aggregates must not blunt the check on what it DOES compare."""
    from scripts.verify_grid_prefix import compare

    ref, new = tmp_path / "ref", tmp_path / "new"
    _write_tree(ref, _ALL_TOPOLOGIES, _THREE_RUNG)
    _write_tree(new, _ALL_TOPOLOGIES, _FOUR_RUNG, perturb=("collider", "L1-oracle"))

    assert (
        compare(
            ref, new, n_seeds=1, topologies=_ALL_TOPOLOGIES, kinds="scoring-only"
        )
        == 1
    )


def test_default_mode_is_unchanged_and_still_compares_aggregates(tmp_path):
    """The PS-1 call shape keeps its exact behaviour (defaults: the N=6 pair, all)."""
    from scripts.verify_grid_prefix import compare

    ref, new = tmp_path / "ref", tmp_path / "new"
    _write_tree(ref, _PREFIX_TOPOLOGIES, _THREE_RUNG)
    _write_tree(new, _PREFIX_TOPOLOGIES, _THREE_RUNG)
    assert compare(ref, new, n_seeds=1) == 0

    # ... and a four-rung tree FAILS the default mode, because its aggregates
    # legitimately diverge. That failure is why note (2) scopes the check.
    four = tmp_path / "four"
    _write_tree(four, _PREFIX_TOPOLOGIES, _FOUR_RUNG)
    assert compare(ref, four, n_seeds=1) == 1


def test_unknown_kinds_is_rejected(tmp_path):
    from scripts.verify_grid_prefix import compare

    with pytest.raises(ValueError, match="unknown kinds"):
        compare(tmp_path, tmp_path, n_seeds=1, kinds="aggregates-only")


# --------------------------------------------------------------------------- #
# 6 — grid-driver flag threading (no grid is launched)
# --------------------------------------------------------------------------- #


def test_grid_flag_threads_conditions_without_mutating_runner_tuples():
    """[PS-4] --l1-discovered overrides per call; the frozen tuples stand."""
    from icknowledge.recourse import run_chain, run_collider, run_triangle
    from scripts.run_cross_seed_grid import L1D_CONDITIONS

    assert L1D_CONDITIONS == _FOUR_RUNG
    # The three driver tuples are frozen three-rung artifacts and stay that way —
    # the fourth rung is a property of the grid INVOCATION, not of a topology.
    for runner in (run_triangle, run_collider, run_chain):
        assert set(runner._CONDITIONS) == set(_THREE_RUNG)


def test_grid_argparse_exposes_the_flag_and_defaults_off():
    import inspect

    from scripts import run_cross_seed_grid

    source = inspect.getsource(run_cross_seed_grid.main)
    assert "--l1-discovered" in source
    # OFF unless asked: the flag is store_true, so an ordinary grid launch keeps
    # the three-rung condition set and the sealed tree's reproducibility.
    assert "action=\"store_true\"" in source
    # `run_cell_seed` takes the override as an explicit parameter defaulting to
    # None (= "use the runner's frozen tuple"), so nothing mutates module state.
    signature = inspect.signature(run_cross_seed_grid.run_cell_seed)
    assert signature.parameters["conditions"].default is None


# --------------------------------------------------------------------------- #
# 7 — analysis-loader parameterization (deviation 2)
# --------------------------------------------------------------------------- #


def test_analysis_loader_conditions_default_is_the_three_rung_ladder():
    import inspect

    from icknowledge.analysis import loading

    assert loading.CONDITIONS == _THREE_RUNG  # unchanged default
    assert loading.CONDITIONS_L1D == _FOUR_RUNG
    assert loading.L1D_ROOT.name == "cross_seed_L1d_N20"
    signature = inspect.signature(loading.load_scoring_tables)
    assert signature.parameters["conditions"].default == _THREE_RUNG
    assert signature.parameters["root"].default == loading.DEFAULT_ROOT


def test_analysis_loader_reads_a_four_rung_cell_when_asked(tmp_path):
    from icknowledge.analysis.loading import Cell, load_scoring_tables

    cell = Cell("triangle", "linear", _REGIME)
    out = cell.seed_dir(0, tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    for condition in _FOUR_RUNG:
        _synthetic_table(condition).to_csv(
            out / f"scoring_table_{_REGIME}_{condition}.csv", index=False
        )

    tables = load_scoring_tables(cell, 0, tmp_path, conditions=_FOUR_RUNG)
    assert set(tables) == set(_FOUR_RUNG)
    assert all(isinstance(t, pd.DataFrame) and not t.empty for t in tables.values())
    # The default (three-rung) read of the same directory silently sees three —
    # which is why root and conditions are passed together, never one alone.
    assert set(load_scoring_tables(cell, 0, tmp_path)) == set(_THREE_RUNG)
