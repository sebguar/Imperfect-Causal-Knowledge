"""CLI entry point: run the recourse pipeline on the TRIANGLE.

    python -m icknowledge.recourse.run_triangle [config_path]

Runs the additive and effect-modifying triangle regimes under the full ladder
L0 + L1-oracle + L2 via `pipeline.run_regime`. Prints the per-regime summary,
writes one per-individual scoring CSV per regime × condition, rolls each cell up
into the aggregate CSVs with ``topology="triangle"`` (a free-string cell key on
`CellKeys`), and writes the run manifest.

The grid driver (`scripts/run_cross_seed_grid.py`) reads `_FAMILIES` and
`_CONDITIONS` off the runner module for the topology it is running, so a
topology reaches the cross-seed grid by EXISTING as a module with those two
names — not by any hardcoded enumeration in the driver.

FUNCTIONAL FAMILY is read from the config's top-level ``family`` key and selects
the SCM builder, the supplied form template, the L1-oracle estimator and the
aggregate CSVs' ``family`` cell key:

    configs/recourse_triangle.yaml      family: linear
    configs/recourse_triangle_nlg.yaml  family: nlg     (tanh mechanism)

NOTE: the combined tables (scoring_table_<regime>.csv) are not rewritten by this
driver. tests/test_l1_oracle_wiring.py gates L0/L2 regression on sha256 digests
recorded in the test; the local CSVs, when present, are an extra per-column
comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from icknowledge.aggregation import AGGREGATION_CODE_VERSION, CellKeys, aggregate_run
from icknowledge.estimation import estimation_provenance
from icknowledge.recourse.pipeline import format_regime_summary, run_regime
from icknowledge.scm import (
    make_linear_triangle,
    make_nonlinear_gaussian_triangle,
    nlg_triangle_form_spec,
    triangle_form_spec,
)
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest
from icknowledge.utils.seeding import seeding_provenance

_DEFAULT_CONFIG = "configs/recourse_triangle.yaml"
_REGIMES = ("additive", "effect_modifying")
_CONDITIONS = ("L0", "L2", "L1-oracle")

#: family -> (SCM builder, supplied form template). The estimator paired with each
#: template is selected inside pipeline.run_regime from the same family key
#: (the L1-oracle form-template-known semantics/the NLG family specification), so template and
#: estimator cannot be mismatched here.
_FAMILIES = {
    "linear": (make_linear_triangle, triangle_form_spec),
    "nlg": (make_nonlinear_gaussian_triangle, nlg_triangle_form_spec),
}


def _format_instrumentation(
    by_cell_paths: dict[str, Path],
    summaries: dict[str, pd.DataFrame],
    wall_clock: dict[str, float],
) -> str:
    """Delta_cost, validity-rate-by-group and wall-clock per (regime x condition).

    # INSTRUMENTATION ONLY — no assertion, no threshold, no gate. [the
    # PS-3 extension (orientation lock): Delta_cost := mean(cost | A=-1) −
    # mean(cost | A=+1), positive = A=-1 more burdened.] These are H1 PRE-NUMBERS
    # from a single-seed smoke run; judging them would be a register decision, not
    # a code choice. Delta_cost is read BACK from the by-cell CSV rather than
    # recomputed, so the printed number is the one the artifact carries.
    #
    # RAW aggregates: the common-found population filter (PS-5) is grid-and-analysis
    # code. This driver reports RAW aggregates.
    """
    # ASCII-only in the PRINTED text (the CSV column keeps its Δ_cost name): stdout
    # here goes to a cp1252 Windows console, which cannot encode Δ and would raise
    # UnicodeEncodeError after the CSVs are written but BEFORE the manifest,
    # leaving a run half-recorded. Reading the Δ_cost column by its real name is
    # unaffected (the CSVs are UTF-8 files, not console output).
    lines = ["=== Delta_cost by regime x condition (INSTRUMENTATION - no gate) ==="]
    for regime, path in by_cell_paths.items():
        frame = pd.read_csv(path, float_precision="round_trip")
        for condition in ("L0", "L1-oracle", "L2"):
            rows = frame[frame["condition"] == condition]
            if rows.empty:
                continue
            value = float(rows.iloc[0]["Δ_cost"])
            lines.append(f"  {regime:<17} {condition:<10} Delta_cost = {value:+.6f}")

    lines.append("")
    lines.append("=== realized validity-rate by group x condition (INSTRUMENTATION) ===")
    for regime, summary in summaries.items():
        for _, row in summary.iterrows():
            lines.append(
                f"  {regime:<17} {row['condition']:<10} A={row['A']:+g}  "
                f"validity = {row['realized_validity_rate']:.4f}  "
                f"(n={int(row['n'])}, valid={int(row['n_realized_valid'])})"
            )

    lines.append("")
    lines.append("=== wall-clock per regime (seed-grid sizing input) ===")
    for regime, seconds in wall_clock.items():
        lines.append(f"  {regime:<17} {seconds:7.1f} s  (L0+L1-oracle+L2, full pool)")
    return "\n".join(lines)


def main(config_path: str = _DEFAULT_CONFIG) -> dict[str, float]:
    """Run the triangle ladder for the config's family; return per-regime seconds."""
    cfg = load_config(config_path)
    family = str(cfg.get("family", "linear"))
    if family not in _FAMILIES:
        raise ValueError(
            f"unknown functional family {family!r} in {config_path}; "
            f"wired: {sorted(_FAMILIES)}."
        )
    scm_builder, form_spec_fn = _FAMILIES[family]
    # NO G1 ANCHOR OFF THE LINEAR FAMILY: the Ehyaei closed form assumes a linear
    # SCM and a linear classifier (the ℓ₂ intervention-cost convention), so it does not
    # apply to an NLG
    # cell. run_regime refuses run_anchor=True for a non-linear family outright.
    run_anchor = family == "linear"

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimation_records: dict[str, dict] = {}
    aggregation_records: dict[str, dict] = {}
    by_cell_paths: dict[str, Path] = {}
    summaries: dict[str, pd.DataFrame] = {}
    wall_clock: dict[str, float] = {}
    for regime in _REGIMES:
        t0 = time.perf_counter()
        run = run_regime(
            cfg,
            regime,
            conditions=_CONDITIONS,
            scm_builder=scm_builder,
            form_spec_fn=form_spec_fn,
            run_anchor=run_anchor,
            family=family,
        )
        wall_clock[regime] = time.perf_counter() - t0
        summaries[regime] = run.summary
        print(format_regime_summary(run))
        # [the classifier construction] eps_scale is auto-calibrated per cell so trained-h test
        # accuracy lands in the band; the NLG value WILL differ from the linear
        # triangle's, which is correct — the classifier construction holds QUALITY
        # constant,
        # not the noise scale. Printed so the calibrated value is recorded.
        spec = run.classifier.label_spec
        print(
            f"calibrated eps_scale = {spec.eps_scale:.4f} (the classifier construction), "
            f"tau = {spec.tau:.4f}, test accuracy = {run.classifier.accuracy_overall:.4f}"
        )
        print(f"regime wall-clock (L0+L1-oracle+L2, full pool): {wall_clock[regime]:.1f} s")
        print()

        # One per-individual scoring CSV per regime x condition (2 x 3 = 6).
        for condition in _CONDITIONS:
            sub = run.table[run.table["condition"] == condition]
            table_path = out_dir / f"scoring_table_{regime}_{condition}.csv"
            sub.to_csv(table_path, index=False)
            print(f"wrote per-individual scoring table -> {table_path}")
        print()
        # [the L1-oracle form-template-known semantics] estimation provenance per regime: estimator
        # id, serialized
        # FormSpec, coefficients, dataset identity (= the classifier's training
        # sample), regime, family — the auditor's evidence for sample sharing.
        estimation_records[regime] = estimation_provenance(run.fitted)

        # Roll this cell's per-individual CSVs up into the two aggregate
        # CSVs (semantics per PS-6 + PS-3 extension). The
        # aggregation only consumes what was just written to disk — the
        # per-individual CSVs remain the source of truth. The aggregation layer is
        # family-agnostic; ``family`` is a cell KEY, so nothing there changes.
        cell = CellKeys(
            topology="triangle", family=family, regime=regime, seed=int(cfg.seed)
        )
        by_group_path, by_cell_path = aggregate_run(out_dir, out_dir / "aggregates", cell)
        print(f"wrote aggregate by-group table -> {by_group_path}")
        print(f"wrote aggregate by-cell table  -> {by_cell_path}")
        print()
        aggregation_records[regime] = {
            "code_version": AGGREGATION_CODE_VERSION,
            "by_group": str(by_group_path),
            "by_cell": str(by_cell_path),
        }
        by_cell_paths[regime] = by_cell_path

    print(_format_instrumentation(by_cell_paths, summaries, wall_clock))
    print()

    # [the seed-generation mechanics] Seeding provenance: the pinned grid meta-entropy + this run's
    # master seed. This is a single-cell run (one seed), so no n_runs is passed
    # and the spawned per-run seed list is not materialized here; the replication
    # grid (PS-1) spawns it in scripts/run_cross_seed_grid.py. The meta-entropy is recorded
    # in every run's manifest so the whole grid is reproducible from one integer.
    seeding_record = seeding_provenance(int(cfg.seed))

    manifest_path = write_run_manifest(
        out_dir,
        cfg,
        estimation=estimation_records,
        aggregation=aggregation_records,
        seeding=seeding_record,
    )
    print(f"wrote run manifest -> {manifest_path}")
    return wall_clock


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CONFIG)
