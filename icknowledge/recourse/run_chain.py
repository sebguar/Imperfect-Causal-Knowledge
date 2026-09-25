"""CLI entry point: run the recourse pipeline on the CHAIN.

    python -m icknowledge.recourse.run_chain [config_path]

Runs the additive and effect-modifying chain regimes under the full ladder
L0 + L1-oracle + L2 via `pipeline.run_regime`. Prints the per-regime summary,
writes one per-individual scoring CSV per regime × condition, rolls each cell up
into the aggregate CSVs with ``topology="chain"`` (a free-string cell key on
`CellKeys`), and writes the run manifest.

The grid driver (`scripts/run_cross_seed_grid.py`) reads `_FAMILIES` and
`_CONDITIONS` off the runner module for the topology it is running, so a
topology reaches the cross-seed grid by EXISTING as a module with those two
names — not by any hardcoded enumeration in the driver.

FUNCTIONAL FAMILY is read from the config's top-level ``family`` key and selects
the SCM builder, the supplied form template, the L1-oracle estimator and the
aggregate CSVs' ``family`` cell key:

    configs/recourse_chain.yaml      family: linear
    configs/recourse_chain_nlg.yaml  family: nlg     (tanh mechanism)

NO G1 ANCHOR OFF THE LINEAR FAMILY: `pipeline.run_regime` refuses
``run_anchor=True`` outside it. On linear, the check evaluates all 2^k − 1
non-empty acted subsets, so it is correct for the chain's 3 actionable variables.
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
    chain_form_spec,
    make_linear_chain,
    make_nonlinear_gaussian_chain,
    nlg_chain_form_spec,
)
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest
from icknowledge.utils.seeding import seeding_provenance

_DEFAULT_CONFIG = "configs/recourse_chain.yaml"
_REGIMES = ("additive", "effect_modifying")

#: family -> (SCM builder, supplied form template). The estimator paired with each
#: template is selected inside pipeline.run_regime from the same family key
#: (the L1-oracle form-template-known semantics/the NLG family specification), so template and
#: estimator cannot be mismatched here. Mirrors
#: run_collider._FAMILIES exactly — one plumbing pattern across all three topologies.
_FAMILIES = {
    "linear": (make_linear_chain, chain_form_spec),
    "nlg": (make_nonlinear_gaussian_chain, nlg_chain_form_spec),
}
# [form-template-known] The L1-oracle rung is fitted by unregularized OLS on the classifier's
# TRAINING SAMPLE (no redraw, no held-out split, mean function only) under the
# SUPPLIED chain form template (`chain_form_spec` / `nlg_chain_form_spec`, the
# chain SCM specification: X₂'s A·X₁ interaction column present in the effect-modifying regime,
# hard-zero by OMISSION in the additive control; X₃ main-effects-only in BOTH
# regimes because γ sits on the upstream edge).
# L1-discovered is NOT in this tuple: the four-rung tree is produced by the grid
# driver with ``--l1-discovered`` (PS-4), so the chain carries the same three rungs
# as every other cell here.
_CONDITIONS = ("L0", "L1-oracle", "L2")


def _format_instrumentation(
    by_cell_paths: dict[str, Path],
    summaries: dict[str, pd.DataFrame],
    wall_clock: dict[str, float],
) -> str:
    """Δ_cost, validity-rate-by-group and wall-clock per (regime × condition).

    # INSTRUMENTATION ONLY — no assertion, no threshold, no gate. [PS-3
    # extension (orientation lock): Δ_cost := mean(cost | A=−1) − mean(cost | A=+1),
    # positive = A=−1 more burdened.] These are single-seed H1 pre-numbers; judging
    # them would be a register decision, not a code choice. In particular the L2
    # Δ_cost printed here FEEDS the PS-2 detectability gate — that gate is evaluated
    # at the PS-1 first-6-seed early look, NOT here, and this driver neither judges
    # nor gates on the value. Δ_cost is read BACK from the by-cell CSV rather than
    # recomputed, so the printed number is the one the artifact carries.
    """
    # ASCII-only in the PRINTED text (the CSV column keeps its Δ_cost name): this
    # driver's stdout goes to a cp1252 Windows console, which cannot encode Δ and
    # would raise UnicodeEncodeError after the CSVs are written but BEFORE the
    # manifest, leaving a run half-recorded.
    lines = ["=== Delta_cost by regime x condition (INSTRUMENTATION - no gate) ==="]
    for regime, path in by_cell_paths.items():
        frame = pd.read_csv(path, float_precision="round_trip")
        for condition in _CONDITIONS:
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
    lines.append("=== wall-clock per regime (chain resolution-pin input) ===")
    for regime, seconds in wall_clock.items():
        lines.append(f"  {regime:<17} {seconds:7.1f} s  (full ladder, full pool)")
    return "\n".join(lines)


def main(
    config_path: str = _DEFAULT_CONFIG,
    conditions: tuple[str, ...] = _CONDITIONS,
) -> dict[str, float]:
    """Run the chain ladder for the config's family; return per-regime seconds.

    ``conditions`` is exposed so the grid-resolution timing spike can run the L0+L2 subset
    without paying for the L1-oracle rung it does not measure.
    """
    cfg = load_config(config_path)
    family = str(cfg.get("family", "linear"))
    if family not in _FAMILIES:
        raise ValueError(
            f"unknown functional family {family!r} in {config_path}; "
            f"wired: {sorted(_FAMILIES)}."
        )
    scm_builder, form_spec_fn = _FAMILIES[family]
    # anchor.py enumerates all 2^k − 1 subsets, so it is correct for k=3 — but only
    # on the LINEAR family (the ℓ₂ intervention-cost convention).
    run_anchor = family == "linear"

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimation_records: dict[str, dict] = {}
    aggregation_records: dict[str, dict] = {}
    # [PS-4] Populated only on the regimes whose run actually scored
    # L1-discovered — i.e. only when the caller passed a four-rung ``conditions``
    # override. This driver is the one single-cell entry point that accepts such an
    # override, so it is the one that can
    # produce a discovery record; run_triangle/run_collider run their frozen
    # three-rung tuples and stay untouched.
    discovery_records: dict[str, dict] = {}
    by_cell_paths: dict[str, Path] = {}
    summaries: dict[str, pd.DataFrame] = {}
    wall_clock: dict[str, float] = {}
    for regime in _REGIMES:
        t0 = time.perf_counter()
        run = run_regime(
            cfg,
            regime,
            conditions=conditions,
            scm_builder=scm_builder,
            form_spec_fn=form_spec_fn,
            run_anchor=run_anchor,
            family=family,
        )
        wall_clock[regime] = time.perf_counter() - t0
        summaries[regime] = run.summary
        print(format_regime_summary(run))
        # [the classifier construction] eps_scale is auto-calibrated per cell so trained-h test
        # accuracy lands in the band; the chain's value WILL differ from the
        # collider's and the NLG chain's from the linear chain's, which is correct —
        # The classifier construction holds QUALITY constant, not the noise scale.
        spec = run.classifier.label_spec
        print(
            f"calibrated eps_scale = {spec.eps_scale:.4f} (the classifier construction), "
            f"tau = {spec.tau:.4f}, test accuracy = {run.classifier.accuracy_overall:.4f}"
        )
        print(f"regime wall-clock ({'+'.join(conditions)}, full pool): {wall_clock[regime]:.1f} s")
        print()

        for condition in conditions:
            sub = run.table[run.table["condition"] == condition]
            table_path = out_dir / f"scoring_table_{regime}_{condition}.csv"
            sub.to_csv(table_path, index=False)
            print(f"wrote per-individual scoring table -> {table_path}")
        print()

        # [the L1-oracle form-template-known semantics] Estimation provenance per regime: estimator
        # id, serialized
        # FormSpec, fitted coefficients, and the dataset identity (= the classifier's
        # training sample) — the auditor's evidence that the scored L1-oracle rung
        # was fitted on the sample the L1-oracle form-template-known semantics requires.
        estimation_records[regime] = estimation_provenance(run.fitted)
        if run.discovery is not None:
            discovery_records[regime] = run.discovery.as_dict()

        # Roll this cell up with topology="chain" — NO aggregation-layer edits. The
        # layer is topology- and family-agnostic (`topology` and
        # `family` are free-string cell keys on `CellKeys`), so a third topology
        # lands by passing a new string, exactly as the SCM specification's collider did.
        cell = CellKeys(
            topology="chain", family=family, regime=regime, seed=int(cfg.seed)
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

    seeding_record = seeding_provenance(int(cfg.seed))
    manifest_path = write_run_manifest(
        out_dir,
        cfg,
        estimation=estimation_records,
        aggregation=aggregation_records,
        seeding=seeding_record,
        # None (not {}) on a three-rung run, so the key is OMITTED and the manifest
        # is byte-identical to the sealed three-rung output (PS-4).
        discovery=discovery_records or None,
    )
    print(f"wrote run manifest -> {manifest_path}")
    return wall_clock


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CONFIG)
