"""Run the H4 analysis over the frozen N=20 grid.

    python -m scripts.run_h4_analysis [--root results/cross_seed_N20]

Reads only. Writes, into ``<root>/summary/``:

    h4_per_seed_ddb.csv     ΔΔB per (cell, seed, rung)
    h4_ddb_by_cell.csv      per (cell, rung): mean/SD, seed sign counts, gate
    h4_regime_contrast.csv  |ΔΔB| treatment vs control, per topology × family
    h4_verdict.md           the mechanical PS-9 verdict report

The verdict is whatever PS-9's procedure returns on these numbers. This script has
no branch that could change it, and nothing downstream of it re-reads the grid.

Once ``h4_verdict.md`` exists it is a FROZEN AUDIT OBJECT and this script REFUSES
to run at all (see ``_refuse_if_frozen``): **PS-7** / **PS-4** isolate the
H4 verdict from every L1-discovered development, and an isolation that depends on
nobody happening to re-run the driver is not an isolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from icknowledge.analysis import h4
from icknowledge.analysis.loading import N20_ROOT, N_SEEDS_FULL

#: The frozen H4 verdict. Named here rather than inline so the guard and the write
#: below cannot drift apart onto two different filenames.
FROZEN_VERDICT = "h4_verdict.md"


def _refuse_if_frozen(summary: Path) -> None:
    """Refuse the whole run if the frozen H4 verdict already exists.

    Checked BEFORE the first write rather than immediately before the verdict
    write: the four artifacts are one run's output, and a re-run that replaced the
    three CSVs while sparing the verdict would leave the summary internally
    inconsistent — worse than not running. So this raises with nothing written.

    Refuse-only, with no ``--force``: H4 has no re-emission need (unlike H3's v2
    layer, which writes distinct ``_v2`` names precisely so the superseded audit
    objects are never a write target). A genuine re-emission is an operator
    decision that must leave a trace in the working tree — remove or rename the
    artifact deliberately first.
    """
    path = summary / FROZEN_VERDICT
    if path.exists():
        raise SystemExit(
            f"FATAL: {path} already exists and is a FROZEN audit object "
            "(PS-7/PS-4 — the H4 verdict is isolated from all L1-discovered "
            "work). Nothing was written. If a genuine re-emission is intended, "
            "remove or rename the frozen artifact deliberately first."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(N20_ROOT))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_FULL)
    args = parser.parse_args(argv)
    root = Path(args.root)
    summary = root / "summary"
    _refuse_if_frozen(summary)

    per_seed = h4.load_per_seed_ddb(root, args.n_seeds)
    delta_s = h4.load_delta_s(root)
    h4.assert_delta_s_preconditions(delta_s)

    cells = h4.by_cell(per_seed, delta_s)
    contrast = h4.regime_contrast(cells)
    correlation = h4.supporting_correlation(cells)
    verdicts = {c: h4.verdict_for_rung(cells, contrast, c) for c in h4.RUNGS}
    no_op = h4.common_found_is_no_op(per_seed, root)
    bimodal = h4.bimodality_flags(per_seed)

    summary.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(summary / "h4_per_seed_ddb.csv", index=False)
    cells.to_csv(summary / "h4_ddb_by_cell.csv", index=False)
    contrast.to_csv(summary / "h4_regime_contrast.csv", index=False)
    report = h4.build_verdict_report(
        cells, contrast, verdicts, correlation, no_op, bimodal
    )
    (summary / FROZEN_VERDICT).write_text(report, encoding="utf-8")

    # ASCII-safe console echo: this stdout is a cp1252 Windows console, which cannot
    # encode Δ and would raise UnicodeEncodeError AFTER the artifacts are written.
    print(report.encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote 4 artifacts -> {summary}")
    for condition in h4.RUNGS:
        print(f"  {condition:<10} VERDICT: {verdicts[condition].label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
