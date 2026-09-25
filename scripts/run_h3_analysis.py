"""Run the H3 analysis over the frozen four-rung L1-discovered grid.

    python -m scripts.run_h3_analysis [--root results/cross_seed_L1d_N20]

Reads only. Writes, into ``<root>/summary/``:

    h3_covariates.csv       per (cell, seed): skeleton-SHD, orientation split, SID
    h3_per_seed.csv         per (cell, seed): dcost_ld, dcost_l0, region, t, event
    h3_instance_table.csv   per instance: region tallies, gate, covariate means
    h3_verdict.md           the mechanical PS-8 verdict record

Nothing here re-runs discovery, estimation, recourse or scoring, and no
pre-existing file under ``results/`` is modified. The verdict is whatever PS-8's
procedure returns on these numbers; this script has no branch that could change
it. Halts are real: a linear-control HARNESS HALT or an N_valid flag (PS-8)
suppresses the affected label(s) and the record says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from icknowledge.analysis import h3, h3_covariates
from icknowledge.analysis.loading import L1D_ROOT, N_SEEDS_FULL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(L1D_ROOT))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_FULL)
    args = parser.parse_args(argv)
    root = Path(args.root)

    covariates = h3_covariates.build_covariate_frame(root=root, n_seeds=args.n_seeds)
    costs = h3.load_realized_costs(root=root, n_seeds=args.n_seeds)
    n_valid = h3.n_valid_table(costs)
    per_seed = h3.per_seed_quantities(costs, covariates)
    instances = h3.instance_table(per_seed, n_valid)
    pairs = h3.interaction_pairs(per_seed, n_valid)
    diagnostics = h3.linear_diagnostics(per_seed, instances)
    propagation = h3.halt_propagation(diagnostics)
    supporting = h3.supporting_evidence(root)

    report = h3.build_verdict_report(
        per_seed, instances, pairs, n_valid, diagnostics, propagation, supporting
    )

    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(summary / "h3_covariates.csv", index=False, encoding="utf-8")
    per_seed.to_csv(summary / "h3_per_seed.csv", index=False, encoding="utf-8")
    instances.to_csv(
        summary / "h3_instance_table.csv", index=False, encoding="utf-8"
    )
    (summary / "h3_verdict.md").write_text(report, encoding="utf-8")

    # ASCII-safe console echo: this stdout is a cp1252 Windows console, which
    # cannot encode Δ / ₃ and would raise AFTER the artifacts are written.
    print(report.encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote 4 artifacts -> {summary}")
    for diagnostic in diagnostics:
        if diagnostic.halt:
            print(
                f"  HARNESS HALT: {diagnostic.topology}/linear/{diagnostic.regime}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
