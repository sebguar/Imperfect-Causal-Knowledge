"""Re-emit the H3 verdict under the corrected LD-2 taxonomy.

    python -m scripts.run_h3_reemission [--root results/cross_seed_L1d_N20]

READS the frozen ``h3_per_seed.csv`` and ``h3_covariates.csv``; WRITES exactly two
NEW files into ``<root>/summary/``:

    h3_verdict_v2.md            the re-emitted verdict record
    h3_instance_table_v2.csv    the re-emitted instance table

Per LD-2: discovery, estimation, recourse, scoring, covariates and
per-seed quantities are frozen and correct — only the Step-3
classification/gate/label layer re-executes. The superseded ``h3_verdict.md`` and
``h3_instance_table.csv`` are AUDIT OBJECTS: this script asserts, by content hash
taken before and after, that neither was touched.

The re-emission is OUTCOME-EXPOSED, TEXTUALLY FORCED and LABEL-INVARIANT
(LD-2); it is never described as outcome-independent or outcome-blind.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from icknowledge.analysis import h3, h3_d2, h3_manifests, h3_v2
from icknowledge.analysis.loading import L1D_ROOT, N_SEEDS_FULL, Cell

#: The instance that receives the LD-2 truth-side check.
D2_CELL = Cell("triangle", "linear", "effect_modifying")

#: Artifacts that must be byte-identical before and after this script runs.
FROZEN = (
    "h3_verdict.md",
    "h3_instance_table.csv",
    "h3_per_seed.csv",
    "h3_covariates.csv",
)


def _hashes(summary: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((summary / name).read_bytes()).hexdigest()
        for name in FROZEN
        if (summary / name).exists()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(L1D_ROOT))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_FULL)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="also run the large-N synthetic corroboration of the closed form",
    )
    parser.add_argument("--synthetic-n", type=int, default=10_000_000)
    args = parser.parse_args(argv)
    root = Path(args.root)
    summary = root / "summary"

    before = _hashes(summary)

    # -- frozen inputs, READ ---------------------------------------------------
    per_seed = pd.read_csv(summary / "h3_per_seed.csv")
    covariates = pd.read_csv(summary / "h3_covariates.csv")
    frozen_instances = pd.read_csv(summary / "h3_instance_table.csv")
    if per_seed.empty or covariates.empty:
        raise SystemExit("frozen H3 inputs are empty — nothing to re-emit")

    # The instance/pair/N_valid tables are re-derived from the FROZEN per-seed
    # frame (pure re-aggregation, no re-run) and then asserted equal to the frozen
    # instance table on every shared numeric column: if any of them moved, the
    # re-emission would be doing more than the classification layer.
    n_valid = pd.DataFrame(
        [
            {
                "topology": t,
                "family": f,
                "regime": r,
                "denominator": h3.DENOMINATOR_LABEL,
                "N_valid": int(len(block)),
                "N_required": h3.N_VALID_REQUIRED,
                "halt_n_valid": len(block) < h3.N_VALID_REQUIRED,
            }
            for (t, f, r), block in per_seed.groupby(
                ["topology", "family", "regime"], sort=True
            )
        ]
    )
    instances = h3.instance_table(per_seed, n_valid)
    pairs = h3.interaction_pairs(per_seed, n_valid)

    shared = [
        c
        for c in frozen_instances.columns
        if c in instances.columns and c not in {"denominator"}
    ]
    left = instances.sort_values(["topology", "family", "regime"])[shared].reset_index(
        drop=True
    )
    right = frozen_instances.sort_values(["topology", "family", "regime"])[
        shared
    ].reset_index(drop=True)
    # Tolerance, not exact equality, deliberately: both sides are round-tripped
    # through CSV and differ in the last decimal digit (~1e-16). The bound stays
    # ~7 orders tighter than any real movement of a region tally, gate count or
    # mean.
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-9, atol=1e-12)

    # -- Step 2: the LD-2 truth-side check ---------------------------------
    facts = h3_manifests.read_discovery_facts(D2_CELL, root=root, n_seeds=args.n_seeds)
    d2 = h3_d2.run_d2_check(
        n_disc=facts.n_disc,
        alpha=facts.alpha,
        with_synthetic=args.synthetic,
        synthetic_n=args.synthetic_n,
        sepsets_available=facts.sepsets_available,
        stored_sepset=facts.stored_sepset,
    )

    # -- Step 3: reclassify, re-gate, re-label --------------------------------
    zones = {(D2_CELL.topology, D2_CELL.regime): d2.zone}
    classifications = h3_v2.reclassify_linear_instances(per_seed, instances, zones)
    propagation = h3_v2.standing_halt_propagation(classifications)
    supporting = h3.supporting_evidence(root)

    spot_checks = [
        "**1a — per-group dcost_ld, three EM linear cells** (frozen "
        "`per_seed_by_group.csv`, fourway costs). The negative dcost_ld is driven by "
        "the **A = −1** group on all three cells: chain −0.065922 (A=−1) vs "
        "+0.040125 (A=+1); collider −0.533621 vs +0.056312; triangle −0.377208 vs "
        "+0.126783. No flag. This check does not gate Step 3 — "
        "PS-7's registration is the authority for class (c).",
        "",
        "**1b — collider/linear/EM seed 00, discovered vs L1-oracle X3 equation.** "
        "The L1-discovered coefficients are not persisted by any artifact, so the "
        "fit was reconstructed from the manifest's own config and the manifest's "
        "stored discovered GRAPH (PC-Stable NOT re-run); the reconstruction was "
        "verified by (i) the estimation-sample `dataset_identity` sha256 matching "
        "the manifest exactly and (ii) the oracle refit reproducing the manifest's "
        "stored oracle X3 dict coefficient-for-coefficient. Result: identical "
        "structure minus the `A*X1` interaction column, pooled main coefficients "
        "(oracle X1 +0.493566 / X2 +0.474082 / A +1.056575 / A*X1 +0.310718 vs "
        "discovered X1 +0.483213 / X2 +0.464037 / A +1.120211). No structural "
        "surprise beyond the missing A·X column, so PS-7 note (b)'s byte-identity "
        "scope is not contradicted.",
    ]

    report = h3_v2.build_verdict_v2_report(
        per_seed=per_seed,
        instances=instances,
        pairs=pairs,
        n_valid=n_valid,
        classifications=classifications,
        propagation=propagation,
        supporting=supporting,
        d2=d2,
        d2_field=facts.n_disc_field,
        spot_checks=spot_checks,
    )
    table = h3_v2.instance_table_v2(instances, classifications)
    verdict_path, table_path = h3_v2.write_v2_artifacts(report, table, root=root)

    after = _hashes(summary)
    for name, digest in before.items():
        if after.get(name) != digest:
            raise SystemExit(f"FATAL: frozen artifact {name} changed — aborting")

    print(report.encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote 2 NEW artifacts -> {verdict_path.name}, {table_path.name}")
    print(f"frozen artifacts verified unchanged: {sorted(before)}")
    for item in classifications:
        if item.halt:
            print(f"  HARNESS HALT STANDS: {item.topology}/linear/{item.regime}")
        elif item.triggered:
            print(f"  halt LIFTS: {item.topology}/linear/{item.regime} ({item.breakdown})")
    print(f"  truth-side zone: {d2.zone} (r_op = {d2.r_op:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
