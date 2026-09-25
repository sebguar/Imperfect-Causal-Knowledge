"""Phase-3 provenance sweep over every manifest of the four-rung L1-discovered tree.

Read-only. Asserts nothing about RESULTS — it checks that each of the 240 cells
was produced by the commit, the pinned library, the seed and the sample the
run claims, and reports the per-cell wall-clock envelope from the manifest
timestamps (the empirical validation of the 7-9 h runtime estimate).

    python -m scripts.sweep_l1d_provenance --root results/cross_seed_L1d_N20 \
        --n-seeds 20 --expect-commit <phase-0 HEAD>

Every check is reported as a count over all cells plus the explicit list of
offenders, so a partial failure names the cells rather than just failing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from icknowledge.utils.seeding import EXPERIMENT_META_ENTROPY, experiment_run_seeds

# The four PC-Stable pins PS-7 requires every discovery record to carry. Values
# are compared exactly; a silently re-defaulted argument is the failure this
# catches.
REQUIRED_PINS = {"alpha": 0.05, "indep_test": "fisherz", "stable": True, "uc_rule": 0}
REQUIRED_CAUSAL_LEARN = "0.1.4.8"


def _cell_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and "_seed_" in p.name)


def _parse_cell(name: str) -> tuple[str, str, str, int]:
    topology, rest = name.split("_", 1)
    body, seed_idx = rest.rsplit("_seed_", 1)
    family, regime = body.split("_", 1)
    return topology, family, regime, int(seed_idx)


def sweep(root: Path, n_seeds: int, expect_commit: str | None) -> int:
    dirs = _cell_dirs(root)
    seeds = experiment_run_seeds(n_seeds)
    fail: dict[str, list[str]] = {}

    def bad(check: str, cell: str, detail: str) -> None:
        fail.setdefault(check, []).append(f"{cell}: {detail}")

    commits: dict[str, list[str]] = {}
    stamps: list[tuple[datetime, str]] = []

    for d in dirs:
        cell = d.name
        topology, family, regime, seed_idx = _parse_cell(cell)
        m = json.loads((d / "manifest.json").read_text())

        if m.get("git_dirty") is not False:
            bad("git_dirty", cell, repr(m.get("git_dirty")))
        commits.setdefault(str(m.get("git_commit")), []).append(cell)
        if expect_commit and m.get("git_commit") != expect_commit:
            bad("git_commit", cell, str(m.get("git_commit")))

        cl = (m.get("package_versions") or {}).get("causal-learn")
        if cl != REQUIRED_CAUSAL_LEARN:
            bad("causal_learn_version", cell, repr(cl))

        disc = (m.get("discovery") or {}).get(regime)
        if disc is None:
            bad("discovery_block_present", cell, "absent")
        else:
            args = disc.get("resolved_args") or {}
            for key, want in REQUIRED_PINS.items():
                if args.get(key) != want:
                    bad(f"pin:{key}", cell, f"{args.get(key)!r} != {want!r}")

        est = ((m.get("estimation") or {}).get(regime) or {}).get("metadata") or {}
        est_sha = (est.get("dataset") or {}).get("sha256")
        disc_sha = ((disc or {}).get("dataset") or {}).get("sha256")
        if est_sha is None or disc_sha is None or est_sha != disc_sha:
            bad("dataset_sha_match", cell, f"est={est_sha} disc={disc_sha}")

        seeding = m.get("seeding") or {}
        if seeding.get("meta_entropy") != int(EXPERIMENT_META_ENTROPY):
            bad("meta_entropy", cell, repr(seeding.get("meta_entropy")))
        want_seed = int(seeds[seed_idx])
        if seeding.get("this_run_seed") != want_seed:
            bad("this_run_seed", cell, f"{seeding.get('this_run_seed')} != {want_seed}")
        if seeding.get("seed_index") != seed_idx:
            bad("seed_index", cell, repr(seeding.get("seed_index")))
        if seeding.get("n_runs") != n_seeds:
            bad("n_runs", cell, repr(seeding.get("n_runs")))
        # The estimator records the master seed independently of the seeding
        # block, so agreement between the two is a real cross-check, not a
        # restatement of the same field.
        if est.get("master_seed") not in (None, want_seed):
            bad("estimation_master_seed", cell, f"{est.get('master_seed')} != {want_seed}")

        stamps.append((datetime.fromisoformat(m["timestamp"]), cell))

    print(f"=== provenance sweep: {len(dirs)} cells under {root} ===\n")
    checks = [
        "git_dirty",
        "git_commit",
        "causal_learn_version",
        "discovery_block_present",
        *[f"pin:{k}" for k in REQUIRED_PINS],
        "dataset_sha_match",
        "meta_entropy",
        "this_run_seed",
        "seed_index",
        "n_runs",
        "estimation_master_seed",
    ]
    for check in checks:
        offenders = fail.get(check, [])
        status = "PASS" if not offenders else f"FAIL ({len(offenders)})"
        print(f"  {check:<26} {len(dirs) - len(offenders):>3}/{len(dirs)}  {status}")
        for line in offenders[:10]:
            print(f"      {line}")

    print("\n  git_commit distribution:")
    for commit, cells in sorted(commits.items(), key=lambda kv: -len(kv[1])):
        print(f"      {commit}  {len(cells)} cells")

    # Per-cell wall-clock as consecutive manifest-timestamp deltas, in the order
    # the cells were actually written. The FIRST cell has no predecessor, so the
    # distribution below is over the remaining n-1 gaps; the total is the span
    # from the first to the last manifest and therefore understates the run by
    # exactly that first cell.
    stamps.sort()
    deltas = [
        (stamps[i][0] - stamps[i - 1][0]).total_seconds() for i in range(1, len(stamps))
    ]
    if deltas:
        span = (stamps[-1][0] - stamps[0][0]).total_seconds()
        print(f"\n  per-cell wall-clock from manifest timestamps (n={len(deltas)} gaps):")
        print(f"      min    {min(deltas):8.1f} s")
        print(f"      median {statistics.median(deltas):8.1f} s")
        print(f"      max    {max(deltas):8.1f} s")
        print(f"      mean   {statistics.fmean(deltas):8.1f} s")
        print(f"      span first->last manifest: {span / 3600:.2f} h  ({span:.0f} s)")
        print(f"      first manifest {stamps[0][0].isoformat()}  ({stamps[0][1]})")
        print(f"      last  manifest {stamps[-1][0].isoformat()}  ({stamps[-1][1]})")

    total_fail = sum(len(v) for v in fail.values())
    verdict = "SWEEP CLEAN" if not total_fail else f"SWEEP FAILED: {total_fail} violations"
    print(f"\n=== {verdict} ===")
    return 0 if not total_fail else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/cross_seed_L1d_N20")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--expect-commit", default=None)
    args = parser.parse_args(argv)
    return sweep(Path(args.root), args.n_seeds, args.expect_commit)


if __name__ == "__main__":
    raise SystemExit(main())
