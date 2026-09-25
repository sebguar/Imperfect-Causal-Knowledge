"""PS-1 prefix-property verification: cross_seed_N20 seeds 0-5 vs cross_seed_N6.

    python -m scripts.verify_grid_prefix                  # PS-1: N6 vs N20, all files

    # [PS-4 note (2)] the four-rung tree's STRICT-SUBSET assertion:
    python -m scripts.verify_grid_prefix --ref results/cross_seed_N20 \\
        --new results/cross_seed_L1d_N20 --n-seeds 20 \\
        --all-topologies --kinds scoring-only

PS-1 stages the seed count 6 -> 20 by appending: ``spawn(6)`` is a prefix of
``spawn(20)``. That claim is checked here against the artifacts rather than
trusted — the seeds being equal is necessary but not sufficient, since any drift
in the builders, the classifier path, the recourse grid or the aggregation layer
would move the numbers while leaving the seeds untouched.

Compares, for the 8 triangle+collider cells the N=6 grid contains, at seeds 0-5:
3 per-condition scoring tables + 3 aggregate CSVs (by_group, by_cell, pairwise)
= 8 x 6 x 6 = 288 comparisons, all float-exact after a round-trip CSV parse. The
chain has no counterpart in cross_seed_N6/ and is not compared.

Exit code 0 iff every comparison is exact; a mismatch prints the first differing
cell / seed / column with both values and stops. cross_seed_N6/ is never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from icknowledge.analysis.loading import (
    CONDITIONS,
    DEFAULT_ROOT,
    N20_ROOT,
    N_SEEDS,
    TOPOLOGIES,
    grid_cells,
)

#: The topologies present in the N=6 grid. The chain (the chain SCM specification)
#: has no counterpart there, so it is out of scope for a PREFIX check.
PREFIX_TOPOLOGIES = ("triangle", "collider")

AGGREGATES = ("by_group", "by_cell", "pairwise")

#: [PS-4 note (2)] The two comparison scopes.
#:
#: ``"all"`` — scoring tables AND aggregates (the PS-1 scope).
#: ``"scoring-only"`` — the three shared per-condition scoring tables alone (the
#:     PS-4 STRICT-SUBSET assertion). Aggregates are excluded BY DESIGN, not by
#:     omission: the four-rung tree's aggregates carry an L1-discovered row and
#:     _fourway denominators (schema.common_found_suffix), so demanding
#:     byte-identity there would assert something false.
KINDS = ("all", "scoring-only")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def first_difference(ref: pd.DataFrame, new: pd.DataFrame) -> str:
    """Locate the first differing column/row, for a report that names the number."""
    if list(ref.columns) != list(new.columns):
        return f"column sets differ: ref={list(ref.columns)} new={list(new.columns)}"
    if len(ref) != len(new):
        return f"row counts differ: ref={len(ref)} new={len(new)}"
    for column in ref.columns:
        left, right = ref[column], new[column]
        unequal = ~((left == right) | (left.isna() & right.isna()))
        if unequal.any():
            i = int(unequal.to_numpy().nonzero()[0][0])
            return (
                f"column {column!r} row {i}: ref={left.iloc[i]!r} new={right.iloc[i]!r}"
            )
    return "frames differ but no per-column difference located (dtype mismatch?)"


def compare(
    ref_root: Path,
    new_root: Path,
    n_seeds: int = N_SEEDS,
    *,
    topologies: tuple[str, ...] = PREFIX_TOPOLOGIES,
    kinds: str = "all",
) -> int:
    """Float-exact compare two grid trees; return 0 iff every comparison is exact.

    ``topologies`` selects which cells are enumerated — the N=6 pair by default
    (the PS-1 prefix check has no chain counterpart), all three for the PS-4
    strict-subset check against the four-rung tree.

    ``kinds`` selects WHAT is compared, per PS-4 note (2): ``"all"`` for scoring
    tables plus aggregates (PS-1), ``"scoring-only"`` for the three shared
    per-condition scoring tables alone (PS-4 — aggregates diverge by design; see
    the KINDS constant).
    """
    if kinds not in KINDS:
        raise ValueError(f"unknown kinds={kinds!r}; wired: {list(KINDS)}.")
    exact = total = 0
    for cell in grid_cells(topologies):
        for seed_idx in range(n_seeds):
            ref_dir = cell.seed_dir(seed_idx, ref_root)
            new_dir = cell.seed_dir(seed_idx, new_root)
            # CONDITIONS is the THREE-rung tuple, and that is correct in both
            # modes: even against a four-rung tree the shared conditions are these
            # three, and L1-discovered has no counterpart in the reference tree to
            # compare against.
            names = [
                f"scoring_table_{cell.regime}_{condition}.csv" for condition in CONDITIONS
            ]
            if kinds == "all":
                names += [f"aggregate_{kind}_{cell.regime}.csv" for kind in AGGREGATES]
            for name in names:
                total += 1
                ref_path, new_path = ref_dir / name, new_dir / name
                if not ref_path.exists() or not new_path.exists():
                    missing = ref_path if not ref_path.exists() else new_path
                    print(f"MISSING: {missing}", file=sys.stderr)
                    return 1
                ref, new = _read(ref_path), _read(new_path)
                try:
                    assert_frame_equal(ref, new, check_exact=True)
                except AssertionError:
                    print("=== PREFIX-PROPERTY VIOLATION ===", file=sys.stderr)
                    print(f"cell={cell.key}  seed_idx={seed_idx}  file={name}", file=sys.stderr)
                    print(first_difference(ref, new), file=sys.stderr)
                    print(
                        "\nThe compared tree is NOT a bit-identity continuation of "
                        "the reference. The reference tree has not been modified. Do "
                        "not consume further seeds until this is resolved.",
                        file=sys.stderr,
                    )
                    return 1
                exact += 1
    print(f"prefix property VERIFIED: {exact}/{total} comparisons float-exact")
    per_cell = (
        f"{len(CONDITIONS)} scoring tables"
        if kinds == "scoring-only"
        else f"{len(CONDITIONS)} scoring tables + {len(AGGREGATES)} aggregates"
    )
    print(f"  {len(grid_cells(topologies))} cells x {n_seeds} seeds x ({per_cell})")
    print(f"  topologies={list(topologies)}  kinds={kinds!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=str(DEFAULT_ROOT))
    parser.add_argument("--new", default=str(N20_ROOT))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--kinds",
        choices=KINDS,
        default="all",
        help=(
            "'all' (default, the PS-1 check) compares scoring tables AND "
            "aggregates; 'scoring-only' compares the three shared per-condition "
            "scoring tables alone — the PS-4 note (2) strict-subset "
            "assertion against the four-rung tree, whose aggregates diverge by "
            "design."
        ),
    )
    parser.add_argument(
        "--all-topologies",
        action="store_true",
        help=(
            "enumerate all three topologies instead of the N=6 pair. Required "
            "for the PS-4 check (the chain joined at the chain SCM specification and exists in "
            "both the "
            "three- and four-rung N=20 trees); wrong for the PS-1 check, whose "
            "reference tree cross_seed_N6/ has no chain cells."
        ),
    )
    args = parser.parse_args(argv)
    return compare(
        Path(args.ref),
        Path(args.new),
        args.n_seeds,
        topologies=TOPOLOGIES if args.all_topologies else PREFIX_TOPOLOGIES,
        kinds=args.kinds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
