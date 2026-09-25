"""Manifest reads that LD-2 requires be READ rather than assumed.

Two facts the check needs and must not guess:

  * ``N_disc`` — the discovery estimation-sample size. LD-2: "Read the ACTUAL
    discovery estimation-sample size N_disc from the run manifest/config (never
    assumed)." It is read from ``discovery.<regime>.dataset.n_rows``, the row
    count of the frame PC-Stable actually ran on, and asserted identical across
    every seed of the instance.
  * whether the manifests store PC **separating sets**. They do not (the
    ``discovery`` block records library, resolved args, canonical order, edges,
    adjacency and dataset identity — no sepsets), so the check covers BOTH
    candidate conditioning subsets rather than deferring to a stored one. The
    absence is DETECTED here, not assumed, so a future manifest that gains the
    field is picked up automatically.

Read-only: opens ``manifest.json`` and nothing else, writes nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from icknowledge.analysis.loading import L1D_ROOT, N_SEEDS_FULL, Cell

__all__ = ["DiscoveryManifestFacts", "SEPSET_KEYS", "read_discovery_facts"]

#: Keys any of which would carry a stored PC separating set, checked case-folded.
SEPSET_KEYS = ("sepsets", "sepset", "separating_sets", "sep_sets")


@dataclass(frozen=True)
class DiscoveryManifestFacts:
    """What the frozen manifests of one instance say about its discovery run."""

    cell: Cell
    n_seeds: int
    n_disc: int
    n_disc_field: str
    alpha: float
    indep_test: str
    sepsets_available: bool
    stored_sepset: tuple[str, ...] | None
    adjacency_absent_seeds: tuple[int, ...]

    @property
    def n_disc_uniform(self) -> bool:
        return True  # enforced at construction; kept explicit for report wording


def _find_sepset(block: dict) -> tuple[bool, tuple[str, ...] | None]:
    """Detect a stored separating set anywhere in the discovery block."""
    for key, value in block.items():
        if key.casefold() in SEPSET_KEYS:
            if isinstance(value, dict):
                for pair, sep in value.items():
                    if "A" in str(pair) and "X2" in str(pair):
                        return True, tuple(str(v) for v in sep)
            return True, None
    return False, None


def read_discovery_facts(
    cell: Cell,
    pair: tuple[str, str] = ("A", "X2"),
    root: Path = L1D_ROOT,
    n_seeds: int = N_SEEDS_FULL,
) -> DiscoveryManifestFacts:
    """Read ``N_disc``, the CI-test pins and sepset availability for one instance.

    Raises if ``N_disc`` is not identical across the instance's seeds — a
    per-seed sample-size difference would make a single Fisher-z power figure
    meaningless, so it fails loudly instead of averaging.
    """
    sizes: set[int] = set()
    alphas: set[float] = set()
    tests: set[str] = set()
    sepset_flags: set[bool] = set()
    stored_sepset: tuple[str, ...] | None = None
    absent: list[int] = []

    for seed_idx in range(n_seeds):
        path = cell.seed_dir(seed_idx, root) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — the check reads N_disc from the frozen "
                "manifests and never assumes it."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        block = manifest["discovery"][cell.regime]
        sizes.add(int(block["dataset"]["n_rows"]))
        alphas.add(float(block["resolved_args"]["alpha"]))
        tests.add(str(block["resolved_args"]["indep_test"]))
        has_sepset, sep = _find_sepset(block)
        sepset_flags.add(has_sepset)
        if sep is not None:
            stored_sepset = sep

        order = [str(n) for n in block["canonical_order"]]
        adjacency = block["adjacency"]
        i, j = order.index(pair[0]), order.index(pair[1])
        if not (adjacency[i][j] or adjacency[j][i]):
            absent.append(seed_idx)

    if len(sizes) != 1:
        raise AssertionError(
            f"N_disc is not identical across the {n_seeds} seeds of {cell.label}: "
            f"{sorted(sizes)}. A single Fisher-z power figure would be meaningless."
        )
    if len(alphas) != 1 or len(tests) != 1:
        raise AssertionError(
            f"CI-test pins differ across seeds of {cell.label}: "
            f"alpha={sorted(alphas)}, indep_test={sorted(tests)}."
        )

    return DiscoveryManifestFacts(
        cell=cell,
        n_seeds=n_seeds,
        n_disc=sizes.pop(),
        n_disc_field=f"discovery.{cell.regime}.dataset.n_rows",
        alpha=alphas.pop(),
        indep_test=tests.pop(),
        sepsets_available=any(sepset_flags),
        stored_sepset=stored_sepset,
        adjacency_absent_seeds=tuple(absent),
    )
