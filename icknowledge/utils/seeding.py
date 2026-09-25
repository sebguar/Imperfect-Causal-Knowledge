"""
Reproducible randomness utilities.

Design rationale
----------------
All stochastic components (SCM sampler, classifier, recourse generator,
abduction noise) receive independent RNG streams derived from a single
master seed via NumPy's SeedSequence hierarchy. This ensures:

1. Full reproducibility: identical master seed → identical experiment.
2. Stream independence: modifying one component does not shift the random
   draws of any other component — a failure mode of a single shared global
   seed (np.random.seed) when code paths change.
3. Clean comparison: the additive and effect-modifying SCM conditions can
   be given matched noise realizations, isolating structural differences
   as the sole driver of any observed burden asymmetry.

New code should use spawn_rng / spawn_children.
set_global_seeds exists only to make legacy library calls reproducible.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np


def set_global_seeds(seed: int) -> None:
    warnings.warn(
        "PYTHONHASHSEED only affects hashing in new subprocesses, not the "
        "current running process.",
        stacklevel=2,
    )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG; for reproducibility of any code using it


def spawn_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

# SeedSequence.spawn() guarantees statistically independent streams
def spawn_children(parent_seed: int, n: int) -> list[np.random.Generator]:
    seed_sequences = np.random.SeedSequence(parent_seed).spawn(n)
    return [np.random.default_rng(seq) for seq in seed_sequences]


# --------------------------------------------------------------------------- #
# Cross-seed replication convention (the seed-generation mechanics)
# --------------------------------------------------------------------------- #
# The whole experiment grid draws its per-run master seeds from ONE pinned
# meta-entropy via SeedSequence(meta_entropy).spawn(N). Pinning the meta-entropy
# — not each run seed individually — makes the entire replication grid
# reproducible from a single audited integer and lets the seed count N grow
# without renumbering or reshuffling the seeds already used (spawn appends).
EXPERIMENT_META_ENTROPY: int = 20260710
# ^ the same integer used as the pilot master seed, reused here as the
#   grid meta-seed so the entire project traces to ONE pinned value. The pilot
#   cells used it directly as a run master seed; the replication
#   grid uses the spawned per-run seeds produced below. Both trace to this int.


def experiment_run_seeds(n: int, meta_entropy: int = EXPERIMENT_META_ENTROPY) -> list[int]:
    """The N per-run master seeds for the replication grid (the seed-generation mechanics).

    Deterministic: ``SeedSequence(meta_entropy).spawn(n)`` yields N statistically
    independent child sequences; each is reduced to a uint32 int usable as a
    downstream master seed (fed to ``spawn_children`` for that run's component
    streams). Growing N only appends new seeds — the first k of ``spawn(N)`` are
    identical to ``spawn(k)`` for any k ≤ N — so extending the grid never
    perturbs an existing run's seed.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    children = np.random.SeedSequence(meta_entropy).spawn(n)
    return [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in children]


def seeding_provenance(
    this_run_seed: int,
    n_runs: int | None = None,
    meta_entropy: int = EXPERIMENT_META_ENTROPY,
) -> dict:
    """Manifest record for the seeding convention (the seed-generation mechanics).

    Always records the pinned ``meta_entropy`` and this run's master seed, so
    every run's manifest carries the meta-seed. When ``n_runs`` is given (a
    replication-grid run), also materializes the full spawned per-run seed list
    inline, so an auditor sees every grid seed without re-running the spawn.
    """
    record: dict = {
        "meta_entropy": int(meta_entropy),
        "this_run_seed": int(this_run_seed),
    }
    if n_runs is not None:
        record["n_runs"] = int(n_runs)
        record["run_seeds"] = experiment_run_seeds(n_runs, meta_entropy)
    return record
