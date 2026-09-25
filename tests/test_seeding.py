"""Tests for icknowledge.utils.seeding."""

import numpy as np
import pytest

from icknowledge.utils.seeding import (
    EXPERIMENT_META_ENTROPY,
    experiment_run_seeds,
    seeding_provenance,
    spawn_children,
    spawn_rng,
)


def test_spawn_rng_is_deterministic():
    a = spawn_rng(42).standard_normal(100)
    b = spawn_rng(42).standard_normal(100)
    assert np.array_equal(a, b)


def test_spawn_rng_different_seeds_diverge():
    a = spawn_rng(42).standard_normal(100)
    b = spawn_rng(43).standard_normal(100)
    assert not np.array_equal(a, b)


def test_spawn_children_returns_distinct_generators():
    children = spawn_children(42, 4)
    assert len(children) == 4

    draws = [child.standard_normal(50) for child in children]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])


def test_spawn_children_is_deterministic():
    first = [child.standard_normal(50) for child in spawn_children(42, 4)]
    second = [child.standard_normal(50) for child in spawn_children(42, 4)]
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Cross-seed replication convention (the seed-generation mechanics)
# --------------------------------------------------------------------------- #


def test_experiment_run_seeds_deterministic_and_distinct():
    a = experiment_run_seeds(5)
    b = experiment_run_seeds(5)
    assert a == b  # deterministic from the pinned meta-entropy
    assert len(set(a)) == 5  # statistically independent -> distinct ints
    assert all(isinstance(s, int) for s in a)


def test_experiment_run_seeds_grow_by_append():
    # The seed-generation mechanics: growing N must only APPEND — extending the grid never perturbs
    # an
    # existing run's seed (spawn(k) is a prefix of spawn(N) for k <= N).
    assert experiment_run_seeds(3) == experiment_run_seeds(7)[:3]


def test_experiment_run_seeds_rejects_nonpositive():
    with pytest.raises(ValueError, match="n must be"):
        experiment_run_seeds(0)


def test_seeding_provenance_always_carries_meta_entropy():
    record = seeding_provenance(this_run_seed=20260710)
    assert record["meta_entropy"] == EXPERIMENT_META_ENTROPY
    assert record["this_run_seed"] == 20260710
    assert "run_seeds" not in record  # single-run pilot: seeds not materialized


def test_seeding_provenance_materializes_grid_seeds():
    record = seeding_provenance(this_run_seed=20260710, n_runs=5)
    assert record["meta_entropy"] == EXPERIMENT_META_ENTROPY
    assert record["n_runs"] == 5
    assert record["run_seeds"] == experiment_run_seeds(5)
