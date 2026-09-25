"""Actionable set + per-axis intervention grid.

Defines WHICH variables recourse may act on and the discrete grid of deltas the
single brute-force procedure sweeps. Grid points are DELTAS (relative to each
individual's factual value); the enacted absolute intervention is
``factual + delta`` (applied by the causal models in model_conditions.py).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

# Immutability lives in the action space, NOT the SCM; consistent with
# the group-blind classifier. A remains a node in the SCM for
# group-conditional propagation, but is never a recourse target (root, immutable).
_IMMUTABLE = frozenset({"A"})


@dataclass(frozen=True)
class ActionSpace:
    """Actionable variables and their per-axis delta grids.

    ``acted`` is the fixed column order used everywhere downstream (candidate
    matrix columns, cost, model predict). ``axis_grids[var]`` is that axis's 1-D
    array of deltas, guaranteed to contain the no-change value 0.
    """

    acted: tuple[str, ...]
    axis_grids: dict[str, np.ndarray]

    @property
    def n_candidates(self) -> int:
        """Size of the full joint grid (Cartesian product of the axes)."""
        size = 1
        for var in self.acted:
            size *= len(self.axis_grids[var])
        return size

    def candidate_matrix(self) -> np.ndarray:
        """(K, |acted|) matrix of all joint delta candidates, columns in ``acted`` order.

        # Joint grid incl. no-change per axis covers intervention subsets
        # automatically — a 0 in an axis column means "do not act on that variable",
        # so single-variable, joint, and the no-op (all-zero row) interventions are
        # all present as rows of this one matrix.
        """
        axes = [self.axis_grids[var] for var in self.acted]
        return np.array(list(product(*axes)), dtype=float).reshape(-1, len(self.acted))

    def iter_candidates(self) -> Iterator[dict[str, float]]:
        """Iterate joint delta candidates as {acted_var: delta} dicts."""
        for row in self.candidate_matrix():
            yield {var: float(row[i]) for i, var in enumerate(self.acted)}


def _axis_half_width(values: np.ndarray, mode: str, k: float) -> float:
    """Per-axis half-width of the symmetric delta grid, from the empirical column.

    # Precision/range knob (validated by G1). Two documented span modes, both
    # symmetric around the no-change delta 0:
    #   sd    : half_width = k · SD(feature)        (factual ±k·SD reach, default)
    #   range : half_width = k · 0.5·(max − min)    (empirical [min,max] reach)
    """
    if mode == "sd":
        return float(k * np.std(values))
    if mode == "range":
        return float(k * 0.5 * (np.max(values) - np.min(values)))
    raise ValueError(f"grid mode must be 'sd' or 'range', got {mode!r}.")


def build_action_space(
    data: pd.DataFrame,
    actionable: Sequence[str],
    *,
    mode: str = "sd",
    k: float = 4.0,
    resolution: int = 81,
) -> ActionSpace:
    """Build the actionable set and its per-axis delta grid from the dataset.

    Parameters
    ----------
    data:
        The sampled dataset (raw feature units). Column empirical statistics
        set each axis's span.
    actionable:
        Requested actionable variables. Any immutable variable (A / Gender) present
        here is rejected — immutability is enforced HERE, not in the SCM.
    mode, k:
        Span of the symmetric delta grid per axis (see `_axis_half_width`).
    resolution:
        Points per axis BEFORE forcing 0 in. This is the precision knob, tuned
        against the anchor check (anchor.py).
        # Brute-force must return the (approx-)optimal action so cross-condition
        # differences reflect the causal model, not the optimizer.
    """
    actionable = list(actionable)
    # A (Gender) EXCLUDED — immutable. Guard so it can never enter a delta.
    illegal = [var for var in actionable if var in _IMMUTABLE]
    if illegal:
        raise ValueError(
            f"immutable variable(s) {illegal} cannot be actionable (A is a "
            "root, immutable). Immutability is enforced in the action "
            "space, not the SCM."
        )
    missing = [var for var in actionable if var not in data.columns]
    if missing:
        raise ValueError(f"actionable variable(s) not in data columns: {missing}.")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}.")

    axis_grids: dict[str, np.ndarray] = {}
    for var in actionable:
        half = _axis_half_width(data[var].to_numpy(), mode, k)
        grid = np.linspace(-half, half, resolution)
        # each axis grid MUST include the no-change value 0 (single-variable, joint
        # and no-op interventions must all be reachable); force it in explicitly so
        # the guarantee does not depend on resolution parity.
        grid = np.unique(np.concatenate([grid, [0.0]]))
        axis_grids[var] = grid

    return ActionSpace(acted=tuple(actionable), axis_grids=axis_grids)
