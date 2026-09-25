"""Recourse cost function(s).

The cost is charged in INTERVENTION space on the ACTED variables only, in RAW
feature units (no scaling; the raw-units decision). One fixed norm is used everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# ℓ₂, fixed across every cell/condition. Anchor: Ehyaei runs experiments in
# L2 ("cost(v,a) = ||v - CF(v,a)||_2"); ℓ₂ is self-dual (p*=2) so the closed form is
# ordinary Euclidean geometry. Also Von Kügelgen Def 3.1 (Δ_cost as intervention cost).
# Also Karimi Prop. 4.1: cost is the norm of the ACTION over acted variables, not
# the factual-to-counterfactual displacement — the two coincide only under feature
# independence, which is why the convention is fixed once here and never per-cell.
L_P = 2


def intervention_cost(delta: Mapping[str, float], acted: Sequence[str]) -> float:
    """ℓ₂ norm of the intervention delta on the ACTED variables only.

        cost(factual, intervention) = || delta_S ||_2        (S = ``acted``)

    # Standard parameterization — the delta is measured relative
    # to the individual's FACTUAL value; the enacted move is the hard intervention
    # do(X_i := factual_i + delta_i). Cost therefore depends on the delta alone.
    #
    # Cost is charged on ACTED variables only, NOT on total downstream point
    # movement — the intervention-space convention. The G1 anchor (anchor.py)
    # reconciles it with the point-distance convention by pulling that closed
    # form back through the SCM into intervention space.

    Parameters
    ----------
    delta:
        Maps acted variable -> intervention delta (relative to the factual value).
        Entries for non-acted variables, if present, are ignored.
    acted:
        The acted/intervened set S. Only these coordinates are charged. An empty
        acted set (no-op) costs 0.
    """
    if not acted:
        return 0.0
    delta_S = np.array([float(delta.get(var, 0.0)) for var in acted], dtype=float)
    return float(np.linalg.norm(delta_S, ord=L_P))


def intervention_cost_batch(deltas: np.ndarray) -> np.ndarray:
    """Vectorized ℓ₂ cost over a batch of deltas on the acted variables.

    ``deltas`` has shape (K, |S|): row k holds one candidate's delta over the acted
    variables (in a fixed acted-variable column order). Returns the length-K vector
    of ℓ₂ intervention costs. Semantics identical to `intervention_cost`;
    used by the grid search where all candidates share the acted-variable ordering.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be 2-D (K, |S|), got shape {deltas.shape}.")
    return np.linalg.norm(deltas, ord=L_P, axis=1)
