"""Label generation for classifier training.

Builds a thresholded training target Y from the DESCENDANT features only. The
SCM stays feature-only; labeling is a separate layer on top of sampled data.

    Y = 1{ beta · (descendant features) + eps > tau }

    # The thresholded label depends on the descendant features only, never on
    # A (Gender) directly. Y is a TRAINING target only — recourse targets the
    # classifier h, never Y. Anchor: the Karimi label-generation scheme,
    # adapted to threshold form as in Ehyaei.

Pieces of the label spec:
- beta:      fixed, declared weight vector over the descendant features. Documented
             in config, NEVER fitted.
- eps:       per-sample label noise drawn from its OWN child-RNG stream (NOT the SCM
             feature noise). Represented here as a fixed standardized base draw z
             scaled by ``eps_scale``; ``eps = eps_scale * z``. The scale is
             auto-calibrated downstream (training.py) so trained-h accuracy lands in
             the target band — non-trivial but not ~perfect.
- tau:       the (1 - base_rate_target) quantile of the label signal (beta·features
             + eps), which by construction realizes the target GLOBAL base rate.
             # Balanced base rate. Anchor: the Karimi balanced-dataset design.

The realized (eps_scale, tau) are recorded on the LabelSpec so the instance's
labels are fully reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    """Fully reproducible specification of an instance's training labels.

    Records the fixed declared weights and the two calibrated scalars
    (``eps_scale``, ``tau``) so the labels can be regenerated exactly.
    """

    feature_names: list[str]  # descendant features the label depends on (A excluded)
    beta: dict[str, float]  # fixed declared weights over feature_names (NOT fitted)
    eps_scale: float  # calibrated label-noise scale (eps = eps_scale * z)
    tau: float  # calibrated threshold hitting the target base rate
    base_rate_target: float
    realized_base_rate: float
    beta_vector: np.ndarray = field(repr=False)  # beta aligned to feature_names order


def compute_signal(features: pd.DataFrame, beta: Mapping[str, float]) -> np.ndarray:
    """Linear label signal beta · (descendant features).

    ``features`` must contain exactly the descendant-feature columns (A already
    excluded by the caller). ``beta`` maps each feature name to its fixed declared
    weight; keys must match the feature columns.
    """
    feature_names = list(features.columns)
    missing = set(feature_names) - set(beta)
    extra = set(beta) - set(feature_names)
    if missing or extra:
        raise ValueError(
            "beta keys must match the descendant-feature columns; "
            f"missing weights for {sorted(missing)}, extra weights for {sorted(extra)}."
        )
    beta_vector = np.array([float(beta[name]) for name in feature_names])
    return features.to_numpy() @ beta_vector


def make_labels(
    signal: np.ndarray, z: np.ndarray, eps_scale: float, base_rate_target: float
) -> tuple[np.ndarray, float]:
    """Threshold the noisy signal into binary labels, returning (y, tau).

    ``eps = eps_scale * z`` is the label noise (z is the fixed standardized base
    draw from the label-noise stream). ``tau`` is the (1 - base_rate_target)
    quantile of ``signal + eps``, so mean(y) == base_rate_target by construction
    (up to sample discreteness). Scaling a FIXED z keeps labels a deterministic
    function of eps_scale, which is what makes the downstream noise calibration
    monotone and reproducible.
    """
    label_signal = signal + eps_scale * z
    # tau at the (1 - base_rate) quantile => the top base_rate fraction is labeled 1.
    tau = float(np.quantile(label_signal, 1.0 - base_rate_target))
    y = (label_signal > tau).astype(int)
    return y, tau
