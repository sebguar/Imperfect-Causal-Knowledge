"""Group-blind logistic-regression training + PS-2-aligned calibration gate.

Trains ONE classifier h per SCM instance (h held constant across every
knowledge condition) on the additive and effect-modifying regimes, and calibrates
the label-noise scale so h is non-trivial but not ~perfect.

Pipeline (all randomness through the child-RNG utility, one master seed):
  1. Sample features from the true SCM (feature stream).
  2. Build the label signal beta·(descendant features); draw a fixed standardized
     label-noise base z (label stream, SEPARATE from SCM feature noise).
  3. Split train/test (split stream) — used ONLY for the honest accuracy number.
  4. Bisection-calibrate eps_scale so trained-h TEST accuracy lands in the band.
  5. Refit at the calibrated scale; assemble metrics; run the calibration gate.

Key decisions (stated at their point of use below):
  group-blind h: feature matrix = descendants only, A excluded.
  no scaling anywhere: cost units == classifier units == raw feature units.
  near-unregularized LR so (w,b) sit close to the true linear boundary.
  eps calibrated so accuracy ~0.825, asserted within [0.75, 0.90].
  recourse pool = negatively-classified over the FULL dataset (not the split).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from icknowledge.classifier.labeling import LabelSpec, compute_signal, make_labels
from icknowledge.scm.base import SCM
from icknowledge.utils.seeding import spawn_children

# Integer-seed ceiling for deriving a per-stream int seed from a child RNG.
_SEED_CEIL = 2**31 - 1


@dataclass
class ClassifierResult:
    """Everything the classifier stage must return for one trained SCM instance."""

    model: LogisticRegression  # the single fitted classifier h (reused across all conditions)
    feature_names: list[str]  # group-blind descendant features (A EXCLUDED)
    protected_attr: str  # the excluded group label column (A / Gender)
    accuracy_overall: float  # held-out TEST accuracy (the honest number)
    accuracy_by_group: dict[float, float]  # TEST accuracy per group value of A
    realized_base_rate: float  # mean(y) over the full dataset
    neg_counts_by_group: dict[float, int]  # negatively-classified counts per group
    negative_pool_indices: np.ndarray  # FULL-dataset indices of h-negatives
    label_spec: LabelSpec  # beta + calibrated (eps_scale, tau), reproducible
    n_samples: int
    dataset: pd.DataFrame  # the full sampled feature dataset (incl. A) the indices point into


# --------------------------------------------------------------------------- #
# Group-blind logistic regression
# --------------------------------------------------------------------------- #


def _make_lr(penalty: str | None, max_iter: int) -> LogisticRegression:
    """Near-unregularized LR on RAW features.

    # penalty=None (fallback: very large C) so the learned (w,b) sit close to
    # the true linear boundary, tightening the G1 anchor so a mismatch reads as a
    # bug, not regularization drift (the Ehyaei closed-form r^M(v)).
    """
    if penalty in (None, "none", "None"):
        try:
            return LogisticRegression(penalty=None, solver="lbfgs", max_iter=max_iter)
        except (ValueError, TypeError):
            # Fallback for sklearn builds that reject penalty=None: very large C
            # (i.e. negligible regularization) is the documented equivalent.
            return LogisticRegression(C=1e12, solver="lbfgs", max_iter=max_iter)
    return LogisticRegression(penalty=penalty, solver="lbfgs", max_iter=max_iter)


def _fit_group_blind_lr(
    x_raw: np.ndarray, y: np.ndarray, penalty: str | None, max_iter: int
) -> LogisticRegression:
    """Fit LR on the RAW descendant-feature matrix — NO scaling.

    # Single coordinate system for h, the l_p cost, and the Ehyaei closed-form
    # check. NO standardization/scaling is applied anywhere, so:
    #     cost units == classifier units == raw feature units.
    # Anchor: the Ehyaei closed-form r^M(v) assumes the cost and the classifier
    # share feature units. If scaling ever becomes necessary it must be carried
    # into BOTH the cost norm and the closed-form weights — out-of-scope, defer.
    """
    # invariant guard: x_raw is used verbatim, never transformed / rescaled.
    assert x_raw.ndim == 2, "feature matrix must be 2-D raw descendant features"
    clf = _make_lr(penalty, max_iter)
    clf.fit(x_raw, y)
    return clf


# --------------------------------------------------------------------------- #
# Noise calibration — bisection on eps_scale
# --------------------------------------------------------------------------- #


def _calibrate_eps_scale(
    accuracy_at: Callable[[float], float],
    *,
    target: float,
    band: tuple[float, float],
    scale_lo: float,
    scale_hi: float,
    max_iter: int,
) -> float:
    """Bisection for the eps_scale whose test accuracy is closest to ``target``.

    # Noise tuned so h is non-trivial but NOT ~perfect
    # (realism). Accuracy is MONOTONE DECREASING in eps_scale (more label noise =>
    # labels less predictable from X => lower accuracy), and the base rate is held
    # fixed at every scale by re-thresholding tau, so bisection is well-posed.
    """
    lo_band, hi_band = band
    best_scale: float | None = None
    best_gap = np.inf
    lo, hi = scale_lo, scale_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        acc = accuracy_at(mid)
        if lo_band <= acc <= hi_band and abs(acc - target) < best_gap:
            best_scale, best_gap = mid, abs(acc - target)
        # accuracy decreasing in scale: too accurate => add noise (raise lo).
        if acc > target:
            lo = mid
        else:
            hi = mid
    return best_scale if best_scale is not None else 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def build_classifier(scm: SCM, cfg: DictConfig) -> ClassifierResult:
    """Train one group-blind LR on data from ``scm`` and run the calibration gate.

    ``cfg`` is the FULL config: ``cfg.seed`` is the single master seed and
    ``cfg.classifier`` holds the classifier knobs (N, base_rate_target, accuracy_band,
    min_neg_per_group, beta, noise-calibration settings, penalty, ...).
    """
    ccfg = cfg.classifier
    protected = str(ccfg.protected_attr)
    n = int(ccfg.N)
    base_rate_target = float(ccfg.base_rate_target)

    # -- Group-blind feature set = DESCENDANTS of A only; A EXCLUDED. -------
    # A (Gender) is kept as a separate group-label column for scoring, never in X.
    # (1) isolates the H4 effect-modification mechanism from direct classifier
    #     discrimination — the corpus "unaware baseline";
    # (2) A is non-actionable (root, immutable) so never a recourse target;
    # (3) A still enters the SCM, so the group contrast is preserved.
    descendants = nx.descendants(scm.graph, protected)
    if not descendants:
        raise ValueError(f"protected node {protected!r} has no descendants to use as features.")
    feature_names = [node for node in scm.nodes if node in descendants]  # topo order
    assert protected not in feature_names, "group-blind invariant: A must not be a feature."
    beta = {k: float(v) for k, v in OmegaConf.to_container(ccfg.beta, resolve=True).items()}

    # -- seeding: three INDEPENDENT child streams from ONE master seed ----------
    # Separate streams for feature sampling, label noise, and the train/test split
    # (spawn_children utility). The label-noise stream is DISTINCT from the
    # SCM's own feature-noise streams (eps is NOT the SCM noise).
    feat_rng, label_rng, split_rng = spawn_children(int(cfg.seed), 3)
    feature_seed = int(feat_rng.integers(0, _SEED_CEIL))  # SCM.sample takes an int seed
    data = scm.sample(n, seed=feature_seed)

    x_full = data[feature_names].to_numpy()  # RAW units, A excluded
    groups = data[protected].to_numpy()
    signal = compute_signal(data[feature_names], beta)  # beta · descendant features
    z = label_rng.standard_normal(n)  # fixed standardized label-noise base draw (own stream)

    # -- train/test split (own stream) — the split ONLY sizes the accuracy number.
    split_seed = int(split_rng.integers(0, _SEED_CEIL))
    perm = np.random.default_rng(split_seed).permutation(n)
    n_test = int(round(float(ccfg.test_fraction) * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    penalty = ccfg.get("penalty", None)
    max_iter_lr = int(ccfg.get("max_iter_lr", 1000))

    def _evaluate(eps_scale: float) -> tuple[float, float, np.ndarray, LogisticRegression]:
        """(test accuracy, tau, labels, fitted h) at a candidate eps_scale."""
        y, tau = make_labels(signal, z, eps_scale, base_rate_target)
        model = _fit_group_blind_lr(x_full[train_idx], y[train_idx], penalty, max_iter_lr)
        acc = accuracy_score(y[test_idx], model.predict(x_full[test_idx]))
        return float(acc), tau, y, model

    ncfg = ccfg.noise_calibration
    eps_scale = _calibrate_eps_scale(
        lambda s: _evaluate(s)[0],
        target=float(ccfg.accuracy_target),
        band=tuple(ccfg.accuracy_band),
        scale_lo=float(ncfg.scale_lo),
        scale_hi=float(ncfg.scale_hi),
        max_iter=int(ncfg.max_iter),
    )

    # -- refit at the calibrated scale and assemble metrics --------------------
    accuracy_overall, tau, y, model = _evaluate(eps_scale)

    y_pred_test = model.predict(x_full[test_idx])
    y_test = y[test_idx]
    groups_test = groups[test_idx]
    group_values = sorted(np.unique(groups).tolist())
    # Per-group TEST accuracy: an optional fairness sanity
    # check — guards against h being much worse for one group, which would confound
    # burden readings.
    accuracy_by_group = {
        float(g): (
            float(accuracy_score(y_test[groups_test == g], y_pred_test[groups_test == g]))
            if np.any(groups_test == g)
            else float("nan")
        )
        for g in group_values
    }

    # -- Recourse pool = h-negatives over the FULL dataset (not the split). --
    # The split is only for the honest accuracy number; the out-of-sample
    # (test-negatives-only) recourse pool is a deferred robustness alt, not core.
    y_pred_full = model.predict(x_full)
    neg_mask = y_pred_full == 0
    negative_pool_indices = np.nonzero(neg_mask)[0]
    neg_counts_by_group = {
        float(g): int(np.count_nonzero((groups == g) & neg_mask)) for g in group_values
    }

    beta_vector = np.array([beta[name] for name in feature_names])
    label_spec = LabelSpec(
        feature_names=list(feature_names),
        beta=beta,
        eps_scale=float(eps_scale),
        tau=float(tau),
        base_rate_target=base_rate_target,
        realized_base_rate=float(y.mean()),
        beta_vector=beta_vector,
    )

    result = ClassifierResult(
        model=model,
        feature_names=list(feature_names),
        protected_attr=protected,
        accuracy_overall=float(accuracy_overall),
        accuracy_by_group=accuracy_by_group,
        realized_base_rate=float(y.mean()),
        neg_counts_by_group=neg_counts_by_group,
        negative_pool_indices=negative_pool_indices,
        label_spec=label_spec,
        n_samples=n,
        # The exact sampled dataset (features + A) the negative-pool indices point
        # into, so the recourse harness fetches factual rows without re-deriving the
        # internal feature seed. Recourse acts on X features; A is retained for
        # group-conditional propagation and group-wise scoring.
        dataset=data,
    )

    # -- calibration gate (fail loudly). The gate needs the full feature matrix;
    # bind it via a closure-free helper so the checker stays self-contained.
    _run_calibration_gate(result, x_full, ccfg)
    return result


def _run_calibration_gate(result: ClassifierResult, x_full: np.ndarray, ccfg: DictConfig) -> None:
    """PS-2-aligned gate — assert loudly (scm-specification convention 6)."""
    lo, hi = tuple(ccfg.accuracy_band)
    acc = result.accuracy_overall
    assert lo <= acc <= hi, (
        f"trained-h test accuracy {acc:.3f} outside band [{lo}, {hi}] — h is trivial "
        "or ~perfect; re-calibrate eps_scale."
    )

    # both classes predicted (h non-degenerate).
    assert set(np.unique(result.model.predict(x_full)).tolist()) == {0, 1}, (
        "h is degenerate — it predicts a single class; re-calibrate."
    )

    # global base rate within tolerance of target (balanced design).
    target, tol = float(ccfg.base_rate_target), float(ccfg.base_rate_tol)
    assert abs(result.realized_base_rate - target) <= tol, (
        f"realized base rate {result.realized_base_rate:.3f} not within {tol} of "
        f"target {target} — tau mis-set."
    )

    # per-group negative (recourse) pool non-trivial for BOTH groups.
    # H4 measures PER-GROUP burden, so each group's negative pool must be
    # non-trivial. Anchor: per-group extension of the Karimi balanced design,
    # required by Von Kügelgen Def. 3.1 group-burden disparity.
    min_neg = int(ccfg.min_neg_per_group)
    for group, count in result.neg_counts_by_group.items():
        assert count >= min_neg, (
            f"group A={group:g} has only {count} negatively-classified individuals "
            f"(< min_neg_per_group={min_neg}); per-group recourse pool too thin."
        )


def classifier_provenance(result: ClassifierResult) -> dict:
    """JSON-serializable record of the ONE fitted h for this (cell, seed).

    Mirrors `icknowledge.estimation.base.estimation_provenance`: the persisted
    artifact is the version-controlled evidence that a SINGLE classifier was fit
    per (cell, seed) and reused verbatim across every knowledge condition (
    h is held constant across L0 / L1-oracle / L2; only the causal model
    varies). Without it the coefficients that produced a scoring table are
    recoverable only by re-running the fit.

    ``coef`` is emitted as a NAME -> VALUE mapping rather than a bare vector: the
    ordering of ``LogisticRegression.coef_`` is the ordering of the raw feature
    matrix (group-blind descendants in topological order), and pairing the
    two here means a future feature-order change cannot silently re-map
    coefficients in an already-written artifact.

    ``calibration`` carries the classifier label-noise state (``eps_scale``, ``tau``)
    that determines the labels h was fit on; the coefficients alone do not
    reproduce the fit without it.
    """
    model = result.model
    spec = result.label_spec
    coef = np.asarray(model.coef_, dtype=float).ravel()
    assert len(coef) == len(result.feature_names), (
        "coef_ length must match the group-blind feature set — the name->value "
        "pairing below would otherwise mis-map coefficients."
    )
    return {
        "estimator": f"{type(model).__module__}.{type(model).__qualname__}",
        # A is EXCLUDED; this is the exact column order coef_ is indexed by.
        "feature_names": list(result.feature_names),
        "protected_attr": result.protected_attr,
        "coef": {name: float(v) for name, v in zip(result.feature_names, coef, strict=True)},
        "intercept": float(np.asarray(model.intercept_, dtype=float).ravel()[0]),
        "classes": [float(c) for c in np.asarray(model.classes_).ravel()],
        "n_iter": [int(v) for v in np.asarray(model.n_iter_).ravel()],
        "params": {
            # Near-unregularized. Recorded because penalty=None vs. the C=1e12
            # fallback (see _make_lr) is a build-dependent branch, and which one ran
            # is not recoverable from the coefficients.
            "penalty": model.get_params().get("penalty"),
            "C": float(model.get_params().get("C")),
            "solver": model.get_params().get("solver"),
            "max_iter": int(model.get_params().get("max_iter")),
            "fit_intercept": bool(model.get_params().get("fit_intercept")),
        },
        # [the classifier construction] The calibrated label-noise state. eps_scale varies BY REGIME
        # by design (classifier QUALITY is what is held constant, not noise scale),
        # so persisting it per (cell, seed) is what makes the classifier policy auditable
        # rather than asserted.
        "calibration": {
            "eps_scale": float(spec.eps_scale),
            "tau": float(spec.tau),
            "base_rate_target": float(spec.base_rate_target),
            "realized_base_rate": float(spec.realized_base_rate),
            "beta": {k: float(v) for k, v in spec.beta.items()},
        },
        "metrics": {
            "accuracy_overall": float(result.accuracy_overall),
            "accuracy_by_group": {f"{g:g}": float(a) for g, a in result.accuracy_by_group.items()},
            "neg_counts_by_group": {
                f"{g:g}": int(c) for g, c in result.neg_counts_by_group.items()
            },
            "n_samples": int(result.n_samples),
            "n_negative_pool": int(len(result.negative_pool_indices)),
        },
    }


def format_report(result: ClassifierResult) -> str:
    """Short human-readable report (accuracy overall + per group, base rate, ...)."""
    spec = result.label_spec
    by_group_acc = ", ".join(f"A={g:g}: {a:.3f}" for g, a in result.accuracy_by_group.items())
    by_group_neg = ", ".join(f"A={g:g}: {c}" for g, c in result.neg_counts_by_group.items())
    return (
        f"Classifier h on {result.n_samples} samples "
        f"(features={result.feature_names}, group-blind: {result.protected_attr} excluded)\n"
        f"  accuracy overall : {result.accuracy_overall:.3f}\n"
        f"  accuracy by group: {by_group_acc}\n"
        f"  realized base rate: {result.realized_base_rate:.3f} "
        f"(target {spec.base_rate_target})\n"
        f"  negatives by group: {by_group_neg}\n"
        f"  calibrated eps_scale={spec.eps_scale:.4f}, tau={spec.tau:.4f}"
    )
