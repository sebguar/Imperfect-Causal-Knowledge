"""G1 pulled-back closed-form check.

Reconciles the intervention-space cost convention with Ehyaei's point-to-
boundary distance by pulling that closed form BACK through the SCM into intervention
space, then checks the brute-force optimum against it on a sample of individuals.
This is what tunes grid resolution.

    # The direct-set-all-acted-vars subset has M_S = I -> recovers raw Euclidean
    # point distance |h(x)|/||w||_2; upstream-only subsets encode propagation -> the
    # causal advantage. This reconciles the cost convention with the Ehyaei closed form.
    # Anchor: the Ehyaei closed-form r^M(v), ℓ₂/self-dual simplification.
    # The direct-set-all (M_S = I) subset reproduces Ehyaei's published raw
    # r^M(v) = |h(x)|/‖w‖₂ within the fixed 1.5×grid-diagonal tolerance, via the
    # REAL L2 severing path in
    # tests/test_g1_ehyaei_anchor.py::test_g1_direct_set_all_l2_matches_ehyaei_raw
    # (reachability precondition per the reachability filter). See
    # tests/test_g1_ehyaei_anchor.py and recourse-harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from icknowledge.recourse.model_conditions import L2TrueSCMModel
from icknowledge.recourse.procedure import RecourseSolution


def nonempty_subsets(acted: tuple[str, ...] | list[str]) -> list[tuple[str, ...]]:
    """All 2^k − 1 non-empty subsets of the actionable set, sizes 1..k.

    The classifier-optimal action is the argmin over all non-empty subsets of the
    actionable set of the pulled-back closed-form cost within each subset. For k=2
    (triangle) these are the two singletons and the pair; for k=3 (collider, chain)
    there are seven, including the three pair subsets, which generically win.
    """
    acted = tuple(acted)
    return [combo for r in range(1, len(acted) + 1) for combo in combinations(acted, r)]


def raw_euclidean_reach(model: LogisticRegression, factual_features: np.ndarray) -> float:
    """Raw ℓ₂ point-to-boundary distance |h(x)| / ||w||_2 (self-dual ℓ₂ collapse).

    This is the direct-set-all-acted-vars subset (M_S = I): the classic
    point-distance recourse cost when nothing propagates.
    """
    w = model.coef_[0]
    h = float(model.decision_function(factual_features.reshape(1, -1))[0])
    return abs(h) / float(np.linalg.norm(w, ord=2))


def closed_form_min_cost(
    model: LogisticRegression,
    true_model: L2TrueSCMModel,
    factual: pd.Series,
) -> tuple[float, str]:
    """Pulled-back closed-form minimum intervention cost over actionable subsets.

    For each intervention subset S over the actionable vars, M_S is the linear map
    from delta_S to the realized point displacement under the true SCM; the effective
    weight is w_tilde_S = M_S^T w and the reach (min ℓ₂ intervention cost to reach the
    boundary using S) is |h(x)| / ||w_tilde_S||_2. Returns (min reach, best-subset
    label).

    Enumeration: ALL 2^k − 1 non-empty subsets of the actionable set
    (`nonempty_subsets`). For k=2 (triangle) these are the two singletons and the
    full joint (M_S = I); for k=3 there are seven, including the three pair
    subsets (a pair with a free-propagating descendant generically wins).

    Per-subset effective weight ‖M_Sᵀw‖₂ is probed from the true SCM (exact for the
    linear family, constant Jacobian): keep EVERY member of S acted — a nonzero delta
    is a do, which severs the acted node from its parents — while non-S variables are
    left free to propagate (that free propagation IS the causal advantage). The
    gradient of w·displacement along each acted axis is read by a unit bump on that
    axis holding the rest of S acted, so a free descendant of one acted variable is
    counted, but a descendant that is itself acted (severed) is not.
    """
    features = true_model.feature_names
    w = model.coef_[0]
    factual_features = factual[features].to_numpy(dtype=float)
    h = float(model.decision_function(factual_features.reshape(1, -1))[0])

    best_eff = 0.0
    best_subset = ",".join(true_model.acted)
    for subset in nonempty_subsets(true_model.acted):
        # Base action = unit do on every member of S (all nonzero → all severed);
        # non-S variables absent → free to propagate.
        base = {v: 1.0 for v in subset}
        base_disp = true_model.predict(factual, base) - factual_features
        # M_Sᵀw component per acted axis: bump that axis by one unit with the rest of
        # S still acted, and read the change in w·displacement (linear ⇒ exact).
        grad = []
        for v in subset:
            bumped = dict(base)
            bumped[v] = 2.0
            bump_disp = true_model.predict(factual, bumped) - factual_features
            grad.append(float(w @ (bump_disp - base_disp)))
        eff = float(np.linalg.norm(grad, ord=2))
        if eff > best_eff:
            best_eff, best_subset = eff, ",".join(subset)

    if best_eff < 1e-12:
        # No subset can move h (degenerate classifier/SCM) → infinite reach.
        return float("inf"), best_subset
    return abs(h) / best_eff, best_subset


@dataclass
class AnchorReport:
    """Result of the G1 smoke check on a sampled set of individuals.

    Errors are ABSOLUTE (cost units); the pass threshold is ``tolerance`` multiples
    of one grid diagonal step. A relative-to-optimum tolerance is deliberately NOT
    used: near the decision boundary the closed-form reach is tiny, so a ratio would
    be dominated by discretization rather than by any harness error. The real claim
    the smoke check tests is that the grid resolves the optimum to within its own
    step, which is exactly an absolute, step-scaled statement.
    """

    n_sample: int
    tolerance: float  # allowed abs error as a MULTIPLE of grid_diag_step
    grid_diag_step: float  # ℓ₂ norm of one grid step across acted axes
    abs_threshold: float  # tolerance * grid_diag_step
    # L2 brute-force vs pulled-back closed form (abs cost error)
    l2_max_abs_err: float
    l2_median_abs_err: float
    l2_passed: bool
    # L0 believed optimum vs raw Euclidean (self-dual ℓ₂ collapse; abs cost error)
    l0_max_abs_err: float
    l0_median_abs_err: float
    l0_passed: bool

    @property
    def passed(self) -> bool:
        return self.l2_passed and self.l0_passed

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"G1 anchor smoke check [{status}] on n={self.n_sample} "
            f"(tol={self.tolerance:g}*step, step={self.grid_diag_step:.4f}, "
            f"abs_threshold={self.abs_threshold:.4f})\n"
            f"  L2 brute-force vs pulled-back closed form: "
            f"median abs err={self.l2_median_abs_err:.4f}, max={self.l2_max_abs_err:.4f} "
            f"[{'PASS' if self.l2_passed else 'FAIL'}]\n"
            f"  L0 believed vs raw Euclidean |h|/||w||: "
            f"median abs err={self.l0_median_abs_err:.4f}, max={self.l0_max_abs_err:.4f} "
            f"[{'PASS' if self.l0_passed else 'FAIL'}]"
        )


def anchor_smoke_check(
    model: LogisticRegression,
    true_model: L2TrueSCMModel,
    l2_solutions: list[RecourseSolution],
    l0_solutions: list[RecourseSolution],
    data: pd.DataFrame,
    grid_diag_step: float,
    span_limit: float,
    *,
    tolerance: float,
    n_sample: int,
    seed: int,
) -> AnchorReport:
    """Compare brute-force optima to the pulled-back closed form on a sample.

    Two checks (fairness-scoring):
      L2: brute-force ℓ₂ optimum ≈ pulled-back closed form (over actionable subsets).
      L0: believed optimum == raw Euclidean |h(x)|/||w||_2 (self-dual ℓ₂ collapse) —
          guards against the smoke check itself raising a false alarm.

    Only individuals whose closed-form optimum is REACHABLE within the grid span
    (``cf <= span_limit``) enter the check: if the cheapest action lies beyond the
    grid's ±k·SD reach, the brute force cannot find it and the mismatch would reflect
    span, not resolution. Among the reachable, the grid must land within ``tolerance``
    grid steps of the closed form; resolution is tuned upward until it does.
    """
    l2_by_idx = {s.index: s for s in l2_solutions}
    l0_by_idx = {s.index: s for s in l0_solutions}
    abs_threshold = tolerance * grid_diag_step

    eligible: list[tuple[int, float, float, float, float]] = []  # idx, l2_grid, l0_grid, cf, raw
    for idx, l2_sol in l2_by_idx.items():
        l0_sol = l0_by_idx.get(idx)
        if l0_sol is None or not l2_sol.believed_validity or not l0_sol.believed_validity:
            continue
        factual = data.iloc[idx]
        cf_cost, _ = closed_form_min_cost(model, true_model, factual)
        raw = raw_euclidean_reach(model, factual[true_model.feature_names].to_numpy(dtype=float))
        # reachable within the grid span (leave a small margin so the optimal grid
        # point is not clipped by the boundary of the span).
        if raw > span_limit:
            continue
        eligible.append((idx, l2_sol.believed_cost, l0_sol.believed_cost, cf_cost, raw))

    rng = np.random.default_rng(seed)
    if len(eligible) > n_sample:
        pick = rng.choice(len(eligible), size=n_sample, replace=False)
        eligible = [eligible[i] for i in sorted(pick.tolist())]

    if not eligible:
        raise ValueError(
            "no reachable individuals for the anchor smoke check — grid span too small "
            "relative to boundary distances; increase grid k/span in action_space."
        )

    l2_abs = np.array([abs(g - cf) for (_, g, _, cf, _) in eligible])
    l0_abs = np.array([abs(g - raw) for (_, _, g, _, raw) in eligible])

    return AnchorReport(
        n_sample=len(eligible),
        tolerance=tolerance,
        grid_diag_step=grid_diag_step,
        abs_threshold=abs_threshold,
        l2_max_abs_err=float(l2_abs.max()),
        l2_median_abs_err=float(np.median(l2_abs)),
        l2_passed=bool(l2_abs.max() <= abs_threshold),
        l0_max_abs_err=float(l0_abs.max()),
        l0_median_abs_err=float(np.median(l0_abs)),
        l0_passed=bool(l0_abs.max() <= abs_threshold),
    )


def grid_diagonal_step(axis_grids: dict[str, np.ndarray]) -> float:
    """ℓ₂ norm of one grid step across the acted axes (the resolution floor)."""
    steps = []
    for grid in axis_grids.values():
        diffs = np.diff(np.sort(grid))
        steps.append(float(diffs.min()) if len(diffs) else 0.0)
    return float(np.linalg.norm(steps, ord=2))


def grid_span_limit(axis_grids: dict[str, np.ndarray], margin: float = 0.9) -> float:
    """Largest closed-form optimum the grid can reach, as a conservative bound.

    An axis-aligned action can reach at most that axis's half-width; the smallest
    axis half-width (times ``margin``) is a safe reachability ceiling for the
    Euclidean point distance, ensuring the optimal grid point is not clipped by the
    span boundary.
    """
    half_widths = [float(np.max(np.abs(grid))) for grid in axis_grids.values()]
    return margin * min(half_widths)
