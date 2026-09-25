"""G1 harness anchor: brute-force vs closed-form on linear triangle → LR → h.

Validates the Ehyaei Prop. 10 closed form against the brute-force recourse optimum
— the external tether for the cost-computation pipeline. A mismatch that appears
only at L2 while L0 agrees points at the pullback convention, not the harness.
The closed forms here are an INDEPENDENT reimplementation of
`icknowledge/recourse/anchor.py`, kept separate so the test never checks the
harness against itself.

Notation: the action→counterfactual map is **J**; `J_S` is J restricted to acted
subset S. J and the H4 descriptor S(g) are distinct objects.

    r^M(v)   = |h(v)| / ‖w‖₂          raw form; J_S = I, anchors L0
    r^CAU(v) = |h(v)| / ‖J_Sᵀw‖₂      pulled back, anchors L2
    triangle X₁→X₂ with coefficient θ, acted subset S:
    S = {X₁}      → J_S = [1, θ]ᵀ,  ‖J_Sᵀw‖ = |w₁ + θ w₂|
    S = {X₂}      → J_S = [0, 1]ᵀ,  ‖J_Sᵀw‖ = |w₂|
    S = {X₁,X₂}   → J_S = I₂,       ‖J_Sᵀw‖ = ‖w‖₂     (propagation severed by do)

r^CAU_L2(v) = |h(v)| / max_S ‖J_Sᵀw‖. Since ‖w‖₂ ≥ |w₂|, {X₂} is dominated by
{X₁,X₂} and is never the argmin (locked in TestAnchorArithmetic).

Eligibility, identical to `anchor.py`: only individuals whose closed-form optimum
is REACHABLE within the ±k·SD grid span AND that found a believed-valid action
enter the pass/fail criterion. Ehyaei Prop. 10 assumes an unbounded actionable
variable, so an off-span individual carries no prediction at all; that
box-truncation excess is filtered out (the reachability filter) rather than folded
into the grid-resolution tolerance.

Cost convention (ℓ₂ intervention cost, acted variables only, raw units) and its
grounding: icknowledge/recourse/cost.py.
"""

import inspect
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Anchor arithmetic — pure functions, no dependency on the recourse pipeline
# ---------------------------------------------------------------------------

def signed_margin(X: np.ndarray, w: np.ndarray, b_sk: float) -> np.ndarray:
    """sklearn-convention classifier score: h(x) = wᵀx + b_sk. Negative → deny."""
    return X @ w + b_sk


def ehyaei_raw(X: np.ndarray, w: np.ndarray, b_sk: float) -> np.ndarray:
    """Raw Ehyaei r^M(v) — displacement anchor; validates L0."""
    return np.abs(signed_margin(X, w, b_sk)) / np.linalg.norm(w)


def effective_weights_triangle(w: np.ndarray, theta: float) -> dict:
    """‖J_Sᵀw‖₂ for each acted subset on the triangle SCM.

    J_S is the action→counterfactual map restricted to subset S.
    Triangle-specific. A chain / collider extension needs its own version.
    """
    return {
        frozenset({"X1"}):        abs(w[0] + theta * w[1]),
        frozenset({"X2"}):        abs(w[1]),
        frozenset({"X1", "X2"}):  float(np.linalg.norm(w)),
    }


def winning_subset_triangle(w: np.ndarray, theta: float) -> frozenset:
    """Acted subset that minimizes r^CAU — i.e. the argmax effective weight."""
    weights = effective_weights_triangle(w, theta)
    return max(weights, key=lambda k: weights[k])


def pulled_back_l2_triangle(
    X: np.ndarray, w: np.ndarray, b_sk: float, theta: float
) -> np.ndarray:
    """Pulled-back closed form for L2 on the triangle: min-cost over subsets."""
    weights = effective_weights_triangle(w, theta)
    max_effective = max(weights.values())
    if max_effective < 1e-8:
        pytest.skip(
            "Degenerate cell: max effective weight ≈ 0 → recourse infeasible on "
            "this classifier / SCM combination."
        )
    return np.abs(signed_margin(X, w, b_sk)) / max_effective


# ---------------------------------------------------------------------------
# Fixture — assembles the (X_neg, w, b, θ, costs, grid) bundle from the real run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def g1_cell():
    """Run the additive-linear-triangle L0 + L2 pipeline at the master seed.

    Returns a namespace with:
        X_neg:            (n, 2) features [X₁, X₂] for ELIGIBLE negative rows
        w:                (2)   classifier weights on [X₁, X₂] (group-blind, the classifier
        construction)
        b_sk:             scalar sklearn intercept
        theta:            X₁→X₂ coefficient PROBED from the true SCM (asserted vs spec)
        cost_l0:          (n,)   brute-force believed ℓ₂ intervention cost, L0
        cost_l2:          (n,)   brute-force believed ℓ₂ intervention cost, L2
        cost_l2_directset:(n,)   L2 brute-force cost RESTRICTED to S = {X₁, X₂}
                                 (both axes acted → do() severs X₁→X₂ → J_S = I)
        probe_wX1/wX2:    (n,)   anchor.py-style probed singleton effective weights
        grid_step_diag:   scalar ℓ₂ combination of per-axis grid steps
        tolerance:        scalar multiplier on grid_step_diag (config anchor.tolerance)
        n_total / n_eligible: pool sizes before / after the reachability filter
    """
    from icknowledge.recourse.action_space import build_action_space
    from icknowledge.recourse.anchor import grid_diagonal_step, grid_span_limit
    from icknowledge.recourse.cost import intervention_cost_batch
    from icknowledge.recourse.model_conditions import L2TrueSCMModel
    from icknowledge.recourse.pipeline import run_regime
    from icknowledge.scm import make_linear_triangle
    from icknowledge.utils.config import load_config

    cfg = load_config("configs/recourse_triangle.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        run = run_regime(cfg, "additive")
    clf, table = run.classifier, run.table
    feats = clf.feature_names  # ['X1', 'X2']
    w = clf.model.coef_[0].astype(float)
    b_sk = float(clf.model.intercept_[0])

    scm = make_linear_triangle("additive")
    true_model = L2TrueSCMModel(scm, feats, list(cfg.recourse.actionable))

    # θ probed from the true SCM: a unit do(X₁) displaces X₂ by exactly θ under the
    # linear family (constant Jacobian). Tethered to the SCM, never a magic constant.
    probe = clf.dataset.iloc[int(clf.negative_pool_indices[0])]
    disp = true_model.predict(probe, {"X1": 1.0}) - probe[feats].to_numpy(float)
    theta = float(disp[feats.index("X2")])

    # Probing θ from the realizer makes the closed form inherit any realizer bug,
    # which would cancel against the brute-force and pass silently. This assert
    # recovers the realizer-vs-spec bug class (e.g. `g` wired to the wrong edge in
    # make_linear_triangle). Spec value read from the SCM builder's declared default
    # rather than hardcoded. Manually verified once (indices 19 and 1: dX2 == 0.5 *
    # delta_X1 to 6dp); this makes it a standing regression guard.
    structural_g = float(inspect.signature(make_linear_triangle).parameters["g"].default)
    assert np.isclose(theta, structural_g), (
        f"Realizer propagates theta={theta} but SCM spec declares g={structural_g} "
        "-- structural-map/spec mismatch (e.g. g wired to the wrong edge)."
    )

    # Grid geometry from the SAME action space the run used.
    action_space = build_action_space(
        clf.dataset, list(cfg.recourse.actionable),
        mode=str(cfg.recourse.grid.mode), k=float(cfg.recourse.grid.k),
        resolution=int(cfg.recourse.grid.resolution),
    )
    grid_step_diag = grid_diagonal_step(action_space.axis_grids)
    span_limit = grid_span_limit(action_space.axis_grids)
    tolerance = float(cfg.recourse.anchor.tolerance)

    l0 = table[table.condition == "L0"].reset_index(drop=True)
    l2 = table[table.condition == "L2"].reset_index(drop=True)
    assert (l0["index"].to_numpy() == l2["index"].to_numpy()).all(), "L0/L2 row misalignment"
    X_all = l0[[f"factual_{f}" for f in feats]].to_numpy(float)

    # Eligibility (mirrors anchor.py:167): reachable within the grid span AND a
    # believed-valid action found on BOTH conditions. Reachability uses r^M (the
    # raw distance); the L2 optimum r^CAU ≤ r^M, so r^M ≤ span is conservative.
    rM = np.abs(X_all @ w + b_sk) / np.linalg.norm(w)
    elig = (
        (rM <= span_limit)
        & (l0["believed_validity"].to_numpy() == 1)
        & (l2["believed_validity"].to_numpy() == 1)
        & l0["found"].to_numpy()
        & l2["found"].to_numpy()
    )
    elig_indices = l0["index"].to_numpy()[elig]

    # --- Direct-set-all subset S = {X₁, X₂} under L2 (the actual missing tether) ---
    # Restrict the L2 brute force to grid rows with BOTH axes nonzero. Under L2,
    # do(X₁, X₂) sets both classifier inputs directly and severs X₁→X₂, so J_S = I
    # and the optimum must collapse to Ehyaei's raw |h|/‖w‖₂. Exercises the real
    # L2TrueSCMModel severing path (subsets are implicit in the nonzero pattern),
    # which anchor.py never probes (anchor.py:65 hardcodes ‖w‖ for this subset).
    cand = action_space.candidate_matrix()  # (K, 2)
    both_nonzero = (cand[:, 0] != 0.0) & (cand[:, 1] != 0.0)
    cand_both = cand[both_nonzero]
    cost_both = intervention_cost_batch(cand_both)  # ℓ₂ over both acted axes

    # anchor.py:73 probes each singleton with {v: 1.0}. predict() takes a DELTA
    # (model_conditions.py:30 → :77-79 adds it to factual), so this is do(v := v+1),
    # a unit action → d_v is the unit column of J, invariant across individuals.
    # Recompute per eligible individual so any per-individual scaling could not hide.
    directset, probe_wX1, probe_wX2 = [], [], []
    for i in elig_indices:
        f = clf.dataset.iloc[int(i)]
        ff = f[feats].to_numpy(float)
        cf = true_model.predict_batch(f, cand_both)  # (K_both, 2) believed CF, L2
        valid = clf.model.decision_function(cf) > 0.0
        directset.append(float(cost_both[valid].min()) if valid.any() else float("nan"))
        probe_wX1.append(abs(float(w @ (true_model.predict(f, {"X1": 1.0}) - ff))))
        probe_wX2.append(abs(float(w @ (true_model.predict(f, {"X2": 1.0}) - ff))))

    return SimpleNamespace(
        X_neg=X_all[elig],
        w=w,
        b_sk=b_sk,
        theta=theta,
        cost_l0=l0["believed_cost"].to_numpy(float)[elig],
        cost_l2=l2["believed_cost"].to_numpy(float)[elig],
        cost_l2_directset=np.array(directset),
        probe_wX1=np.array(probe_wX1),
        probe_wX2=np.array(probe_wX2),
        grid_step_diag=grid_step_diag,
        tolerance=tolerance,
        n_total=int(len(X_all)),
        n_eligible=int(elig.sum()),
    )


# ---------------------------------------------------------------------------
# Part A — L0 brute-force vs raw Ehyaei (J_S = I limit, external tether)
# ---------------------------------------------------------------------------

def test_g1_part_a_l0_matches_ehyaei_raw(g1_cell):
    """L0 believes J = I → brute-force should match Ehyaei's Prop. 10 formula.

    Grid-bounded one-sided criterion:
        r^M ≤ cost_l0_brute ≤ r^M + tolerance · grid_step_diag
    """
    # A NaN cost (found=False) makes `NaN >= -1e-8` False and would trip the
    # lower-bound assert below, whose message says "fell BELOW the anchor" — the
    # single most diagnostic signature in G1. Fail here first with the correct cause.
    n_nan = int(np.isnan(g1_cell.cost_l0).sum())
    assert not np.isnan(g1_cell.cost_l0).any(), (
        f"{n_nan} individuals have NaN cost (found=False) after the reachability mask. "
        "This is an infeasibility, NOT a pullback undercut -- do not read this as a "
        "convention bug."
    )

    r_ehyaei = ehyaei_raw(g1_cell.X_neg, g1_cell.w, g1_cell.b_sk)
    excess = g1_cell.cost_l0 - r_ehyaei

    # Lower bound: a discrete-grid minimizer cannot undercut the continuous optimum
    # beyond floating-point noise. A violation means the closed form is wrong or the
    # brute-force objective differs (e.g. displacement, not intervention cost).
    assert (excess >= -1e-8).all(), (
        f"L0 brute-force fell BELOW Ehyaei anchor (max undercut "
        f"{-excess.min():.6e}) — pullback/convention bug, not a harness bug."
    )

    # Upper bound: brute-force overshoots by at most the grid resolution in the
    # action direction. 1.5 × ℓ₂ diagonal is conservative for the axis-aligned
    # single-lever cells L0 actually chooses in this SCM.
    abs_threshold = g1_cell.tolerance * g1_cell.grid_step_diag
    assert (excess <= abs_threshold).all(), (
        f"L0 brute-force exceeds Ehyaei + tolerance. "
        f"max excess = {excess.max():.6e}, threshold = {abs_threshold:.6e}. "
        f"Either the grid is too coarse or the closed form is inapplicable "
        f"in this cell (check w, b, and that X_neg is truly negative-class)."
    )


# ---------------------------------------------------------------------------
# Part B — L2 brute-force vs pulled-back closed form (J_S ≠ I, general case)
# ---------------------------------------------------------------------------

def test_g1_part_b_l2_matches_pulled_back(g1_cell):
    """L2 acts optimally under the true SCM → matches min-subset pulled-back form.

    For the linear-triangle cell with the calibrated classifier, single-lever X₁
    dominates: |w₁ + θ w₂| > ‖w‖₂ > |w₂|. If a future seed / SCM flips this
    dominance, the max-over-subsets formulation still holds.
    """
    n_nan = int(np.isnan(g1_cell.cost_l2).sum())
    assert not np.isnan(g1_cell.cost_l2).any(), (
        f"{n_nan} individuals have NaN cost (found=False) after the reachability mask. "
        "This is an infeasibility, NOT a pullback undercut -- do not read this as a "
        "convention bug."
    )

    r_pullback = pulled_back_l2_triangle(
        g1_cell.X_neg, g1_cell.w, g1_cell.b_sk, g1_cell.theta
    )
    excess = g1_cell.cost_l2 - r_pullback

    assert (excess >= -1e-8).all(), (
        f"L2 brute-force fell BELOW pulled-back anchor (max undercut "
        f"{-excess.min():.6e}) — check J_S enumeration, θ sign, or that the "
        f"brute-force actually enumerates single-lever subsets."
    )

    abs_threshold = g1_cell.tolerance * g1_cell.grid_step_diag
    assert (excess <= abs_threshold).all(), (
        f"L2 brute-force exceeds pullback + tolerance. "
        f"max excess = {excess.max():.6e}, threshold = {abs_threshold:.6e}."
    )


# ---------------------------------------------------------------------------
# Phase 2 — direct-set-all under L2 vs Ehyaei raw (the L2-code-path ↔ Ehyaei tether)
# ---------------------------------------------------------------------------

def test_g1_direct_set_all_l2_matches_ehyaei_raw(g1_cell):
    """Force S = {X₁, X₂} under L2; do() on all inputs severs X₁→X₂ so J_S = I and
    the pullback must collapse to Ehyaei's PUBLISHED raw form |h|/‖w‖₂.

    Why this and not Part A/B: Part A tethers L0, whose code path never builds J_S.
    Part B tethers L2 but against our own reimplementation (internal). This restricts
    the real L2 brute force to both-axes-acted grid rows — the only check that ties
    the L2 severing code to an external published number. A subset engine that forgot
    do(X₂) severs X₁→X₂ would compute ‖(w₁+θw₂, w₂)‖ = 0.9223 instead of ‖w‖ = 0.7208,
    making the buggy cost cheaper and undercutting the correct value — caught here
    unambiguously by Ehyaei's number.

    Excess floor (Amendment 1). Excluding the axes, the smallest admissible action is
    (±s₁, ±s₂) with norm == the diagonal step (0.161139). So for a near-boundary
    individual (r^M → 0) this constrained minimum cannot fall below the diagonal step
    — a QUANTIZATION FLOOR, distinct from the grid-discretization excess of Part A/B
    and from the box-truncation handled by the reachability mask. Bound: round each
    component of the ideal δ* = (|h|/‖w‖²)·w outward to the next grid multiple (min one
    step); rounding outward along +w only raises wᵀδ, so the classifier still flips,
    and the norm exceeds r^M by at most one diagonal cell. Therefore
        excess ≤ grid_step_diag (0.161139) < tolerance·diag (0.241708) → passes.
    Anything above grid_step_diag means the derivation or the filter is wrong, NOT the
    grid — report, do not tune (hard constraint 1). If a future cell exceeds tolerance
    here, restrict this check to r^M above ~1 diagonal step (where the floor does not
    bind); never loosen the tolerance.
    """
    cost = g1_cell.cost_l2_directset
    n_nan = int(np.isnan(cost).sum())
    assert not np.isnan(cost).any(), (
        f"{n_nan} individuals have NaN direct-set cost (no valid both-axes action). "
        "Infeasibility, NOT a pullback undercut."
    )

    r_ehyaei = ehyaei_raw(g1_cell.X_neg, g1_cell.w, g1_cell.b_sk)
    excess = cost - r_ehyaei

    # Lower bound: the continuous {X₁,X₂} optimum IS |h|/‖w‖₂ (J_S = I), so any grid
    # point can only be ≥ it. An undercut means the severing is wrong (J_S ≠ I).
    assert (excess >= -1e-8).all(), (
        f"Direct-set-all L2 fell BELOW Ehyaei raw (max undercut {-excess.min():.6e}) "
        f"— do(X₂) is NOT severing X₁→X₂; the {{X₁,X₂}} subset weight is inflated."
    )

    # Upper bound: grid discretization + the axis-exclusion quantization floor, both
    # bounded by one diagonal cell (see docstring).
    abs_threshold = g1_cell.tolerance * g1_cell.grid_step_diag
    assert (excess <= abs_threshold).all(), (
        f"Direct-set-all L2 exceeds Ehyaei raw + tolerance. "
        f"max excess = {excess.max():.6e}, threshold = {abs_threshold:.6e}. "
        f"If max excess ≤ grid_step_diag ({g1_cell.grid_step_diag:.6f}) it is the "
        f"quantization floor; above that the derivation/filter is wrong (do not tune)."
    )


# ---------------------------------------------------------------------------
# Amendment 2 — anchor.py's probe technique vs the independent reimplementation
# ---------------------------------------------------------------------------

def test_anchor_probe_matches_reimplementation(g1_cell):
    """anchor.py builds J_S by probing the realizer (`predict(factual, {v: 1.0})`,
    anchor.py:73); effective_weights_triangle builds it analytically. If predict()
    took ABSOLUTE values instead of deltas, {v: 1.0} would be do(X₁ := 1.0) and give
    d_v = (1 − x₁)·(1, θ) — scaled per-individual, invisible on a single row. Compare
    across the whole eligible pool so a per-individual scaling cannot hide.

    Two INDEPENDENT computations asserted equal — not the code under test calling
    itself (hard constraint 4): the probe side goes through L2TrueSCMModel, the
    reimpl side is closed-form linear algebra.
    """
    reimpl = effective_weights_triangle(g1_cell.w, g1_cell.theta)
    assert np.allclose(g1_cell.probe_wX1, reimpl[frozenset({"X1"})]), (
        "anchor.py's probed {X1} effective weight disagrees with the independent "
        "reimplementation -- check predict()'s action semantics (absolute vs delta)."
    )
    assert np.allclose(g1_cell.probe_wX2, reimpl[frozenset({"X2"})]), (
        "anchor.py's probed {X2} effective weight disagrees with the independent "
        "reimplementation -- check predict()'s action semantics (absolute vs delta)."
    )
    # The probe must be individual-invariant (unit column of J), not scaled.
    assert np.ptp(g1_cell.probe_wX1) < 1e-9 and np.ptp(g1_cell.probe_wX2) < 1e-9, (
        "Probed effective weights vary across individuals -- predict() is treating "
        "{v: 1.0} as an absolute set, not a unit delta."
    )


# ---------------------------------------------------------------------------
# Diagnostic — surfaces tightness numbers for the anchor-tolerance record
# ---------------------------------------------------------------------------

def test_g1_diagnostic_report(g1_cell, capsys):
    """Not a pass/fail gate — logs the tightness statistics for the
    anchor-tolerance record.

    Run with `pytest -s` to see output.
    """
    r_ehyaei = ehyaei_raw(g1_cell.X_neg, g1_cell.w, g1_cell.b_sk)
    r_pullback = pulled_back_l2_triangle(
        g1_cell.X_neg, g1_cell.w, g1_cell.b_sk, g1_cell.theta
    )
    excess_a = g1_cell.cost_l0 - r_ehyaei
    excess_b = g1_cell.cost_l2 - r_pullback
    excess_d = g1_cell.cost_l2_directset - r_ehyaei

    weights = effective_weights_triangle(g1_cell.w, g1_cell.theta)
    winning = winning_subset_triangle(g1_cell.w, g1_cell.theta)
    step = g1_cell.grid_step_diag

    with capsys.disabled():
        print("\nG1 anchor tightness (additive linear triangle):")
        print(f"  negatively classified: {g1_cell.n_total} "
              f"(eligible after reachability filter: {g1_cell.n_eligible})")
        print(f"  theta (probed X1->X2):   {g1_cell.theta:.6f}")
        print(f"  grid_step_diag:          {step:.6f}")
        print(f"  tolerance × diag_step:   {g1_cell.tolerance * step:.6f}")
        print(f"  L2 predicted winning subset: {set(winning)}")
        print(f"  effective weights: "
              f"{ {tuple(sorted(k)): round(v, 6) for k, v in weights.items()} }")
        print("  Part A (L0 vs Ehyaei raw):")
        print(f"    max excess: {excess_a.max():.6f} ({excess_a.max()/step:.3f} × diag step)")
        print(f"    mean excess: {excess_a.mean():.6f}")
        print("  Part B (L2 vs pullback reimplementation):")
        print(f"    max excess: {excess_b.max():.6f} ({excess_b.max()/step:.3f} × diag step)")
        print(f"    mean excess: {excess_b.mean():.6f}")
        print("  Direct-set-all L2 vs Ehyaei raw (J_S = I tether):")
        print(f"    max excess: {excess_d.max():.6f} ({excess_d.max()/step:.3f} × diag step)")
        print(f"    mean excess: {excess_d.mean():.6f}")


# ---------------------------------------------------------------------------
# Unit tests on the anchor arithmetic itself (no pipeline dependency)
# ---------------------------------------------------------------------------

class TestAnchorArithmetic:
    """Sanity tests on the closed-form functions, independent of pipeline output."""

    def test_ehyaei_matches_known_index_19(self):
        """r^M(v_19) on the pinned 6-dp inputs = 0.8042296885 (exact for these inputs;
        the full-precision pipeline value differs at ~1e-7 and is not what this
        arithmetic regression pins)."""
        w = np.array([0.474961, 0.542195])
        b_sk = -1.269163
        v = np.array([[0.342046, 0.971993]])
        assert np.isclose(ehyaei_raw(v, w, b_sk)[0], 0.8042296885, atol=1e-9)

    def test_pullback_collapses_to_ehyaei_when_theta_zero(self):
        """No propagation → X₁-only weight = |w₁|, both-features weight = ‖w‖.
        With ‖w‖ ≥ |w₁|, the both-features subset wins and r^CAU = r^M."""
        w = np.array([0.5, 0.6])
        b_sk = -0.3
        v = np.array([[1.0, 1.0]])
        theta = 0.0
        assert np.isclose(
            pulled_back_l2_triangle(v, w, b_sk, theta)[0],
            ehyaei_raw(v, w, b_sk)[0],
        )

    def test_pullback_beats_ehyaei_when_propagation_aligned(self):
        """|w₁ + θ w₂| > ‖w‖₂ → single-lever X₁ strictly cheaper than both."""
        w = np.array([0.474961, 0.542195])
        b_sk = -1.269163
        v = np.array([[0.342046, 0.971993]])
        theta = 0.5
        assert pulled_back_l2_triangle(v, w, b_sk, theta)[0] < ehyaei_raw(v, w, b_sk)[0]

    def test_degenerate_weights_skips(self):
        """|w₁ + θw₂| ≈ 0 AND |w₂| ≈ 0 AND ‖w‖ ≈ 0 → recourse infeasible."""
        w = np.array([1e-12, 1e-12])
        b_sk = -0.5
        v = np.array([[0.0, 0.0]])
        with pytest.raises(pytest.skip.Exception):
            pulled_back_l2_triangle(v, w, b_sk, theta=0.5)

    # -- Subset-dominance coverage (Phase 3): exercise each branch as the argmin. ----
    # Note: {X₂} can NEVER be the strict winner. Its weight is |w₂|, and the {X₁,X₂}
    # weight is ‖w‖₂ = sqrt(w₁²+w₂²) ≥ |w₂| for ALL w — so {X₁,X₂} dominates {X₂}
    # everywhere. The realizable winners are {X₁} (θ-propagation aligned) and {X₁,X₂}
    # (θ weak or the lever opposing). The requested "{X₂} wins" case is geometrically
    # impossible in this formulation; the dominated-branch test below locks that.

    def test_subset_winner_x1_when_propagation_aligned(self):
        """{X₁} wins when |w₁ + θ w₂| is the largest effective weight."""
        w = np.array([0.474961, 0.542195])
        theta = 0.5
        assert winning_subset_triangle(w, theta) == frozenset({"X1"})

    def test_subset_winner_both_when_theta_zero(self):
        """{X₁, X₂} wins when θ ≈ 0: the single lever loses to the diagonal ‖w‖."""
        w = np.array([0.5, 0.6])
        theta = 0.0
        assert winning_subset_triangle(w, theta) == frozenset({"X1", "X2"})

    def test_subset_winner_both_when_lever_opposes(self):
        """{X₁, X₂} wins when a negative w₁ makes θ-propagation fight the X₁ lever,
        so |w₁ + θ w₂| shrinks below ‖w‖ — the nominal '{X₂} wins' case,
        which in fact resolves to the dominating joint subset, not {X₂}."""
        w = np.array([-0.1, 0.6])
        theta = 0.5
        assert winning_subset_triangle(w, theta) == frozenset({"X1", "X2"})

    def test_x2_subset_is_dominated_and_never_wins(self):
        """‖w‖₂ ≥ |w₂| for all w → the {X₂} weight is never the max, so {X₂} is never
        the L2 winner. Lock this so the dominated branch cannot silently start winning."""
        rng = np.random.default_rng(0)
        for _ in range(1000):
            w = rng.normal(size=2)
            theta = float(rng.normal())
            weights = effective_weights_triangle(w, theta)
            assert weights[frozenset({"X1", "X2"})] >= weights[frozenset({"X2"})] - 1e-12
            assert winning_subset_triangle(w, theta) != frozenset({"X2"})
