"""Tests for the concrete COLLIDER SCMs (the SCM specification).

Collider (a′): A → X1, A → X2, A → X3, X1 → X3 ← X2. Linear family, both attribute
regimes (additive control, effect-modifying treatment). Eight groups:

  1 — hand-derived counterfactual GATE, three intervention patterns (the primary
      gate — the collider's new two-parent-plus-root abduction pattern).
  2 — abduction round-trip on X3's two-parent-plus-root equation.
  3 — collider conditional-independence (v-structure) structure.
  4 — form-spec structure (interaction column present/absent; β_A in both).
  5 — S(g) flatness guard (PS-3 conformance): exact equality under additive,
      inequality under effect-modification.
  6 — induced-properties report (marginal variances + base rate vs the classifier target).
  7 — pipeline smoke, L0 + L2, both regimes; aggregation layer unchanged.
  8 — G1 pulled-back anchor on the collider L2 cell (topology-correct, independent).

True coefficients are read PROGRAMMATICALLY from the builder-function defaults
(the `structural_g` pattern of the G1 anchor test), except in the hand-derived
gate (Group 1), which constructs with explicit coefficients so the pen-and-paper
arithmetic is self-contained — mirroring tests/test_triangle_scm.py Group 1.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from icknowledge.classifier import build_classifier
from icknowledge.descriptor import s_of_g
from icknowledge.recourse.action_space import build_action_space
from icknowledge.recourse.anchor import (
    closed_form_min_cost,
    grid_diagonal_step,
    grid_span_limit,
)
from icknowledge.recourse.model_conditions import L2TrueSCMModel
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import collider_form_spec, make_linear_collider
from icknowledge.utils.config import load_config

_REGIMES = ("additive", "effect_modifying")


def _coeffs() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_linear_collider).parameters
    keys = ("a1", "a2", "beta", "gamma", "eta", "beta_a", "c1", "c2", "c3", "sigma")
    return {k: float(params[k].default) for k in keys}


# ===========================================================================
# Group 1 — HAND-DERIVED counterfactual gate (THE gate; three patterns)
# ===========================================================================
#
# Fixed builder: make_linear_collider(regime, a1=2, a2=2, beta=0.5, gamma=0.3,
#   eta=0.5, beta_a=1.0, c1=c2=c3=0, sigma=1). Factual: A=1, X1=3, X2=4, X3=5.
#
# X3 = beta*X1 + gamma*(A*X1) + eta*X2 + beta_a*A + eps3   (eff-mod)
# X3 = beta*X1 +                eta*X2 + beta_a*A + eps3   (additive; gamma omitted)
#
# Abduction at the factual (observed X3=5):
#   eff-mod   1.5 + 0.9 + 2.0 + 1.0 + eps3 = 5.4 + eps3  =>  eps3 = -0.4
#   additive  1.5 +       2.0 + 1.0 + eps3 = 4.5 + eps3  =>  eps3 = +0.5
#
# The three do-patterns exercise three different paths through `counterfactual`:
#   (i)   do(X1)      — X2 UNCHANGED (the collider's defining property; on the
#                       triangle do(X1) propagates to X2). X3 changes by
#                       (beta + gamma*A)*Δx1.
#   (ii)  do(X1, X2)  — X3 changes by (beta+gamma*A)*Δx1 + eta*Δx2 (both parents).
#   (iii) do(X3)      — X1, X2 UNCHANGED (a do on a collider leaves its parents).


def _gate_factual() -> pd.DataFrame:
    return pd.DataFrame({"A": [1.0], "X1": [3.0], "X2": [4.0], "X3": [5.0]})


def _gate_scm(regime: str):
    return make_linear_collider(
        regime, a1=2.0, a2=2.0, beta=0.5, gamma=0.3, eta=0.5, beta_a=1.0, sigma=1.0
    )


def test_gate_do_x1_only_effect_modifying_leaves_x2_untouched():
    # do(X1 := 10), A=1. X2 must stay 4 (collider); X3 = (beta+gamma*A)*Δx1 term.
    #   X3 = 0.5*10 + 0.3*(1*10) + 0.5*4 + 1*1 + (-0.4) = 5 + 3 + 2 + 1 - 0.4 = 10.6
    #   ΔX3 = (0.5+0.3)*(10-3) = 0.8*7 = 5.6  =>  5 + 5.6 = 10.6  ✓
    scm = _gate_scm("effect_modifying")
    cf = scm.counterfactual(_gate_factual(), {"X1": 10.0})
    np.testing.assert_allclose(cf["A"].to_numpy(), [1.0])
    np.testing.assert_allclose(cf["X1"].to_numpy(), [10.0])
    np.testing.assert_allclose(cf["X2"].to_numpy(), [4.0])  # UNCHANGED — the collider property
    np.testing.assert_allclose(cf["X3"].to_numpy(), [10.6])


def test_gate_do_x1_only_additive_leaves_x2_untouched():
    #   X3 = 0.5*10 + 0.5*4 + 1*1 + 0.5 = 5 + 2 + 1 + 0.5 = 8.5 ; ΔX3 = 0.5*7 = 3.5
    scm = _gate_scm("additive")
    cf = scm.counterfactual(_gate_factual(), {"X1": 10.0})
    np.testing.assert_allclose(cf["X2"].to_numpy(), [4.0])  # UNCHANGED
    np.testing.assert_allclose(cf["X3"].to_numpy(), [8.5])


def test_gate_do_x1_and_x2_jointly_both_parent_contributions():
    # do(X1 := 10, X2 := 9), A=1. X3 changes by (beta+gamma*A)*Δx1 + eta*Δx2.
    #   eff-mod:  X3 = 0.5*10 + 0.3*10 + 0.5*9 + 1 - 0.4 = 5 + 3 + 4.5 + 1 - 0.4 = 13.1
    #             Δ = 0.8*7 + 0.5*5 = 5.6 + 2.5 = 8.1  =>  5 + 8.1 = 13.1  ✓
    #   additive: X3 = 0.5*10 + 0.5*9 + 1 + 0.5 = 5 + 4.5 + 1.5 = 11.0
    #             Δ = 0.5*7 + 0.5*5 = 3.5 + 2.5 = 6.0  =>  5 + 6.0 = 11.0  ✓
    cf_em = _gate_scm("effect_modifying").counterfactual(_gate_factual(), {"X1": 10.0, "X2": 9.0})
    np.testing.assert_allclose(cf_em["X3"].to_numpy(), [13.1])
    cf_add = _gate_scm("additive").counterfactual(_gate_factual(), {"X1": 10.0, "X2": 9.0})
    np.testing.assert_allclose(cf_add["X3"].to_numpy(), [11.0])


def test_gate_do_x3_directly_leaves_parents_untouched():
    # do(X3 := 100): intervening on the collider must not disturb X1, X2 (or A).
    for regime in _REGIMES:
        cf = _gate_scm(regime).counterfactual(_gate_factual(), {"X3": 100.0})
        np.testing.assert_allclose(cf["A"].to_numpy(), [1.0])
        np.testing.assert_allclose(cf["X1"].to_numpy(), [3.0])  # UNCHANGED
        np.testing.assert_allclose(cf["X2"].to_numpy(), [4.0])  # UNCHANGED
        np.testing.assert_allclose(cf["X3"].to_numpy(), [100.0])


# ===========================================================================
# Group 2 — abduction round-trip on the two-parent-plus-root equation
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_abduction_recovers_true_noise_including_x3(regime):
    """û₃ = x₃ − (β₀+β·x₁+γ·A·x₁+η·x₂+β_A·A) recovers the true ε₃ exactly, and the
    abduct→recompute identity reproduces the factual to machine precision. This is
    where an arithmetic error in X3's two-parent-plus-root equation would hide."""
    scm = make_linear_collider(regime)
    x = scm.sample(3000, seed=5)
    u_abd = scm.abduct(x)
    u_direct = scm.sample_noise(3000, seed=5)  # same seed → identical per-node noise
    for node in ("A", "X1", "X2", "X3"):
        np.testing.assert_allclose(
            u_abd[node].to_numpy(), u_direct[node].to_numpy(), atol=1e-12,
            err_msg=f"{regime}/{node}: abducted noise != sampled noise",
        )
    # abduct-and-recompute with an EMPTY intervention returns the factual exactly.
    identity_cf = scm.counterfactual(x, {})
    np.testing.assert_allclose(identity_cf.to_numpy(), x.to_numpy(), atol=1e-12)


# ===========================================================================
# Group 3 — collider conditional-independence (v-structure) structure
# ===========================================================================


def _partial_corr(cols: dict[str, np.ndarray], x: str, y: str, given: list[str]) -> float:
    """Linear partial correlation of x, y given the ``given`` set (via residualization)."""
    n = len(cols[x])
    design = np.column_stack([np.ones(n)] + [cols[g] for g in given])

    def _resid(col: str) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, cols[col], rcond=None)
        return cols[col] - design @ beta

    return float(np.corrcoef(_resid(x), _resid(y))[0, 1])


@pytest.mark.parametrize("regime", _REGIMES)
def test_v_structure_conditional_independence(regime):
    """Within group (conditioning on A): X1 ⊥ X2 (partial corr ≈ 0), but
    X1 ⊥̸ X2 | X3, A — conditioning on the collider X3 INDUCES dependence. This is
    the v-structure property PC will rely on later; asserting it now catches a
    mis-specified builder before L1-discovered depends on it. The marginal
    dependence of X1, X2 is induced by the shared root A (not an X1–X2 edge)."""
    scm = make_linear_collider(regime)
    data = scm.sample(40000, seed=101)
    cols = {c: data[c].to_numpy(dtype=float) for c in data.columns}

    # X1 ⊥ X2 | A  (they share only the root A, no X1–X2 edge → within-group ≈ 0).
    pc_given_a = _partial_corr(cols, "X1", "X2", ["A"])
    assert abs(pc_given_a) < 0.02, (
        f"{regime}: pcorr(X1,X2|A)={pc_given_a:.4f} — X1 and X2 must be conditionally "
        "independent given A (collider parents share only the root)."
    )
    # X1 ⊥̸ X2 | X3, A  (conditioning on the collider induces dependence; ≈ −0.19).
    pc_given_x3_a = _partial_corr(cols, "X1", "X2", ["X3", "A"])
    assert abs(pc_given_x3_a) > 0.1, (
        f"{regime}: pcorr(X1,X2|X3,A)={pc_given_x3_a:.4f} — conditioning on the collider "
        "X3 must INDUCE dependence between its parents (the v-structure signal)."
    )
    # Marginal dependence is real and root-induced (shared A), not zero.
    marginal = float(np.corrcoef(cols["X1"], cols["X2"])[0, 1])
    assert marginal > 0.5, (
        f"{regime}: marginal corr(X1,X2)={marginal:.4f} — the shared root A must induce "
        "marginal dependence between the collider parents."
    )


# ===========================================================================
# Group 4 — form-spec structure (mirrors the triangle's Test 3)
# ===========================================================================


def test_form_spec_interaction_column_present_only_under_effect_modification():
    """X3's FormSpec carries the (A,X1) interaction under effect_modifying and NOT
    under additive (hard-zero by omission). β_A — the A main-effect term — is
    present in BOTH regimes (A ∈ parents of X3), so the A→X3 edge is matched."""
    add = collider_form_spec("additive")
    em = collider_form_spec("effect_modifying")

    # X3 parent (main-effect) set is identical across regimes and INCLUDES A (β_A).
    assert add["X3"].parents == ("A", "X1", "X2") == em["X3"].parents, (
        "X3 must carry A as a main-effect parent (β_A) in BOTH regimes so the graph "
        "is matched across regimes."
    )
    # Interaction (A·X1) present only under effect-modification.
    assert add["X3"].interactions == (), "additive X3 must carry NO interaction (γ hard-zero)."
    assert em["X3"].interactions == (("A", "X1"),), (
        "effect-mod X3 must carry the (A,X1) interaction."
    )
    # X1, X2 identical in both regimes; single parent {A}, no interaction possible.
    for regime_spec in (add, em):
        assert regime_spec["X1"].parents == ("A",) and regime_spec["X1"].interactions == ()
        assert regime_spec["X2"].parents == ("A",) and regime_spec["X2"].interactions == ()


def test_form_spec_matches_true_graph_parent_sets():
    """The form spec's declared parent sets equal the true builder's graph parent
    sets (the precondition build_estimated_scm enforces for L1-oracle)."""
    for regime in _REGIMES:
        scm = make_linear_collider(regime)
        spec = collider_form_spec(regime)
        for node in ("X1", "X2", "X3"):
            assert set(spec[node].parents) == set(scm.parents(node)), (
                f"{regime}/{node}: form parents {spec[node].parents} != true graph parents "
                f"{sorted(scm.parents(node))}"
            )


# ===========================================================================
# Group 5 — S(g) flatness guard (PS-3 conformance check)
# ===========================================================================
#
# S(g) := sqrt( E_{u|A=g} ‖ M̃(g;u) − 𝟙 ‖²_F ),  M(g) = (I − B(g))⁻¹, restricted to
# rows V_h ∩ Desc(A) and columns S ∩ Desc(A). The linear family is constant in u,
# so the RMS reduces to the deterministic Frobenius value (PS-3).
#
# CRUCIAL — A is a ROOT and is EXCLUDED from B's row and column index sets, since
# A ∉ Desc(A); B is formed over the ENDOGENOUS variables {X1, X2, X3} only. On the
# collider the only endogenous edges are X1→X3 and X2→X3, so B(g) is nilpotent and
#   M(g) − I = B(g) = [[0,0,0],[0,0,0],[β+γ·g, η, 0]]   (rows/cols = X1,X2,X3)
# giving the closed form  S(g) = √((β + γ·g)² + η²).  Under the additive control
# (γ omitted) this is √(β² + η²) for BOTH groups — flat by construction, which is
# exactly what H4's control arm requires. Admitting A would fold its main effect
# β_A into S(g), conflating "A modifies one endogenous edge" with "A shifts levels".
#
# S(g) is computed by the production descriptor module this test calls. The
# analytic closed form in scripts/collider_spec_verification.py is an
# INDEPENDENT reimplementation (a redundant cross-check).


def test_s_of_g_flat_under_additive_and_distinct_under_effect_modification():
    d = _coeffs()
    v_h = {"X1", "X2", "X3"}       # group-blind classifier features h(X1,X2,X3)
    actionable = {"X1", "X2", "X3"}  # 3-D action space (PS-8)

    # Additive control: S(−1) == S(+1) EXACTLY (to machine precision) — flat by
    # construction. Closed form √(β² + η²) for both groups.
    scm_add = make_linear_collider("additive")
    s_neg = s_of_g(scm_add, -1.0, V_h=v_h, S=actionable)
    s_pos = s_of_g(scm_add, +1.0, V_h=v_h, S=actionable)
    expected_add = np.hypot(d["beta"], d["eta"])
    assert abs(s_neg - s_pos) < 1e-12, (
        f"additive S(−1)={s_neg} != S(+1)={s_pos} — the control arm MUST be flat "
        "across groups (H4 control). A leak here means A entered the propagation "
        "sub-block or γ is not hard-zero. Closed form is √(β²+η²)="
        f"{expected_add:.6f} for BOTH groups."
    )
    np.testing.assert_allclose([s_neg, s_pos], [expected_add, expected_add], atol=1e-12)

    # Effect-modifying: S(−1) != S(+1), both finite. Closed form √((β+γ·g)²+η²).
    scm_em = make_linear_collider("effect_modifying")
    e_neg = s_of_g(scm_em, -1.0, V_h=v_h, S=actionable)
    e_pos = s_of_g(scm_em, +1.0, V_h=v_h, S=actionable)
    exp_neg = np.hypot(d["beta"] + d["gamma"] * (-1.0), d["eta"])
    exp_pos = np.hypot(d["beta"] + d["gamma"] * (+1.0), d["eta"])
    assert np.isfinite(e_neg) and np.isfinite(e_pos)
    assert abs(e_neg - e_pos) > 1e-6, (
        f"effect-modifying S(−1)={e_neg:.6f} == S(+1)={e_pos:.6f} — the treatment arm "
        "MUST be group-distinct (A modifies X3's X1-coefficient)."
    )
    np.testing.assert_allclose([e_neg, e_pos], [exp_neg, exp_pos], atol=1e-12)


# ===========================================================================
# Group 6 — induced-properties report (marginal variances + base rate vs the classifier
# construction)
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_induced_variances_finite_and_base_rate_in_band(regime):
    """Marginal variances of X1, X2, X3 are finite (X3's exceeds X1/X2 — two
    endogenous parents plus a root term plus noise), and the realized classifier
    base rate lands within the classifier band. A base-rate miss is a PS-2 CALIBRATION
    issue (register-logged recalibration), NOT a silent coefficient retune."""
    scm = make_linear_collider(regime)
    data = scm.sample(20000, seed=int(np.random.SeedSequence(20260710).generate_state(1)[0]))
    variances = {c: float(np.var(data[c].to_numpy())) for c in ("X1", "X2", "X3")}
    assert all(np.isfinite(v) and v > 0 for v in variances.values()), variances
    # X3 has strictly the largest induced marginal variance (two parents + root + noise).
    assert variances["X3"] > variances["X1"] and variances["X3"] > variances["X2"], (
        f"{regime}: X3 marginal variance {variances['X3']:.3f} must exceed X1/X2 "
        f"({variances['X1']:.3f}/{variances['X2']:.3f}) — two endogenous parents + root."
    )

    cfg = load_config("configs/recourse_collider.yaml")
    target = float(cfg.classifier.base_rate_target)
    tol = float(cfg.classifier.base_rate_tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = build_classifier(scm, cfg)
    assert abs(clf.realized_base_rate - target) <= tol, (
        f"{regime}: realized base rate {clf.realized_base_rate:.3f} outside "
        f"{target}±{tol} — this is a PS-2 CALIBRATION issue (the collider's higher X3 "
        "marginal variance shifted the label signal). Remedy is a register-logged "
        "recalibration, NOT a silent coefficient retune."
    )


# ===========================================================================
# Group 7 + 8 — pipeline smoke and G1 anchor (shared small-scale fixture)
# ===========================================================================
#
# The full-pool run at triangle-parity resolution (161³ ≈ 4.2M candidates) is
# infeasible (≈14 s / individual in L2). These correctness checks run at a small
# resolution/N — the faithful-resolution wall-clock is reported by the driver's
# spec-verification artifact, not asserted here.


def _small_cfg():
    cfg = load_config("configs/recourse_collider.yaml")
    cfg.classifier.N = 600
    cfg.classifier.min_neg_per_group = 30
    cfg.recourse.grid.resolution = 21
    return cfg


@pytest.fixture(scope="module")
def collider_runs():
    cfg = _small_cfg()
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime in _REGIMES:
            runs[regime] = run_regime(
                cfg,
                regime,
                conditions=("L0", "L2"),
                scm_builder=make_linear_collider,
                form_spec_fn=collider_form_spec,
                run_anchor=True,  # anchor.py now enumerates all 2^k − 1 subsets (correct for k=3)
            )
    return cfg, runs


def test_builtin_g1_anchor_passes_on_collider_both_regimes(collider_runs):
    """The driver's built-in G1 anchor check passes on the collider L2 cell in
    BOTH regimes, within the reachability-filter tolerance."""
    _cfg, runs = collider_runs
    for regime, run in runs.items():
        assert run.anchor is not None and run.anchor.passed, (
            f"{regime}: built-in G1 anchor failed on the collider — "
            f"{run.anchor.format() if run.anchor else 'anchor not run'}"
        )


def test_pipeline_smoke_produces_tables_and_aggregates_unchanged(collider_runs, tmp_path):
    """End-to-end L0 + L2, both regimes: per-individual scoring CSVs roll up through
    the EXISTING aggregation layer with topology="collider" and ZERO modifications
    to icknowledge/aggregation/ (the layer is topology-agnostic; the SCM specification)."""
    from icknowledge.aggregation import CellKeys, aggregate_run
    from icknowledge.aggregation.schema import by_cell_columns, by_group_columns

    _cfg, runs = collider_runs
    for regime, run in runs.items():
        assert set(run.table["condition"].unique()) == {"L0", "L2"}
        # X3 is now a scored feature: delta/factual/cf columns exist for all three.
        for f in ("X1", "X2", "X3"):
            assert f"delta_{f}" in run.table.columns and f"realized_cf_{f}" in run.table.columns
        for condition in ("L0", "L2"):
            sub = run.table[run.table["condition"] == condition]
            sub.to_csv(tmp_path / f"scoring_table_{regime}_{condition}.csv", index=False)

        # aggregate_run must succeed unchanged with topology="collider".
        cell = CellKeys(topology="collider", family="linear", regime=regime, seed=20260710)
        by_group_path, by_cell_path = aggregate_run(tmp_path, tmp_path / "agg", cell)
        by_group = pd.read_csv(by_group_path)
        by_cell = pd.read_csv(by_cell_path)
        # [PS-4 note (1)] Asserted against the schema module's
        # answer FOR THIS ARITY, not against the frozen three-condition tuple.
        # This cell scores TWO conditions, so its common-found columns carry the
        # `_twoway` denominator label. The property under test: no schema DRIFT
        # between what the layer writes and what schema.py declares.
        assert list(by_group.columns) == list(by_group_columns(2))
        assert list(by_cell.columns) == list(by_cell_columns(2))
        assert (by_group["topology"] == "collider").all()
        assert (by_cell["topology"] == "collider").all()
        assert set(by_group["condition"].unique()) == {"L0", "L2"}
        assert set(by_group["group"].unique()) == {-1, 1}


# --- Group 8: topology-correct pulled-back G1 anchor (independent reimplementation) ---
#
# anchor.py's closed_form_min_cost enumerates ALL 2^k − 1 non-empty acted subsets,
# so it is correct on the 3-var collider. This test keeps a FULLY INDEPENDENT
# pulled-back closed form (all 7 subsets, built by hand and tethered to the realizer
# by PROBING the true SCM for p = β+γ·A (X1→X3) and η (X2→X3)), validating that
# enumeration from a separate code path. Pulled-back
# r^CAU_L2 = |h(x)| / max_S ‖(∂ feature-displacement/∂δ_S)ᵀ w‖₂.


def _effective_weights_collider(w: np.ndarray, p: float, eta: float) -> dict[frozenset, float]:
    """‖J_Sᵀw‖₂ for every non-empty acted subset S ⊆ {X1,X2,X3} on the collider (a′).

    Feature axes are (X1, X2, X3) = w indices (0, 1, 2). For a subset S:
      • acted X3 is SEVERED (do severs X1,X2 → X3), contributing w₃ on its own axis;
      • a FREE X3 propagates: it adds (β+γ·A)=p per unit δ_X1 and η per unit δ_X2.
    """
    names = ("X1", "X2", "X3")
    weights: dict[frozenset, float] = {}
    for mask in range(1, 8):
        S = {names[i] for i in range(3) if mask & (1 << i)}
        grad: list[float] = []
        if "X3" in S:  # X3 severed
            if "X1" in S:
                grad.append(w[0])
            if "X2" in S:
                grad.append(w[1])
            grad.append(w[2])
        else:  # X3 free → propagates from acted parents
            if "X1" in S:
                grad.append(w[0] + w[2] * p)
            if "X2" in S:
                grad.append(w[1] + w[2] * eta)
        weights[frozenset(S)] = float(np.linalg.norm(grad, ord=2))
    return weights


def test_g1_pulled_back_anchor_on_collider_l2_cell(collider_runs):
    """The Ehyaei pulled-back anchor is topology-agnostic in PRINCIPLE — verify the
    collider L2 brute-force matches the full-subset pulled-back closed form within
    the reachability tolerance, with the reachability filter applied. A failure here while
    the triangle passes is a pullback/Jacobian issue specific to the 3-var structural
    map and must be reported, not tolerated (do NOT loosen the tolerance)."""
    cfg, runs = collider_runs
    run = runs["additive"]  # additive cell (p = β for both groups), parallel to triangle G1
    clf = run.classifier
    model = clf.model
    feats = clf.feature_names  # ['X1','X2','X3']
    w = model.coef_[0].astype(float)

    scm = make_linear_collider("additive")
    acted = list(cfg.recourse.actionable)
    true_model = L2TrueSCMModel(scm, feats, acted)

    action_space = build_action_space(
        clf.dataset, acted, mode=str(cfg.recourse.grid.mode),
        k=float(cfg.recourse.grid.k), resolution=int(cfg.recourse.grid.resolution),
    )
    grid_step = grid_diagonal_step(action_space.axis_grids)
    span_limit = grid_span_limit(action_space.axis_grids)
    tolerance = float(cfg.recourse.anchor.tolerance)
    abs_threshold = tolerance * grid_step

    l2 = run.table[run.table["condition"] == "L2"].reset_index(drop=True)

    excesses = []
    n_eligible = 0
    for _, row in l2.iterrows():
        if not (row["found"] and row["believed_validity"] == 1):
            continue
        idx = int(row["index"])
        factual = clf.dataset.iloc[idx]
        ff = factual[feats].to_numpy(dtype=float)
        h = float(model.decision_function(ff.reshape(1, -1))[0])
        raw_reach = abs(h) / float(np.linalg.norm(w, ord=2))
        if raw_reach > span_limit:  # the reachability filter
            continue
        # Probe the realizer for the propagation coefficients (tether, not self-check):
        # a unit do(X1) displaces X3 by p=β+γ·A; a unit do(X2) displaces X3 by η.
        p_probe = float((true_model.predict(factual, {"X1": 1.0}) - ff)[feats.index("X3")])
        eta_probe = float((true_model.predict(factual, {"X2": 1.0}) - ff)[feats.index("X3")])
        weights = _effective_weights_collider(w, p_probe, eta_probe)
        max_eff = max(weights.values())
        if max_eff < 1e-8:
            continue
        closed_form = abs(h) / max_eff
        # Redundant cross-check: anchor.py's completed enumeration must agree with this
        # independent reimplementation to numerical precision (validates its subset set
        # and per-subset effective weight from a separate code path).
        cf_anchor, _ = closed_form_min_cost(model, true_model, factual)
        assert np.isclose(cf_anchor, closed_form, rtol=1e-9, atol=1e-9), (
            f"anchor.py closed form {cf_anchor:.9f} disagrees with the independent "
            f"reimplementation {closed_form:.9f} on the collider — enumeration mismatch."
        )
        excess = float(row["believed_cost"]) - closed_form
        excesses.append(excess)
        n_eligible += 1

    assert n_eligible > 0, "no reachable eligible individuals for the collider G1 anchor."
    excesses = np.array(excesses)
    # Lower bound: a discrete-grid minimizer cannot undercut the continuous optimum.
    assert (excesses >= -1e-8).all(), (
        f"collider L2 brute-force fell BELOW the pulled-back anchor (max undercut "
        f"{-excesses.min():.6e}) — a subset-enumeration / Jacobian bug specific to the "
        "3-variable collider map. Report, do not tolerate."
    )
    # Upper bound: grid discretization, bounded by the tolerance × diagonal step.
    assert (excesses <= abs_threshold).all(), (
        f"collider L2 brute-force exceeds pullback + tolerance: max excess "
        f"{excesses.max():.6e} > {abs_threshold:.6e} (grid_step={grid_step:.6f}). If a "
        "3-var map issue, report it — do NOT loosen the tolerance."
    )
