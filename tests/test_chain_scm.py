"""Tests for the concrete CHAIN SCMs (the chain SCM specification).

The chain is the ONLY topology whose endogenous block composes: X₁ → X₂ → X₃ gives
(I − B)⁻¹ a nonzero (3,1) entry b₂₁·b₃₂. PS-3 convention (1) is written around this
cell, so several gates below are chain-specific rather than ported.

Covers, per the chain SCM specification:

  1. HAND-DERIVED counterfactual gate — the composed path do(X1) → X2 → X3, the
     sharpest contrast with the collider, where do(X1) leaves X2 untouched.
  2. ABDUCTION round-trip on both endogenous equations.
  3. STRUCTURAL-EQUATION smoke — group-conditional means and variances against
     closed-form analytic expectations, all four cells.
  4. NON-SATURATION empirical check (NLG ONLY — see below).
  5. FORM TEMPLATE — the A·X₁ column appears in X₂'s equation only under effect
     modification; X₃ is main-effects-only in BOTH regimes.
  6. S(g) LINEAR-CHAIN ANALYTIC GATE — the closed-form ‖·‖²_F below against
     production `s_of_g` to full precision.
  7. S(g) NLG FD REGRESSION at h=1e-4 — the NLG family specification's tether.
  8. S(g) additive flatness / effect-modifying sign margin, both families.
  9. PULLED-BACK G1 ANCHOR on the linear × LR × L2 chain cell, with the chain's
     own structural map (independent of anchor.py).
 10. AGGREGATION SCHEMA — topology="chain" rolls up with no schema drift.

NON-SATURATION IS NLG-ONLY: the specification's constraint binds the TANH INPUTS
(X₁ feeding tanh in X₂'s equation, X₂ feeding tanh in X₃'s). The linear chain has
no tanh, so the constraint does not apply; its marginals are reported as
instrumentation instead.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from icknowledge.aggregation import CellKeys, aggregate_run
from icknowledge.classifier import build_classifier
from icknowledge.descriptor import S_OF_G_MC_SAMPLES, s_of_g
from icknowledge.recourse.action_space import build_action_space
from icknowledge.recourse.anchor import grid_diagonal_step, grid_span_limit
from icknowledge.recourse.model_conditions import L2TrueSCMModel
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import (
    chain_form_spec,
    make_linear_chain,
    make_nonlinear_gaussian_chain,
    nlg_chain_form_spec,
)
from icknowledge.utils.config import load_config

# ONE tether and ONE tolerance across all three NLG topologies: the NLG-triangle S(g) gate
# owns the central-FD construction (which reads nothing from `jacobian_fn` and
# differentiates the realizer itself), the 1e-4 absolute tolerance, and the 0.02
# MC-noise envelope. Importing them is what keeps the chain from drifting to a
# looser gate, exactly as the NLG-collider gate imports them.
from tests.test_s_of_g_nlg import (
    _FD_ATOL,
    _MC_NOISE_ENVELOPE,
    _finite_difference_jacobian,
)

_REGIMES = ("additive", "effect_modifying")
_GROUPS = (-1.0, 1.0)

# Topology-defined: the classifier is group-blind over A's descendants (the classifier
# construction) and
# the actionable set excludes A, so V_h == S == {X1, X2, X3} on the chain —
# which is why PS-3 convention (3)'s "identity removed" reduces to the ordinary
# identity matrix here (the chain SCM specification).
_CHAIN_SETS = {"V_h": ("X1", "X2", "X3"), "S": ("X1", "X2", "X3")}

#: Sign-margin bar on the effect-modifying arm: |ΔS| must clear this many MC SEs.
#: [the rule of thumb; PS-2's gate condition (ii)] Deliberately a slack floor,
#: so the assertion fails on a BROKEN cell (a sign collapsed toward noise, the
#: failure mode PS-3 convention 4 exists to catch) and not on a PS-2 retune that
#: legitimately moves the level.
_SIGN_MARGIN_MC_SE = 5.0

#: The chain SCM specification's non-saturation band and the fraction of mass required inside it.
_NON_SATURATION_BAND = 1.5
_NON_SATURATION_FRACTION = 0.95


def _linear_coeffs() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_linear_chain).parameters
    keys = ("a", "beta", "gamma", "beta_a2", "delta", "beta_a3", "sigma1", "sigma2", "sigma3")
    return {k: float(params[k].default) for k in keys}


def _nlg_coeffs() -> dict[str, float]:
    params = inspect.signature(make_nonlinear_gaussian_chain).parameters
    keys = ("a", "beta", "gamma", "beta_a2", "delta", "beta_a3", "sigma1", "sigma2", "sigma3")
    return {k: float(params[k].default) for k in keys}


# ===========================================================================
# Group 1 — HAND-DERIVED counterfactual gate (THE gate; the composed path)
# ===========================================================================
#
# Fixed builder: make_linear_chain(regime, a=1, beta=0.5, gamma=0.3, beta_a2=0.5,
#   delta=0.5, beta_a3=0.5, sigma*=0.5). Factual: A=1, X1=3, X2=4, X3=5.
#
# X2 = beta*X1 + gamma*(A*X1) + beta_a2*A + eps2   (eff-mod)
# X2 = beta*X1 +                beta_a2*A + eps2   (additive; gamma omitted)
# X3 = delta*X2 + beta_a3*A + eps3                 (BOTH regimes — gamma is upstream)
#
# Abduction at the factual (observed X2=4, X3=5):
#   eff-mod   X2 = 1.5 + 0.9 + 0.5 + eps2 = 2.9 + eps2  =>  eps2 = +1.1
#   additive  X2 = 1.5 +       0.5 + eps2 = 2.0 + eps2  =>  eps2 = +2.0
#   both      X3 = 2.0 +       0.5 + eps3 = 2.5 + eps3  =>  eps3 = +2.5
#
# The do-patterns exercise the paths `counterfactual` must get right:
#   (i)   do(X1)      — X2 AND X3 BOTH move; ΔX3 = delta*(beta + gamma*A)*ΔX1, the
#                       COMPOSED PATH, the (3,1) entry.
#   (ii)  do(X2)      — X1 unchanged (upstream); X3 moves by delta*ΔX2.
#   (iii) do(X1, X2)  — X2's own do-value WINS over the X1 propagation (severance),
#                       so ΔX3 depends on ΔX2 only.
#   (iv)  do(X3)      — X1, X2 unchanged (terminal node).


def _gate_factual() -> pd.DataFrame:
    return pd.DataFrame({"A": [1.0], "X1": [3.0], "X2": [4.0], "X3": [5.0]})


def _gate_scm(regime: str):
    return make_linear_chain(
        regime, a=1.0, beta=0.5, gamma=0.3, beta_a2=0.5,
        delta=0.5, beta_a3=0.5, sigma1=0.5, sigma2=0.5, sigma3=0.5,
    )


def test_gate_do_x1_only_propagates_along_the_composed_path_effect_modifying():
    # do(X1 := 10), A=1. b21 = beta + gamma*A = 0.8.
    #   X2 = 0.5*10 + 0.3*10 + 0.5 + 1.1 = 5 + 3 + 0.5 + 1.1 = 9.6
    #        ΔX2 = 0.8*(10-3) = 5.6  =>  4 + 5.6 = 9.6  ✓
    #   X3 = 0.5*9.6 + 0.5 + 2.5 = 4.8 + 3.0 = 7.8
    #        ΔX3 = delta*b21*ΔX1 = 0.5*0.8*7 = 2.8  =>  5 + 2.8 = 7.8  ✓  (composed)
    scm = _gate_scm("effect_modifying")
    cf = scm.counterfactual(_gate_factual(), {"X1": 10.0})
    np.testing.assert_allclose(cf["A"].to_numpy(), [1.0])
    np.testing.assert_allclose(cf["X1"].to_numpy(), [10.0])
    np.testing.assert_allclose(cf["X2"].to_numpy(), [9.6])  # MOVED — the chain property
    np.testing.assert_allclose(cf["X3"].to_numpy(), [7.8])  # MOVED via X2 — composed path


def test_gate_do_x1_only_propagates_along_the_composed_path_additive():
    #   X2 = 0.5*10 + 0.5 + 2.0 = 7.5 ; ΔX2 = 0.5*7 = 3.5  =>  4 + 3.5 = 7.5  ✓
    #   X3 = 0.5*7.5 + 0.5 + 2.5 = 3.75 + 3.0 = 6.75 ; ΔX3 = 0.5*0.5*7 = 1.75  ✓
    scm = _gate_scm("additive")
    cf = scm.counterfactual(_gate_factual(), {"X1": 10.0})
    np.testing.assert_allclose(cf["X2"].to_numpy(), [7.5])
    np.testing.assert_allclose(cf["X3"].to_numpy(), [6.75])


def test_gate_do_x2_severs_the_upstream_edge():
    # do(X2 := 9): X1 unchanged (upstream of the intervention); X3 moves by
    # delta*ΔX2 = 0.5*(9-4) = 2.5  =>  5 + 2.5 = 7.5. IDENTICAL in both regimes,
    # because severing X1→X2 removes the only path gamma sits on.
    for regime in _REGIMES:
        cf = _gate_scm(regime).counterfactual(_gate_factual(), {"X2": 9.0})
        np.testing.assert_allclose(cf["X1"].to_numpy(), [3.0])  # UNCHANGED
        np.testing.assert_allclose(cf["X2"].to_numpy(), [9.0])
        np.testing.assert_allclose(cf["X3"].to_numpy(), [7.5])


def test_gate_do_x1_and_x2_jointly_lets_the_do_value_win_over_propagation():
    # do(X1 := 10, X2 := 9): X2's do-value SEVERS the X1→X2 edge, so the X1 change
    # cannot reach X3. X3 must equal the do(X2)-only answer, 7.5 — NOT the composed
    # value. A builder that applied propagation before severance would give 9.6-based
    # arithmetic here and pass every other gate in this file.
    for regime in _REGIMES:
        cf = _gate_scm(regime).counterfactual(_gate_factual(), {"X1": 10.0, "X2": 9.0})
        np.testing.assert_allclose(cf["X2"].to_numpy(), [9.0])
        np.testing.assert_allclose(cf["X3"].to_numpy(), [7.5])


def test_gate_do_x3_directly_leaves_parents_untouched():
    # do(X3 := 100): intervening on the terminal node must not disturb X1, X2 (or A).
    for regime in _REGIMES:
        cf = _gate_scm(regime).counterfactual(_gate_factual(), {"X3": 100.0})
        np.testing.assert_allclose(cf["A"].to_numpy(), [1.0])
        np.testing.assert_allclose(cf["X1"].to_numpy(), [3.0])  # UNCHANGED
        np.testing.assert_allclose(cf["X2"].to_numpy(), [4.0])  # UNCHANGED
        np.testing.assert_allclose(cf["X3"].to_numpy(), [100.0])


# ===========================================================================
# Group 2 — abduction round-trip on BOTH endogenous equations
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_abduction_recovers_true_noise_on_both_endogenous_nodes(regime):
    """û_i = x_i − f_i(pa, 0) recovers the true ε_i exactly at X₂ AND X₃.

    Both endogenous nodes are checked because the chain is the first topology whose
    two endogenous equations are IN SERIES: X₃'s abduction reads X₂'s realized
    value, so an abduction bug at X₂ would be masked at X₃ if only the terminal node
    were checked (the wrong X₂ is still self-consistent downstream).
    """
    scm = make_linear_chain(regime)
    data = scm.sample(500, seed=11)
    u = scm.abduct(data)
    c = _linear_coeffs()
    a_col = data["A"].to_numpy()

    x2_mean = c["beta"] * data["X1"].to_numpy() + c["beta_a2"] * a_col
    if regime == "effect_modifying":
        x2_mean = x2_mean + c["gamma"] * (a_col * data["X1"].to_numpy())
    np.testing.assert_allclose(
        u["X2"].to_numpy(), data["X2"].to_numpy() - x2_mean, atol=1e-12
    )

    x3_mean = c["delta"] * data["X2"].to_numpy() + c["beta_a3"] * a_col
    np.testing.assert_allclose(
        u["X3"].to_numpy(), data["X3"].to_numpy() - x3_mean, atol=1e-12
    )


# ===========================================================================
# Group 3 — structural-equation smoke against ANALYTIC expectations
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_linear_chain_group_means_and_variances_match_closed_form(regime):
    """Group-conditional means and marginal variances against hand algebra.

    Analytic, per group g (all noises independent, mean zero):
        E[X1 | g] = a·g                      Var = σ₁²
        E[X2 | g] = b21(g)·a·g + β_{A,2}·g   Var = b21(g)²σ₁² + σ₂²
        E[X3 | g] = δ·E[X2 | g] + β_{A,3}·g  Var = δ²Var[X2] + σ₃²
    with b21(g) = β + γ·g under effect modification and β under the control. The
    variance recursion is the composed path in second-moment form — it is what makes
    the chain's induced Δ_cost scale its own (PS-8: levels are topology-specific).
    """
    c = _linear_coeffs()
    scm = make_linear_chain(regime)
    data = scm.sample(400_000, seed=13)
    for g in _GROUPS:
        sub = data[data["A"] == g]
        b21 = c["beta"] + (c["gamma"] * g if regime == "effect_modifying" else 0.0)
        m1 = c["a"] * g
        m2 = b21 * m1 + c["beta_a2"] * g
        m3 = c["delta"] * m2 + c["beta_a3"] * g
        v1 = c["sigma1"] ** 2
        v2 = b21**2 * v1 + c["sigma2"] ** 2
        v3 = c["delta"] ** 2 * v2 + c["sigma3"] ** 2
        for node, mean, var in (("X1", m1, v1), ("X2", m2, v2), ("X3", m3, v3)):
            observed = sub[node].to_numpy()
            # 4 SE on the mean; 3% relative on the variance. Loose enough that MC
            # noise never trips it, tight enough that a wrong coefficient does.
            se = np.sqrt(var / observed.size)
            assert abs(observed.mean() - mean) < 4 * se, (
                f"{regime} {node} | A={g:+g}: mean {observed.mean():.4f} vs analytic "
                f"{mean:.4f} (4 SE = {4 * se:.4f})"
            )
            assert abs(observed.var() / var - 1.0) < 0.03, (
                f"{regime} {node} | A={g:+g}: var {observed.var():.4f} vs analytic {var:.4f}"
            )


def test_nlg_chain_is_actually_nonlinear_in_its_endogenous_parents():
    """The NLG chain must NOT pass the descriptor's affineness probe.

    Guards the family boundary from the other side: if a refactor dropped the tanh,
    every S(g) number in this file would still be computed — by the LINEAR branch,
    silently, at a wrong value. `s_of_g` routes on VERIFIED affineness, so this
    asserts the routing input rather than trusting the label.
    """
    from icknowledge.descriptor.s_of_g import _NonAffineEquation, _structural_adjacency

    scm = make_nonlinear_gaussian_chain("effect_modifying")
    with pytest.raises(_NonAffineEquation):
        _structural_adjacency(scm, 1.0, "A")


# ===========================================================================
# Group 4 — NON-SATURATION empirical check (the chain SCM specification; NLG ONLY)
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_nlg_chain_non_saturation_on_both_tanh_inputs(regime):
    """≥95% of X₁ AND X₂ inside [−1.5, 1.5] — the chain SCM specification's constraint, verified
    empirically.

    The chain SCM specification says "Verification is empirical (build-time check on sampled X₁, X₂
    from the
    group-conditional joint)", so this samples the joint rather than evaluating the
    closed-form inequality. Checking BOTH tanh inputs is the chain-specific part: the
    collider's two inputs are independent A-children, but the chain's X₂ is fed by X₁
    THROUGH the composed path, so its marginal compounds the group signal and can
    saturate even when X₁ comfortably does not.

    A failure here is a CONSTRUCTION failure, not a finding: the remedy is an chain SCM
    specification
    coefficient adjustment logged in the register, never a relaxed band here.
    """
    data = make_nonlinear_gaussian_chain(regime).sample(200_000, seed=17)
    for node in ("X1", "X2"):
        inside = float(np.mean(np.abs(data[node].to_numpy()) <= _NON_SATURATION_BAND))
        assert inside >= _NON_SATURATION_FRACTION, (
            f"NLG chain [{regime}] {node}: only {inside:.4f} of mass inside "
            f"±{_NON_SATURATION_BAND} (need ≥ {_NON_SATURATION_FRACTION}). The chain SCM "
            f"specification's "
            "non-saturation constraint is violated — adjust the builder coefficients "
            "and log it in the register; do NOT widen this band."
        )


@pytest.mark.parametrize("regime", _REGIMES)
def test_nlg_chain_group_separation_is_not_sacrificed_to_non_saturation(regime):
    """The other half of the chain SCM specification's separation-vs-saturation balance.

    Non-saturation alone is trivially satisfiable by shrinking `a` to zero — which
    would also null ΔS and make the cell degenerate. This asserts the group means of
    the tanh INPUT are actually separated, so the constraint above cannot be passed
    by collapsing the signal.
    """
    c = _nlg_coeffs()
    data = make_nonlinear_gaussian_chain(regime).sample(200_000, seed=19)
    m_neg = data[data["A"] == -1.0]["X1"].mean()
    m_pos = data[data["A"] == 1.0]["X1"].mean()
    assert abs(m_pos - m_neg) > c["a"], (
        f"NLG chain [{regime}]: X₁ group separation {abs(m_pos - m_neg):.4f} has "
        f"collapsed (expected ≈ 2a = {2 * c['a']:.2f}) — non-saturation was bought "
        "by nulling the signal."
    )


# ===========================================================================
# Group 5 — form template (the L1-oracle form-template-known semantics/the chain SCM specification)
# ===========================================================================


def test_form_spec_interaction_column_present_only_under_effect_modification():
    """A·X₁ enters X₂'s equation under EM only, and X₃ is main-effects-only in BOTH.

    The second half is the chain's structural signature: the collider's γ lands on
    the terminal node, so ITS template gains a column at X₃. Here γ is upstream, so
    X₃'s template is regime-invariant — a template that grew an interaction at X₃
    would be fitting a term the truth does not contain.
    """
    additive, em = chain_form_spec("additive"), chain_form_spec("effect_modifying")
    assert additive["X2"].interactions == ()
    assert em["X2"].interactions == (("A", "X1"),)
    assert additive["X3"].interactions == () == em["X3"].interactions
    assert additive["X1"].interactions == () == em["X1"].interactions


def test_nlg_form_spec_columns_match_the_builder_term_for_term():
    """The NLG template's design columns, by canonical name, against the chain SCM specification's "
    "form."""
    from icknowledge.estimation.forms import term_names

    additive, em = nlg_chain_form_spec("additive"), nlg_chain_form_spec("effect_modifying")
    assert term_names(additive["X2"]) == ["tanh(X1)", "A"]
    assert term_names(em["X2"]) == ["tanh(X1)", "A*tanh(X1)", "A"]
    # X₃ regime-invariant, tanh on its ONE endogenous parent (the chain SCM specification).
    assert term_names(additive["X3"]) == ["tanh(X2)", "A"] == term_names(em["X3"])


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("spec_fn", [chain_form_spec, nlg_chain_form_spec])
def test_form_spec_matches_true_graph_parent_sets(regime, spec_fn):
    """Every template's declared parents == the true graph's parents (the L1-oracle
    form-template-known semantics).

    X₃'s parent set must be {A, X2} and NOT contain X1 — an X₁→X₃ edge would collapse
    the chain into a triangle-with-an-extra-edge and destroy the composed path the chain SCM
    specification
    exists to contribute.
    """
    scm = make_linear_chain(regime)
    spec = spec_fn(regime)
    for node, node_form in spec.items():
        assert set(node_form.parents) == set(scm.parents(node)), node
    assert "X1" not in spec["X3"].parents, (
        "X₃ must not take X₁ as a parent — that is a triangle, not a chain (the chain SCM "
        "specification)."
    )


# ===========================================================================
# Group 6 — S(g) LINEAR-chain analytic gate (the composed-path closed form)
# ===========================================================================


def _linear_chain_s_closed_form(beta: float, gamma: float, delta: float, g: float) -> float:
    """‖M̃(g) − 𝟙‖_F from the chain's composed-path structure, in closed form.

    B(g) is strictly lower-triangular with exactly two live entries,
    b₂₁ = β + γ·g and b₃₂ = δ, so (I − B)⁻¹ = I + B + B² and

        M[2,1] = b₂₁,  M[3,2] = b₃₂,  M[3,1] = b₂₁·b₃₂   ← composed path

    hence ‖M − I‖²_F = b₂₁² + b₃₂² + (b₂₁·b₃₂)² = b₂₁²(1 + b₃₂²) + b₃₂².
    The third term is the chain's own contribution: on the triangle B² = 0 (one
    endogenous edge) and on the collider B² = 0 (two edges sharing a head), so
    neither carries it.
    """
    b21 = beta + gamma * g
    b32 = delta
    return float(np.sqrt(b21**2 * (1.0 + b32**2) + b32**2))


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_s_of_g_linear_chain_matches_the_composed_path_closed_form(regime, group):
    """Production `s_of_g` == the hand-derived closed form, to full precision.

    Exact (rtol 1e-12), not approximate: the linear branch evaluates B(g) by
    unit-bump probes in exact float arithmetic and takes no expectation, so any
    daylight here is a structural error — a missing composed path, a transposed
    index, or A leaking into the endogenous block — not numerical noise.
    """
    c = _linear_coeffs()
    gamma = c["gamma"] if regime == "effect_modifying" else 0.0
    expected = _linear_chain_s_closed_form(c["beta"], gamma, c["delta"], group)
    actual = s_of_g(make_linear_chain(regime), group, **_CHAIN_SETS)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_linear_chain_composed_path_term_is_actually_load_bearing():
    """Dropping the (3,1) term would give a DIFFERENT number — the gate has teeth.

    Without this, `test_..._matches_the_composed_path_closed_form` would still pass
    against a closed form that forgot the composed path if the production code
    forgot it too. Asserting the two-term and three-term formulas DISAGREE is what
    makes the agreement above informative.
    """
    c = _linear_coeffs()
    b21, b32 = c["beta"], c["delta"]
    with_composed = np.sqrt(b21**2 * (1 + b32**2) + b32**2)
    without_composed = np.sqrt(b21**2 + b32**2)
    assert with_composed > without_composed
    np.testing.assert_allclose(
        s_of_g(make_linear_chain("additive"), 1.0, **_CHAIN_SETS), with_composed, rtol=1e-12
    )


def test_s_of_g_linear_chain_additive_is_flat_exactly():
    """ΔS == 0 to machine precision on the linear additive control (H4's control arm).

    EXACT here, unlike the NLG additive cell: B(g) is constant in u and γ is absent,
    so both groups evaluate the identical expression. β_{A,2} and β_{A,3} are present
    and carry A — flatness holding WITH them is the content of PS-3's A-exclusion
    guard, not an artifact of a graph without A→X_i arrows.
    """
    s_neg = s_of_g(make_linear_chain("additive"), -1.0, **_CHAIN_SETS)
    s_pos = s_of_g(make_linear_chain("additive"), +1.0, **_CHAIN_SETS)
    assert s_neg == s_pos, f"S(−1)={s_neg!r} != S(+1)={s_pos!r} on the additive control"


def test_s_of_g_linear_chain_effect_modifying_sign_is_negative():
    """ΔS < 0 under effect modification — the chain SCM specification's sign convention, sign(γ) =
    sign(β).

    A = +1's sub-SCM has gain (β + γ) > (β − γ) on the upstream edge, so it
    propagates more and L0's no-propagation model approximates it WORSE. The SIGN is
    what H4 reads (sign(ΔS · ΔΔB) per PS-3), so it is asserted on its own — a flip
    here would break the cross-topology ΔS-sign pattern PS-8 pools on.
    """
    scm = make_linear_chain("effect_modifying")
    delta_s = s_of_g(scm, -1.0, **_CHAIN_SETS) - s_of_g(scm, +1.0, **_CHAIN_SETS)
    assert delta_s < 0, f"ΔS = {delta_s:+.6f}; the chain SCM specification pins sign(ΔS) negative"


# ===========================================================================
# Group 7 — S(g) NLG chain: FD tether (the NLG family specification) + structural zeros
# ===========================================================================

#: FD probe points, as U₁ / U₂ draws. With a = 0.6 the operating point is
#: X₁ = ±0.6 + U₁, so this set gives each group the group-mean interior (|X₁| ≈ 0.6,
#: where the cell operates), the edge of the chain SCM specification's non-saturation band (|X₁| ≈
#: 1.5), and
#: points beyond it — a probe set clustered at the mean would never exercise the
#: curvature the MC integrates over. U₂ uses a rotation so an X₁/X₂ column swap
#: cannot pass unnoticed.
_FD_U1_POINTS = (-2.1, -0.9, -0.4, 0.0, 0.4, 0.9, 2.1)
_FD_U2_POINTS = (0.4, 2.1, -0.9, 0.9, -2.1, -0.4, 0.0)
_N_FD = len(_FD_U1_POINTS)


def _fd_exogenous(group: float) -> dict[str, np.ndarray]:
    """The FD probe's exogenous vector at A = g (X₃'s noise is a sink)."""
    return {
        "A": np.full(_N_FD, float(group)),
        "X1": np.array(_FD_U1_POINTS),
        "X2": np.array(_FD_U2_POINTS),
        "X3": np.linspace(-1.0, 1.0, _N_FD),
    }


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_analytic_jacobian_matches_central_finite_difference(regime, group):
    """# [the NLG family specification] "validated by a central finite-difference regression test".

    The analytic Jacobian is hand-derived from the builder's equations, so nothing
    but an independent numerical derivative of THE SAME CLOSURES can catch a
    mis-derivation. The FD machinery is imported from the NLG-triangle S(g) gate and
    reads nothing from `jacobian_fn`, so the two paths are genuinely independent.
    """
    scm = make_nonlinear_gaussian_chain(regime)
    u = _fd_exogenous(group)
    analytic = np.asarray(scm.jacobian_fn(u, float(group), scm), dtype=float)
    numeric = _finite_difference_jacobian(scm, u, float(group))
    np.testing.assert_allclose(analytic, numeric, atol=_FD_ATOL)


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_structural_zeros_are_exactly_zero(regime, group):
    """Everywhere the chain graph has no edge, B is EXACTLY 0 — not merely small.

    In particular B[X3, X1] must be exactly zero: the composed path is created by the
    INVERSE (I − B)⁻¹, never by a direct X₁→X₃ entry. A nonzero slot there would mean
    the builder grew the edge the chain SCM specification forbids, and S(g) would still look
    plausible.
    """
    scm = make_nonlinear_gaussian_chain(regime)
    B = np.asarray(scm.jacobian_fn(_fd_exogenous(group), float(group), scm), dtype=float)
    endog = [n for n in scm.nodes if scm.parents(n)]
    i = {n: k for k, n in enumerate(endog)}
    live = {(i["X2"], i["X1"]), (i["X3"], i["X2"])}
    for r in range(len(endog)):
        for col in range(len(endog)):
            if (r, col) not in live:
                assert np.all(B[:, r, col] == 0.0), (
                    f"B[{endog[r]}, {endog[col]}] is nonzero — the chain graph has no "
                    "such edge (an X₁→X₃ slot would be the triangle collapse the chain SCM "
                    "specification forbids)."
                )


def test_analytic_jacobian_is_sech_squared_not_one_minus_tanh_squared():
    """The 1/cosh² form, live-asserted (mirrors the NLG triangle's and collider's precision guards).

    (1 − tanh²(x)) cancels two nearly-equal numbers once |x| grows and bleeds
    relative precision. The frozen coefficients keep the operating point out of
    saturation, but the MC integrates over the whole group marginal, whose tails do
    reach it — so the guard is asserted rather than left to a comment.
    """
    c = _nlg_coeffs()
    scm = make_nonlinear_gaussian_chain("additive")
    big = 20.0  # deep in the tail, where the two forms visibly diverge in float
    u = {
        "A": np.array([1.0]),
        "X1": np.array([big - c["a"]]),  # so X1 lands at `big`
        "X2": np.array([0.0]),
        "X3": np.array([0.0]),
    }
    B = np.asarray(scm.jacobian_fn(u, 1.0, scm), dtype=float)
    endog = [n for n in scm.nodes if scm.parents(n)]
    entry = float(B[0, endog.index("X2"), endog.index("X1")])
    sech2 = c["beta"] / np.cosh(big) ** 2
    naive = c["beta"] * (1.0 - np.tanh(big) ** 2)

    # EXACT equality, and no pytest.approx anywhere here: approx carries a default
    # ABSOLUTE floor of 1e-12, which at this magnitude (~1e-17) would call every
    # value equal to every other and make the whole guard vacuous.
    assert entry == sech2, f"jacobian entry {entry!r} is not the 1/cosh² form {sech2!r}"
    # The naive form does not merely lose precision at this probe point — it loses
    # the value ENTIRELY: tanh(20) rounds to exactly 1.0 in float64, so 1 − tanh²
    # cancels to a hard zero while sech² still carries ~1e-17. That total collapse is
    # the failure mode the convention exists to prevent.
    assert naive == 0.0 and sech2 > 0.0, (
        f"probe point is not deep enough for the two forms to diverge "
        f"(naive={naive!r}, sech²={sech2!r}) — strengthen it"
    )


def _mc_standard_error(scm, group: float, seed: int) -> tuple[float, float]:
    """(Ŝ, MC SE of Ŝ) at A = g, from an INDEPENDENT draw of the same size.

    Production `s_of_g` returns the point estimate only, and its `sg_mc` stream is
    pinned at module level (the NLG family specification) so repeated calls are bit-identical —
    there is no
    repetition to take a spread over. This estimates the MC error the way the
    estimator's definition allows: S = √(E[Q]) with Q := ‖M̃ − 𝟙‖²_F per draw, so by
    the delta method SE(Ŝ) = SD(Q) / (2·Ŝ·√n_mc). Drawn at a TEST-LOCAL seed,
    deliberately not the `sg_mc` one: this measures the estimator's sampling error,
    not a re-derivation of the production number.

    Unlike the collider's, this helper's inverse is NOT optional: on the chain
    M − I ≠ B on the restricted block, because B² carries the composed-path entry.
    """
    rng = np.random.default_rng(seed)
    n = S_OF_G_MC_SAMPLES
    u = {
        node: np.asarray(scm.noise_samplers[node](rng, n), dtype=float)
        for node in scm.nodes
    }
    u["A"] = np.full(n, float(group))  # A := U_A, so conditioning pins its entry
    B = np.asarray(scm.jacobian_fn(u, float(group), scm), dtype=float)
    k = B.shape[-1]
    deviation = np.linalg.inv(np.eye(k) - B) - np.eye(k)
    q = np.sum(deviation**2, axis=(-2, -1))
    s_hat = float(np.sqrt(q.mean()))
    return s_hat, float(q.std(ddof=1) / np.sqrt(n) / (2.0 * s_hat))


def test_nlg_chain_additive_is_flat_within_mc_noise():
    """S(−1) ≈ S(+1) on the NLG additive control — CLOSE, not EQUAL.

    DO NOT TIGHTEN TO EQUALITY. The two groups' integrands are equal in distribution
    (X₁ = ±a + U₁ and sech² is even) but are evaluated on DIFFERENT realizations of
    the same draws, so the residual is MC noise, not asymmetry. The 0.02 envelope is
    the NLG-triangle S(g) convention, imported rather than restated.
    """
    scm = make_nonlinear_gaussian_chain("additive")
    s_neg = s_of_g(scm, -1.0, **_CHAIN_SETS)
    s_pos = s_of_g(scm, +1.0, **_CHAIN_SETS)
    assert abs(s_neg - s_pos) < _MC_NOISE_ENVELOPE, (
        f"NLG chain additive S(−1)={s_neg:.6f} vs S(+1)={s_pos:.6f}: |ΔS|="
        f"{abs(s_neg - s_pos):.6f} exceeds the {_MC_NOISE_ENVELOPE} MC-noise envelope. "
        "The control arm must be flat — a real gap means A entered the endogenous "
        "propagation block, e.g. β_{A,2}·A leaking into B(g; u)."
    )


def test_nlg_chain_effect_modifying_is_distinct_with_a_5_se_sign_margin():
    """|ΔS| ≥ 5 × MC SE with ΔS < 0 — the chain SCM specification's sign convention on the NLG
    family.

    The margin is in MC SEs rather than absolute units because the failure mode
    PS-2's gate condition (ii) is written against is precisely "a nominally-correct
    ΔS sitting within MC noise of zero".
    """
    scm = make_nonlinear_gaussian_chain("effect_modifying")
    s_neg = s_of_g(scm, -1.0, **_CHAIN_SETS)
    s_pos = s_of_g(scm, +1.0, **_CHAIN_SETS)
    delta_s = s_neg - s_pos
    assert delta_s < 0, f"ΔS = {delta_s:+.6f}; the chain SCM specification pins sign(ΔS) negative"

    _, se_neg = _mc_standard_error(scm, -1.0, seed=8101)
    _, se_pos = _mc_standard_error(scm, +1.0, seed=8102)
    se = float(np.hypot(se_neg, se_pos))
    assert abs(delta_s) >= _SIGN_MARGIN_MC_SE * se, (
        f"|ΔS| = {abs(delta_s):.6f} is only {abs(delta_s) / se:.1f} MC SE from zero "
        f"(need ≥ {_SIGN_MARGIN_MC_SE}); the effect-modifying cell is not detectable."
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_nlg_chain_s_of_g_is_finite_and_nonnegative(regime, group):
    value = s_of_g(make_nonlinear_gaussian_chain(regime), group, **_CHAIN_SETS)
    assert np.isfinite(value) and value >= 0.0


# ===========================================================================
# Groups 9 + 10 — pipeline smoke, pulled-back G1 anchor, aggregation schema
# ===========================================================================


@pytest.fixture(scope="module")
def chain_runs():
    """One L0+L2 linear-chain run per regime, shared across the pipeline tests.

    L0+L2 only: this file gates CONSTRUCTION, and the L1-oracle rung's end-to-end
    behaviour is exercised by the cross-seed grid. Running the full ladder here
    would triple the fixture cost for no additional construction coverage.
    """
    cfg = load_config("configs/recourse_chain.yaml")
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime in _REGIMES:
            runs[regime] = run_regime(
                cfg,
                regime,
                conditions=("L0", "L2"),
                scm_builder=make_linear_chain,
                form_spec_fn=chain_form_spec,
                run_anchor=True,
                family="linear",
            )
    return cfg, runs


@pytest.mark.parametrize("regime", _REGIMES)
def test_classifier_calibration_gate_passes_on_chain(regime):
    """the classifier construction: accuracy in band, base rate on
    target, both pools non-trivial.

    `build_classifier` asserts all three internally, so reaching the assertions below
    already means the gate passed; they restate the numbers so a failure REPORT names
    the chain rather than a generic calibration message.
    """
    cfg = load_config("configs/recourse_chain.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = build_classifier(make_linear_chain(regime), cfg)
    lo, hi = tuple(cfg.classifier.accuracy_band)
    assert lo <= result.accuracy_overall <= hi
    assert abs(result.realized_base_rate - float(cfg.classifier.base_rate_target)) <= float(
        cfg.classifier.base_rate_tol
    )
    for group, count in result.neg_counts_by_group.items():
        assert count >= int(cfg.classifier.min_neg_per_group), (group, count)


def test_builtin_g1_anchor_passes_on_chain_both_regimes(chain_runs):
    """anchor.py's own L2 smoke check passes on the chain (k=3, all 2^k−1 subsets)."""
    _, runs = chain_runs
    for regime in _REGIMES:
        report = runs[regime].anchor
        assert report is not None, f"anchor did not run on the linear chain [{regime}]"
        assert report.passed, (
            f"G1 anchor smoke check FAILED on the chain [{regime}]:\n{report.format()}"
        )


def _effective_weights_chain(
    w: np.ndarray, b21: float, delta: float
) -> dict[frozenset, float]:
    """‖J_Sᵀw‖₂ for every non-empty acted subset S ⊆ {X1,X2,X3} on the chain (a″).

    Feature axes are (X1, X2, X3) = w indices (0, 1, 2). Per acted variable, the
    displacement gradient follows the chain's severance rules:
      • δ_X1 moves X1 always; moves X2 only if X2 is FREE (else the edge is severed);
        and moves X3 only if X2 AND X3 are both free — the COMPOSED path, the one
        term the triangle and collider cannot produce.
      • δ_X2 moves X2 always; moves X3 only if X3 is free.
      • δ_X3 moves X3 only (terminal).
    """
    names = ("X1", "X2", "X3")
    weights: dict[frozenset, float] = {}
    for mask in range(1, 8):
        S = {names[i] for i in range(3) if mask & (1 << i)}
        x2_free, x3_free = "X2" not in S, "X3" not in S
        grad: list[float] = []
        if "X1" in S:
            g = w[0]
            if x2_free:
                g += w[1] * b21
                if x3_free:
                    g += w[2] * b21 * delta  # composed path X1 -> X2 -> X3
            grad.append(g)
        if "X2" in S:
            grad.append(w[1] + (w[2] * delta if x3_free else 0.0))
        if "X3" in S:
            grad.append(w[2])
        weights[frozenset(S)] = float(np.linalg.norm(grad, ord=2))
    return weights


def test_g1_pulled_back_anchor_on_chain_l2_cell(chain_runs):
    """Chain L2 brute-force vs. the pulled-back Ehyaei closed form, within the reachability filter.

    An INDEPENDENT reimplementation of the pullback for the chain's structural map
    (all 7 acted subsets, built by hand), mirroring the collider's belt-and-braces
    discipline: it validates anchor.py's enumeration from a separate code path, and
    it is the only place the composed-path term of the Jacobian is checked against
    the actual optimizer. A failure here while the triangle and collider pass is a
    pullback issue specific to the chain's serial map and must be reported, NOT
    tolerated by loosening the tolerance (the reachability filter fixes it at 1.5 × grid diagonal
    step
    and handles unreachable individuals by FILTERING, never by widening).
    """
    cfg, runs = chain_runs
    run = runs["additive"]  # b21 = beta for both groups, parallel to the triangle G1
    clf = run.classifier
    feats = clf.feature_names
    w = clf.model.coef_[0].astype(float)
    c = _linear_coeffs()

    acted = list(cfg.recourse.actionable)
    action_space = build_action_space(
        clf.dataset, acted, mode=str(cfg.recourse.grid.mode),
        k=float(cfg.recourse.grid.k), resolution=int(cfg.recourse.grid.resolution),
    )
    # Tether the closed form to the REALIZER: probe the true SCM for b21 and delta
    # rather than reading the builder defaults, so a coefficient change cannot make
    # the test and the code agree on a stale number.
    scm = make_linear_chain("additive")
    true_model = L2TrueSCMModel(scm, feats, acted)
    assert true_model is not None
    base = pd.DataFrame({"A": [1.0], "X1": [0.0], "X2": [0.0], "X3": [0.0]})
    b21 = float(scm.counterfactual(base, {"X1": 1.0})["X2"].iloc[0]
                - scm.counterfactual(base, {"X1": 0.0})["X2"].iloc[0])
    delta = float(scm.counterfactual(base, {"X2": 1.0})["X3"].iloc[0]
                  - scm.counterfactual(base, {"X2": 0.0})["X3"].iloc[0])
    np.testing.assert_allclose([b21, delta], [c["beta"], c["delta"]], atol=1e-12)

    effective = _effective_weights_chain(w, b21, delta)
    best_weight = max(effective.values())

    grid_step = grid_diagonal_step(action_space.axis_grids)
    span_limit = grid_span_limit(action_space.axis_grids)
    abs_threshold = float(cfg.recourse.anchor.tolerance) * grid_step

    l2 = run.table[run.table["condition"] == "L2"].reset_index(drop=True)
    excesses, n_compared = [], 0
    for _, row in l2.iterrows():
        if not (row["found"] and row["believed_validity"] == 1):
            continue
        factual = clf.dataset.iloc[int(row["index"])][feats].to_numpy(dtype=float)
        h = float(clf.model.decision_function(factual.reshape(1, -1))[0])
        if abs(h) / float(np.linalg.norm(w, ord=2)) > span_limit:
            continue  # the reachability filter — excluded, never accommodated
        n_compared += 1
        excesses.append(float(row["realized_cost"]) - abs(h) / best_weight)

    assert n_compared >= 50, f"only {n_compared} grid-reachable individuals to compare"
    excess = np.array(excesses)
    # One-sided: the brute force searches a FINITE grid, so it can only ever cost
    # MORE than the continuous optimum. A negative excess beyond float noise would
    # mean the closed form is over-predicting, i.e. the pullback is wrong.
    assert excess.min() > -1e-9, f"brute force beat the closed form by {-excess.min():.6f}"
    assert np.median(excess) <= abs_threshold, (
        f"chain L2 median excess {np.median(excess):.6f} exceeds the reachability tolerance "
        f"{abs_threshold:.6f} ({cfg.recourse.anchor.tolerance} x grid diagonal step)."
    )


def test_aggregation_accepts_topology_chain_with_no_schema_drift(chain_runs, tmp_path):
    """topology="chain" rolls up with the same columns as every other cell.

    The aggregation layer is topology-agnostic (`topology` is a
    free-string key on `CellKeys`), so this asserts that property still holds rather
    than exercising new code — the same check the collider got at the SCM specification.
    """
    from icknowledge.aggregation.schema import by_cell_columns, by_group_columns

    cfg, runs = chain_runs
    regime = "additive"
    out = tmp_path / "chain_cell"
    out.mkdir()
    for condition in ("L0", "L2"):
        sub = runs[regime].table[runs[regime].table["condition"] == condition]
        sub.to_csv(out / f"scoring_table_{regime}_{condition}.csv", index=False)

    cell = CellKeys(topology="chain", family="linear", regime=regime, seed=int(cfg.seed))
    by_group_path, by_cell_path = aggregate_run(out, out, cell)

    by_group = pd.read_csv(by_group_path, float_precision="round_trip")
    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")
    # [PS-4 note (1)] Two conditions are scored here, so the
    # common-found columns carry the `_twoway` denominator label. Compared against
    # the schema module's arity-2 answer rather than the raw ByGroupRow/ByCellRow
    # field lists (which are written at the canonical three-condition arity) — the
    # no-schema-drift property this test exists for is unchanged.
    assert list(by_group.columns) == list(by_group_columns(2))
    assert list(by_cell.columns) == list(by_cell_columns(2))
    assert set(by_group["topology"]) == {"chain"}
    assert set(by_cell["topology"]) == {"chain"}
    assert set(by_group["group"]) == {-1, 1}
