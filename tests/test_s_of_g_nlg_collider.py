"""S(g) on the NONLINEAR-GAUSSIAN collider (PS-3 + the SCM specification).

Gates the NLG collider's analytic local Jacobian B(g; u) — attached by
`make_nonlinear_gaussian_collider` — through the PRODUCTION `s_of_g` module, whose
NLG branch picks it up as an optional `jacobian_fn` attribute. A failure here is a
BUILDER BUG (a mis-derived derivative, a transposed index) or a MODULE regression.

Covers, per the NLG collider spec:

  1. FD REGRESSION — the analytic Jacobian against a central finite difference of the
     SCM's own equation closures, at 7 probe points × both regimes × both groups.
  2. STRUCTURAL ZEROS — exactly zero (not merely small) everywhere the graph says
     there is no edge, so the A-leak failure mode stays guarded.
  3. SECH² PRECISION GUARD — the 1/cosh² form, with a live assertion that fails if
     the fragile (1 − tanh²) form is reinstated.
  4. ADDITIVE FLATNESS — S(−1) ≈ S(+1) on the additive control (H4's control arm).
  5. EFFECT-MODIFYING DISTINCTNESS — |ΔS| ≥ 5 × MC SE with ΔS < 0.
  6. SANITY — finite and ≥ 0 on both NLG collider cells.

The FD machinery, FD tolerance and MC-noise envelope are IMPORTED from
tests/test_s_of_g_nlg.py: one tether and one tolerance for both topologies.

EXACT-EQUALITY WARNING, load-bearing for the flatness, sign and margin contracts
below. On the LINEAR additive collider S(−1) == S(+1) to machine precision because
B(g) is a constant. Here they do not: the RMS is a Monte-Carlo estimate and, although
the two groups' integrands are equal in distribution (X_i = ±α + U_i and sech² is
even), they are evaluated on DIFFERENT realizations. The residual is MC noise, not
asymmetry, and no tolerance in this file may be tightened toward equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from icknowledge.descriptor import S_OF_G_MC_SAMPLES, s_of_g
from icknowledge.scm import make_linear_collider, make_nonlinear_gaussian_collider

# ONE tether and ONE tolerance across both NLG topologies: the NLG-triangle S(g) gate owns
# the central-FD construction (which reads nothing from `jacobian_fn` and
# differentiates the realizer itself), the 1e-4 absolute tolerance, and the 0.02
# MC-noise envelope. Importing them is what keeps the collider from drifting to a
# looser gate. `_finite_difference_jacobian` is topology-generic already.
from tests.test_s_of_g_nlg import (
    _FD_ATOL,
    _FD_STEP,
    _MC_NOISE_ENVELOPE,
    _finite_difference_jacobian,
)

# Topology-defined, identical to tests/test_s_of_g_triangle.py's collider sets: the
# classifier is group-blind over A's descendants (the classifier construction) and the actionable
# set
# excludes A, so V_h == S == {X1, X2, X3} here.
_COLLIDER_SETS = {"V_h": ("X1", "X2", "X3"), "S": ("X1", "X2", "X3")}
_REGIMES = ("additive", "effect_modifying")
_GROUPS = (-1.0, 1.0)

#: Sign-margin bar on the effect-modifying arm: |ΔS| must clear this many MC SEs.
#: [the rule of thumb; the SCM specification] The pre-freeze calibration
#: recorded |ΔS|/MC_SE ≈ 131, so 5 is an enormously slack floor — deliberately, so
#: the assertion fails on a BROKEN CELL (a sign that has collapsed toward noise,
#: the failure mode PS-3 convention 4 exists to catch) and not on a PS-2 retune
#: that legitimately moves the level. The PS-2 detectability gate condition (ii)
#: reads this same quantity in the cross-seed grid; this is its production-side precursor.
_SIGN_MARGIN_MC_SE = 5.0

#: Evaluation points for the FD tether, as U₁ draws (U₂ uses a rotation of the same values, so a
#: tanh(X1)/tanh(X2) column swap cannot pass unnoticed — the two channels are numerically
#: exchangeable under the SCM defaults). With α = 0.6 the operating point is X_i = ±0.6 + U_i, so
#: this set gives EACH group both the group-mean interior (|X_i| ≈ 0.6, where the cell actually
#: operates) and moderate tails at |X_i| ≈ 1.5 — the edge of the non-saturation band the SCM
#: specification pins — plus points beyond it.
#:     A = +1:  X₁ ∈ {−1.5, −0.3, 0.2, 0.6, 1.0, 1.5, 2.7}
#:     A = −1:  X₁ ∈ {−2.7, −1.5, −1.0, −0.6, −0.2, 0.3, 1.5}
_FD_U1_POINTS = (-2.1, -0.9, -0.4, 0.0, 0.4, 0.9, 2.1)
_FD_U2_POINTS = (0.4, 2.1, -0.9, 0.9, -2.1, -0.4, 0.0)
_N_FD = len(_FD_U1_POINTS)


def _fd_exogenous(group: float) -> dict[str, np.ndarray]:
    """The FD probe's exogenous vector at A = g (X₃'s noise is a sink — see below)."""
    return {
        "A": np.full(_N_FD, float(group)),
        "X1": np.array(_FD_U1_POINTS),
        "X2": np.array(_FD_U2_POINTS),
        # X₃ is a sink: its noise cannot reach any derivative, but it IS drawn in the
        # production path, so it is supplied here too rather than special-cased.
        "X3": np.linspace(-1.0, 1.0, _N_FD),
    }


def _mc_standard_error(scm, group: float, seed: int) -> tuple[float, float]:
    """(Ŝ, MC SE of Ŝ) at A = g, from an INDEPENDENT draw of the same size.

    The production `s_of_g` returns the point estimate only, and its `sg_mc` stream
    is pinned at module level (the NLG family specification) so repeated calls are bit-identical —
    there is
    no repetition to take a spread over. This helper therefore estimates the MC
    error the way the estimator's own definition allows: S = √(E[Q]) with
    Q := ‖M̃ − 𝟙‖²_F per draw, so by the delta method
        SE(Ŝ) = SD(Q) / (2 · Ŝ · √n_mc).
    Drawn at a TEST-LOCAL seed, deliberately not the `sg_mc` one: this is a
    measurement of the estimator's sampling error, not a re-derivation of the
    production number, and reusing the pinned stream would make the SE and the
    value share their noise.

    On this topology M − I equals B on the restricted block exactly (B is strictly
    lower-triangular with B² = 0 on the collider), but the inverse is taken here
    anyway so the helper measures what the production path computes.
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


# ===========================================================================
# 1 + 2 — the NLG family specification's mandated tether: analytic Jacobian vs. central FD, + zeros
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_analytic_jacobian_matches_central_finite_difference(regime, group):
    """# [the NLG family specification] "validated by a central finite-difference regression test".

    The analytic Jacobian is hand-derived from the builder's equations, so nothing
    but this test stands between a transcription slip (a dropped gain, a sech² vs.
    sech confusion, a transposed [child, parent] slot, a tanh(X1)/tanh(X2) swap) and
    a silently wrong S(g). Both regimes are covered because they differ exactly in
    X₃'s X₁-gain — additive's β vs. effect-modifying's (β + γ·A) — which is the term
    a slip would hit, and both groups because that gain is group-dependent.
    """
    scm = make_nonlinear_gaussian_collider(regime)
    u = _fd_exogenous(group)
    analytic = np.asarray(scm.jacobian_fn(u, float(group), scm), dtype=float)
    numeric = _finite_difference_jacobian(scm, u, float(group))

    assert analytic.shape == numeric.shape == (_N_FD, 3, 3)
    np.testing.assert_allclose(
        analytic,
        numeric,
        atol=_FD_ATOL,
        rtol=0.0,
        err_msg=(
            f"NLG collider [{regime}] at A={group:g}: analytic B(g; u) disagrees with "
            f"a central finite difference (h={_FD_STEP:g}) beyond {_FD_ATOL:g}. "
            "Either the closed-form derivative in `_nlg_collider_jacobian` no longer "
            "matches `make_nonlinear_gaussian_collider`'s equations, or the "
            "[child, parent] index convention drifted."
        ),
    )


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_structural_zeros_are_exactly_zero(regime, group):
    """Every no-edge slot of B is EXACTLY 0.0 — the A-leak guard.

    B is indexed over the ENDOGENOUS nodes (X1, X2, X3) in topological order, so
    PS-3's restriction to Desc(A) has already excluded A from both axes. Only
    [X3, X1] and [X3, X2] are live. In particular:

      * the X1 and X2 ROWS are entirely zero — their only parent is the ROOT A, and
        if A's influence leaked into the endogenous block the additive control arm
        would stop being flat (the β_A·A main effect on X₃ is exactly the term that
        would leak), contradicting H4's design;
      * [X1, X3] and [X2, X3] are zero — X₃ is a sink, there is no back-edge;
      * the [X3, X3] self-slot is zero — no self-loop, which is also what keeps
        (I − B) unit-triangular and always invertible.

    Asserted as exact equality, not `allclose`: these are absences, not small
    numbers, and a leak would typically be O(1) (β_A = 1.0) rather than epsilon.
    """
    scm = make_nonlinear_gaussian_collider(regime)
    analytic = np.asarray(
        scm.jacobian_fn(_fd_exogenous(group), float(group), scm), dtype=float
    )
    endog = [node for node in scm.nodes if scm.parents(node)]
    assert endog == ["X1", "X2", "X3"], f"endogenous index drifted: {endog}"

    live = {(2, 0), (2, 1)}  # (X3, X1) and (X3, X2)
    for j in range(3):
        for i in range(3):
            if (j, i) in live:
                continue
            assert analytic[:, j, i].tolist() == [0.0] * _N_FD, (
                f"NLG collider [{regime}] A={group:g}: B[{endog[j]}, {endog[i]}] is "
                "nonzero, but the graph has no such endogenous edge. If this is the "
                "X1 or X2 row, A has leaked into the endogenous propagation block."
            )
    # …and the live slots really are live, so the loop above is not vacuously true
    # against an all-zero Jacobian.
    assert np.all(analytic[:, 2, 0] != 0.0) and np.all(analytic[:, 2, 1] != 0.0)


def test_analytic_jacobian_is_sech_squared_not_one_minus_tanh_squared():
    """Saturation guard for the 1/cosh² form (the NLG-triangle precision convention).

    The frozen the SCM coefficients keep the OPERATING point out of saturation
    (|α| + 2σ = 1.4 ≤ 1.5), so unlike the NLG triangle at a = 2.0 this cell is not
    routinely in the fragile regime — but the MC integrates over the whole group
    marginal, whose tails do reach it, and the precision convention is a standing
    one rather than a per-cell judgement call. `(1 − tanh(x)²)` cancels two
    nearly-equal numbers at large |x| and bleeds relative precision; `1/cosh(x)²`
    does not. The assertion is on the VALUE, so this test fails on the CHANGE, not
    merely on the code shape.
    """
    scm = make_nonlinear_gaussian_collider("additive")
    deep = np.array([7.4, 9.4, 11.4])  # X₁ = 0.6 + u₁ reaches 8, 10, 12
    u = {"A": np.ones(3), "X1": deep, "X2": np.zeros(3), "X3": np.zeros(3)}
    analytic = np.asarray(scm.jacobian_fn(u, 1.0, scm), dtype=float)[:, 2, 0]

    x1 = 0.6 + deep
    stable = 0.7 / np.cosh(x1) ** 2
    fragile = 0.7 * (1.0 - np.tanh(x1) ** 2)
    np.testing.assert_allclose(analytic, stable, rtol=1e-12, atol=0.0)
    # …and the fragile form is genuinely OUTSIDE that tolerance here, which is what
    # makes the assertion above a live guard rather than a tautology: at X₁ = 12,
    # 1 − tanh² subtracts two numbers agreeing to ~1e-11, so its relative error is
    # ~3e-7 against a true value of ~1.5e-10. Swapping the implementation fails this.
    assert (np.abs(fragile - stable) / stable > 1e-12).any()


# ===========================================================================
# 4 — additive flatness (H4's control arm), WITHIN the MC-noise envelope
# ===========================================================================


def test_nlg_collider_additive_is_flat_within_mc_noise():
    """S(−1) ≈ S(+1) on the NLG collider's additive control — CLOSE, not EQUAL.

    DO NOT TIGHTEN THIS TO AN EQUALITY (see the module docstring). The 0.02 envelope
    is the NLG-triangle convention, IMPORTED rather than restated: it was the MC noise floor at
    n_mc = 10,000 on the triangle, and it stays conservative here because this cell's
    S(g) LEVEL is larger (≈ 0.72 vs the triangle's), so the same absolute envelope is
    a looser relative bar. The assertion is that the arm is FLAT UP TO SAMPLING
    ERROR, which is what H4's control requires once the descriptor is estimated
    rather than solved.

    The additive NLG collider carries β_A·A (an A→X₃ main effect) exactly as the
    linear one does: flatness holding WITH that term present is the content of the
    A-exclusion guard, not an artifact of a graph without an A→X₃ arrow.
    """
    scm = make_nonlinear_gaussian_collider("additive")
    s_neg = s_of_g(scm, -1.0, **_COLLIDER_SETS)
    s_pos = s_of_g(scm, +1.0, **_COLLIDER_SETS)
    assert abs(s_neg - s_pos) < _MC_NOISE_ENVELOPE, (
        f"NLG collider additive S(−1)={s_neg:.6f} vs S(+1)={s_pos:.6f}: |ΔS|="
        f"{abs(s_neg - s_pos):.6f} exceeds the {_MC_NOISE_ENVELOPE} MC-noise "
        "envelope. The control arm must be flat — a real gap means A entered the "
        "endogenous propagation block, e.g. the additive builder's β_A·A term "
        "leaking into B(g; u)."
    )


# ===========================================================================
# 5 — effect-modifying distinctness, sign, and the 5×MC-SE margin
# ===========================================================================


def test_nlg_collider_effect_modifying_is_distinct_with_a_5_se_sign_margin():
    """|ΔS| ≥ 5 × MC SE with ΔS < 0, matching the linear collider and the SCM specification.

    ΔS := S(−1) − S(+1) is negative because γ > 0: the A = +1 sub-SCM's gain on
    tanh(X₁) is (β + γ) > (β − γ), so it propagates MORE and L0's no-propagation
    model approximates it WORSE. The SIGN is what H4 reads (sign(ΔS · ΔΔB) per
    PS-3), so it is asserted separately from the magnitude — a sign flip between
    families or topologies would break the pooling PS-8 pins.

    The margin is measured in MC SEs rather than in absolute units, because the
    failure mode PS-2's detectability gate condition (ii) was written against is
    precisely "a nominally-correct ΔS sitting within MC noise of zero". 5 SE is the
    rule-of-thumb floor; the SCM specification's pre-freeze calibration
    recorded ≈ 131 SE, so a run anywhere near the floor is itself a signal.
    """
    scm = make_nonlinear_gaussian_collider("effect_modifying")
    s_neg = s_of_g(scm, -1.0, **_COLLIDER_SETS)
    s_pos = s_of_g(scm, +1.0, **_COLLIDER_SETS)
    delta = s_neg - s_pos

    # Independent test-local draws (see `_mc_standard_error`), combined as
    # independent — conservative, since the production estimates at the two groups
    # share their draws and are positively correlated, which SHRINKS SE(ΔS).
    _s, se_neg = _mc_standard_error(scm, -1.0, seed=20260731)
    _s, se_pos = _mc_standard_error(scm, +1.0, seed=20260732)
    se_delta = float(np.hypot(se_neg, se_pos))

    assert delta < 0.0, (
        f"NLG collider effect-modifying ΔS={delta:.6f} ≥ 0, but γ > 0 makes the "
        "A=+1 sub-SCM propagate more, and the LINEAR collider gives ΔS < 0. The "
        "cells must agree on the SIGN of ΔS — H4's prediction is on "
        "sign(ΔS · ΔΔB(c)), pooled across families and topologies (PS-8)."
    )
    assert abs(delta) >= _SIGN_MARGIN_MC_SE * se_delta, (
        f"NLG collider effect-modifying |ΔS|={abs(delta):.6f} is only "
        f"{abs(delta) / se_delta:.1f} MC SE from zero (SE={se_delta:.6f}), under the "
        f"{_SIGN_MARGIN_MC_SE:g}× floor. The SCM specification's pre-freeze calibration recorded "
        "≈131 SE, so this is a broken cell, not a tight one."
    )
    assert abs(delta) > _MC_NOISE_ENVELOPE, (
        "…and it must also clear the additive arm's flatness envelope, or treatment "
        "and control would be indistinguishable by the same yardstick."
    )

    linear = make_linear_collider("effect_modifying")
    linear_delta = s_of_g(linear, -1.0, **_COLLIDER_SETS) - s_of_g(
        linear, +1.0, **_COLLIDER_SETS
    )
    assert np.sign(delta) == np.sign(linear_delta)
    # [PS-8] LEVELS are not compared — only the sign. sech² < 1 damps every
    # NLG derivative relative to the linear collider's constants, and that is a
    # scale difference, not a finding.


# ===========================================================================
# 6 — sanity on both NLG collider cells
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_nlg_collider_s_of_g_is_finite_and_nonnegative(regime, group):
    scm = make_nonlinear_gaussian_collider(regime)
    value = s_of_g(scm, group, **_COLLIDER_SETS)
    assert np.isfinite(value) and value >= 0.0, (
        f"NLG collider/{regime}/A={group:g}: S(g)={value} — must be a finite "
        "non-negative RMS Frobenius norm."
    )


# ===========================================================================
# REPORTING (no assertions) — the H4 pre-numbers record
# ===========================================================================


def test_report_nlg_collider_s_of_g_values(capsys):
    """Print the four NLG collider S(g) values, ΔS, and the MC SE. Assertion-free.

    These go to the H4 pre-numbers record and are the PRODUCTION counterpart of the
    pre-freeze MC recorded in the SCM specification. They are PRINTED and not
    asserted so that a PS-2 coefficient retune re-reports rather than fails: the
    behavioural contracts live above (flatness, sign, margin), which are
    retune-invariant, while the LEVELS are not (PS-8). ASCII-only — this goes to a
    cp1252 Windows console.
    """
    with capsys.disabled():
        print(
            "\n  NLG collider S(g)  [PS-3 + the SCM specification, "
            f"n_mc={S_OF_G_MC_SAMPLES:,}, seed stream `sg_mc`]"
        )
        for regime in _REGIMES:
            scm = make_nonlinear_gaussian_collider(regime)
            s_neg = s_of_g(scm, -1.0, **_COLLIDER_SETS)
            s_pos = s_of_g(scm, +1.0, **_COLLIDER_SETS)
            _s, se_neg = _mc_standard_error(scm, -1.0, seed=20260731)
            _s, se_pos = _mc_standard_error(scm, +1.0, seed=20260732)
            se_delta = float(np.hypot(se_neg, se_pos))
            delta = s_neg - s_pos
            print(
                f"    {regime:17s}  S(-1)={s_neg:.6f}  S(+1)={s_pos:.6f}  "
                f"DeltaS={delta:+.6f}  MC_SE={se_delta:.6f}  "
                f"|DeltaS|/MC_SE={abs(delta) / se_delta:.1f}"
            )
