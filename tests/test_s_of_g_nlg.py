"""S(g) on the NONLINEAR-GAUSSIAN triangle (PS-3).

Gates the NLG branch of `icknowledge/descriptor/s_of_g.py`: the analytic local
Jacobian B(g; u) attached by `make_nonlinear_gaussian_triangle`, and its Monte-Carlo
integration over u | A = g at the pinned `sg_mc` seed. A failure is a MODULE BUG.

Covers, in the order the NLG family specification pins them:

  1. FD REGRESSION — analytic Jacobian vs a central finite difference of the SCM's
     own closures, at several (u, A = g) points, both regimes; the NLG FD tether.
  2. ADDITIVE FLATNESS — S(−1) ≈ S(+1) on the additive control (H4's control arm).
  3. EFFECT-MODIFYING DISTINCTNESS — S(−1) ≠ S(+1) with the sign ΔS < 0 the linear
     branch already shows, so the two families agree on direction.
  4. SANITY — finite and ≥ 0 on both NLG cells.
  5. SEED DETERMINISM — repeated calls bit-identical (the MC seed is construction-time).
  6. LINEAR REGRESSION GUARD — the four linear S(g) values are untouched by this branch.
  7. REFUSAL — a nonlinear SCM with no analytic Jacobian raises rather than falling
     back to the linear point-value path.

The NLG COLLIDER has its own file, tests/test_s_of_g_nlg_collider.py, which IMPORTS
this file's FD machinery, FD tolerance and MC-noise envelope: one tether, one tolerance.

EXACT-EQUALITY WARNING, load-bearing for the flatness, distinctness and sign
contracts below. On the LINEAR additive triangle S(−1) == S(+1) to 1e-12 because B(g)
is a constant. Here they do not: the RMS is a Monte-Carlo estimate and, although the
two groups' integrands are equal in distribution (X₁ = ±a + U₁ and sech² is even), they
are evaluated on DIFFERENT realizations of the same draws. The residual is MC noise,
not asymmetry, and no tolerance in this file may be tightened toward equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from icknowledge.descriptor import NLG_NOT_IMPLEMENTED, s_of_g
from icknowledge.scm import make_linear_triangle, make_nonlinear_gaussian_triangle

# Topology-defined, identical to tests/test_s_of_g_triangle.py: the classifier is
# group-blind over A's descendants (the classifier construction) and the actionable set excludes A
# itself.
_TRIANGLE_SETS = {"V_h": ("X1", "X2"), "S": ("X1", "X2")}
_REGIMES = ("additive", "effect_modifying")
_GROUPS = (-1.0, 1.0)

#: MC-noise envelope on the additive arm at n_mc = 10,000 (the NLG family specification). NOT an
#: equality
#: tolerance — see the module docstring.
_MC_NOISE_ENVELOPE = 0.02

#: Central-difference step and tolerance for the NLG-family regression tether. At h = 1e-5 the
#: truncation error is O(h²·f‴) ≈ 1e-10 and the cancellation error O(ε/h) ≈ 1e-11, so
#: 1e-4 is loose against the numerics and tight against any real derivative error.
_FD_STEP = 1e-5
_FD_ATOL = 1e-4

#: Evaluation points for the FD tether, as U₁ draws. Deliberately spans BOTH regimes of
#: tanh: with a = 2.0 the operating point is X₁ = ±2 + U₁, so u₁ = −2 / +2 lands X₁ at
#: the linear centre for one group and deep in saturation (|X₁| = 4, sech² ≈ 1.3e-3) for
#: the other. A probe set clustered near X₁ = 0 would never exercise the saturated
#: regime the sech² precision note is about.
_FD_U1_POINTS = (-4.0, -2.0, -0.5, 0.0, 1.5, 3.0)


def _endogenous(scm) -> list[str]:
    """Non-root nodes in topological order — the index the Jacobian is built against."""
    return [node for node in scm.nodes if scm.parents(node)]


def _realize(scm, u: dict[str, np.ndarray], group: float, protected: str = "A") -> dict:
    """Endogenous values at (u, A = g), by ancestral evaluation of the SCM's equations."""
    n = len(u["X1"])
    values: dict[str, np.ndarray] = {}
    for node in scm.nodes:
        if node == protected:
            values[node] = np.full(n, float(group))
            continue
        parents = {p: values[p] for p in scm.parents(node)}
        values[node] = np.asarray(scm.equations[node](parents, u[node]), dtype=float)
    return values


def _finite_difference_jacobian(
    scm, u: dict[str, np.ndarray], group: float, protected: str = "A"
) -> np.ndarray:
    """CENTRAL finite-difference B(g; u), shape (n, k, k), from the equation closures.

    The independent construction the NLG family specification requires: it reads nothing from
    `jacobian_fn` and
    differentiates the realizer itself — ``(f_j(x_i + h) − f_j(x_i − h)) / 2h`` at the
    endogenous values realized from (u, A = g), with noise zero (under ANM the noise is
    additive, so it cancels in the difference and contributes nothing to ∂f_j/∂x_i).
    """
    endog = _endogenous(scm)
    values = _realize(scm, u, group, protected)
    n = len(u["X1"])
    zero = np.zeros(n)
    B = np.zeros((n, len(endog), len(endog)))
    for j, child in enumerate(endog):
        parents = scm.parents(child)
        for i, parent in enumerate(endog):
            if parent not in parents:
                continue  # structural zero — no edge, no derivative
            up = {p: values[p].copy() for p in parents}
            down = {p: values[p].copy() for p in parents}
            up[parent] = values[parent] + _FD_STEP
            down[parent] = values[parent] - _FD_STEP
            forward = np.asarray(scm.equations[child](up, zero), dtype=float)
            backward = np.asarray(scm.equations[child](down, zero), dtype=float)
            B[:, j, i] = (forward - backward) / (2.0 * _FD_STEP)
    return B


# ===========================================================================
# 1 — the NLG family specification's mandated validation tether: analytic Jacobian vs. central FD
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_analytic_jacobian_matches_central_finite_difference(regime, group):
    """# [the NLG family specification] "validated by a central finite-difference regression test".

    The analytic Jacobian is hand-derived from the builder's equations, so nothing but
    this test stands between a transcription slip (a dropped gain, a sech² vs. sech
    confusion, a transposed [child, parent] slot) and a silently wrong S(g). Both
    regimes are covered because they differ exactly in the gain — additive's shared `g`
    vs. effect-modifying's group-dependent g_of_A — which is the term a slip would hit.
    """
    scm = make_nonlinear_gaussian_triangle(regime)
    u = {
        "A": np.full(len(_FD_U1_POINTS), float(group)),
        "X1": np.array(_FD_U1_POINTS),
        # X₂ is a sink: its noise cannot reach any derivative, but it is drawn in the
        # production path so it is supplied here too rather than special-cased.
        "X2": np.linspace(-1.0, 1.0, len(_FD_U1_POINTS)),
    }
    analytic = np.asarray(scm.jacobian_fn(u, float(group), scm), dtype=float)
    numeric = _finite_difference_jacobian(scm, u, float(group))

    assert analytic.shape == numeric.shape == (len(_FD_U1_POINTS), 2, 2)
    np.testing.assert_allclose(
        analytic,
        numeric,
        atol=_FD_ATOL,
        rtol=0.0,
        err_msg=(
            f"NLG triangle [{regime}] at A={group:g}: analytic B(g; u) disagrees with a "
            f"central finite difference (h={_FD_STEP:g}) beyond {_FD_ATOL:g}. Either the "
            "closed-form derivative in `_nlg_triangle_jacobian` no longer matches "
            "`make_nonlinear_gaussian_triangle`'s equations, or the [child, parent] "
            "index convention drifted."
        ),
    )
    # The structural zeros must be exactly zero, not merely small: B is indexed over the
    # endogenous nodes, X₁ has no endogenous parent, and X₂ is a sink, so [X₂, X₁] is the
    # ONLY live slot. A nonzero elsewhere means A leaked into the endogenous block.
    assert analytic[:, 0, 0].tolist() == analytic[:, 0, 1].tolist() == [0.0] * len(
        _FD_U1_POINTS
    )
    assert analytic[:, 1, 1].tolist() == [0.0] * len(_FD_U1_POINTS)


def test_analytic_jacobian_is_sech_squared_not_one_minus_tanh_squared():
    """Saturation guard for the 1/cosh² form (implementation note, the NLG family specification
    branch).

    At a = 2.0 the operating point is already saturated (|X₁| ≈ 2, tanh ≈ 0.964), and it
    is far worse in the tails this test probes. `(1 − tanh(x)²)` cancels two nearly-equal
    numbers there and bleeds relative precision; `1/cosh(x)²` does not. The assertion is
    on the FD tether, which is the thing that would degrade if the fragile form were
    reinstated — so this test fails on the CHANGE, not merely on the code shape.
    """
    scm = make_nonlinear_gaussian_triangle("additive")
    deep = np.array([6.0, 8.0, 10.0])  # X₁ = 2 + u₁ reaches 8, 10, 12
    u = {"A": np.ones(3), "X1": deep, "X2": np.zeros(3)}
    analytic = np.asarray(scm.jacobian_fn(u, 1.0, scm), dtype=float)[:, 1, 0]

    x1 = 2.0 + deep
    stable = 0.5 / np.cosh(x1) ** 2
    fragile = 0.5 * (1.0 - np.tanh(x1) ** 2)
    np.testing.assert_allclose(analytic, stable, rtol=1e-12, atol=0.0)
    # …and the fragile form is genuinely OUTSIDE that tolerance here, which is what makes
    # the assertion above a live guard rather than a tautology: at X₁ = 12, 1 − tanh²
    # subtracts two numbers agreeing to ~1e-11, so its relative error is ~3e-7 against a
    # true value of ~1.5e-10. Swapping the implementation would fail this test.
    assert (np.abs(fragile - stable) / stable > 1e-12).any()


# ===========================================================================
# 2 — additive flatness (H4's control arm), WITHIN the MC-noise envelope
# ===========================================================================


def test_nlg_additive_is_flat_within_mc_noise():
    """S(−1) ≈ S(+1) on the NLG additive control — CLOSE, deliberately not EQUAL.

    DO NOT TIGHTEN THIS TO AN EQUALITY. On the LINEAR additive triangle S(−1) == S(+1)
    exactly (B is a constant, so both groups evaluate one expression). Here the two are
    Monte-Carlo estimates: the integrands are equal in DISTRIBUTION — X₁ = ±a + U₁ and
    sech² is even, so sech²(a + U₁) and sech²(−a + U₁) share a law — but they are
    evaluated on different realizations of the same n_mc draws, so the estimates differ
    by MC noise. The 0.02 envelope is that noise at n_mc = 10,000 (the NLG family specification);
    the assertion
    is that the arm is FLAT UP TO SAMPLING ERROR, which is what H4's control requires
    once the descriptor is estimated rather than solved. Asserting equality here would
    make the test fail on the seed, not on the science.
    """
    scm = make_nonlinear_gaussian_triangle("additive")
    s_neg = s_of_g(scm, -1.0, **_TRIANGLE_SETS)
    s_pos = s_of_g(scm, +1.0, **_TRIANGLE_SETS)
    assert abs(s_neg - s_pos) < _MC_NOISE_ENVELOPE, (
        f"NLG additive S(−1)={s_neg:.6f} vs S(+1)={s_pos:.6f}: |ΔS|="
        f"{abs(s_neg - s_pos):.6f} exceeds the {_MC_NOISE_ENVELOPE} MC-noise envelope. "
        "The control arm must be flat — a real gap means A entered the endogenous "
        "propagation block, e.g. the additive builder's b·A term leaking into B(g; u)."
    )
    # The additive NLG triangle carries b·A (an A→X₂ main effect) exactly as the linear
    # one does: flatness holding WITH that term present is the content of the A-exclusion
    # guard, not an artifact of a graph without an A→X₂ arrow.


# ===========================================================================
# 3 — effect-modifying distinctness, and sign agreement with the linear branch
# ===========================================================================


def test_nlg_effect_modifying_is_distinct_and_signs_with_the_linear_branch():
    """S(−1) ≠ S(+1) well above MC noise, with ΔS < 0 as on the linear family.

    ΔS := S(−1) − S(+1) is negative because g_neg < g_pos: the A = −1 sub-SCM propagates
    LESS, so L0's no-propagation model approximates it BETTER. The sign is what H4 reads
    (sign(ΔS · ΔΔB) per PS-3), so it is asserted rather than left to the magnitude test —
    a sign flip between families would break the six-point pooling PS-8 pins.
    """
    scm = make_nonlinear_gaussian_triangle("effect_modifying")
    s_neg = s_of_g(scm, -1.0, **_TRIANGLE_SETS)
    s_pos = s_of_g(scm, +1.0, **_TRIANGLE_SETS)
    delta = s_neg - s_pos

    assert abs(delta) > _MC_NOISE_ENVELOPE, (
        f"NLG effect-modifying S(−1)={s_neg:.6f} vs S(+1)={s_pos:.6f}: |ΔS|="
        f"{abs(delta):.6f} is inside the {_MC_NOISE_ENVELOPE} MC-noise envelope, so the "
        "treatment arm is indistinguishable from the control. A modifies X₂'s gain on "
        "tanh(X₁), so this must separate."
    )
    assert delta < 0.0, (
        f"NLG effect-modifying ΔS={delta:.6f} ≥ 0, but the linear family gives ΔS = "
        "|g_neg| − |g_pos| < 0. The two families must agree on the SIGN of ΔS — H4's "
        "prediction is on sign(ΔS · ΔΔB(c)), pooled across families (PS-8)."
    )

    linear = make_linear_triangle("effect_modifying")
    linear_delta = s_of_g(linear, -1.0, **_TRIANGLE_SETS) - s_of_g(
        linear, +1.0, **_TRIANGLE_SETS
    )
    assert np.sign(delta) == np.sign(linear_delta)
    # [PS-8] LEVELS are not compared — only the sign. The NLG |ΔS| is much smaller
    # than the linear one (sech² < 1 damps every derivative), and that is a scale
    # difference, not a finding.


# ===========================================================================
# 4 — sanity on both NLG cells
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_nlg_s_of_g_is_finite_and_nonnegative(regime, group):
    scm = make_nonlinear_gaussian_triangle(regime)
    value = s_of_g(scm, group, **_TRIANGLE_SETS)
    assert np.isfinite(value) and value >= 0.0, (
        f"NLG triangle/{regime}/A={group:g}: S(g)={value} — must be a finite "
        "non-negative RMS Frobenius norm."
    )


# ===========================================================================
# 5 — seed determinism: the `sg_mc` stream is construction-time, not per-call
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
def test_repeated_calls_are_bit_identical(regime):
    """# [the NLG family specification] The `sg_mc` stream is spawned ONCE from the seed-generation
    meta-entropy.

    Bit-identity — not near-equality — is the assertion: a fresh Generator is built from
    the pinned seed on every call, so no RNG state may advance between them. If it did,
    S(g) would drift with call ORDER, and a descriptor whose value depends on when it was
    computed is not pre-committed in any meaningful sense (PS-3). Distinct SCM objects of
    the same cell must also agree, since S(g) is a property of the specification.
    """
    scm = make_nonlinear_gaussian_triangle(regime)
    for group in _GROUPS:
        first = s_of_g(scm, group, **_TRIANGLE_SETS)
        second = s_of_g(scm, group, **_TRIANGLE_SETS)
        assert first == second, (
            f"NLG {regime} A={group:g}: repeated s_of_g calls returned {first!r} then "
            f"{second!r}. The MC RNG state advanced across calls — the `sg_mc` seed must "
            "be re-derived per call, never held as a live Generator."
        )
        rebuilt = s_of_g(
            make_nonlinear_gaussian_triangle(regime), group, **_TRIANGLE_SETS
        )
        assert rebuilt == first


# ===========================================================================
# 6 — regression guard: the linear branch is untouched
# ===========================================================================


def test_linear_s_of_g_values_unchanged_by_the_nlg_branch():
    """All four linear triangle S(g) values, at the pre-NLG tolerance.

    The NLG branch reorganized `s_of_g`'s body (the affineness failure now routes rather
    than raises, and the restriction is written to broadcast over an MC axis). These four
    numbers are the closed forms — |g| for both groups on the additive control, |g_neg| /
    |g_pos| under effect modification — and they must be BYTE-stable through that
    refactor, not merely close.
    """
    additive = make_linear_triangle("additive")
    assert s_of_g(additive, -1.0, **_TRIANGLE_SETS) == 0.5
    assert s_of_g(additive, +1.0, **_TRIANGLE_SETS) == 0.5

    effect_modifying = make_linear_triangle("effect_modifying")
    np.testing.assert_allclose(
        [
            s_of_g(effect_modifying, -1.0, **_TRIANGLE_SETS),
            s_of_g(effect_modifying, +1.0, **_TRIANGLE_SETS),
        ],
        [0.3, 0.9],
        atol=1e-12,
    )


# ===========================================================================
# 7 — refusal survives: nonlinear WITHOUT an analytic Jacobian still raises
# ===========================================================================


def test_nonlinear_scm_without_a_jacobian_is_refused():
    """No silent linear fallback for a nonlinear cell that supplies no `jacobian_fn`.

    This is the guard the NLG-refusal test carried, and
    it must outlive it: every NLG builder shipped so far attaches a `jacobian_fn` (the
    the NLG triangle, the NLG collider), so the refusal has no live cell to fire on —
    which is exactly why it is simulated here. The failure mode to prevent is S(g)
    quietly returning the derivative at one arbitrary base point on the NEXT nonlinear
    cell whose builder forgets one.
    """
    scm = make_nonlinear_gaussian_triangle("effect_modifying")
    del scm.jacobian_fn  # simulate a builder that forgot to attach one

    with pytest.raises(NotImplementedError) as excinfo:
        s_of_g(scm, -1.0, **_TRIANGLE_SETS)
    message = str(excinfo.value)
    assert NLG_NOT_IMPLEMENTED in message
    # The refusal must also name the node that failed affineness — otherwise the message
    # says "nonlinear" without saying where, which is useless on a bigger topology.
    assert "'X2'" in message and "not affine" in message


# ===========================================================================
# REPORTING (no assertions) — the H4 pre-numbers record
# ===========================================================================


def test_report_nlg_triangle_s_of_g_values(capsys):
    """Print the four NLG triangle S(g) values. Deliberately assertion-free.

    These go to the H4 pre-numbers record. They are PRINTED and not asserted so that an
    PS-2 coefficient retune re-reports rather than fails: the behavioural contracts live
    above (flatness, distinctness, sign), which are retune-invariant, while the
    LEVELS are not (PS-8).
    """
    with capsys.disabled():
        print("\n  NLG triangle S(g)  [PS-3, n_mc=10,000, seed stream `sg_mc`]")
        for regime in _REGIMES:
            scm = make_nonlinear_gaussian_triangle(regime)
            s_neg = s_of_g(scm, -1.0, **_TRIANGLE_SETS)
            s_pos = s_of_g(scm, +1.0, **_TRIANGLE_SETS)
            print(
                f"    {regime:17s}  S(-1)={s_neg:.6f}  S(+1)={s_pos:.6f}  "
                f"DeltaS={s_neg - s_pos:+.6f}"
            )
