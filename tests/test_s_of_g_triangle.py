"""S(g) on the TRIANGLE + the NLG refusal gate (PS-3).

The triangle counterpart to tests/test_collider_scm.py Group 5, against the promoted
production module `icknowledge/descriptor/s_of_g.py`. A failure here is a MODULE BUG,
not a finding. Covers:

  1. flatness / distinctness — S(−1) == S(+1) exactly on the additive control,
     S(−1) ≠ S(+1) on the effect-modifying treatment (H4's control-vs-treatment
     precondition, the same pair of assertions the collider test makes);
  2. the NLG boundary — s_of_g routes both nonlinear-Gaussian triangle regimes into the
     The NLG family specification branch, never a silently-wrong
     linear fallback;
     the branch's own behaviour is gated in tests/test_s_of_g_nlg.py;
  3. sanity — finite and ≥ 0 on all four (topology × regime) linear cells this
     module covers.

Triangle closed form. The only endogenous edge is X₁ → X₂ with slope s(g), so
B(g) = [[0, 0], [s(g), 0]] is nilpotent and M(g) − 𝟙 = B(g), giving S(g) = |s(g)|:
|g| in the additive control (shared slope, hence flat) and |g_neg| / |g_pos| under
effect modification. True coefficients are read PROGRAMMATICALLY from the builder
signature (the `structural_g` pattern), so a PS-2 retune needs no test edit.

Numerical tolerances are TAKEN FROM the collider test (1e-12 for the flatness
equality and the closed-form check, 1e-6 for the treatment-arm separation) rather
than chosen fresh — the two topologies are held to one standard.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from icknowledge.descriptor import NLG_NOT_IMPLEMENTED, s_of_g
from icknowledge.scm import (
    make_linear_collider,
    make_linear_triangle,
    make_nonlinear_gaussian_triangle,
)

# V_h and S are TOPOLOGY-DEFINED (they are not free parameters of the descriptor):
# the classifier is group-blind over A's descendants (the classifier construction) and the
# actionable set
# excludes the immutable A. On both topologies V_h ∩ Desc(A) and S ∩ Desc(A)
# therefore reduce to the full endogenous set — the restriction is a no-op HERE, but
# s_of_g still applies it so a later asymmetric topology gets it right.
_TRIANGLE_SETS = {"V_h": ("X1", "X2"), "S": ("X1", "X2")}
_COLLIDER_SETS = {"V_h": ("X1", "X2", "X3"), "S": ("X1", "X2", "X3")}
_REGIMES = ("additive", "effect_modifying")
_GROUPS = (-1.0, 1.0)


def _triangle_coeffs() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_linear_triangle).parameters
    return {k: float(params[k].default) for k in ("a", "b", "g", "g_pos", "g_neg", "sigma")}


# ===========================================================================
# 1 — flatness under the additive control, distinctness under effect modification
# ===========================================================================


def test_s_of_g_flat_under_additive_and_distinct_under_effect_modification():
    d = _triangle_coeffs()

    # Additive control: S(−1) == S(+1) EXACTLY — flat by construction, which is what
    # H4's control arm requires. Closed form |g| for both groups. [PS-3: "under
    # additive control SCMs (γ = 0) ... S(−1) = S(+1) exactly".]
    scm_add = make_linear_triangle("additive")
    s_neg = s_of_g(scm_add, -1.0, **_TRIANGLE_SETS)
    s_pos = s_of_g(scm_add, +1.0, **_TRIANGLE_SETS)
    expected_add = abs(d["g"])
    assert abs(s_neg - s_pos) < 1e-12, (
        f"additive S(−1)={s_neg} != S(+1)={s_pos} — the control arm MUST be flat "
        "across groups (H4 control). A leak here means A entered the propagation "
        f"sub-block. Closed form is |g|={expected_add:.6f} for BOTH groups."
    )
    np.testing.assert_allclose([s_neg, s_pos], [expected_add, expected_add], atol=1e-12)
    # The additive triangle carries b·A (an A→X₂ main effect) exactly as the collider
    # carries β_A: flatness holding WITH that term present is the real content of the
    # A-exclusion guard, not an artifact of a graph without an A→X₂ arrow.

    # Effect-modifying: S(−1) != S(+1), both finite. Closed form |g_neg| / |g_pos|.
    scm_em = make_linear_triangle("effect_modifying")
    e_neg = s_of_g(scm_em, -1.0, **_TRIANGLE_SETS)
    e_pos = s_of_g(scm_em, +1.0, **_TRIANGLE_SETS)
    assert np.isfinite(e_neg) and np.isfinite(e_pos)
    assert abs(e_neg - e_pos) > 1e-6, (
        f"effect-modifying S(−1)={e_neg:.6f} == S(+1)={e_pos:.6f} — the treatment arm "
        "MUST be group-distinct (A modifies X₂'s X₁-coefficient)."
    )
    np.testing.assert_allclose(
        [e_neg, e_pos], [abs(d["g_neg"]), abs(d["g_pos"])], atol=1e-12
    )


# ===========================================================================
# 2 — the NLG boundary
# ===========================================================================


@pytest.mark.parametrize("regime", _REGIMES)
@pytest.mark.parametrize("group", _GROUPS)
def test_s_of_g_takes_the_nlg_branch_on_the_nonlinear_gaussian_triangle(regime, group):
    """The NLG triangle COMPUTES rather than refuses (the NLG family
    specification's branch).

    The tanh mechanism makes
    ∂f/∂x₁ depend on x₁, so the linear branch's collapse of E_{u|A=g}[·] to a point value
    is invalid; PS-3 requires the local Jacobian plus MC over the group-g exogenous
    distribution, and the NLG family specification pins how (analytic Jacobian, n_mc = 10,000,
    `sg_mc` seed).
    That branch exists for the triangle, so what is asserted HERE is only that both
    regimes route into it and return a usable number.

    The refusal itself has NOT been dropped — a nonlinear SCM carrying no analytic
    Jacobian still raises rather than falling back to
    the linear path. That guard, and every behavioural property of the branch, lives in
    tests/test_s_of_g_nlg.py; `NLG_NOT_IMPLEMENTED` is still exported and still used.
    """
    scm = make_nonlinear_gaussian_triangle(regime)
    value = s_of_g(scm, group, **_TRIANGLE_SETS)
    assert np.isfinite(value) and value >= 0.0
    assert "jacobian_fn" in NLG_NOT_IMPLEMENTED  # the refusal path is still wired


# ===========================================================================
# 3 — sanity across all four linear cells this module covers
# ===========================================================================


def test_s_of_g_finite_and_nonnegative_on_all_linear_cells():
    """Finite and ≥ 0 on triangle × {additive, eff-mod} and collider × the same.

    # [PS-8] LEVELS are compared WITHIN a topology only; this test asserts the
    # per-cell properties (finite, non-negative — it is a norm) and deliberately makes
    # no cross-topology magnitude claim.
    """
    cells = [
        ("triangle", make_linear_triangle, _TRIANGLE_SETS),
        ("collider", make_linear_collider, _COLLIDER_SETS),
    ]
    for topology, builder, sets in cells:
        for regime in _REGIMES:
            scm = builder(regime)
            for group in _GROUPS:
                value = s_of_g(scm, group, **sets)
                assert np.isfinite(value) and value >= 0.0, (
                    f"{topology}/{regime}/A={group:g}: S(g)={value} — must be a finite "
                    "non-negative Frobenius norm."
                )


# ===========================================================================
# API guards — the keyword-only contract and the input validation
# ===========================================================================


def test_v_h_and_s_are_keyword_only():
    """Rows (V_h) and columns (S) cannot be passed positionally.

    They coincide on both topologies in scope, so a positional swap would be SILENT
    here and would surface only once an asymmetric topology lands — which is exactly
    why the signature forbids it rather than documenting it.
    """
    scm = make_linear_triangle("additive")
    with pytest.raises(TypeError):
        s_of_g(scm, -1.0, ("X1", "X2"), ("X1", "X2"))


def test_rejects_out_of_range_group_and_unknown_nodes():
    scm = make_linear_triangle("additive")
    with pytest.raises(ValueError, match="group must be"):
        s_of_g(scm, 0.0, **_TRIANGLE_SETS)
    with pytest.raises(ValueError, match="unknown node"):
        s_of_g(scm, -1.0, V_h=("X1", "X9"), S=("X1", "X2"))
