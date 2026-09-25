"""Concrete TRIANGLE ground-truth SCMs.

Builder functions that CONSTRUCT AND RETURN configured `SCM` instances (base.py,
convention: no subclassing). Two functional families × two attribute regimes =
four SCMs, all sharing one graph:

    Triangle over A, X1, X2 with edges  A->X1,  A->X2,  X1->X2.

A is the protected root (Gender), with nonzero downstream influence (convention
1). The regime is DECLARED at construction (convention 2), never inferred. All
four SCMs are additive-noise (is_anm=True, convention 5), so abduction
(u = x - f(parents, 0)) is valid.

Structural equations
--------------------
LINEAR family:
  additive:          X1 := a*A + U1 ;  X2 := b*A + g*X1 + U2
  effect_modifying:  X1 := a*A + U1 ;  X2 := g_of_A*X1 + U2   (no additive b*A term)

NONLINEAR-GAUSSIAN family (nonlinear in parents, additive Gaussian noise):
  additive:          X1 := a*A + U1 ;  X2 := b*A + g*tanh(X1) + U2
  effect_modifying:  X1 := a*A + U1 ;  X2 := g_of_A*tanh(X1) + U2

g_of_A = g_pos if A>0 else g_neg — the multiplicative A×X1 interaction that
makes the mechanism effect-modifying. U1, U2 ~ N(0, sigma^2); A ∈ {-1, +1}
centered. The PS-2 gate is evaluated by the analysis layer
(`analysis/t4_reports.py`); this module carries only the hand-derivable
mechanism precondition (tests/test_triangle_scm.py, Group 2).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from icknowledge.estimation.forms import TANH, Factor, FormSpec, NodeForm, TermSpec
from icknowledge.scm.base import SCM

_VALID_REGIMES = frozenset({"additive", "effect_modifying"})


def _triangle_graph() -> nx.DiGraph:
    """Triangle DiGraph: A->X1, A->X2, X1->X2."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "X1", "X2"])
    graph.add_edges_from([("A", "X1"), ("A", "X2"), ("X1", "X2")])
    return graph


def _make_noise_samplers(sigma: float) -> dict:
    """Noise samplers: A draws a centered ±1 group indicator; U1, U2 ~ N(0, sigma^2)."""
    # A ∈ {-1, +1}: centered binary group indicator (keeps additive shifts symmetric).
    def _sample_A(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice([-1.0, 1.0], size=n)

    def _sample_gaussian(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, sigma, size=n)

    return {"A": _sample_A, "X1": _sample_gaussian, "X2": _sample_gaussian}


def _eq_A(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
    """Root A ignores parents and returns its own noise (the ±1 draw)."""
    return noise


def _eq_X1(a: float):
    """X1 := a*A + U1 (shared across both regimes and families)."""

    def _f(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return a * parents["A"] + noise

    return _f


def make_linear_triangle(
    regime: str,
    *,
    a: float = 2.0,
    b: float = 1.0,
    g: float = 0.5,
    g_pos: float = 0.9,
    g_neg: float = 0.3,
    sigma: float = 1.0,
    name: str | None = None,
) -> SCM:
    """Build a LINEAR triangle SCM in the given attribute regime.

    Triangle topology A->X1, A->X2, X1->X2 (module docstring: shared conventions).

    Structural equations
    --------------------
    additive:          X1 := a*A + U1 ;  X2 := b*A + g*X1 + U2   (slope g SHARED)
    effect_modifying:  X1 := a*A + U1 ;  X2 := g_of_A*X1 + U2    (g_of_A: g_pos if
                       A>0 else g_neg — a multiplicative A*X1 interaction)

    U1, U2 ~ N(0, sigma^2). Coefficients carry nonzero defaults (convention 1).

    Parameters
    ----------
    regime: {"additive", "effect_modifying"}; else ValueError.
    a, b, g: additive-regime coefficients (b, g additive only).
    g_pos, g_neg: effect-modifying group slopes (effect_modifying only).
    sigma: std of U1, U2.
    name: optional label (descriptive default if None).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X2 := b*A + g*X1 + U2 (slope g shared across groups).
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return b * parents["A"] + g * parents["X1"] + noise
    else:  # effect_modifying
        # X2 := g_of_A*X1 + U2 ; g_of_A = np.where(A>0, g_pos, g_neg) — the A*X1 interaction.
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            slope = np.where(parents["A"] > 0, g_pos, g_neg)
            return slope * parents["X1"] + noise

    equations = {"A": _eq_A, "X1": _eq_X1(a), "X2": _eq_X2}
    return SCM(
        graph=_triangle_graph(),
        equations=equations,
        noise_samplers=_make_noise_samplers(sigma),
        regime=regime,
        is_anm=True,
        name=name or f"linear-triangle[{regime}]",
    )


def triangle_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the LINEAR triangle's estimated equations.

    # Form-template-known L1-oracle. [the L1-oracle form-template-known
    # semantics]
    # The template (which parents enter; whether an A×parent interaction is
    # present) is SUPPLIED to the estimator, which fits ONLY the coefficients. It
    # lives HERE, next to the true triangle builders, so the supplied form and the
    # true equations sit in one file and cannot silently drift apart. The true
    # graph alone does not determine this form (A→X₂ says nothing about main
    # effect vs. interaction); see estimation/base.py for the full form-template-known rationale.

    # One equation per non-root endogenous node; interaction explicit.
    # [skill:scm-specification; attribute regimes]
    #   A  — exogenous root, NO fit (roots are absent from the FormSpec).
    #   X1 — parents {A}:      X1 = α₀ + α₁·A + ε₁. Identical in both regimes:
    #        with a single parent no interaction is structurally possible.
    #   X2 — parents {A, X1}:  X2 = β₀ + β₁·X1 + β₂·A + γ·(A·X1) + ε₂, with γ
    #        HARD-ZERO in the additive regime (implemented as OMISSION of the
    #        interaction column from the design matrix, not fit-then-discard) and
    #        γ estimated in the effect-modifying regime.

    Linear family only; the nonlinear-Gaussian template is `nlg_triangle_form_spec`
    below, behind the same pluggable interface.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    x2_interactions = () if regime == "additive" else (("A", "X1"),)
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A", "X1"), interactions=x2_interactions),
    }


def _nlg_triangle_jacobian(regime: str, *, g: float, g_pos: float, g_neg: float):
    """Analytic endogenous Jacobian B(g; u) for the NLG triangle.

    Closed-form derivatives of `make_nonlinear_gaussian_triangle`'s equations,
    evaluated at (u, A = g) — not a finite-difference probe. Validated against a
    central finite difference in tests/test_s_of_g_nlg.py; co-located with its
    builder so derivative and truth cannot drift apart.

    B is indexed over the ENDOGENOUS nodes only (PS-3): 2×2 with the single entry
    [X2, X1] — the slot the linear triangle fills with the constant g, now
    u-dependent through X1.

    Returns
    -------
    A callback ``jacobian(u, group, scm) -> ndarray`` of shape ``(n, k, k)``.
    ``u`` maps node -> exogenous draws (root entry pinned at the group value);
    descendants are reconstructed by the SCM's OWN equations via ``scm``, which
    also supplies the endogenous index ordering; ``group`` is the scalar A = g
    (fixes g_of_A).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    def _jacobian(
        u: dict[str, np.ndarray], group: float, scm: SCM
    ) -> np.ndarray:
        x1_noise = np.asarray(u["X1"], dtype=float)
        n = x1_noise.shape[0]
        a_column = np.full(n, float(group))
        # X1 from the SCM's own equation closure — the Jacobian differentiates the
        # equations actually in play, so only the DERIVATIVE is written out here.
        x1 = np.asarray(
            scm.equations["X1"]({"A": a_column}, x1_noise), dtype=float
        )
        gain = g if regime == "additive" else (g_pos if float(group) > 0 else g_neg)

        # sech²(x) as 1/cosh(x)² and NOT (1 − tanh(x)²): at the saturated operating
        # point (a = 2.0 puts |X1| ≈ 2 before noise) tanh(X1) → ±1 and the
        # 1 − tanh² form cancels catastrophically, while 1/cosh² stays exact.
        d_x2_d_x1 = gain / np.cosh(x1) ** 2

        endog = [node for node in scm.nodes if scm.parents(node)]
        B = np.zeros((n, len(endog), len(endog)))
        B[:, endog.index("X2"), endog.index("X1")] = d_x2_d_x1
        return B

    return _jacobian


def make_nonlinear_gaussian_triangle(
    regime: str,
    *,
    a: float = 2.0,
    b: float = 1.0,
    g: float = 0.5,
    g_pos: float = 0.9,
    g_neg: float = 0.3,
    sigma: float = 1.0,
    name: str | None = None,
) -> SCM:
    """Build a NONLINEAR-GAUSSIAN triangle SCM in the given attribute regime.

    Triangle topology A->X1, A->X2, X1->X2. A ∈ {-1, +1} is the
    centered protected group indicator (Gender), a root node with nonzero
    downstream influence (scm-specification convention 1). The ``regime`` is
    DECLARED, never inferred (convention 2). Noise stays additive Gaussian
    (is_anm=True, convention 5), so abduction u = x - f(parents, 0) remains valid
    despite the nonlinear mechanism.

    Structural equations (tanh nonlinearity)
    ----------------------------------------
    additive:          X1 := a*A + U1 ;  X2 := b*A + g*tanh(X1) + U2   (shared mechanism)
    effect_modifying:  X1 := a*A + U1 ;  X2 := g_of_A*tanh(X1) + U2    (group-dependent gain
                       g_of_A on the SAME nonlinearity: g_pos if A>0 else g_neg)

    U1, U2 ~ N(0, sigma^2). Coefficients are construction parameters with nonzero
    defaults (convention 1).

    Parameters mirror `make_linear_triangle`.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X2 := b*A + g*tanh(X1) + U2 (shared nonlinear mechanism).
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return b * parents["A"] + g * np.tanh(parents["X1"]) + noise
    else:  # effect_modifying
        # X2 := g_of_A*tanh(X1) + U2 ; group-dependent gain on the same nonlinearity.
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            gain = np.where(parents["A"] > 0, g_pos, g_neg)
            return gain * np.tanh(parents["X1"]) + noise

    equations = {"A": _eq_A, "X1": _eq_X1(a), "X2": _eq_X2}
    scm = SCM(
        graph=_triangle_graph(),
        equations=equations,
        noise_samplers=_make_noise_samplers(sigma),
        regime=regime,
        is_anm=True,
        name=name or f"nonlinear-gaussian-triangle[{regime}]",
    )
    # [the NLG family specification] DISCOVERY PATTERN for the analytic Jacobian: an OPTIONAL
    # `jacobian_fn` ATTRIBUTE, attached by the builder that owns the equations.
    # Chosen over (a) a constructor parameter — that would change `SCM.__init__`'s
    # signature for every linear cell that will never carry one — and over (b) a
    # name-keyed registry — `name` is a caller-overridable label, so a registry
    # would silently miss `make_nonlinear_gaussian_triangle(..., name="pilot")`.
    # `descriptor/s_of_g.py` reads it with getattr and refuses (never falls back to
    # the linear branch) when a nonlinear SCM carries none.
    scm.jacobian_fn = _nlg_triangle_jacobian(
        regime, g=g, g_pos=g_pos, g_neg=g_neg
    )
    return scm


def nlg_triangle_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the NONLINEAR-GAUSSIAN triangle.

    # L1-oracle template convention: the template is SUPPLIED (parents, fixed
    # transform, interaction presence) and the estimator fits ONLY the
    # coefficients; columns mirror the builder above TERM BY TERM (linear-in-
    # parameters once tanh(·) is applied, so unregularized OLS carries over).
    # Co-located with its builder so template and truth cannot drift apart.
    # NO A MAIN EFFECT in the effect-modifying template: the builder carries
    # none, and a term absent from the truth is hard-zero by COLUMN OMISSION
    # (the L1-oracle template convention). A is still a DECLARED parent of X₂ —
    # L1-oracle carries the TRUE graph; `build_estimated_scm` checks the parent
    # set. The intercept is estimated regardless (fit_intercept=True); the truth
    # has none, so it comes back ≈0 and is harmless.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    a_main = TermSpec((Factor("A"),))
    tanh_x1 = TermSpec((Factor("X1", TANH),))
    a_tanh_x1 = TermSpec((Factor("A"), Factor("X1", TANH)))
    x2_terms = (a_main, tanh_x1) if regime == "additive" else (tanh_x1, a_tanh_x1)
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A", "X1"), terms=x2_terms),
    }
