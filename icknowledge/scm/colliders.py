"""Concrete COLLIDER ground-truth SCMs (linear + nonlinear-Gaussian families).

Builder functions that CONSTRUCT AND RETURN configured `SCM` instances (base.py,
convention: no subclassing), paralleling `triangles.py`. Two functional families
× two attribute regimes share one graph:

    Collider (a′) over A, X1, X2, X3 with edges
        A -> X1,  A -> X2,  A -> X3,  X1 -> X3 <- X2.

A is the protected root (Gender), with nonzero downstream influence (convention
1). The regime is DECLARED at construction (convention 2), never inferred.
Additive-noise throughout (is_anm=True, convention 5), so abduction
(u = x - f(parents, 0)) is valid at every node.

`gamma` (on X₃) is the effect-modifying channel, hard-zero in the additive
control by COLUMN OMISSION; `beta_a*A` is carried in BOTH regimes, so the
A→X₃ edge is matched across regimes (see the SCM specification). X₁ ⊥ X₂ | A
holds by construction, so X₃ is a genuine collider within each group. X₁ is
the modified channel by the a-priori LOWEST-INDEX symmetry convention.
Coefficient and noise scales match the triangle; induced variances differ and
are never silently retuned (PS-2); cross-topology comparison is pattern-only
(PS-8).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from icknowledge.estimation.forms import TANH, Factor, FormSpec, NodeForm, TermSpec
from icknowledge.scm.base import SCM

_VALID_REGIMES = frozenset({"additive", "effect_modifying"})


def _collider_graph() -> nx.DiGraph:
    """Collider (a′) DiGraph: A->X1, A->X2, A->X3, X1->X3, X2->X3."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "X1", "X2", "X3"])
    graph.add_edges_from(
        [("A", "X1"), ("A", "X2"), ("A", "X3"), ("X1", "X3"), ("X2", "X3")]
    )
    return graph


def _make_noise_samplers(sigma: float) -> dict:
    """Noise samplers: A draws a centered ±1 group indicator; U1, U2, U3 ~ N(0, sigma^2)."""
    # A ∈ {-1, +1}: centered binary group indicator (keeps additive shifts symmetric).
    def _sample_A(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice([-1.0, 1.0], size=n)

    def _sample_gaussian(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, sigma, size=n)

    return {
        "A": _sample_A,
        "X1": _sample_gaussian,
        "X2": _sample_gaussian,
        "X3": _sample_gaussian,
    }


def _eq_A(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
    """Root A ignores parents and returns its own noise (the ±1 draw)."""
    return noise


def _eq_root_child(intercept: float, slope: float):
    """A-child equation X := intercept + slope*A + U (shared across both regimes)."""

    def _f(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return intercept + slope * parents["A"] + noise

    return _f


def make_linear_collider(
    regime: str,
    *,
    a1: float = 2.0,
    a2: float = 2.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    eta: float = 0.5,
    beta_a: float = 1.0,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    sigma: float = 1.0,
    name: str | None = None,
) -> SCM:
    """Build a LINEAR collider SCM in the given attribute regime.

    Collider (a′) topology A->X1, A->X2, A->X3, X1->X3<-X2 (module docstring:
    shared conventions).

    Structural equations
    --------------------
    A   := U_A                                            (±1 centered root)
    X1  := c1 + a1*A + U1                                 (both regimes)
    X2  := c2 + a2*A + U2                                 (both regimes)
    X3  := c3 + beta*X1 + gamma*(A*X1) + eta*X2 + beta_a*A + U3

    ``gamma`` is HARD-ZERO by column omission in the additive control; ``beta_a*A``
    is carried in BOTH regimes. U1, U2, U3 ~ N(0, sigma^2). Defaults keep X₁ and
    X₂ exchangeable in the additive control (a1 == a2, beta == eta).

    Parameters
    ----------
    regime: {"additive", "effect_modifying"}; else ValueError.
    a1, a2: A→X₁ / A→X₂ root-child slopes.
    beta, eta: X₁→X₃ / X₂→X₃ collider-parent coefficients.
    gamma: effect-modifying A×X₁ interaction (effect_modifying only).
    beta_a: A→X₃ main effect, both regimes.
    c1, c2, c3: intercepts (default 0.0).
    sigma: std of U1, U2, U3.
    name: optional label (descriptive default if None).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X3 := c3 + beta*X1 + eta*X2 + beta_a*A + U3.
        # γ hard-zero by OMISSION of the A·X1 term (by omission, not
        # fit-then-discard); beta_a*A carried so A→X3 exists identically to the
        # effect-modifying regime — the two regimes match on graph.
        def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                c3
                + beta * parents["X1"]
                + eta * parents["X2"]
                + beta_a * parents["A"]
                + noise
            )
    else:  # effect_modifying
        # X3 := c3 + beta*X1 + gamma*(A*X1) + eta*X2 + beta_a*A + U3.
        # A modifies X3's X1-coefficient only: effective slope on X1 is
        # (beta + gamma*A), group-differential; X2's coefficient eta and the A
        # main effect beta_a are unchanged (channel (i), lowest-index — the SCM specification).
        def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                c3
                + beta * parents["X1"]
                + gamma * (parents["A"] * parents["X1"])
                + eta * parents["X2"]
                + beta_a * parents["A"]
                + noise
            )

    equations = {
        "A": _eq_A,
        "X1": _eq_root_child(c1, a1),
        "X2": _eq_root_child(c2, a2),
        "X3": _eq_X3,
    }
    return SCM(
        graph=_collider_graph(),
        equations=equations,
        noise_samplers=_make_noise_samplers(sigma),
        regime=regime,
        is_anm=True,
        name=name or f"linear-collider[{regime}]",
    )


def collider_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the LINEAR collider's estimated equations.

    # Form-template-known L1-oracle. [the L1-oracle form-template-known
    # semantics; the SCM specification]
    # The template (which parents enter; whether an A×parent interaction is
    # present) is SUPPLIED to the estimator, which fits ONLY the coefficients. It
    # lives HERE, next to the true collider builders, so the supplied form and the
    # true equations sit in one file and cannot silently drift apart (the SCM
    # convention). The form spec EXISTS for the collider.

    # One equation per non-root endogenous node; interaction explicit.
    # [skill:scm-specification; attribute regimes; the SCM specification]
    #   A  — exogenous root, NO fit (roots are absent from the FormSpec).
    #   X1 — parents {A}:      X1 = α₀ + α₁·A + ε₁. Identical in both regimes:
    #        a single parent admits no interaction.
    #   X2 — parents {A}:      X2 = α'₀ + α'₁·A + ε₂. Identical in both regimes.
    #   X3 — parents {A, X1, X2}:
    #        X3 = β₀ + β·X1 + η·X2 + β_A·A + γ·(A·X1) + ε₃, with γ HARD-ZERO in the
    #        additive regime (OMISSION of the A·X1 column, not fit-then-discard)
    #        and γ estimated in the effect-modifying regime. β_A (the A main-effect
    #        term) is present in BOTH regimes.

    Linear family only; the nonlinear-Gaussian template lands with the NLG
    estimator behind the same pluggable interface.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    x3_interactions = () if regime == "additive" else (("A", "X1"),)
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A",)),
        "X3": NodeForm(
            node="X3", parents=("A", "X1", "X2"), interactions=x3_interactions
        ),
    }


def _make_nlg_noise_samplers(sigma1: float, sigma2: float, sigma3: float) -> dict:
    """Noise samplers with a PER-NODE sigma (A draws the centered ±1 indicator).

    The linear collider's `_make_noise_samplers` takes ONE sigma because all three
    of its structural noises share a scale. The NLG collider does not: the SCM specification
    [Extended] sets σ₁ = σ₂ = 0.4 on the tanh INPUTS (they are what the
    non-saturation constraint |α| + 2σ ≤ 1.5 constrains) and σ₃ = 1.0 on the
    collider node, whose noise enters no tanh. A separate factory rather than a
    signature change to the linear one — the linear cells are frozen production.
    """

    def _sample_A(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice([-1.0, 1.0], size=n)

    def _gaussian(sigma: float):
        def _f(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.normal(0.0, sigma, size=n)

        return _f

    return {
        "A": _sample_A,
        "X1": _gaussian(sigma1),
        "X2": _gaussian(sigma2),
        "X3": _gaussian(sigma3),
    }


def _nlg_collider_jacobian(
    regime: str, *, beta: float, gamma: float, eta: float
):
    """Analytic endogenous Jacobian B(g; u) for the NLG collider.

    Closed-form derivatives of `make_nonlinear_gaussian_collider`'s equations,
    evaluated at (u, A = g) — not a finite-difference probe. Validated against a
    central finite difference in tests/test_s_of_g_nlg_collider.py; co-located
    with its builder so derivative and truth cannot drift apart.

    B is indexed over the ENDOGENOUS nodes only (PS-3 restricts M̃ to Desc(A) on
    both axes): 3×3 over (X1, X2, X3), strictly lower-triangular, live slots
    [X3, X1] and [X3, X2]; γ = 0 in the additive regime.

    Returns
    -------
    A callback ``jacobian(u, group, scm) -> ndarray`` of shape ``(n, k, k)``,
    matching `_nlg_triangle_jacobian`'s contract (descriptor/s_of_g.py discovers
    it as the SCM's optional ``jacobian_fn`` and checks the shape).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    def _jacobian(u: dict[str, np.ndarray], group: float, scm: SCM) -> np.ndarray:
        x1_noise = np.asarray(u["X1"], dtype=float)
        x2_noise = np.asarray(u["X2"], dtype=float)
        n = x1_noise.shape[0]
        a_column = np.full(n, float(group))
        # X1, X2 from the SCM's OWN equation closures — the Jacobian differentiates
        # the equations actually in play, so only the DERIVATIVE is written here.
        x1 = np.asarray(scm.equations["X1"]({"A": a_column}, x1_noise), dtype=float)
        x2 = np.asarray(scm.equations["X2"]({"A": a_column}, x2_noise), dtype=float)

        # γ hard-zero in the additive control, so the X1 gain is β there and
        # (β + γ·A) — group-differential — under effect modification (the SCM channel
        # (i), lowest-index convention). η is NOT group-differential in either
        # regime: A modifies X₃'s X₁-coefficient ONLY.
        x1_gain = beta if regime == "additive" else beta + gamma * float(group)

        # sech²(x) as 1/cosh(x)² and NOT (1 − tanh(x)²): the latter cancels two
        # nearly-equal numbers once |x| grows and bleeds relative precision. The
        # frozen coefficients keep the operating point OUT of saturation
        # (|α| + 2σ = 1.4 ≤ 1.5, the SCM specification), but the MC integrates
        # over the whole group marginal, whose tails do reach it — and the NLG S(g)
        # precision guard is a standing convention, not a per-cell judgement call.
        d_x3_d_x1 = x1_gain / np.cosh(x1) ** 2
        d_x3_d_x2 = eta / np.cosh(x2) ** 2

        endog = [node for node in scm.nodes if scm.parents(node)]
        B = np.zeros((n, len(endog), len(endog)))
        B[:, endog.index("X3"), endog.index("X1")] = d_x3_d_x1
        B[:, endog.index("X3"), endog.index("X2")] = d_x3_d_x2
        return B

    return _jacobian


def make_nonlinear_gaussian_collider(
    regime: str,
    *,
    alpha1: float = 0.6,
    alpha2: float = 0.6,
    sigma1: float = 0.4,
    sigma2: float = 0.4,
    beta: float = 0.7,
    eta: float = 0.7,
    gamma: float = 0.3,
    beta_A: float = 1.0,
    sigma3: float = 1.0,
    beta0: float = 0.0,
    name: str | None = None,
) -> SCM:
    """Build a NONLINEAR-GAUSSIAN collider SCM in the given attribute regime.

    Same graph as `make_linear_collider`, so the family is the only axis that
    moves (module docstring: shared conventions). Noise stays additive Gaussian
    (is_anm=True), so abduction remains valid despite the nonlinear mechanism.

    Structural equations (tanh nonlinearity)
    ----------------------------------------
    A   := U_A                                            (±1 centered root)
    X1  := α₁·A + U₁                                      (both regimes)
    X2  := α₂·A + U₂                                      (both regimes)
    X3  := β₀ + β·tanh(X1) + γ·(A·tanh(X1)) + η·tanh(X2) + β_A·A + U₃

    ``gamma`` is HARD-ZERO by column OMISSION in the additive control; ``eta`` and
    ``beta_A`` are carried in BOTH regimes. Defaults keep the channels
    exchangeable in the additive control (α₁ == α₂, σ₁ == σ₂, β == η) and satisfy
    the non-saturation constraint (the NLG family specification).

    Parameters
    ----------
    regime: {"additive", "effect_modifying"}; else ValueError.
    alpha1, alpha2: A→X₁ / A→X₂ root-child slopes.
    sigma1, sigma2: std of U₁, U₂ (tanh inputs; non-saturation-bound).
    beta, eta: gains on tanh(X1) / tanh(X2) in X₃'s equation.
    gamma: effect-modifying A×tanh(X1) gain (effect_modifying only).
    beta_A: A→X₃ main effect, both regimes.
    sigma3: std of U₃ (no tanh; unconstrained).
    beta0: X₃'s intercept (default 0.0).
    name: optional label (descriptive default if None).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X3 := β₀ + β·tanh(X1) + η·tanh(X2) + β_A·A + U₃.
        # γ hard-zero by OMISSION of the A·tanh(X1) term; β_A·A carried so A→X₃
        # exists identically to the effect-modifying regime (graph-matched, the SCM specification).
        def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                beta0
                + beta * np.tanh(parents["X1"])
                + eta * np.tanh(parents["X2"])
                + beta_A * parents["A"]
                + noise
            )
    else:  # effect_modifying
        # X3 := β₀ + β·tanh(X1) + γ·(A·tanh(X1)) + η·tanh(X2) + β_A·A + U₃.
        # A modifies X₃'s tanh(X1) gain only: the effective gain is (β + γ·A),
        # group-differential; η and β_A are unchanged (channel (i), lowest index).
        def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                beta0
                + beta * np.tanh(parents["X1"])
                + gamma * (parents["A"] * np.tanh(parents["X1"]))
                + eta * np.tanh(parents["X2"])
                + beta_A * parents["A"]
                + noise
            )

    equations = {
        "A": _eq_A,
        "X1": _eq_root_child(0.0, alpha1),
        "X2": _eq_root_child(0.0, alpha2),
        "X3": _eq_X3,
    }
    scm = SCM(
        graph=_collider_graph(),
        equations=equations,
        noise_samplers=_make_nlg_noise_samplers(sigma1, sigma2, sigma3),
        regime=regime,
        is_anm=True,
        name=name or f"nonlinear-gaussian-collider[{regime}]",
    )
    # [the NLG family specification] Same DISCOVERY PATTERN as the NLG triangle: an OPTIONAL
    # `jacobian_fn` ATTRIBUTE attached by the builder that owns the equations.
    # `descriptor/s_of_g.py` reads it with getattr and refuses (never falls back to
    # the linear branch) when a nonlinear SCM carries none.
    scm.jacobian_fn = _nlg_collider_jacobian(regime, beta=beta, gamma=gamma, eta=eta)
    return scm


def nlg_collider_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the NONLINEAR-GAUSSIAN collider.

    # L1-oracle template convention: the template is SUPPLIED (parents, fixed
    # transform, interaction presence) and the estimator fits ONLY the
    # coefficients; columns mirror the builder above TERM BY TERM (linear-in-
    # parameters once tanh(·) is applied, so unregularized OLS carries over).
    # Co-located with its builder so template and truth cannot drift apart.
    # Unlike the NLG triangle template, the A main effect is present in BOTH
    # regimes (the builder carries β_A·A in both). The intercept is estimated
    # in both regimes (fit_intercept=True); here the truth has one (β₀).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    a_main = TermSpec((Factor("A"),))
    tanh_x1 = TermSpec((Factor("X1", TANH),))
    tanh_x2 = TermSpec((Factor("X2", TANH),))
    a_tanh_x1 = TermSpec((Factor("A"), Factor("X1", TANH)))
    x3_terms = (
        (tanh_x1, tanh_x2, a_main)
        if regime == "additive"
        else (tanh_x1, a_tanh_x1, tanh_x2, a_main)
    )
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A",)),
        "X3": NodeForm(node="X3", parents=("A", "X1", "X2"), terms=x3_terms),
    }
