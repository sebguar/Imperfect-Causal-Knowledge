"""Concrete CHAIN ground-truth SCMs (linear + nonlinear-Gaussian families).

Builder functions that CONSTRUCT AND RETURN configured `SCM` instances (base.py,
convention: no subclassing), paralleling `colliders.py`. Two functional families
× two attribute regimes share one graph:

    Chain (a″) over A, X1, X2, X3 with edges
        A -> X1,  A -> X2,  A -> X3,  X1 -> X2 -> X3.

A is the protected root (Gender), with nonzero downstream influence (convention
1). X2 is the MEDIATOR; X₃'s only endogenous parent is X₂, and the composed
path X1→X2→X3 is this topology's contribution (PS-3 convention (1); see the
chain SCM specification). The regime is DECLARED at construction (convention
2). Additive-noise throughout (is_anm=True, convention 5), so abduction
(u = x - f(parents, 0)) is valid.

`gamma` (on X₂) is the effect-modifying channel on the UPSTREAM interior edge —
placed by register-logged design, not symmetry (the chain SCM specification).
Hard-zero in the additive control by COLUMN OMISSION; `beta_a2*A` and
`beta_a3*A` are carried in BOTH regimes, so the graph is matched across
regimes. Coefficient and noise scales match the other topologies; induced
variances differ (raw-units convention), are never silently retuned (PS-2),
and cross-topology comparison is pattern-only (PS-8).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from icknowledge.estimation.forms import TANH, Factor, FormSpec, NodeForm, TermSpec
from icknowledge.scm.base import SCM

_VALID_REGIMES = frozenset({"additive", "effect_modifying"})


def _chain_graph() -> nx.DiGraph:
    """Chain (a″) DiGraph: A->X1, A->X2, A->X3, X1->X2, X2->X3."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "X1", "X2", "X3"])
    graph.add_edges_from(
        [("A", "X1"), ("A", "X2"), ("A", "X3"), ("X1", "X2"), ("X2", "X3")]
    )
    return graph


def _make_noise_samplers(sigma1: float, sigma2: float, sigma3: float) -> dict:
    """Per-node noise samplers (A draws the centered ±1 group indicator).

    Per-node sigmas from the outset, unlike the LINEAR collider's single-sigma
    factory: the chain SCM specification's non-saturation constraint binds σ₁ and σ₂ separately on
    the NLG
    family (they scale the two tanh INPUTS, X₁ and X₂), while σ₃ is unconstrained
    (X₃ feeds no tanh — it is the terminal node). One factory serving both families
    keeps the linear and NLG chains on the same noise plumbing.
    """

    # A ∈ {-1, +1}: centered binary group indicator (keeps additive shifts symmetric).
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


def _eq_A(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
    """Root A ignores parents and returns its own noise (the ±1 draw)."""
    return noise


def _eq_root_child(slope: float):
    """A-child equation X1 := slope*A + U1 (shared across both regimes and families)."""

    def _f(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return slope * parents["A"] + noise

    return _f


def make_linear_chain(
    regime: str,
    *,
    a: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    beta_a2: float = 0.5,
    delta: float = 0.5,
    beta_a3: float = 0.5,
    sigma1: float = 0.5,
    sigma2: float = 0.5,
    sigma3: float = 0.5,
    name: str | None = None,
) -> SCM:
    """Build a LINEAR chain SCM in the given attribute regime.

    Chain (a″) topology A->X1, A->X2, A->X3, X1->X2->X3 (module docstring:
    shared conventions).

    Structural equations
    --------------------
    A   := U_A                                            (±1 centered root)
    X1  := a*A + U1                                       (both regimes)
    X2  := beta*X1 + gamma*(A*X1) + beta_a2*A + U2
    X3  := delta*X2 + beta_a3*A + U3                      (both regimes)

    ``gamma`` is HARD-ZERO by column omission in the additive control;
    ``beta_a2*A`` and ``beta_a3*A`` are carried in BOTH regimes. Default signs
    follow the chain SCM specification's sign convention.

    Parameters
    ----------
    regime: {"additive", "effect_modifying"}; else ValueError.
    a: A→X₁ root-child slope.
    beta: X₁→X₂ upstream interior-edge coefficient.
    gamma: effect-modifying A×X₁ interaction, upstream edge (effect_modifying only).
    beta_a2, beta_a3: A→X₂ / A→X₃ main effects, both regimes.
    delta: X₂→X₃ downstream interior-edge coefficient.
    sigma1, sigma2, sigma3: stds of U1, U2, U3.
    name: optional label (descriptive default if None).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X2 := beta*X1 + beta_a2*A + U2. γ hard-zero by OMISSION of the A·X1 term
        # (hard-zero by omission, not fit-then-discard); beta_a2*A carried so A→X2
        # exists identically to the effect-modifying regime (graph-matched, the chain SCM
        # specification).
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return beta * parents["X1"] + beta_a2 * parents["A"] + noise
    else:  # effect_modifying
        # X2 := beta*X1 + gamma*(A*X1) + beta_a2*A + U2. A modifies X2's X1
        # coefficient only: effective slope (beta + gamma*A), group-differential.
        # delta and beta_a3 downstream are unchanged — γ is on X1→X2 ONLY (the chain SCM
        # specification).
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                beta * parents["X1"]
                + gamma * (parents["A"] * parents["X1"])
                + beta_a2 * parents["A"]
                + noise
            )

    # X3 is IDENTICAL across regimes: γ sits upstream, so X₃'s equation is
    # main-effects-only in both arms (the chain SCM specification's L1-oracle template says the
    # same).
    def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return delta * parents["X2"] + beta_a3 * parents["A"] + noise

    equations = {
        "A": _eq_A,
        "X1": _eq_root_child(a),
        "X2": _eq_X2,
        "X3": _eq_X3,
    }
    return SCM(
        graph=_chain_graph(),
        equations=equations,
        noise_samplers=_make_noise_samplers(sigma1, sigma2, sigma3),
        regime=regime,
        is_anm=True,
        name=name or f"linear-chain[{regime}]",
    )


def chain_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the LINEAR chain's estimated equations.

    # Form-template-known L1-oracle. [the L1-oracle form-template-known semantics; the chain
    # SCM specification]
    # The template (which parents enter; whether an A×parent interaction is
    # present) is SUPPLIED to the estimator, which fits ONLY the coefficients. It
    # lives HERE, next to the true chain builders, so the supplied form and the true
    # equations sit in one file and cannot silently drift apart — the same
    # co-location convention `triangle_form_spec` and `collider_form_spec` follow.
    # NOTE this is why nothing is added to `icknowledge/estimation/`: that package
    # holds the form-template TYPES and the estimators, never a per-topology
    # registry.

    # One equation per non-root endogenous node; interaction explicit.
    #   A  — exogenous root, NO fit (roots are absent from the FormSpec).
    #   X1 — parents {A}:      X1 = α₀ + α₁·A + ε₁. Identical in both regimes:
    #        a single parent admits no interaction.
    #   X2 — parents {A, X1}:  X2 = β₀ + β·X1 + β_{A,2}·A + γ·(A·X1) + ε₂, with γ
    #        HARD-ZERO in the additive regime (OMISSION of the A·X1 column, not
    #        fit-then-discard) and estimated under effect modification.
    #   X3 — parents {A, X2}:  X3 = δ₀ + δ·X2 + β_{A,3}·A + ε₃. MAIN-EFFECTS-ONLY in
    #        BOTH regimes — γ is on the upstream edge X1→X2 only (the chain SCM specification), so
    # X₃'s
    #        equation carries no interaction column in either arm. This is the one
    #        template asymmetry vs. the collider, whose γ sits on the terminal node.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    x2_interactions = () if regime == "additive" else (("A", "X1"),)
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A", "X1"), interactions=x2_interactions),
        "X3": NodeForm(node="X3", parents=("A", "X2")),
    }


def _nlg_chain_jacobian(regime: str, *, beta: float, gamma: float, delta: float):
    """Analytic endogenous Jacobian B(g; u) for the NLG chain.

    Closed-form derivatives of `make_nonlinear_gaussian_chain`'s equations,
    evaluated at (u, A = g) — not a finite-difference probe. Validated against a
    central finite difference in tests/test_chain_scm.py; co-located with its
    builder so derivative and truth cannot drift apart.

    B is indexed over the ENDOGENOUS nodes only (PS-3): 3×3, strictly
    lower-triangular, live slots [X2, X1] and [X3, X2] on a subdiagonal that
    COMPOSES — (I − B)⁻¹ carries [X3, X1] = [X2,X1]·[X3,X2] (PS-3 convention (1));
    γ = 0 in the additive regime.

    Returns
    -------
    A callback ``jacobian(u, group, scm) -> ndarray`` of shape ``(n, k, k)``,
    matching `_nlg_collider_jacobian`'s contract (descriptor/s_of_g.py discovers
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
        # X1 and X2 from the SCM's OWN equation closures — the Jacobian
        # differentiates the equations actually in play, so only the DERIVATIVE is
        # written here. X2 needs X1, which is the chain's serial structure showing
        # up in the Jacobian evaluation itself (the collider could evaluate its two
        # tanh inputs independently; the chain cannot).
        x1 = np.asarray(scm.equations["X1"]({"A": a_column}, x1_noise), dtype=float)
        x2 = np.asarray(
            scm.equations["X2"]({"A": a_column, "X1": x1}, x2_noise), dtype=float
        )

        # γ hard-zero in the additive control, so the X1 gain is β there and
        # (β + γ·A) — group-differential — under effect modification (the chain SCM specification's
        # upstream channel). δ is NOT group-differential in either regime.
        x1_gain = beta if regime == "additive" else beta + gamma * float(group)

        # sech²(x) as 1/cosh(x)² and NOT (1 − tanh(x)²): the latter cancels two
        # nearly-equal numbers once |x| grows and bleeds relative precision. The
        # frozen coefficients keep the operating point OUT of saturation (the chain SCM
        # specification's
        # non-saturation constraint), but the MC integrates over the whole group
        # marginal, whose tails do reach it — and the NLG S(g) precision guard is a
        # standing convention, not a per-cell judgement call.
        d_x2_d_x1 = x1_gain / np.cosh(x1) ** 2
        d_x3_d_x2 = delta / np.cosh(x2) ** 2

        endog = [node for node in scm.nodes if scm.parents(node)]
        B = np.zeros((n, len(endog), len(endog)))
        B[:, endog.index("X2"), endog.index("X1")] = d_x2_d_x1
        B[:, endog.index("X3"), endog.index("X2")] = d_x3_d_x2
        return B

    return _jacobian


def make_nonlinear_gaussian_chain(
    regime: str,
    *,
    a: float = 0.6,
    beta: float = 0.6,
    gamma: float = 0.4,
    beta_a2: float = 0.3,
    delta: float = 0.6,
    beta_a3: float = 0.4,
    sigma1: float = 0.4,
    sigma2: float = 0.3,
    sigma3: float = 0.4,
    name: str | None = None,
) -> SCM:
    """Build a NONLINEAR-GAUSSIAN chain SCM in the given attribute regime.

    Same graph as `make_linear_chain`, so the family is the only axis that moves
    (module docstring: shared conventions). Noise stays additive Gaussian
    (is_anm=True), so abduction remains valid despite the nonlinear mechanism.

    Structural equations (tanh nonlinearity)
    ----------------------------------------
    A   := U_A                                            (±1 centered root)
    X1  := a·A + U₁                                       (both regimes; A is a
                                                           root, so NO tanh here)
    X2  := β·tanh(X1) + γ·(A·tanh(X1)) + β_{A,2}·A + U₂
    X3  := δ·tanh(X2) + β_{A,3}·A + U₃                    (both regimes)

    ``gamma`` is HARD-ZERO by column OMISSION in the additive control; ``beta_a2``
    and ``beta_a3`` are carried in BOTH regimes. The defaults satisfy the
    non-saturation constraint on BOTH tanh inputs (the NLG family specification;
    verified empirically in tests/test_chain_scm.py). σ₃ is unconstrained — X₃ is
    terminal and feeds no tanh.

    Parameters
    ----------
    regime: {"additive", "effect_modifying"}; else ValueError.
    a: A→X₁ root-child slope (tanh input; non-saturation-bound with sigma1).
    beta, delta: gains on tanh(X1) in X₂'s equation / tanh(X2) in X₃'s.
    gamma: effect-modifying A×tanh(X1) gain, upstream edge (effect_modifying only).
    beta_a2, beta_a3: A→X₂ / A→X₃ main effects, both regimes.
    sigma1, sigma2: stds of U₁, U₂ (tanh inputs; non-saturation-bound).
    sigma3: std of U₃ (terminal; unconstrained).
    name: optional label (descriptive default if None).
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )

    if regime == "additive":
        # X2 := β·tanh(X1) + β_{A,2}·A + U₂. γ hard-zero by OMISSION of the
        # A·tanh(X1) term; β_{A,2}·A carried so A→X₂ is graph-matched (the chain SCM specification).
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                beta * np.tanh(parents["X1"]) + beta_a2 * parents["A"] + noise
            )
    else:  # effect_modifying
        # X2 := β·tanh(X1) + γ·(A·tanh(X1)) + β_{A,2}·A + U₂. A modifies X₂'s
        # tanh(X1) gain only: effective gain (β + γ·A), group-differential.
        def _eq_X2(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return (
                beta * np.tanh(parents["X1"])
                + gamma * (parents["A"] * np.tanh(parents["X1"]))
                + beta_a2 * parents["A"]
                + noise
            )

    # X3 IDENTICAL across regimes — γ is upstream (the chain SCM specification), so the terminal
    # node's
    # equation carries no interaction in either arm.
    def _eq_X3(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        return delta * np.tanh(parents["X2"]) + beta_a3 * parents["A"] + noise

    equations = {
        "A": _eq_A,
        "X1": _eq_root_child(a),
        "X2": _eq_X2,
        "X3": _eq_X3,
    }
    scm = SCM(
        graph=_chain_graph(),
        equations=equations,
        noise_samplers=_make_noise_samplers(sigma1, sigma2, sigma3),
        regime=regime,
        is_anm=True,
        name=name or f"nonlinear-gaussian-chain[{regime}]",
    )
    # [the NLG family specification] Same DISCOVERY PATTERN as the NLG triangle and collider: an
    # OPTIONAL `jacobian_fn` ATTRIBUTE attached by the builder that owns the
    # equations. `descriptor/s_of_g.py` reads it with getattr and refuses (never
    # falls back to the linear branch) when a nonlinear SCM carries none — which is
    # why nothing is registered in that module for the chain.
    scm.jacobian_fn = _nlg_chain_jacobian(
        regime, beta=beta, gamma=gamma, delta=delta
    )
    return scm


def nlg_chain_form_spec(regime: str) -> FormSpec:
    """Supplied parametric form template for the NONLINEAR-GAUSSIAN chain.

    # L1-oracle template convention: the template is SUPPLIED (parents, fixed
    # transform, interaction presence) and the estimator fits ONLY the
    # coefficients; columns mirror the builder above TERM BY TERM (linear-in-
    # parameters once tanh(·) is applied, so unregularized OLS carries over).
    # Co-located with its builder so template and truth cannot drift apart.
    # No interaction column on X₃ in either arm: γ sits on the UPSTREAM edge.
    # The intercept is estimated regardless (fit_intercept=True); the truth has
    # none, so it is a harmless nuisance.
    """
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime must be one of {sorted(_VALID_REGIMES)}, got {regime!r}."
        )
    a_main = TermSpec((Factor("A"),))
    tanh_x1 = TermSpec((Factor("X1", TANH),))
    tanh_x2 = TermSpec((Factor("X2", TANH),))
    a_tanh_x1 = TermSpec((Factor("A"), Factor("X1", TANH)))
    x2_terms = (
        (tanh_x1, a_main)
        if regime == "additive"
        else (tanh_x1, a_tanh_x1, a_main)
    )
    return {
        "X1": NodeForm(node="X1", parents=("A",)),
        "X2": NodeForm(node="X2", parents=("A", "X1"), terms=x2_terms),
        "X3": NodeForm(node="X3", parents=("A", "X2"), terms=(tanh_x2, a_main)),
    }
