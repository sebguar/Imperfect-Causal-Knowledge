"""S(g), the pre-committed structural descriptor for H4 (PS-3).

Its own subpackage: S(g) is a function OF an SCM but not a property of one (it
needs V_h and S), and it is data-, classifier-, and fit-independent BY
CONSTRUCTION (PS-3) — so it sits where it cannot reach aggregation outputs,
fitted coefficients, or manifest data.

Definition (PS-3, verbatim structure)
-------------------------------------
    S(g) := √( E_{u | A = g} [ ‖ M̃(g; u) − 𝟙 ‖²_F ] )

with M(g) := (I − B(g))⁻¹ the total-effect propagation matrix, B(g)[j, i] =
∂f_j/∂x_i evaluated at A = g, M̃(g) the restriction of M(g) to rows
j ∈ V_h ∩ Desc(A) and columns i ∈ S ∩ Desc(A), and 𝟙 the entry-wise indicator
[j = i] over those index sets.

Two branches, selected by a VERIFIED affineness test on the SCM's own
equations, never by a declared family flag: LINEAR (deterministic Frobenius
value) and NONLINEAR-GAUSSIAN (Monte Carlo over the builder's analytic
Jacobian at a pinned seed). A nonlinear SCM with no analytic Jacobian is
REFUSED (NotImplementedError), never silently approximated.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from icknowledge.scm.base import SCM
from icknowledge.utils.seeding import EXPERIMENT_META_ENTROPY, spawn_children

NLG_NOT_IMPLEMENTED = (
    "S(g) on a NONLINEAR cell needs the analytic local Jacobian B(g; u) (PS-3 / "
    "the NLG family specification), which this SCM does not supply: it carries no `jacobian_fn` "
    "attribute."
)

#: Monte-Carlo sample size for the NLG branch's E_{u|A=g}[·].
#: [the NLG family specification] n_mc = 10,000 draws of the group-g exogenous vector — pinned in
#: the
#: register, so it is a named constant here rather than a literal at the call site.
S_OF_G_MC_SAMPLES = 10_000

#: [the NLG family specification] SEED — one CONSTRUCTION-TIME child stream labelled `sg_mc`,
#: spawned
#: ONCE from the seed-generation meta-entropy (20260710). Emphatically NOT a per-run stream:
#: S(g) is data-, classifier- and fit-independent by construction (PS-3), so
#: redrawing it per experiment seed would make the descriptor move with the run and
#: silently reintroduce exactly the fit-dependence PS-3 rejected. Because the seed is
#: fixed at MODULE level and a fresh Generator is built from it on every call, repeated
#: `s_of_g` calls are bit-identical — no RNG state advances across calls.
SG_MC_STREAM_LABEL = "sg_mc"
#: The `sg_mc` stream is a SIBLING BRANCH of the run-seed line, not a member of it.
#: `experiment_run_seeds` consumes `SeedSequence(meta).spawn(N)`, whose children carry
#: spawn_key (0,), (1,), … — so spawning a child the ordinary way would hand S(g) the
#: byte-identical stream of run seed 0. Deriving the branch from the label's own bytes
#: gives a spawn_key that the run-seed line can never reach at any N (PS-1 caps it at 20).
_SG_MC_SPAWN_KEY = tuple(SG_MC_STREAM_LABEL.encode("ascii"))
S_OF_G_MC_SEED: int = int(
    np.random.SeedSequence(
        entropy=EXPERIMENT_META_ENTROPY, spawn_key=_SG_MC_SPAWN_KEY
    ).generate_state(1, dtype=np.uint32)[0]
)

#: Parent values used to VERIFY that each equation is affine in its endogenous
#: parents at A = g (see `_structural_adjacency`). Deliberately asymmetric and of
#: mixed magnitude: an odd nonlinearity such as tanh has tanh(1) − tanh(0) ==
#: tanh(0) − tanh(−1), so symmetric probe points around zero would let it pass a
#: second-difference test by construction. Offsetting per parent index also varies
#: the parents against each other, so a parent–parent product term (zero under
#: one-at-a-time unit bumps) cannot slip through either.
_PROBE_VALUES = (2.0, -3.0, 0.5, 4.0, -1.5, 1.25)
_N_PROBES = len(_PROBE_VALUES)

#: Affineness tolerance. The linear builders evaluate in exact float arithmetic, so a
#: genuine linear cell agrees to ~1e-15 relative; anything above this is a real
#: functional-form departure, not accumulated rounding.
_AFFINE_RTOL = 1e-9
_AFFINE_ATOL = 1e-9


class _NonAffineEquation(Exception):
    """Internal signal: an equation failed the affineness probe — route to the NLG branch.

    Private and never raised out of this module: `s_of_g` catches it and either takes
    the NLG branch (SCM supplies an analytic Jacobian) or re-raises it as the public
    NotImplementedError. It exists so the linear branch's affineness verification stays
    a single expression of one idea, with the branch decision made once, at the top.
    """

    def __init__(self, node: str, group: float, deviation: float) -> None:
        super().__init__(
            f"Equation for node {node!r} is not affine in its endogenous parents at "
            f"A={group:g} (max superposition deviation {deviation:.3e}), so ∂f_j/∂x_i "
            "is not constant in u and the linear branch's collapse of E_{u|A=g}[·] to "
            "a point value does not hold."
        )


def _endogenous_nodes(scm: SCM) -> list[str]:
    """Endogenous (non-root) nodes in TOPOLOGICAL order.

    # [PS-3] The protected root A is excluded from BOTH index sets automatically:
    # the restriction is to V_h ∩ Desc(A) and S ∩ Desc(A), and A ∉ Desc(A). This is
    # load-bearing, not incidental — were A admitted, its total-effect column (the
    # A → X_j paths, including the β_A main effect on the collider) would fold "A
    # shifts levels" into a descriptor meant to measure "A modifies an endogenous
    # edge", and the additive control arm would stop being flat, contradicting H4's
    # design. Topological order also makes B strictly lower-triangular, hence
    # nilpotent, so (I − B) is always invertible on a DAG.
    """
    return [node for node in scm.nodes if scm.parents(node)]


def _evaluate_at(
    scm: SCM,
    node: str,
    values: dict[str, np.ndarray],
    n: int,
    group: float,
    protected: str,
) -> np.ndarray:
    """f_node(parents, noise = 0) with A pinned at g, other roots at 0, over ``n`` rows.

    Parents absent from ``values`` are held at their base level. Noise is zero: under
    the additive-noise convention (is_anm) it would cancel in every difference taken
    below anyway, and f(pa, 0) is the mean function `abduct` already inverts.
    """
    columns = {
        p: values.get(p, np.full(n, group if p == protected else 0.0))
        for p in scm.parents(node)
    }
    return np.asarray(scm.equations[node](columns, np.zeros(n)), dtype=float)


def _structural_adjacency(
    scm: SCM, group: float, protected: str
) -> tuple[np.ndarray, list[str]]:
    """B(g)[j, i] = ∂f_j/∂x_i at A = g over the endogenous nodes, plus its index.

    # IMPLEMENTATION PATH — finite-difference (unit-bump) probe of the equation
    # closures, NOT analytic coefficient reading. [PS-3: "B(g)[j, i] = ∂f_j/∂x_i"]
    # `SCM` stores each structural equation as an opaque `Callable[[parents, noise]]`
    # closure (scm/base.py) and exposes NO coefficient table on the linear family —
    # `make_linear_triangle`'s a/b/g and `make_linear_collider`'s β/γ/η live only in
    # the closures' cell variables. Probing the realizer is therefore the only path
    # that reads the SCM actually in play rather than a parallel transcription of it,
    # which is also the stronger tether: a coefficient table could drift from the
    # equations it claims to describe, a probe cannot.
    #
    # For an affine f_j the derivative is constant, so the base point is irrelevant —
    # which is exactly why the linear branch needs no expectation over u (PS-3) and
    # why noise is passed as zero and non-A root parents (none in the triangle or
    # collider) as zero. That constancy is not assumed: it is VERIFIED below, and the
    # verification is what gates the NLG refusal.
    """
    endog = _endogenous_nodes(scm)
    index = {node: k for k, node in enumerate(endog)}
    B = np.zeros((len(endog), len(endog)))

    for node in endog:
        endogenous_parents = [p for p in scm.parents(node) if p in index]

        base = float(_evaluate_at(scm, node, {}, 1, group, protected)[0])
        for parent in endogenous_parents:
            bumped = _evaluate_at(scm, node, {parent: np.ones(1)}, 1, group, protected)
            B[index[node], index[parent]] = float(bumped[0]) - base

        if not endogenous_parents:
            continue

        # -- affineness verification == the linear/NLG gate -----------------------
        # Superposition must hold at points the unit bumps never visited: predicted
        # = f(0) + Σ_i (∂f/∂x_i)·v_i. tanh, a parent–parent product, or any other
        # curvature breaks this. The equations are vectorized over rows, so all
        # probe points are evaluated in ONE call.
        probes = np.array(
            [
                [_PROBE_VALUES[(t + i) % _N_PROBES] for i in range(len(endogenous_parents))]
                for t in range(_N_PROBES)
            ]
        )
        actual = _evaluate_at(
            scm,
            node,
            {p: probes[:, i] for i, p in enumerate(endogenous_parents)},
            _N_PROBES,
            group,
            protected,
        )
        slopes = np.array([B[index[node], index[p]] for p in endogenous_parents])
        predicted = base + probes @ slopes
        if not np.allclose(actual, predicted, rtol=_AFFINE_RTOL, atol=_AFFINE_ATOL):
            raise _NonAffineEquation(
                node, group, float(np.max(np.abs(actual - predicted)))
            )

    return B, endog


# ===========================================================================
# NLG branch — analytic local Jacobian (the NLG family specification) + Monte Carlo over u | A = g
# ===========================================================================


def _exogenous_draws(scm: SCM, n: int, group: float, protected: str) -> dict[str, np.ndarray]:
    """``n`` draws of the exogenous vector u | A = g, from the SCM's OWN noise samplers.

    # [the NLG family specification / PS-3 convention (5)] The same `noise_samplers` that
    # `scm.sample()`
    # uses, mapped to per-node child streams in topological order exactly as
    # `SCM._child_rngs` does — so the MC integrates the distribution the experiment
    # actually draws from, not a parallel description of it. The expectation is over the
    # FULL group-g exogenous vector: NON-A ROOTS ARE SAMPLED, never held at a base point
    # (PS-3). Moot on the triangle and collider — A is the only
    # root — and load-bearing the first time a topology has another one.
    #
    # A's own entry is PINNED at g rather than drawn: A := U_A, so conditioning the
    # exogenous distribution on A = g IS fixing its exogenous entry. Its stream is still
    # spawned and discarded, keeping every other node's draws independent of whether A
    # is pinned.
    """
    rngs = spawn_children(S_OF_G_MC_SEED, len(scm.nodes))
    u = {
        node: np.asarray(scm.noise_samplers[node](rng, n), dtype=float)
        for node, rng in zip(scm.nodes, rngs, strict=True)
    }
    u[protected] = np.full(n, float(group))
    return u


def _jacobian_samples(
    scm: SCM, group: float, protected: str, endog: list[str]
) -> np.ndarray:
    """B(g; u_1), …, B(g; u_n_mc) stacked as ``(n_mc, k, k)`` over the endogenous nodes.

    # [the NLG family specification] The Jacobian is ANALYTIC PER NLG BUILDER, discovered as an
    # optional
    # `jacobian_fn` attribute on the SCM (see `make_nonlinear_gaussian_triangle`), and
    # is NOT approximated by finite differences here — FD lives in the test suite as the
    # validation tether, not in the production path.
    """
    jacobian_fn = getattr(scm, "jacobian_fn", None)
    if jacobian_fn is None:
        raise NotImplementedError(NLG_NOT_IMPLEMENTED)

    u = _exogenous_draws(scm, S_OF_G_MC_SAMPLES, group, protected)
    B = np.asarray(jacobian_fn(u, group, scm), dtype=float)
    expected = (S_OF_G_MC_SAMPLES, len(endog), len(endog))
    if B.shape != expected:
        raise ValueError(
            f"{scm.name!r}'s jacobian_fn returned shape {B.shape}, expected {expected} "
            f"— rows/columns must be the endogenous nodes {endog} in topological order."
        )
    return B


def s_of_g(
    scm: SCM,
    group: float,
    *,
    V_h: Sequence[str],
    S: Sequence[str],
    protected: str = "A",
) -> float:
    """S(g) for one group (PS-3). Linear branch closed-form, NLG branch by MC.

    Parameters
    ----------
    scm: the ground-truth `SCM` the run was built from — no wrapper type.
    group: the protected attribute's value, −1 or +1.
    V_h: the classifier's feature set — ROW index set V_h ∩ Desc(A).
    S: the actionable set — COLUMN index set S ∩ Desc(A). Keyword-only with
        ``V_h`` so rows and columns cannot be swapped positionally (they coincide
        on the triangle and collider, so a swap would be silent there).
    protected: name of the protected root A (default "A"); needed for Desc(A).

    Returns
    -------
    float — LINEAR cells: the deterministic Frobenius value; NONLINEAR cells: the
    Monte-Carlo RMS over ``S_OF_G_MC_SAMPLES`` draws of u | A = g at the pinned
    `sg_mc` seed.

    Raises
    ------
    NotImplementedError — nonlinear SCM with no analytic `jacobian_fn`; the
    linear branch is never a fallback (it would be confidently wrong with no
    runtime signal).

    Notes
    -----
    # The five locked PS-3 conventions (total-effect matrix, Frobenius, identity
    # removed, RMS over the group exogenous distribution, full group marginal)
    # are stated in the S(g) specification. [PS-8] Levels are topology-scaled;
    # cross-topology reading happens downstream and is pattern-only.
    """
    if float(group) not in (-1.0, 1.0):
        raise ValueError(
            f"group must be -1 or +1 (PS-3: A ∈ {{−1, +1}}), got {group!r}."
        )
    graph = scm.graph
    if protected not in graph:
        raise ValueError(f"protected node {protected!r} is not in the SCM graph.")
    if scm.parents(protected):
        raise ValueError(
            f"protected node {protected!r} must be a ROOT (PS-3); it has parents "
            f"{scm.parents(protected)}."
        )
    unknown = sorted({*V_h, *S} - set(graph.nodes()))
    if unknown:
        raise ValueError(f"V_h / S reference unknown node(s): {unknown}.")

    # -- BRANCH SELECTION. Decided by VERIFIED affineness of the SCM's own equations,
    # not by a declared family flag: a cell mislabelled linear would otherwise take the
    # point-value path and return a wrong number with no runtime signal. `B` carries a
    # leading MC axis on the NLG branch and none on the linear branch; everything below
    # is written once against both via `...` indexing.
    try:
        B, endog = _structural_adjacency(scm, float(group), protected)
        nonlinear = False
    except _NonAffineEquation as non_affine:
        endog = _endogenous_nodes(scm)
        try:
            B = _jacobian_samples(scm, float(group), protected, endog)
        except NotImplementedError as missing_jacobian:
            raise NotImplementedError(
                f"{missing_jacobian} {non_affine}"
            ) from non_affine
        nonlinear = True

    # Convention (1): the TOTAL-effect matrix. B is strictly lower-triangular in
    # topological order, so (I − B) is unit-triangular and always invertible — at every
    # MC draw alike, since the Jacobian inherits the graph's acyclicity. `np.linalg.inv`
    # broadcasts over the leading MC axis, so the NLG branch inverts all
    # S_OF_G_MC_SAMPLES matrices in one vectorized call.
    M = np.linalg.inv(np.eye(len(endog)) - B)

    descendants = nx.descendants(graph, protected)
    row_nodes = [n for n in endog if n in set(V_h) & descendants]
    col_nodes = [n for n in endog if n in set(S) & descendants]
    if not row_nodes or not col_nodes:
        # Empty sub-block: nothing of A's propagation is both classifier-visible and
        # actionable, so there is no deviation from the identity to measure.
        return 0.0

    rows = [endog.index(n) for n in row_nodes]
    cols = [endog.index(n) for n in col_nodes]
    # Leading `...` so ONE expression restricts both a bare (k, k) linear M and a
    # stacked (n_mc, k, k) NLG M; row and column selections are taken one axis at a time.
    sub = M[..., rows, :][..., :, cols]
    # Convention (3): subtract the entry-wise indicator [row node == col node] — NOT
    # np.eye, which would be wrong the moment the row and column index sets differ.
    indicator = np.array(
        [[1.0 if rn == cn else 0.0 for cn in col_nodes] for rn in row_nodes]
    )
    deviation = sub - indicator
    if not nonlinear:
        # Convention (2): Frobenius. Convention (4): on the linear branch the RMS over
        # u of a u-constant quantity is that quantity, so no expectation is taken.
        return float(np.linalg.norm(deviation, ord="fro"))
    # Conventions (2) + (4) on the NLG branch: ‖·‖²_F per draw, MEAN over the n_mc
    # draws of u | A = g, then the square root — S(g) := √(E[‖·‖²_F]), the RMS of PS-3.
    # Averaging the SQUARED norms (not the norms) is what makes this the RMS rather
    # than a mean-Frobenius; the two differ exactly by the within-group derivative
    # dispersion that convention (4) exists to capture.
    return float(np.sqrt(np.mean(np.sum(deviation**2, axis=(-2, -1)))))
