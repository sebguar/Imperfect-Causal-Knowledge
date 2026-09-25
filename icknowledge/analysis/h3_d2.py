"""Truth-side check for triangle/linear/effect-modifying (LD-2).

Answers one question: on the frozen triangle/linear/EM builder, is the
systematic 20/20 loss of the ``A—X2`` adjacency what a correctly applied linear
CI test produces on this DGP (class (d)), or a broken harness (class (a))?

TRUTH-SIDE ONLY. Every number is a POPULATION quantity derived in closed form
from the frozen builder's coefficients, read programmatically off
``scm/triangles.py``, never transcribed. No production data is read, nothing
under ``results/`` is touched, and PC-Stable is not re-run. The one run-derived
input is ``N_disc``, which LD-2 requires be READ from the run manifest — see
``analysis/h3_manifests.py``.

PC removes an adjacency as soon as independence is accepted on ANY tested
subset. For the (A, X2) pair the tested subsets are ``{}`` and ``{X1}``, so both
are computed and the OPERATIVE one is that with the SMALLEST ``|rho|``.

With ``A`` a centered +/-1 indicator and no ``A`` main effect in the ``X2``
equation (the builder puts the whole group difference in the slope ``g_of_A``),
``triangle_em_population_correlations`` gives
``Cov(A, X2 | X1) = gbar*a - a*gbar = 0``: the partial correlation
``rho(A, X2 | X1)`` is exactly zero for every admissible parameter value. The
pinned Fisher-z test is not malfunctioning: it is correctly reporting that there
is no *linear* conditional dependence to find.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import numpy as np

from icknowledge.scm.triangles import make_linear_triangle

__all__ = [
    "ALPHA",
    "ALARM_MIN",
    "GENUINE_MAX",
    "D2Result",
    "SubsetCorrelation",
    "TriangleEMCoefficients",
    "d2_fork",
    "fisher_z_rejection_probability",
    "run_d2_check",
    "triangle_em_builder_coefficients",
    "triangle_em_population_correlations",
    "triangle_em_synthetic_correlations",
]

#: [PS-7] The pinned CI-test level, as recorded in every run manifest.
ALPHA = 0.05

#: [LD-2] The three-zone fork, PRE-COMMITTED before any rho existed.
#: GENUINE iff r_op <= 0.5; ALARM iff r_op >= 0.95; INDETERMINATE strictly between.
GENUINE_MAX = 0.5
ALARM_MIN = 0.95

#: [LD-2] The three zone texts.
ZONE_TEXT = {
    "GENUINE": (
        "**GENUINE (class d)** iff r_op ≤ 0.5: the pinned linear test, correctly "
        "applying its linear lens, more-likely-than-not drops an interaction-carried "
        "edge; the systematic 20/20 SHD = 1 is regime-induced skeleton "
        "misspecification, the halt LIFTS, the edge loss is reported as a finding "
        "(H3's misspecification mechanism arriving via the interaction/regime channel "
        "rather than the predicted nonlinearity/family channel)."
    ),
    "ALARM": (
        "**ALARM (class a)** iff r_op ≥ 0.95: a correctly-behaving test keeps the edge "
        "almost always, yet PC dropped it 20/20 — harness alarm CONFIRMED, halt STANDS, "
        "no label, fix in a follow-up adjudication."
    ),
    "INDETERMINATE": (
        "**INDETERMINATE** iff 0.5 < r_op < 0.95: the halt STANDS pending a dedicated "
        "adjudication — an intermediate r_op is itself in tension with a 20/20 "
        "drop and must be examined, not rounded to either branch."
    ),
}


# --------------------------------------------------------------------------- #
# Frozen builder coefficients, read programmatically
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TriangleEMCoefficients:
    """The triangle/linear/effect-modifying structural coefficients.

    Read off ``make_linear_triangle``'s own signature defaults, which is how the
    grid called it (``run_cross_seed_grid`` passes ``regime`` only, per the PS-8
    frozen-sigma verification). Nothing here is transcribed by hand.
    """

    a: float
    g_pos: float
    g_neg: float
    sigma: float

    @property
    def g_bar(self) -> float:
        """Pooled slope ``(g_pos + g_neg)/2`` — the X1 main-effect coefficient."""
        return (self.g_pos + self.g_neg) / 2.0

    @property
    def gamma(self) -> float:
        """Interaction coefficient ``(g_pos - g_neg)/2`` on the ``A·X1`` column."""
        return (self.g_pos - self.g_neg) / 2.0

    @property
    def beta_a(self) -> float:
        """The ``A`` MAIN-EFFECT coefficient in the X2 equation: zero by construction.

        The effect-modifying builder carries no additive ``b*A`` term — the whole
        group difference lives in the slope. This is the fact that makes the
        partial correlation vanish identically.
        """
        return 0.0


def triangle_em_builder_coefficients() -> TriangleEMCoefficients:
    """Read the frozen builder's effect-modifying coefficients programmatically."""
    params = inspect.signature(make_linear_triangle).parameters
    return TriangleEMCoefficients(
        a=float(params["a"].default),
        g_pos=float(params["g_pos"].default),
        g_neg=float(params["g_neg"].default),
        sigma=float(params["sigma"].default),
    )


# --------------------------------------------------------------------------- #
# Population correlations over both tested conditioning sets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubsetCorrelation:
    """One conditioning subset's population correlation and its Fisher-z power."""

    conditioning: tuple[str, ...]
    rho: float
    n_disc: int
    alpha: float
    rejection_prob: float

    @property
    def size(self) -> int:
        """``|S|`` — the number of conditioning variables."""
        return len(self.conditioning)

    @property
    def label(self) -> str:
        return "∅" if not self.conditioning else "{" + ", ".join(self.conditioning) + "}"


def triangle_em_population_correlations(
    coefficients: TriangleEMCoefficients | None = None,
) -> dict[tuple[str, ...], float]:
    """Population ``rho(A, X2)`` and ``rho(A, X2 | X1)`` in closed form.

    Returns a mapping keyed by the conditioning subset. See the module docstring
    for the derivation; the partial is identically zero for every admissible
    parameter value, which the unit tests assert on randomized coefficients.
    """
    c = coefficients or triangle_em_builder_coefficients()
    var_a = 1.0  # A is a centered +/-1 indicator: E[A]=0, E[A^2]=1
    var_x1 = c.a**2 + c.sigma**2
    var_x2 = c.g_bar**2 * (c.a**2 + c.sigma**2) + c.gamma**2 * c.sigma**2 + c.sigma**2
    cov_a_x2 = c.g_bar * c.a
    cov_a_x1 = c.a
    cov_x1_x2 = c.g_bar * var_x1

    rho_marginal = cov_a_x2 / math.sqrt(var_a * var_x2)

    # Partial correlation via residualization on X1.
    resid_cov = cov_a_x2 - cov_a_x1 * cov_x1_x2 / var_x1
    resid_var_a = var_a - cov_a_x1**2 / var_x1
    resid_var_x2 = var_x2 - cov_x1_x2**2 / var_x1
    rho_partial = resid_cov / math.sqrt(resid_var_a * resid_var_x2)

    return {(): rho_marginal, ("X1",): rho_partial}


def triangle_em_synthetic_correlations(
    n: int = 10_000_000,
    seed: int = 0,
    coefficients: TriangleEMCoefficients | None = None,
) -> dict[tuple[str, ...], float]:
    """Large-N corroboration of the closed form, drawn FROM THE BUILDER.

    LD-2 permits "analytic or large-N synthetic FROM THE BUILDER"; the analytic
    route is the operative one and this is only its cross-check. The SCM's own
    equations and noise samplers generate the sample — no equation is re-typed.
    """
    if coefficients is None:
        scm = make_linear_triangle("effect_modifying")
    else:
        scm = make_linear_triangle(
            "effect_modifying",
            a=coefficients.a,
            g_pos=coefficients.g_pos,
            g_neg=coefficients.g_neg,
            sigma=coefficients.sigma,
        )
    frame = scm.sample(n, seed=seed)
    corr = np.corrcoef(
        np.vstack(
            [
                frame["A"].to_numpy(dtype=float),
                frame["X1"].to_numpy(dtype=float),
                frame["X2"].to_numpy(dtype=float),
            ]
        )
    )
    r_a_x2, r_a_x1, r_x1_x2 = corr[0, 2], corr[0, 1], corr[1, 2]
    # First-order partial correlation from the 3x3 correlation matrix — the same
    # quantity the residualized route gives, without materializing residuals.
    rho_partial = (r_a_x2 - r_a_x1 * r_x1_x2) / math.sqrt(
        (1.0 - r_a_x1**2) * (1.0 - r_x1_x2**2)
    )
    return {(): float(r_a_x2), ("X1",): float(rho_partial)}


# --------------------------------------------------------------------------- #
# Fisher-z rejection probability and the pre-committed fork
# --------------------------------------------------------------------------- #


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (bisection; exact enough at 1e-15 tolerance)."""
    lo, hi = -40.0, 40.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fisher_z_rejection_probability(
    rho: float, n_disc: int, n_conditioning: int, alpha: float = ALPHA
) -> float:
    """[LD-2] Fisher-z rejection probability at ``N_disc``, verbatim formula.

        r = Φ( sqrt(N_disc − |S| − 3) · |atanh ρ| − z_{1−alpha/2} )

    Under the Fisher-z transform ``z = atanh(rho_hat)`` is approximately normal
    with mean ``atanh(rho)`` and variance ``1/(N − |S| − 3)``, so the two-sided
    test rejects when ``sqrt(N−|S|−3)·|z| > z_{1−alpha/2}``. The registered form
    keeps the dominant tail only; at ``rho = 0`` it therefore returns
    ``Φ(−z_{0.975}) = alpha/2`` rather than the exact two-sided size ``alpha``.
    Both readings are reported in the artifact, and on this grid both land in the
    same fork zone by two orders of magnitude, so the distinction moves no label.
    """
    if n_disc - n_conditioning - 3 <= 0:
        raise ValueError(
            f"N_disc - |S| - 3 must be positive; got N_disc={n_disc}, "
            f"|S|={n_conditioning}."
        )
    z_crit = _normal_ppf(1.0 - alpha / 2.0)
    scale = math.sqrt(n_disc - n_conditioning - 3)
    return _normal_cdf(scale * abs(math.atanh(rho)) - z_crit)


def d2_fork(r_op: float) -> str:
    """[LD-2] The three zones, applied EXACTLY as committed."""
    if r_op <= GENUINE_MAX:
        return "GENUINE"
    if r_op >= ALARM_MIN:
        return "ALARM"
    return "INDETERMINATE"


@dataclass(frozen=True)
class D2Result:
    """The complete truth-side adjudication for triangle/linear/effect_modifying."""

    coefficients: TriangleEMCoefficients
    n_disc: int
    alpha: float
    subsets: tuple[SubsetCorrelation, ...]
    operative: SubsetCorrelation
    zone: str
    synthetic: dict[tuple[str, ...], float] | None = None
    sepsets_available: bool = False
    stored_sepset: tuple[str, ...] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def r_op(self) -> float:
        return self.operative.rejection_prob

    @property
    def halt_lifts(self) -> bool:
        """Only the GENUINE zone lifts the halt. ALARM and INDETERMINATE stand."""
        return self.zone == "GENUINE"

    @property
    def seed_drop_probability(self) -> float:
        """Per-seed probability that the test ACCEPTS independence (edge dropped)."""
        return 1.0 - self.r_op

    def all_seeds_drop_probability(self, n_seeds: int = 20) -> float:
        """Probability all ``n_seeds`` independent runs drop the edge."""
        return self.seed_drop_probability**n_seeds


def run_d2_check(
    n_disc: int,
    alpha: float = ALPHA,
    *,
    with_synthetic: bool = False,
    synthetic_n: int = 10_000_000,
    sepsets_available: bool = False,
    stored_sepset: tuple[str, ...] | None = None,
) -> D2Result:
    """Run the LD-2 truth-side check end to end.

    ``n_disc`` MUST come from the run manifests (LD-2: "never assumed").
    """
    coefficients = triangle_em_builder_coefficients()
    rhos = triangle_em_population_correlations(coefficients)
    subsets = tuple(
        SubsetCorrelation(
            conditioning=conditioning,
            rho=rho,
            n_disc=n_disc,
            alpha=alpha,
            rejection_prob=fisher_z_rejection_probability(
                rho, n_disc, len(conditioning), alpha
            ),
        )
        # Deterministic order: marginal first, then the size-1 subsets.
        for conditioning, rho in sorted(rhos.items(), key=lambda kv: (len(kv[0]), kv[0]))
    )
    # [LD-2] "the operative subset is the one with the smallest |ρ|".
    operative = min(subsets, key=lambda s: abs(s.rho))
    synthetic = triangle_em_synthetic_correlations(synthetic_n) if with_synthetic else None
    return D2Result(
        coefficients=coefficients,
        n_disc=n_disc,
        alpha=alpha,
        subsets=subsets,
        operative=operative,
        zone=d2_fork(operative.rejection_prob),
        synthetic=synthetic,
        sepsets_available=sepsets_available,
        stored_sepset=stored_sepset,
    )
