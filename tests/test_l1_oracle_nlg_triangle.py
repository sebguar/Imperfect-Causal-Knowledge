"""L1-oracle on the NONLINEAR-GAUSSIAN triangle: recovery, wiring, PS-6.

The NLG-family analogue of tests/test_l1_oracle_estimation.py (linear triangle)
and tests/test_l1_oracle_collider.py (linear collider): same topology as the
former, a new FUNCTIONAL FAMILY. A failure here is a WIRING or ESTIMATOR bug, not
a finding. Covers, per the NLG triangle spec:

  1. the supplied template's columns match the BUILDER's actual terms — additive
     {A, tanh(X1)}, effect-modifying {tanh(X1), A*tanh(X1)} with NO A main effect;
  2. coefficient recovery both regimes (each templated coefficient within ~4 SE of
     the builder default, under the pooled mapping for the effect-modifying gain);
  3. the effect-modifying group gains g_pos / g_neg recovered with correct sign,
     correct ordering, and non-trivial magnitude;
  4. estimated-SCM counterfactual sanity — exact collapse to the true SCM at TRUE
     coefficients (the assembly/abduction tether) and finite-sample agreement at
     the ESTIMATED ones;
  5. live small-scale ladder: believed_cf populated and non-null on NLG L1-oracle
     rows, and the believed path genuinely diverges from the realized one;
  6. PS-6 collapse on the NLG triangle — Gap_cost identically 0 at L1-oracle, with
     believed_cost == realized_cost per row (the intervention-cost identity it rests on);
  7. no G1 anchor off the linear family (run_regime refuses run_anchor=True);
  8. the shipped NLG aggregate CSVs carry family=nlg rows for all three conditions.

True coefficients are read PROGRAMMATICALLY from `make_nonlinear_gaussian_triangle`'s
defaults, so a register-logged PS-2 retune does not require test edits. The 4-SE
recovery criterion is IMPORTED from the linear-triangle L1-oracle estimation test
rather than restated, so the two FAMILIES are held to one criterion by construction
(the same discipline test_l1_oracle_collider.py applies across topologies).
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from icknowledge.aggregation import CellKeys, aggregate_run
from icknowledge.classifier import build_classifier
from icknowledge.estimation import (
    FittedEquations,
    NLGOLSEstimator,
    build_estimated_scm,
    design_matrix,
    equation_from_coefficients,
)
from icknowledge.recourse import run_triangle
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import make_nonlinear_gaussian_triangle, nlg_triangle_form_spec
from icknowledge.utils.config import load_config

# ONE criterion for both FAMILIES: the linear-triangle test owns the SE machinery
# and the 4-SE bound; importing it (rather than restating) is what keeps the NLG
# family from silently drifting to a looser gate. Same precedent as the collider
# test importing it across topologies.
from tests.test_l1_oracle_estimation import _assert_recovery

_CONFIG_PATH = "configs/recourse_triangle_nlg.yaml"
_RESULTS_DIR = Path("results/recourse_triangle_nlg")
_REGIMES = ("additive", "effect_modifying")
_CONDITIONS = ("L0", "L1-oracle", "L2")
_FEATURES = ("X1", "X2")
_SEED = 20260710  # the config's master seed (the seed-generation meta-entropy); the CSVs' cell key


def _builder_defaults() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_nonlinear_gaussian_triangle).parameters
    return {k: float(params[k].default) for k in ("a", "b", "g", "g_pos", "g_neg", "sigma")}


def _true_coefficients(regime: str, d: dict[str, float]) -> dict[str, dict[str, float]]:
    """True values of every templated term, in the estimator's parametrization.

    Mapping VERIFIED term by term against `make_nonlinear_gaussian_triangle`:

    X1 := a·A + U1 in both regimes — IDENTICAL to the linear family (the tanh
    nonlinearity is confined to X2's equation) → intercept 0, A-coef a.

    X2, additive: X2 := b·A + g·tanh(X1) + U2 → intercept 0, A-coef b,
    tanh(X1)-coef g. Templated columns {A, tanh(X1)} — no interaction.

    X2, effect-modifying: the builder computes gain = np.where(A > 0, g_pos, g_neg)
    and X2 := gain·tanh(X1) + U2, with NO additive A term and NO intercept. Under
    the centered A ∈ {−1, +1} encoding (asserted on the data below),
    gain(A) = (g_pos + g_neg)/2 + A·(g_pos − g_neg)/2, so the pooled
    LINEAR-IN-PARAMETERS form X2 = β₀ + β₁·tanh(X1) + γ·(A·tanh(X1)) + ε₂ has
        β₁ = (g_pos + g_neg)/2,  γ = (g_pos − g_neg)/2,  β₀ = 0.
    This is exactly the NLG-family claim the whole NLG L1-oracle rests on: after the
    FIXED tanh(·) transform the mechanism is linear in its parameters, so the L1-oracle
    form-template-known semantics's
    unregularized OLS carries over unchanged.
    """
    truths = {"X1": {"intercept": 0.0, "A": d["a"]}}
    if regime == "additive":
        truths["X2"] = {"intercept": 0.0, "A": d["b"], "tanh(X1)": d["g"]}
    else:
        truths["X2"] = {
            "intercept": 0.0,
            "tanh(X1)": (d["g_pos"] + d["g_neg"]) / 2.0,
            "A*tanh(X1)": (d["g_pos"] - d["g_neg"]) / 2.0,
        }
    return truths


def _make_cell(regime: str) -> SimpleNamespace:
    """One fitted NLG estimation cell, on the sample the L1-oracle form-template-known semantics
    mandates.

    # [the L1-oracle form-template-known semantics] The estimation sample IS the classifier's
    # training sample
    # (ClassifierResult.dataset) — no redraw, no held-out split — so this fixture
    # exercises the same data the pipeline's L1-oracle rung is fitted on. The
    # no-penalty (lstsq) guard lives in the linear-triangle test; per the NLG family specification
    # the NLG
    # estimator IS that estimator, so the guard covers this family too.
    """
    cfg = load_config(_CONFIG_PATH)
    scm = make_nonlinear_gaussian_triangle(regime)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        clf = build_classifier(scm, cfg)
    data = clf.dataset
    # Premise of the pooled mapping in _true_coefficients: centered ±1 encoding.
    assert set(np.unique(data["A"]).tolist()) == {-1.0, 1.0}
    form = nlg_triangle_form_spec(regime)
    fitted = NLGOLSEstimator().fit(
        data, form, extra_metadata={"regime": regime, "family": "nlg"}
    )
    return SimpleNamespace(
        regime=regime,
        scm=scm,
        data=data,
        n=len(data),
        form=form,
        fitted=fitted,
        defaults=_builder_defaults(),
        truths=_true_coefficients(regime, _builder_defaults()),
    )


@pytest.fixture(scope="module")
def cells() -> dict[str, SimpleNamespace]:
    return {regime: _make_cell(regime) for regime in _REGIMES}


# ---------------------------------------------------------------------------
# 1 — the supplied template's columns ARE the builder's terms
# ---------------------------------------------------------------------------


def test_template_columns_match_the_builder(cells):
    """Design columns per regime, read against `make_nonlinear_gaussian_triangle`.

    # [the L1-oracle form-template-known semantics] The template is SUPPLIED, and what it
    # supplies must be
    # the builder's ACTUAL term set — a column the truth does not have is a
    # (harmless but undeclared) over-parametrization, and a column it does have
    # that is missing is a misspecification, which the L1-oracle form-template-known semantics
    # excludes from this rung by
    # construction. The effect-modifying template carries NO A main effect because
    # the builder carries none: the group difference lives entirely in the gain on
    # tanh(X1), and an absent term is hard-zero by COLUMN OMISSION, never
    # fit-then-discarded.
    """
    x_add, terms_add = design_matrix(cells["additive"].data, cells["additive"].form["X2"])
    assert terms_add == ["A", "tanh(X1)"] and x_add.shape[1] == 2, (
        f"additive NLG X2 columns must be {{A, tanh(X1)}}, got {terms_add}"
    )
    x_em, terms_em = design_matrix(
        cells["effect_modifying"].data, cells["effect_modifying"].form["X2"]
    )
    assert terms_em == ["tanh(X1)", "A*tanh(X1)"] and x_em.shape[1] == 2, (
        f"effect-modifying NLG X2 columns must be {{tanh(X1), A*tanh(X1)}}, got {terms_em}"
    )
    assert "A" not in cells["effect_modifying"].fitted.coefficients["X2"], (
        "the effect-modifying NLG builder has no A main effect, so no A key may "
        "exist in the fitted coefficients (hard-zero by omission)."
    )
    # A is still DECLARED a parent of X2 in both regimes — L1-oracle carries the
    # TRUE graph, and build_estimated_scm checks the declared parent set against it.
    for regime in _REGIMES:
        assert set(cells[regime].form["X2"].parents) == {"A", "X1"}

    # The columns really are the fixed transform of the parent, not a relabelling.
    np.testing.assert_allclose(
        x_em[:, 0], np.tanh(cells["effect_modifying"].data["X1"].to_numpy(dtype=float))
    )


# ---------------------------------------------------------------------------
# 2 — coefficient recovery, both regimes (the imported 4-SE gate)
# ---------------------------------------------------------------------------


def test_nlg_coefficient_recovery_additive(cells):
    """Every fitted NLG coefficient within ~4 SE of its builder default (additive).

    The new surface relative to the linear triangle is the tanh(X1) column: a
    transform applied at fit time but not at counterfactual time (or vice versa)
    would show up as a badly-off g and a badly-off b here.
    """
    _assert_recovery(cells["additive"])


def test_nlg_coefficient_recovery_effect_modifying(cells):
    """Same for the effect-modifying regime, INCLUDING γ via the pooled mapping.

    NOTE on precision, not on correctness: with a = 2.0 the tanh saturates, so
    tanh(X1) ≈ ±0.93 by group and the design columns are strongly collinear
    (corr(A, tanh(X1)) ≈ 0.96; A·tanh(X1) is nearly constant, hence nearly
    collinear with the intercept). OLS stays unbiased, but its SEs are inflated —
    which is why this is an SE-scaled bound rather than an absolute one, and why
    the NLG L1-oracle rung carries visibly more estimation error than the linear
    one at the same n. That is a property of the SCM's construction parameters,
    not of the estimator.
    """
    _assert_recovery(cells["effect_modifying"])


# ---------------------------------------------------------------------------
# 3 — the effect-modifying group gains g_pos / g_neg
# ---------------------------------------------------------------------------


def _ols_covariance(cell: SimpleNamespace, node: str) -> tuple[np.ndarray, list[str]]:
    """Full classical OLS covariance from the SAME design matrix the estimator used.

    The imported `_ols_standard_errors` returns only the diagonal; the implied
    group gain ĝ(a) = β̂₁ + a·γ̂ needs the off-diagonal too (delta method).
    """
    x, terms = design_matrix(cell.data, cell.form[node])
    xi = np.column_stack([np.ones(len(x)), x])
    y = cell.data[node].to_numpy(dtype=float)
    coefs = cell.fitted.coefficients[node]
    names = ["intercept", *terms]
    beta_hat = np.array([coefs[n] for n in names])
    resid = y - xi @ beta_hat
    sigma2 = float(resid @ resid) / (len(y) - xi.shape[1])
    return sigma2 * np.linalg.inv(xi.T @ xi), names


def test_effect_modifying_group_gains_recovered(cells):
    """ĝ_pos and ĝ_neg recovered: right ordering, right sign, non-trivial magnitude.

    The pooled coefficients map back to the builder's group gains as
    ĝ(a) = β̂₁ + a·γ̂ for a ∈ {−1, +1} — the quantity the effect-modifying regime
    is DEFINED by. The 4-SE recovery bound is applied to each gain through the
    delta-method SE (β̂₁ and γ̂ are strongly negatively correlated here, so summing
    their marginal SEs would be the wrong scale).

    The distinguishable-from-zero bar on γ̂ is 2 SE, NOT the 4 SE the collider test
    uses: the near-collinearity documented above inflates SE(γ̂) enough that the
    true γ = 0.3 sits only ~4.5 SE from zero at this n, so a 4-SE bar would be a
    coin-flip on sampling noise rather than a statement about the mechanism. The
    substantive "the channel is real" claim is instead carried by the RELATIVE
    magnitude assertion, which is scale-free and independent of the SE inflation.
    """
    cell = cells["effect_modifying"]
    d = cell.defaults
    gamma_true = (d["g_pos"] - d["g_neg"]) / 2.0
    cov, names = _ols_covariance(cell, "X2")
    i1, i2 = names.index("tanh(X1)"), names.index("A*tanh(X1)")
    coefs = cell.fitted.coefficients["X2"]
    gamma_hat = coefs["A*tanh(X1)"]
    se_gamma = float(np.sqrt(cov[i2, i2]))

    assert np.sign(gamma_hat) == np.sign(gamma_true), (
        f"γ̂={gamma_hat:.6f} has the wrong sign against the builder's implied "
        f"γ=(g_pos−g_neg)/2={gamma_true:.6f} — the A·tanh(X1) column is mis-signed."
    )
    assert abs(gamma_hat) / se_gamma > 2.0, (
        f"γ̂={gamma_hat:.6f} is only {abs(gamma_hat) / se_gamma:.2f} SE from zero at "
        f"n={cell.n} — the NLG effect-modifying channel is not recoverable, so the "
        "L1-oracle treatment arm would carry no signal."
    )
    assert abs(gamma_hat) > 0.5 * abs(gamma_true), (
        f"γ̂={gamma_hat:.6f} is under half the builder's γ={gamma_true:.6f} — the "
        "fitted effect modification is a fraction of the truth, not merely noisy."
    )

    gains = {}
    for a_val, truth in ((1.0, d["g_pos"]), (-1.0, d["g_neg"])):
        gain = coefs["tanh(X1)"] + a_val * gamma_hat
        var = cov[i1, i1] + a_val**2 * cov[i2, i2] + 2.0 * a_val * cov[i1, i2]
        se = float(np.sqrt(var))
        assert abs(gain - truth) / se <= 4.0, (
            f"implied group gain ĝ(A={a_val:+g})={gain:.6f} is "
            f"{abs(gain - truth) / se:.2f} SE from the builder's {truth:.6f} at "
            f"n={cell.n} — beyond the 4-SE recovery bound."
        )
        gains[a_val] = gain

    assert gains[1.0] > gains[-1.0], (
        f"ĝ_pos={gains[1.0]:.6f} must exceed ĝ_neg={gains[-1.0]:.6f} — the builder's "
        f"defaults are g_pos={d['g_pos']} > g_neg={d['g_neg']}."
    )
    assert (gains[1.0] - gains[-1.0]) > 0.5 * (d["g_pos"] - d["g_neg"]), (
        "the recovered gain SPREAD is under half the builder's — effect modification "
        "is present but too attenuated to drive the H4 contrast."
    )


# ---------------------------------------------------------------------------
# 4 — estimated-SCM counterfactual sanity (abduction exactness)
# ---------------------------------------------------------------------------


def _counterfactual_actions(batch: pd.DataFrame) -> list[dict]:
    x1 = batch["X1"].to_numpy(dtype=float)
    x2 = batch["X2"].to_numpy(dtype=float)
    return [
        {},  # identity counterfactual
        {"X1": x1 + 1.0},  # per-row do(X1 := x1 + 1): X2 must propagate through tanh
        {"X2": 0.0},  # scalar do(X2 := 0): severs the tanh channel
        {"X1": x1 - 0.5, "X2": x2 + 2.0},  # joint action severs X1→X2
    ]


@pytest.mark.parametrize("regime", _REGIMES)
def test_zero_estimation_error_collapses_to_l2(cells, regime):
    """Estimated NLG SCM built from the TRUE coefficients == true SCM, to machine precision.

    The abduction-exactness check: it pins the estimated-SCM assembly path
    (equation_from_coefficients + build_estimated_scm) to the known-correct L2 path
    at the point where the two must coincide by construction — INCLUDING the
    tanh transform, which must be applied identically at fit time and at
    counterfactual time. Exactness here is what makes abduction exact on the NLG
    rung (û = x − f̂(pa); the nonlinearity is in the parents, never in the noise),
    so any later L1-oracle discrepancy is attributable to ESTIMATION ERROR rather
    than to a transform applied on one side only.
    """
    cell = cells[regime]
    perfect = FittedEquations(
        coefficients=cell.truths,
        equations={
            node: equation_from_coefficients(cell.form[node], cell.truths[node])
            for node in cell.truths
        },
        form=cell.form,
        metadata={"estimator": "true-coefficients (test tether)", "regime": regime},
    )
    est_scm = build_estimated_scm(cell.scm, perfect)

    batch = cell.data.iloc[:200]
    for action in _counterfactual_actions(batch):
        cf_true = cell.scm.counterfactual(batch, action)
        cf_est = est_scm.counterfactual(batch, action)
        # 1e-12 absolute on O(1–10) values == machine precision up to reassociation
        # (ḡ + γ·A vs g_of_A differs at ~1e-16).
        max_diff = float(np.max(np.abs(cf_true.to_numpy() - cf_est.to_numpy())))
        assert max_diff < 1e-12, (
            f"{regime}: estimated NLG SCM with TRUE coefficients deviates from the true "
            f"SCM (max |diff| = {max_diff:.3e} for action {sorted(action)}) — an "
            "assembly/transform bug, NOT estimation error."
        )


@pytest.mark.parametrize("regime", _REGIMES)
def test_estimated_counterfactual_tracks_truth_within_sampling_noise(cells, regime):
    """At the ESTIMATED coefficients the NLG counterfactual tracks the true one.

    NOT an equality test: the gap IS the finite-sample estimation error the
    L2→L1-oracle contrast is defined to measure (H1). Bounded generously —
    the tanh saturation makes the design columns near-collinear (see test 2), so
    the coefficient SEs, and hence this gap, are larger than the linear triangle's
    at the same n. Zero would mean an oracle leak; O(1) would mean the estimated
    equation is not the fitted one.
    """
    cell = cells[regime]
    est_scm = build_estimated_scm(cell.scm, cell.fitted)
    batch = cell.data.iloc[:200]
    for action in _counterfactual_actions(batch):
        cf_true = cell.scm.counterfactual(batch, action).to_numpy()
        cf_est = est_scm.counterfactual(batch, action).to_numpy()
        max_diff = float(np.max(np.abs(cf_true - cf_est)))
        assert max_diff < 0.5, (
            f"{regime}: estimated NLG counterfactual deviates by {max_diff:.4f} for "
            f"action {sorted(action)} at n={cell.n} — beyond finite-sample error."
        )
    # A ≠ B check on the propagating action: with estimated coefficients the two
    # SCMs must NOT coincide, or the estimated path is secretly the true one.
    action = {"X1": batch["X1"].to_numpy(dtype=float) + 1.0}
    assert not np.allclose(
        cell.scm.counterfactual(batch, action).to_numpy(),
        est_scm.counterfactual(batch, action).to_numpy(),
    ), "estimated and true NLG counterfactuals coincide exactly — oracle leak."


# ---------------------------------------------------------------------------
# 5 + 6 — live small-scale ladder: believed_cf channel + PS-6 collapse
# ---------------------------------------------------------------------------
#
# Small resolution / N: the faithful-resolution numbers are the driver's artifact,
# not this test's job. These assertions are STRUCTURAL (a column is populated; an
# identity holds), so they are resolution-independent — and unlike test 8 they do
# not depend on results/ existing, which makes them the durable half of the gate.


def _small_cfg():
    cfg = load_config(_CONFIG_PATH)
    cfg.classifier.N = 600
    cfg.classifier.min_neg_per_group = 30
    cfg.recourse.grid.resolution = 21  # NOT the grid-resolution production pin (161) — test scale
    return cfg


@pytest.fixture(scope="module")
def ladder_runs():
    cfg = _small_cfg()
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime in _REGIMES:
            runs[regime] = run_regime(
                cfg,
                regime,
                conditions=_CONDITIONS,
                scm_builder=make_nonlinear_gaussian_triangle,
                form_spec_fn=nlg_triangle_form_spec,
                family="nlg",
                run_anchor=False,  # NO G1 anchor off the linear family (see test 7)
            )
    return cfg, runs


@pytest.mark.parametrize("regime", _REGIMES)
def test_believed_cf_populated_on_nlg_l1_oracle_rows(ladder_runs, regime):
    """believed_cf exists and is non-null on every found NLG L1-oracle row.

    The believed_cf mechanism is family-agnostic (scoring.py routes the
    believed side through whatever model object generated the solution), so this
    VERIFIES rather than reimplements — the new surface is only that the estimated
    SCM behind the L1-oracle model carries a tanh equation.
    """
    table = ladder_runs[1][regime].table
    l1 = table[(table["condition"] == "L1-oracle") & table["found"]]
    assert len(l1) > 0, "no found L1-oracle rows on the NLG triangle"
    for feature in _FEATURES:
        col = f"believed_cf_{feature}"
        assert col in table.columns, f"{col} missing from the NLG scoring table"
        assert l1[col].notna().all(), (
            f"{regime}: null {col} on a found L1-oracle row — the believed path is "
            "not being routed through the estimated NLG SCM."
        )


def test_believed_cf_diverges_at_nlg_l1_oracle(ladder_runs):
    """The believed and realized CF paths are computed independently on the NLG cell.

    NO magnitude assertion (mirrors the linear wiring tests): the point is only
    that the channel EXISTS — f̂_L1 ≠ f_true maps some individual's chosen δ to a
    different downstream X2 than the true SCM does.
    """
    table = ladder_runs[1]["effect_modifying"].table
    sub = table[(table["condition"] == "L1-oracle") & table["found"]]
    believed = sub[[f"believed_cf_{f}" for f in _FEATURES]].to_numpy()
    realized = sub[[f"realized_cf_{f}" for f in _FEATURES]].to_numpy()
    assert np.any(believed != realized), (
        "believed_cf identical to realized_cf on every NLG L1-oracle row — the "
        "believed path is not going through the estimated SCM."
    )


def test_nlg_estimator_is_wired_into_the_ladder(ladder_runs):
    """The L1-oracle rung was fitted by the NLG estimator on the NLG template.

    Guards the form-template-known provenance claim end to end: family selection is by the
    CONFIG/RUN family field (the SCM carries no family attribute), so a silent
    fallback to the linear estimator/template would leave the ladder running and
    only show up here.
    """
    for regime in _REGIMES:
        fitted = ladder_runs[1][regime].fitted
        assert fitted.metadata["family"] == "nlg"
        assert "NLGOLSEstimator" in fitted.metadata["estimator"]
        assert "tanh(X1)" in fitted.coefficients["X2"]


@pytest.mark.parametrize("regime", _REGIMES)
def test_gap_cost_collapse_at_nlg_l1_oracle(ladder_runs, regime, tmp_path):
    """Gap_cost == 0 EXACTLY at L1-oracle on the NLG triangle (PS-6 carries over).

    # [PS-6; the ℓ₂ intervention-cost convention] Expected, NOT a bug: cost is a property of the
    # ACTION, so
    # per-row believed cost equals realized cost bitwise for the chosen δ at every
    # SCM-carrying condition — a statement about the cost functional that the
    # functional family cannot touch. The H2 signal on the NLG cell therefore lives
    # in ValidityGap_g(c), the same as on the linear triangle and the collider.
    # Asserted on a live run through the real aggregation layer (tmp dir), so it
    # does not depend on results/ being present.
    """
    _cfg, runs = ladder_runs
    table = runs[regime].table
    for condition in _CONDITIONS:
        sub = table[table["condition"] == condition]
        sub.to_csv(tmp_path / f"scoring_table_{regime}_{condition}.csv", index=False)

    cell = CellKeys(topology="triangle", family="nlg", regime=regime, seed=_SEED)
    _by_group_path, by_cell_path = aggregate_run(tmp_path, tmp_path / "agg", cell)
    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")

    l1_rows = by_cell[by_cell["condition"] == "L1-oracle"]
    assert len(l1_rows) == 1, "expected exactly one NLG L1-oracle by-cell row"
    assert float(l1_rows.iloc[0]["Gap_cost"]) == 0.0, (
        f"{regime}: PS-6 collapse violated at the NLG L1-oracle row — Gap_cost must "
        "be identically zero at an SCM-carrying condition."
    )
    assert (by_cell["family"] == "nlg").all()
    # The per- the ℓ₂ intervention-cost convention identity the cell-level collapse rests on.
    found = table[(table["condition"] == "L1-oracle") & table["found"]]
    np.testing.assert_array_equal(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy()
    )


# ---------------------------------------------------------------------------
# 7 — no G1 anchor off the linear family
# ---------------------------------------------------------------------------


def test_no_g1_anchor_on_nlg():
    """run_regime refuses the G1 anchor for a non-linear family, loudly.

    The Ehyaei closed form the anchor pulls back assumes a LINEAR SCM and a linear
    classifier (the ℓ₂ intervention-cost convention), so on an NLG cell it is
    inapplicable, not merely
    loose — reporting its mismatch as a smoke-check failure would be a false alarm.
    The driver enforces the same rule from the config's family key.
    """
    cfg = _small_cfg()
    with pytest.raises(ValueError, match="family='linear'"):
        run_regime(
            cfg,
            "additive",
            conditions=_CONDITIONS,
            scm_builder=make_nonlinear_gaussian_triangle,
            form_spec_fn=nlg_triangle_form_spec,
            family="nlg",
            run_anchor=True,
        )
    assert set(run_triangle._FAMILIES) == {"linear", "nlg"}


# ---------------------------------------------------------------------------
# 8 — the shipped NLG aggregates carry family=nlg rows for the full ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", _REGIMES)
def test_shipped_nlg_aggregates_carry_family_rows(regime):
    """The regenerated production CSVs carry family=nlg rows for L0/L1-oracle/L2.

    results/ is gitignored and machine-local, so this SKIPS cleanly on a fresh
    clone (the same discipline as test_aggregation_layer.py); tests 5-6 above are
    the durable, run-from-source half of the same gate.
    """
    agg_dir = _RESULTS_DIR / "aggregates"
    by_cell_path = agg_dir / f"aggregate_by_cell_{regime}.csv"
    by_group_path = agg_dir / f"aggregate_by_group_{regime}.csv"
    per_individual = _RESULTS_DIR / f"scoring_table_{regime}_L1-oracle.csv"
    missing = [
        str(p) for p in (by_cell_path, by_group_path, per_individual) if not p.exists()
    ]
    if missing:
        pytest.skip(f"NLG triangle result CSVs not present locally: {missing}")

    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")
    assert set(by_cell["condition"]) == set(_CONDITIONS), (
        f"{regime}: aggregate_by_cell is missing conditions — regenerate with "
        "`python -m icknowledge.recourse.run_triangle configs/recourse_triangle_nlg.yaml`."
    )
    assert (by_cell["family"] == "nlg").all()
    assert (by_cell["topology"] == "triangle").all()
    assert float(by_cell[by_cell["condition"] == "L1-oracle"].iloc[0]["Gap_cost"]) == 0.0

    by_group = pd.read_csv(by_group_path, float_precision="round_trip")
    assert (by_group["family"] == "nlg").all()
    assert set(by_group["condition"]) == set(_CONDITIONS)

    table = pd.read_csv(per_individual, float_precision="round_trip")
    found = table[table["found"]]
    # believed_cf present and non-null on the shipped L1-oracle rows.
    for feature in _FEATURES:
        assert found[f"believed_cf_{feature}"].notna().all()
    # The ℓ₂ intervention-cost identity, per row, on the shipped
    # artifact.
    np.testing.assert_array_equal(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy()
    )
