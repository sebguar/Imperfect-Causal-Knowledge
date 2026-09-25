"""L1-oracle on the NONLINEAR-GAUSSIAN collider: recovery, wiring, PS-6.

The fourth corner of the topology × family square: tests/test_l1_oracle_estimation.py
(linear triangle), tests/test_l1_oracle_collider.py (linear collider) and
tests/test_l1_oracle_nlg_triangle.py (NLG triangle) are the other three. A failure
here is a WIRING or ESTIMATOR bug, not a finding. Covers, per the NLG collider spec:

  1. the supplied template's columns match the BUILDER's actual terms — additive
     {tanh(X1), tanh(X2), A}, effect-modifying {tanh(X1), A*tanh(X1), tanh(X2), A};
     and the additive regime carries NO A*tanh(X1) column at all;
  2. coefficient recovery both regimes (each templated coefficient within ~4 SE of
     the SCM specification builder default);
  3. γ̂ recovered with the correct sign and non-trivial magnitude;
  4. driver wiring — run_collider runs the full L0 + L1-oracle + L2 ladder on NLG;
  5. live small-scale ladder: believed_cf populated and non-null on NLG L1-oracle
     rows, and the believed path genuinely diverges from the realized one;
  6. PS-6 collapse on the NLG collider — Gap_cost identically 0 at L1-oracle, with
     believed_cost == realized_cost per row (the intervention-cost identity it rests on);
  7. no G1 anchor off the linear family (run_regime refuses run_anchor=True);
  8. the shipped NLG collider aggregate CSVs carry family=nlg rows for all three
     conditions.

True coefficients are read PROGRAMMATICALLY from
`make_nonlinear_gaussian_collider`'s defaults, so a register-logged PS-2 retune does
not require test edits. The 4-SE recovery criterion is IMPORTED from the
linear-triangle L1-oracle estimation test rather than restated, so all four
topology × family cells are held to ONE criterion by construction.
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
from icknowledge.estimation import NLGOLSEstimator, design_matrix
from icknowledge.recourse import run_collider
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import make_nonlinear_gaussian_collider, nlg_collider_form_spec
from icknowledge.utils.config import load_config

# ONE criterion across BOTH axes of the cell square: the linear-triangle test owns
# the SE machinery and the 4-SE bound; importing it (rather than restating) is what
# keeps a new topology × family corner from silently drifting to a looser gate.
# Same discipline as test_l1_oracle_collider.py (across topologies) and
# test_l1_oracle_nlg_triangle.py (across families).
from tests.test_l1_oracle_estimation import _assert_recovery, _ols_standard_errors

_CONFIG_PATH = "configs/recourse_collider_nlg.yaml"
_RESULTS_DIR = Path("results/recourse_collider_nlg")
_REGIMES = ("additive", "effect_modifying")
_CONDITIONS = ("L0", "L1-oracle", "L2")
_FEATURES = ("X1", "X2", "X3")
_SEED = 20260710  # the config's master seed (the seed-generation meta-entropy); the CSVs' cell key


def _builder_defaults() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_nonlinear_gaussian_collider).parameters
    keys = (
        "alpha1", "alpha2", "sigma1", "sigma2",
        "beta", "eta", "gamma", "beta_A", "sigma3", "beta0",
    )
    return {k: float(params[k].default) for k in keys}


def _true_coefficients(regime: str, d: dict[str, float]) -> dict[str, dict[str, float]]:
    """True values of every templated term, in the estimator's parametrization.

    # [the SCM specification; the NLG family specification] Like the LINEAR collider — and unlike
    # either triangle, whose effect-modifying builders use an np.where group-slope
    # switch that must be mapped onto a pooled basis — the NLG collider builder is
    # ALREADY WRITTEN in the pooled basis the form spec templates:
    #     X1 := α₁·A + ε₁
    #     X2 := α₂·A + ε₂
    #     X3 := β₀ + β·tanh(X1) + γ·(A·tanh(X1)) + η·tanh(X2) + β_A·A + ε₃
    # so each fitted term maps 1:1 to a builder default with NO reparametrization.
    # This is the NLG family specification LIP-after-tanh claim in its most direct form: the design
    # columns ARE the builder's terms with tanh already applied.
    #
    # γ is absent from the additive dict because it is hard-zero by COLUMN OMISSION
    # there, not fit-then-discarded — see test 1. X1 and X2 carry NO
    # intercept term in the builder, but the L1-oracle form-template-known semantics fixes
    # fit_intercept=True, so their
    # intercepts are templated with true value 0. X3's intercept is β₀, a REAL
    # builder parameter that happens to default to 0.
    """
    truths = {
        "X1": {"intercept": 0.0, "A": d["alpha1"]},
        "X2": {"intercept": 0.0, "A": d["alpha2"]},
        "X3": {
            "intercept": d["beta0"],
            "tanh(X1)": d["beta"],
            "tanh(X2)": d["eta"],
            "A": d["beta_A"],
        },
    }
    if regime == "effect_modifying":
        truths["X3"]["A*tanh(X1)"] = d["gamma"]
    return truths


def _make_cell(regime: str) -> SimpleNamespace:
    """One fitted NLG collider estimation cell, on the sample the L1-oracle form-template-known
    semantics mandates.

    # [the L1-oracle form-template-known semantics] The estimation sample IS the classifier's
    # training sample
    # (ClassifierResult.dataset) — no redraw, no held-out split — so this fixture
    # exercises the same data the pipeline's L1-oracle rung is fitted on. The
    # no-penalty (lstsq) guard lives in the linear-triangle test; per the NLG family specification
    # the NLG
    # estimator IS that estimator, so the guard covers this cell too.
    """
    cfg = load_config(_CONFIG_PATH)
    scm = make_nonlinear_gaussian_collider(regime)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        clf = build_classifier(scm, cfg)
    data = clf.dataset
    # Premise of the centered ±1 encoding the A·tanh(X1) column's reading rests on.
    assert set(np.unique(data["A"]).tolist()) == {-1.0, 1.0}
    form = nlg_collider_form_spec(regime)
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
    """X3's design columns per regime, read against the NLG collider builder.

    # [the L1-oracle form-template-known semantics] The template is SUPPLIED, and what it
    # supplies must be
    # the builder's ACTUAL term set — a column the truth does not have is an
    # undeclared over-parametrization, and a missing one is a misspecification,
    # which the L1-oracle form-template-known semantics excludes from this rung by construction.
    #
    # The A MAIN EFFECT is present in BOTH regimes here (β_A·A is carried in both
    # by the SCM specification's graph-matching rule), which is where this template departs from
    # `nlg_triangle_form_spec` — whose effect-modifying builder carries no additive
    # A term at all. Both templates match their own builder; the difference is in
    # the truths, not in the discipline.
    """
    x_add, terms_add = design_matrix(cells["additive"].data, cells["additive"].form["X3"])
    assert terms_add == ["tanh(X1)", "tanh(X2)", "A"] and x_add.shape[1] == 3, (
        f"additive NLG collider X3 columns must be {{tanh(X1), tanh(X2), A}}, "
        f"got {terms_add}"
    )
    x_em, terms_em = design_matrix(
        cells["effect_modifying"].data, cells["effect_modifying"].form["X3"]
    )
    assert terms_em == ["tanh(X1)", "A*tanh(X1)", "tanh(X2)", "A"] and x_em.shape[1] == 4, (
        f"effect-modifying NLG collider X3 columns must be {{tanh(X1), A*tanh(X1), "
        f"tanh(X2), A}}, got {terms_em}"
    )
    # A, X1, X2 are all DECLARED parents of X3 in both regimes — L1-oracle carries
    # the TRUE graph, and build_estimated_scm checks the declared parent set.
    for regime in _REGIMES:
        assert set(cells[regime].form["X3"].parents) == {"A", "X1", "X2"}
        # X1 and X2 stay IDENTICAL to the linear family: the tanh sits in X3 only.
        for node in ("X1", "X2"):
            _x, terms = design_matrix(cells[regime].data, cells[regime].form[node])
            assert terms == ["A"], f"{regime}/{node} must template a bare A column"

    # The columns really are the fixed transform of the parent, not a relabelling,
    # and tanh(X1) / tanh(X2) are not swapped (the two channels are symmetric by
    # construction under the SCM defaults, so a swap would otherwise be invisible).
    em = cells["effect_modifying"]
    np.testing.assert_allclose(
        x_em[:, 0], np.tanh(em.data["X1"].to_numpy(dtype=float))
    )
    np.testing.assert_allclose(
        x_em[:, 2], np.tanh(em.data["X2"].to_numpy(dtype=float))
    )


def test_additive_regime_has_no_interaction_column(cells):
    """γ is hard-zero by COLUMN OMISSION in the additive control (the SCM specification).

    Structural, mirroring the linear collider's additive test: not
    fit-then-discarded — no A*tanh(X1) column may exist in X₃'s design matrix and
    no A*tanh(X1) key may exist in the fitted coefficients.
    """
    cell = cells["additive"]
    _x, terms = design_matrix(cell.data, cell.form["X3"])
    assert "A*tanh(X1)" not in terms, (
        f"additive NLG collider X3 carries an interaction column: {terms}"
    )
    assert "A*tanh(X1)" not in cell.fitted.coefficients["X3"], (
        "γ must be hard-zero by OMISSION from the design matrix (the SCM specification), "
        "not "
        "fit-then-discarded — no γ key may exist in the additive regime."
    )


# ---------------------------------------------------------------------------
# 2 — coefficient recovery, both regimes (the imported 4-SE gate)
# ---------------------------------------------------------------------------


def test_nlg_collider_coefficient_recovery_additive(cells):
    """Every fitted coefficient within ~4 SE of its SCM builder default (additive).

    The new surface relative to BOTH predecessors: X₃'s equation carries two
    tanh-transformed endogenous parents plus a root main effect, estimated jointly.
    A transform applied at fit time but not at counterfactual time (or vice versa),
    or a tanh(X1)/tanh(X2) column swap, shows up here as badly-off β̂ and η̂.
    """
    _assert_recovery(cells["additive"])


def test_nlg_collider_coefficient_recovery_effect_modifying(cells):
    """Same for the effect-modifying regime, INCLUDING γ (the A·tanh(X1) column).

    η̂ and β̂_A must ALSO come back at their additive values: the SCM specification's channel (i)
    modifies X₃'s tanh(X1) gain ONLY, so a γ̂ that ate part of η̂ or β̂_A would be an
    interaction-column construction bug, and this joint bound catches it.
    """
    _assert_recovery(cells["effect_modifying"])


# ---------------------------------------------------------------------------
# 3 — γ̂: correct sign and non-trivial magnitude
# ---------------------------------------------------------------------------


def test_gamma_hat_sign_and_nontrivial_magnitude(cells):
    """γ̂ carries the builder's sign and is distinguishable from zero.

    The distinguishable-from-zero bar is 2 SE, NOT the 4 SE the LINEAR collider
    test uses — the same relaxation tests/test_l1_oracle_nlg_triangle.py had to
    make, and it is documented here rather than silently inherited.

    PRE-COMMITMENT AND OUTCOME. The pre-commitment was to start at the
    4-SE bar, on the reasoning that the NLG collider sits FURTHER from tanh
    saturation than the NLG triangle did (α = 0.6 vs a = 2.0), so the SE inflation
    that forced the triangle down to 2 SE should be milder here. The 4-SE bar
    FAILED: γ̂ lands ≈ 2.9 SE from zero at n = 1500. Saturation was the wrong
    diagnosis for this cell — the binding constraint is the opposite one. Keeping
    X₁ out of saturation keeps its marginal NARROW (α = 0.6, σ = 0.4), so
    tanh(X₁) ∈ ≈ [−0.8, 0.8] has little spread, the A·tanh(X₁) column carries
    little variation, and σ₃ = 1.0 is large against it. SE(γ̂) is therefore inflated
    by WEAK REGRESSOR VARIANCE rather than by collinearity. Both routes end at the
    same place: γ = 0.3 sits only a few SE from zero at this n, so a 4-SE bar would
    be a coin-flip on sampling noise rather than a statement about the mechanism.

    The substantive "the channel is real" claim is carried instead by the
    RELATIVE-MAGNITUDE assertion below, which is scale-free and independent of the
    SE inflation. NOTE that the two-sided 4-SE RECOVERY bound is NOT relaxed —
    tests 2 above still apply it to A*tanh(X1) via `_assert_recovery`.
    """
    cell = cells["effect_modifying"]
    gamma_true = cell.defaults["gamma"]
    gamma_hat = cell.fitted.coefficients["X3"]["A*tanh(X1)"]
    se = _ols_standard_errors(cell, "X3")["A*tanh(X1)"]

    assert np.sign(gamma_hat) == np.sign(gamma_true), (
        f"γ̂={gamma_hat:.6f} has the wrong sign against the SCM default "
        f"γ={gamma_true:.6f} — the A·tanh(X₁) column is mis-signed or mis-ordered."
    )
    assert abs(gamma_hat) / se > 2.0, (
        f"γ̂={gamma_hat:.6f} is only {abs(gamma_hat) / se:.2f} SE from zero at "
        f"n={cell.n} — the NLG effect-modifying channel is not recoverable, so the "
        "L1-oracle collider treatment arm would carry no signal."
    )
    assert abs(gamma_hat) > 0.5 * abs(gamma_true), (
        f"γ̂={gamma_hat:.6f} is under half the builder's γ={gamma_true:.6f} — the "
        "fitted effect modification is a fraction of the truth, not merely noisy."
    )


# NOT mirrored from the triangle: `test_group_stratified_equivalence`. Same reason
# the LINEAR collider test gives, and it survives the family change unchanged. The
# triangle's pooled X₂ model is SATURATED in A, so per-group OLS is algebraically
# identical to it. The NLG collider's X₃ model is NOT — it carries
# {1, tanh(X1), A·tanh(X1), tanh(X2), A} with NO A·tanh(X2) interaction (the SCM specification
# channel (i) modifies the X₁ coefficient only), so η is CONSTRAINED equal across
# groups while a per-group fit would let it float. The two agree only up to sampling
# noise, and asserting an exact identity would be wrong, not merely tight.


# ---------------------------------------------------------------------------
# 4 — driver wiring: the collider runs the full ladder on the NLG family
# ---------------------------------------------------------------------------


def test_run_collider_runs_full_ladder_on_nlg():
    """The collider driver's condition set is L0 + L1-oracle + L2, and nlg is wired.

    The condition tuple is family-independent (it is a module constant, not a
    per-family choice), so asserting both here is what makes "the NLG cell runs the
    FULL three-condition ladder" a checked claim rather than an inference.
    """
    assert set(run_collider._CONDITIONS) == {"L0", "L1-oracle", "L2"}
    assert set(run_collider._FAMILIES) == {"linear", "nlg"}
    builder, form_spec_fn = run_collider._FAMILIES["nlg"]
    assert builder is make_nonlinear_gaussian_collider
    assert form_spec_fn is nlg_collider_form_spec


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
    cfg.recourse.grid.resolution = 21  # NOT the grid-resolution production pin (41) — test scale
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
                scm_builder=make_nonlinear_gaussian_collider,
                form_spec_fn=nlg_collider_form_spec,
                family="nlg",
                run_anchor=False,  # NO G1 anchor off the linear family (see test 7)
            )
    return cfg, runs


@pytest.mark.parametrize("regime", _REGIMES)
def test_believed_cf_populated_on_nlg_collider_l1_oracle_rows(ladder_runs, regime):
    """believed_cf exists and is non-null on every found NLG L1-oracle row.

    The believed_cf mechanism is topology- AND family-agnostic (scoring.py
    routes the believed side through whatever model object generated the solution),
    so this VERIFIES rather than reimplements — the new surface is only that the
    estimated SCM behind the L1-oracle model carries two tanh channels into X₃.
    """
    table = ladder_runs[1][regime].table
    l1 = table[(table["condition"] == "L1-oracle") & table["found"]]
    assert len(l1) > 0, "no found L1-oracle rows on the NLG collider"
    for feature in _FEATURES:
        col = f"believed_cf_{feature}"
        assert col in table.columns, f"{col} missing from the NLG collider table"
        assert l1[col].notna().all(), (
            f"{regime}: null {col} on a found L1-oracle row — the believed path is "
            "not being routed through the estimated NLG SCM."
        )


def test_believed_cf_diverges_at_nlg_collider_l1_oracle(ladder_runs):
    """The believed and realized CF paths are computed independently.

    NO magnitude assertion (mirrors every other wiring test): the point is only
    that the channel EXISTS — f̂_L1 ≠ f_true maps some individual's chosen δ to a
    different downstream X₃ than the true SCM does.
    """
    table = ladder_runs[1]["effect_modifying"].table
    sub = table[(table["condition"] == "L1-oracle") & table["found"]]
    believed = sub[[f"believed_cf_{f}" for f in _FEATURES]].to_numpy()
    realized = sub[[f"realized_cf_{f}" for f in _FEATURES]].to_numpy()
    assert np.any(believed != realized), (
        "believed_cf identical to realized_cf on every NLG collider L1-oracle row "
        "— the believed path is not going through the estimated SCM."
    )


def test_nlg_estimator_is_wired_into_the_collider_ladder(ladder_runs):
    """The L1-oracle rung was fitted by the NLG estimator on the NLG template.

    Guards the form-template-known provenance claim end to end: family selection is by the
    CONFIG/RUN family field (the SCM carries no family attribute), so a silent
    fallback to the linear estimator/template would leave the ladder running and
    only show up here. This is also the ASSERTION that discharges the NLG-collider
    question "is NLGOLSEstimator topology-agnostic?" — it is: nothing in
    estimation/ was touched for the collider, and the estimator reaches the NLG
    collider's transformed columns through the NLG-triangle resolved-terms channel.
    """
    for regime in _REGIMES:
        fitted = ladder_runs[1][regime].fitted
        assert fitted.metadata["family"] == "nlg"
        assert "NLGOLSEstimator" in fitted.metadata["estimator"]
        assert "tanh(X1)" in fitted.coefficients["X3"]
        assert "tanh(X2)" in fitted.coefficients["X3"]


@pytest.mark.parametrize("regime", _REGIMES)
def test_gap_cost_collapse_at_nlg_collider_l1_oracle(ladder_runs, regime, tmp_path):
    """Gap_cost == 0 EXACTLY at L1-oracle on the NLG collider (PS-6 carries over).

    # [PS-6; the ℓ₂ intervention-cost convention] Expected, NOT a bug: cost is a property of the
    # ACTION, so
    # per-row believed cost equals realized cost bitwise for the chosen δ at every
    # SCM-carrying condition — a statement about the cost functional that neither
    # the topology nor the functional family can touch. The H2 signal on this cell
    # therefore lives in ValidityGap_g(c), as everywhere else. Asserted on a live
    # run through the real aggregation layer (tmp dir), so it does not depend on
    # results/ being present.
    """
    _cfg, runs = ladder_runs
    table = runs[regime].table
    for condition in _CONDITIONS:
        sub = table[table["condition"] == condition]
        sub.to_csv(tmp_path / f"scoring_table_{regime}_{condition}.csv", index=False)

    cell = CellKeys(topology="collider", family="nlg", regime=regime, seed=_SEED)
    _by_group_path, by_cell_path = aggregate_run(tmp_path, tmp_path / "agg", cell)
    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")

    l1_rows = by_cell[by_cell["condition"] == "L1-oracle"]
    assert len(l1_rows) == 1, "expected exactly one NLG collider L1-oracle by-cell row"
    assert float(l1_rows.iloc[0]["Gap_cost"]) == 0.0, (
        f"{regime}: PS-6 collapse violated at the NLG collider L1-oracle row — "
        "Gap_cost must be identically zero at an SCM-carrying condition."
    )
    assert (by_cell["family"] == "nlg").all()
    assert (by_cell["topology"] == "collider").all()
    # The per- the ℓ₂ intervention-cost convention identity the cell-level collapse rests on.
    found = table[(table["condition"] == "L1-oracle") & table["found"]]
    np.testing.assert_array_equal(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy()
    )


# ---------------------------------------------------------------------------
# 7 — no G1 anchor off the linear family
# ---------------------------------------------------------------------------


def test_no_g1_anchor_on_the_nlg_collider():
    """run_regime refuses the G1 anchor for a non-linear family, loudly.

    The Ehyaei closed form the anchor pulls back assumes a LINEAR SCM and a linear
    classifier (the ℓ₂ intervention-cost convention), so on an NLG cell it is
    inapplicable, not merely
    loose. The collider driver enforces the same rule from the config's family key
    (`run_anchor = family == "linear"`).
    """
    cfg = _small_cfg()
    with pytest.raises(ValueError, match="family='linear'"):
        run_regime(
            cfg,
            "additive",
            conditions=_CONDITIONS,
            scm_builder=make_nonlinear_gaussian_collider,
            form_spec_fn=nlg_collider_form_spec,
            family="nlg",
            run_anchor=True,
        )


# ---------------------------------------------------------------------------
# 8 — the shipped NLG collider aggregates carry family=nlg rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", _REGIMES)
def test_shipped_nlg_collider_aggregates_carry_family_rows(regime):
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
        pytest.skip(f"NLG collider result CSVs not present locally: {missing}")

    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")
    assert set(by_cell["condition"]) == set(_CONDITIONS), (
        f"{regime}: aggregate_by_cell is missing conditions — regenerate with "
        "`python -m icknowledge.recourse.run_collider "
        "configs/recourse_collider_nlg.yaml`."
    )
    assert (by_cell["family"] == "nlg").all()
    assert (by_cell["topology"] == "collider").all()
    assert float(by_cell[by_cell["condition"] == "L1-oracle"].iloc[0]["Gap_cost"]) == 0.0

    by_group = pd.read_csv(by_group_path, float_precision="round_trip")
    assert (by_group["family"] == "nlg").all()
    assert set(by_group["condition"]) == set(_CONDITIONS)

    table = pd.read_csv(per_individual, float_precision="round_trip")
    found = table[table["found"]]
    # believed_cf present and non-null on the shipped L1-oracle rows (all features).
    for feature in _FEATURES:
        assert found[f"believed_cf_{feature}"].notna().all()
    # The ℓ₂ intervention-cost identity, per row, on the shipped
    # artifact.
    np.testing.assert_array_equal(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy()
    )
