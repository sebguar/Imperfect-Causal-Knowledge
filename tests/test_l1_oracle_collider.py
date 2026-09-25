"""L1-oracle on the COLLIDER topology: recovery, wiring, PS-6 collapse.

The collider analogue of tests/test_l1_oracle_estimation.py (recovery) plus the
collider-relevant slice of tests/test_l1_oracle_wiring.py (believed_cf channel):
a failure here is a WIRING BUG, not a finding. Covers, per the L1-oracle collider spec:

  1. coefficient recovery on the linear collider, additive regime (each templated
     coefficient within ~4 SE of the SCM builder default);
  2. coefficient recovery, effect-modifying regime, INCLUDING the γ interaction;
  3. γ̂ sign + non-trivial magnitude, and the additive regime's hard-zero-by-
     omission (no A*X1 key exists at all);
  4. driver wiring — run_collider runs the full L0 + L1-oracle + L2 ladder;
  5. live small-scale run: believed_cf populated and non-null on L1-oracle rows,
     and the believed path genuinely diverges from the realized one;
  6. PS-6 collapse on the collider — Gap_cost identically 0 at L1-oracle, with
     believed_cost == realized_cost per row (the intervention-cost identity it rests on);
  7. the shipped collider aggregate CSVs carry an L1-oracle row (skipped when
     results/ is absent — it is gitignored and machine-local).

True coefficients are read PROGRAMMATICALLY from `make_linear_collider`'s
defaults (the `structural_g` pattern), so a register-logged PS-2 retune does not
require test edits. The 4-SE recovery criterion is IMPORTED from the triangle
L1-oracle estimation test rather than restated, so the two topologies are held to
one criterion by construction.
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
from icknowledge.estimation import LinearOLSEstimator, design_matrix
from icknowledge.recourse import run_collider
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import collider_form_spec, make_linear_collider
from icknowledge.utils.config import load_config

# ONE criterion for both topologies: the triangle test owns the SE machinery and
# the 4-SE bound; importing it (rather than restating) is what keeps the collider
# from silently drifting to a looser gate. Precedent: test_aggregation_layer.py
# imports the pinned checksum constants from test_l1_oracle_wiring.py.
from tests.test_l1_oracle_estimation import _assert_recovery, _ols_standard_errors

_CONFIG_PATH = "configs/recourse_collider.yaml"
_RESULTS_DIR = Path("results/recourse_collider")
_REGIMES = ("additive", "effect_modifying")
_FEATURES = ("X1", "X2", "X3")
_SEED = 20260710  # the config's master seed (the seed-generation meta-entropy); the CSVs' cell key


def _builder_defaults() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_linear_collider).parameters
    keys = ("a1", "a2", "beta", "gamma", "eta", "beta_a", "c1", "c2", "c3", "sigma")
    return {k: float(params[k].default) for k in keys}


def _true_coefficients(regime: str, d: dict[str, float]) -> dict[str, dict[str, float]]:
    """True values of every templated term, in the estimator's parametrization.

    # [the SCM specification] Unlike the triangle — whose effect-modifying builder uses a
    # np.where group-slope switch that must be mapped onto the pooled
    # {1, A, X1, A·X1} basis — the collider builder is ALREADY WRITTEN in the
    # pooled basis the form spec templates:
    #     X1 := c1 + a1·A + ε₁
    #     X2 := c2 + a2·A + ε₂
    #     X3 := c3 + β·X1 + γ·(A·X1) + η·X2 + β_A·A + ε₃
    # so each fitted term maps 1:1 to a builder default with NO reparametrization.
    # γ is absent from the additive dict because it is hard-zero by COLUMN
    # OMISSION there, not fit-then-discarded — see test 3.
    """
    truths = {
        "X1": {"intercept": d["c1"], "A": d["a1"]},
        "X2": {"intercept": d["c2"], "A": d["a2"]},
        "X3": {
            "intercept": d["c3"],
            "A": d["beta_a"],
            "X1": d["beta"],
            "X2": d["eta"],
        },
    }
    if regime == "effect_modifying":
        truths["X3"]["A*X1"] = d["gamma"]
    return truths


def _make_cell(regime: str) -> SimpleNamespace:
    """One fitted collider estimation cell, on the sample the L1-oracle form-template-known
    semantics mandates.

    # [the L1-oracle form-template-known semantics] The estimation sample IS the classifier's
    # training sample
    # (ClassifierResult.dataset) — no redraw, no held-out split — so this fixture
    # exercises the same data the pipeline's L1-oracle rung is fitted on, not a
    # convenience draw. Unregularized OLS, mean function only; the no-penalty
    # guard itself lives in the triangle test (same estimator class).
    """
    cfg = load_config(_CONFIG_PATH)
    scm = make_linear_collider(regime)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        clf = build_classifier(scm, cfg)
    data = clf.dataset
    # Premise of the centered ±1 encoding the A·X1 column's interpretation rests on.
    assert set(np.unique(data["A"]).tolist()) == {-1.0, 1.0}
    form = collider_form_spec(regime)
    fitted = LinearOLSEstimator().fit(
        data, form, extra_metadata={"regime": regime, "family": "linear"}
    )
    return SimpleNamespace(
        regime=regime,
        scm=scm,
        data=data,
        n=len(data),
        form=form,
        fitted=fitted,
        truths=_true_coefficients(regime, _builder_defaults()),
    )


@pytest.fixture(scope="module")
def cells() -> dict[str, SimpleNamespace]:
    return {regime: _make_cell(regime) for regime in _REGIMES}


# ---------------------------------------------------------------------------
# 1 + 2 — coefficient recovery on the collider (both regimes)
# ---------------------------------------------------------------------------


def test_collider_coefficient_recovery_additive(cells):
    """Every fitted collider coefficient within ~4 SE of its SCM builder default.

    X₃'s two-endogenous-parent-plus-root equation is the new surface relative to
    the triangle: β (X₁→X₃), η (X₂→X₃) and β_A (A→X₃) are estimated jointly, so a
    parent-set or design-matrix-ordering error on X₃ surfaces here.
    """
    _assert_recovery(cells["additive"])


def test_collider_coefficient_recovery_effect_modifying(cells):
    """Same for the effect-modifying regime, INCLUDING γ (the A·X₁ interaction).

    Note η and β_A must ALSO come back at their additive values: the SCM specification's channel (i)
    modifies X₃'s X₁-coefficient ONLY, so a γ̂ that ate part of η̂ or β̂_A would be
    an interaction-column construction bug, and this joint bound catches it.
    """
    _assert_recovery(cells["effect_modifying"])


# ---------------------------------------------------------------------------
# 3 — γ̂: correct sign, non-trivial magnitude; hard-zero by omission in additive
# ---------------------------------------------------------------------------


def test_gamma_hat_sign_and_nontrivial_magnitude(cells):
    """γ̂ carries the builder's sign and is many SE from zero (not a noise blip).

    The triangle's interaction check is the 4-SE two-sided recovery bound (tests
    1-2 above already apply it to A*X1 via `_assert_recovery`); this adds the two
    directional statements that bound alone does not make — SIGN agreement with
    the SCM default, and distinguishability from zero at the same 4-SE scale.
    Effect-modification that is real but unrecoverable would pass recovery and
    fail here.
    """
    cell = cells["effect_modifying"]
    gamma_true = _builder_defaults()["gamma"]
    gamma_hat = cell.fitted.coefficients["X3"]["A*X1"]
    se = _ols_standard_errors(cell, "X3")["A*X1"]

    assert np.sign(gamma_hat) == np.sign(gamma_true), (
        f"γ̂={gamma_hat:.6f} has the wrong sign against the SCM default "
        f"γ={gamma_true:.6f} — the A·X₁ column is mis-signed or mis-ordered."
    )
    assert abs(gamma_hat) / se > 4.0, (
        f"γ̂={gamma_hat:.6f} is only {abs(gamma_hat) / se:.2f} SE from zero at "
        f"n={cell.n} — the effect-modifying channel is not recoverable at this "
        "sample size, so the L1-oracle collider treatment arm carries no signal."
    )


def test_additive_regime_has_no_interaction_column(cells):
    """γ is hard-zero by COLUMN OMISSION in the additive control (the SCM specification).

    Not fit-then-discarded: no A*X1 column may exist in X₃'s design matrix and no
    A*X1 key may exist in the fitted coefficients.
    """
    cell = cells["additive"]
    x, terms = design_matrix(cell.data, cell.form["X3"])
    assert terms == ["A", "X1", "X2"] and x.shape[1] == 3, (
        f"additive X3 design matrix must hold exactly the parent columns, got {terms}"
    )
    assert "A*X1" not in cell.fitted.coefficients["X3"], (
        "γ must be hard-zero by OMISSION from the design matrix (the SCM specification), "
        "not fit-then-discarded — no γ key may exist in the additive regime."
    )


# NOT mirrored from the triangle: `test_group_stratified_equivalence`. The
# triangle's pooled X₂ model {1, A, X1, A·X1} is SATURATED in A, so per-group OLS
# is algebraically identical to it. The collider's X₃ model is NOT — it carries
# {1, A, X1, X2, A·X1} with NO A·X2 interaction (the SCM channel (i) modifies the X₁
# coefficient only), so η is CONSTRAINED equal across groups while a per-group fit
# would let it float. The two agree only up to sampling noise, and asserting an
# exact identity would be wrong, not merely tight.


# ---------------------------------------------------------------------------
# 4 — driver wiring: the collider runs the full ladder
# ---------------------------------------------------------------------------


def test_run_collider_runs_full_ladder():
    """The collider driver's condition set is L0 + L1-oracle + L2."""
    assert set(run_collider._CONDITIONS) == {"L0", "L1-oracle", "L2"}


# ---------------------------------------------------------------------------
# 5 + 6 — live small-scale ladder: believed_cf channel + PS-6 collapse
# ---------------------------------------------------------------------------
#
# Small resolution / N (as in tests/test_collider_scm.py Groups 7-8): the
# faithful-resolution numbers are the driver's artifact, not this test's job.
# These assertions are STRUCTURAL (a column is populated; an identity holds), so
# they are resolution-independent — and unlike test 7 they do not depend on
# results/ existing, which makes them the durable half of the gate.


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
                conditions=("L0", "L1-oracle", "L2"),
                scm_builder=make_linear_collider,
                form_spec_fn=collider_form_spec,
                run_anchor=False,  # anchor is test_collider_scm.py's gate, not this one
            )
    return cfg, runs


@pytest.mark.parametrize("regime", _REGIMES)
def test_believed_cf_populated_on_collider_l1_oracle_rows(ladder_runs, regime):
    """believed_cf exists and is non-null on every found L1-oracle row (all 3 features).

    The believed_cf mechanism is topology-agnostic (scoring.py routes the
    believed side through whatever model object generated the solution), so this
    verifies rather than reimplements — the collider's third feature X₃ is the
    only new surface.
    """
    table = ladder_runs[1][regime].table
    l1 = table[(table["condition"] == "L1-oracle") & table["found"]]
    assert len(l1) > 0, "no found L1-oracle rows on the collider"
    for feature in _FEATURES:
        col = f"believed_cf_{feature}"
        assert col in table.columns, f"{col} missing from the collider scoring table"
        assert l1[col].notna().all(), (
            f"{regime}: null {col} on a found L1-oracle row — the believed path is "
            "not being routed through the estimated SCM."
        )


def test_believed_cf_diverges_at_collider_l1_oracle(ladder_runs):
    """The believed and realized CF paths are computed independently on the collider.

    NO magnitude assertion (mirrors the triangle wiring test): the point is only
    that the channel EXISTS — f̂_L1 ≠ f_true maps some individual's chosen δ to a
    different downstream X₃ than the true SCM does.
    """
    table = ladder_runs[1]["effect_modifying"].table
    sub = table[(table["condition"] == "L1-oracle") & table["found"]]
    believed = sub[[f"believed_cf_{f}" for f in _FEATURES]].to_numpy()
    realized = sub[[f"realized_cf_{f}" for f in _FEATURES]].to_numpy()
    assert np.any(believed != realized), (
        "believed_cf identical to realized_cf on every collider L1-oracle row — "
        "the believed path is not going through the estimated SCM."
    )


@pytest.mark.parametrize("regime", _REGIMES)
def test_gap_cost_collapse_at_collider_l1_oracle(ladder_runs, regime, tmp_path):
    """Gap_cost == 0 EXACTLY at L1-oracle on the collider (PS-6 carries over).

    # [PS-6; the ℓ₂ intervention-cost convention] Expected, NOT a bug: cost is a property of the
    # ACTION, so
    # per-row believed cost equals realized cost bitwise for the chosen δ at every
    # SCM-carrying condition, and the cell-level Gap collapses to an algebraic
    # zero. The H2 signal on the collider therefore lives in ValidityGap_g(c), the
    # same as on the triangle. Asserted on a live run through the real aggregation
    # layer (tmp dir), so it does not depend on results/ being present.
    """
    _cfg, runs = ladder_runs
    table = runs[regime].table
    for condition in ("L0", "L1-oracle", "L2"):
        sub = table[table["condition"] == condition]
        sub.to_csv(tmp_path / f"scoring_table_{regime}_{condition}.csv", index=False)

    cell = CellKeys(topology="collider", family="linear", regime=regime, seed=_SEED)
    _by_group_path, by_cell_path = aggregate_run(tmp_path, tmp_path / "agg", cell)
    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")

    l1_rows = by_cell[by_cell["condition"] == "L1-oracle"]
    assert len(l1_rows) == 1, "expected exactly one collider L1-oracle by-cell row"
    assert float(l1_rows.iloc[0]["Gap_cost"]) == 0.0, (
        f"{regime}: PS-6 collapse violated at the collider L1-oracle row — "
        "Gap_cost must be identically zero at an SCM-carrying condition."
    )
    # The per- the ℓ₂ intervention-cost convention identity the cell-level collapse rests on.
    found = table[(table["condition"] == "L1-oracle") & table["found"]]
    np.testing.assert_array_equal(
        found["believed_cost"].to_numpy(), found["realized_cost"].to_numpy()
    )


# ---------------------------------------------------------------------------
# 7 — the shipped collider aggregates carry an L1-oracle row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", _REGIMES)
def test_shipped_collider_aggregates_carry_l1_oracle_row(regime):
    """The regenerated production CSVs contain a condition=L1-oracle row, Gap_cost 0.

    results/ is gitignored and machine-local, so this SKIPS cleanly on a fresh
    clone (the same discipline as test_aggregation_layer.py); tests 5-6 above are
    the durable, run-from-source half of the same gate.
    """
    by_cell_path = _RESULTS_DIR / "aggregates" / f"aggregate_by_cell_{regime}.csv"
    per_individual = _RESULTS_DIR / f"scoring_table_{regime}_L1-oracle.csv"
    missing = [str(p) for p in (by_cell_path, per_individual) if not p.exists()]
    if missing:
        pytest.skip(f"collider result CSVs not present locally: {missing}")

    by_cell = pd.read_csv(by_cell_path, float_precision="round_trip")
    rows = by_cell[by_cell["condition"] == "L1-oracle"]
    assert len(rows) == 1, (
        f"{regime}: aggregate_by_cell carries no L1-oracle row — regenerate with "
        "`python -m icknowledge.recourse.run_collider`."
    )
    assert (rows["topology"] == "collider").all()
    assert float(rows.iloc[0]["Gap_cost"]) == 0.0

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
