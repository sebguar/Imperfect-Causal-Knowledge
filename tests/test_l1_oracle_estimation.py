"""L1-oracle equation estimation: coefficient recovery + estimated-SCM assembly gates.

Permanent coefficient-recovery smoke test, parallel to the
hand-derived counterfactual gate: a failure here is an ESTIMATION BUG, not a finding
(G1 style). True coefficients are read PROGRAMMATICALLY from the
builder-function defaults (the `structural_g` pattern of the G1 anchor test), so
the tests survive a coefficient retune without edits.

Covers, per the estimation spec:
  1. coefficient recovery, additive linear triangle (each coef within ~4 SE);
  2. coefficient recovery, effect-modifying linear triangle (incl. γ, with the
     builder-verified pooled mapping β₁ = (g_pos+g_neg)/2, γ = (g_pos−g_neg)/2);
  3. additive regime carries NO interaction column (hard-zero by omission);
  4. group-stratified OLS ≡ pooled interaction model (design-matrix sanity);
  5. zero-estimation-error collapse to L2 (the assembly-path tether);
  6. abduction consistency (û tracks true u within sampling noise);
  7. no-regularization guard: OLS == numpy.linalg.lstsq to ~1e-10;
  plus no sampling from the estimated SCM, and manifest-provenance checks.
"""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from icknowledge.estimation import (
    FittedEquations,
    LinearOLSEstimator,
    build_estimated_scm,
    dataset_identity,
    design_matrix,
    equation_from_coefficients,
    estimation_provenance,
)
from icknowledge.scm import make_linear_triangle, triangle_form_spec
from icknowledge.utils.config import load_config

REGIMES = ["additive", "effect_modifying"]


def _builder_defaults() -> dict[str, float]:
    """True coefficients read from the builder signature — never hardcoded targets."""
    params = inspect.signature(make_linear_triangle).parameters
    return {k: float(params[k].default) for k in ("a", "b", "g", "g_pos", "g_neg", "sigma")}


def _true_coefficients(regime: str, d: dict[str, float]) -> dict[str, dict[str, float]]:
    """True values of every templated term, in the pooled parametrization.

    X1 := a·A + U1 in both regimes → intercept 0, A-coef a.

    X2, additive: X2 := b·A + g·X1 + U2 → intercept 0, A-coef b, X1-coef g.

    X2, effect-modifying — mapping VERIFIED against the actual builder
    (make_linear_triangle): the builder computes slope = np.where(A > 0, g_pos,
    g_neg) and X2 := slope·X1 + U2, with NO additive b·A term and NO intercept.
    Under the centered A ∈ {−1, +1} encoding (asserted on the data below),
    slope(A) = (g_pos + g_neg)/2 + A·(g_pos − g_neg)/2, so the pooled
    parametrization X2 = β₀ + β₁·X1 + β₂·A + γ·(A·X1) + ε₂ has
        β₁ = (g_pos + g_neg)/2,  γ = (g_pos − g_neg)/2,  β₀ = β₂ = 0.
    """
    truths = {"X1": {"intercept": 0.0, "A": d["a"]}}
    if regime == "additive":
        truths["X2"] = {"intercept": 0.0, "A": d["b"], "X1": d["g"]}
    else:
        truths["X2"] = {
            "intercept": 0.0,
            "A": 0.0,
            "X1": (d["g_pos"] + d["g_neg"]) / 2.0,
            "A*X1": (d["g_pos"] - d["g_neg"]) / 2.0,
        }
    return truths


def _make_cell(regime: str) -> SimpleNamespace:
    """One fitted estimation cell at the production sample size and master seed."""
    cfg = load_config("configs/recourse_triangle.yaml")
    n = int(cfg.classifier.N)  # production sample size, read from config — not hardcoded
    seed = int(cfg.seed)
    defaults = _builder_defaults()
    scm = make_linear_triangle(regime)
    data = scm.sample(n, seed=seed)
    # Premise of the pooled mapping in _true_coefficients: centered ±1 encoding.
    assert set(np.unique(data["A"]).tolist()) == {-1.0, 1.0}
    form = triangle_form_spec(regime)
    fitted = LinearOLSEstimator().fit(
        data, form, extra_metadata={"regime": regime, "family": "linear", "dataset_seed": seed}
    )
    return SimpleNamespace(
        regime=regime,
        scm=scm,
        data=data,
        n=n,
        seed=seed,
        defaults=defaults,
        form=form,
        fitted=fitted,
        truths=_true_coefficients(regime, defaults),
    )


@pytest.fixture(scope="module")
def cells() -> dict[str, SimpleNamespace]:
    return {regime: _make_cell(regime) for regime in REGIMES}


def _ols_standard_errors(cell: SimpleNamespace, node: str) -> dict[str, float]:
    """Classical OLS SEs from the SAME design matrix the estimator used."""
    x, terms = design_matrix(cell.data, cell.form[node])
    xi = np.column_stack([np.ones(len(x)), x])
    y = cell.data[node].to_numpy(dtype=float)
    coefs = cell.fitted.coefficients[node]
    beta_hat = np.array([coefs["intercept"], *(coefs[t] for t in terms)])
    resid = y - xi @ beta_hat
    sigma2 = float(resid @ resid) / (len(y) - xi.shape[1])
    cov = sigma2 * np.linalg.inv(xi.T @ xi)
    se = np.sqrt(np.diag(cov))
    return dict(zip(["intercept", *terms], se, strict=True))


def _assert_recovery(cell: SimpleNamespace, max_se_multiple: float = 4.0) -> None:
    for node, truth in cell.truths.items():
        fitted = cell.fitted.coefficients[node]
        assert set(fitted) == set(truth), (
            f"{cell.regime}/{node}: fitted terms {sorted(fitted)} != templated terms "
            f"{sorted(truth)}"
        )
        se = _ols_standard_errors(cell, node)
        multiples = {t: abs(fitted[t] - truth[t]) / se[t] for t in truth}
        report = ", ".join(f"{t}: {m:.2f} SE" for t, m in multiples.items())
        assert all(m <= max_se_multiple for m in multiples.values()), (
            f"{cell.regime}/{node}: coefficient(s) beyond {max_se_multiple} SE of truth "
            f"at n={cell.n} — SE-multiples: {report}"
        )


# ---------------------------------------------------------------------------
# 1 + 2 — coefficient recovery (smoke gate)
# ---------------------------------------------------------------------------

def test_coefficient_recovery_additive(cells):
    """Every fitted coefficient within ~4 SE of the builder's true value (additive)."""
    _assert_recovery(cells["additive"])


def test_coefficient_recovery_effect_modifying(cells):
    """Same for the effect-modifying regime, INCLUDING γ via the pooled mapping."""
    _assert_recovery(cells["effect_modifying"])


# ---------------------------------------------------------------------------
# 3 — additive regime carries no interaction column (hard-zero by omission)
# ---------------------------------------------------------------------------

def test_additive_has_no_interaction_column(cells):
    cell = cells["additive"]
    x, terms = design_matrix(cell.data, cell.form["X2"])
    assert "A*X1" not in terms and x.shape[1] == 2, (
        f"additive X2 design matrix must hold exactly the parent columns, got {terms}"
    )
    assert "A*X1" not in cell.fitted.coefficients["X2"], (
        "γ must be hard-zero by OMISSION from the design matrix, "
        "not fit-then-discarded — no γ key may exist in the additive regime."
    )


# ---------------------------------------------------------------------------
# 4 — group-stratified equivalence (saturated pooled model == per-group OLS)
# ---------------------------------------------------------------------------

def test_group_stratified_equivalence(cells):
    """Per-group OLS must reproduce the pooled interaction model's implied group lines.

    The pooled model {1, A, X1, A·X1} is SATURATED in A, hence algebraically
    equivalent to fitting X2 = c_g + s_g·X1 separately per group: implied
    s_g = β₁ + γ·a, c_g = β₀ + β₂·a for a ∈ {−1, +1}. A discrepancy beyond
    numerical precision means the design matrix is malformed.
    """
    from sklearn.linear_model import LinearRegression

    cell = cells["effect_modifying"]
    pooled = cell.fitted.coefficients["X2"]
    for a_val in (-1.0, 1.0):
        sub = cell.data[cell.data["A"] == a_val]
        reg = LinearRegression(fit_intercept=True).fit(
            sub[["X1"]].to_numpy(dtype=float), sub["X2"].to_numpy(dtype=float)
        )
        implied_slope = pooled["X1"] + pooled["A*X1"] * a_val
        implied_intercept = pooled["intercept"] + pooled["A"] * a_val
        assert abs(float(reg.coef_[0]) - implied_slope) < 1e-8, (
            f"A={a_val:g}: per-group slope {float(reg.coef_[0]):.12f} != pooled-implied "
            f"{implied_slope:.12f} — malformed design matrix."
        )
        assert abs(float(reg.intercept_) - implied_intercept) < 1e-8, (
            f"A={a_val:g}: per-group intercept {float(reg.intercept_):.12f} != "
            f"pooled-implied {implied_intercept:.12f} — malformed design matrix."
        )


# ---------------------------------------------------------------------------
# 5 — zero-estimation-error collapse to L2 (the important one)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", REGIMES)
def test_zero_estimation_error_collapses_to_l2(cells, regime):
    """Estimated SCM built from the TRUE coefficients == true SCM, to machine precision.

    Same style of tether as the M_S = I collapse in the G1 anchor check: it pins
    the estimated-SCM ASSEMBLY path (equation_from_coefficients +
    build_estimated_scm) to the known-correct L2 path at the point where the two
    must coincide by construction — so a later L1-oracle discrepancy is
    attributable to ESTIMATION ERROR rather than to a bug in build_estimated_scm.
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
    x1 = batch["X1"].to_numpy(dtype=float)
    x2 = batch["X2"].to_numpy(dtype=float)
    actions = [
        {},  # identity counterfactual
        {"X1": x1 + 1.0},  # per-row do(X1 := x1 + 1): X2 must propagate identically
        {"X2": 0.0},  # scalar do(X2 := 0)
        {"X1": x1 - 0.5, "X2": x2 + 2.0},  # joint action severs X1→X2
    ]
    for action in actions:
        cf_true = cell.scm.counterfactual(batch, action)
        cf_est = est_scm.counterfactual(batch, action)
        # 1e-12 absolute on O(1–10) values == machine precision up to reassociation
        # (e.g. (g_pos+g_neg)/2 + (g_pos−g_neg)/2 vs g_pos differs at ~1e-16).
        max_diff = float(np.max(np.abs(cf_true.to_numpy() - cf_est.to_numpy())))
        assert max_diff < 1e-12, (
            f"{regime}: estimated SCM with TRUE coefficients deviates from the true SCM "
            f"(max |diff| = {max_diff:.3e} for action {sorted(action)}) — this is an "
            "assembly bug in build_estimated_scm, NOT estimation error."
        )


# ---------------------------------------------------------------------------
# 6 — abduction consistency (the residual gap IS the estimation error)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", REGIMES)
def test_abduction_consistency(cells, regime):
    """û = x − f̂(pa) under the estimated SCM tracks the true u within sampling noise.

    NOT an equality test: the per-row gap û − u equals f(pa) − f̂(pa), i.e. the
    coefficient estimation error itself — expected small (O(SE) ≈ σ/√n per term)
    but nonzero. The root A abducts exactly (u_A = x_A).
    """
    cell = cells[regime]
    fitted = LinearOLSEstimator().fit(cell.data, cell.form)
    est_scm = build_estimated_scm(cell.scm, fitted)
    u_true = cell.scm.sample_noise(cell.n, seed=cell.seed)  # same seed → same noise as sample
    u_hat = est_scm.abduct(cell.data)

    assert np.array_equal(u_hat["A"].to_numpy(), u_true["A"].to_numpy()), (
        "root A must abduct exactly (u = x, no fitted mean function)."
    )
    for node in ("X1", "X2"):
        gap = u_hat[node].to_numpy() - u_true[node].to_numpy()
        rms = float(np.sqrt(np.mean(gap**2)))
        max_abs = float(np.max(np.abs(gap)))
        assert 0.0 < rms < 0.1, (
            f"{regime}/{node}: û-vs-u RMS gap {rms:.4f} outside (0, 0.1) at n={cell.n} — "
            "zero means an oracle leak; large means the fit or abduction is broken."
        )
        assert max_abs < 0.5, (
            f"{regime}/{node}: max |û − u| = {max_abs:.4f} ≥ 0.5 — beyond sampling noise."
        )


# ---------------------------------------------------------------------------
# 7 — no-regularization guard (numerical — fails loudly on a penalized swap)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", REGIMES)
def test_no_regularization_guard_lstsq(cells, regime):
    """Fitted coefficients must equal numpy.linalg.lstsq on the same design matrix.

    A numerical guard, not a type check: Ridge/Lasso/ElasticNet (or any penalized
    estimator swapped in later) shrink coefficients away from the least-squares
    solution and fail this to ~1e-10.
    """
    cell = cells[regime]
    for node, node_form in cell.form.items():
        x, terms = design_matrix(cell.data, node_form)
        xi = np.column_stack([np.ones(len(x)), x])
        y = cell.data[node].to_numpy(dtype=float)
        beta_lstsq = np.linalg.lstsq(xi, y, rcond=None)[0]
        coefs = cell.fitted.coefficients[node]
        beta_fit = np.array([coefs["intercept"], *(coefs[t] for t in terms)])
        max_diff = float(np.max(np.abs(beta_fit - beta_lstsq)))
        assert max_diff < 1e-10, (
            f"{regime}/{node}: fitted coefficients deviate from lstsq by {max_diff:.3e} "
            "— a penalized/regularized estimator has been swapped in "
            "(violates the unregularized-OLS rule)."
        )


# ---------------------------------------------------------------------------
# Sampling guard + provenance (manifest evidence for the form template and the sample)
# ---------------------------------------------------------------------------

def test_estimated_scm_refuses_to_sample(cells):
    """Only the mean function is fitted — ancestral sampling must fail LOUDLY,
    not silently draw from ground-truth noise distributions (oracle leak)."""
    cell = cells["additive"]
    est_scm = build_estimated_scm(cell.scm, cell.fitted)
    with pytest.raises(RuntimeError, match="mean function only"):
        est_scm.sample(10, seed=0)


def test_manifest_provenance_roundtrip(cells, tmp_path):
    """The manifest carries estimator id, serialized FormSpec, coefficients, dataset
    identity (shared-sample audit trail), regime, and family — and survives JSON round-trip."""
    from icknowledge.utils.manifest import write_run_manifest

    cell = cells["effect_modifying"]
    cfg = load_config("configs/recourse_triangle.yaml")
    path = write_run_manifest(tmp_path / "run", cfg, estimation=estimation_provenance(cell.fitted))
    manifest = json.loads(path.read_text())

    est = manifest["estimation"]
    assert "LinearRegression" in est["metadata"]["estimator"]
    assert est["metadata"]["regime"] == "effect_modifying"
    assert est["metadata"]["family"] == "linear"
    assert est["form_spec"]["X2"]["interactions"] == [["A", "X1"]]
    assert set(est["coefficients"]["X2"]) == {"intercept", "A", "X1", "A*X1"}
    # The recorded identity must match the dataset object the estimator was
    # handed — the same sample the L1-discovered PC step is bound to reuse.
    assert est["metadata"]["dataset"] == _to_plain(dataset_identity(cell.data))


def _to_plain(obj):
    """Normalize via JSON so dict/list comparisons match the round-tripped manifest."""
    return json.loads(json.dumps(obj))


# ---------------------------------------------------------------------------
# Estimation-sample contract (structural)
# ---------------------------------------------------------------------------

def test_estimator_takes_dataset_object_not_seed():
    """The shared-sample contract is enforced by the entry point's signature: `fit` takes the
    upstream
    dataset OBJECT (a DataFrame), not a seed or a re-sampling callable."""
    params = inspect.signature(LinearOLSEstimator.fit).parameters
    assert list(params)[:3] == ["self", "data", "form"]
    assert params["data"].annotation in ("pd.DataFrame", pd.DataFrame)
