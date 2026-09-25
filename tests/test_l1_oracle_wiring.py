"""L1-oracle wiring gates: contract, believed_cf schema, manifest, regression.

Permanent wiring tests, parallel in style to the G1 anchor gate: a failure here is
a WIRING BUG, not a finding. Covers, per the L1-oracle spec:

  1. contract compliance — L1OracleEstimatedSCMModel exposes the SAME
     predict(factual, intervention) contract as L0/L2;
  2. believed_cf == realized_cf at L0 and L2 (the believed_cf within-experiment sanity
     anchor, with the grid-edge exception set at L0);
  3. believed_cf ≠ realized_cf exists at L1-oracle (divergence channel wired);
  4. manifest carries the estimation provenance (the L1-oracle form-template-known semantics audit
  trail, incl. the
     classifier-training-sample identity);
  5. Regression — L0/L2 rows reproduce the frozen pilot CSVs exactly on
     every pre-existing column;
  6. sampler-guard non-reachability — the full 3-condition run never invokes the
     estimated SCM's guarded noise samplers (sampler-guard pin).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icknowledge.estimation import (
    build_estimated_scm,
    dataset_identity,
    estimation_provenance,
)
from icknowledge.recourse import (
    L0AssociationalModel,
    L1OracleEstimatedSCMModel,
    L2TrueSCMModel,
)
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import make_linear_triangle
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest

_CONFIG_PATH = "configs/recourse_triangle.yaml"
_REGIMES = ("additive", "effect_modifying")
_CONDITIONS = ("L0", "L2", "L1-oracle")  # run_triangle's full-ladder run set
_FEATURES = ("X1", "X2")

# Pilot schema, in pilot column order — the pre-existing columns the regression
# gate (test 5) protects. believed_cf_* insertion must not touch any of these.
_PRE_EXISTING_COLS = [
    "condition", "index", "A", "acted_set", "believed_cost", "realized_cost",
    "believed_validity", "realized_validity", "found",
    "delta_X1", "factual_X1", "realized_cf_X1",
    "delta_X2", "factual_X2", "realized_cf_X2",
]


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CONFIG_PATH)


# The full 3-condition pipeline (classifier bisection + grid over ~2500 negatives
# x 3 causal models) is expensive; run each regime once for the whole module.
_RUN_CACHE: dict = {}


def _run(cfg, regime):
    if regime not in _RUN_CACHE:
        _RUN_CACHE[regime] = run_regime(cfg, regime, conditions=_CONDITIONS)
    return _RUN_CACHE[regime]


def _cf_matrices(
    table: pd.DataFrame, condition: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(found-rows subframe, believed-CF matrix, realized-CF matrix) for one condition."""
    sub = table[(table["condition"] == condition) & table["found"]]
    believed = sub[[f"believed_cf_{f}" for f in _FEATURES]].to_numpy()
    realized = sub[[f"realized_cf_{f}" for f in _FEATURES]].to_numpy()
    return sub, believed, realized


# --------------------------------------------------------------------------- #
# 1 — contract compliance (structural, not behavioural)
# --------------------------------------------------------------------------- #


def test_l1_oracle_matches_shared_predict_contract():
    """L1-oracle exposes the SAME predict / predict_batch signatures as L0 and L2."""
    ref_predict = inspect.signature(L2TrueSCMModel.predict)
    ref_batch = inspect.signature(L2TrueSCMModel.predict_batch)
    for cls in (L0AssociationalModel, L1OracleEstimatedSCMModel):
        assert inspect.signature(cls.predict) == ref_predict
        assert inspect.signature(cls.predict_batch) == ref_batch
    # L1-oracle joins L2's SCM-dispatch hierarchy (same base, same code path), so
    # the L2 vs. L1-oracle contrast is attributable to the equations alone.
    assert L1OracleEstimatedSCMModel.__mro__[1] is L2TrueSCMModel.__mro__[1]


def test_l1_condition_holds_only_the_estimated_scm(cfg):
    """The condition object is constructed from (and holds) the ESTIMATED SCM only."""
    run = _run(cfg, "additive")
    est_scm = build_estimated_scm(make_linear_triangle("additive"), run.fitted)
    model = L1OracleEstimatedSCMModel(est_scm, list(_FEATURES), list(_FEATURES))
    assert model.name == "L1-oracle"
    assert model.scm.name.startswith("estimated[")


# --------------------------------------------------------------------------- #
# 2 — believed_cf == realized_cf at L0 and L2 (the believed_cf within-experiment check)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_believed_cf_equals_realized_cf_at_l0_and_l2(cfg, regime):
    """Schema-level analogue of the Gap_L2 ≈ 0 calibration anchor for H2 (H2).

    A failure means the scoring pipeline is not routing believed CFs through the
    condition-held SCM correctly. At L2 believed == realized bitwise (same model
    object on both sides). At L0 the equality holds on FULL-acted rows; grid-edge
    rows with a partial acted set are the documented exception (zero delta = NOT
    intervened, recourse-harness, so the free axis propagates under the
    true SCM while the believed J = I point holds it fixed) — divergence must be
    CONFINED to that exception set (the believed_cf amendment).
    """
    table = _run(cfg, regime).table

    _, believed_l2, realized_l2 = _cf_matrices(table, "L2")
    np.testing.assert_array_equal(believed_l2, realized_l2)

    sub, believed_l0, realized_l0 = _cf_matrices(table, "L0")
    full_acted = ((sub["delta_X1"] != 0.0) & (sub["delta_X2"] != 0.0)).to_numpy()
    np.testing.assert_array_equal(believed_l0[full_acted], realized_l0[full_acted])
    diverged = np.any(believed_l0 != realized_l0, axis=1)
    assert not np.any(diverged & full_acted), (
        "L0 believed_cf deviates from realized_cf on a FULL-acted row — the scoring "
        "layer is not routing believed CFs through the condition-held model."
    )


# --------------------------------------------------------------------------- #
# 3 — believed_cf ≠ realized_cf exists at L1-oracle (divergence channel wired)
# --------------------------------------------------------------------------- #


def test_believed_cf_diverges_at_l1_oracle_effect_modifying(cfg):
    """The believed/realized storage-and-computation paths are wired independently.

    NO magnitude assertion (spec): at n=4000 the OLS divergence is O(sampling
    noise); the point is only that the channel EXISTS — some individual's chosen
    δ leaves a downstream variable free, where f̂_L1 ≠ f_true maps it elsewhere.
    """
    _, believed, realized = _cf_matrices(_run(cfg, "effect_modifying").table, "L1-oracle")
    assert np.any(believed != realized), (
        "believed_cf identical to realized_cf on every L1-oracle row — the believed "
        "path is not going through the estimated SCM."
    )


# --------------------------------------------------------------------------- #
# 4 — manifest carries the estimation provenance (the L1-oracle form-template-known semantics audit
# trail)
# --------------------------------------------------------------------------- #


def test_manifest_carries_estimation_provenance(cfg, tmp_path):
    """Structural check on the manifest file, assembled exactly as run_triangle does."""
    run = _run(cfg, "effect_modifying")
    path = write_run_manifest(
        tmp_path / "run", cfg, estimation={run.regime: estimation_provenance(run.fitted)}
    )
    record = json.loads(path.read_text())["estimation"]["effect_modifying"]

    assert "LinearRegression" in record["metadata"]["estimator"]
    assert record["metadata"]["regime"] == "effect_modifying"
    assert record["metadata"]["family"] == "linear"
    assert record["form_spec"]["X2"]["interactions"] == [["A", "X1"]]
    assert set(record["coefficients"]["X2"]) == {"intercept", "A", "X1", "A*X1"}
    # The L1-oracle form-template-known semantics's sample-sharing constraint, auditable: the
    # recorded estimation-sample
    # identity IS the classifier's training sample (no redraw, no split) — the
    # same identity the L1-discovered PC step must later be asserted against.
    expected = json.loads(json.dumps(dataset_identity(run.classifier.dataset)))
    assert record["metadata"]["dataset"] == expected


# --------------------------------------------------------------------------- #
# 5 — Regression ("did we accidentally shift semantics" gate)
# --------------------------------------------------------------------------- #

# sha256 of the canonical serialization (to_csv, index=False) of each (regime,
# condition) slice restricted to _PRE_EXISTING_COLS, computed from the pilot
# outputs (results/ is gitignored, so the local CSVs alone would not travel with
# the repo — these constants make the gate durable). Recorded after
# verifying fresh-run == pilot-CSV per-column equality to the last float bit.
_PRE_EXISTING_COL_DIGESTS = {
    ("additive", "L0"): "ca1315463ef33a0ad73a63fae2389af038c9d1675c603158c9befcf199bf5509",
    ("additive", "L2"): "cfe456533ffe65f7e4eebd0e55ecb25613dcaf61ab8cbb7b84d1c0f3b252e2fe",
    ("effect_modifying", "L0"): "87f6c532fd416541c188041f1c884f11808eac9f75271aa07db44cc05f1b7b27",
    ("effect_modifying", "L2"): "1fa65cccf1cf67018a2871dd914564dcc38b0162f1ba99fe720045d331755d5e",
}


def _canonical_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


@pytest.mark.parametrize("regime", _REGIMES)
def test_pre_existing_cols_regression_l0_l2(cfg, regime):
    """L0/L2 rows must reproduce the frozen pilot outputs on every pre-existing column.

    Column ORDER differs from the pilot (the believed_cf_* columns), so equality is
    asserted on the pre-existing column subset, exact including float bits: via the
    recorded checksums above, plus a full per-column frame comparison against the
    local pilot combined CSV when it exists. A mismatch means the schema extension
    shifted semantics somewhere.
    """
    table = _run(cfg, regime).table
    fresh_slices = {}
    for condition in ("L0", "L2"):
        fresh = (
            table[table["condition"] == condition][_PRE_EXISTING_COLS]
            .reset_index(drop=True)
        )
        fresh_slices[condition] = fresh
        assert _canonical_digest(fresh) == _PRE_EXISTING_COL_DIGESTS[(regime, condition)], (
            f"{regime}/{condition}: pre-existing columns no longer reproduce the "
            "pilot outputs — the schema extension shifted semantics."
        )

    baseline_path = Path(f"results/recourse_triangle/scoring_table_{regime}.csv")
    if not baseline_path.exists():  # results/ is gitignored; checksum gate above suffices
        return
    # float_precision="round_trip": the default C-parser mode can be 1 ulp off on
    # the last digit, which check_exact would misread as a semantics shift.
    baseline = pd.read_csv(baseline_path, float_precision="round_trip")
    baseline["acted_set"] = baseline["acted_set"].fillna("")
    for condition in ("L0", "L2"):
        base = (
            baseline[baseline["condition"] == condition][_PRE_EXISTING_COLS]
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            fresh_slices[condition], base, check_exact=True, check_dtype=False
        )


# --------------------------------------------------------------------------- #
# 6 — sampler-guard non-reachability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("regime", _REGIMES)
def test_sampler_guard_not_reached_by_pipeline(cfg, regime):
    """The sampler-guard audit classified the pipeline's ONLY .sample() call as on
    the TRUE SCM upstream
    (classifier/training.py — class (a)). This pins it: the full 3-condition run
    completes without a RuntimeError from the estimated SCM's guarded samplers
    (procedure.py and scoring.py never sample from a condition-held SCM), while
    the guard itself stays armed on the estimated SCM."""
    run = _run(cfg, regime)  # would have raised RuntimeError if the guard were hit
    assert (run.table["condition"] == "L1-oracle").any()
    est_scm = build_estimated_scm(make_linear_triangle(regime), run.fitted)
    with pytest.raises(RuntimeError, match="mean function only"):
        est_scm.sample(5, seed=0)
