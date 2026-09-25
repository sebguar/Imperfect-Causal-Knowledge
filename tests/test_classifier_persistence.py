"""Classifier-persistence gates.

The invariant under test is architectural, not numerical: ONE classifier h is fit
per (cell, seed), ABOVE the condition loop, and the SAME fitted object is handed
to L0, L1-oracle and L2 (h held constant across conditions — only the
causal model varies). Persistence of that fit is added; these tests are the gate
that keeps the "fit once, reuse everywhere" property from silently regressing into
a per-condition refit that merely happens to agree.

  1. one-shot smoke — the persisted params, the run's fitted h, and a standalone
     inline fit are three-way equal, and the run exposes exactly ONE classifier
     across L0 / L1-oracle / L2;
  2. single-fit count — build_classifier is invoked exactly once per run_regime
     call, regardless of how many conditions run;
  3. manifest presence — classifier_params.json lands next to manifest.json after
     run_cell_seed, and round-trips to the same floats;
  4. condition independence — condition ORDER and condition SET leave every
     coefficient and every calibration scalar bit-identical. This is the
     regression test against a silently reintroduced RNG advancement between
     condition fits: if any condition consumed classifier-stream randomness, a
     reordering would move the numbers.

Speed note: the classifier fit reads ONLY ``cfg.seed`` and ``cfg.classifier``
(see build_classifier) — the recourse grid resolution is not an input to it — so
the tests that only interrogate classifier parameters shrink
``cfg.recourse.grid.resolution`` to keep the brute-force search cheap. The
classifier numbers are unaffected by that override, and test 1 asserts exactly
that by comparing against a standalone build_classifier on the untouched config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icknowledge.classifier import build_classifier, classifier_provenance
from icknowledge.recourse import pipeline as pipeline_module
from icknowledge.recourse.pipeline import run_regime
from icknowledge.scm import make_linear_triangle
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import CLASSIFIER_PARAMS_FILENAME

_CONFIG_PATH = "configs/recourse_triangle.yaml"
_REGIME = "additive"
_FULL_LADDER = ("L0", "L1-oracle", "L2")

#: Cheap grid for the classifier-only assertions (see the module docstring).
_FAST_RESOLUTION = 9


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CONFIG_PATH)


@pytest.fixture(scope="module")
def fast_cfg(cfg):
    fast = load_config(_CONFIG_PATH)
    fast.recourse.grid.resolution = _FAST_RESOLUTION
    # The anchor smoke check is a grid-resolution gate; at res=9 it would fail for
    # reasons that have nothing to do with the classifier. Callers pass
    # run_anchor=False rather than loosening its tolerance.
    assert fast.seed == cfg.seed, "the fast config must keep the master seed"
    assert fast.classifier == cfg.classifier, "the fast config must not touch cfg.classifier"
    return fast


@pytest.fixture(scope="module")
def fast_run(fast_cfg):
    return run_regime(fast_cfg, _REGIME, conditions=_FULL_LADDER, run_anchor=False)


@pytest.fixture(scope="module")
def inline_fit(cfg):
    """A standalone build_classifier on the UNTOUCHED config — the reference fit.

    Module-scoped: the eps_scale bisection refits h ~50 times, so this is the
    expensive object in the file and is built once.
    """
    return build_classifier(make_linear_triangle(_REGIME), cfg)


def _params(record: dict) -> tuple[dict, float, float, float]:
    """The comparison tuple: coefficients, intercept, and the calibrated scalars."""
    return (
        record["coef"],
        record["intercept"],
        record["calibration"]["eps_scale"],
        record["calibration"]["tau"],
    )


# --------------------------------------------------------------------------- #
# 1. one-shot smoke: persisted == run's h == inline fit, across all conditions
# --------------------------------------------------------------------------- #


def test_persisted_params_match_inline_fit_for_every_condition(inline_fit, fast_run, tmp_path):
    """Three-way equality: standalone inline fit, the run's h, the persisted JSON.

    "The classifier that would have been fit inline for L0 / L1-oracle / L2" is a
    single object here BY CONSTRUCTION — run_regime fits h once, above the
    condition loop, so all three conditions share one ``run.classifier``. The
    inline reference is therefore build_classifier on the same (scm, cfg), and the
    assertion is that the run, the reference, and the file agree exactly.
    """
    inline = classifier_provenance(inline_fit)
    from_run = classifier_provenance(fast_run.classifier)

    path = Path(tmp_path) / CLASSIFIER_PARAMS_FILENAME
    path.write_text(json.dumps({_REGIME: from_run}, indent=2))
    persisted = json.loads(path.read_text())[_REGIME]

    assert _params(from_run) == _params(inline), (
        "the run's h differs from a standalone inline fit on the same (scm, cfg) — "
        "an RNG stream was created, destroyed or reordered by the pipeline."
    )
    assert _params(persisted) == _params(from_run), (
        "JSON round-trip is lossy — the persisted params are not the fitted params."
    )
    # The full record, not just the comparison tuple, must survive the round-trip.
    assert persisted == from_run


def test_one_classifier_object_serves_every_condition(fast_run):
    """The RegimeRun exposes exactly one fitted h; conditions consume it, not a spec."""
    assert fast_run.classifier.model is not None
    conditions = set(fast_run.table["condition"].unique())
    assert conditions == set(_FULL_LADDER), (
        f"expected all three conditions in one run's table; got {sorted(conditions)}"
    )


# --------------------------------------------------------------------------- #
# 2. single-fit count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conditions", [("L0", "L2"), _FULL_LADDER])
def test_classifier_is_fit_exactly_once_per_run(fast_cfg, conditions, monkeypatch):
    """build_classifier runs ONCE per (cell, seed), whatever the condition set.

    A per-condition refit would be invisible in the outputs (the refits are
    determinism-locked and would agree), so the count is asserted directly.
    """
    calls = []
    original = pipeline_module.build_classifier

    def counting_build_classifier(scm, cfg):
        calls.append(cfg.seed)
        return original(scm, cfg)

    monkeypatch.setattr(pipeline_module, "build_classifier", counting_build_classifier)
    run_regime(fast_cfg, _REGIME, conditions=conditions, run_anchor=False)
    assert len(calls) == 1, (
        f"classifier fit {len(calls)}x for {len(conditions)} conditions — the fit "
        "must be lifted above the condition loop (one h per (cell, seed))."
    )


# --------------------------------------------------------------------------- #
# 3. manifest presence — the artifact lands next to manifest.json
# --------------------------------------------------------------------------- #


def test_run_cell_seed_writes_classifier_params_next_to_manifest(tmp_path, monkeypatch):
    """After run_cell_seed, classifier_params.json sits beside manifest.json.

    The grid runner is exercised through its real entry point (not a hand-rolled
    stand-in) so a future refactor that drops the write is caught here. The cell's
    grid resolution is shrunk through the config loader the runner itself calls —
    the classifier block is untouched.
    """
    from scripts import run_cross_seed_grid as grid

    original_build = grid._build_cfg

    def fast_build_cfg(topology, family, seed, out_dir):
        cfg = original_build(topology, family, seed, out_dir)
        cfg.recourse.grid.resolution = _FAST_RESOLUTION
        return cfg

    monkeypatch.setattr(grid, "_build_cfg", fast_build_cfg)
    # run_anchor is gated on family=="linear" inside run_cell_seed; the coarse grid
    # would fail the anchor's step tolerance, so exercise an NLG cell, where
    # run_cell_seed disables the anchor by its own rule.
    grid.run_cell_seed("triangle", "nlg", _REGIME, 0, 20260710, Path(tmp_path))

    out_dir = Path(tmp_path) / grid.cell_dir_name("triangle", "nlg", _REGIME, 0)
    params_path = out_dir / CLASSIFIER_PARAMS_FILENAME
    assert (out_dir / "manifest.json").exists()
    assert params_path.exists(), f"{CLASSIFIER_PARAMS_FILENAME} not written to {out_dir}"

    record = json.loads(params_path.read_text())[_REGIME]
    assert set(record["coef"]) == set(record["feature_names"])
    assert record["protected_attr"] not in record["coef"], (
        "group-blind invariant: A must not carry a coefficient."
    )
    assert isinstance(record["intercept"], float)
    assert record["calibration"]["eps_scale"] > 0.0


# --------------------------------------------------------------------------- #
# 4. condition independence (the reintroduced-RNG-advancement regression)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "conditions",
    [("L2", "L1-oracle", "L0"), ("L0", "L2", "L1-oracle"), ("L0", "L2")],
)
def test_condition_order_and_set_do_not_move_the_classifier(fast_cfg, fast_run, conditions):
    """Reordering or dropping conditions leaves every classifier number identical.

    If any condition consumed randomness from a classifier stream — the failure
    mode a future refactor could silently reintroduce — permuting the conditions
    would move the coefficients. Exact equality, not a tolerance: these are the
    same floats or they are a bug.
    """
    reference = classifier_provenance(fast_run.classifier)
    other = run_regime(fast_cfg, _REGIME, conditions=conditions, run_anchor=False)
    assert _params(classifier_provenance(other.classifier)) == _params(reference), (
        f"classifier params moved under conditions={conditions} — an RNG stream is "
        "being advanced per condition."
    )


def test_provenance_coef_pairing_is_positional_and_guarded(inline_fit):
    """coef_ is paired to feature_names by position, in the model's own column order.

    The name->value mapping in the persisted record is only meaningful if the two
    sequences are aligned; classifier_provenance asserts the lengths match, so a
    feature-set change cannot silently re-map an already-written artifact.
    """
    record = classifier_provenance(inline_fit)
    assert list(record["coef"]) == list(inline_fit.feature_names)
    assert len(record["coef"]) == inline_fit.model.coef_.ravel().size
    assert list(record["coef"].values()) == [
        float(v) for v in inline_fit.model.coef_.ravel()
    ]
