"""`L1DiscoveredEstimatedSCMModel`: the condition class behind the shared predict contract.

Class-level assertions of the contract and the truth-freedom rule, plus the two
cheap OPT-IN invariants the wiring must preserve: the default ladder is
still three rungs, and a default run carries no discovery provenance.

The condition is REGISTERED in `recourse/pipeline.py` (PS-7,
PS-4) but only when a caller names it in ``conditions`` — see
tests/test_l1_discovered_wiring.py for the byte-identity regression that proves
opting out costs nothing.
"""

import json
import warnings

import networkx as nx
import numpy as np
import pytest

from icknowledge.discovery import synthesize_form_spec
from icknowledge.estimation import LinearOLSEstimator, estimation_provenance
from icknowledge.estimation.discovered import build_estimated_scm_from_discovery
from icknowledge.recourse import pipeline as recourse_pipeline
from icknowledge.recourse.model_conditions import (
    L1DiscoveredEstimatedSCMModel,
    L1OracleEstimatedSCMModel,
    L2TrueSCMModel,
    _SCMDispatchModel,
)
from icknowledge.scm import make_linear_triangle
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest

N = 800
SEED = 20260808


@pytest.fixture(scope="module")
def discovered_model():
    """L1-discovered on a WRONG graph (X1->X2 reversed) — a plausible tie-break error."""
    scm = make_linear_triangle("additive")
    data = scm.sample(N, seed=SEED)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "X1", "X2"])
    graph.add_edges_from([("A", "X1"), ("A", "X2"), ("X2", "X1")])
    fitted = LinearOLSEstimator().fit(data, synthesize_form_spec(graph, "linear"))
    estimated = build_estimated_scm_from_discovery(graph, fitted, data)
    model = L1DiscoveredEstimatedSCMModel(
        estimated, feature_names=["X1", "X2"], acted=["X1", "X2"]
    )
    return model, scm, data


def test_name_and_dispatch_shape(discovered_model):
    """Same dispatch as L1-oracle / L2 — that is what makes the gap attributable."""
    model, _, _ = discovered_model
    assert model.name == "L1-discovered"
    assert issubclass(L1DiscoveredEstimatedSCMModel, _SCMDispatchModel)
    assert L1DiscoveredEstimatedSCMModel.predict is _SCMDispatchModel.predict
    assert L1DiscoveredEstimatedSCMModel.predict_batch is _SCMDispatchModel.predict_batch


def test_predict_contract(discovered_model):
    """predict -> (F,) aligned to feature_names; predict_batch -> (K, F)."""
    model, _, data = discovered_model
    factual = data.iloc[0]

    single = model.predict(factual, {"X1": 1.0, "X2": 0.0})
    batch = model.predict_batch(factual, np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]]))

    assert single.shape == (2,)
    assert batch.shape == (3, 2)
    np.testing.assert_allclose(batch[0], single)
    # A zero delta on every axis is the identity counterfactual.
    np.testing.assert_allclose(batch[1], factual[["X1", "X2"]].to_numpy())


def test_the_condition_holds_only_the_discovered_scm(discovered_model):
    """[mirrors the L1-oracle rule] The condition must not hold or expose the truth.

    Its SCM carries the DISCOVERED edges, not the true ones; anything needing the
    truth lives at the scoring layer (the realizer), never inside the condition.
    """
    model, true_scm, _ = discovered_model

    assert model.scm.parents("X1") == ["A", "X2"]  # discovered (reversed)
    assert true_scm.parents("X1") == ["A"]  # truth
    held = {id(value) for value in vars(model).values()}
    assert id(true_scm) not in held
    assert not any(getattr(value, "name", None) == true_scm.name for value in vars(model).values())


def test_wrong_graph_actually_changes_the_believed_prediction(discovered_model):
    """Sanity: the reversed edge is load-bearing, so the rung is not a relabelled L1-oracle."""
    model, true_scm, data = discovered_model
    factual = data.iloc[0]
    l2 = L2TrueSCMModel(true_scm, feature_names=["X1", "X2"], acted=["X1", "X2"])

    delta = {"X1": 1.0, "X2": 0.0}
    assert not np.allclose(model.predict(factual, delta), l2.predict(factual, delta))


def test_the_rung_is_opt_in_not_default():
    """[PS-4] The rung is WIRED but LAZY — the default ladder stays at three.

    A caller who does not ask for the rung pays nothing for it and gets a
    bit-identical run. The byte-identity half lives in
    tests/test_l1_discovered_wiring.py; this test pins the two cheap invariants:
    the default ladder, and no discovery provenance off the rung.
    """
    assert recourse_pipeline._DEFAULT_CONDITIONS == ("L0", "L1-oracle", "L2")
    assert L1OracleEstimatedSCMModel.name == "L1-oracle"  # untouched


def test_default_run_produces_no_discovery_provenance(tmp_path):
    """A default-conditions run leaves both PS-4 fields None and writes no block."""
    cfg = load_config("configs/recourse_triangle.yaml")
    cfg.classifier.N = 500
    cfg.classifier.min_neg_per_group = 30
    cfg.recourse.grid.resolution = 9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = recourse_pipeline.run_regime(cfg, "additive", run_anchor=False)

    assert set(run.table["condition"].unique()) == {"L0", "L1-oracle", "L2"}
    # The search never ran, so there is nothing to record and nothing was fitted
    # against a discovered graph.
    assert run.discovery is None
    assert run.fitted_discovered is None

    # ... and the manifest therefore carries no `discovery` key at all. The
    # absence IS the signal: PS-4 makes the block's presence the marker that
    # distinguishes the four-rung tree from the sealed three-rung one, so an
    # empty-dict block would be as wrong as a populated one.
    manifest_path = write_run_manifest(
        tmp_path,
        cfg,
        estimation={"additive": estimation_provenance(run.fitted)},
        discovery=None,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "discovery" not in manifest
    assert "estimation" in manifest  # the sibling block still writes normally
