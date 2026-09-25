"""L1-discovered assembly: zero-term fits, parentless marginals, guards, drift guard.

Graphs are HAND-DRAWN — no PC runs here. Production SCM builders are
used for DATA ONLY, so the numbers are realistic without any of this being
discoverer calibration.

Covers:
  1. the Step-2 zero-term tolerance at the shared-primitive level (an
     intercept-only node fits, and its reconstructed equation propagates);
  2. the PS-7-note-(b) parentless marginal: abduction round-trip exactness, and
     inertness under an intervention elsewhere in the graph;
  3. the discovered-flavoured guard and the assembly's other refusals;
  4. the PS-7-note-(c) scoped drift guard: on the TRUE graph, additive regime,
     both families, both topological shapes — the discovered path's X-node
     coefficient dicts equal `build_estimated_scm`'s, fitted on identical data.
     A is excluded (the one intended difference, per note b) and EM is excluded
     (interaction-blindness by PS-7 is intended, not drift).
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from icknowledge.discovery import synthesize_form_spec
from icknowledge.estimation import (
    LinearOLSEstimator,
    NLGOLSEstimator,
    NodeForm,
    build_estimated_scm,
)
from icknowledge.estimation.discovered import build_estimated_scm_from_discovery
from icknowledge.estimation.forms import term_names
from icknowledge.scm import (
    chain_form_spec,
    collider_form_spec,
    make_linear_chain,
    make_linear_collider,
    make_linear_triangle,
    make_nonlinear_gaussian_chain,
    make_nonlinear_gaussian_collider,
    make_nonlinear_gaussian_triangle,
    nlg_chain_form_spec,
    nlg_collider_form_spec,
    nlg_triangle_form_spec,
    triangle_form_spec,
)

N = 1500
SEED = 20260808

#: (topology, family) -> (builder, ORACLE form-spec fn, estimator class). The
#: oracle template and estimator are the frozen pair `pipeline._ESTIMATORS` /
#: the drivers' `_FAMILIES` select; the discovered path reuses the estimator and
#: swaps only the template.
CELLS = {
    ("triangle", "linear"): (make_linear_triangle, triangle_form_spec, LinearOLSEstimator),
    ("triangle", "nlg"): (
        make_nonlinear_gaussian_triangle,
        nlg_triangle_form_spec,
        NLGOLSEstimator,
    ),
    ("chain", "linear"): (make_linear_chain, chain_form_spec, LinearOLSEstimator),
    ("chain", "nlg"): (make_nonlinear_gaussian_chain, nlg_chain_form_spec, NLGOLSEstimator),
    ("collider", "linear"): (make_linear_collider, collider_form_spec, LinearOLSEstimator),
    ("collider", "nlg"): (
        make_nonlinear_gaussian_collider,
        nlg_collider_form_spec,
        NLGOLSEstimator,
    ),
}


@pytest.fixture(scope="module")
def chain_data() -> pd.DataFrame:
    """Linear-chain sample (A, X1, X2, X3) — data only; the graphs below are drawn by hand."""
    return make_linear_chain("additive").sample(N, seed=SEED)


def _graph(edges, nodes) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def _orphan_chain_graph() -> nx.DiGraph:
    """A and X1 parentless; A->X2<-X1; X2->X3. X1 is a DISCOVERED root (search miss)."""
    return _graph(
        [("A", "X2"), ("X1", "X2"), ("X2", "X3")], ["A", "X1", "X2", "X3"]
    )


def _build(graph: nx.DiGraph, data: pd.DataFrame, family: str = "linear"):
    """synthesize -> fit (the L1-oracle form-template-known semantics estimator) -> assemble, the "
    "whole discovered path."""
    form = synthesize_form_spec(graph, family)
    estimator = LinearOLSEstimator if family == "linear" else NLGOLSEstimator
    fitted = estimator().fit(data, form, extra_metadata={"regime": "additive"})
    return build_estimated_scm_from_discovery(graph, fitted, data), fitted


# -- 1. Step 2: zero-term tolerance in the shared primitive ------------------


def test_zero_term_form_fits_to_the_sample_mean(chain_data):
    """An intercept-only node fits: no design matrix, coefficients = {intercept: ȳ}.

    The unregularized-OLS solution with an intercept and no regressors IS the
    sample mean, so this is the form-template-known estimator, not a second one.
    """
    form = {"X1": NodeForm(node="X1", parents=())}
    assert term_names(form["X1"]) == []

    fitted = LinearOLSEstimator().fit(chain_data, form)

    assert set(fitted.coefficients["X1"]) == {"intercept"}
    assert fitted.coefficients["X1"]["intercept"] == pytest.approx(
        float(chain_data["X1"].mean()), rel=0, abs=1e-12
    )


def test_zero_term_equation_propagates_the_intercept(chain_data):
    """The reconstructed equation broadcasts the intercept and stays additive in u."""
    fitted = LinearOLSEstimator().fit(chain_data, {"X1": NodeForm(node="X1", parents=())})
    intercept = fitted.coefficients["X1"]["intercept"]
    equation = fitted.equations["X1"]

    noise = np.array([0.0, 1.5, -2.25, 7.0])
    np.testing.assert_allclose(equation({}, np.zeros(4)), np.full(4, intercept))
    np.testing.assert_allclose(equation({}, noise), intercept + noise)


# -- 2. PS-7 note (b): the parentless marginal -------------------------------


def test_parentless_intercepts_are_the_sample_means(chain_data):
    """A and the orphaned X1 both get ȳ — A is NOT special-cased (uniformity)."""
    scm, _ = _build(_orphan_chain_graph(), chain_data)

    assert set(scm.parentless_intercepts) == {"A", "X1"}
    for node, intercept in scm.parentless_intercepts.items():
        assert intercept == pytest.approx(float(chain_data[node].mean()), rel=0, abs=1e-12)


def test_parentless_nodes_return_their_factual_value_after_abduction(chain_data):
    """abduct -> counterfactual(identity) reproduces the factual row exactly enough.

    u = x − ȳ is recovered, and prediction returns ȳ + u = x, so the marginal is
    behaviourally INERT whatever ȳ happens to be.
    """
    scm, _ = _build(_orphan_chain_graph(), chain_data)
    factual = chain_data.head(50)

    recovered = scm.abduct(factual)
    np.testing.assert_allclose(
        recovered["A"].to_numpy(),
        factual["A"].to_numpy() - scm.parentless_intercepts["A"],
        atol=1e-12,
    )
    cf = scm.counterfactual(factual, {})
    for node in scm.nodes:
        np.testing.assert_allclose(
            cf[node].to_numpy(), factual[node].to_numpy(), atol=1e-10, rtol=0
        )


def test_parentless_node_stays_factual_under_an_unrelated_intervention(chain_data):
    """Intervening on X1 moves its descendants; A and the acted-on node's non-descendants don't.

    A is parentless AND not a descendant of X1 in the discovered graph, so it must
    come back at its factual value — the marginal contributes no drift.
    """
    scm, _ = _build(_orphan_chain_graph(), chain_data)
    factual = chain_data.head(50)

    cf = scm.counterfactual(factual, {"X1": factual["X1"].to_numpy() + 3.0})

    np.testing.assert_allclose(
        cf["A"].to_numpy(), factual["A"].to_numpy(), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(cf["X1"].to_numpy(), factual["X1"].to_numpy() + 3.0)
    assert not np.allclose(cf["X2"].to_numpy(), factual["X2"].to_numpy())
    assert not np.allclose(cf["X3"].to_numpy(), factual["X3"].to_numpy())


def test_estimated_discovered_scm_refuses_to_be_sampled(chain_data):
    """The mean function only — no σ̂², so ancestral sampling is undefined."""
    scm, _ = _build(_orphan_chain_graph(), chain_data)
    with pytest.raises(RuntimeError, match="no noise distribution"):
        scm.sample(10, seed=1)


def test_builder_signature_is_truth_free():
    """[PS-7 note (b)] The discovered builder cannot reach a true SCM: no such parameter."""
    import inspect

    params = inspect.signature(build_estimated_scm_from_discovery).parameters
    assert list(params) == ["discovered_graph", "fitted", "data"]
    assert not any("true" in name for name in params)


# -- 3. guards ---------------------------------------------------------------


def test_discovered_guard_raises_on_a_parent_set_mismatch(chain_data):
    """The form must be synthesized from the graph it is assembled against."""
    fitted_for = _graph([("A", "X2"), ("X1", "X2"), ("X2", "X3")], ["A", "X1", "X2", "X3"])
    assembled_against = _graph(
        [("A", "X2"), ("X2", "X3")], ["A", "X1", "X2", "X3"]
    )  # X1 -> X2 dropped
    fitted = LinearOLSEstimator().fit(chain_data, synthesize_form_spec(fitted_for, "linear"))

    with pytest.raises(ValueError, match="DISCOVERED graph's parent set"):
        build_estimated_scm_from_discovery(assembled_against, fitted, chain_data)


def test_assembly_raises_on_a_cyclic_graph(chain_data):
    fitted = LinearOLSEstimator().fit(
        chain_data, synthesize_form_spec(_orphan_chain_graph(), "linear")
    )
    cyclic = _graph(
        [("A", "X2"), ("X1", "X2"), ("X2", "X3"), ("X3", "X1")], ["A", "X1", "X2", "X3"]
    )
    with pytest.raises(ValueError, match="not a DAG"):
        build_estimated_scm_from_discovery(cyclic, fitted, chain_data)


def test_assembly_requires_the_estimation_sample_for_parentless_nodes(chain_data):
    graph = _orphan_chain_graph()
    fitted = LinearOLSEstimator().fit(chain_data, synthesize_form_spec(graph, "linear"))
    with pytest.raises(ValueError, match="`data` .*is required"):
        build_estimated_scm_from_discovery(graph, fitted)


def test_assembly_rejects_a_form_covering_a_node_outside_the_graph(chain_data):
    graph = _orphan_chain_graph()
    fitted = LinearOLSEstimator().fit(chain_data, synthesize_form_spec(graph, "linear"))
    smaller = _graph([("A", "X2"), ("X1", "X2")], ["A", "X1", "X2"])
    with pytest.raises(ValueError, match="not in the discovered graph"):
        build_estimated_scm_from_discovery(smaller, fitted, chain_data)


def test_assembly_rejects_a_supplied_form_for_a_parentless_node(chain_data):
    """A parentless node's mechanism is built here (note b), never handed in."""
    graph = _orphan_chain_graph()
    form = dict(synthesize_form_spec(graph, "linear"))
    form["X1"] = NodeForm(node="X1", parents=())  # zero-term form for a discovered root
    fitted = LinearOLSEstimator().fit(chain_data, form)

    with pytest.raises(ValueError, match="carry a form spec"):
        build_estimated_scm_from_discovery(graph, fitted, chain_data)


# -- 4. PS-7 note (c): scoped drift guard ------------------------------------


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_discovered_path_on_the_true_graph_matches_the_oracle_on_x_nodes(cell):
    """[PS-7 note (c)] Hand the discovered path the TRUE graph and it must reproduce
    `build_estimated_scm`'s X-node coefficients, additive regime, both families.

    SCOPE, per the note: endogenous NON-ROOT equations only. A is excluded — the
    oracle copies A's true root mechanism verbatim while the discovered builder
    fits the note-(b) marginal, which is the one intended difference. The
    effect-modifying arm is excluded too: there the discovered form is
    interaction-free by PS-7 while the oracle carries the A·X column, which is
    intended divergence, not drift.

    Doubles as the confirmation of note (a): if the synthesized per-parent basis
    were not the oracle's basis, these coefficients could not coincide.
    """
    builder, oracle_form_fn, estimator = CELLS[cell]
    _, family = cell
    scm = builder("additive")
    data = scm.sample(N, seed=SEED)

    oracle_form = oracle_form_fn("additive")
    discovered_form = synthesize_form_spec(scm.graph, family)
    oracle = estimator().fit(data, oracle_form)
    discovered = estimator().fit(data, discovered_form)

    x_nodes = [node for node in scm.nodes if scm.parents(node)]
    assert x_nodes, "every production cell has at least one endogenous non-root"
    for node in x_nodes:
        assert set(oracle.coefficients[node]) == set(discovered.coefficients[node]), (
            f"{node}: the synthesized basis produced a different COLUMN SET than the "
            "oracle template — that is a form-preserving estimation (PS-7 note a) violation."
        )
        if term_names(oracle_form[node]) == term_names(discovered_form[node]):
            # Same columns in the same order => the same lstsq call => bitwise equal.
            assert oracle.coefficients[node] == discovered.coefficients[node]
        else:
            # Same columns, PERMUTED (the synthesized order is sorted-by-parent-name,
            # some oracle NLG templates are not). LAPACK is not permutation-exact, so
            # the residual is float round-off, ~1e-15 — orders of magnitude below any
            # coefficient difference a basis change would produce.
            for term, value in oracle.coefficients[node].items():
                assert value == pytest.approx(
                    discovered.coefficients[node][term], rel=1e-11, abs=1e-11
                )


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_discovered_and_oracle_scms_agree_on_counterfactuals_on_the_true_graph(cell):
    """The assembly-level tether behind the coefficient guard: same graph, same
    numbers, so the two estimated SCMs must answer counterfactual queries alike.

    Includes A, which the two build differently (true root mechanism vs. note-(b)
    marginal) — non-intervened, both return the factual value, so agreement here
    is the positive statement that the (b) mechanism is behaviourally inert.
    """
    builder, oracle_form_fn, estimator = CELLS[cell]
    _, family = cell
    scm = builder("additive")
    data = scm.sample(N, seed=SEED)

    oracle_scm = build_estimated_scm(scm, estimator().fit(data, oracle_form_fn("additive")))
    discovered_scm = build_estimated_scm_from_discovery(
        scm.graph,
        estimator().fit(data, synthesize_form_spec(scm.graph, family)),
        data,
    )

    factual = data.head(50)
    action = {"X1": factual["X1"].to_numpy() + 1.0}
    oracle_cf = oracle_scm.counterfactual(factual, action)
    discovered_cf = discovered_scm.counterfactual(factual, action)

    for node in scm.nodes:
        np.testing.assert_allclose(
            discovered_cf[node].to_numpy(), oracle_cf[node].to_numpy(), rtol=1e-9, atol=1e-9
        )
