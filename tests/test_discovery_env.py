"""Environment guard for the causal-discovery dependency.

PURPOSE IS ENVIRONMENT VERIFICATION, NOT CAUSAL-DISCOVERY VALIDATION. These
tests assert that the pinned causal-learn build imports, that the PC call-site
signature named in PS-7 still accepts every argument we pass explicitly,
and that the BackgroundKnowledge surface discovery depends on ("A is a root")
still exists. They say nothing about whether PC recovers the right graph —
discovery quality is the synthesis layer's concern and is tested there.

The toy data below is HAND-MADE and lives deliberately outside the icknowledge
SCM builders. The method-switch authority (amended) forbids calibrating discovery on production
SCMs;
keeping the fixture local to this test file makes that boundary explicit and
prevents this environment guard from ever becoming a discovery benchmark.

Per PS-7, PC-Stable and Fisher-z are the library defaults, but both are
passed explicitly so the manifest records them rather than inheriting them.
"""

from __future__ import annotations

import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

TOY_NODE_NAMES = ["X1", "X2", "X3"]
TOY_N = 500
TOY_SEED = 20250808


def _toy_chain_data() -> np.ndarray:
    """Hand-drawn linear chain X1 -> X2 -> X3, ~500 samples.

    Not an icknowledge SCM. Written out inline so this file has no dependency
    on the production generators (see module docstring).
    """
    rng = np.random.default_rng(TOY_SEED)
    x1 = rng.normal(0.0, 1.0, TOY_N)
    x2 = 2.0 * x1 + rng.normal(0.0, 0.5, TOY_N)
    x3 = 1.5 * x2 + rng.normal(0.0, 0.5, TOY_N)
    return np.column_stack([x1, x2, x3])


def test_pc_stable_fisherz_call_site_runs_and_returns_a_cpdag():
    """The PS-7 call site: every argument passed explicitly."""
    data = _toy_chain_data()

    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=fisherz,
        stable=True,
        background_knowledge=None,
        node_names=TOY_NODE_NAMES,
    )

    assert cg is not None
    adj = cg.G.graph
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (len(TOY_NODE_NAMES), len(TOY_NODE_NAMES))
    assert [n.get_name() for n in cg.G.get_nodes()] == TOY_NODE_NAMES


def test_fisherz_is_the_identifier_pc_expects():
    """Guard against a rename of the Fisher-z test identifier."""
    assert fisherz == "fisherz"


def test_background_knowledge_surface_exists_and_is_accepted_by_pc():
    """The 'A is a root' mechanism discovery needs, smoke-tested end to end."""
    x1, x2, x3 = (GraphNode(name) for name in TOY_NODE_NAMES)

    bk = BackgroundKnowledge()
    bk.add_forbidden_by_node(x2, x1)
    bk.add_forbidden_by_node(x3, x1)
    bk.add_node_to_tier(x1, 0)
    bk.add_node_to_tier(x2, 1)
    bk.add_node_to_tier(x3, 2)

    assert bk.is_forbidden(x2, x1)
    assert not bk.is_forbidden(x1, x2)
    assert bk.is_in_which_tier(x1) == 0

    # Node identity is by name (GraphNode.__eq__), so knowledge built from
    # freshly constructed nodes matches the nodes PC creates from node_names.
    cg = pc(
        data=_toy_chain_data(),
        alpha=0.05,
        indep_test=fisherz,
        stable=True,
        background_knowledge=bk,
        node_names=TOY_NODE_NAMES,
    )

    assert cg is not None
    assert cg.G.graph.shape == (len(TOY_NODE_NAMES), len(TOY_NODE_NAMES))
