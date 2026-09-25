"""SCM base class: graph + structural equations + ancestral sampling.

The ancestral-sampling engine is identical for every experimental cell; only the 
(graph, equations, noise) specification varies.
Therefore the engine lives in this single concrete, parametrized `SCM` class and
the concrete SCMs will be builder functions that construct and return
configured `SCM` instances. There is deliberately no abstract base class.

See the `scm-specification` skill for the locked construction conventions.

Noise is IMPLICIT: exogenous U variables are NOT graph nodes (Karimi's
three-tuple convention). The graph holds only endogenous variables.

RNG scheme: every node's exogenous noise is derived through the project seeding 
utility, never ad-hoc `np.random`. Both `sample` and `sample_noise` call 
`spawn_children(seed, n_nodes)` and map the resulting child generators to nodes 
in topological order, so a given seed reproduces the same per-node noise across 
both methods (abduction relies on this consistency).
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd

from icknowledge.utils.seeding import spawn_children

StructuralEquation = Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]
# receives {parent_name: parent_column of shape (n,)} and a noise array of
# shape (n,); returns a column of shape (n,)
NoiseSampler = Callable[[np.random.Generator, int], np.ndarray]
# receives (rng, n); returns exogenous noise draws of shape (n,)

_VALID_REGIMES = frozenset({"additive", "effect_modifying"})


class SCM:
    """A parametrized structural causal model over endogenous variables.

    A single concrete class serves every cell: graph, structural equations and
    noise samplers are supplied at construction and the shared ancestral-
    sampling engine draws data from them.

    Parameters
    ----------
    graph: DAG over endogenous variable names, edges cause -> effect (noise implicit).
    equations: node -> structural equation `f_j(parents, noise)`, vectorized over n.
    noise_samplers: node -> exogenous noise sampler `g_j(rng, n)`.
    regime: optional declared regime, {"additive", "effect_modifying"} (convention 2).
    is_anm: DECLARATION (default True) of additive-noise structure — not a
        verification; gates `abduct`/`counterfactual` (valid only for ANM), which
        refuse when False.
    name: optional human-readable label.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        equations: dict[str, StructuralEquation],
        noise_samplers: dict[str, NoiseSampler],
        regime: str | None = None,
        name: str | None = None,
        is_anm: bool = True,
    ) -> None:
        errors: list[str] = []

        if not nx.is_directed_acyclic_graph(graph):
            errors.append("graph must be a directed acyclic graph (DAG).")

        graph_nodes = set(graph.nodes())

        eq_missing = graph_nodes - set(equations)
        eq_extra = set(equations) - graph_nodes
        if eq_missing or eq_extra:
            errors.append(
                "equations keys must match graph nodes; "
                f"missing equations for {sorted(eq_missing)}, "
                f"extra equations for {sorted(eq_extra)}."
            )

        noise_missing = graph_nodes - set(noise_samplers)
        noise_extra = set(noise_samplers) - graph_nodes
        if noise_missing or noise_extra:
            errors.append(
                "noise_samplers keys must match graph nodes; "
                f"missing noise samplers for {sorted(noise_missing)}, "
                f"extra noise samplers for {sorted(noise_extra)}."
            )

        if regime is not None and regime not in _VALID_REGIMES:
            errors.append(
                f"regime must be one of {sorted(_VALID_REGIMES)} or None, got {regime!r}."
            )

        if errors:
            raise ValueError("Invalid SCM specification:\n  - " + "\n  - ".join(errors))

        self._graph = graph.copy()  # defensive copy
        self.equations = dict(equations)
        self.noise_samplers = dict(noise_samplers)
        self.regime = regime
        self.name = name
        self.is_anm = is_anm
        self._topo_order = list(nx.topological_sort(self._graph))

    # -- accessors -----------------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        """Variable names in topological order."""
        return list(self._topo_order)

    @property
    def graph(self) -> nx.DiGraph:
        """A defensive copy of the underlying DiGraph."""
        return self._graph.copy()

    def parents(self, node: str) -> list[str]:
        """Direct parents (causes) of ``node``."""
        return list(self._graph.predecessors(node))

    def children(self, node: str) -> list[str]:
        """Direct children (effects) of ``node``."""
        return list(self._graph.successors(node))

    def __repr__(self) -> str:
        return (
            f"SCM(name={self.name!r}, n_nodes={len(self._topo_order)}, "
            f"regime={self.regime!r})"
        )

    # -- RNG / noise ---------------------------------------------------------

    def _child_rngs(self, seed: int) -> dict[str, np.random.Generator]:
        """Per-node RNGs, mapped to nodes in topological order.

        Node ``i`` (in topological order) uses ``spawn_children(seed, n)[i]``.
        Both `sample` and `sample_noise` route through this so a given seed
        reproduces identical per-node noise across both.
        """
        rngs = spawn_children(seed, len(self._topo_order))
        return {node: rng for node, rng in zip(self._topo_order, rngs, strict=True)}

    def sample_noise(self, n: int, seed: int) -> pd.DataFrame:
        """Draw each node's exogenous noise.

        Returns a DataFrame with one column per node (topological column order)
        and ``n`` rows. Uses the same seeded per-node RNGs as `sample`.
        """
        child_rngs = self._child_rngs(seed)
        noise = {
            node: self.noise_samplers[node](child_rngs[node], n)
            for node in self._topo_order
        }
        return pd.DataFrame(noise, columns=self._topo_order)

    # -- sampling / interventions -------------------------------------------

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Ancestral sampling of ``n`` rows.

        Derives the same per-node noise as `sample_noise(n, seed)`, then walks
        the graph in topological order, evaluating each structural equation on
        its parents' already-computed columns and its own noise. Roots receive
        an empty parents dict. Deterministic given ``seed``.
        """
        noise = self.sample_noise(n, seed)
        data: dict[str, np.ndarray] = {}
        for node in self._topo_order:
            parents_dict = {p: data[p] for p in self.parents(node)}
            data[node] = self.equations[node](parents_dict, noise[node].to_numpy())
        return pd.DataFrame(data, columns=self._topo_order)

    def intervene(self, node: str, value: float) -> SCM:
        """Return a new SCM under the hard intervention do(``node`` = ``value``).

        Pure structural surgery / do-operator: incoming edges to
        ``node`` are removed and its equation is replaced by a constant that
        ignores parents and noise. NO abduction is performed — this is NOT the
        counterfactual (see `counterfactual`). Other equations, noise samplers
        and the declared ``regime`` are preserved.
        """
        if node not in self._graph:
            raise ValueError(f"cannot intervene on unknown node {node!r}.")

        new_graph = self._graph.copy()
        for parent in list(new_graph.predecessors(node)):
            new_graph.remove_edge(parent, node)

        def _constant(parents: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
            return np.full(noise.shape, value)

        new_equations = dict(self.equations)
        new_equations[node] = _constant

        return SCM(
            graph=new_graph,
            equations=new_equations,
            noise_samplers=self.noise_samplers,
            regime=self.regime,
            name=f"{self.name} | do({node}={value})",
        )

    # -- counterfactuals (abduction-action-prediction) ----------------------

    def abduct(self, x_factual: pd.DataFrame) -> pd.DataFrame:
        """ABDUCTION step: recover each individual's exogenous noise.

        Inverts the structural equations on the factual observation to recover
        the individual-specific exogenous noise. For an additive-noise model
        X_j := f_j(parents_j) + U_j the inversion is
        ``u_j = x_j − f_j(parents_j, 0)`` — evaluate the structural equation with
        the FACTUAL parent values and zero noise, then subtract from the observed
        value. For a root X := U this reduces to ``u = x`` (empty parents,
        f({}, 0) = 0 for the additive form).

        This is valid ONLY for additive-noise families; it is gated on the declared ``is_anm`` flag.

        Parameters
        ----------
        x_factual:
            DataFrame with one column per node and ``n`` rows: the factual
            observation(s) whose noise is to be recovered.

        Returns
        -------
        DataFrame of recovered exogenous noise, columns in topological order,
        ``n`` rows (index taken from ``x_factual``).
        """
        if not self.is_anm:
            raise NotImplementedError(
                "abduct requires additive-noise structure: abduction recovers noise "
                "as u = x − f(parents, 0), which is valid only for additive-noise (ANM) "
                "families. This SCM was declared is_anm=False."
            )

        missing = [node for node in self._topo_order if node not in x_factual.columns]
        if missing:
            raise ValueError(
                f"x_factual is missing required node column(s): {missing}."
            )

        n = len(x_factual)
        zeros = np.zeros(n)
        recovered: dict[str, np.ndarray] = {}
        for node in self._topo_order:
            parents_dict = {p: x_factual[p].to_numpy() for p in self.parents(node)}
            noise_free = self.equations[node](parents_dict, zeros)
            recovered[node] = x_factual[node].to_numpy() - noise_free
        return pd.DataFrame(recovered, columns=self._topo_order, index=x_factual.index)

    def counterfactual(
        self, x_factual: pd.DataFrame, action: dict[str, float]
    ) -> pd.DataFrame:
        """Structural counterfactual via ABDUCTION - ACTION - PREDICTION.

        The individual-specific counterfactual the scorer uses for r^CAU / Δ_cost:
        (1) ABDUCTION — recover exogenous noise via `abduct`; (2) ACTION —
        absolute-value do-interventions (each ``action`` value is the NEW ABSOLUTE
        VALUE, not a delta; surgery overrides the node's equation, so its abducted
        noise does not re-enter); (3) PREDICTION — re-evaluate remaining equations in
        topological order. Non-intervened DESCENDANTS change; non-intervened
        ANCESTORS keep their factual values. Gated on ``is_anm`` (see `abduct`).

        Parameters
        ----------
        x_factual: factual observation(s), DataFrame with one column per node, n rows.
        action: intervened node -> new absolute value; scalar (broadcast) OR
            length-n array — the per-row form lets one call sweep n distinct
            candidate actions (the harness's vectorized grid search). May be empty
            (identity counterfactual). Immutability (e.g. Gender) is enforced by the
            recourse harness, not the SCM.

        Returns
        -------
        DataFrame of counterfactual values, columns in topological order, n rows
        (index from ``x_factual``).
        """
        if not self.is_anm:
            raise NotImplementedError(
                "counterfactual requires additive-noise structure: abduction recovers "
                "noise as u = x − f(parents, 0), which is valid only for additive-noise "
                "(ANM) families. This SCM was declared "
                "is_anm=False."
            )

        unknown = [key for key in action if key not in self._graph]
        if unknown:
            raise ValueError(f"action references unknown node(s): {unknown}.")

        # ABDUCTION (also validates x_factual has every node column).
        u = self.abduct(x_factual)

        n = len(x_factual)
        cf: dict[str, np.ndarray] = {}
        for node in self._topo_order:
            if node in action:
                # ACTION: absolute-value do-surgery; the structural equation is
                # overridden, so the abducted noise for this node does not enter.
                # Value may be a scalar (broadcast) or a length-n per-row array,
                # so a single call can enact n distinct actions at once.
                value = np.asarray(action[node], dtype=float)
                if value.ndim == 0:
                    cf[node] = np.full(n, float(value))
                elif value.shape == (n,):
                    cf[node] = value.copy()
                else:
                    raise ValueError(
                        f"action[{node!r}] must be a scalar or a length-{n} array "
                        f"matching x_factual rows, got shape {value.shape}."
                    )
            else:
                # PREDICTION: parents are the COUNTERFACTUAL values already computed
                # (so interventions propagate downstream), with the abducted noise.
                parents_dict = {p: cf[p] for p in self.parents(node)}
                cf[node] = self.equations[node](parents_dict, u[node].to_numpy())
        return pd.DataFrame(cf, columns=self._topo_order, index=x_factual.index)
