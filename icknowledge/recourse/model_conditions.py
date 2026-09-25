"""Knowledge conditions as pluggable causal models.

Each condition is a causal model exposing the SAME model-agnostic contract:

    predict(factual, delta)         -> feature_vector   (shape (F,))
    predict_batch(factual, deltas)  -> feature_matrix    (shape (K, F))

``factual`` is the individual's full node row (a Series over ALL SCM nodes,
including the immutable A, which L2 needs for abduction and L0 ignores). ``delta``
is the intervention delta over the ACTED variables (relative to the factual value).
The returned feature vector is aligned to the classifier's ``feature_names``
(the group-blind descendants), so the procedure can push it straight through h.

L0, L2, L1-oracle and L1-discovered are wired here behind the same contract
(L1-oracle: an estimated SCM from `icknowledge.estimation`; L1-discovered: an
estimated-on-discovered SCM). `L1DiscoveredEstimatedSCMModel` is instantiated by
`recourse/pipeline.py` only when the rung is requested by name (PS-7, PS-4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from icknowledge.scm.base import SCM


def _as_deltas_matrix(delta: Mapping[str, float], acted: Sequence[str]) -> np.ndarray:
    """Turn a single {acted_var: delta} dict into a (1, |acted|) matrix in acted order."""
    return np.array([[float(delta.get(var, 0.0)) for var in acted]], dtype=float)


class _SCMDispatchModel:
    """Shared dispatch for conditions that hold an SCM behind the predict contract.

    predict runs the abduction–action–prediction engine on the HELD SCM
    (whichever one the condition was constructed with), so downstream descendants
    of an acted variable propagate under that SCM's equations. L2 holds the true
    SCM; L1-oracle holds the estimated one — the dispatch is identical in shape,
    which is exactly what makes the L2 vs. L1-oracle contrast attributable to the
    equations alone.
    """

    name: str  # set by the concrete condition subclasses

    def __init__(self, scm: SCM, feature_names: Sequence[str], acted: Sequence[str]):
        self.scm = scm
        self.feature_names = list(feature_names)
        self.acted = list(acted)

    def predict_batch(self, factual: pd.Series, deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(deltas, dtype=float)
        k = deltas.shape[0]
        nodes = self.scm.nodes
        feature_cols = self.feature_names
        base_row = factual[nodes].to_numpy(dtype=float)
        factual_features = factual[feature_cols].to_numpy(dtype=float)
        out = np.empty((k, len(feature_cols)), dtype=float)

        # A zero delta on an axis means that variable is NOT intervened — it is left
        # FREE to propagate under the held SCM (a 0 per axis = "act on a subset").
        # This is exactly what creates the causal advantage: acting on an upstream
        # variable lets its descendants move for free. Because one counterfactual call
        # must intervene on a FIXED node set across its rows, group the candidates by
        # their nonzero-delta pattern and run one vectorized counterfactual per group.
        acted_nonzero = deltas != 0.0  # (K, |acted|)
        patterns = np.unique(acted_nonzero, axis=0)
        for pattern in patterns:
            rows = np.nonzero((acted_nonzero == pattern).all(axis=1))[0]
            acted_vars = [self.acted[j] for j in range(len(self.acted)) if pattern[j]]
            if not acted_vars:
                # No-op: the identity counterfactual reproduces the factual features.
                out[rows] = factual_features
                continue
            rep = pd.DataFrame(np.tile(base_row, (len(rows), 1)), columns=nodes)
            # ACTION values are ABSOLUTE (factual + delta), per the SCM's
            # absolute-value do-semantics; only the intervened subset is set, the rest
            # propagate.
            action = {
                var: rep[var].to_numpy() + deltas[rows, self.acted.index(var)]
                for var in acted_vars
            }
            cf = self.scm.counterfactual(rep, action)
            out[rows] = cf[feature_cols].to_numpy()
        return out

    def predict(self, factual: pd.Series, delta: Mapping[str, float]) -> np.ndarray:
        return self.predict_batch(factual, _as_deltas_matrix(delta, self.acted))[0]


class L2TrueSCMModel(_SCMDispatchModel):
    """L2: full causal knowledge — the TRUE SCM itself.

    believed == realized here (same SCM); this class is ALSO the realizer/scorer
    used for EVERY condition (scoring.py). Anchor: the oracle upper bound
    (Karimi; Majumdar).
    """

    name = "L2"


class L1OracleEstimatedSCMModel(_SCMDispatchModel):
    """L1-oracle: true graph + ESTIMATED structural equations.

    # [the L1-oracle form-template-known semantics; recourse-harness] The
    # generator
    # receives `build_estimated_scm(true_scm, fitted)` — the TRUE graph with
    # OLS-fitted equations from the classifier's training sample under a SUPPLIED
    # form template. This class must NOT hold or expose the true SCM: L1-oracle
    # only sees the estimated one, and any need to reach the true SCM belongs at
    # the scoring layer (the realizer), never inside the condition. The dispatch
    # is identical in shape to L2's — the L2→L1-oracle gap therefore quantifies
    # equation-estimation error alone (H1).

    Constructor takes the ESTIMATED `SCM` (the return value of
    `build_estimated_scm`); `counterfactual` works unchanged on it because the
    fitted equations preserve the additive-noise convention.
    """

    name = "L1-oracle"


class L1DiscoveredEstimatedSCMModel(_SCMDispatchModel):
    """L1-discovered: DISCOVERED graph + ESTIMATED structural equations.

    # [PS-7; recourse-harness] The generator
    # receives `build_estimated_scm_from_discovery(discovered_graph, fitted,
    # data)` — the oriented DAG with OLS-fitted equations under a form
    # template SYNTHESIZED from that same graph (the form-preserving estimation basis;
    # no A·X column in either regime). Like L1-oracle, this class must NOT hold or
    # expose the true SCM: L1-discovered sees only the discovered graph and the
    # estimation sample, and any need to reach the truth belongs at the scoring
    # layer (the realizer), never inside the condition. The dispatch is identical
    # in shape to L1-oracle's — the L1-oracle→L1-discovered gap therefore
    # quantifies graph error (plus interaction-blindness under EM, which
    # PS-7 puts on record rather than hiding).

    Constructor takes the ESTIMATED-ON-DISCOVERED `SCM`;  `counterfactual` works
    unchanged on it because the fitted equations — including the intercept-only
    marginals of parentless nodes (PS-7 note b) — preserve the additive-noise
    convention.

    Instantiated by `recourse/pipeline.py` only when the rung is requested by
    name (PS-4).
    """

    name = "L1-discovered"


class L0AssociationalModel:
    """L0: associational / independence-of-mechanisms (J = I) baseline.

    # L0 = associational baseline = independence-of-mechanisms assumption (J=I).
    # Believed recourse is the plain point-distance solve: the generator believes the
    # features are independent, so acting on a variable moves ONLY that feature and
    # NOTHING propagates.

    predict(factual, delta) = factual features, with acted features replaced by
    (x + delta) and every other feature held fixed. A is never acted and is not
    a feature, so it plays no role here.
    """

    def __init__(self, feature_names: Sequence[str], acted: Sequence[str]):
        self.feature_names = list(feature_names)
        self.acted = list(acted)
        # index of each acted var within feature_names (acted vars that are not
        # features contribute nothing to the believed feature vector under J=I).
        self._acted_feature_cols = {
            var: self.feature_names.index(var)
            for var in self.acted
            if var in self.feature_names
        }
        self.name = "L0"

    def predict_batch(self, factual: pd.Series, deltas: np.ndarray) -> np.ndarray:
        deltas = np.asarray(deltas, dtype=float)
        k = deltas.shape[0]
        base = np.tile(factual[self.feature_names].to_numpy(dtype=float), (k, 1))
        # J = I: add the delta ONLY to its own feature column; no propagation to any
        # other feature (non-acted features held fixed).
        for i, var in enumerate(self.acted):
            col = self._acted_feature_cols.get(var)
            if col is not None:
                base[:, col] += deltas[:, i]
        return base

    def predict(self, factual: pd.Series, delta: Mapping[str, float]) -> np.ndarray:
        return self.predict_batch(factual, _as_deltas_matrix(delta, self.acted))[0]
