"""Readers over the ``results/cross_seed_N6/`` cross-seed tree + the coefficient map.

Nothing here recomputes an experiment. The grid enumeration and cell-directory
naming are imported from the grid runner rather than restated, so a cell rename
cannot silently desynchronize the analysis layer from the tree it reads.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from icknowledge.scm.chains import (
    make_linear_chain,
    make_nonlinear_gaussian_chain,
)
from icknowledge.scm.colliders import (
    make_linear_collider,
    make_nonlinear_gaussian_collider,
)
from icknowledge.scm.triangles import (
    make_linear_triangle,
    make_nonlinear_gaussian_triangle,
)

#: Root of the early-look grid (PS-1 first-6 of the staged N=20). Kept as the
#: default so the N=6 reports keep reproducing off the artifact they were
#: written against; the N=20 tree is passed explicitly by its callers.
DEFAULT_ROOT = Path("results/cross_seed_N6")

#: Root of the completed PS-1 grid: all 12 cells x 20 seeds.
N20_ROOT = Path("results/cross_seed_N20")

#: PS-1 early look: 6 seeds. The analysis layer asserts this rather than inferring it, so a
#: partially-written tree fails loudly instead of being averaged over silently.
N_SEEDS = 6

#: PS-1's pinned CEILING. Not a default anywhere — every N=20 read passes it
#: explicitly, so a partially-materialized tree cannot be averaged as if complete.
N_SEEDS_FULL = 20

#: Root of the PS-4 four-rung tree. Written only by a grid launched with
#: ``--l1-discovered``; kept DISTINCT from N20_ROOT because the two trees carry
#: different common-found denominators and are not interchangeable (PS-4).
L1D_ROOT = Path("results/cross_seed_L1d_N20")

#: The knowledge ladder, ordered cheap-belief -> true-belief.
#: THE THREE-RUNG DEFAULT, unchanged: every three-rung report and figure reads the
#: sealed three-rung tree through it and must keep doing so.
CONDITIONS = ("L0", "L1-oracle", "L2")

#: [PS-4] The four-rung ladder, in `aggregation._CONDITION_ORDER`'s canonical
#: order. Not a default anywhere — H3 passes it EXPLICITLY alongside
#: ``L1D_ROOT``, so a three-rung tree can never be read as if it had four rungs
#: (the loader would simply fail to find the L1-discovered file, loudly).
CONDITIONS_L1D = ("L0", "L1-oracle", "L1-discovered", "L2")

# The chain is appended, not inserted: `grid_cells()`
# iterates topology-major, so appending leaves every pre-existing cell at the same
# position in every report table that enumerates cells in this order.
TOPOLOGIES = ("triangle", "collider", "chain")
FAMILIES = ("linear", "nlg")
REGIMES = ("additive", "effect_modifying")

#: The ValidityDisp guard threshold: below this the valid-subset cost is a near-empty
#: average and must not enter a cross-seed mean unflagged (the ValidityDisp guard).
VALID_SUBSET_MIN_N = 30


@dataclass(frozen=True)
class Cell:
    """One experimental cell of the 2x2x2 grid (seeds are the within-cell axis)."""

    topology: str
    family: str
    regime: str

    @property
    def key(self) -> str:
        return f"{self.topology}_{self.family}_{self.regime}"

    @property
    def label(self) -> str:
        return f"{self.topology} / {self.family} / {self.regime}"

    def seed_dir(self, seed_idx: int, root: Path = DEFAULT_ROOT) -> Path:
        return root / f"{self.key}_seed_{seed_idx:02d}"


def grid_cells(topologies: tuple[str, ...] = TOPOLOGIES) -> list[Cell]:
    """The grid's cells, topology-major (collider cells are the PS-9 focus).

    ``topologies`` narrows the enumeration — used to evaluate a gate on one
    topology's cells while the rest of the grid is still materializing, without
    a report silently averaging over cells that do not exist yet.
    """
    return [
        Cell(topology, family, regime)
        for topology in topologies
        for family in FAMILIES
        for regime in REGIMES
    ]


def collider_cells() -> list[Cell]:
    return [c for c in grid_cells() if c.topology == "collider"]


def chain_cells() -> list[Cell]:
    return [c for c in grid_cells() if c.topology == "chain"]


# --------------------------------------------------------------------------- #
# Artifact loaders
# --------------------------------------------------------------------------- #


def load_summary(name: str, root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Load one of the grid run's ``summary/*.csv`` tables."""
    path = root / "summary" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Grid summary artifact {path} is missing — the analysis layer reads "
            "the grid outputs and "
            "does not regenerate them; re-run scripts/run_cross_seed_grid.py first."
        )
    return pd.read_csv(path)


def load_manifest(cell: Cell, seed_idx: int, root: Path = DEFAULT_ROOT) -> dict:
    path = cell.seed_dir(seed_idx, root) / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_classifier_params(
    cell: Cell, seed_idx: int, root: Path = DEFAULT_ROOT
) -> tuple[np.ndarray, float, list[str]]:
    """(w, b, feature_names) of the ONE classifier this (cell, seed) fit.

    Reads ``classifier_params.json``, the persisted classifier fit (persistence
    of the single per-(cell, seed) fit). ``w`` is ordered to match
    ``feature_names``, which is the classifier's own column order.

    RAISES rather than falling back when the file is absent. The N=6 tree at
    ``cross_seed_N6/`` predates persistence and legitimately has none; a silent
    fallback to the L0-rung proxy would make two different quantities share one
    column name across trees, which is exactly the ambiguity persistence was added
    to remove. Callers that want the proxy ask for it by name.
    """
    path = cell.seed_dir(seed_idx, root) / "classifier_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Persisted classifier params are written by the grid runner; "
            "trees written before it (results/cross_seed_N6/) carry none. Read the "
            "L0-rung proxy explicitly if that is what you want — this loader does not "
            "silently substitute one for the other."
        )
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)[cell.regime]
    features = [str(f) for f in record["feature_names"]]
    w = np.array([float(record["coef"][f]) for f in features], dtype=float)
    return w, float(record["intercept"]), features


def load_scoring(
    cell: Cell, seed_idx: int, condition: str, root: Path = DEFAULT_ROOT
) -> pd.DataFrame:
    """Per-individual scoring table for one (cell, seed, condition)."""
    path = (
        cell.seed_dir(seed_idx, root)
        / f"scoring_table_{cell.regime}_{condition}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing grid scoring table {path}")
    return pd.read_csv(path)


def load_scoring_tables(
    cell: Cell,
    seed_idx: int,
    root: Path = DEFAULT_ROOT,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, pd.DataFrame]:
    """Per-individual scoring tables for one (cell, seed), keyed by condition.

    ``conditions`` defaults to the THREE-rung ladder, so every existing reader
    keeps its exact behaviour on the sealed tree. H3 reads the PS-4
    four-rung tree by passing ``conditions=CONDITIONS_L1D, root=L1D_ROOT``
    together — the pair is the whole parameterization, and passing one without
    the other fails loudly (a three-rung tree has no L1-discovered file; a
    four-rung tree read at three rungs silently drops the rung under test, which
    is why nothing here defaults ``root`` to L1D_ROOT).

    This is the shape `aggregation.build_by_group_frame` consumes, so a H3
    re-aggregation off the four-rung tree needs no reshaping step.
    """
    return {
        condition: load_scoring(cell, seed_idx, condition, root)
        for condition in conditions
    }


def cell_frame(frame: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    """Restrict a long summary frame to one cell (keys are columns, not an index)."""
    mask = (
        (frame["topology"] == cell.topology)
        & (frame["family"] == cell.family)
        & (frame["regime"] == cell.regime)
    )
    return frame.loc[mask]


def seed_order(frame: pd.DataFrame) -> list[int]:
    """Seed *indices* in the grid run's spawn order (the seed-generation
    mechanics), not raw seed integers."""
    return sorted(int(v) for v in frame["seed_idx"].unique())


# --------------------------------------------------------------------------- #
# Ground-truth vs estimated load-bearing coefficients (the mechanism diagnostic's Signal 2)
# --------------------------------------------------------------------------- #

#: The builders the grid runner used, keyed exactly as its ``_FAMILIES`` tables are.
_BUILDERS = {
    ("triangle", "linear"): make_linear_triangle,
    ("triangle", "nlg"): make_nonlinear_gaussian_triangle,
    ("collider", "linear"): make_linear_collider,
    ("collider", "nlg"): make_nonlinear_gaussian_collider,
    ("chain", "linear"): make_linear_chain,
    ("chain", "nlg"): make_nonlinear_gaussian_chain,
}


def _builder_defaults(topology: str, family: str) -> dict[str, float]:
    """Ground-truth structural coefficients, read off the builder signature.

    ``pipeline.run_regime`` calls ``scm_builder(regime)`` with no coefficient
    overrides, so the signature defaults ARE the ground truth every grid cell ran
    on. Reading them by introspection rather than restating the numbers here is
    deliberate: a builder retune cannot leave a stale copy in the analysis layer.
    """
    signature = inspect.signature(_BUILDERS[(topology, family)])
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
        and isinstance(param.default, (int, float))
    }


#: Per (topology, family): for each load-bearing edge, the
#: (edge label, FITTED NODE, main-effect term, A-interaction term) 4-tuple naming
#: where the coefficient lives in the manifest's per-node coefficient dict.
#: ``None`` interaction == the edge carries no A modification.
#:
#: The fitted node is per-EDGE rather than per-cell because the CHAIN's two
#: load-bearing edges live in DIFFERENT equations: X1->X2 is a coefficient of X2's
#: equation and X2->X3 a coefficient of X3's. The triangle and collider both have
#: all their load-bearing edges converging on one node; the chain's serial
#: structure does not.
_EDGE_TERMS = {
    ("triangle", "linear"): [("X1->X2", "X2", "X1", "A*X1")],
    ("triangle", "nlg"): [("X1->X2", "X2", "tanh(X1)", "A*tanh(X1)")],
    ("collider", "linear"): [
        ("X1->X3", "X3", "X1", "A*X1"),
        ("X2->X3", "X3", "X2", None),
    ],
    ("collider", "nlg"): [
        ("X1->X3", "X3", "tanh(X1)", "A*tanh(X1)"),
        ("X2->X3", "X3", "tanh(X2)", None),
    ],
    # [the chain SCM specification] gamma sits on the UPSTREAM edge X1->X2, so that is where the
    # A-interaction term appears; X2->X3 is unmodified in both regimes.
    ("chain", "linear"): [
        ("X1->X2", "X2", "X1", "A*X1"),
        ("X2->X3", "X3", "X2", None),
    ],
    ("chain", "nlg"): [
        ("X1->X2", "X2", "tanh(X1)", "A*tanh(X1)"),
        ("X2->X3", "X3", "tanh(X2)", None),
    ],
}


def true_load_bearing(cell: Cell, group: float) -> dict[str, float]:
    """Ground-truth load-bearing coefficient(s) as seen by group ``A = group``.

    Triangle: the X1->X2 gain (``g`` additive; ``g_neg``/``g_pos`` under effect
    modification). Collider: the X1->X3 gain (``beta + gamma*A`` under effect
    modification, ``beta`` additive) and the unmodified X2->X3 gain ``eta``.
    """
    d = _builder_defaults(cell.topology, cell.family)
    if cell.topology == "triangle":
        if cell.regime == "additive":
            return {"X1->X2": float(d["g"])}
        return {"X1->X2": float(d["g_pos"] if group > 0 else d["g_neg"])}

    # Collider and chain: beta_a / beta_A / beta_a2 / beta_a3 are the A->X_i DIRECT
    # edges and are deliberately NOT load-bearing here — they sit outside the
    # actionable -> classifier propagation sub-block the diagnostic is about
    # (the SCM specification's and the chain SCM specification's identical lever note).
    gamma = float(d["gamma"])
    beta = float(d["beta"])
    modified = beta + gamma * float(group) if cell.regime == "effect_modifying" else beta
    if cell.topology == "collider":
        return {"X1->X3": modified, "X2->X3": float(d["eta"])}
    # Chain: the MODIFIED edge is the upstream X1->X2 (not the
    # terminal one, as on the collider), and the downstream X2->X3 gain is delta,
    # unmodified in both regimes.
    return {"X1->X2": modified, "X2->X3": float(d["delta"])}


def estimated_load_bearing(
    cell: Cell, coefficients: dict, group: float
) -> dict[str, float]:
    """L1-oracle's fitted counterpart of :func:`true_load_bearing`.

    ``coefficients`` is the manifest's ``estimation.<regime>.coefficients`` dict.
    An A-interaction term is folded in at ``A = group`` so the estimated and true
    quantities are the same object: the gain THIS group actually propagates at.
    """
    out: dict[str, float] = {}
    for edge, node, main_term, interaction_term in _EDGE_TERMS[
        (cell.topology, cell.family)
    ]:
        fitted = coefficients[node]
        value = float(fitted[main_term])
        if interaction_term is not None and interaction_term in fitted:
            value += float(group) * float(fitted[interaction_term])
        out[edge] = value
    return out


def tanh_input_features(cell: Cell) -> list[str]:
    """Features entering a ``tanh(.)`` in the true equations (the mechanism diagnostic's Signal 3).

    Linear cells have none — Signal 3 is N/A there, reported as such rather than
    silently zero-filled.
    """
    if cell.family != "nlg":
        return []
    # Triangle: only X1 is tanh-wrapped (in X2's equation). Collider: X1 and X2 are
    # both wrapped, in X3's equation. Chain (the chain SCM specification): X1 is wrapped in X2's
    # equation
    # and X2 in X3's — two inputs, but reached serially rather than in parallel,
    # which is why the chain SCM specification's non-saturation constraint had to be written on X2's
    # MARGINAL rather than on its noise alone.
    return ["X1"] if cell.topology == "triangle" else ["X1", "X2"]
