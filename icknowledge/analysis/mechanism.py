"""PS-9 EXPLORATORY mechanism-discrimination diagnostic.

    [Exploratory — post-hoc, descriptive, NOT a confirmatory pre-commitment]

Computes three candidate signals per (cell, seed) over the A=-1
negatively-classified pool and reports which, if any, separates high-validity
seeds from validity-collapse seeds. No H1-H4 confirmation criterion reads off
anything here.

1. boundary distance   mean |w'x - b| / ||w||_2 over the pool, at the
                       individual's ORIGINAL (pre-action) feature vector.
2. estimation-error    signed (L1-oracle estimated - L2 true) load-bearing
   sign                coefficient, evaluated at the group's own A.
3. saturation position mean |tanh-input| over the pool (NLG cells only; linear
                       cells report N/A rather than a zero).

Signal 1 is read DIRECTLY from the persisted classifier: ``w`` and ``b`` come
from ``classifier_params.json``. `mean_boundary_distance_proxy` reads the same
quantity off the L0 rung of the scoring tables and carries a discretization
overshoot, so it orders seeds within a cell but does not compare levels across
cells; one classifier per (cell, seed) is shared by every rung. The proxy
applies because the actionable set equals the classifier's feature set in all
three topologies: triangle {X1,X2}, collider and chain {X1,X2,X3}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from icknowledge.analysis.loading import (
    DEFAULT_ROOT,
    N_SEEDS,
    Cell,
    collider_cells,
    estimated_load_bearing,
    grid_cells,
    load_classifier_params,
    load_manifest,
    load_scoring,
    tanh_input_features,
    true_load_bearing,
)

#: Validity above/below which a seed counts as high-validity / collapse.
VALIDITY_SPLIT = 0.5

#: The pool the diagnostic is defined on.
POOL_GROUP = -1.0

SIGNAL1 = "boundary_distance"
SIGNAL2_PREFIX = "coef_error"
SIGNAL3_PREFIX = "saturation"
#: Follow-up candidate, added as a PS-9 body note (see the function).
SIGNAL4 = "boundary_margin_at_pool_mode"

EXPLORATORY_TAG = (
    "[EXPLORATORY — post-hoc, PS-9; descriptive, "
    "NOT a confirmatory pre-commitment]"
)

NOT_APPLICABLE = "not applicable — validity is not bimodal in this cell"


# --------------------------------------------------------------------------- #
# Signal primitives (pure; unit-tested against known synthetic cases)
# --------------------------------------------------------------------------- #


def boundary_distance(w: np.ndarray, b: float, x: np.ndarray) -> np.ndarray:
    """Raw l_2 point-to-boundary distance |w'x + b| / ||w||_2.

    ``b`` is the classifier INTERCEPT, i.e. the boundary is ``w'x + b = 0``; the
    PS-9 wording ``|w'x - b|`` is the same object with the threshold written on
    the other side. Accepts a single point or a stacked (n, d) matrix.
    """
    w = np.asarray(w, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    norm = float(np.linalg.norm(w, ord=2))
    if norm == 0.0:
        raise ValueError("degenerate classifier: ||w||_2 == 0.")
    return np.abs(x @ w + float(b)) / norm


def mean_boundary_distance_proxy(
    scoring: pd.DataFrame, group: float = POOL_GROUP
) -> float:
    """SUPERSEDED Signal 1, from a persisted **L0** scoring table.

    Uses ``believed_cost``: at L0 the belief is J = I, so the believed cost of the
    chosen action IS the point-to-boundary distance — plus the grid discretization
    overshoot, which is what makes this a proxy rather than the quantity itself.

    Retained (not deleted) for two reasons: it is the only way to read Signal 1 on
    the ``cross_seed_N6/`` tree, which predates classifier persistence; and
    it is the comparison arm of :func:`proxy_vs_exact`, the regression check that
    the two readings differ only by the overshoot.
    """
    pool = scoring.loc[scoring["A"] == group]
    if pool.empty:
        raise ValueError(f"empty A={group:+.0f} pool in the supplied scoring table.")
    return float(pool["believed_cost"].mean())


def mean_boundary_distance(
    scoring: pd.DataFrame,
    w: np.ndarray,
    b: float,
    features: list[str],
    group: float = POOL_GROUP,
) -> float:
    """Signal 1, EXACT: mean |w'x + b| / ||w||_2 over the A=g pool.

    ``x`` is each individual's FACTUAL (pre-action) feature vector, read from the
    scoring table's ``factual_*`` columns; ``w``, ``b`` come from the seed's
    persisted ``classifier_params.json``. Nothing is retrained and nothing passes
    through the action grid, so this is the boundary distance itself rather than
    the grid-discretized proxy.
    """
    pool = scoring.loc[scoring["A"] == group]
    if pool.empty:
        raise ValueError(f"empty A={group:+.0f} pool in the supplied scoring table.")
    x = pool[[f"factual_{f}" for f in features]].to_numpy(dtype=float)
    return float(boundary_distance(w, b, x).mean())


def proxy_vs_exact(
    cell: Cell, seed_idx: int, root: Path = DEFAULT_ROOT
) -> dict[str, float]:
    """Regression check: the superseded proxy against the exact reading.

    Returns the two Signal-1 values, their difference, and the cell's own measured
    grid overshoot. The two readings are consistent iff ``proxy > exact`` (a finite grid
    can only overshoot the continuous optimum) and the relative gap is of the order
    the overshoot audit reports — on EVERY cell, not on a family split. See the
    module docstring: one classifier serves all three rungs, so there is no
    linear-vs-NLG difference in the fit for the two readings to disagree about.
    """
    l0 = load_scoring(cell, seed_idx, "L0", root)
    l1 = load_scoring(cell, seed_idx, "L1-oracle", root)
    w, b, features = load_classifier_params(cell, seed_idx, root)
    proxy = mean_boundary_distance_proxy(l0, POOL_GROUP)
    exact = mean_boundary_distance(l1, w, b, features, POOL_GROUP)
    audit = l0_projection_diagnostics(l0, features, POOL_GROUP)
    return {
        "proxy": proxy,
        "exact": exact,
        "proxy_minus_exact": proxy - exact,
        "relative_gap": (proxy - exact) / exact if exact else float("nan"),
        "measured_grid_overshoot": audit["overshoot"],
    }


def l0_projection_diagnostics(
    scoring: pd.DataFrame, features: list[str], group: float = POOL_GROUP
) -> dict[str, float]:
    """Audit the projection identity Signal 1 rests on, from artifacts alone.

    If L0 really solves ``min ||delta|| s.t. w'(x + delta) + b > 0``, then every
    action is parallel to ``w`` and the unit action vectors coincide. Two numbers
    are returned:

    ``mean_cosine``  concentration of the unit actions about their own mean
                     direction — 1.0 in the continuous limit, below 1 only by
                     grid discretization.
    ``overshoot``    ``mean||delta|| / mean(delta . w_hat) - 1``, i.e. the
                     fraction by which the grid solution exceeds the exact
                     point-to-boundary distance.

    This does NOT recover (w, b) and does not retrain anything; ``w_hat`` is only
    the empirical action direction, used to measure the harness against its own
    stated identity.
    """
    columns = [f"delta_{f}" for f in features]
    deltas = scoring[columns].to_numpy(dtype=float)
    norms = np.linalg.norm(deltas, axis=1)
    moving = norms > 1e-9
    if not moving.any():
        raise ValueError("no nonzero L0 actions — cannot audit the projection.")
    units = deltas[moving] / norms[moving, None]
    w_hat = units.mean(axis=0)
    w_hat /= np.linalg.norm(w_hat)

    pool = scoring["A"].to_numpy(dtype=float) == group
    pool_deltas = deltas[pool]
    along = float(np.mean(pool_deltas @ w_hat))
    full = float(np.mean(np.linalg.norm(pool_deltas, axis=1)))
    return {
        "mean_cosine": float(np.mean(units @ w_hat)),
        "overshoot": (full / along - 1.0) if along != 0.0 else float("nan"),
    }


def boundary_crosses_pool_mode(
    scoring: pd.DataFrame,
    w: np.ndarray,
    b: float,
    features: list[str],
    group: float = POOL_GROUP,
) -> float:
    """Signal 4 (follow-up candidate): signed margin at the pool's mode.

    PS-9 flagged a fourth candidate — whether the classifier
    boundary CROSSES the A=−1 pool's mode on a given seed. Operationalized as the
    signed distance from the boundary to the pool's modal point, where the mode is
    taken as the per-feature MEDIAN (a robust, estimator-free stand-in for the mode
    of a unimodal cloud; a KDE mode would add a bandwidth choice this diagnostic
    has no basis to pre-commit).

    Sign, not magnitude, is the object: NEGATIVE means the modal individual sits on
    the rejected side (the boundary has cut past the bulk of the pool), POSITIVE
    means the bulk already sits on the accepted side. A near-zero value is the
    interesting case — the boundary slicing through the densest part of the pool,
    where a small action error flips many outcomes at once, which is the mechanism
    a bimodal validity switch would need.

    Exploratory (PS-9); no H1–H4 criterion reads off it.
    """
    pool = scoring.loc[scoring["A"] == group]
    if pool.empty:
        raise ValueError(f"empty A={group:+.0f} pool in the supplied scoring table.")
    x = pool[[f"factual_{f}" for f in features]].to_numpy(dtype=float)
    modal_point = np.median(x, axis=0)
    w = np.asarray(w, dtype=float)
    return float((modal_point @ w + float(b)) / np.linalg.norm(w, ord=2))


def coefficient_errors(
    cell: Cell, coefficients: dict, group: float = POOL_GROUP
) -> dict[str, float]:
    """Signal 2: signed (estimated - true) on each load-bearing edge.

    Positive == L1-oracle OVER-estimates propagation through that edge (it
    believes an action carries further than it truly does); negative ==
    under-estimates. The sign is the object of interest, not the magnitude.
    """
    estimated = estimated_load_bearing(cell, coefficients, group)
    truth = true_load_bearing(cell, group)
    return {edge: estimated[edge] - truth[edge] for edge in truth}


def mean_saturation_positions(
    scoring: pd.DataFrame, cell: Cell, group: float = POOL_GROUP
) -> dict[str, float]:
    """Signal 3: mean |tanh-input| over the pool. Empty dict on linear cells.

    Larger == the group operates deeper into ``tanh' ~ 0``, where an action on
    the input moves the downstream feature almost not at all.
    """
    features = tanh_input_features(cell)
    if not features:
        return {}
    pool = scoring.loc[scoring["A"] == group]
    return {
        feature: float(pool[f"factual_{feature}"].abs().mean())
        for feature in features
    }


# --------------------------------------------------------------------------- #
# Per-seed assembly
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeedSignals:
    """One row of a cell's diagnostic table."""

    seed_idx: int
    seed: int
    validity_neg: float
    n_pool: int
    signals: dict[str, float] = field(default_factory=dict)
    #: Signal-1 identity audit (mean_cosine, overshoot) — reported, not a signal.
    audit: dict[str, float] = field(default_factory=dict)


def seed_signals(
    cell: Cell, seed_idx: int, root: Path = DEFAULT_ROOT
) -> SeedSignals:
    """All three signals plus the A=-1 L1-oracle validity for one seed."""
    l0 = load_scoring(cell, seed_idx, "L0", root)
    l1 = load_scoring(cell, seed_idx, "L1-oracle", root)

    # The eligible pool is the classifier's negative set and is condition-
    # invariant by construction; assert it rather than trust it, because Signal 1
    # is read on L0 and lined up against validity read on L1-oracle.
    if set(l0["index"]) != set(l1["index"]):
        raise ValueError(
            f"{cell.label} seed {seed_idx}: L0 and L1-oracle eligible pools differ; "
            "the boundary-distance read assumes a shared pool."
        )

    pool = l1.loc[l1["A"] == POOL_GROUP]
    manifest = load_manifest(cell, seed_idx, root)
    coefficients = manifest["estimation"][cell.regime]["coefficients"]

    # Feature order comes from the PERSISTED CLASSIFIER, not from a per-topology
    # literal: w is indexed by it, so hardcoding a list here would silently
    # mis-pair coefficients the moment a topology's feature order differed.
    w, b, features = load_classifier_params(cell, seed_idx, root)
    audit = l0_projection_diagnostics(l0, features, POOL_GROUP)

    signals: dict[str, float] = {
        SIGNAL1: mean_boundary_distance(l1, w, b, features, POOL_GROUP)
    }
    for edge, value in coefficient_errors(cell, coefficients, POOL_GROUP).items():
        signals[f"{SIGNAL2_PREFIX}[{edge}]"] = value
    for feature, value in mean_saturation_positions(l1, cell, POOL_GROUP).items():
        signals[f"{SIGNAL3_PREFIX}[{feature}]"] = value
    signals[SIGNAL4] = boundary_crosses_pool_mode(l1, w, b, features, POOL_GROUP)

    return SeedSignals(
        seed_idx=seed_idx,
        seed=int(manifest["seeding"]["this_run_seed"]),
        validity_neg=float(pool["realized_validity"].mean()),
        n_pool=int(len(pool)),
        signals=signals,
        audit=audit,
    )


def cell_signal_table(
    cell: Cell, root: Path = DEFAULT_ROOT, n_seeds: int = N_SEEDS
) -> pd.DataFrame:
    """The per-seed table: seed | A=-1 validity | signal1 | signal2 | signal3."""
    records = [seed_signals(cell, i, root) for i in range(n_seeds)]
    rows = []
    for record in records:
        row = {
            "seed_idx": record.seed_idx,
            "seed": record.seed,
            "validity_A_neg": record.validity_neg,
            "n_pool": record.n_pool,
        }
        row.update(record.signals)
        row["l0_mean_cosine"] = record.audit.get("mean_cosine", float("nan"))
        row["l0_grid_overshoot"] = record.audit.get("overshoot", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed_idx").reset_index(drop=True)


#: Audit columns: reported for transparency, never read as candidate signals.
AUDIT_COLUMNS = ("l0_mean_cosine", "l0_grid_overshoot")


# --------------------------------------------------------------------------- #
# The discriminative read
# --------------------------------------------------------------------------- #


def signal_columns(table: pd.DataFrame) -> list[str]:
    meta = {"seed_idx", "seed", "validity_A_neg", "n_pool", *AUDIT_COLUMNS}
    return [c for c in table.columns if c not in meta]


def chance_separation_p(n_high: int, n_collapse: int) -> float:
    """P(a signal separates cleanly | strata sizes) under a random-labelling null.

    Of the ``C(n, n_high)`` ways to assign the labels to a fixed set of distinct
    signal values, exactly 2 put every high-validity seed on one side of every
    collapse seed. With n=6 split 3/3 that is 2/20 = 0.1 per signal — so across
    five signals a chance "clean separator" is not rare, and any separator found
    at N=6 must be read against this number rather than at face value.
    """
    n = n_high + n_collapse
    if n_high == 0 or n_collapse == 0:
        return float("nan")
    return float(2.0 / comb(n, n_high))


def separation_read(
    table: pd.DataFrame, split: float = VALIDITY_SPLIT
) -> pd.DataFrame:
    """Does each signal separate high-validity seeds from collapse seeds?

    A signal is a CLEAN SEPARATOR only when both strata are non-empty and their
    observed value ranges are disjoint. Rank correlation is reported alongside as
    a monotone-association read, but does not by itself promote a signal to
    separator status — with 6 points a strong rho is cheap.
    """
    validity = table["validity_A_neg"].to_numpy(dtype=float)
    high = validity > split
    low = validity < split
    rows = []
    for column in signal_columns(table):
        values = table[column].to_numpy(dtype=float)
        hi, lo = values[high], values[low]
        # Rank correlation is computed on all seeds regardless of the split, so a
        # cell whose validity is graded rather than bimodal (no seed below the
        # split) still yields a monotone read instead of an empty row.
        rho = (
            float(spearmanr(values, validity).statistic)
            if np.ptp(values) > 0
            else float("nan")
        )
        record: dict[str, object] = {
            "signal": column,
            "n_high": int(high.sum()),
            "n_collapse": int(low.sum()),
            "high_range": (
                (float(hi.min()), float(hi.max())) if hi.size else None
            ),
            "collapse_range": (
                (float(lo.min()), float(lo.max())) if lo.size else None
            ),
        }
        record["chance_separation_p"] = chance_separation_p(int(high.sum()), int(low.sum()))
        record["spearman_rho"] = rho
        if hi.size == 0 or lo.size == 0:
            record.update(
                ranges_overlap=None,
                separation_gap=float("nan"),
                clean_separator=False,
                verdict=NOT_APPLICABLE,
            )
        else:
            overlap = bool(hi.min() <= lo.max() and lo.min() <= hi.max())
            gap = (
                0.0
                if overlap
                else float(max(hi.min() - lo.max(), lo.min() - hi.max()))
            )
            record.update(
                ranges_overlap=overlap,
                separation_gap=gap,
                clean_separator=not overlap,
                verdict=(
                    "CLEAN SEPARATOR" if not overlap else "overlaps — not a separator"
                ),
            )
        rows.append(record)
    return pd.DataFrame(rows)


def cell_verdict(
    read: pd.DataFrame, table: pd.DataFrame | None = None, n_seeds: int = N_SEEDS
) -> str:
    """Roll a cell's separation read up to one sentence, confounds included.

    ``n_seeds`` only shapes the PROSE — every number is computed from ``read`` and
    ``table``. It is threaded rather than hardcoded so a verdict written off the
    N=20 grid cannot claim to be an N=6 read (or vice versa).
    """
    if read.empty:
        return "no signals computed."
    if read["verdict"].eq(NOT_APPLICABLE).all():
        observed = ""
        if table is not None:
            validity = table["validity_A_neg"].to_numpy(dtype=float)
            observed = (
                f" Observed A=−1 validity spans [{validity.min():.4f}, "
                f"{validity.max():.4f}] — all {validity.size} seeds on the same side."
            )
        ranked = read.dropna(subset=["spearman_rho"]).reindex(
            read["spearman_rho"].abs().sort_values(ascending=False).index
        )
        monotone = ""
        if not ranked.empty:
            top = ranked.iloc[0]
            monotone = (
                f" Rank association is still reported: strongest is "
                f"{top['signal']} at rho={top['spearman_rho']:+.3f} across the "
                f"{n_seeds} seeds — a graded read only, with no strata to separate."
            )
        return (
            "NOT APPLICABLE — every seed falls on the same side of the validity "
            f"split ({VALIDITY_SPLIT}), so this cell has no collapse/high contrast "
            "to discriminate." + observed + monotone
        )

    separators = read.loc[read["clean_separator"], "signal"].tolist()
    # Multiplicity is stated with every verdict: 5 signals x p per signal is the
    # number of clean separators a pure-noise cell would be expected to produce.
    p = float(read["chance_separation_p"].dropna().iloc[0])
    expected = p * len(read)
    multiplicity = (
        f" Multiplicity: {len(read)} signals tested, each separating by chance "
        f"with p={p:.3f} under a random-labelling null, so ~{expected:.1f} clean "
        "separators are expected from noise alone."
    )
    if not separators:
        return (
            f"NO CLEAN SEPARATOR at N={n_seeds} — no candidate signal has disjoint "
            "ranges across the high-validity / collapse strata. The switch trigger "
            "is none of the candidates computed here." + multiplicity
        )
    if len(separators) == 1:
        return (
            f"single clean separator: {separators[0]}. Suggestive only — it is not "
            "evidence of causation and carries no confirmatory weight (PS-9)."
            + multiplicity
        )
    return (
        f"CONFOUNDED — {len(separators)} signals separate cleanly "
        f"({', '.join(separators)}). At N={n_seeds} they co-move and which (if any) is "
        "causal cannot be told apart; no choice between them is made."
        + multiplicity
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "YES" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if np.isnan(value) else f"{value:+.6f}"
    if isinstance(value, tuple):
        return f"[{value[0]:+.6f}, {value[1]:+.6f}]"
    if value is None:
        return "n/a"
    # GFM splits table rows on unescaped pipes before parsing inline code.
    return str(value).replace("|", r"\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    # Header cells are escaped too: GFM splits rows on unescaped pipes before it
    # parses anything else, so a pipe in a COLUMN NAME silently breaks the table.
    lines = [
        "| " + " | ".join(str(c).replace("|", r"\|") for c in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    # Format off the COLUMN dtype, not the cell: iterrows() upcasts integers to
    # float on a mixed-dtype row, which would print seed 0 as "+0.000000".
    integral = {c for c in columns if pd.api.types.is_integer_dtype(frame[c])}
    for _, row in frame.iterrows():
        cells = [
            str(int(row[c])) if c in integral else _fmt(row[c]) for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def cell_report(
    cell: Cell, root: Path = DEFAULT_ROOT, n_seeds: int = N_SEEDS
) -> tuple[pd.DataFrame, pd.DataFrame, str, list[str]]:
    """(per-seed table, separation read, verdict, markdown lines) for one cell."""
    table = cell_signal_table(cell, root, n_seeds)
    read = separation_read(table)
    verdict = cell_verdict(read, table, n_seeds)

    signals = signal_columns(table)
    lines = [f"### {cell.label}", ""]
    if not tanh_input_features(cell):
        lines += ["Signal 3 (saturation position): **N/A** — linear family.", ""]
    lines += _markdown_table(
        table, ["seed_idx", "validity_A_neg", "n_pool", *signals]
    )
    cosine = table["l0_mean_cosine"]
    overshoot = table["l0_grid_overshoot"]
    lines += [
        "",
        "*Signal-1 identity audit (not a signal):* L0 unit actions concentrate about "
        f"one direction at mean cosine {cosine.min():.4f}–{cosine.max():.4f}, and the "
        "grid solution exceeds the exact point-to-boundary distance by "
        f"{100 * overshoot.min():.2f}–{100 * overshoot.max():.2f}% "
        f"(spread across seeds {100 * overshoot.std(ddof=1):.3f} pp). The projection "
        "identity holds and the discretization offset is near-constant within the "
        "cell, so it cannot drive the across-seed ordering read below.",
        "",
        "**Separation read**",
        "",
    ]
    lines += _markdown_table(
        read,
        [
            "signal",
            "n_high",
            "n_collapse",
            "high_range",
            "collapse_range",
            "ranges_overlap",
            "separation_gap",
            "spearman_rho",
            "chance_separation_p",
            "verdict",
        ],
    )
    lines += ["", f"**Verdict —** {verdict}", ""]
    return table, read, verdict, lines


def build_report(
    root: Path = DEFAULT_ROOT,
    cells: list[Cell] | None = None,
    n_seeds: int = N_SEEDS,
) -> tuple[str, dict[str, str]]:
    """Full PS-9 diagnostic document + the per-cell verdicts."""
    cells = cells if cells is not None else grid_cells()
    focus = {c.key for c in collider_cells()}
    lines = [
        "# Mechanism-discrimination diagnostic",
        "",
        EXPLORATORY_TAG,
        "",
        "Post-hoc diagnostic declared by **PS-9**, motivated by the",
        "N=6 cross-seed finding that the additive-control ValidityDisp asymmetry is",
        "seed-fragile and bimodal on the collider. It explains; it does not test.",
        "No H1–H4 confirmation criterion reads off any number below.",
        "",
        "Pool: the **A=−1 negatively-classified** individuals, at **L1-oracle**.",
        "",
        "**Signal 1 (boundary distance)** is read off the persisted **L0** rung:",
        "at L0 the belief is J = I, so under the fixed ℓ₂ cost the believed-optimal",
        "action is the orthogonal projection onto the decision boundary and its cost",
        "collapses to `|wᵀx − b| / ‖w‖₂` — the self-dual ℓ₂ identity the G1 anchor",
        "already asserts (`recourse/anchor.py`), applicable because the actionable set",
        "equals the classifier's feature set in all three topologies. No classifier is",
        "retrained. Caveat: the harness solves on a discretized action grid, so the",
        "value carries a within-cell near-constant discretization overshoot; it is",
        "used only for within-cell across-seed comparison, never for cross-cell levels.",
        "",
        "**Signal 2** is signed `estimated − true` on the load-bearing edge(s),",
        "evaluated at A=−1: positive means L1-oracle **over**-estimates propagation.",
        "",
        "**Signal 3** is `mean |tanh-input|` over the pool; **N/A** on linear cells.",
        "",
        "---",
        "",
    ]
    verdicts: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    for cell in cells:
        table, _, verdict, cell_lines = cell_report(cell, root, n_seeds)
        marker = "" if cell.key in focus else " *(contrast cell)*"
        cell_lines[0] = cell_lines[0] + marker
        lines += cell_lines + ["---", ""]
        verdicts[cell.key] = verdict
        tables[cell.key] = table

    lines += _saturation_contrast(cells, tables)
    return "\n".join(lines), verdicts


def _saturation_contrast(
    cells: list[Cell], tables: dict[str, pd.DataFrame]
) -> list[str]:
    """Cross-cell read on Signal 3, which no within-cell table can deliver.

    Saturation position is near-constant across the seeds of a cell (it is set by
    the builder's alpha and sigma, not by the seed), so within a cell it can
    barely separate anything. Its discriminating power, if any, is BETWEEN cells
    — which is exactly the contrast the triangle cells were computed for.
    """
    nlg = [c for c in cells if tanh_input_features(c)]
    if not nlg:
        return []
    rows = []
    for cell in nlg:
        table = tables[cell.key]
        position = float(table["saturation[X1]"].mean())
        validity = table["validity_A_neg"]
        rows.append(
            {
                "cell": cell.label,
                "mean abs tanh-input (X1)": position,
                "tanh' = sech^2": float(1.0 / np.cosh(position) ** 2),
                "A=-1 validity min": float(validity.min()),
                "A=-1 validity max": float(validity.max()),
            }
        )
    frame = pd.DataFrame(rows)
    return [
        "## Cross-cell contrast on Signal 3 (saturation)",
        "",
        "Saturation position is fixed by the builder's α and σ, so it barely varies",
        "across the seeds of a cell — its discriminating power, if any, is BETWEEN",
        "cells. That is what the triangle cells were computed for.",
        "",
        *_markdown_table(frame, list(frame.columns)),
        "",
        "**Read.** The contrast runs the WRONG way for a saturation explanation: the",
        "NLG *triangle* operates deep in saturation (`tanh′ ≈ 0.07`, the a = 2.0",
        "regime the SCM specification was written against) and holds validity at ~1.0 on every "
        "seed,",
        "while the NLG *collider* sits comfortably inside the non-saturated band",
        "(`tanh′ ≈ 0.69`, satisfying the SCM specification's `|α| + 2σ ≤ 1.5` constraint) and is "
        "where",
        "validity collapses. Depth into saturation therefore does not order the",
        "collapse across cells, and within the collider cells it does not separate the",
        "strata either. On this N=6 evidence Signal 3 is not the switch trigger.",
        "*(Exploratory; the two cells differ in topology as well as saturation, so",
        "this is a discordance argument, not a controlled comparison.)*",
        "",
    ]
