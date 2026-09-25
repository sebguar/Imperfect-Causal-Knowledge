"""Figure 1 — the cost degradation ladder (H1 read), at N=20 on the full grid.

Presentation layer ONLY — runs no experiment and recomputes no verdict quantity.
Reads one FROZEN artifact:

    <root>/summary/per_seed_by_group.csv

    python -m scripts.fig1_cost_ladder [--root results/cross_seed_N20] [--out DIR]

12 panels = 6 rows (topology x family) x 2 cols (regime). PS-8 forbids reading
levels across topologies, so the y-ranges are free per topology
(``_ylim_by_topology``).

POPULATION. The DV is realized cost over the **common-found three-way** set; the
arity word is carried on the axis and in the caption per PS-4's dynamic-arity rule.

DESCRIPTIVE SPREAD ONLY — no SE, no CI (PS-9; the inference stance).
"""

from __future__ import annotations

import argparse
import hashlib
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icknowledge.analysis.loading import (  # noqa: E402
    CONDITIONS,
    N20_ROOT,
    N_SEEDS_FULL,
    TOPOLOGIES,
    Cell,
    cell_frame,
    grid_cells,
    load_summary,
)

#: SHA-256 of the frozen N=20 by-group artifact this figure is licensed to render.
#: The N=20 tree is sealed (PS-1 ceiling), so pinning the digest makes "rendered
#: against the frozen inputs" checkable rather than asserted. Rebasing this
#: constant is a register act, not an edit.
BY_GROUP_SHA256 = "c761f9029043b01ebd0d1d3d3afcc14d4108817e9d8619e1ca003b8749dde6bb"

#: common-found primary population: realized cost over the common-found three-way set.
COST_COLUMN = "realized_cost_common_found_threeway"

#: [PS-4 dynamic-arity suffix] The denominator word carried on the axis and in the
#: caption. Derived from the column rather than typed twice, so the label and the
#: data can never disagree about which population is plotted.
DENOMINATOR_LABEL = COST_COLUMN.rsplit("_", 1)[-1]

#: Group -> colour and marker. Okabe-Ito, colour-blind safe.
#:
#: SEMANTIC SEPARATION. Neither hex may be one of the RESERVED
#: regime hues (#0072B2 effect-modifying, #D55E00 additive) used by fig4-fig7, and
#: neither glyph may be one of the RESERVED topology markers ({o, D, s}) — group is
#: not regime and is not topology, so it may not borrow either encoding. Marker
#: shape doubles the contrast so colour is never the sole distinguisher.
GROUP_STYLE = {
    -1.0: {"colour": "#CC79A7", "marker": "v", "label": "A = −1"},
    1.0: {"colour": "#009E73", "marker": "^", "label": "A = +1"},
}

#: Row order: the canonical `grid_cells()` enumeration, topology-major, so each
#: topology's two family rows are contiguous and the free-y grouping (PS-8) is
#: visually obvious. Same order as every analysis table this figure is read beside.
ROWS = [(topology, family) for topology in TOPOLOGIES for family in ("linear", "nlg")]
COLUMNS = ["additive", "effect_modifying"]

#: Group offset from the condition tick, then a deterministic within-group spread
#: so 20 coincident seed values stay countable. No RNG: the figure is
#: byte-reproducible across runs, and the spread is sized by the imported seed
#: count rather than a literal.
GROUP_OFFSET = 0.14
SEED_SPREAD = np.linspace(-0.075, 0.075, N_SEEDS_FULL)

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}
REGIME_DISPLAY = {"additive": "additive",
                  "effect_modifying": "effect-modifying"}

#: Ladder order on the x axis, most-knowledge-first. Plotted values are
#: unchanged; only each condition's x position follows this order.
DISPLAY_CONDITIONS = tuple(reversed(CONDITIONS))

CAPTION_TEMPLATE = """Figure 1. Realized recourse cost across the knowledge ladder
(L2 -> L1-oracle -> L0), by protected group, for each of the {n_cells}
(topology x family x regime) cells of the full grid; small markers are the
{n_seeds} per-seed cell means and the connecting line joins their seed averages; the
vertical whisker at each average is +/-1 descriptive SD (ddof = 1).

POPULATION: the common-found `{denominator}` set — individuals for whom a
feasible action is returned in every one of the three conditions. The arity word
is carried on the value axis, in the caption and nowhere abbreviated, because the
four-way L1-discovered series (PS-4) sits on a different denominator and the two
are not interchangeable. On this grid the three-way common-found read coincides
EXACTLY with the all-eligible read (max |difference| = {equality}), because the
exhaustive grid search always returns a feasible action; the primary column is
plotted regardless, since the provenance is the point.

PS-8: levels are NOT comparable across topologies (different action-space
dimension, induced marginal variances and classifier-boundary geometry), so each
topology's two rows carry one shared y-range, the three topologies are independent
of one another, and any cross-topology reading is pattern-only. Within a topology
the vertical scan (family) and the horizontal scan (regime) are legitimate.

CHAIN ROWS ARE A FIRST APPEARANCE. The chain was added to the grid after the
early look, so its four cells appear in an H1 figure here for the first
time and are uncorroborated by any prior look — there is no earlier rendering to
agree or disagree with them (the same flag LD-3 carries on the H1 verdict).

N = {n_seeds} seeds per cell (PS-1's pinned ceiling). All {n_seeds} per-seed values
are shown rather than summarized. There are no standard errors or confidence
intervals on this figure: PS-9 re-tasks SD as descriptive
dispersion, and a formal interval would over-promise precision this structure
lacks (the inference stance)."""


def assert_frozen_digest(path: Path, expected: str | None = None) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact.

    Read from the module constant at CALL time (not as a default argument) so a
    test can substitute a fixture's digest without the guard silently passing.
    """
    expected = expected if expected is not None else BY_GROUP_SHA256
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"FROZEN-INPUT DIGEST MISMATCH for {path}: expected SHA-256 {expected}, "
            f"got {digest}. This figure renders the sealed N=20 grid and may not be "
            "drawn against a modified input — re-check the tree against the PS-1 "
            "record rather than rebasing this constant."
        )
    return digest


def common_found_cost_equality(by_group: pd.DataFrame) -> float:
    """max |cost(common-found) − cost(all-eligible)| — the H1-read no-op, re-checked.

    The caption asserts the two populations coincide exactly; the number behind
    that assertion is recomputed here rather than copied, so the claim cannot go
    stale if the grid is ever re-run.
    """
    diff = (by_group[COST_COLUMN] - by_group["realized_mean_cost"]).abs()
    return float(diff.max())


def _panel_data(by_group: pd.DataFrame, cell: Cell) -> dict[float, dict[str, np.ndarray]]:
    """Per-seed cost vectors for one panel, keyed group -> condition -> values."""
    sub = cell_frame(by_group, cell)
    out: dict[float, dict[str, np.ndarray]] = {}
    for group in GROUP_STYLE:
        per_condition = {}
        for condition in CONDITIONS:
            rows = sub[(sub["condition"] == condition) & (sub["group"] == group)]
            values = rows.sort_values("seed_idx")[COST_COLUMN].to_numpy(dtype=float)
            if values.size != N_SEEDS_FULL:
                raise ValueError(
                    f"{cell.label} / {condition} / A={group:+.0f}: expected "
                    f"{N_SEEDS_FULL} seeds (PS-1 pinned ceiling), got {values.size}"
                )
            per_condition[condition] = values
        out[group] = per_condition
    return out


def _ylim_by_topology(panels: dict[Cell, dict]) -> dict[str, tuple[float, float]]:
    """PS-8 encoding: one shared y-range per topology, independent between them.

    Sharing a single range across topologies would invite exactly the level
    comparison PS-8 forbids; sharing within a topology keeps the family and regime
    scans (which ARE allowed) on a common scale. Enumerated from the panels
    actually present, so adding a topology cannot leave one silently unscaled.
    """
    limits = {}
    for topology in sorted({cell.topology for cell in panels}):
        values = np.concatenate(
            [
                series
                for cell, data in panels.items()
                if cell.topology == topology
                for per_condition in data.values()
                for series in per_condition.values()
            ]
        )
        low, high = float(values.min()), float(values.max())
        pad = 0.08 * (high - low)
        limits[topology] = (max(0.0, low - pad), high + pad)
    return limits


def build_figure(by_group: pd.DataFrame) -> plt.Figure:
    cells = {cell: _panel_data(by_group, cell) for cell in grid_cells()}
    limits = _ylim_by_topology(cells)
    x = np.arange(len(DISPLAY_CONDITIONS), dtype=float)

    figure, axes = plt.subplots(
        len(ROWS), len(COLUMNS), figsize=(6.3, 11.6), sharex=True
    )
    for row, (topology, family) in enumerate(ROWS):
        for column, regime in enumerate(COLUMNS):
            axis = axes[row][column]
            data = cells[Cell(topology, family, regime)]
            for group, style in GROUP_STYLE.items():
                base = x + (GROUP_OFFSET if group > 0 else -GROUP_OFFSET)
                means = [float(data[group][c].mean()) for c in DISPLAY_CONDITIONS]
                axis.plot(
                    base, means, color=style["colour"], linewidth=1.1,
                    zorder=2, solid_capstyle="round",
                )
                sds = [
                    float(data[group][c].std(ddof=1)) for c in DISPLAY_CONDITIONS
                ]
                axis.errorbar(
                    base, means, yerr=sds,
                    color=style["colour"], ecolor=style["colour"],
                    elinewidth=0.9, capsize=2.0, fmt="none", zorder=4,
                )
                for index, condition in enumerate(DISPLAY_CONDITIONS):
                    axis.plot(
                        base[index] + SEED_SPREAD,
                        data[group][condition],
                        linestyle="none",
                        marker=style["marker"],
                        markersize=2.6,
                        markerfacecolor=style["colour"],
                        markeredgecolor="white",
                        markeredgewidth=0.25,
                        alpha=0.85,
                        zorder=3,
                    )
            axis.set_title(
                f"{topology} · {FAMILY_DISPLAY[family]} · {REGIME_DISPLAY[regime]}",
                fontsize=8, pad=4,
            )
            axis.set_ylim(*limits[topology])
            axis.set_xlim(-0.5, len(DISPLAY_CONDITIONS) - 0.5)
            axis.set_xticks(x)
            axis.set_xticklabels(DISPLAY_CONDITIONS, fontsize=7)
            axis.tick_params(axis="y", labelsize=8)
            axis.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if column == 1:
                axis.tick_params(axis="y", labelleft=False)
            else:
                axis.set_ylabel(
                    "realized cost\n(common-found, three-way)",
                    fontsize=8,
                )
            if row == len(ROWS) - 1:
                axis.set_xlabel("knowledge condition", fontsize=9)

    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker=style["marker"], color=style["colour"],
            markersize=5, label=style["label"],
        )
        for style in GROUP_STYLE.values()
    ] + [
        plt.Line2D([], [], color="0.35", linewidth=1.1, label="seed mean"),
        plt.Line2D([], [], linestyle="none", marker="|", color="0.35", markersize=10,
                   markeredgewidth=2.0, label="mean ± 1 SD (descriptive)"),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.018),
    )
    figure.tight_layout()
    return figure


def _wrap(text: str) -> str:
    """Reflow the caption to a fixed width — templating leaves ragged lines."""
    # break_on_hyphens=False: the register's vocabulary is full of hyphenated
    # terms that must survive as single tokens — "REGISTER-LOGGED",
    # "non-gating", "effect-modifying", "L1-oracle". textwrap's default
    # splits them across a line boundary, which both hurts readability and
    # makes any fragment check depend on where the wrap happened to land.
    return textwrap.fill(
        " ".join(text.split()), width=78, break_on_hyphens=False
    ) + "\n"


def _style() -> None:
    """Self-contained styling — no rcParams are inherited from another script."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,  # embed TrueType, not Type 3 — LaTeX-friendly
            "savefig.bbox": "tight",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=N20_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)

    assert_frozen_digest(args.root / "summary" / "per_seed_by_group.csv")
    by_group = load_summary("per_seed_by_group", args.root)

    _style()
    figure = build_figure(by_group)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"fig1_cost_ladder.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    # Every number in the caption is read off the data at render time, so the
    # caption cannot drift from the figure it describes.
    text = CAPTION_TEMPLATE.format(
        equality=f"{common_found_cost_equality(by_group):.3e}",
        n_cells=len(grid_cells()),
        n_seeds=N_SEEDS_FULL,
        denominator=DENOMINATOR_LABEL,
    )
    caption = out / "fig1_cost_ladder.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
