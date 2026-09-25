"""Figure 3 — the detectability-gate record (PS-2 discharge), two-populations layout.

Presentation layer ONLY. Runs no experiment and RE-EVALUATES NO GATE. Reads two
FROZEN artifacts:

    <root>/summary/per_seed_by_cell.csv               the 20-seed dispersion strip
    <root>/summary/detectability_gate_provenance.csv  the frozen 6-seed gate

    python -m scripts.fig3_detectability [--root results/cross_seed_N20] [--out DIR]

TWO POPULATIONS ON ONE PANEL — the chain SCM specification, pinned before this
script was written. The figure PLOTS the full 20-seed Δ_cost(L2) strip (the
dispersion the grid actually produced) and ANNOTATES the gate computed on the
seed-0..5 prefix (``n_seeds_evaluated_for_gate = 6``). These are different
populations and are labelled as such ON THE FIGURE, not only in the caption: a
reader who took the annotated |mean|/sd to describe the plotted cloud would be
reading a 6-seed statistic off a 20-seed strip.

Every gate quantity drawn is READ from the provenance CSV, never recomputed from
the plotted strip; ``assert_gate_is_not_derived`` enforces that structurally (PS-2).

DESCRIPTIVE SPREAD ONLY — no SE, no CI (PS-9; the inference stance).
"""

from __future__ import annotations

import argparse
import hashlib
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icknowledge.analysis.loading import (  # noqa: E402
    N20_ROOT,
    N_SEEDS_FULL,
    TOPOLOGIES,
    Cell,
    cell_frame,
    grid_cells,
    load_summary,
)

#: SHA-256 of the two frozen inputs. The strip is the sealed N=20 grid; the
#: provenance record is the frozen-facts gate consolidation. Rebasing either
#: constant is a register act, not an edit.
BY_CELL_SHA256 = "c00e54c3eea973e73f96fd2fd7257697082386545a10132734d76f50d8855b3f"
GATE_PROVENANCE_SHA256 = "8402056a9602282af69e70c64d5b3ea3ba098fd83f99020d8ced804caa6154c0"

GATE_PROVENANCE_NAME = "detectability_gate_provenance.csv"

#: The strip column. Δ_cost here is the SIGNED group difference
#: mean(cost | A=−1) − mean(cost | A=+1) (signed primary); |Δ_cost| is never
#: plotted, because sign-consistency is exactly what gate condition (i) reads on.
STRIP_COLUMN = "Δ_cost"
COMMON_FOUND_COLUMN = "Δ_cost_common_found_threeway"
CONDITION = "L2"

#: The only columns this figure is permitted to display from the provenance record.
GATE_FIELDS = ("mean", "sd", "abs_mean_over_sd", "gate_pass", "n_seeds_evaluated_for_gate")

#: Regime -> colour, IDENTICAL literals to fig4-fig7. Regime is the dimension these
#: hexes encode everywhere in this project, and it is the dimension they encode
#: here, so the reserved pair is reused rather than avoided (a hex may
#: not mean something DIFFERENT, not "may not recur").
REGIME_COLOUR = {"effect_modifying": "#0072B2", "additive": "#D55E00"}

#: Topology -> marker, the reserved glyphs. The panel already carries the topology,
#: so the marker is redundant with it BY DESIGN — that is what keeps the glyph
#: meaning topology and nothing else.
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}

#: Family -> fill. Filled = linear, hollow = nonlinear-Gaussian; the fig6/fig7
#: grammar, reused unchanged.
FAMILY_FILLED = {"linear": True, "nlg": False}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}
REGIME_DISPLAY = {"additive": "additive",
                  "effect_modifying": "effect-modifying"}

SEED_SPREAD = np.linspace(-0.22, 0.22, N_SEEDS_FULL)

CAPTION_TEMPLATE = """Figure 3. Per-seed signed Δ_cost at L2 — the group difference
mean(cost | A = −1) − mean(cost | A = +1) — for each of the {n_cells} cells of the
full grid, one panel per topology, with the Δ_cost = 0 reference line.

TWO POPULATIONS, LABELLED AS SUCH (the chain SCM specification). The plotted
cloud is the DISPERSION STRIP: all {n_seeds} seeds of the sealed N = 20 grid. The
annotated |mean|/sd above each column is the GATE: the PS-2
detectability gate as evaluated on seeds 0-{last_gate_seed} only
(n_seeds_evaluated_for_gate = {gate_seeds}), read from
`{provenance}` and never recomputed from the strip. PS-1 forecloses re-evaluating
the gate on a growing seed count, because that would reintroduce the
stopping-rule freedom the staged 6->20 design exists to prevent; a 6-seed
statistic sitting beside a 20-seed cloud is that discipline working, not a stale
number. The two are drawn in different registers — grey annotation for the gate,
coloured markers for the strip — and the panel legend names both.

THE RECORD. The gate passes in {n_pass}/{n_cells} cells: sign(Δ_cost(L2)) is
consistent across all {gate_seeds} early-look seeds in every cell and |mean|
exceeds the seed SD everywhere, with a minimum ratio of {min_ratio} at
{min_cell}. This discharges PS-2 for all three topologies — triangle,
collider (the SCM specification) and chain (the chain SCM specification).

The strip's own three-way common-found variant is cross-checked against Δ_cost at
render time and must agree exactly (the common-found no-op); a divergence aborts the
figure rather than being drawn.

PS-8: Δ_cost levels are not comparable across topologies (different action-space
dimension, induced marginal variances and classifier-boundary geometry), so the
three panels carry independent y-ranges and the cross-panel reading is
pattern-only. There are no standard errors or confidence intervals on this
figure: PS-9 re-tasks SD as descriptive dispersion, and a
formal interval would over-promise precision this structure lacks (the inference stance)."""


def assert_frozen_digest(path: Path, expected: str) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"FROZEN-INPUT DIGEST MISMATCH for {path}: expected SHA-256 {expected}, "
            f"got {digest}. This figure renders sealed artifacts and may not be drawn "
            "against a modified input — re-check the tree against the PS-1 / PS-2 "
            "record rather than rebasing this constant."
        )
    return digest


def strip_series(by_cell: pd.DataFrame, cell: Cell) -> np.ndarray:
    """The 20-seed Δ_cost(L2) dispersion strip for one cell. NOT a gate quantity.

    Nothing gate-like is computed from this vector anywhere in the module — see
    :func:`assert_gate_is_not_derived`, which is what makes that checkable.
    """
    sub = cell_frame(by_cell, cell)
    rows = sub[sub["condition"] == CONDITION].sort_values("seed_idx")
    values = rows[STRIP_COLUMN].to_numpy(dtype=float)
    if values.size != N_SEEDS_FULL:
        raise ValueError(
            f"{cell.label}: expected {N_SEEDS_FULL} seeds at {CONDITION} (PS-1 pinned "
            f"ceiling), got {values.size}"
        )
    # The common-found population: on this grid the common-found population is the same population,
    # and
    # that is checked, not trusted.
    common = rows[COMMON_FOUND_COLUMN].to_numpy(dtype=float)
    if not np.allclose(values, common, rtol=0, atol=0):
        raise ValueError(
            f"{cell.label}: Δ_cost and its common-found variant differ — the common-found "
            f"population "
            "no-op no longer holds, so this figure's provenance note is wrong."
        )
    return values


def load_gate(root: Path) -> pd.DataFrame:
    """The FROZEN gate record — the only source of a gate quantity in this figure."""
    path = root / "summary" / GATE_PROVENANCE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Per the chain SCM specification this figure reads "
            "its gate values from the frozen provenance record and may not fall back "
            "to recomputing them from the plotted strip — that would re-evaluate the "
            "PS-1-frozen gate. Emit the record with run_t4_analysis --reports t4c."
        )
    assert_frozen_digest(path, GATE_PROVENANCE_SHA256)
    gate = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in GATE_FIELDS if column not in gate.columns]
    if missing:
        raise KeyError(f"{path} has no {missing} column(s) — not a gate provenance record")
    return gate


def gate_row(gate: pd.DataFrame, cell: Cell) -> pd.Series:
    """The frozen gate record for one cell, joined on topology x family x regime."""
    match = gate[
        (gate["topology"] == cell.topology)
        & (gate["family"] == cell.family)
        & (gate["regime"] == cell.regime)
    ]
    if len(match) != 1:
        raise LookupError(
            f"{cell.label}: expected exactly 1 frozen gate row, got {len(match)}"
        )
    return match.iloc[0]


def assert_gate_is_not_derived(gate: pd.DataFrame, by_cell: pd.DataFrame) -> None:
    """FAIL if any displayed gate quantity could have come from the 20-seed strip.

    Two independent checks, per the chain SCM specification:

    1. **Seed count.** Every provenance row must declare the 6-seed prefix; a
       record stamped 20 is a re-evaluated gate wearing the frozen record's
       filename.
    2. **Numeric separation.** The gate's |mean|/sd must NOT equal what the
       plotted strip would produce. The strip statistic is computed HERE only to
       prove it is not what gets displayed, and is discarded.
    """
    stamped = set(gate["n_seeds_evaluated_for_gate"].astype(int))
    if stamped != {6}:
        raise ValueError(
            f"gate provenance declares n_seeds_evaluated_for_gate = {sorted(stamped)}; "
            "PS-1 / PS-2 define the gate on the FIRST-6-SEED early look. A record "
            "stamped with the strip's seed count is a re-evaluated gate, not the "
            "frozen one, and must not be rendered."
        )
    for cell in grid_cells():
        values = strip_series(by_cell, cell)
        strip_ratio = abs(float(values.mean())) / float(values.std(ddof=1))
        frozen_ratio = float(gate_row(gate, cell)["abs_mean_over_sd"])
        if np.isclose(strip_ratio, frozen_ratio, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"{cell.label}: the displayed gate ratio {frozen_ratio:.6f} equals the "
                f"ratio the 20-seed strip produces ({strip_ratio:.6f}). Either the "
                "provenance record was regenerated from the full grid or this figure "
                "is computing the gate from the strip — both re-evaluate the "
                "PS-1-frozen gate (PS-2)."
            )


def build_figure(by_cell: pd.DataFrame, gate: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(len(TOPOLOGIES), 1, figsize=(6.3, 7.4))
    for axis, topology in zip(axes, TOPOLOGIES, strict=True):
        cells = [cell for cell in grid_cells() if cell.topology == topology]
        marker = TOPOLOGY_MARKER[topology]
        strips = [strip_series(by_cell, cell) for cell in cells]

        for position, (cell, values) in enumerate(zip(cells, strips, strict=True)):
            colour = REGIME_COLOUR[cell.regime]
            filled = FAMILY_FILLED[cell.family]
            axis.plot(
                position + SEED_SPREAD,
                values,
                linestyle="none",
                marker=marker,
                markersize=3.6,
                markerfacecolor=colour if filled else "white",
                markeredgecolor=colour,
                markeredgewidth=0.8,
                alpha=0.9,
                zorder=3,
            )

        # The gate band: the frozen 6-seed mean, drawn in a DIFFERENT register
        # (grey, dashed) from the coloured strip so the two populations are
        # separable at a glance and not only in the caption (the chain SCM specification ext.
        # clause 1).
        for position, cell in enumerate(cells):
            row = gate_row(gate, cell)
            axis.plot(
                [position - 0.32, position + 0.32],
                [float(row["mean"])] * 2,
                color="0.25", linewidth=1.3, linestyle=(0, (3, 1.6)), zorder=4,
            )

        values = np.concatenate(strips)
        high = float(values.max())
        axis.set_ylim(-0.10 * high, high * 1.30)
        axis.axhline(0.0, color="0.35", linewidth=1.0, zorder=2)
        for position, cell in enumerate(cells):
            row = gate_row(gate, cell)
            axis.annotate(
                f"|μ|/σ = {float(row['abs_mean_over_sd']):.2f}\n"
                f"(gate: seeds 0–5, n = {int(row['n_seeds_evaluated_for_gate'])})",
                xy=(position, high * 1.15),
                ha="center", va="center", fontsize=6.2, color="0.25",
                linespacing=1.25,
            )
        axis.set_xticks(range(len(cells)))
        axis.set_xticklabels(
            [
                f"{FAMILY_DISPLAY[cell.family]}"
                f"\n{REGIME_DISPLAY[cell.regime]}"
                for cell in cells
            ],
            fontsize=7,
        )
        axis.set_xlim(-0.6, len(cells) - 0.4)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_title(
            f"{topology} — strip: all {N_SEEDS_FULL} seeds · gate: seeds 0–5 "
            "· free y-axis (PS-8)",
            fontsize=8, pad=5,
        )
        axis.set_ylabel("signed Δ_cost at L2", fontsize=8)
        axis.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[-1].set_xlabel("cell (family · regime)", fontsize=9)

    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", color=REGIME_COLOUR[regime],
            markersize=5, label=f"strip, {regime.replace('_', '-')} (20 seeds)",
        )
        for regime in ("additive", "effect_modifying")
    ] + [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor="0.35", markersize=5,
            label="hollow = nonlinear-Gaussian",
        ),
        plt.Line2D(
            [], [], color="0.25", linewidth=1.3, linestyle=(0, (3, 1.6)),
            label="frozen gate mean (seeds 0–5, PS-1)",
        ),
        plt.Line2D(
            [], [], linestyle="none", marker="none",
            label="gate threshold: |μ|/σ > 1",
        ),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.045),
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
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def caption_facts(gate: pd.DataFrame) -> dict[str, object]:
    """Every caption number, read from the FROZEN provenance record.

    The pass count, the minimum ratio and the cell it belongs to are read here and
    nowhere else: they are gate quantities, so the strip may not supply them.
    """
    weakest = gate.loc[gate["abs_mean_over_sd"].idxmin()]
    gate_seeds = int(gate["n_seeds_evaluated_for_gate"].iloc[0])
    return {
        "n_cells": int(len(gate)),
        "n_pass": int(gate["gate_pass"].astype(bool).sum()),
        "n_seeds": N_SEEDS_FULL,
        "gate_seeds": gate_seeds,
        "last_gate_seed": gate_seeds - 1,
        "min_ratio": f"{float(weakest['abs_mean_over_sd']):.6f}",
        "min_cell": (
            f"{weakest['topology']} · {FAMILY_DISPLAY[weakest['family']]} · "
            f"{REGIME_DISPLAY[weakest['regime']]}"
        ),
        "provenance": GATE_PROVENANCE_NAME,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=N20_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)

    assert_frozen_digest(args.root / "summary" / "per_seed_by_cell.csv", BY_CELL_SHA256)
    by_cell = load_summary("per_seed_by_cell", args.root)
    gate = load_gate(args.root)
    assert_gate_is_not_derived(gate, by_cell)

    _style()
    figure = build_figure(by_cell, gate)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"fig3_detectability.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    text = CAPTION_TEMPLATE.format(**caption_facts(gate))
    caption = out / "fig3_detectability.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
