"""Figure 2 — the A=−1 validity distribution at L1-oracle (H2 read), N=20.

Presentation layer ONLY — runs no experiment. Reads one FROZEN artifact:

    <root>/summary/per_seed_by_group.csv

    python -m scripts.fig2_validity_gap [--root results/cross_seed_N20] [--out DIR]

No per-cell centre marker is drawn: the SHAPE is the finding.

DESCRIPTIVE READ ONLY — no SE, no CI (PS-9; the inference stance).
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
    Cell,
    cell_frame,
    grid_cells,
    load_summary,
)

#: SHA-256 of the frozen N=20 by-group artifact. See fig1 for the rationale.
BY_GROUP_SHA256 = "c761f9029043b01ebd0d1d3d3afcc14d4108817e9d8619e1ca003b8749dde6bb"

#: H2's live signal is validity, not cost: PS-6 records that Gap_c on cost is
#: identically zero at L1-oracle and L2, so a Gap_c gradient is NOT a H2 signal
#: and is deliberately absent from this figure.
CONDITION = "L1-oracle"
FOCUS_GROUP = -1.0
VALIDITY_COLUMN = "realized_validity_rate"

#: (topology, family) -> style, one entry per block on the x-axis.
#: Colour encodes FAMILY; marker shape encodes topology via the reserved
#: TOPOLOGY_MARKER glyphs, which mean the same thing in fig4-fig7. Neither family
#: hue is one of the RESERVED regime hues (#0072B2 / #D55E00) or one of fig1's
#: group hues: no hex in this project means two different things.
FAMILY_COLOUR = {"linear": "#E69F00", "nlg": "#56B4E9"}
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}
BLOCK_STYLE = {
    (topology, family): {
        "colour": FAMILY_COLOUR[family],
        "marker": TOPOLOGY_MARKER[topology],
    }
    for topology in TOPOLOGY_MARKER
    for family in FAMILY_COLOUR
}

#: Deterministic within-cell spread — no RNG, so the figure is reproducible and
#: the 20 seeds stay countable where they pile up at exactly 1.0 (many do).
SEED_SPREAD = np.linspace(-0.30, 0.30, N_SEEDS_FULL)

#: Twelve cells share one text-block width, so the regime name is written in full
#: on the tick label and the ticks are rotated rather than overrun their
#: neighbours. The cell ORDER is untouched — only the rendering of the name is.
REGIME_LABEL = {"additive": "additive", "effect_modifying": "effect-modifying"}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}
REGIME_DISPLAY = {"additive": "additive",
                  "effect_modifying": "effect-modifying"}

#: A seed is "collapsed" when the A=−1 group's recourse essentially never flips the
#: true classifier. The ValidityDisp read uses the same reading; the threshold only
#: DESCRIBES the figure and never enters a statistic or a verdict.
COLLAPSE_THRESHOLD = 0.1

#: Below this a seed is in a cell's lower tail rather than its ≈1.0 mode.
HIGH_BAND_FLOOR = 0.9

CAPTION_TEMPLATE = """Figure 2. Per-seed REALIZED VALIDITY RATE of the A = −1 group
at L1-oracle, one marker per seed, for each of the {n_cells} cells in a fixed
(topology, family, regime) order — cells are never re-ordered by value, so the
comparison reads against the labelled axis. This figure plots ONE GROUP'S LEVEL:
it is neither
ValidityGap_g(c) (believed − realized, per group) nor ValidityDisp(c) (realized
A = −1 − realized A = +1); no cross-group difference is drawn. Colour carries
the functional family and marker shape carries the topology, which the axis
order also encodes.

WHAT THE BANDS DO. Every cell has a mode at validity ≈ 1.0, and every one of the
{n_cells} cells reaches 1.000 on at least one seed. What differs is the LOWER
TAIL. {below_high_total} of {n_total} (cell, seed) observations fall below
{high_floor}, and {collapse_total} fall below {threshold} — a near-total failure
of recourse for that group on that seed. Those {collapse_total} collapse
observations are confined to {collapse_cells} of the {n_cells} cells:
{collapse_detail}. Per topology, the count of observations below {high_floor} is
{below_high_by_topology}, and the lowest single value is {floor_by_topology}.

NO TOPOLOGY IS FREE OF THE LOWER TAIL. The triangle is the mildest — no collapse
seed anywhere, and {triangle_clean_n} of its {triangle_cells} cells hold every
seed at or above {triangle_clean_floor} — but its linear additive cell still puts
{triangle_low_n} seeds between {triangle_low_range}, so "no switch on the
triangle" is not what these data show.

THE CHAIN, FIRST APPEARANCE. The chain was added to the grid after the early look and has never
appeared in a validity figure; its read here is uncorroborated by any prior look.
Its low-tail count {chain_rank_phrase} ({chain_below_high} of {chain_total}
observations below {high_floor}), its linear additive cell is the single most
tail-heavy on the grid ({chain_worst_n}/{n_seeds} seeds below {high_floor}), and
its two nonlinear-Gaussian cells contain collapse seeds.

WHY NO MEAN OR SD IS DRAWN. At the N = 6 early look this distribution read as
near-binary with a sparse middle. At N = 20 the middle is POPULATED
({mid_band_total} observations between {threshold} and {high_floor}), so the
accurate description is a heavy mode at ≈ 1.0 with a long left tail rather than
two clean modes. The conclusion is unchanged: a mean ± SD would report a centre
where little mass sits, so the moments are omitted and every seed is shown.

Validity is a proportion and so is scale-free across topologies, but the
cross-topology reading remains pattern-only per PS-8. N = {n_seeds} seeds per cell
(PS-1's pinned ceiling). This figure is descriptive: H2's verdict is recorded at
LD-3 and in t4b_validity_gap.md, not here. There are no standard errors or
confidence intervals on this figure: PS-9 re-tasks SD as
descriptive dispersion, and a formal interval would over-promise precision this
structure lacks (the inference stance)."""


def assert_frozen_digest(path: Path, expected: str | None = None) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact."""
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


def validity_by_cell(by_group: pd.DataFrame) -> dict[Cell, np.ndarray]:
    """Per-seed A=−1 validity at L1-oracle, one vector per cell, seed-ordered."""
    out = {}
    for cell in grid_cells():
        sub = cell_frame(by_group, cell)
        rows = sub[(sub["condition"] == CONDITION) & (sub["group"] == FOCUS_GROUP)]
        values = rows.sort_values("seed_idx")[VALIDITY_COLUMN].to_numpy(dtype=float)
        if values.size != N_SEEDS_FULL:
            raise ValueError(
                f"{cell.label}: expected {N_SEEDS_FULL} seeds at {CONDITION} "
                f"(PS-1 pinned ceiling), got {values.size}"
            )
        out[cell] = values
    return out


def caption_facts(data: dict[Cell, np.ndarray]) -> dict[str, object]:
    """Every descriptive number in the caption, recomputed from the plotted data.

    Nothing here is copied from the ValidityDisp read's table or from an earlier
    caption: the narrative is only allowed to say what this function returns.
    """
    everything = np.concatenate(list(data.values()))
    collapse_cells = {
        cell: int((values < COLLAPSE_THRESHOLD).sum())
        for cell, values in data.items()
        if (values < COLLAPSE_THRESHOLD).any()
    }
    by_topology = {
        topology: np.concatenate(
            [values for cell, values in data.items() if cell.topology == topology]
        )
        for topology in TOPOLOGY_MARKER
    }
    triangle_low = np.concatenate(
        [
            values[values < HIGH_BAND_FLOOR]
            for cell, values in data.items()
            if cell.topology == "triangle"
        ]
    )
    chain_worst = max(
        (int((values < HIGH_BAND_FLOOR).sum()) for cell, values in data.items()
         if cell.topology == "chain"),
        default=0,
    )
    mid = everything[
        (everything >= COLLAPSE_THRESHOLD) & (everything < HIGH_BAND_FLOOR)
    ]

    # "largest low-tail count" is only sayable if it IS the largest. The chain and
    # the collider tie on this grid, and a caption that claimed otherwise would be
    # wrong in exactly the way the N=6 narrative was — so the comparison is
    # computed and the phrase follows the numbers.
    below = {
        topology: int((values < HIGH_BAND_FLOOR).sum())
        for topology, values in by_topology.items()
    }
    chain_below = below["chain"]
    tied = sorted(t for t, n in below.items() if n == chain_below and t != "chain")
    if chain_below < max(below.values()):
        rank_phrase = "is not the largest of the three topologies"
    elif tied:
        rank_phrase = f"ties the {' and '.join(tied)} for the largest of the three"
    else:
        rank_phrase = "is the largest of the three topologies"

    # The triangle's "clean" cells: those whose every seed sits in the ≈1.0 mode.
    triangle_cells = [
        values for cell, values in data.items() if cell.topology == "triangle"
    ]
    clean = [values for values in triangle_cells if (values >= HIGH_BAND_FLOOR).all()]
    return {
        "n_cells": len(data),
        "n_seeds": N_SEEDS_FULL,
        "n_total": int(everything.size),
        "threshold": f"{COLLAPSE_THRESHOLD:.1f}",
        "high_floor": f"{HIGH_BAND_FLOOR:.1f}",
        "below_high_total": int((everything < HIGH_BAND_FLOOR).sum()),
        "collapse_total": int((everything < COLLAPSE_THRESHOLD).sum()),
        "collapse_cells": len(collapse_cells),
        "collapse_detail": "; ".join(
            f"{cell.topology} · {FAMILY_DISPLAY[cell.family]} · "
            f"{REGIME_DISPLAY[cell.regime]} {count}/{N_SEEDS_FULL}"
            for cell, count in collapse_cells.items()
        ),
        "below_high_by_topology": ", ".join(
            f"{topology} {int((values < HIGH_BAND_FLOOR).sum())}/{values.size}"
            for topology, values in by_topology.items()
        ),
        "floor_by_topology": ", ".join(
            f"{topology} {values.min():.3f}" for topology, values in by_topology.items()
        ),
        "triangle_low_n": int(triangle_low.size),
        "triangle_low_range": (
            f"{triangle_low.min():.3f} and {triangle_low.max():.3f}"
            if triangle_low.size
            else "n/a"
        ),
        "chain_below_high": chain_below,
        "chain_total": int(by_topology["chain"].size),
        "chain_worst_n": chain_worst,
        "chain_rank_phrase": rank_phrase,
        "triangle_cells": len(triangle_cells),
        "triangle_clean_n": len(clean),
        "triangle_clean_floor": (
            f"{min(float(values.min()) for values in clean):.2f}" if clean else "n/a"
        ),
        "mid_band_total": int(mid.size),
    }


def build_figure(by_group: pd.DataFrame) -> plt.Figure:
    data = validity_by_cell(by_group)
    cells = list(data)

    figure, axis = plt.subplots(figsize=(6.3, 4.6))
    seen = set()
    for position, cell in enumerate(cells):
        style = BLOCK_STYLE[(cell.topology, cell.family)]
        label = f"{cell.topology} · {cell.family}"
        axis.plot(
            position + SEED_SPREAD,
            data[cell],
            linestyle="none",
            marker=style["marker"],
            markersize=3.4,
            markerfacecolor=style["colour"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.9,
            zorder=3,
            label=label if label not in seen else None,
        )
        seen.add(label)

    # Topology dividers, COMPUTED from the cell order rather than hardcoded, so a
    # re-ordering or a fourth topology cannot leave a divider in the wrong place.
    for boundary in _topology_boundaries(cells):
        axis.axvline(
            boundary, color="0.55", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1
        )
    for reference in (0.0, 1.0):
        axis.axhline(reference, color="0.85", linewidth=0.7, zorder=0)
    axis.axhline(
        HIGH_BAND_FLOOR, color="0.75", linewidth=0.6, linestyle=(0, (1, 2)), zorder=0
    )

    axis.set_xticks(range(len(cells)))
    axis.set_xticklabels(
        [
            f"{cell.topology}\n{FAMILY_DISPLAY[cell.family]}"
            f"\n{REGIME_LABEL[cell.regime]}"
            for cell in cells
        ],
        fontsize=6,
        rotation=90,
    )
    axis.set_xlim(-0.7, len(cells) - 0.3)
    axis.set_ylim(-0.05, 1.05)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.tick_params(axis="y", labelsize=8)
    axis.set_ylabel("realized validity, A = −1, at L1-oracle\n(all-eligible)", fontsize=9)
    axis.set_xlabel("cell (topology · family · regime)", fontsize=9)
    axis.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    handles = [
        plt.Line2D(
            [], [], linestyle="none",
            marker=BLOCK_STYLE[(topology, family)]["marker"],
            color=BLOCK_STYLE[(topology, family)]["colour"],
            markersize=5, label=f"{topology} · {FAMILY_DISPLAY[family]}",
        )
        for topology in TOPOLOGY_MARKER
        for family in FAMILY_COLOUR
    ] + [
        plt.Line2D([], [], color="0.75", linewidth=0.6, linestyle=(0, (1, 2)),
                   label="high-validity band floor (0.9)"),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.17),
    )
    figure.tight_layout()
    return figure


def _topology_boundaries(cells: list[Cell]) -> list[float]:
    """x positions where the topology changes — computed, never hardcoded."""
    return [
        position - 0.5
        for position, (previous, cell) in enumerate(
            zip(cells[:-1], cells[1:], strict=True), 1
        )
        if previous.topology != cell.topology
    ]


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
        path = out / f"fig2_validity_gap.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    text = CAPTION_TEMPLATE.format(**caption_facts(validity_by_cell(by_group)))
    caption = out / "fig2_validity_gap.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
