"""Figure 5 — the regime contrast: |ΔΔB| treatment vs control, per rung.

Renders the numbers `icknowledge/analysis/h4.py` produced; runs no experiment and
recomputes no verdict. Reads ``<root>/summary/h4_per_seed_ddb.csv`` and
``h4_ddb_by_cell.csv``.

    python -m scripts.fig5_regime_contrast [--root results/cross_seed_N20] [--out DIR]

One panel per degraded rung; within a panel each topology × family appears as a
PAIR of strips — effect-modifying beside its additive control (PS-9).

|ΔΔB| is plotted, not ΔΔB: PS-9's contrast criterion is stated on magnitudes
(|ΔΔB|_treatment > |ΔΔB|_control). The signed values are fig4's subject.

Descriptive SD only — no SE, no CI (PS-9; the inference stance).
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

from icknowledge.analysis.h4 import RUNGS, SOURCE_COLUMN  # noqa: E402
from icknowledge.analysis.loading import (  # noqa: E402
    N20_ROOT,
    N_SEEDS_FULL,
    TOPOLOGIES,
)

#: SHA-256 of the frozen H4 artifacts this figure is licensed to render. Same
#: rationale as fig4; rebasing a constant here is a register act, not an edit.
PER_SEED_DDB_SHA256 = (
    "bc8e6af7d749f87fea00767e85a77bbe1caa524e04d40afcd0edd7b58cddd3c4"
)
DDB_BY_CELL_SHA256 = (
    "6c618ca5797b11a128443b7f13b86229e0ac17d6036e8a8825eedbedfc50056e"
)
REGIME_CONTRAST_SHA256 = (
    "0c0da7e7c65a02cfe879fb21cd45474bfd9e655ef8902c5adf1d7eba8dd929d3"
)

REGIME_COLOUR = {"effect_modifying": "#0072B2", "additive": "#D55E00"}
REGIME_LABEL = {"effect_modifying": "treatment", "additive": "control"}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}

#: Topology -> marker. The reserved glyphs, meaning exactly what they mean in
#: fig2/fig3/fig4/fig6/fig7. Encoding topology as well as labelling it means the
#: reader never has to trust a tick label to know which instance a strip belongs
#: to.
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}

#: [PS-4 dynamic-arity suffix] ΔΔB sits on the PAIRWISE-vs-L2 common-found
#: population — arity TWO, not fig1's three-way or fig6/fig7's four-way.
DENOMINATOR_LABEL = "twoway"
#: The population COUNT column in the pairwise artifact, and the VALUE column
#: read off it. Named separately because they are different things: the first
#: says who is in the denominator, the second says what is measured on them.
POPULATION_COLUMN = "N_common_found_pairwise_vs_L2"
VALUE_COLUMN = SOURCE_COLUMN

#: Within one topology's sub-axis, the two families side by side.
FAMILIES = ("linear", "nlg")

CAPTION_TEMPLATE = """
Figure 5. The regime contrast (H4's primary evidence): |DDB(c)| under the
effect-modifying treatment beside its matched additive control, at each degraded
knowledge rung. Each pair is one SCM instance; small translucent points are the
{n_seeds} per-seed |DDB| values, the bar is the instance mean and the whisker is
+/-1 descriptive SD. No standard errors or confidence intervals (PS-9; the
inference stance). Magnitudes are plotted because PS-9 states the contrast
criterion on magnitudes; the signed values are Figure 4's subject.

POPULATION: DDB(c) is built on the PAIRWISE-vs-L2 common-found set
(`{population}`, value column `{value_column}`), whose denominator arity is
TWO. The `{denominator}` word is
carried on the value axis and here, per PS-4's dynamic-arity rule.

LAYOUT — PS-8 ENFORCED BY THE ENCODING, NOT BY THIS SENTENCE. Each topology has
its OWN sub-axis with its OWN y-range, in every rung row. There is deliberately no
shared vertical scale across topologies, so a cross-topology magnitude comparison
is not available to make rather than merely warned against; marker shape carries
topology as well (circle = triangle, diamond = collider, square = chain). What
remains on one common scale is the comparison PS-9 actually reads: treatment
beside its matched control, within one instance. Read each pair against its own
control, never against another topology's bar.

PS-9 CONTRAST CRITERION: |DDB(c)|_treatment > |DDB(c)|_control, read per topology
across the three topologies. OBSERVED: at L1-oracle the contrast {l1_read}; at L0
it {l0_read}.

SIGNAL-2 DISCRIMINATOR (before these numbers existed): the PS-9
N=20 diagnostic found the dominant validity-collapse separator to be estimation
error on an UNMODIFIED, regime-shared downstream edge. Such a channel would move
BOTH arms, so a clean contrast (treatment signed, control flat) rules it out as the
driver of DDB, while a non-flat control makes it a live alternative explanation
that must be named rather than shelved. {signal2}

"""


def _wrap(text: str) -> str:
    # break_on_hyphens=False: the register's vocabulary is full of hyphenated
    # terms that must survive as single tokens — "REGISTER-LOGGED",
    # "non-gating", "effect-modifying", "L1-oracle". textwrap's default
    # splits them across a line boundary, which both hurts readability and
    # makes any fragment check depend on where the wrap happened to land.
    return textwrap.fill(
        " ".join(text.split()), width=78, break_on_hyphens=False
    ) + "\n"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def assert_frozen_digest(path: Path, expected: str) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"FROZEN-INPUT DIGEST MISMATCH for {path}: expected SHA-256 {expected}, "
            f"got {digest}. This figure renders the frozen H4 artifacts and may not "
            "be drawn against a modified input — re-check the tree against the "
            "PS-1/PS-9 record rather than rebasing this constant."
        )
    return digest


def assert_seed_counts(per_seed: pd.DataFrame, n_seeds: int = N_SEEDS_FULL) -> None:
    """A partially-written grid fails loudly rather than being plotted silently."""
    counts = per_seed.groupby(
        ["topology", "family", "regime", "condition"], sort=True
    ).size()
    wrong = counts[counts != n_seeds]
    if not wrong.empty:
        detail = ", ".join(
            f"{topology}/{family}/{regime}/{condition}={int(value)}"
            for (topology, family, regime, condition), value in wrong.items()
        )
        raise ValueError(
            f"expected {n_seeds} seeds per instance (PS-1 pinned ceiling); got {detail}"
        )


def build_figure(per_seed: pd.DataFrame, by_cell: pd.DataFrame) -> plt.Figure:
    """One sub-axis per (rung x topology), each with its OWN y-range.

    PS-8 COMPLIANCE IS STRUCTURAL HERE, NOT DISCLAIMED. Each topology has its own
    axis with an independent range, so there is no shared scale on the page
    against which two topologies' bars could be compared — the forbidden reading
    is not available to make.

    What the layout PRESERVES is the comparison PS-9 actually reads: treatment
    beside its matched control, within one instance, on one axis. The contrast is
    still adjacent and still on a common scale, because that pair is the evidence.
    """
    figure, axes = plt.subplots(
        len(RUNGS), len(TOPOLOGIES), figsize=(6.3, 6.4), squeeze=False
    )
    # Deterministic within-strip spread — no RNG, so the figure is reproducible and
    # the seeds stay countable where they coincide. Sized by the imported seed
    # count, never a literal.
    spread = np.linspace(-0.11, 0.11, N_SEEDS_FULL)

    for row, condition in enumerate(RUNGS):
        for column, topology in enumerate(TOPOLOGIES):
            axis = axes[row][column]
            marker = TOPOLOGY_MARKER[topology]
            ticks, labels = [], []
            for index, family in enumerate(FAMILIES):
                for offset, regime in (
                    (-0.17, "effect_modifying"),
                    (+0.17, "additive"),
                ):
                    position = index + offset
                    values = np.abs(
                        per_seed[
                            (per_seed["condition"] == condition)
                            & (per_seed["topology"] == topology)
                            & (per_seed["family"] == family)
                            & (per_seed["regime"] == regime)
                        ]["ddB"].to_numpy(dtype=float)
                    )
                    colour = REGIME_COLOUR[regime]
                    axis.scatter(
                        position + spread[: values.size], values,
                        s=5, alpha=0.30, color=colour, marker=marker,
                        linewidths=0, zorder=2,
                    )
                    # The bar is the mean of |ΔΔB| per seed, NOT |mean ΔΔB|: on a
                    # cell whose sign flips across seeds those differ a lot, and
                    # the strip beside it would otherwise contradict its own
                    # summary marker.
                    axis.errorbar(
                        position, values.mean(), yerr=values.std(ddof=1),
                        color=colour, ecolor=colour, elinewidth=1.1, capsize=2.5,
                        marker="_", markersize=11, markeredgewidth=2.0,
                        linestyle="none", zorder=3,
                    )
                ticks.append(index)
                labels.append(FAMILY_DISPLAY[family])
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels, fontsize=7)
            axis.set_xlim(-0.6, len(FAMILIES) - 0.4)
            axis.axhline(0.0, color="0.55", linewidth=0.8, zorder=0)
            axis.margins(y=0.12)
            axis.tick_params(axis="y", labelsize=7)
            axis.set_title(f"{topology}\nrung {condition} · own y-axis", fontsize=7.5)
            if column == 0:
                axis.set_ylabel(
                    "|ΔΔB(c)| per seed\n(common-found, two-way)",
                    fontsize=7.5,
                )
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([], [], linestyle="none", marker="s", color=REGIME_COLOUR[r],
                   markersize=6, label=REGIME_LABEL[r])
        for r in ("effect_modifying", "additive")
    ] + [
        plt.Line2D([], [], linestyle="none", marker=TOPOLOGY_MARKER[topology],
                   color="0.35", markersize=5, label=topology)
        for topology in TOPOLOGIES
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.5, -0.045))
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=N20_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)
    summary = args.root / "summary"
    assert_frozen_digest(summary / "h4_per_seed_ddb.csv", PER_SEED_DDB_SHA256)
    assert_frozen_digest(summary / "h4_ddb_by_cell.csv", DDB_BY_CELL_SHA256)
    assert_frozen_digest(summary / "h4_regime_contrast.csv", REGIME_CONTRAST_SHA256)
    per_seed = pd.read_csv(summary / "h4_per_seed_ddb.csv", float_precision="round_trip")
    by_cell = pd.read_csv(summary / "h4_ddb_by_cell.csv", float_precision="round_trip")
    contrast = pd.read_csv(
        summary / "h4_regime_contrast.csv", float_precision="round_trip"
    )
    assert_seed_counts(per_seed)

    _style()
    figure = build_figure(per_seed, by_cell)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"fig5_regime_contrast.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    reads = {}
    for condition in RUNGS:
        scope = contrast[
            (contrast["condition"] == condition) & (contrast["family"] == "ALL")
        ]
        failures = [
            r.topology for r in scope.itertuples()
            if not r.contrast_holds and not r.saturated_both_arms
        ]
        holds = [r.topology for r in scope.itertuples() if r.contrast_holds]
        if len(holds) == 3:
            reads[condition] = f"holds on all three topologies ({', '.join(holds)})"
        else:
            where = ", ".join(holds) if holds else "no topology"
            failed = (
                ", ".join(failures)
                if failures
                else "none — the remaining topologies are exempted as "
                "saturated-tiny in both arms"
            )
            reads[condition] = f"holds on {where} and FAILS on {failed}"
    non_flat = by_cell[
        (by_cell["regime"] == "additive")
        & (by_cell["control_read"].str.startswith("ATTENUATION"))
    ]
    signal2 = (
        "OBSERVED: every additive control reads as absence (flat within seed "
        "noise), so the Signal-2 channel is ruled out as the DDB driver."
        if non_flat.empty
        else (
            f"OBSERVED: {len(non_flat)} control (instance, rung) readings are "
            "non-flat, so the Signal-2 channel is a LIVE alternative explanation "
            "and is named as such in the H4 write-up."
        )
    )
    text = CAPTION_TEMPLATE.format(
        l1_read=reads["L1-oracle"],
        l0_read=reads["L0"],
        signal2=signal2,
        n_seeds=N_SEEDS_FULL,
        denominator=DENOMINATOR_LABEL,
        population=POPULATION_COLUMN,
        value_column=VALUE_COLUMN,
    )
    caption = out / "fig5_regime_contrast.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
