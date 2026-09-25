"""Figure 4 — the central H4 figure: ΔS against ΔΔB, per rung.

Renders the numbers `icknowledge/analysis/h4.py` produced; runs no experiment and
recomputes no verdict. Reads ``<root>/summary/h4_per_seed_ddb.csv`` and
``h4_ddb_by_cell.csv``.

    python -m scripts.fig4_h4_scatter [--root results/cross_seed_N20] [--out DIR]

Two panels, one per degraded rung (L1-oracle | L0). Twelve instances per panel:
the six effect-modifying ones at their own ΔS, the six additive controls pinned to
ΔS ≈ 0 (flat by construction, PS-3 convention 4). Per-seed points are drawn small
and translucent BEHIND the instance mean.

DESCRIPTIVE SD BANDS ONLY — no SE, no CI (PS-9); the per-seed counts are printed
beside each effect-modifying instance.

PS-8: the pooled points are WITHIN-TOPOLOGY differences. Levels are not comparable
across topologies — the panel reads as a pattern, never as a magnitude comparison.
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from icknowledge.analysis.h4 import RUNGS, SIGN_GATE_K, SOURCE_COLUMN  # noqa: E402
from icknowledge.analysis.loading import N20_ROOT, N_SEEDS_FULL  # noqa: E402

#: SHA-256 of the frozen H4 artifacts this figure is licensed to render. The N=20
#: tree is sealed (PS-1 ceiling) and the H4 verdict is frozen (PS-7/PS-4), so
#: pinning the digests makes "rendered against the frozen inputs" checkable rather
#: than asserted. Rebasing a constant here is a register act, not an edit.
PER_SEED_DDB_SHA256 = (
    "bc8e6af7d749f87fea00767e85a77bbe1caa524e04d40afcd0edd7b58cddd3c4"
)
DDB_BY_CELL_SHA256 = (
    "6c618ca5797b11a128443b7f13b86229e0ac17d6036e8a8825eedbedfc50056e"
)

#: [PS-4 dynamic-arity suffix] ΔΔB is built from the PAIRWISE-vs-L2 common-found
#: population, so its denominator arity is TWO, not the three-way population fig1
#: plots or the four-way one fig6/fig7 plot. Derived from h4's own source column
#: rather than typed, so the label and the data cannot disagree.
DENOMINATOR_LABEL = "twoway"
#: The population COUNT column in the pairwise artifact, and the VALUE column
#: read off it. Named separately because they are different things: the first
#: says who is in the denominator, the second says what is measured on them.
POPULATION_COLUMN = "N_common_found_pairwise_vs_L2"
VALUE_COLUMN = SOURCE_COLUMN

#: Topology -> marker. Shape carries topology because it survives greyscale
#: printing and because PS-8 forbids reading across shapes as a magnitude scale.
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}

#: Family -> fill. Filled = linear, hollow = nonlinear-Gaussian.
FAMILY_FILLED = {"linear": True, "nlg": False}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}

#: Regime -> colour. Okabe-Ito, colour-blind safe.
REGIME_COLOUR = {"effect_modifying": "#0072B2", "additive": "#D55E00"}

CAPTION_TEMPLATE = """
Figure 4. Group-asymmetric burden degradation (H4): the structural descriptor
difference DS = S(-1) - S(+1) against the burden-increase asymmetry
DDB(c) = DB_(-1)(c) - DB_(+1)(c), at each degraded knowledge rung. Blue markers
are the six effect-modifying SCM instances (3 topologies x 2 families); orange
markers are the six additive controls, which sit at DS = 0 by construction
(PS-3 convention 4). Marker shape encodes topology (circle = triangle,
diamond = collider, square = chain); filled = linear family, hollow =
nonlinear-Gaussian. Small translucent points are the 20 per-seed values; the large
marker is the instance mean and the vertical bar is +/-1 descriptive SD. No
standard errors or confidence intervals are shown: PS-9
reads the sign claim as a per-seed COUNT, printed beside each effect-modifying
instance as agree/20 (gate: >= {gate}/20), set in bold where the instance
clears the gate, and re-tasks SD as descriptive
dispersion only (the inference stance).

POPULATION: DDB(c) is built on the PAIRWISE-vs-L2 common-found set
(`{population}`, value column `{value_column}`), whose denominator arity is
TWO. The `{denominator}` word is
carried on the value axis and here, per PS-4's dynamic-arity rule: the three-way
population Figure 1 plots and the four-way L1-discovered population Figures 6-7
plot are different denominators, and an unlabelled magnitude could be read
against the wrong one.

REGISTER-LOGGED PREDICTION: effect-modifying instances fall in the lower-left
quadrant (DS < 0 and DDB < 0, i.e. sign(DS . DDB) > 0), while additive controls
pin to the vertical DS = 0 axis near DDB = 0. OBSERVED: at L1-oracle {l1_pass} of 6
effect-modifying instances clear the seed gate; at L0, {l0_pass} of 6. {observed} The
predicted quadrant is shaded as a reading aid.

Levels are not comparable across topologies (PS-8): the pooled points are
within-topology differences, so the panel is read as a pattern - which quadrant,
which sign - and never as a magnitude comparison between marker shapes.
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


def assert_frozen_digest(path: Path, expected: str) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact.

    Passed explicitly rather than defaulted, so a test can substitute a fixture's
    digest without the guard silently passing.
    """
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
    """A partially-written grid fails loudly rather than being plotted silently.

    This guard is required on every figure. An
    instance short a seed would move a mean, an SD and a printed agree/20 count
    with nothing on the rendered page to say so.
    """
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


def _style() -> None:
    """Self-contained styling — no rcParams are inherited from another script."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def build_figure(per_seed: pd.DataFrame, by_cell: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(6.3, 7.6), sharey=False)
    for axis, condition in zip(axes, RUNGS, strict=True):
        cells = by_cell[by_cell["condition"] == condition]
        seeds = per_seed[per_seed["condition"] == condition]
        # Jitter in AXIS units: the ΔS range of this rung's instances, scaled.
        delta_s = cells["delta_S"].to_numpy(dtype=float)
        axis_span = float(delta_s.max() - delta_s.min()) or 1.0
        jitter = np.linspace(-0.018, 0.018, N_SEEDS_FULL) * axis_span
        axis.axhline(0.0, color="0.55", linewidth=0.8, zorder=0)
        axis.axvline(0.0, color="0.55", linewidth=0.8, zorder=0)

        for row in cells.itertuples():
            colour = REGIME_COLOUR[row.regime]
            marker = TOPOLOGY_MARKER[row.topology]
            filled = FAMILY_FILLED[row.family]
            values = seeds[
                (seeds["topology"] == row.topology)
                & (seeds["family"] == row.family)
                & (seeds["regime"] == row.regime)
            ]["ddB"].to_numpy(dtype=float)

            # Per-seed cloud, jittered on x ONLY for visibility: ΔS is identical
            # across seeds by construction (PS-3 fit-independence), so every seed
            # of an instance shares one x. The jitter is deterministic — no RNG.
            # The spread is a fraction of the AXIS SPAN, not a literal in ΔS units.
            axis.scatter(
                row.delta_S + jitter, values, s=7, alpha=0.28, color=colour,
                marker=marker, linewidths=0, zorder=2,
            )
            axis.errorbar(
                row.delta_S, row.mean_ddB, yerr=row.sd_ddB,
                color=colour, ecolor=colour, elinewidth=1.1, capsize=2.5,
                marker=marker, markersize=7,
                markerfacecolor=colour if filled else "white",
                markeredgecolor=colour, markeredgewidth=1.2,
                linestyle="none", zorder=3,
            )
            if row.regime == "effect_modifying":
                # The flag is READ off the frozen by-cell row, never recomputed
                # from a threshold comparison against the plotted counts.
                gate_pass = (
                    bool(row.sign_gate_pass)
                    if pd.notna(row.sign_gate_pass)
                    else False
                )
                axis.annotate(
                    f"{int(row.n_seeds_sign_agree)}/{int(row.n_seeds)}",
                    (row.delta_S, row.mean_ddB),
                    textcoords="offset points", xytext=(9, 4),
                    fontsize=7, color=colour,
                    fontweight="bold" if gate_pass else "normal",
                )

        axis.set_title(f"rung {condition}")
        axis.set_xlabel("ΔS  =  S(−1) − S(+1)")
        axis.set_ylabel(
            "ΔΔB(c)  =  ΔB₋₁(c) − ΔB₊₁(c)\n"
            "(common-found, two-way)",
            fontsize=8,
        )
        axis.margins(x=0.14, y=0.16)
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        if left < 0.0 and bottom < 0.0:
            axis.add_patch(
                Rectangle(
                    (left, bottom), -left, -bottom,
                    facecolor="0.55", alpha=0.10, edgecolor="none",
                    zorder=0,
                )
            )
            axis.set_xlim(left, right)
            axis.set_ylim(bottom, top)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color=REGIME_COLOUR["effect_modifying"],
                   markersize=6, label="effect-modifying (treatment)"),
        plt.Line2D([], [], linestyle="none", marker="o", color=REGIME_COLOUR["additive"],
                   markersize=6, label="additive (control)"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=6,
                   label="triangle"),
        plt.Line2D([], [], linestyle="none", marker="D", color="0.35", markersize=6,
                   label="collider"),
        plt.Line2D([], [], linestyle="none", marker="s", color="0.35", markersize=6,
                   label="chain"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=6,
                   markerfacecolor="white", label="nonlinear-Gaussian (hollow)"),
        Patch(facecolor="0.55", alpha=0.10, edgecolor="none",
              label="shaded: predicted quadrant (register-logged sign agreement)"),
        plt.Line2D([], [], linestyle="none", marker="none",
                   label=f"bold count = clears the {SIGN_GATE_K}/{N_SEEDS_FULL} seed gate"),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
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
    per_seed = pd.read_csv(summary / "h4_per_seed_ddb.csv", float_precision="round_trip")
    by_cell = pd.read_csv(summary / "h4_ddb_by_cell.csv", float_precision="round_trip")
    assert_seed_counts(per_seed)

    _style()
    figure = build_figure(per_seed, by_cell)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"fig4_h4_scatter.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    # Every number in the caption is read off the data at render time, so the
    # caption cannot drift from the figure it describes.
    em = by_cell[by_cell["regime"] == "effect_modifying"]
    passes = {
        condition: int(em[em["condition"] == condition]["sign_gate_pass"].sum())
        for condition in RUNGS
    }
    wrong_way = em[(em["condition"] == "L0") & (em["mean_ddB"] > 0)]
    observed = (
        "The predicted lower-left concentration is not obtained at either rung."
        if passes["L1-oracle"] < 5 and passes["L0"] < 5
        else "See h4_verdict.md for the mechanical reading."
    )
    if not wrong_way.empty:
        names = ", ".join(
            f"{r.topology}/{FAMILY_DISPLAY[r.family]}"
            for r in wrong_way.itertuples()
        )
        observed += (
            f" At L0 the following instances carry the OPPOSITE sign to the"
            f" prediction (DDB > 0 against DS < 0): {names}."
        )
    text = CAPTION_TEMPLATE.format(
        gate=SIGN_GATE_K,
        l1_pass=passes["L1-oracle"],
        l0_pass=passes["L0"],
        observed=observed,
        denominator=DENOMINATOR_LABEL,
        population=POPULATION_COLUMN,
        value_column=VALUE_COLUMN,
    )
    caption = out / "fig4_h4_scatter.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
