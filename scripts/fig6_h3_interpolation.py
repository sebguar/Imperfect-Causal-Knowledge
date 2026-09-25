"""Figure 6 — the interpolation frame: three regions, per instance.

Presentation layer ONLY — runs no experiment, re-fits nothing and recomputes no
verdict quantity. Reads two FROZEN artifacts:

    <root>/summary/h3_per_seed.csv          (regions, dcost_ld, dcost_l0)
    <root>/summary/h3_instance_table_v2.csv (LD-2 class counts, halts)

    python -m scripts.fig6_h3_interpolation [--root results/cross_seed_L1d_N20] [--out DIR]

GAP FRAME. L1-oracle is the reference rung, so its gap is identically 0 by
construction. ``dcost_ld``, ``dcost_l0`` and region membership are all READ from
the frozen per-seed table — the shading boundaries are drawn from the strip MEAN
``dcost_l0`` (a drawing device), but no seed's region is re-derived from it.

Region 2 is drawn and labelled even when empty — an omitted empty zone reads as
an absent measurement rather than as a measured absence.

DESCRIPTIVE SD ONLY — no SE, no CI (PS-9; the inference stance).
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from icknowledge.analysis.h3 import (  # noqa: E402
    DENOMINATOR_LABEL,
    N_VALID_REQUIRED,
)
from icknowledge.analysis.loading import L1D_ROOT, N_SEEDS_FULL  # noqa: E402

#: SHA-256 of the frozen per-seed artifact this figure is licensed to render, as
#: carried on the LD-2 register record. The figure layer had no input-digest
#: guard before this script; the hash is already on the record, so asserting it
#: costs nothing and makes "rendered against the frozen inputs" checkable rather
#: than asserted. Rebasing this constant is a register act, not an edit.
PER_SEED_SHA256 = "108de1ccbae5b8bde9bd4f5f218259bccca9b05db80b6211ad163b490034372f"

#: Regime -> colour. Okabe-Ito, colour-blind safe. Identical literal to fig4/fig5.
REGIME_COLOUR = {"effect_modifying": "#0072B2", "additive": "#D55E00"}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}
REGIME_DISPLAY = {"additive": "additive",
                  "effect_modifying": "effect-modifying"}

#: Topology -> marker. Shape carries topology because it survives greyscale
#: printing and because PS-8 forbids reading across shapes as a magnitude scale.
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}

# NOTE: this module deliberately carries NO family->fill mapping.
# Family is encoded by the COLUMN FACET — every panel title names it — and the
# strip markers here are drawn at s=4..9 points with a "|" summary marker whose
# face is invisible by construction, so a filled/hollow distinction would encode
# nothing a reader could see and would need a legend entry for an invisible
# channel. fig4 and fig7 keep FAMILY_FILLED because their markers are large
# enough to carry it; this one does not.

#: Region -> shading. Neutral = holds; warm = the predicted-direction violation
#: zone (region 2); cool = better-than-both (region 3). Deliberately very light
#: AND strongly desaturated: the zones are the frame, the points are the data,
#: and a saturated warm/cool pair would collide with the Okabe-Ito regime hues —
#: an orange point inside an orange band reads as a region assignment it is not.
#: The dotted boundary line drawn at each strip's interval end carries the
#: partition where the tints alone would be too subtle.
REGION_FILL = {1: "#F2F2F2", 2: "#F8EEE3", 3: "#E7EDF3"}
REGION_LABEL = {
    1: "region 1 (holds)",
    2: "region 2 (worse than no graph)",
    3: "region 3 (better than both)",
}

#: Row order: topology-major so each topology's two family panels are contiguous
#: and the free-x grouping (PS-8) is visually obvious — the fig1/fig5 rationale.
ROWS = ("triangle", "chain", "collider")
COLUMNS = ("linear", "nlg")

#: Two horizontal strips per panel: additive above, effect-modifying below.
STRIPS = (("additive", 1.5), ("effect_modifying", 0.5))
STRIP_HALF = 0.5

#: Deterministic within-strip spread — no RNG, so the figure is reproducible and
#: 20 coincident seed values stay countable where they pile up (exact zeros do).
LD_SPREAD = np.linspace(-0.085, 0.085, N_SEEDS_FULL)
L0_SPREAD = np.linspace(-0.055, 0.055, N_SEEDS_FULL)
LD_OFFSET = +0.13
L0_OFFSET = -0.14

CAPTION_TEMPLATE = """
Figure 6. The H3 interpolation frame: per-seed realized-cost gaps against the
L1-oracle reference, one panel per (topology x family) and two strips per panel
(additive above, effect-modifying below). POPULATION: the PS-4 four-way
common-found set; the `{denominator}` denominator label is carried on the value
axis, in every panel title and here, because a four-way number may never sit
under an unlabelled or three-way header (PS-4 JOINT TABLES / dynamic-arity
suffix). GAP FRAME: L1-oracle is the reference rung, so its gap is identically
zero by construction (the vertical reference line); the large points are the
{n_seeds} per-seed dcost_ld values (cost at L1-discovered minus cost at
L1-oracle) and the small, lighter points of the same colour are the per-seed
dcost_l0 values (cost at L0 minus cost at L1-oracle), which supply the far end
of each seed's interpolation interval and vary from seed to seed.

THREE REGIONS (the PS-8 region partition, as corrected to the mutually
exclusive interval partition). With m the strip's MEAN dcost_l0: region 1
(holds) is the closed interval [min(0, m), max(0, m)], shaded grey; region 2
(the predicted-direction violation, worse than BOTH baselines) is beyond
max(0, m), shaded warm; region 3 (better than both) is below min(0, m), shaded
cool. The shaded boundaries are drawn from the mean, which is a drawing device
only: per-seed interval ends differ and are plotted as the small points, and
every region COUNT printed on a strip is the per-seed assignment read verbatim
from the frozen h3_per_seed.csv `region` column, never re-derived from m.

THE worse-than-no-graph NULL IN ONE NUMBER: region 2 contains {region_2_total} of {n_total}
per-seed observations across the grid. The empty predicted zone is drawn and
labelled on every strip rather than omitted, so the figure shows a measured
absence and not an absent measurement. REGION 3, by contrast, is occupied:
{region_3_em} of {n_em} effect-modifying seeds and {region_3_add} of {n_add}
additive seeds sit below both baselines, i.e. the discovered graph was CHEAPER
than the oracle graph.

CHANNELS (LD-2 classification/label layer, read from h3_instance_table_v2.csv;
no channel is inferred here). Class (c) is the registered interaction-blindness
channel (PS-7) — a perfect skeleton with correct orientations, where the
discovered rung's group-agnostic form prescribes cheaper pooled actions. Class
(d) is regime-induced skeleton misspecification: the A-X2 dependence is carried
by the interaction and is therefore structurally invisible to the pinned linear
CI test (LD-2). Additive strips whose seeds are all region 1 at exactly zero
carry the form-preserving estimation identity of PS-7 note (c) — a positive confirmation, not
missing data.

PS-8: levels are not comparable across topologies (different action-space
dimension, induced marginal variances and classifier-boundary geometry), so
each topology row carries one shared x-range for its two family panels and the
rows are independent of one another; a cross-row reading is pattern-only. All
{n_seeds} per-seed values are shown; the bar is the mean dcost_ld with +/-1
DESCRIPTIVE SD (ddof = 1). There are no standard errors or confidence intervals
on this figure: PS-9 re-tasks SD as descriptive
dispersion, and a formal interval would over-promise precision this structure
lacks (the inference stance).
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


# --------------------------------------------------------------------------- #
# Frozen-input guards
# --------------------------------------------------------------------------- #


def assert_frozen_digest(path: Path, expected: str | None = None) -> str:
    """Abort loudly unless ``path`` is byte-identical to the registered artifact.

    Read from the module constant at CALL time (not as a default argument) so a
    test can substitute a fixture's digest without the guard silently passing.
    """
    expected = expected if expected is not None else PER_SEED_SHA256
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"FROZEN-INPUT DIGEST MISMATCH for {path}: expected SHA-256 {expected}, "
            f"got {digest}. This figure renders the frozen H3 verdict artifacts and "
            "may not be drawn against a modified input — re-check the tree against "
            "the LD-2 register record rather than rebasing this constant."
        )
    return digest


def assert_seed_counts(per_seed: pd.DataFrame, n_seeds: int = N_SEEDS_FULL) -> None:
    """A partially-written grid fails loudly rather than being plotted silently.

    This guard is carried here: a
    19-seed instance would shift a mean, an SD and a region count with nothing on
    the rendered page to say so.
    """
    counts = per_seed.groupby(["topology", "family", "regime"], sort=True).size()
    wrong = counts[counts != n_seeds]
    if not wrong.empty:
        detail = ", ".join(
            f"{topology}/{family}/{regime}={int(value)}"
            for (topology, family, regime), value in wrong.items()
        )
        raise ValueError(
            f"expected {n_seeds} seeds per instance (PS-1 pinned ceiling); got {detail}"
        )


# --------------------------------------------------------------------------- #
# Counts — every caption number is read off the frozen `region` column
# --------------------------------------------------------------------------- #


def region_counts(per_seed: pd.DataFrame) -> dict[str, int]:
    """Grid-level region tallies, counted from the frozen per-seed column.

    Nothing here re-derives a region from dcost_ld / dcost_l0: the partition was
    settled once and applied once, in the analysis layer.
    """
    regions = per_seed["region"].astype(int)
    em = per_seed["regime"] == "effect_modifying"
    return {
        "n_total": int(len(per_seed)),
        "region_1_total": int((regions == 1).sum()),
        "region_2_total": int((regions == 2).sum()),
        "region_3_total": int((regions == 3).sum()),
        "n_em": int(em.sum()),
        "n_add": int((~em).sum()),
        "region_3_em": int(((regions == 3) & em).sum()),
        "region_3_add": int(((regions == 3) & ~em).sum()),
    }


def linear_status(
    instances: pd.DataFrame,
    topology: str,
    regime: str,
    ld: np.ndarray,
    regions: np.ndarray,
) -> str:
    """The LD-2 status for one LINEAR strip. NEVER empty.

    The wording is fixed; WHICH status a strip carries is decided by the frozen
    ``halt``/``halt_class_*`` columns, so the figure cannot assert a channel the
    verdict layer did not emit. Every branch returns a string — an absent status
    would read as an unexamined cell rather than an examined clean one.

    Order matters: a standing halt outranks a channel label, and a channel label
    outranks the form-preserving estimation identity note.
    """
    row = instances[
        (instances["topology"] == topology)
        & (instances["family"] == "linear")
        & (instances["regime"] == regime)
    ]
    if row.empty:
        raise LookupError(
            f"{topology}/linear/{regime} has no row in the frozen instance table — "
            "this figure renders statuses from that table and will not invent one."
        )
    if bool(row["halt"].iloc[0]):
        return "HARNESS HALT STANDS (LD-2)"
    if int(row["halt_class_d"].iloc[0]) > 0:
        return "class (d): A→X2 structurally invisible to linear CI (LD-2)"
    if int(row["halt_class_c"].iloc[0]) > 0:
        return "class (c): interaction-blindness (PS-7)"
    if (regions == 1).all() and not ld.any():
        # Positive confirmation of the PS-7 note (c) identity, not missing data.
        return "clean — form-preserving estimation exact zero (PS-7 note c)"
    return "clean — interpolation held, diagnostic not triggered (LD-2)"


def _xlim_by_topology(per_seed: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """PS-8 encoding: one shared x-range per topology, independent between them.

    Sharing a single range across topologies would invite exactly the level
    comparison PS-8 forbids; sharing within a topology keeps the family scan
    (which IS allowed, and is H3's own contrast) on a common scale. The right pad
    is the larger of the two so the region-2 zone is always visibly drawn even
    where the strip's mean dcost_l0 sits at the data maximum.
    """
    limits: dict[str, tuple[float, float]] = {}
    for topology in ROWS:
        block = per_seed[per_seed["topology"] == topology]
        values = np.concatenate(
            [
                block["dcost_ld"].to_numpy(dtype=float),
                block["dcost_l0"].to_numpy(dtype=float),
                np.zeros(1),
            ]
        )
        low, high = float(values.min()), float(values.max())
        span = (high - low) or 1.0
        limits[topology] = (low - 0.14 * span, high + 0.26 * span)
    return limits


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _draw_strip(
    axis,
    block: pd.DataFrame,
    instances: pd.DataFrame,
    topology: str,
    family: str,
    regime: str,
    centre: float,
    xlim: tuple[float, float],
) -> None:
    colour = REGIME_COLOUR[regime]
    ordered = block.sort_values("seed_idx")
    ld = ordered["dcost_ld"].to_numpy(dtype=float)
    l0 = ordered["dcost_l0"].to_numpy(dtype=float)
    regions = ordered["region"].astype(int).to_numpy()

    # Shading boundaries from the strip MEAN dcost_l0 — a drawing device. Region
    # MEMBERSHIP below is the frozen per-seed column, never re-derived from m.
    m = float(l0.mean())
    lower, upper = min(0.0, m), max(0.0, m)
    bottom, height = centre - STRIP_HALF, 2 * STRIP_HALF
    for region, (x0, x1) in (
        (3, (xlim[0], lower)),
        (1, (lower, upper)),
        (2, (upper, xlim[1])),
    ):
        if x1 > x0:
            axis.add_patch(
                Rectangle(
                    (x0, bottom), x1 - x0, height,
                    facecolor=REGION_FILL[region], edgecolor="none", zorder=0,
                )
            )
    # The interval's far end, drawn only across THIS strip: the three-region
    # partition must stay readable where the tints are near-identical in print.
    for boundary in (lower, upper):
        if boundary != 0.0:
            axis.plot(
                [boundary, boundary], [bottom + 0.02, bottom + height - 0.02],
                color="0.62", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1,
            )
    axis.axvline(0.0, color="0.55", linewidth=0.8, zorder=1)

    marker = TOPOLOGY_MARKER[topology]
    # dcost_l0: the interval's far end. Smaller and lighter, on its own sub-band,
    # because it is the FRAME the gap is read against, not the quantity under
    # test — but it varies by seed, so it is shown rather than hidden in m.
    axis.scatter(
        l0, np.full(l0.size, centre + L0_OFFSET) + L0_SPREAD[: l0.size],
        s=4, alpha=0.32, color=colour, marker=marker, linewidths=0, zorder=2,
    )
    axis.scatter(
        ld, np.full(ld.size, centre + LD_OFFSET) + LD_SPREAD[: ld.size],
        s=9, alpha=0.55, color=colour, marker=marker, linewidths=0, zorder=3,
    )
    # Mean +/- 1 DESCRIPTIVE SD (ddof=1). The fig5 mean-marker grammar, rotated:
    # the value axis is horizontal here, so the perpendicular tick is "|".
    axis.errorbar(
        float(ld.mean()), centre + LD_OFFSET, xerr=float(ld.std(ddof=1)),
        color=colour, ecolor=colour, elinewidth=1.2, capsize=3,
        marker="|", markersize=13, markeredgewidth=2.0,
        markeredgecolor=colour, linestyle="none", zorder=4,
    )

    n2 = int((regions == 2).sum())
    n3 = int((regions == 3).sum())
    # Stacked, not side by side: at 6.3 in over two columns the two labels are
    # wider than one panel line and would overprint each other. Each keeps the
    # side its region is on, so the alignment still reads as a pointer.
    pad = 0.02 * (xlim[1] - xlim[0])
    axis.text(
        xlim[1] - pad, centre + STRIP_HALF - 0.03,
        f"region 2 — worse than no graph: {n2}/{N_VALID_REQUIRED}",
        ha="right", va="top", fontsize=6, color="0.35", zorder=5,
    )
    axis.text(
        xlim[0] + pad, centre + STRIP_HALF - 0.16,
        f"region 3: {n3}/{N_VALID_REQUIRED}",
        ha="left", va="top", fontsize=6, color="0.35", zorder=5,
    )

    # Every LINEAR strip carries exactly one status; the nlg strips
    # carry none, because the LD-2 taxonomy is defined on the linear
    # control arm and inventing a label for the nlg arm would over-report it.
    note = linear_status(instances, topology, regime, ld, regions) if family == "linear" else ""
    if note:
        axis.text(
            xlim[0] + pad, centre - STRIP_HALF + 0.04,
            textwrap.fill(note, 46),
            ha="left", va="bottom", fontsize=6.5, color="0.25", zorder=5,
        )


def build_figure(per_seed: pd.DataFrame, instances: pd.DataFrame) -> plt.Figure:
    limits = _xlim_by_topology(per_seed)
    figure, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(6.3, 8.6))

    for row, topology in enumerate(ROWS):
        for column, family in enumerate(COLUMNS):
            axis = axes[row][column]
            xlim = limits[topology]
            for regime, centre in STRIPS:
                block = per_seed[
                    (per_seed["topology"] == topology)
                    & (per_seed["family"] == family)
                    & (per_seed["regime"] == regime)
                ]
                _draw_strip(
                    axis, block, instances, topology, family, regime, centre, xlim
                )
            axis.set_xlim(*xlim)
            axis.set_ylim(0.0, 2.0)
            axis.set_yticks([centre for _, centre in STRIPS])
            axis.set_yticklabels(
                [REGIME_DISPLAY[regime] for regime, _ in STRIPS],
                fontsize=7,
            )
            axis.axhline(1.0, color="white", linewidth=1.2, zorder=1)
            axis.set_title(
                f"{topology} · {FAMILY_DISPLAY[family]} · four-way", fontsize=8, pad=4
            )
            axis.tick_params(axis="x", labelsize=7)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row == len(ROWS) - 1:
                axis.set_xlabel(
                    "cost gap vs L1-oracle\n(common-found, four-way)",
                    fontsize=8,
                )

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=5,
                   label="per-seed ΔCost_ld"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=3,
                   alpha=0.4, label="per-seed ΔCost_l0 (interval far end)"),
        plt.Line2D([], [], linestyle="none", marker="|", color="0.35", markersize=10,
                   markeredgewidth=2.0, label="mean ± 1 SD (descriptive)"),
        plt.Line2D([], [], linestyle="none", marker="s", markersize=6,
                   color=REGIME_COLOUR["additive"], label="additive"),
        plt.Line2D([], [], linestyle="none", marker="s", markersize=6,
                   color=REGIME_COLOUR["effect_modifying"], label="effect-modifying"),
    ] + [
        Patch(facecolor=REGION_FILL[region], edgecolor="0.8", label=REGION_LABEL[region])
        for region in (1, 2, 3)
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=L1D_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)
    summary = args.root / "summary"

    per_seed_path = summary / "h3_per_seed.csv"
    assert_frozen_digest(per_seed_path)
    per_seed = pd.read_csv(per_seed_path, float_precision="round_trip")
    assert_seed_counts(per_seed)
    instances = pd.read_csv(
        summary / "h3_instance_table_v2.csv", comment="#", float_precision="round_trip"
    )

    _style()
    figure = build_figure(per_seed, instances)
    written = []
    for suffix in ("pdf", "png"):
        path = out / f"fig6_h3_interpolation.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    # Every number in the caption is read off the data at render time, so the
    # caption cannot drift from the figure it describes.
    counts = region_counts(per_seed)
    text = CAPTION_TEMPLATE.format(
        denominator=DENOMINATOR_LABEL, n_seeds=N_SEEDS_FULL, **counts
    )
    caption = out / "fig6_h3_interpolation.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
