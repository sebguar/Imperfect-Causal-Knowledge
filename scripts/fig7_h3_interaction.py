"""Figure 7 — the family × provenance interaction, as paired arms.

Presentation layer ONLY — runs no experiment and recomputes no verdict quantity.
Reads two FROZEN artifacts:

    <root>/summary/h3_per_seed.csv          (dcost_ld, interaction event, ties)
    <root>/summary/h3_instance_table_v2.csv (LD-2 standing halts)

    python -m scripts.fig7_h3_interaction [--root results/cross_seed_L1d_N20] [--out DIR]

TOPOLOGY HIERARCHY, DRAWN. PS-8 makes the COLLIDER the confirmatory-primary
orientation channel, so it gets the large top panel; the chain and triangle pairs
sit below, smaller, titled as non-gating. Each topology keeps its own y-axis
(PS-8 forbids reading levels across topologies).

SEED PAIRING is the PS-8 common-random-numbers device: each line joins the SAME
seed_idx across the two families. The count is a STABILITY count, not a paired test.

MANDATORY ANNOTATION. The emitted orientation label is produced from
region-3 differentials, so PS-8 requires its clause on the figure itself, not
in the caption alone.

DESCRIPTIVE ONLY — no SE, no CI (PS-9; the inference stance).
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
from matplotlib.patches import Patch  # noqa: E402

from icknowledge.analysis.h3 import (  # noqa: E402
    CORROBORATING_TOPOLOGIES,
    DENOMINATOR_LABEL,
    N_VALID_REQUIRED,
    ORIENTATION_TOPOLOGY,
    REGIMES,
    SIGN_GATE_K,
    orientation_label,
)
from icknowledge.analysis.h3_v2 import D3_REGION3_CLAUSE  # noqa: E402
from icknowledge.analysis.loading import L1D_ROOT, N_SEEDS_FULL  # noqa: E402

#: SHA-256 of the frozen per-seed artifact this figure is licensed to render, as
#: carried on the LD-2 register record. See fig6 for the rationale; the constant
#: is duplicated deliberately so each script is self-contained.
PER_SEED_SHA256 = "108de1ccbae5b8bde9bd4f5f218259bccca9b05db80b6211ad163b490034372f"

#: [PS-8] The clause a region-3-driven orientation label carries
#: VERBATIM. Held as a literal here so the on-figure text is auditable in the
#: source, and asserted identical to the analysis layer's copy at render time so
#: the two can never drift apart.
D3_ON_FIGURE_TEXT = (
    "interaction reflects form-blindness magnitude in the region-3 (cheaper) "
    "direction, opposite to H3's predicted discovery-penalty direction; cannot be "
    "read as a discovery-penalty differential."
)

#: Regime -> colour. Okabe-Ito, colour-blind safe. Identical literal to fig4/fig5.
REGIME_COLOUR = {"effect_modifying": "#0072B2", "additive": "#D55E00"}

#: Display spellings for the raw data tokens carried in tick and title text.
FAMILY_DISPLAY = {"linear": "linear", "nlg": "nonlinear-Gaussian"}
REGIME_DISPLAY = {"additive": "additive",
                  "effect_modifying": "effect-modifying"}

#: Topology -> marker. Shape carries topology because it survives greyscale
#: printing and because PS-8 forbids reading across shapes as a magnitude scale.
TOPOLOGY_MARKER = {"triangle": "o", "collider": "D", "chain": "s"}

#: Family -> fill. Filled = linear, hollow = nonlinear-Gaussian.
FAMILY_FILLED = {"linear": True, "nlg": False}

#: Cool tint for the below-reference half-plane, matching fig6's region-3 fill.
REGION_3_FILL = "#DEEBF5"

#: One block per regime: two x positions (linear, nlg) with a gap between blocks
#: and a light dashed divider in it — the fig2 topology-divider grammar.
BLOCK_X = {"additive": (0.0, 1.0), "effect_modifying": (2.2, 3.2)}
DIVIDER_X = 1.6

CAPTION_TEMPLATE = """
Figure 7. The H3 family x provenance interaction (H3a) as paired arms: each
line joins one seed's dcost_ld under the LINEAR arm to the same seed's dcost_ld
under the nonlinear-Gaussian arm, at matched (topology, regime). POPULATION: the
PS-4 four-way common-found set, `{denominator}` denominator, labelled on the
value axis, in every panel title and here (PS-4).

TOP PANEL - ORIENTATION CHANNEL, CONFIRMATORY-PRIMARY (PS-8). The collider is
the only one of the three topologies whose graph admits any CI-based orientation, so it alone
carries the interaction channel. Per-seed event = dcost_ld(nonlinear-Gaussian, s) -
dcost_ld(linear, s) > 0 STRICTLY (PS-8); the printed counts are read
from the frozen h3_per_seed.csv event column at render time. OBSERVED: additive
{add_events}/{n_seeds} with {add_ties} exact zero differences (non-firing under
the strict inequality), effect-modifying {em_events}/{n_seeds}, against the
k = {gate}/{denominator_n} gate. ORIENTATION-CHANNEL VERDICT: {verdict}
({passing}/2 pairs pass).

MANDATORY ANNOTATION (PS-8), reproduced on the figure itself and
here: "{d3}" Region occupancy behind that clause: the collider effect-modifying
arms sit in region 3 on {collider_em_region3} of {collider_em_total} per-seed
observations, i.e. the discovered graph was CHEAPER than the oracle graph, which
is the opposite of H3's predicted discovery penalty. PER ARM, as PS-8 sub-clause
(i) requires — the pooled figure above can be carried by one arm, so each is
stated separately: effect-modifying {em_linear_arm} and {em_nlg_arm}; additive
{add_linear_arm} and {add_nlg_arm}. Region membership is the per-seed assignment
read from the frozen region column; the shaded band below the L1-oracle reference
marks the region-3 SIDE of the frame and is a reading aid, not a per-seed
classification.

BOTTOM PANELS - CORROBORATING AND non-gating (pinned as purely descriptive by
the PS-8 non-gating corroboration read). Chain and triangle carry the
adjacency/skeleton channel. They do not gate the orientation channel and no
threshold is applied to
them; per the registered SYMMETRIC no-consequence clause, agreement strengthens
the Discussion narrative and disagreement is reported as a tension, and neither
moves a label, a gate or a band. OBSERVED: {corroborating}.

SEED PAIRING is the PS-8 common-random-numbers device - the same seed_idx across
two independently generated cells - so each count is a STABILITY count and NOT a
matched-population paired test. PS-8: levels are not comparable across
topologies, so every topology carries its own y-axis and a cross-panel reading is
pattern-only. All {n_seeds} per-seed values are shown, and there are no standard
errors or confidence intervals on this figure: PS-9
re-tasks SD as descriptive dispersion only, and a formal interval would
over-promise precision this structure lacks (the inference stance).
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
            "xtick.labelsize": 8,
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

    Seed-completeness guard: an instance short a seed
    would drop a line from a paired-arms block with nothing on the page to say so.
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


def assert_d3_text_matches_the_analysis_copy() -> None:
    """The on-figure annotation literal must equal the analysis layer's PS-8 copy."""
    if D3_ON_FIGURE_TEXT != D3_REGION3_CLAUSE:
        raise RuntimeError(
            "the on-figure annotation clause has drifted from "
            "icknowledge.analysis.h3_v2.D3_REGION3_CLAUSE; PS-8 requires it "
            "VERBATIM, so the figure refuses to render."
        )


# --------------------------------------------------------------------------- #
# Counts — read from the frozen columns, never re-derived
# --------------------------------------------------------------------------- #


def gate_counts(per_seed: pd.DataFrame, topology: str, regime: str) -> dict[str, int]:
    """Per-pair interaction-event and exact-tie counts, off the frozen columns.

    ``interaction_event`` and ``interaction_diff`` are recorded on BOTH family
    rows of a pair by the analysis layer; the NLG rows are read so each pair is
    counted exactly once.
    """
    nlg = per_seed[
        (per_seed["topology"] == topology)
        & (per_seed["regime"] == regime)
        & (per_seed["family"] == "nlg")
    ]
    return {
        "events": int(nlg["interaction_event"].astype(bool).sum()),
        "ties": int((nlg["interaction_diff"].astype(float) == 0.0).sum()),
        "n_seeds": int(len(nlg)),
    }


def standing_halt(instances: pd.DataFrame, topology: str) -> bool:
    """Whether a HARNESS HALT still stands on a topology's control arm (PS-8)."""
    rows = instances[
        (instances["topology"] == topology) & (instances["family"] == "linear")
    ]
    return bool(rows["halt"].astype(bool).any())


def orientation_verdict(
    per_seed: pd.DataFrame, instances: pd.DataFrame
) -> tuple[str, int]:
    """[PS-8] The collider channel's label and pass count, recomputed at render.

    A STANDING collider-linear halt suppresses the label entirely (PS-8);
    on the frozen v2 artifacts every halt lifted, but the branch is kept so the
    figure cannot silently band a halted channel.
    """
    if standing_halt(instances, ORIENTATION_TOPOLOGY):
        return "NO LABEL — HALTED", 0
    passing = sum(
        gate_counts(per_seed, ORIENTATION_TOPOLOGY, regime)["events"] >= SIGN_GATE_K
        for regime in REGIMES
    )
    return orientation_label(passing), passing


def region_3_occupancy(per_seed: pd.DataFrame, topology: str, regime: str) -> tuple[int, int]:
    block = per_seed[
        (per_seed["topology"] == topology) & (per_seed["regime"] == regime)
    ]
    return int((block["region"].astype(int) == 3).sum()), int(len(block))


def arm_region_occupancy(
    per_seed: pd.DataFrame, topology: str, regime: str, family: str
) -> tuple[int, int, int]:
    """[PS-8 sub-clause (i)] Which region ONE family arm's seeds predominantly occupied.

    Returns (dominant_region, count_in_it, n_seeds), all read from the frozen
    ``region`` column — never re-derived from dcost_ld against a strip mean.

    PS-8 sub-clause (i) asks for the region occupancy of each family ARM, not a
    pooled figure over both. An aggregate can be dominated by one arm and still be
    reported as though it described the pair, which is precisely the reading the
    sub-clause exists to prevent; the counts are therefore per arm.
    """
    block = per_seed[
        (per_seed["topology"] == topology)
        & (per_seed["regime"] == regime)
        & (per_seed["family"] == family)
    ]
    regions = block["region"].astype(int)
    counts = {region: int((regions == region).sum()) for region in (1, 2, 3)}
    dominant = max(counts, key=lambda region: counts[region])
    return dominant, counts[dominant], int(len(block))


def arm_region_phrase(per_seed: pd.DataFrame, topology: str, regime: str) -> str:
    """The per-arm occupancy line PS-8 sub-clause (i) requires, for one regime block."""
    parts = []
    for family in ("linear", "nlg"):
        region, count, total = arm_region_occupancy(per_seed, topology, regime, family)
        parts.append(
            f"{FAMILY_DISPLAY[family]} {count}/{total} region {region}"
        )
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _paired_arms(per_seed: pd.DataFrame, topology: str, regime: str):
    """dcost_ld(linear, s) and dcost_ld(nlg, s), aligned on seed_idx (CRN)."""
    block = per_seed[
        (per_seed["topology"] == topology) & (per_seed["regime"] == regime)
    ]
    wide = block.pivot_table(
        index="seed_idx", columns="family", values="dcost_ld"
    ).sort_index()
    return (
        wide["linear"].to_numpy(dtype=float),
        wide["nlg"].to_numpy(dtype=float),
    )


def _draw_panel(
    axis,
    per_seed: pd.DataFrame,
    topology: str,
    headroom: float,
    annotation_y: float,
    wrap: int,
    fontsize: float,
    instances: pd.DataFrame | None = None,
) -> None:
    """One topology's two paired-arm blocks, with recomputed gate annotations."""
    marker = TOPOLOGY_MARKER[topology]
    values = []
    for regime in REGIMES:
        linear, nlg = _paired_arms(per_seed, topology, regime)
        values.append(np.concatenate([linear, nlg]))
        x_linear, x_nlg = BLOCK_X[regime]
        colour = REGIME_COLOUR[regime]
        for a, b in zip(linear, nlg, strict=True):
            axis.plot(
                [x_linear, x_nlg], [a, b],
                color=colour, alpha=0.30, linewidth=0.9, zorder=2,
            )
        for x, arm, family in ((x_linear, linear, "linear"), (x_nlg, nlg, "nlg")):
            axis.plot(
                np.full(arm.size, x), arm,
                linestyle="none", marker=marker, markersize=4.0,
                markerfacecolor=colour if FAMILY_FILLED[family] else "white",
                markeredgecolor=colour, markeredgewidth=0.9,
                alpha=0.85, zorder=3,
            )

    pooled = np.concatenate(values)
    low, high = float(pooled.min()), float(pooled.max())
    span = (high - low) or 1.0
    axis.set_ylim(low - 0.12 * span, high + headroom * span)
    # Region-3 SIDE of the frame: below the L1-oracle reference. A reading aid —
    # per-seed region membership is the frozen column, not this band.
    axis.axhspan(
        axis.get_ylim()[0], 0.0, facecolor=REGION_3_FILL, edgecolor="none", zorder=0
    )
    axis.axhline(0.0, color="0.55", linewidth=0.8, zorder=1)
    axis.axvline(
        DIVIDER_X, color="0.55", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1
    )

    for regime in REGIMES:
        counts = gate_counts(per_seed, topology, regime)
        lines = [REGIME_DISPLAY[regime]]
        lines += textwrap.wrap(
            f"interaction event {counts['events']}/{counts['n_seeds']} "
            f"(gate ≥{SIGN_GATE_K})",
            wrap,
        )
        if counts["ties"]:
            lines += textwrap.wrap(
                f"{counts['ties']}/{counts['n_seeds']} exact zero differences — "
                "non-firing under strict >",
                wrap,
            )
        # [PS-8 sub-clause (i)] PER-ARM region occupancy, on the figure itself. Read from
        # the frozen region column at render, so the annotation cannot claim an
        # occupancy the verdict layer did not record.
        lines += textwrap.wrap(arm_region_phrase(per_seed, topology, regime), wrap)
        axis.text(
            float(np.mean(BLOCK_X[regime])), annotation_y, "\n".join(lines),
            transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=fontsize, color="0.25", zorder=5,
        )

    if instances is not None and standing_halt(instances, topology):
        # [LD-2] A standing halt is a PAGE fact, not merely an
        # internal switch that removes a label: a reader must not have to infer a
        # halt from a missing verdict string.
        axis.text(
            0.5, 0.5, "HARNESS HALT STANDS (LD-2)",
            transform=axis.transAxes, ha="center", va="center",
            fontsize=8, color="#B03030", zorder=7,
            bbox={"facecolor": "white", "edgecolor": "#B03030", "linewidth": 0.8,
                  "boxstyle": "round,pad=0.35", "alpha": 0.94},
        )

    ticks = [x for regime in REGIMES for x in BLOCK_X[regime]]
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        ["linear", "nonlinear-Gaussian", "linear", "nonlinear-Gaussian"],
        fontsize=fontsize, rotation=30, ha="right",
    )
    axis.set_xlim(BLOCK_X["additive"][0] - 0.6, BLOCK_X["effect_modifying"][1] + 0.6)
    axis.set_ylabel(
        "ΔCost_ld\n(common-found, four-way)", fontsize=7
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def build_figure(per_seed: pd.DataFrame, instances: pd.DataFrame) -> plt.Figure:
    assert_d3_text_matches_the_analysis_copy()
    # Mosaic rather than a bare gridspec: the collider spans the top row (it is
    # the confirmatory-primary channel) while chain and triangle keep SEPARATE
    # axes below, because PS-8 forbids reading cost levels across topologies and
    # one shared y-axis would invite exactly that.
    figure, panels = plt.subplot_mosaic(
        [[ORIENTATION_TOPOLOGY, ORIENTATION_TOPOLOGY], list(CORROBORATING_TOPOLOGIES)],
        figsize=(6.3, 7.6),
        gridspec_kw={"height_ratios": [1.55, 1.0]},
    )

    top = panels[ORIENTATION_TOPOLOGY]
    _draw_panel(
        top, per_seed, ORIENTATION_TOPOLOGY,
        headroom=0.95, annotation_y=0.78, wrap=34, fontsize=7,
        instances=instances,
    )
    top.set_title(
        f"{ORIENTATION_TOPOLOGY} — orientation channel (PS-8, "
        "confirmatory-primary) · four-way",
        fontsize=8, pad=6,
    )
    # [PS-8] MANDATORY, on the figure itself — not caption-only. Parked in the
    # headroom reserved above the data so it never overlaps a seed line.
    top.text(
        0.015, 0.985, textwrap.fill(D3_ON_FIGURE_TEXT, 78),
        transform=top.transAxes, ha="left", va="top",
        fontsize=8, color="0.25", zorder=6,
        bbox={"facecolor": "white", "edgecolor": "0.80", "linewidth": 0.6,
              "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )

    for topology in CORROBORATING_TOPOLOGIES:
        axis = panels[topology]
        _draw_panel(
            axis, per_seed, topology,
            headroom=0.80, annotation_y=0.995, wrap=24, fontsize=5.5,
            instances=instances,
        )
        axis.set_title(
            f"{topology} · four-way\n"
            "corroborating — non-gating (PS-8)",
            fontsize=7, pad=4,
        )

    handles = [
        plt.Line2D([], [], color=REGIME_COLOUR["additive"], linewidth=1.4,
                   label="additive"),
        plt.Line2D([], [], color=REGIME_COLOUR["effect_modifying"], linewidth=1.4,
                   label="effect-modifying"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=5,
                   label="linear arm (filled)"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.35", markersize=5,
                   markerfacecolor="white", label="nonlinear-Gaussian arm (hollow)"),
        Patch(facecolor=REGION_3_FILL, edgecolor="0.8",
              label="below the L1-oracle reference (region-3 side)"),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
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
        path = out / f"fig7_h3_interaction.{suffix}"
        meta = {"metadata": {"CreationDate": None}} if suffix == "pdf" else {}
        figure.savefig(path, **meta)
        written.append(path)
    plt.close(figure)

    # Every number in the caption is read off the data at render time, so the
    # caption cannot drift from the figure it describes.
    verdict, passing = orientation_verdict(per_seed, instances)
    additive = gate_counts(per_seed, ORIENTATION_TOPOLOGY, "additive")
    modifying = gate_counts(per_seed, ORIENTATION_TOPOLOGY, "effect_modifying")
    em_region3, em_total = region_3_occupancy(
        per_seed, ORIENTATION_TOPOLOGY, "effect_modifying"
    )
    # [PS-8 sub-clause (i)] One phrase per family ARM of the orientation channel, each read
    # from the frozen region column. Named em_/add_ so the caption template reads
    # as prose rather than as an index into a tuple.
    arms = {}
    for regime in REGIMES:
        prefix = "em" if regime == "effect_modifying" else "add"
        for family in ("linear", "nlg"):
            region, count, total = arm_region_occupancy(
                per_seed, ORIENTATION_TOPOLOGY, regime, family
            )
            arms[f"{prefix}_{family}_arm"] = (
                f"{FAMILY_DISPLAY[family]} {count}/{total} in region {region}"
            )

    corroborating_parts = []
    for topology in CORROBORATING_TOPOLOGIES:
        for regime in REGIMES:
            counts = gate_counts(per_seed, topology, regime)
            part = f"{topology} {REGIME_DISPLAY[regime]} {counts['events']}/{counts['n_seeds']}"
            if counts["ties"]:
                part += f" ({counts['ties']} exact ties)"
            corroborating_parts.append(part)
    corroborating = "; ".join(corroborating_parts)
    text = CAPTION_TEMPLATE.format(
        denominator=DENOMINATOR_LABEL,
        denominator_n=N_VALID_REQUIRED,
        n_seeds=N_SEEDS_FULL,
        gate=SIGN_GATE_K,
        add_events=additive["events"],
        add_ties=additive["ties"],
        em_events=modifying["events"],
        verdict=verdict,
        passing=passing,
        d3=D3_ON_FIGURE_TEXT,
        collider_em_region3=em_region3,
        collider_em_total=em_total,
        corroborating=corroborating,
        **arms,
    )
    caption = out / "fig7_h3_interaction.caption.txt"
    caption.write_text(_wrap(text), encoding="utf-8", newline="")
    written.append(caption)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
