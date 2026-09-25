"""Analysis pass over a cross-seed grid. Runs NO experiments.

Reads a grid root (per-seed aggregates, per-individual scoring CSVs and manifests)
and emits, into ``<root>/summary/``:

  t4a_h1_monotonicity.md H1 signed-monotonicity + the PS-5 criterion read
  t4b_validity_gap.md H2 / ValidityDisp + Gap_c + the gap ladder read
  t4c_detectability_gate.md PS-2 formal pass/fail record
  detectability_gate_provenance.csv PS-2 consolidated machine-readable gate record
  mechanism_diagnostic.md PS-9 EXPLORATORY mechanism-discrimination diagnostic

Run with ``-m`` from the repo root:

    python -m scripts.run_t4_analysis --root results/cross_seed_N20 --n-seeds 20 \\
        --reports t4a t4b t4c

``--n-seeds`` is REQUIRED for any H1/H2 path and is asserted against the tree. The
DETECTABILITY GATE is NOT governed by it: per PS-1 / PS-2 it is always computed over
the seed-0..5 prefix and always stamps ``n_seeds_evaluated_for_gate = 6``.

Every target is refused if it already exists, before the first write (see
``_refuse_if_frozen``). ``results/cross_seed_N6/`` is a frozen PS-1 early-look audit
object; pointing this script at it refuses rather than re-emitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from icknowledge.analysis import mechanism
from icknowledge.analysis.loading import (
    DEFAULT_ROOT,
    VALID_SUBSET_MIN_N,
    cell_frame,
    grid_cells,
    load_summary,
)
from icknowledge.analysis.t4_reports import (
    G2_CRITERION,
    G3_CRITERION,
    GATE_SEEDS,
    assert_gate_matches_frozen,
    cost_table,
    detectability_gate,
    gap_cost_check,
    gate_provenance,
    h1_g2_verdict,
    h1_rung_bands,
    h1_topology_read,
    h2_g3_verdict,
    h2_gap_ladder,
    h2_topology_read,
    monotonicity_by_cell,
    monotonicity_by_group,
    no_op_summary,
    reversal_table,
    seed_prefix,
    valid_subset_guard,
    validity_disp,
)

CELL_KEYS = ["topology", "family", "regime"]

#: Every artifact this driver can emit, in the order it writes them.
REPORTS = ("t4a", "t4b", "t4c", "mechanism")

#: What each report writes into ``<root>/summary/``.
ARTIFACTS: dict[str, tuple[str, ...]] = {
    "t4a": ("t4a_h1_monotonicity.md",),
    "t4b": ("t4b_validity_gap.md",),
    "t4c": ("t4c_detectability_gate.md", "detectability_gate_provenance.csv"),
    "mechanism": ("mechanism_diagnostic.md",),
}

#: The N=6 artifacts a re-emission at N=20 supersedes, named BY PATH. Mirrors the
#: LD-2 supersession-header discipline: the superseded files are audit objects and
#: are never overwritten, so the new artifact has to say what it replaces.
SUPERSEDED = {
    "t4a": ("results/cross_seed_N6/summary/t4a_h1_monotonicity.md",),
    "t4b": ("results/cross_seed_N6/summary/t4b_validity_gap.md",),
    "t4c": (
        "results/cross_seed_N6/summary/t4c_detectability_gate.md",
        "results/cross_seed_N20/summary/chain_detectability_gate.md",
    ),
}


def _refuse_if_frozen(summary: Path, names: tuple[str, ...]) -> None:
    """Refuse the whole run if any target artifact already exists.

    Checked BEFORE the first write, so a refused run leaves nothing half-replaced.
    ``results/cross_seed_N6/`` is the case this exists for: its analysis artifacts are the
    PS-1 early-look record the thesis cites, and re-pointing this script at that root
    must fail loudly rather than regenerate them under a later code revision.
    """
    existing = [summary / name for name in names if (summary / name).exists()]
    if existing:
        listed = ", ".join(str(p) for p in existing)
        raise SystemExit(
            f"FATAL: {listed} already exist(s) and is a FROZEN audit object "
            "(PS-1 — the early-look analysis artifacts are cited as they stand). Nothing was "
            "written. If a genuine re-emission is intended, remove or rename the "
            "frozen artifact deliberately first, or point --root at a different tree."
        )


def _assert_seed_count(frame: pd.DataFrame, n_seeds: int, name: str) -> None:
    """The tree must carry exactly ``n_seeds`` seeds — never inferred, always checked.

    H1 and H2 are cross-seed reads, so a silently-short tree would produce a real
    number computed over the wrong denominator. `--n-seeds` is the operator's claim
    about the tree; this is where the claim is tested.
    """
    present = sorted(int(v) for v in frame["seed_idx"].unique())
    if present != list(range(n_seeds)):
        raise SystemExit(
            f"FATAL: {name} carries seed indices {present}, not 0..{n_seeds - 1}. "
            "--n-seeds is asserted against the tree, never inferred from it."
        )


def _md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Markdown table with GFM pipe-escaping inside every cell (RP lesson)."""
    columns = columns or list(frame.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.6f}" if abs(value) < 1e6 else f"{value:.3e}"
            elif isinstance(value, bool):
                text = "YES" if value else "NO"
            else:
                text = str(value)
            cells.append(text.replace("|", r"\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# H1
# --------------------------------------------------------------------------- #


def _supersession_block(kind: str, n_seeds: int, n_cells: int) -> list[str]:
    """Name the N=6 artifacts this re-emission supersedes, BY PATH (LD-2 style).

    The superseded files are audit objects: they are named, not overwritten.
    """
    paths = SUPERSEDED.get(kind, ())
    if not paths:
        return []
    return [
        "## SUPERSEDES",
        "",
        *[f"- `{path}`" for path in paths],
        "",
        f"Those artifacts are **audit objects** and are NOT overwritten. They record "
        f"the **PS-1 early look** — 8 cells × 6 seeds, before the chain joined the "
        f"grid at the chain SCM specification. This artifact is the **full registered scope**: "
        f"{n_cells} cells × {n_seeds} seeds. Where the two disagree, the disagreement "
        "is reported below rather than resolved in favour of the earlier read.",
        "",
    ]


def _outcome_exposed_block(criterion_name: str, criterion: str) -> list[str]:
    """State plainly that the rule is pre-data and this application of it is not."""
    return [
        "## Outcome exposure — stated, not claimed away",
        "",
        f"**{criterion_name} was register-logged prior to outcome "
        "exposure.** It is quoted below as it now stands. No threshold, "
        "band, quantifier or population in it was moved for this read: "
        "every one applied here is the one on record before these "
        "numbers were computed.",
        "",
        f"> {criterion}",
        "",
        "**This application of it is OUTCOME-EXPOSED.** The numbers existed before "
        "this document was generated, and the choice to compute the read now, on this "
        "tree, was made with the N=6 early look already known. That is disclosed "
        "rather than described away: this is **not** an outcome-independent read, and "
        "nothing here should be cited as one. What is pre-committed is the RULE; what "
        "is post-data is the OCCASION of applying it.",
        "",
    ]


def _comparison_block(kind: str, compare: dict | None, new: pd.DataFrame) -> list[str]:
    """Per-topology N=6 → N=20 comparison, including topologies that are NEW at N=20.

    Emitted for its disagreements, not its agreements. A topology whose bands
    separated at N=6 and overlap at N=20 is stated as a FLIP in those words; a
    topology with no N=6 counterpart (the chain, which joined the grid at the chain SCM
    specification) is
    stated as first-time and uncorroborated.
    """
    lines = ["## Comparison against the N=6 early look", ""]
    if compare is None:
        return [
            *lines,
            "**Not computed.** No comparison root was available on this run "
            "(`--compare-root` absent or the tree is not present), so the N=6 → N=20 "
            "comparison is UNAVAILABLE rather than assumed to agree.",
            "",
        ]
    if compare.get("unavailable"):
        return [
            *lines,
            f"**Not computed.** {compare['unavailable']}",
            "",
        ]

    old = compare[f"{kind}_old"]
    column = compare[f"{kind}_column"]
    old_map = dict(zip(old["topology"], old[column], strict=True))
    rows, flips, first_time = [], [], []
    for _, row in new.iterrows():
        topology = row["topology"]
        before = old_map.get(topology)
        after = bool(row[column])
        if before is None:
            state = "NEW at N=20 — no N=6 counterpart"
            first_time.append(topology)
        elif bool(before) == after:
            state = "unchanged"
        else:
            state = f"**FLIP** — {bool(before)} at N=6 → {after} at N=20"
            flips.append((topology, bool(before), after))
        rows.append(
            {
                "topology": topology,
                f"{column}_N6": "n/a" if before is None else ("YES" if before else "NO"),
                f"{column}_N20": "YES" if after else "NO",
                "change": state,
            }
        )

    lines += [
        f"Comparison root: `{compare['root']}` (N={compare['n_seeds']}, "
        f"{compare['n_cells']} cells). Verdict quantity: `{column}`.",
        "",
        _md(pd.DataFrame(rows)),
        "",
    ]

    # The verdict boolean alone can hide a real movement underneath it — a count that
    # changed while the verdict did not. The supporting counts are therefore compared
    # too, so "unchanged" above is never mistaken for "nothing moved".
    detail_columns = compare.get(f"{kind}_detail_columns", ())
    if detail_columns:
        detail_rows = []
        for _, row in new.iterrows():
            before = old[old["topology"] == row["topology"]]
            record = {"topology": row["topology"]}
            for name in detail_columns:
                now = str(row[name])
                if before.empty:  # NEW topology — nothing to have moved from
                    record[name] = f"n/a → {now}"
                    continue
                was = str(before.iloc[0][name])
                record[name] = f"{was} → {now}" + (" **(moved)**" if was != now else "")
            detail_rows.append(record)
        lines += [
            "**Supporting counts, N=6 → N=20.** A verdict that did not move can still "
            "sit on numbers that did; those are shown rather than left implicit.",
            "",
            _md(pd.DataFrame(detail_rows)),
            "",
        ]
    if flips:
        lines += [
            "**Flips, stated plainly:**",
            "",
            *[
                f"- **{t}**: `{column}` was {a} at N=6 and is {b} at N=20. The N=20 "
                "read is the one on the full registered scope; it is NOT reconciled "
                "toward the early look."
                for t, a, b in flips
            ],
            "",
        ]
    else:
        lines += ["No flip on any topology present in both reads.", ""]
    if first_time:
        lines += [
            f"**First-time result — {', '.join(first_time)}.** This topology joined the "
            "grid at the chain SCM specification and has never appeared in a H1/H2 report. Its "
            "read here is "
            "**uncorroborated by any prior look**: there is no early-look counterpart "
            "to agree or disagree with it, so it carries no replication.",
            "",
        ]
    return lines


def report_t4a(
    by_group: pd.DataFrame,
    by_cell: pd.DataFrame,
    n_seeds: int,
    compare: dict | None = None,
) -> tuple[str, dict]:
    summary = no_op_summary(by_group)
    costs = cost_table(by_group)
    by_group_mono = monotonicity_by_group(by_group)
    by_cell_mono = monotonicity_by_cell(by_cell)
    reversal = reversal_table(by_cell)
    bands = h1_rung_bands(by_group)
    topology_read = h1_topology_read(bands)
    verdict = h1_g2_verdict(topology_read)
    n_cells = len(grid_cells())

    wide = costs.pivot_table(
        index=[*CELL_KEYS, "group"],
        columns="condition",
        values=["cost_mean", "cost_sd"],
    ).reset_index()
    wide.columns = [
        c[0] if not c[1] else f"{c[1]}_{c[0].replace('cost_', '')}"
        for c in wide.columns
    ]

    lines = [
        f"# H1 monotonicity and the PS-5 criterion read (N={n_seeds}, {n_cells} cells)",
        "",
        "DV: realized cost over the **common-found three-way** population",
        f"(the common-found population primary). Mean ± SD over the {n_seeds} seeds of this grid.",
        "",
        *_supersession_block("t4a", n_seeds, n_cells),
        *_outcome_exposed_block("PS-5", G2_CRITERION),
        "## 1. Common-found realized cost, mean ± SD over seeds",
        "",
        _md(wide),
        "",
        "## 2. The common-found no-op — reported as a FINDING, not a caveat",
        "",
        f"Rows checked: **{summary['n_rows']}** ({n_cells} cells × 3 conditions × 2 "
        f"groups × {n_seeds} seeds).",
        "",
        "- `max |N_common_found_threeway − N_eligible|` = "
        f"**{summary['max_abs_N_diff_common_vs_eligible']}**",
        "- `max |N_found − N_eligible|` = "
        f"**{summary['max_abs_N_diff_found_vs_eligible']}**",
        f"- `max |cost(common-found) − cost(all-eligible)|` = "
        f"**{summary['max_abs_cost_diff_common_vs_all_eligible']:.3e}**",
        f"- exact equality across every row: **{'YES' if summary['exact_equality'] else 'NO'}**",
        "",
        "**Substantive reading.** Found is complete under exhaustive grid search on",
        "this DGP, so the common-found primary coincides *exactly* with the",
        "all-eligible legacy read. This is a statement about the design, not a",
        "technicality: **recourse failure here is always VALIDITY failure, never",
        "FINDABILITY failure** — the brute-force procedure always returns a feasible",
        "action; that action may fail to flip the true classifier, but it is never",
        "absent. The common-found population's machinery is correct and binding in "
        "principle; it",
        "resolves to equality on this grid because the DGP makes findability trivial.",
        "",
        "## 3. H1 signed monotonicity — per cell × group (signed primary)",
        "",
        "Expected: `cost(L2) ≤ cost(L1-oracle) ≤ cost(L0)`. `diff = lower − upper`;",
        "the rung holds when `diff_mean ≤ 0`. `within_seed_noise` flags a rung whose",
        "`|mean|` does not exceed its seed SD.",
        "",
        _md(
            by_group_mono,
            [
                *CELL_KEYS,
                "group",
                "rung",
                "diff_mean",
                "diff_sd",
                "holds_on_mean",
                "sign_consistent_across_seeds",
                "n_seeds_holding",
                "abs_mean_over_sd",
                "within_seed_noise",
            ],
        ),
        "",
        "## 4. Same ladder on the cell-level Δ_cost (common-found)",
        "",
        _md(
            by_cell_mono,
            [
                *CELL_KEYS,
                "rung",
                "diff_mean",
                "diff_sd",
                "holds_on_mean",
                "sign_consistent_across_seeds",
                "abs_mean_over_sd",
                "within_seed_noise",
            ],
        ),
        "",
        "## 5. The L2 → L1-oracle reversal, per cell, with its seed SD",
        "",
        "`[Δ_cost(L1-oracle) − Δ_cost(L2)]`. A **positive** sign means the ladder",
        "ordering holds at cell level; **negative** is the L2→L1-oracle reversal",
        "(L1-oracle buys cheaper actions that fail the true classifier more often).",
        "",
        _md(
            reversal,
            [
                *CELL_KEYS,
                "per_seed",
                "delta_mean",
                "delta_sd",
                "sign",
                "sign_consistent",
                "abs_mean_over_sd",
                "within_seed_noise",
            ],
        ),
        "",
        "## 6. H1 — the mechanical PS-5 criterion read, per topology",
        "",
        "The criterion (PS-5) asks two things of the ladder: that the **ordering** "
        "`L2 ≤ L1-oracle ≤ L0`",
        "hold, and that its **variability be non-overlapping**. Both are reported, and",
        "the variability question is reported under BOTH available readings rather than",
        "one being chosen silently:",
        "",
        "- **paired** — `|mean(diff)| / SD(diff)` on the per-seed difference. Seeds are",
        "  shared across conditions, so this is the tighter and more powerful reading,",
        "  and it is the one reported here. Separated iff the ratio > 1.",
        "- **marginal** — `band_gap = (mean_upper − SD_upper) − (mean_lower + SD_lower)`,",
        "  in raw cost units. **Positive** = the two ±1 SD bands are disjoint, and the",
        "  value is the clear space between them. **Negative** = they overlap, and",
        "  `band_overlap_width` is by how much.",
        "",
        "A rung counts as separated only when the ordering ALSO holds on the mean: a",
        "disjoint band on the wrong side of the ordering is a violation, not a pass.",
        "",
        "### 6.1 Per (cell, group, rung) — the numbers",
        "",
        _md(
            bands,
            [
                *CELL_KEYS,
                "group",
                "rung",
                "diff_mean",
                "diff_sd",
                "holds_on_mean",
                "n_seeds_holding",
                "abs_mean_over_sd",
                "band_gap",
                "bands_overlap",
                "band_overlap_width",
                "paired_separates",
                "marginal_separates",
            ],
        ),
        "",
        "### 6.2 Rolled up to the topology — the unit the criterion names",
        "",
        "A **series** is one (cell, group). It is *ordered* when both rungs hold on the",
        "mean, and *separated* when both rungs also separate under the stated reading.",
        "",
        _md(
            topology_read,
            [
                "topology",
                "n_series",
                "n_series_ordered",
                "n_series_paired_separated",
                "n_series_marginal_separated",
                "binding_rung",
                "max_abs_mean_over_sd_on_binding_rung",
                "min_band_gap",
                "max_band_overlap_width",
                "any_series_meets_h1_criterion",
                "all_series_meet_h1_criterion",
            ],
        ),
        "",
        "### 6.3 VERDICT under the pinned rule",
        "",
        f"> **PS-5:** {G2_CRITERION}",
        "",
        f"**{verdict['label']}.**",
        "",
        "The criterion makes the topology the unit and quantifies existentially ACROSS topologies",
        "(\"at least one\"). It does not say how the cells and groups WITHIN a topology",
        "combine, so the verdict is computed under both the weak reading (at least one",
        "series in the topology separates) and the strong reading (every series does):",
        "",
        f"- topologies meeting the criterion, **weak** reading: "
        f"**{verdict['topologies_meeting_weak_reading'] or 'none'}**",
        f"- topologies meeting the criterion, **strong** reading: "
        f"**{verdict['topologies_meeting_strong_reading'] or 'none'}**",
        f"- the two readings agree: **{'YES' if verdict['reading_invariant'] else 'NO'}**"
        + (
            " — so the verdict does not depend on which within-topology aggregation is"
            " adopted, and no aggregation choice is doing any work here."
            if verdict["reading_invariant"]
            else " — the verdict DEPENDS on the aggregation, which is a register"
            " question, not a code one. Escalate before citing either reading."
        ),
        "",
        "*No prose beyond the rule is written here: the PS-5 text is the whole criterion,*",
        "*and the table above is the whole evidence.*",
        "",
        *_comparison_block("t4a", compare, topology_read),
    ]
    return "\n".join(lines), {
        "no_op": summary,
        "reversal": reversal,
        "mono_group": by_group_mono,
        "costs": wide,
        "bands": bands,
        "topology_read": topology_read,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# H2
# --------------------------------------------------------------------------- #


def report_t4b(
    by_group: pd.DataFrame,
    by_cell: pd.DataFrame,
    n_seeds: int,
    compare: dict | None = None,
) -> tuple[str, dict]:
    disps = validity_disp(by_group)
    gap_c = gap_cost_check(by_cell)
    guard = valid_subset_guard(by_group)
    ladder = h2_gap_ladder(by_cell)
    topology_read = h2_topology_read(ladder)
    verdict = h2_g3_verdict(topology_read)
    n_cells = len(grid_cells())

    collider = disps[disps["topology"] == "collider"]
    per_seed_lines = []
    for _, row in disps.iterrows():
        values = " , ".join(f"{v:.4f}" for v in row["validity_neg_per_seed"])
        per_seed_lines.append(
            f"| {row['topology']} | {row['family']} | {row['regime']} | {values} |"
        )

    flagged = guard[guard["n_seeds_flagged_small"] > 0]

    lines = [
        f"# H2 / ValidityDisp and the Gap_c read (N={n_seeds}, {n_cells} cells)",
        "",
        *_supersession_block("t4b", n_seeds, n_cells),
        *_outcome_exposed_block("Gap_c (PS-6)", G3_CRITERION),
        "## 1. ValidityDisp at L1-oracle = realized validity(A=−1) − realized validity(A=+1)",
        "",
        _md(
            disps,
            [
                *CELL_KEYS,
                "validity_neg_mean",
                "validity_neg_sd",
                "validity_pos_mean",
                "validity_pos_sd",
                "disp_mean",
                "disp_sd",
                "sd_ge_abs_mean_disp",
                "sd_ge_abs_mean_validity_neg",
                "n_high_validity_seeds",
                "n_collapse_seeds",
            ],
        ),
        "",
        "### Per-seed A=−1 realized validity at L1-oracle (seed order)",
        "",
        "The distribution shape is the finding: on the collider the series switches",
        "near-binary rather than varying smoothly, so the moments above misdescribe it.",
        "",
        f"| topology | family | regime | seed {' , '.join(str(i) for i in range(n_seeds))} |",
        "|---|---|---|---|",
        *per_seed_lines,
        "",
        f"Collider cells with `SD ≥ |mean|` on the **ValidityDisp**: "
        f"**{int(collider['sd_ge_abs_mean_disp'].sum())} / {len(collider)}** — this is",
        "the seed-fragility statement PS-9 was opened on, and it reproduces.",
        "",
        f"Collider cells with `SD ≥ |mean|` on the raw **A=−1 validity level**: "
        f"**{int(collider['sd_ge_abs_mean_validity_neg'].sum())} / {len(collider)}**.",
        "The two flags do not agree, and the distinction matters: the *disparity* is",
        "fragile in every collider cell, while the *level* only reads as fragile",
        "where enough seeds have actually collapsed to ~0.",
        "",
        "> **Read the flag with its cause.** `SD ≥ |mean|` fires on the triangle cells",
        "> too, but for the opposite reason: there the disparity is ~0 with ~0",
        "> dispersion, so the ratio is uninformative — the finding is that the",
        "> asymmetry is ABSENT, not that it is fragile. On the collider the disparity",
        "> is large (|mean| 0.20–0.36) AND its SD exceeds it, which is genuine seed",
        "> fragility. The flag alone does not distinguish the two; the `disp_mean`",
        "> column beside it does.",
        "",
        "## 2. Gap_c confirmation (PS-6): zero at L1-oracle and L2, nonzero at L0",
        "",
        _md(
            gap_c,
            [
                *CELL_KEYS,
                "max_abs_Gap_cost_L0",
                "min_abs_Gap_cost_L0",
                "max_abs_Gap_cost_L1-oracle",
                "max_abs_Gap_cost_L2",
                "zero_at_L1_and_L2",
                "nonzero_at_L0_all_seeds",
            ],
        ),
        "",
        "## 3. valid-subset small-N guard",
        "",
        f"Threshold: `N_valid_subset < {VALID_SUBSET_MIN_N}` at L1-oracle. On a flagged",
        "(cell, group, seed) the valid-subset cost is a near-empty average and its",
        "contribution to the common-found population **secondary** contrast is unreliable. The "
        "cross-seed",
        f"cost is therefore given twice — over all {n_seeds} seeds, and over the seeds",
        "clearing the threshold — so small-N contamination is separable from cost signal.",
        "",
        _md(
            guard,
            [
                *CELL_KEYS,
                "group",
                "N_valid_subset_per_seed",
                "n_seeds_flagged_small",
                "flagged_seed_idx",
                "collapse_seeds_passing_guard",
                "cost_mean_all_seeds",
                "cost_sd_all_seeds",
                "cost_mean_guarded",
                "cost_sd_guarded",
                "n_seeds_guarded",
            ],
        ),
        "",
        f"Flagged (cell, group) rows: **{len(flagged)} / {len(guard)}**.",
        "",
        "> **Limit of the guard as specified.** `N_valid_subset ≥ 30` is an absolute",
        "> COUNT threshold, not a validity-rate one, so a collapse seed with a large",
        "> eligible pool can clear it — `collapse_seeds_passing_guard` names every seed",
        "> that passes the count guard while its realized validity is below 0.5. Those",
        "> seeds still enter the guarded mean, so the guarded column is *less*",
        "> contaminated than the all-seed column, not contamination-free.",
        "",
        "## 4. H2 — the mechanical Gap_c read, per topology",
        "",
        "The quantity is **the per-condition gap `|believed − realized|`** — PS-6's",
        "`Gap_cost`. The ladder is read in the criterion's direction, knowledge DECREASING:",
        "`L2 → L1-oracle → L0`.",
        "",
        "### 4.1 The gap ladder, per cell × condition",
        "",
        _md(
            ladder,
            [
                *CELL_KEYS,
                "condition",
                "n_seeds",
                "abs_gap_mean",
                "abs_gap_sd",
                "abs_gap_min",
                "abs_gap_max",
            ],
        ),
        "",
        "### 4.2 Rolled up to the topology, clause by clause",
        "",
        "The criterion's clauses are counted separately because they can disagree.",
        "`non_decreasing` and `largest_at_L0` are what \"grows as knowledge decreases,",
        "largest at L0\" asserts; `strictly_increasing` is the stronger reading in which",
        "EVERY rung grows; `flat` is the criterion's own disconfirming case.",
        "",
        _md(
            topology_read,
            [
                "topology",
                "n_cells",
                "n_non_decreasing",
                "n_largest_at_L0",
                "n_strictly_increasing",
                "n_flat",
                "meets_h2_criterion",
            ],
        ),
        "",
        "### 4.3 VERDICT under the pinned rule",
        "",
        f"> **Gap_c:** {G3_CRITERION}",
        "",
        f"**{verdict['label']}** — satisfied on "
        f"{len(verdict['topologies_meeting'])}/{verdict['n_topologies']} topologies "
        f"({', '.join(verdict['topologies_meeting']) or 'none'}).",
        "",
        f"- cells with a **flat** gap (the disconfirming case): "
        f"**{verdict['n_flat_cells']}**",
        f"- cells where the gap is **strictly increasing at every rung**: "
        f"**{verdict['n_strictly_increasing_cells']}**",
        "",
        "**Structure of the satisfied ladder, stated rather than smoothed.** PS-6 makes",
        "`Gap_cost` exactly **zero** at L1-oracle and at L2 *by construction* — at those",
        "rungs the acting model is the true one, so believed and realized coincide. The",
        "ladder that satisfies Gap_c here is therefore `0 = 0 < positive`: **non-decreasing",
        "and strictly largest at L0, but not strictly increasing at every rung.**",
        "The criterion's operative clauses (\"largest at L0\"; \"a flat gap disconfirms\") are",
        "met; its \"continuum\" language is met only in the two-point sense the design permits.",
        "The empirical content of this table is the **L0 magnitude** and the fact that it",
        "is nonzero on every cell — not the two zeros above it, which are structural.",
        "",
        *_comparison_block("t4b", compare, topology_read),
    ]
    return "\n".join(lines), {
        "disps": disps,
        "gap_c": gap_c,
        "guard": guard,
        "ladder": ladder,
        "topology_read": topology_read,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# detectability gate
# --------------------------------------------------------------------------- #


def report_t4c(
    by_cell: pd.DataFrame, s_of_g: pd.DataFrame, n_seeds: int
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """The PS-1-frozen gate, always over the seed-0..5 prefix — never re-evaluated.

    ``n_seeds`` is the size of the TREE, used only to say so in the prose. The gate
    itself is computed over :data:`GATE_SEEDS` seeds regardless: PS-1 forecloses
    re-evaluating it on a growing seed count, and the seed-generation prefix property
    makes the
    prefix reproduce the frozen record exactly. That reproduction is ASSERTED here,
    not assumed — a drift raises rather than being written out as a new gate.
    """
    gate = detectability_gate(
        seed_prefix(by_cell, GATE_SEEDS), seed_prefix(s_of_g, GATE_SEEDS), n_seeds=GATE_SEEDS
    )
    drift_check = assert_gate_matches_frozen(gate)
    provenance = gate_provenance(gate)
    n_pass = int(gate["gate_pass"].sum())
    lines = [
        "# Detectability-gate formal record (PS-2)",
        "",
        "Formal discharge of **PS-2** for the collider and the analogous check for",
        f"every cell, evaluated on the **PS-1 first-{GATE_SEEDS}-seed early look**.",
        "",
        f"> **This is a {GATE_SEEDS}-seed gate on an N={n_seeds} tree, and that is the",
        "> discipline working, not a stale number.** PS-1 forecloses re-evaluating the",
        "> gate on a growing seed count — doing so would reintroduce exactly the",
        "> stopping-rule freedom the staged 6→20 design exists to prevent. The gate is",
        f"> computed over the seed-0..{GATE_SEEDS - 1} prefix (the seed-generation "
        f"prefix property),",
        "> and every cell is asserted equal to its frozen record before this document is",
        "> written. The `n_seeds_evaluated_for_gate` column below carries that fact into",
        "> the artifact, as the chain SCM specification requires.",
        "",
        f"- **(i)** `sign(Δ_cost(L2))` consistent across all {GATE_SEEDS} seeds **and**",
        "  `|mean(Δ_cost(L2))| > SD(Δ_cost(L2))`.",
        "- **(ii)** `sign(ΔS)` unambiguous under the **production** S(g). S(g) is",
        "  data-, classifier- and fit-independent by construction (PS-3), so it",
        "  must also be identical across seeds; that invariance is re-checked here",
        "  rather than assumed. In the **additive control** ΔS is flat *by",
        " construction* (PS-3 convention 4), so (ii) requires only the invariance.",
        "",
        _md(
            gate,
            [
                *CELL_KEYS,
                "mean",
                "sd",
                "abs_mean_over_sd",
                "cond_i_sign_consistent",
                "cond_i_magnitude",
                "cond_i_pass",
                "delta_S",
                "delta_S_identical_across_seeds",
                "cond_ii_pass",
                "cond_ii_note",
                "gate_pass",
                "n_seeds_evaluated_for_gate",
            ],
        ),
        "",
        "### Per-seed Δ_cost(L2)",
        "",
        _md(gate, [*CELL_KEYS, "delta_cost_L2_per_seed"]),
        "",
        f"**Record: {n_pass} / {len(gate)} cells PASS.**",
        "",
        "### Provenance — where each cell's gate was originally discharged",
        "",
        "The three frozen records cover the grid between them but none covers it alone.",
        "Every row below was re-derived from the seed-0..5 prefix of this tree and",
        "checked, field by field, against the record named in `source_record` at that",
        "record's own rendered precision. A mismatch would have raised before this",
        "document was written.",
        "",
        _md(
            drift_check,
            [
                *CELL_KEYS,
                "mean_frozen",
                "mean_first6",
                "sd_frozen",
                "sd_first6",
                "abs_mean_over_sd_frozen",
                "abs_mean_over_sd_first6",
                "delta_S_frozen",
                "delta_S_first6",
            ],
        ),
        "",
        "The machine-readable consolidation of this table — one row per cell, all three",
        "topologies, carrying `n_seeds_evaluated_for_gate` and `source_record` — is",
        "written alongside as `detectability_gate_provenance.csv`. It carries FROZEN",
        "FACTS ONLY and establishes no display convention for any figure.",
        "",
    ]
    if n_pass != len(gate):
        lines += [
            "> **STOP.** A failing cell triggers the SCM specification's fallback-lever protocol",
            "> (γ → β/η → β_A last resort). That is a register-logged DECISION, not",
            "> a code path — escalate before any downstream analysis uses the cell.",
            "",
        ]
    return "\n".join(lines), gate, provenance


# --------------------------------------------------------------------------- #


def _load_comparison(root: Path | None, n_seeds: int) -> dict | None:
    """The earlier read this one is compared against, or a stated reason it is absent.

    Read-only, and tolerant: a missing comparison tree makes the comparison section say
    UNAVAILABLE rather than silently claiming agreement. That tolerance is what keeps
    the suite green on a clean checkout, where results/ is gitignored and absent.
    """
    if root is None:
        return None
    try:
        by_group = load_summary("per_seed_by_group", root)
        by_cell = load_summary("per_seed_by_cell", root)
    except FileNotFoundError as error:
        return {"unavailable": f"comparison root `{root}` is not present ({error})."}

    present = sorted(int(v) for v in by_group["seed_idx"].unique())
    if present != list(range(n_seeds)):
        return {
            "unavailable": (
                f"comparison root `{root}` carries seed indices {present}, not "
                f"0..{n_seeds - 1} — refusing to compare against an unexpected tree."
            )
        }
    # The comparison tree is not required to have the same cells: the N=6 grid
    # predates the chain (the chain SCM specification) and carries 8, not 12. Enumerate what it
    # actually has,
    # so a topology missing there shows up as "NEW at N=20" rather than as an error.
    present_cells = [c for c in grid_cells() if not cell_frame(by_group, c).empty]
    return {
        "root": root.as_posix(),
        "n_seeds": n_seeds,
        "n_cells": len(present_cells),
        "t4a_old": h1_topology_read(h1_rung_bands(by_group, cells=present_cells)),
        "t4a_column": "any_series_meets_h1_criterion",
        "t4a_detail_columns": (
            "n_series_ordered",
            "n_series_paired_separated",
            "n_series_marginal_separated",
        ),
        "t4b_old": h2_topology_read(h2_gap_ladder(by_cell, cells=present_cells)),
        "t4b_column": "meets_h2_criterion",
        "t4b_detail_columns": ("n_non_decreasing", "n_largest_at_L0", "n_flat"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=True,
        help="seed count of --root. ASSERTED against the tree, never inferred: a "
        "H1/H2 path must never silently default to the PS-1 early-look 6.",
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        choices=REPORTS,
        default=list(REPORTS),
        help="which artifacts to emit. Narrowed when part of the target tree is "
        "already frozen — the guard refuses the run otherwise.",
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        default=None,
        help="an earlier tree to compare the H1/H2 reads against (typically "
        "results/cross_seed_N6). Absent or missing => the comparison section says so.",
    )
    parser.add_argument("--compare-n-seeds", type=int, default=6)
    args = parser.parse_args(argv)
    root: Path = args.root
    n_seeds: int = args.n_seeds
    selected = tuple(r for r in REPORTS if r in set(args.reports))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = root / "summary"
    targets = tuple(name for report in selected for name in ARTIFACTS[report])
    _refuse_if_frozen(out, targets)

    by_group = load_summary("per_seed_by_group", root)
    by_cell = load_summary("per_seed_by_cell", root)
    s_of_g = load_summary("s_of_g_by_seed", root)
    for name, frame in (
        ("per_seed_by_group", by_group),
        ("per_seed_by_cell", by_cell),
        ("s_of_g_by_seed", s_of_g),
    ):
        _assert_seed_count(frame, n_seeds, name)

    compare = _load_comparison(args.compare_root, args.compare_n_seeds)

    texts: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    gate = None
    if "t4a" in selected:
        texts["t4a_h1_monotonicity.md"], _ = report_t4a(by_group, by_cell, n_seeds, compare)
    if "t4b" in selected:
        texts["t4b_validity_gap.md"], _ = report_t4b(by_group, by_cell, n_seeds, compare)
    if "t4c" in selected:
        t4c_text, gate, provenance = report_t4c(by_cell, s_of_g, n_seeds)
        texts["t4c_detectability_gate.md"] = t4c_text
        frames["detectability_gate_provenance.csv"] = provenance
    if "mechanism" in selected:
        texts["mechanism_diagnostic.md"], verdicts = mechanism.build_report(root, grid_cells())
    else:
        verdicts = {}

    paths = [_write(out / name, text) for name, text in texts.items()]
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
        paths.append(out / name)

    print("=" * 78)
    print(f"Analysis pass over {root} (N={n_seeds}, no experiments re-run)")
    print("=" * 78)
    for text in texts.values():
        print(text)
    if verdicts:
        print("=" * 78)
        print("PS-9 per-cell verdicts")
        print("=" * 78)
        for key, verdict in verdicts.items():
            print(f"  {key}: {verdict}")
    print()
    for path in paths:
        print(f"wrote {path}")

    # A failed gate is an escalation, not a silent pass.
    if gate is None:
        return 0
    return 0 if bool(gate["gate_pass"].all()) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
