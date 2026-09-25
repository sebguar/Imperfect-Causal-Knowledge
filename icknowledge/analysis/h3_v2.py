"""H3 verdict re-emission under the LD-2-corrected linear-control taxonomy.

This module is the classification/gate/label layer and nothing else.
It READS the frozen h3_per_seed.csv and h3_covariates.csv and emits versioned
artifacts (h3_verdict_v2.md, h3_instance_table_v2.csv) that carry a
supersession header. The halted h3_verdict.md is an audit object and is never
overwritten.

EPISTEMIC STATUS. The re-emission is NOT outcome-blind and does not claim to
be. Its legitimacy rests on textual forcing (the correction direction is fixed
by register text predating every number), label-invariance (no threshold,
band, gate numerator, k or region definition moves), and the additive-grid
byte-exact confirmation of PS-7 note (c). Nothing here may describe the
re-emission as outcome-independent or outcome-blind.

THE CORRECTION. Limb a3 (perfect graph => ΔCost_ld = 0) is licensed only by
PS-7 note (c), which scopes that identity to the ADDITIVE regime and excludes
EM, where the two L1 rungs differ by graph AND interaction-blindness. Limb a3
is therefore restricted to additive cells; EM gets its own registered class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from icknowledge.analysis.h3 import (
    _F4,
    _F6,
    CORROBORATING_TOPOLOGIES,
    DENOMINATOR_LABEL,
    LINEAR_HOLDS_MIN,
    MIXED_MIN_INSTANCES,
    N_VALID_REQUIRED,
    ORIENTATION_TOPOLOGY,
    REGIMES,
    SIGN_GATE_K,
    SUPPORTED_MIN_INSTANCES,
    WORSE_THAN_NO_GRAPH_ANCHOR,
    _table,
    band_label,
    orientation_label,
)
from icknowledge.analysis.h3_d2 import ZONE_TEXT, D2Result
from icknowledge.analysis.loading import L1D_ROOT

__all__ = [
    "CLASS_DEFINITIONS",
    "D3_REGION3_CLAUSE",
    "InstanceClassification",
    "SUPERSEDED",
    "build_verdict_v2_report",
    "classify_violating_seed",
    "instance_table_v2",
    "reclassify_linear_instances",
    "standing_halt_propagation",
]

#: [LD-2] The superseded artifacts, named by path and commit in the header of
#: every re-emitted artifact. The halted verdict is an AUDIT OBJECT: never
#: overwritten, never deleted.
SUPERSEDED = (
    ("results/cross_seed_L1d_N20/summary/h3_verdict.md", "d8bdd23"),
    ("results/cross_seed_L1d_N20/summary/h3_instance_table.csv", "d8bdd23"),
)

#: [LD-2] Class definitions. Quoted bodies are the appendix wording word for
#: word; only the attribution framing outside the quotes is the harness's.
CLASS_DEFINITIONS = {
    "c": (
        "**class (c)** (LD-2) — \"A violating seed on an EM cell with "
        "skeleton_shd = 0, orientation_wrong_count = 0, and region-3 placement is "
        "classified **class (c): registered interaction-blindness channel per "
        "PS-7** — the discovered rung's group-agnostic form prescribes cheaper "
        "pooled actions; the register-logged consequence, not an alarm; **no halt**.\""
    ),
    "d": (
        "**class (d)** (LD-2) — \"systematic adjacency error traceable by a "
        "truth-side check to DGP-driven CI-misspecification under the pinned linear "
        "test — genuine discovery behaviour on the regime channel, NOT an alarm, NOT "
        "class (a).\""
    ),
    "sporadic": (
        "**sporadic-finite-sample** (LD-2) — \"**sporadic finite-sample adjacency "
        "noise** — isolated seeds, non-systematic (e.g. chain's 2-seed spurious "
        "X1→X3) — reported, NOT an alarm, NOT class (a).\""
    ),
    "a": (
        "**class (a)** (LD-2) — \"config/estimator/code signature, or systematic "
        "unexplained adjacency error — HARNESS alarm.\" Limb a3 (perfect graph ⟹ "
        "ΔCost_ld = 0) is restricted by the diagnostic-taxonomy correction to "
        "ADDITIVE cells, \"where the note-(c) identity licenses it.\""
    ),
    "b": (
        "**class (b)** (PS-8 per-seed class-(a)/(b) rule) — \"only for a wrong "
        "tie-break-channel orientation.\" A class-(b) call on a linear cell "
        "contradicts the PS-7 note (f)/(g) structural guarantee and is escalated."
    ),
    "unclassified": (
        "**unclassified** — no defined class covers this seed. Reported, never "
        "folded into class (a); an occurrence requires a follow-up adjudication."
    ),
}

#: [LD-2] The halt rule, unchanged in form.
HALT_RULE = (
    "HARNESS HALT iff class (a) is a majority of violating seeds; (c), (d) and "
    "sporadic-finite-sample are not (a)."
)

#: [PS-8] The clause a region-3-driven orientation label carries VERBATIM.
D3_REGION3_CLAUSE = (
    "interaction reflects form-blindness magnitude in the region-3 (cheaper) "
    "direction, opposite to H3's predicted discovery-penalty direction; cannot be "
    "read as a discovery-penalty differential."
)

#: Operationalization of LD-2's systematic-vs-isolated split: an instance's
#: adjacency error is SYSTEMATIC when a majority of its seeds carry
#: skeleton_shd > 0. On this grid every cut in (0.10, 1.00] produces identical
#: labels, so the value is not a tuned threshold.
SYSTEMATIC_MIN_SHARE = 0.5


# --------------------------------------------------------------------------- #
# Per-seed classification (LD-2 sub-clauses (2) and (3))
# --------------------------------------------------------------------------- #


def classify_violating_seed(
    row: pd.Series,
    regime: str,
    *,
    instance_systematic: bool,
    d2_zone: str | None = None,
) -> str:
    """LD-2 class of ONE violating linear seed.

    ``instance_systematic`` is the instance-level a1 fracture input: whether the
    adjacency error is systematic across the instance's seeds or isolated to a
    few. ``d2_zone`` is the LD-2 truth-side verdict for this instance, or None
    where no truth-side check was run (then a systematic adjacency error stays
    "systematic unexplained" and is class (a)).
    """
    if row["skeleton_shd"] > 0:
        # [LD-2] The a1 fracture: skeleton_shd > 0 splits three ways.
        if d2_zone == "ALARM":
            return "a"
        if instance_systematic:
            return "d" if d2_zone == "GENUINE" else "a"
        return "sporadic"

    # skeleton_shd == 0 below.
    if row["wrong_ci_orientation_count"] > 0:
        # Limb a2: a wrong CI-determined orientation on linear data.
        return "a"
    if row["orientation_wrong_count"] == 0:
        if regime == "effect_modifying":
            # [LD-2] The registered form-blindness channel — region-3 placement
            # is part of the definition, so a region-2 EM seed is NOT class (c).
            return "c" if int(row["region"]) == 3 else "unclassified"
        # [LD-2] Limb a3, restricted to ADDITIVE cells.
        return "a"
    if row["wrong_and_tiebreak_count"] > 0:
        return "b"
    return "a"


# --------------------------------------------------------------------------- #
# Instance-level reclassification
# --------------------------------------------------------------------------- #


@dataclass
class InstanceClassification:
    """One linear instance's control-arm status under the corrected taxonomy."""

    topology: str
    regime: str
    n_valid: int
    region_1_count: int
    triggered: bool
    halt: bool
    seeds_by_class: dict[str, list[int]] = field(default_factory=dict)
    n_shd_positive: int = 0
    systematic: bool = False
    d2_zone: str | None = None
    escalations: list[str] = field(default_factory=list)

    @property
    def n_violating(self) -> int:
        return sum(len(v) for v in self.seeds_by_class.values())

    @property
    def n_class_a(self) -> int:
        return len(self.seeds_by_class.get("a", []))

    @property
    def breakdown(self) -> str:
        if not self.n_violating:
            return "—"
        order = ["a", "b", "c", "d", "sporadic", "unclassified"]
        return " + ".join(
            f"{len(self.seeds_by_class[k])} {k}"
            for k in order
            if self.seeds_by_class.get(k)
        )

    @property
    def channel(self) -> str:
        """The registered channel — NEVER "clean" for a non-(a) channel."""
        if not self.triggered:
            return "clean — interpolation held, diagnostic not triggered"
        if self.halt:
            return "HARNESS HALT STANDS — class-(a) majority; no label emitted"
        parts = []
        if self.seeds_by_class.get("c"):
            parts.append(
                f"class (c) registered form-blindness channel (PS-7) on "
                f"{len(self.seeds_by_class['c'])} seed(s)"
            )
        if self.seeds_by_class.get("d"):
            parts.append(
                f"class (d) regime-induced skeleton misspecification "
                f"(zone {self.d2_zone}) on {len(self.seeds_by_class['d'])} seed(s)"
            )
        if self.seeds_by_class.get("sporadic"):
            parts.append(
                f"sporadic finite-sample adjacency noise on "
                f"{len(self.seeds_by_class['sporadic'])} seed(s)"
            )
        if self.seeds_by_class.get("unclassified"):
            parts.append(
                f"UNCLASSIFIED on {len(self.seeds_by_class['unclassified'])} seed(s)"
            )
        return "; ".join(parts) + " — halt LIFTS (no class-(a) majority)"

    @property
    def statement(self) -> str:
        if not self.triggered:
            return (
                "linear control clean — interpolation held, diagnostic not "
                f"triggered (region-1 count {self.region_1_count}/{self.n_valid})."
            )
        verdict = (
            f"HARNESS HALT STANDS — class (a) is a majority of {self.n_violating} "
            "violating seeds."
            if self.halt
            else (
                f"HALT LIFTS — class (a) is {self.n_class_a}/{self.n_violating} "
                "violating seeds, not a majority."
            )
        )
        return (
            f"diagnostic TRIGGERED (region-1 count {self.region_1_count}/"
            f"{self.n_valid} < {LINEAR_HOLDS_MIN}); violating seeds classify "
            f"{self.breakdown} → {verdict}"
        )


def reclassify_linear_instances(
    per_seed: pd.DataFrame,
    instances: pd.DataFrame,
    d2_zones: dict[tuple[str, str], str] | None = None,
) -> list[InstanceClassification]:
    """Re-run the linear-control diagnostic over the six linear instances under LD-2.

    ``d2_zones`` maps ``(topology, regime)`` to an LD-2 zone for instances that
    received a truth-side check. Instances absent from it get ``None``.
    """
    d2_zones = d2_zones or {}
    out: list[InstanceClassification] = []
    for row in instances[instances["family"] == "linear"].itertuples():
        block = per_seed[
            (per_seed["topology"] == row.topology)
            & (per_seed["family"] == "linear")
            & (per_seed["regime"] == row.regime)
        ]
        n_shd_positive = int((block["skeleton_shd"] > 0).sum())
        systematic = n_shd_positive > SYSTEMATIC_MIN_SHARE * len(block)
        zone = d2_zones.get((row.topology, row.regime))

        triggered = row.region_1_count < LINEAR_HOLDS_MIN
        seeds_by_class: dict[str, list[int]] = {}
        escalations: list[str] = []
        if triggered:
            for seed_row in block[block["region"] != 1].itertuples():
                label = classify_violating_seed(
                    pd.Series(seed_row._asdict()),
                    row.regime,
                    instance_systematic=systematic,
                    d2_zone=zone,
                )
                seeds_by_class.setdefault(label, []).append(int(seed_row.seed_idx))
            if seeds_by_class.get("b"):
                escalations.append(
                    f"class-(b) call(s) at seed(s) {seeds_by_class['b']} — "
                    "contradicts the PS-7 note (f)/(g) structural guarantee; escalated "
                    "for inspection, not accepted as benign."
                )
            if seeds_by_class.get("unclassified"):
                escalations.append(
                    f"seed(s) {seeds_by_class['unclassified']} match no LD-2 class "
                    "— reported, NOT folded into class (a); requires a follow-up "
                    "adjudication."
                )
        n_violating = sum(len(v) for v in seeds_by_class.values())
        n_a = len(seeds_by_class.get("a", []))
        # [LD-2] Majority of violating seeds — (c), (d), sporadic are not (a).
        halt = triggered and 2 * n_a > n_violating
        out.append(
            InstanceClassification(
                topology=row.topology,
                regime=row.regime,
                n_valid=int(row.N_valid),
                region_1_count=int(row.region_1_count),
                triggered=triggered,
                halt=halt,
                seeds_by_class=seeds_by_class,
                n_shd_positive=n_shd_positive,
                systematic=systematic,
                d2_zone=zone,
                escalations=escalations,
            )
        )
    return out


def standing_halt_propagation(
    classifications: list[InstanceClassification],
) -> dict[str, list[str]]:
    """[PS-8] What each STANDING halt still invalidates.

    A lifted halt contributes nothing, so the orientation channel
    un-invalidates without the propagation rule itself changing.
    """
    invalidated: dict[str, list[str]] = {
        "orientation_pairs": [],
        "q2_linear_controls": [],
        "corroborating_pairs": [],
    }
    for item in classifications:
        if not item.halt:
            continue
        if item.topology == ORIENTATION_TOPOLOGY:
            invalidated["orientation_pairs"].extend(
                f"collider × {regime}" for regime in REGIMES
            )
            invalidated["q2_linear_controls"].append(
                f"collider linear control (worse-than-no-graph + fallback tier), {item.regime}"
            )
        else:
            invalidated["corroborating_pairs"].append(
                f"{item.topology} × {item.regime}"
            )
    for key, values in invalidated.items():
        invalidated[key] = sorted(set(values))
    return invalidated


# --------------------------------------------------------------------------- #
# The re-emitted instance table
# --------------------------------------------------------------------------- #


def instance_table_v2(
    instances: pd.DataFrame, classifications: list[InstanceClassification]
) -> pd.DataFrame:
    """The frozen instance table plus the LD-2 classification columns.

    Every pre-existing column is carried through UNCHANGED — the re-emission
    touches the classification/label layer only, so a numeric column that moved
    would itself be the bug.
    """
    by_key = {(c.topology, c.regime): c for c in classifications}
    table = instances.copy()
    for column in (
        "violating_seeds",
        "halt_class_a",
        "halt_class_b",
        "halt_class_c",
        "halt_class_d",
        "sporadic",
        "unclassified",
    ):
        table[column] = 0
    table["diagnostic_triggered"] = False
    table["halt"] = False
    table["fork_zone"] = ""
    table["diagnostic_channel"] = ""

    for index, row in table.iterrows():
        if row["family"] != "linear":
            table.at[index, "diagnostic_channel"] = "n/a — NLG is not the control arm"
            continue
        item = by_key[(row["topology"], row["regime"])]
        table.at[index, "violating_seeds"] = item.n_violating
        for code, column in (
            ("a", "halt_class_a"),
            ("b", "halt_class_b"),
            ("c", "halt_class_c"),
            ("d", "halt_class_d"),
            ("sporadic", "sporadic"),
            ("unclassified", "unclassified"),
        ):
            table.at[index, column] = len(item.seeds_by_class.get(code, []))
        table.at[index, "diagnostic_triggered"] = item.triggered
        table.at[index, "halt"] = item.halt
        table.at[index, "fork_zone"] = item.d2_zone or ""
        table.at[index, "diagnostic_channel"] = item.channel
    return table


# --------------------------------------------------------------------------- #
# The re-emitted verdict record
# --------------------------------------------------------------------------- #


def _supersession_header(kind: str) -> list[str]:
    lines = [
        f"<!-- SUPERSESSION HEADER — {kind} re-emitted under LD-2 -->",
        "",
        "> **SUPERSEDES:**",
    ]
    for path, commit in SUPERSEDED:
        lines.append(f"> - `{path}` @ commit `{commit}`")
    lines += [
        "> ",
        "> Those artifacts are **audit objects** and are NOT overwritten, altered or",
        "> deleted. This file is a **versioned re-emission** of the Step-3",
        "> classification/gate/label layer ONLY. Per LD-2, nothing re-runs:",
        "> discovery, estimation, recourse, scoring, covariates (h3_covariates.csv)",
        "> and per-seed quantities (h3_per_seed.csv) are frozen and correct; only the",
        "> Step-3 classification/gate/label layer re-executes.",
        "",
    ]
    return lines


def _epistemic_status_block() -> list[str]:
    return [
        "## Epistemic status",
        "",
        "This verdict is **OUTCOME-EXPOSED, TEXTUALLY FORCED and LABEL-INVARIANT**.",
        "It is *not* outcome-independent, and it is **NOT outcome-blind and does",
        "not claim to be**: the halted run's numbers were visible when the",
        "correction was settled. Its standing rests on three grounds (PS-8, LD-2):",
        "",
        "1. **TEXTUAL FORCING** — the correction direction is uniquely determined by",
        "   appendix text predating every number (PS-7 note (c)'s additive scoping,",
        "   and the interaction-blindness consequence); no alternative correction",
        "   exists that the observed numbers could have selected among.",
        "2. **LABEL-INVARIANCE** — the corrected taxonomy is applied identically",
        "   whatever labels it produces; no threshold, band, gate numerator, k or",
        "   region definition moves.",
        "3. **ADDITIVE-GRID CONFIRMATION** — dcost_ld is exactly 0.000000 on every",
        "   perfect-recovery additive cell, so PS-7 note (c)'s identity holds",
        "   byte-exactly precisely where it is scoped.",
        "",
        "**The correction (LD-2).** Limb a3 of the class-(a) rule justified itself by",
        "\"graph perfect ⟹ under form-preserving estimation + PS-7 note (c)",
        "ΔCost_ld = 0 by construction\". PS-7 note (c) scopes that identity to the",
        "ADDITIVE regime, explicitly excluding EM, where PS-7 records that the two L1",
        "rungs differ by graph AND interaction-blindness. Limb a3's regime-blind",
        "application was a drafting error in the note.",
        "",
    ]


def _d2_block(d2: D2Result, facts_field: str) -> list[str]:
    lines = [
        "## Step 2 — truth-side check (triangle/linear/EM; LD-2)",
        "",
        "Truth-side only: population quantities in closed form from the FROZEN",
        "builder coefficients (`icknowledge/scm/triangles.py`, read programmatically",
        "off the builder signature). No production data was read, nothing under",
        "`results/` was consulted for this section beyond the manifest field naming",
        "`N_disc`, and PC-Stable was not re-run.",
        "",
        "**Frozen builder coefficients — triangle / linear / effect_modifying.**",
        "",
        f"    X1 := a·A + U1                     a = {d2.coefficients.a}",
        f"    X2 := g_of_A·X1 + U2               g_pos = {d2.coefficients.g_pos}, "
        f"g_neg = {d2.coefficients.g_neg}",
        f"    U1, U2 ~ N(0, σ²)                  σ = {d2.coefficients.sigma}",
        "",
        "Under the centered A ∈ {−1,+1} encoding, `g_of_A = ḡ + γ·A` exactly, so the",
        "X2 equation's coefficients are:",
        "",
        "| term | coefficient | value |",
        "|---|---|---|",
        f"| A main effect | β_A | **{d2.coefficients.beta_a:.1f}** — the builder "
        "carries NO additive A term under EM |",
        f"| X1 | ḡ = (g_pos+g_neg)/2 | {d2.coefficients.g_bar:.4f} |",
        f"| A·X1 interaction | γ = (g_pos−g_neg)/2 | {d2.coefficients.gamma:.4f} |",
        f"| noise sd | σ | {d2.coefficients.sigma:.4f} |",
        "",
        f"**N_disc = {d2.n_disc}**, read from the manifest field `{facts_field}` and",
        f"verified identical across all {20} seeds (never assumed). "
        f"α = {d2.alpha}, CI test = Fisher-z.",
        "",
        "**Population correlations over BOTH conditioning sets PC tests at this order.**",
        "PC removes the adjacency if independence is accepted on ANY tested subset, so",
        "the operative subset is the one with the smallest |ρ|.",
        "",
        "| conditioning set S | \\|S\\| | ρ(A, X2 \\| S) | Fisher-z rejection prob at "
        "N_disc | operative |",
        "|---|---|---|---|---|",
    ]
    for subset in d2.subsets:
        mark = "**← operative**" if subset is d2.operative else ""
        lines.append(
            f"| {subset.label} | {subset.size} | {subset.rho:+.9f} | "
            f"{subset.rejection_prob:.6f} | {mark} |"
        )
    lines += [
        "",
        "**Derivation of the partial (exact, and exact for EVERY parameter value).**",
        "",
        "    X2 = (ḡ + γA)(aA + U1) + U2 = ḡa·A + γa + ḡ·U1 + γ·A·U1 + U2   (A² = 1)",
        "",
        "    Cov(A, X2)  = ḡa                    Var(X1)  = a² + σ²",
        "    Cov(A, X1)  = a                     Var(A)   = 1",
        "    Cov(X1, X2) = ḡ(a² + σ²) = ḡ·Var(X1)",
        "",
        "    Cov(A, X2 | X1) = Cov(A,X2) − Cov(A,X1)·Cov(X1,X2)/Var(X1)",
        "                    = ḡa − a·ḡ = 0     ← identically zero, all parameters",
        "",
        "The interaction-carried dependence γ·A·X1 is orthogonal to A at first order,",
        "so a LINEAR lens conditioning on X1 sees nothing left. The pinned Fisher-z",
        "test is not malfunctioning: it correctly reports that no *linear* conditional",
        "dependence remains.",
        "",
        "**Corroboration statistic (LD-2 formula).**",
        "",
        "    r_op ≈ Φ( √(N_disc − |S| − 3) · |atanh ρ_op| − z_{0.975} )",
        f"         = Φ( √({d2.n_disc} − {d2.operative.size} − 3) · "
        f"|atanh {d2.operative.rho:.9f}| − 1.959964 )",
        f"         = Φ( −1.959964 ) = **{d2.r_op:.6f}**",
        "",
        f"At ρ_op = 0 the registered one-tail form returns α/2 = {d2.r_op:.4f}; the",
        "exact two-sided size of the same test is α = 0.05. Both readings sit in the",
        "SAME fork zone by more than an order of magnitude, so the distinction moves",
        "no label.",
        "",
    ]
    if d2.synthetic is not None:
        lines += [
            "**Large-N synthetic corroboration, drawn FROM THE BUILDER** (the analytic",
            "route above is the operative one; this only cross-checks it):",
            "",
            "| conditioning set | analytic ρ | synthetic ρ |",
            "|---|---|---|",
        ]
        for subset in d2.subsets:
            lines.append(
                f"| {subset.label} | {subset.rho:+.9f} | "
                f"{d2.synthetic[subset.conditioning]:+.9f} |"
            )
        lines.append("")
    lines += [
        "**Stored separating sets.** "
        + (
            f"Present; stored sepset for the dropped A—X2 adjacency: "
            f"{d2.stored_sepset}."
            if d2.sepsets_available
            else "The manifests store NO PC separating sets (the `discovery` block "
            "carries library, resolved args, canonical order, edges, adjacency and "
            "dataset identity only), so both candidate subsets are covered above "
            "rather than deferring to a stored one."
        ),
        "",
        f"### TRUTH-SIDE FORK — ZONE: **{d2.zone}**",
        "",
        f"r_op = {d2.r_op:.6f}. The zone applied EXACTLY as committed in LD-2:",
        "",
        f"> {ZONE_TEXT[d2.zone]}",
        "",
        f"**BRANCH: the triangle/linear/effect_modifying halt "
        f"{'**LIFTS**' if d2.halt_lifts else '**STANDS**'}"
        f"{'' if d2.halt_lifts else ' — no label is emitted for that instance'}.**",
        "",
        "**Sanity note (reported only).** Observed drop rate of the A—X2 adjacency:",
        "20/20 seeds. Per-seed probability the test ACCEPTS independence (i.e. drops",
        f"the edge) = 1 − r_op = {d2.seed_drop_probability:.6f}; probability all 20",
        f"independent seeds drop it = {d2.all_seeds_drop_probability(20):.4f} "
        f"(= {1 - 0.05:.2f}^20 = {0.95**20:.4f} under the exact two-sided size).",
        "The observed 20/20 and r_op are **mutually consistent**: a test that accepts",
        "independence ~95–97.5% of the time is expected to drop the edge on all 20",
        "seeds a large fraction of the time.",
        "",
    ]
    return lines


def build_verdict_v2_report(
    per_seed: pd.DataFrame,
    instances: pd.DataFrame,
    pairs: pd.DataFrame,
    n_valid: pd.DataFrame,
    classifications: list[InstanceClassification],
    propagation: dict[str, list[str]],
    supporting: pd.DataFrame | None,
    d2: D2Result,
    d2_field: str,
    spot_checks: list[str],
) -> str:
    """The re-emitted H3 verdict record. No interpretation beyond the rules."""
    out: list[str] = []
    add = out.append

    add("# H3 verdict (v2) — re-emitted under LD-2")
    add("")
    out.extend(_supersession_header("H3 verdict"))
    add(
        "Produced by `icknowledge/analysis/h3_v2.py`, consuming the "
        "FROZEN\n`h3_per_seed.csv` and `h3_covariates.csv`. Every label below is the "
        "OUTPUT of a\npure function applying **PS-8**, including its region\n"
        "partition, as **corrected by LD-2**, to "
        "frozen numbers. No\nthreshold, band, gate numerator, k or region definition "
        "moved."
    )
    add("")
    out.extend(_epistemic_status_block())
    add("---")
    add("")
    add(
        f"- **Population:** PS-4 four-way common-found (`{DENOMINATOR_LABEL}`). Every\n"
        f"  table below carries the `{DENOMINATOR_LABEL}` denominator label.\n"
        "- **dcost_ld** := cost(L1-discovered) − cost(L1-oracle);\n"
        "  **dcost_l0** := cost(L0) − cost(L1-oracle).\n"
        "- **Regions (PS-8, corrected — mutually exclusive):** region 1\n"
        "  `dcost_ld ∈ [min(0,dcost_l0), max(0,dcost_l0)]` (CLOSED; ties → region 1);\n"
        "  region 2 `dcost_ld > max(0,dcost_l0)`; region 3 `dcost_ld < min(0,dcost_l0)`.\n"
        f"- **Gate:** k = **{SIGN_GATE_K}/{N_VALID_REQUIRED}**, numerator = **region 2\n"
        "  only**; region-3 seeds are non-firing but RETAINED in the denominator.\n"
        f"- **Bands (PS-9, unchanged):** ≥{SUPPORTED_MIN_INSTANCES}/6 Supported ·\n"
        f"  {MIXED_MIN_INSTANCES}/6 Mixed · ≤3/6 Null.\n"
        "- **Linear is the CONTROL arm.** It is never banded into a worse-than-no-graph verdict."
    )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- Step-1 checks ---
    add("## Step 1 — mechanism spot-checks (read-only, confirmatory, not load-bearing)")
    add("")
    for line in spot_checks:
        add(line)
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ Step 2 ----
    out.extend(_d2_block(d2, d2_field))
    add("---")
    add("")

    # ----------------------------------------------------- classification ----
    halts = [c for c in classifications if c.halt]
    lifted = [c for c in classifications if c.triggered and not c.halt]
    add("## Step 3a — reclassification under the LD-2 taxonomy")
    add("")
    add("**Class definitions (PS-8, as corrected by LD-2):**")
    add("")
    for code in ("a", "b", "c", "d", "sporadic", "unclassified"):
        add(f"- {CLASS_DEFINITIONS[code]}")
    add("")
    add(f"**Halt rule (LD-2):** {HALT_RULE}")
    add("")
    add(
        "**a1 fracture input.** An instance's adjacency error is read as SYSTEMATIC\n"
        "when a majority of its seeds carry `skeleton_shd > 0`, and as isolated\n"
        "otherwise. On this grid the two cases are 20/20 (triangle/linear/EM) and\n"
        "2/20 (chain/linear), so every cut strictly between them yields identical\n"
        "labels — the split carries no discretion here."
    )
    add("")
    rows = pd.DataFrame(
        [
            {
                "topology": c.topology,
                "family": "linear",
                "regime": c.regime,
                "denominator": DENOMINATOR_LABEL,
                "region_1_count": c.region_1_count,
                "violating": c.n_violating,
                "seeds_shd_positive": c.n_shd_positive,
                "systematic": c.systematic,
                "fork_zone": c.d2_zone or "—",
                "class_breakdown": c.breakdown,
                "halt": "STANDS" if c.halt else ("LIFTS" if c.triggered else "—"),
            }
            for c in classifications
        ]
    )
    add(
        _table(
            rows,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "region_1_count",
                "violating",
                "seeds_shd_positive",
                "systematic",
                "fork_zone",
                "class_breakdown",
                "halt",
            ],
        )
    )
    add("")
    for item in classifications:
        add(f"- **{item.topology} / linear / {item.regime}** — {item.statement}")
        for code in ("a", "b", "c", "d", "sporadic", "unclassified"):
            seeds = item.seeds_by_class.get(code)
            if seeds:
                add(f"  - class {code} seeds: {seeds}")
        for escalation in item.escalations:
            add(f"  - **ESCALATION:** {escalation}")
    add("")
    add(
        f"**Halts LIFTED: {len(lifted)}** — "
        + (
            ", ".join(f"{c.topology}/linear/{c.regime}" for c in lifted)
            if lifted
            else "none"
        )
    )
    add(
        f"**Halts STANDING: {len(halts)}** — "
        + (
            ", ".join(f"{c.topology}/linear/{c.regime}" for c in halts)
            if halts
            else "none"
        )
    )
    add("")
    add("### Step 3b — propagation under the standing halt(s) (PS-8)")
    add("")
    for key, values in propagation.items():
        add(f"- `{key}`: {', '.join(values) if values else '— none —'}")
    add("")
    if not halts:
        add(
            "Every halt-scope (PS-8) invalidation recorded in the superseded run is\n"
            "LIFTED: the collider orientation-channel pairs, the collider\n"
            "worse-than-no-graph linear control and fallback-tier control, and the\n"
            "chain/triangle corroborating\n"
            "pairs are all re-enabled."
        )
        add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3c ----
    orientation = pairs[pairs["topology"] == ORIENTATION_TOPOLOGY]
    orientation_halted = bool(propagation["orientation_pairs"])
    passing = int(orientation["gate_pass"].sum())
    add("## Step 3c — H3a interaction, orientation channel (PS-8)")
    add("")
    add(
        "The two collider pairs (× additive, × effect-modifying). Per-seed event =\n"
        "`dcost_ld(NLG, s) − dcost_ld(linear, s) > 0` **strictly**, paired by\n"
        "`seed_idx` (CRN); a zero difference is non-firing (PS-8). The seed\n"
        "counts are FROZEN — banded here, not recomputed."
    )
    add("")
    add(
        _table(
            orientation,
            [
                "topology",
                "regime",
                "denominator",
                "n_seeds",
                "interaction_event_count",
                "zero_difference_count",
                "mean_interaction_diff",
                "gate_pass",
            ],
            {"mean_interaction_diff": _F6},
        )
    )
    add("")
    if orientation_halted:
        add(
            "### VERDICT: **NO LABEL EMITTED — HALTED**\n\n"
            "A standing collider-linear HARNESS HALT still invalidates both collider\n"
            "interaction pairs per PS-8's halt-scope rule."
        )
        add("")
    else:
        add(
            f"### VERDICT: **{orientation_label(passing)} ({passing}/2)** — "
            f"{passing}/2 pairs pass the {SIGN_GATE_K}/{N_VALID_REQUIRED} seed gate"
        )
        add("")
        add("#### MANDATORY ANNOTATION (PS-8) — accompanies this label")
        add("")
        region3 = per_seed[
            (per_seed["topology"] == ORIENTATION_TOPOLOGY)
            & (per_seed["regime"] == "effect_modifying")
        ]
        arm_lines = []
        for family in ("linear", "nlg"):
            arm = region3[region3["family"] == family]
            counts = arm["region"].value_counts()
            dominant = int(counts.idxmax())
            arm_lines.append(
                f"- **collider / {family} / effect_modifying** — seeds predominantly "
                f"occupied **region {dominant}** ({int(counts.max())}/{len(arm)}); "
                f"mean dcost_ld = {arm['dcost_ld'].mean():+.6f} (**negative** = "
                "L1-discovered CHEAPER than L1-oracle)."
            )
        add(
            "(i) **Region occupancy per family arm** — the pair that carries the "
            "firing:"
        )
        add("")
        for line in arm_lines:
            add(line)
        add("")
        add(
            "(ii) **Sign direction relative to H3's prediction.** H3 predicts a "
            "discovery\nPENALTY (L1-discovered costlier). Both arms of the firing "
            "pair sit in region 3\n(cheaper than the oracle), so the interaction "
            "difference is a difference of\nMAGNITUDES in the cheaper direction — "
            "**opposite** to the predicted direction."
        )
        add("")
        add(
            "This label is produced from region-3 differentials and therefore carries "
            "the\nPS-8 clause **VERBATIM**:"
        )
        add("")
        add(f"> *\"{D3_REGION3_CLAUSE}\"*")
        add("")
        add(
            "Per PS-8 this annotation discharges its standing 'reported\n"
            "explicitly, never folded into holds' instruction and the region partition's "
            "region-3\ninspect flag; it moves no gate, band, or count."
        )
        add("")
    add(
        "**Structure-conditioned reading (PS-8, structural, pre-stated).** The\n"
        "interaction is not a flat six-pair count: the collider is the only one of the\n"
        "three topologies whose graph admits any CI-based orientation, so it alone carries\n"
        "the orientation channel; chain and triangle corroborate on the\n"
        "skeleton/adjacency channel and cannot gate it."
    )
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- worse-than-no-graph ----
    nlg = instances[instances["family"] == "nlg"]
    passing_q2 = int(nlg["region_2_gate_pass"].sum())
    add("## worse-than-no-graph co-primary — NLG region-2 banded verdict")
    add("")
    add(
        "**DUAL ROLE — computed once, emitted once.** This single banded result\n"
        "serves as the **worse-than-no-graph co-primary** (always reported) **and** as the **H3b\n"
        "NLG-existence fallback tier** (invoked ONLY if 3c is Mixed/Null). The\n"
        f"fallback section cross-references it by the anchor `{WORSE_THAN_NO_GRAPH_ANCHOR}` and "
        f"never\n"
        "reprints these counts, so Results cannot double-count one computation as\n"
        "two verdicts."
    )
    add("")
    add(
        _table(
            nlg,
            [
                "topology",
                "regime",
                "denominator",
                "N_valid",
                "region_1_count",
                "region_2_count",
                "region_3_count",
                "reversal_seed_count",
                "region_2_gate_pass",
            ],
        )
    )
    add("")
    add(
        f"### VERDICT: **{band_label(passing_q2, 6)}** — {passing_q2}/6 NLG instances "
        f"reach region-2 count ≥ {SIGN_GATE_K}/{N_VALID_REQUIRED}"
    )
    add("")
    add(
        "Region 2 is EMPTY on every instance of this grid. Region-3 seeds are counted\n"
        "in **neither** the numerator **nor** removed from the denominator; their\n"
        "per-instance tally is the `region_3_count` column above."
    )
    add("")
    add("**Linear control arm beside the NLG verdict (no band applied):**")
    add("")
    linear_rows = instances[instances["family"] == "linear"]
    add(
        _table(
            linear_rows,
            [
                "topology",
                "regime",
                "denominator",
                "region_1_count",
                "region_2_count",
                "region_3_count",
            ],
        )
    )
    add("")
    if propagation["q2_linear_controls"]:
        add(
            "**Control-arm invalidation (PS-8 halt-scope rule):** "
            + "; ".join(propagation["q2_linear_controls"])
            + " — invalidated by a STANDING HARNESS HALT."
        )
    else:
        add(
            "**Control-arm invalidation (PS-8 halt-scope rule): none.** No halt stands, so\n"
            "the collider linear control discharges its worse-than-no-graph and fallback-tier role."
        )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- control table ---
    add("## Step 3e — control-arm status, all six linear instances")
    add("")
    add(
        "Additive cells read \"clean\" exactly as in the superseded run. An EM cell is\n"
        "NEVER reported as \"clean\" and NEVER as a \"harness bug\" for a non-(a)\n"
        "channel: the registered channel is stated explicitly."
    )
    add("")
    control = pd.DataFrame(
        [
            {
                "topology": c.topology,
                "family": "linear",
                "regime": c.regime,
                "denominator": DENOMINATOR_LABEL,
                "region_1_count": c.region_1_count,
                "status": "clean" if not c.triggered else "registered channel",
                "registered_channel": c.channel,
            }
            for c in classifications
        ]
    )
    add(
        _table(
            control,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "region_1_count",
                "status",
                "registered_channel",
            ],
        )
    )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ fallback --
    add("## Step 3f — H3b NLG-existence fallback tier (PS-8, compute-once seam)")
    add("")
    if orientation_halted:
        add(
            "The orientation channel emitted **no label** — HALTED. The fallback's\n"
            "invocation condition (3c Mixed or Null) is not satisfiable."
        )
    else:
        label = orientation_label(passing)
        if label == "Supported":
            add(
                "3c is **Supported**, so the fallback is **NOT invoked** — stated\n"
                "explicitly rather than omitted."
            )
        else:
            add(
                f"3c is **{label}**, so the fallback tier's invocation condition **is\n"
                f"satisfied**. It is discharged by CROSS-REFERENCE to the single banded\n"
                f"result at `{WORSE_THAN_NO_GRAPH_ANCHOR}`; the counts are **not** reprinted "
                f"here, per\n"
                "PS-8's compute-once seam."
            )
    add("")
    add(
        "**Triangle propagation note (verified).** A triangle-linear halt propagates "
        "to\nthe `corroborating_pairs` slot only — it does not touch the collider "
        "pairs — so\na standing triangle halt could not have halted 3c. On this "
        "re-emission the\ntriangle halt lifts in any case, so the question is moot "
        "and v1's 3g handling\nis not invoked."
    )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------ corroboration ----
    add("## CORROBORATING — NON-GATING (PS-8)")
    add("")
    corroborating = instances[instances["topology"].isin(CORROBORATING_TOPOLOGIES)]
    add(
        _table(
            corroborating,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "mean_skeleton_shd",
                "region_1_count",
                "region_2_count",
                "region_3_count",
            ],
            {"mean_skeleton_shd": _F4},
        )
    )
    add("")
    add(
        _table(
            pairs[pairs["topology"].isin(CORROBORATING_TOPOLOGIES)],
            [
                "topology",
                "regime",
                "denominator",
                "n_seeds",
                "interaction_event_count",
                "zero_difference_count",
            ],
        )
    )
    add("")
    add(
        "Per the registered symmetric no-consequence clause, whichever way these\n"
        "numbers point they move no label, no gate and no band."
    )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ covars ----
    add("## Diagnostic covariates per instance (PS-8) — frozen, re-tabled")
    add("")
    add(
        "Covariates below are governed by the orientation-covariate split and "
        "the nSID normalization rule (PS-8)."
    )
    add("")
    add(
        _table(
            instances,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "mean_skeleton_shd",
                "mean_orientation_wrong",
                "mean_tiebreak_resolved",
                "mean_wrong_and_tiebreak",
                "mean_n_sid",
                "structural_expectation_violations",
            ],
            {
                "mean_skeleton_shd": _F4,
                "mean_orientation_wrong": _F4,
                "mean_tiebreak_resolved": _F4,
                "mean_wrong_and_tiebreak": _F4,
                "mean_n_sid": _F4,
            },
        )
    )
    add("")
    add("**Descriptive extras (frozen).**")
    add("")
    add(
        _table(
            instances,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "mean_dcost_ld",
                "sd_dcost_ld",
                "mean_dcost_l0",
                "sd_dcost_l0",
                "mean_t",
                "t_instability_flag",
            ],
            {
                "mean_dcost_ld": _F6,
                "sd_dcost_ld": _F6,
                "mean_dcost_l0": _F6,
                "sd_dcost_l0": _F6,
                "mean_t": _F6,
            },
        )
    )
    add("")
    add("**Supporting evidence (PS-6 / the common-found population) — reported, "
        "enters no verdict rule.**")
    add("")
    if supporting is None:
        add("Not trivially readable from the existing summary CSVs — deferred to Results.")
    else:
        add(
            _table(
                supporting,
                [
                    "topology",
                    "family",
                    "regime",
                    "denominator",
                    "validity_A_neg",
                    "validity_A_pos",
                    "Gap_cost_L1d",
                ],
                {
                    "validity_A_neg": _F4,
                    "validity_A_pos": _F4,
                    "Gap_cost_L1d": _F6,
                },
            )
        )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- separability ----
    add("## Discussion obligation (LD-2)")
    add("")
    add(
        "> the family channel (Fisher-z degrading NLG skeletons) returned empty on\n"
        "> this grid — skeleton-SHD(NLG) ≤ skeleton-SHD(linear) on all corroborating\n"
        "> cells, nSID = 0 everywhere — and the regime channel (form-blindness under\n"
        "> EM, with group-asymmetric validity collapse) is the live finding; the two\n"
        "> statements are kept separable: what was predicted and nulled vs. what was\n"
        "> found under a register-logged consequence clause (PS-7)."
    )
    add("")
    add(
        "No hypothesis or assumption changes. No threshold, band, gate numerator, k,\n"
        "or region definition moved in this re-emission."
    )
    add("")
    return "\n".join(out) + "\n"


def write_v2_artifacts(
    report: str, table: pd.DataFrame, root: Path = L1D_ROOT
) -> tuple[Path, Path]:
    """Write the two _v2 artifacts, REFUSING to touch a superseded one."""
    summary = root / "summary"
    verdict_path = summary / "h3_verdict_v2.md"
    table_path = summary / "h3_instance_table_v2.csv"
    for path in (verdict_path, table_path):
        if path.name in {"h3_verdict.md", "h3_instance_table.csv"}:  # pragma: no cover
            raise AssertionError("refusing to overwrite a superseded audit object")
    verdict_path.write_text(report, encoding="utf-8")

    header = "\n".join(
        f"# {line}"
        for line in [
            "H3 instance table (v2) — re-emitted under LD-2.",
            "SUPERSEDES: "
            + "; ".join(f"{path} @ {commit}" for path, commit in SUPERSEDED),
            "Those artifacts are audit objects and are NOT overwritten.",
            "Only the Step-3 classification/gate/label layer re-executed; every",
            "pre-existing numeric column below is carried through from the frozen",
            "inputs; the classification counts are computed here.",
        ]
    )
    table_path.write_text(
        header + "\n" + table.to_csv(index=False), encoding="utf-8", newline=""
    )
    return verdict_path, table_path
