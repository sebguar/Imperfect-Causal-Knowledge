"""H3 — the price of discovery: per-seed regions, gates, diagnostic, verdict.

Reads PS-8 on the PS-4 four-way common-found population.

    dcost_ld := realized cost at L1-discovered − realized cost at L1-oracle
    dcost_l0 := realized cost at L0            − realized cost at L1-oracle

Both levels are read from the stored per-group aggregate
``realized_cost_common_found_fourway`` in ``summary/per_seed_by_group.csv``,
combined across groups by their stored ``N_common_found_fourway`` counts. The
cell-level ``Δ_cost_common_found_fourway`` column is the BETWEEN-GROUP gap
(A=−1 minus A=+1) and is NOT used.

    region 1  holds        dcost_ld ∈ [min(0, dcost_l0), max(0, dcost_l0)]
    region 2  violation    dcost_ld > max(0, dcost_l0)
    region 3  better       dcost_ld < min(0, dcost_l0)

The regions are mutually exclusive and the interval is CLOSED: ties are region
1. Under reversal (dcost_l0 < 0) a point between the two baselines is a partial
degradation and reads "holds" — it is not the worse-than-no-graph event. The
gate numerator is region 2 only; region-3 seeds are non-firing but RETAINED in
the denominator. Channels: H3a orientation on the two collider pairs; the
worse-than-no-graph co-primary on the six NLG instances, region-2 count under
the same k=16/20 gate and the PS-9 bands, computed once and doubling as the H3b
fallback tier; chain and triangle corroborate with no threshold. Linear is the
control arm and is never banded. A class-(a) majority halt or an
``N_valid < 20`` flag suppresses the affected label(s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from icknowledge.analysis.loading import (
    L1D_ROOT,
    N_SEEDS_FULL,
)

#: [PS-4] Per-group realized-cost LEVEL on the four-way common-found
#: population, and its per-group denominator. Named once so the choice is
#: auditable and cannot drift to the between-group gap column.
COST_COLUMN = "realized_cost_common_found_fourway"
COUNT_COLUMN = "N_common_found_fourway"

#: The PS-4 denominator label carried on every table in the verdict record.
DENOMINATOR_LABEL = "fourway"

#: [PS-9, carried unchanged by PS-8] Per-seed sign-agreement gate.
SIGN_GATE_K = 16

#: [PS-8] The gate denominator. An instance with fewer
#: valid seeds HALTS TO A FLAG — the integer threshold is never re-based.
N_VALID_REQUIRED = 20

#: [PS-9, carried unchanged by PS-8] Per-instance bands.
SUPPORTED_MIN_INSTANCES = 5
MIXED_MIN_INSTANCES = 4

#: [PS-8] The linear control's clean-interpolation
#: expectation: the diagnostic activates below this many region-1 seeds.
LINEAR_HOLDS_MIN = 16

#: [PS-8] The orientation channel's topology and its two pairs.
ORIENTATION_TOPOLOGY = "collider"
REGIMES = ("additive", "effect_modifying")

#: [PS-8] The corroborating (non-gating) topologies.
CORROBORATING_TOPOLOGIES = ("chain", "triangle")


# --------------------------------------------------------------------------- #
# Region partition (PS-8, corrected)
# --------------------------------------------------------------------------- #


def classify_region(dcost_ld: float, dcost_l0: float) -> int:
    """The mutually-exclusive interval region of one seed.

    ``dcost_l0`` supplies the far end of the ordered interval; 0 (the L1-oracle
    reference) supplies the near end. Membership is decided on the CLOSED
    interval, so both boundary ties fall in region 1.
    """
    lower = min(0.0, dcost_l0)
    upper = max(0.0, dcost_l0)
    if dcost_ld > upper:
        return 2
    if dcost_ld < lower:
        return 3
    return 1


def _ratio(dcost_ld: float, dcost_l0: float) -> float:
    """Normalized interpolation coordinate t (descriptive only; NaN-safe)."""
    if dcost_l0 == 0.0:
        return float("nan")
    return dcost_ld / dcost_l0


# --------------------------------------------------------------------------- #
# Step 2 — per-seed quantities
# --------------------------------------------------------------------------- #


def load_realized_costs(
    root: Path = L1D_ROOT, n_seeds: int = N_SEEDS_FULL
) -> pd.DataFrame:
    """Realized cost LEVEL per (topology, family, regime, seed_idx, condition).

    Combines the two stored per-group aggregate means by their stored per-group
    common-found counts. Nothing individual-level is read or recomputed.
    """
    path = root / "summary" / "per_seed_by_group.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — H3 reads the frozen four-rung aggregates and "
            "does not regenerate them."
        )
    frame = pd.read_csv(path)
    for column in (COST_COLUMN, COUNT_COLUMN):
        if column not in frame.columns:
            raise KeyError(f"{path} carries no `{column}` column")

    frame = frame.copy()
    frame["_weighted"] = frame[COST_COLUMN] * frame[COUNT_COLUMN]
    keys = ["topology", "family", "regime", "seed", "seed_idx", "condition"]
    grouped = frame.groupby(keys, as_index=False).agg(
        weighted=("_weighted", "sum"),
        n_common_found=(COUNT_COLUMN, "sum"),
        n_groups=("group", "nunique"),
    )
    if not (grouped["n_groups"] == 2).all():
        raise AssertionError("a (cell, seed, condition) is missing a protected group")
    grouped["cost"] = grouped["weighted"] / grouped["n_common_found"]
    grouped = grouped.drop(columns=["weighted", "n_groups"])
    if grouped["seed_idx"].max() >= n_seeds:
        grouped = grouped[grouped["seed_idx"] < n_seeds]
    return grouped


def n_valid_table(costs: pd.DataFrame) -> pd.DataFrame:
    """[PS-8] Per-instance count of seeds with a usable four-way population.

    A seed is valid iff all four conditions are present AND its four-way
    common-found population is non-empty in every one of them.
    """
    rows: list[dict[str, object]] = []
    required = {"L0", "L1-oracle", "L1-discovered", "L2"}
    for (topology, family, regime), block in costs.groupby(
        ["topology", "family", "regime"], sort=False
    ):
        n_valid = 0
        for _seed_idx, seed_block in block.groupby("seed_idx"):
            conditions = set(seed_block["condition"])
            non_empty = (seed_block["n_common_found"] > 0).all()
            if required.issubset(conditions) and non_empty:
                n_valid += 1
        rows.append(
            {
                "topology": topology,
                "family": family,
                "regime": regime,
                "denominator": DENOMINATOR_LABEL,
                "N_valid": n_valid,
                "N_required": N_VALID_REQUIRED,
                "halt_n_valid": n_valid < N_VALID_REQUIRED,
            }
        )
    return pd.DataFrame(rows).sort_values(["topology", "family", "regime"]).reset_index(
        drop=True
    )


def per_seed_quantities(
    costs: pd.DataFrame, covariates: pd.DataFrame | None = None
) -> pd.DataFrame:
    """dcost_ld, dcost_l0, region, t and the seed-paired interaction event."""
    wide = costs.pivot_table(
        index=["topology", "family", "regime", "seed", "seed_idx"],
        columns="condition",
        values="cost",
    ).reset_index()
    wide.columns.name = None
    for condition in ("L0", "L1-oracle", "L1-discovered"):
        if condition not in wide.columns:
            raise KeyError(f"the four-rung aggregate carries no `{condition}` rung")

    wide["cost_l1_oracle"] = wide["L1-oracle"]
    wide["cost_l1_discovered"] = wide["L1-discovered"]
    wide["cost_l0"] = wide["L0"]
    wide["dcost_ld"] = wide["L1-discovered"] - wide["L1-oracle"]
    wide["dcost_l0"] = wide["L0"] - wide["L1-oracle"]
    wide["region"] = [
        classify_region(ld, l0)
        for ld, l0 in zip(wide["dcost_ld"], wide["dcost_l0"], strict=True)
    ]
    wide["t"] = [
        _ratio(ld, l0)
        for ld, l0 in zip(wide["dcost_ld"], wide["dcost_l0"], strict=True)
    ]
    wide["reversal_seed"] = wide["dcost_l0"] < 0

    # [PS-8] Cross-family interaction event, paired by seed_idx (CRN).
    # Recorded on BOTH family rows of a pair so the artifact is self-describing.
    pivot = wide.pivot_table(
        index=["topology", "regime", "seed_idx"], columns="family", values="dcost_ld"
    )
    if {"nlg", "linear"}.issubset(pivot.columns):
        diff = (pivot["nlg"] - pivot["linear"]).rename("interaction_diff")
        wide = wide.merge(
            diff.reset_index(), on=["topology", "regime", "seed_idx"], how="left"
        )
    else:
        # A single-family frame carries no cross-family pair; NaN rather than a
        # fabricated zero, so a partial frame can never fire an interaction event.
        wide["interaction_diff"] = np.nan
    wide["interaction_event"] = wide["interaction_diff"] > 0

    wide = wide.drop(columns=["L0", "L1-oracle", "L1-discovered", "L2"], errors="ignore")

    if covariates is not None:
        wide = wide.merge(
            covariates,
            on=["topology", "family", "regime", "seed_idx"],
            how="left",
            suffixes=("", "_cov"),
        )
        # The covariate frame carries its own master seed, read from the
        # manifest. Assert the two provenance chains agree before dropping the
        # duplicate — a mismatch would mean the aggregate and the manifest
        # disagree about which run a row describes.
        if "seed_cov" in wide.columns:
            if not (wide["seed"] == wide["seed_cov"]).all():
                raise AssertionError(
                    "master seed disagrees between the aggregate and the manifest"
                )
            wide = wide.drop(columns=["seed_cov"])
    return wide.sort_values(
        ["topology", "family", "regime", "seed_idx"]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 3 — instance-level counts, gates, diagnostic
# --------------------------------------------------------------------------- #


def instance_table(per_seed: pd.DataFrame, n_valid: pd.DataFrame) -> pd.DataFrame:
    """Per-instance region tallies, gate counts and descriptive extras."""
    rows: list[dict[str, object]] = []
    for (topology, family, regime), block in per_seed.groupby(
        ["topology", "family", "regime"], sort=False
    ):
        regions = block["region"]
        mean_l0 = float(block["dcost_l0"].mean())
        sd_l0 = float(block["dcost_l0"].std(ddof=1))
        rows.append(
            {
                "topology": topology,
                "family": family,
                "regime": regime,
                "denominator": DENOMINATOR_LABEL,
                "n_seeds": int(len(block)),
                "region_1_count": int((regions == 1).sum()),
                "region_2_count": int((regions == 2).sum()),
                "region_3_count": int((regions == 3).sum()),
                "reversal_seed_count": int(block["reversal_seed"].sum()),
                "mean_dcost_ld": float(block["dcost_ld"].mean()),
                "sd_dcost_ld": float(block["dcost_ld"].std(ddof=1)),
                "mean_dcost_l0": mean_l0,
                "sd_dcost_l0": sd_l0,
                "mean_t": float(np.nanmean(block["t"])),
                "t_instability_flag": bool(abs(mean_l0) <= sd_l0),
                "mean_skeleton_shd": float(block["skeleton_shd"].mean()),
                "mean_orientation_wrong": float(block["orientation_wrong_count"].mean()),
                "mean_tiebreak_resolved": float(
                    block["tiebreak_resolved_count"].mean()
                ),
                "mean_wrong_and_tiebreak": float(
                    block["wrong_and_tiebreak_count"].mean()
                ),
                "mean_n_sid": float(block["n_sid"].mean()),
                "structural_expectation_violations": int(
                    block["structural_expectation_violated"].sum()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table = table.merge(
        n_valid[["topology", "family", "regime", "N_valid", "halt_n_valid"]],
        on=["topology", "family", "regime"],
        how="left",
    )
    # The worse-than-no-graph / fallback gate counts region 2 only, against the fixed denominator.
    table["region_2_gate_pass"] = (table["region_2_count"] >= SIGN_GATE_K) & (
        ~table["halt_n_valid"]
    )
    return table.sort_values(["topology", "family", "regime"]).reset_index(drop=True)


def interaction_pairs(per_seed: pd.DataFrame, n_valid: pd.DataFrame) -> pd.DataFrame:
    """[PS-8] Per (topology, regime) pair: seed count of the interaction event."""
    halted = {
        (r.topology, r.family, r.regime)
        for r in n_valid.itertuples()
        if r.halt_n_valid
    }
    rows: list[dict[str, object]] = []
    for (topology, regime), block in per_seed.groupby(
        ["topology", "regime"], sort=False
    ):
        nlg = block[block["family"] == "nlg"]
        events = int(nlg["interaction_event"].sum())
        ties = int((nlg["interaction_diff"] == 0).sum())
        rows.append(
            {
                "topology": topology,
                "regime": regime,
                "denominator": DENOMINATOR_LABEL,
                "n_seeds": int(len(nlg)),
                "interaction_event_count": events,
                "zero_difference_count": ties,
                "mean_interaction_diff": float(nlg["interaction_diff"].mean()),
                "sd_interaction_diff": float(nlg["interaction_diff"].std(ddof=1)),
                "n_valid_halt": (topology, "nlg", regime) in halted
                or (topology, "linear", regime) in halted,
                "gate_pass": events >= SIGN_GATE_K
                and (topology, "nlg", regime) not in halted
                and (topology, "linear", regime) not in halted,
            }
        )
    return pd.DataFrame(rows).sort_values(["topology", "regime"]).reset_index(drop=True)


def _classify_violating_seed(row: pd.Series) -> str:
    """[PS-8] class (a) / class (b) for one violating LINEAR seed."""
    if row["skeleton_shd"] > 0:
        return "a"
    if row["wrong_ci_orientation_count"] > 0:
        return "a"
    if row["orientation_wrong_count"] == 0:
        # Graph perfect and orientation correct: under form-preserving estimation + PS-7 note (c)
        # dcost_ld is 0 by construction, so a violation is config / CI / estimator
        # / numerical — the purest harness alarm, one the drift guard owed us.
        return "a"
    if row["wrong_and_tiebreak_count"] > 0:
        return "b"
    return "a"


@dataclass
class LinearDiagnostic:
    """[PS-8] One linear instance's control-arm status."""

    topology: str
    family: str
    regime: str
    n_valid: int
    region_1_count: int
    triggered: bool
    halt: bool
    class_a_seeds: list[int] = field(default_factory=list)
    class_b_seeds: list[int] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.topology, self.regime)

    @property
    def statement(self) -> str:
        if self.halt:
            return (
                f"HARNESS HALT — class-(a) majority among {len(self.class_a_seeds)}"
                f"+{len(self.class_b_seeds)} violating seeds; no verdict emitted "
                "for the affected component(s)."
            )
        if self.triggered:
            return (
                f"diagnostic TRIGGERED (region-1 count {self.region_1_count}/"
                f"{self.n_valid} < {LINEAR_HOLDS_MIN}) — violating seeds classify "
                f"{len(self.class_a_seeds)} class-(a) / {len(self.class_b_seeds)} "
                "class-(b); no class-(a) majority, so no halt."
            )
        return (
            "linear control clean — interpolation held, diagnostic not triggered "
            f"(region-1 count {self.region_1_count}/{self.n_valid})."
        )


def linear_diagnostics(
    per_seed: pd.DataFrame, instances: pd.DataFrame
) -> list[LinearDiagnostic]:
    """[PS-8] Run the control-arm diagnostic over the six linear instances."""
    out: list[LinearDiagnostic] = []
    linear = instances[instances["family"] == "linear"]
    for row in linear.itertuples():
        block = per_seed[
            (per_seed["topology"] == row.topology)
            & (per_seed["family"] == "linear")
            & (per_seed["regime"] == row.regime)
        ]
        triggered = row.region_1_count < LINEAR_HOLDS_MIN
        class_a: list[int] = []
        class_b: list[int] = []
        escalations: list[str] = []
        if triggered:
            for seed_row in block[block["region"] != 1].itertuples():
                seed_series = pd.Series(seed_row._asdict())
                if _classify_violating_seed(seed_series) == "a":
                    class_a.append(int(seed_row.seed_idx))
                else:
                    class_b.append(int(seed_row.seed_idx))
            # [PS-7 note (f)/(g)] A class-(b) call on a linear cell
            # contradicts a pre-stated structural guarantee: it is escalated for
            # inspection, never accepted as benign identifiability-limitation.
            if class_b:
                escalations.append(
                    f"class-(b) call(s) on a linear cell at seed(s) "
                    f"{class_b} — contradicts the PS-7 note (f)/(g) structural "
                    "guarantee; escalated for inspection, not accepted as benign."
                )
        halt = triggered and len(class_a) > len(class_b)
        out.append(
            LinearDiagnostic(
                topology=row.topology,
                family="linear",
                regime=row.regime,
                n_valid=int(row.N_valid),
                region_1_count=int(row.region_1_count),
                triggered=triggered,
                halt=halt,
                class_a_seeds=class_a,
                class_b_seeds=class_b,
                escalations=escalations,
            )
        )
    return out


def halt_propagation(diagnostics: list[LinearDiagnostic]) -> dict[str, list[str]]:
    """[PS-8] What each linear HALT invalidates."""
    invalidated: dict[str, list[str]] = {
        "orientation_pairs": [],
        "q2_linear_controls": [],
        "corroborating_pairs": [],
    }
    for diagnostic in diagnostics:
        if not diagnostic.halt:
            continue
        if diagnostic.topology == ORIENTATION_TOPOLOGY:
            invalidated["orientation_pairs"].extend(
                f"collider × {regime}" for regime in REGIMES
            )
            invalidated["q2_linear_controls"].append(
                f"collider linear control (worse-than-no-graph + fallback tier), "
                f"{diagnostic.regime}"
            )
        else:
            invalidated["corroborating_pairs"].append(
                f"{diagnostic.topology} × {diagnostic.regime}"
            )
    for key, values in invalidated.items():
        invalidated[key] = sorted(set(values))
    return invalidated


# --------------------------------------------------------------------------- #
# Verdict labels
# --------------------------------------------------------------------------- #


def band_label(passing: int, total: int) -> str:
    """[PS-9, carried by PS-8] ≥5/6 Supported · 4/6 Mixed · ≤3/6 Null."""
    del total
    if passing >= SUPPORTED_MIN_INSTANCES:
        return "Supported"
    if passing == MIXED_MIN_INSTANCES:
        return "Mixed"
    return "Null"


def orientation_label(passing: int) -> str:
    """[PS-8] Two collider pairs: 2/2 Supported · 1/2 Mixed · 0/2 Null."""
    return {2: "Supported", 1: "Mixed", 0: "Null"}[passing]


# --------------------------------------------------------------------------- #
# Supporting evidence (PS-6 / the common-found population), read as-is or deferred
# --------------------------------------------------------------------------- #


def supporting_evidence(root: Path = L1D_ROOT) -> pd.DataFrame | None:
    """Validity-by-group and Gap_c at L1-discovered, IF trivially readable.

    Read verbatim off the existing cross-seed summaries; nothing is derived. If
    either table is absent the caller reports the item as deferred to Results
    rather than deriving it (PS-8 SUPPORTING EVIDENCE is not a H3 verdict input).
    """
    by_group = root / "summary" / "cross_seed_by_group.csv"
    by_cell = root / "summary" / "cross_seed_by_cell.csv"
    if not (by_group.exists() and by_cell.exists()):
        return None
    groups = pd.read_csv(by_group)
    cells = pd.read_csv(by_cell)
    needed_g = {"realized_validity_rate_mean", "group", "condition"}
    needed_c = {"Gap_cost_mean", "condition"}
    if not needed_g.issubset(groups.columns) or not needed_c.issubset(cells.columns):
        return None

    groups = groups[groups["condition"] == "L1-discovered"]
    validity = groups.pivot_table(
        index=["topology", "family", "regime"],
        columns="group",
        values="realized_validity_rate_mean",
    ).rename(columns={-1: "validity_A_neg", 1: "validity_A_pos"})
    gap = (
        cells[cells["condition"] == "L1-discovered"]
        .set_index(["topology", "family", "regime"])["Gap_cost_mean"]
        .rename("Gap_cost_L1d")
    )
    out = validity.join(gap).reset_index()
    out["denominator"] = "all-eligible (PS-6/common-found reporting population)"
    return out


# --------------------------------------------------------------------------- #
# Step 4 — the verdict record
# --------------------------------------------------------------------------- #

#: Anchor of the SINGLE banded NLG region-2 result. The fallback tier
#: cross-references this anchor and never reprints the counts (PS-8's
#: compute-once seam), so Results cannot double-count one computation.
WORSE_THAN_NO_GRAPH_ANCHOR = "#worse-than-no-graph-co-primary--nlg-region-2-banded-verdict"


def _table(frame: pd.DataFrame, columns: list[str], formats: dict | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "---|" * len(columns)
    lines = [header, rule]
    for row in frame.itertuples():
        values = []
        for column in columns:
            value = getattr(row, column)
            fmt = formats.get(column)
            values.append(fmt(value) if fmt else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


_F6 = lambda v: f"{v:+.6f}"  # noqa: E731
_F4 = lambda v: f"{v:.4f}"  # noqa: E731


def build_verdict_report(
    per_seed: pd.DataFrame,
    instances: pd.DataFrame,
    pairs: pd.DataFrame,
    n_valid: pd.DataFrame,
    diagnostics: list[LinearDiagnostic],
    propagation: dict[str, list[str]],
    supporting: pd.DataFrame | None,
) -> str:
    """The mechanical H3 verdict record. No interpretation beyond the rules."""
    out: list[str] = []
    add = out.append
    escalations = [
        (d, text) for d in diagnostics for text in d.escalations
    ]

    add("# H3 verdict — mechanical PS-8 reading at N=20 (four-rung tree)")
    add("")
    add(
        "Produced by `icknowledge/analysis/h3.py` + `h3_covariates.py`. Every label\n"
        "below is the OUTPUT of a pure function applying **PS-8** with the\n"
        "**region-partition correction**, to frozen numbers. Nothing is narrated; no\n"
        "threshold was chosen after the numbers existed."
    )
    add("")
    add(
        f"- **Population:** PS-4 four-way common-found (`{DENOMINATOR_LABEL}`). Every\n"
        f"  table below carries the `{DENOMINATOR_LABEL}` denominator label.\n"
        f"- **Realized cost source:** `{COST_COLUMN}` in `per_seed_by_group.csv`,\n"
        f"  combined across groups by the stored `{COUNT_COLUMN}` counts. The\n"
        "  cell-level `Δ_cost_common_found_fourway` column is the between-group GAP\n"
        "  and is deliberately not used.\n"
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

    # -------------------------------------------------------- at a glance ---
    halts_glance = [d for d in diagnostics if d.halt]
    n_valid_halts = n_valid[n_valid["halt_n_valid"]]
    add("## HALTS · FLAGS · ESCALATIONS — at a glance")
    add("")
    add(
        f"- **N_valid halts (PS-8):** {len(n_valid_halts)}"
        + (
            ""
            if n_valid_halts.empty
            else " — "
            + ", ".join(
                f"{r.topology}/{r.family}/{r.regime} (N_valid={r.N_valid})"
                for r in n_valid_halts.itertuples()
            )
        )
    )
    add(
        f"- **HARNESS HALTs (PS-8 halt-scope rule):** {len(halts_glance)}"
        + (
            ""
            if not halts_glance
            else " — "
            + ", ".join(f"{d.topology}/linear/{d.regime}" for d in halts_glance)
        )
    )
    add(
        f"- **PS-7 note (f)/(g) class-(b) escalations:** {len(escalations)}"
        + ("" if escalations else " — none: no class-(b) call was made on any linear cell.")
    )
    add(
        "- **Structural-expectation violations (PS-8 orientation-covariate split):** "
        f"{int(instances['structural_expectation_violations'].sum())}"
    )
    add(
        "- **Reversal seeds (dcost_l0 < 0):** "
        f"{int(instances['reversal_seed_count'].sum())} of "
        f"{int(instances['n_seeds'].sum())}"
    )
    add(
        "- **t instability flags:** "
        f"{int(instances['t_instability_flag'].sum())}"
    )
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3a ----
    add("## 3a — N_valid gate (PS-8)")
    add("")
    add(
        _table(
            n_valid,
            [
                "topology",
                "family",
                "regime",
                "denominator",
                "N_valid",
                "N_required",
                "halt_n_valid",
            ],
        )
    )
    add("")
    halted_n = n_valid[n_valid["halt_n_valid"]]
    if halted_n.empty:
        add(
            f"**No N_valid halt.** Every instance carries N_valid = "
            f"{N_VALID_REQUIRED}/{N_VALID_REQUIRED} on the `{DENOMINATOR_LABEL}` "
            "population, so the common-found no-op holds and the integer gate is applied "
            "against its calibrated denominator without re-basing."
        )
    else:
        add(
            "**N_valid HALT-TO-FLAG.** The following instances have N_valid < "
            f"{N_VALID_REQUIRED} and receive **no verdict label**; the exclusion is "
            "adjudicated in a follow-up before any label is written:"
        )
        for row in halted_n.itertuples():
            add(f"- {row.topology} / {row.family} / {row.regime}: N_valid = {row.N_valid}")
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------- CONTROL ----
    add("## CONTROL ARM (linear) — PS-8")
    add("")
    add(
        "All six linear instances, banded or not. A clean control is a POSITIVE\n"
        "reportable outcome (PS-8: \"the linear control is what separates 'discovery\n"
        "genuinely hurts NLG' from 'seed noise'\"), so it is stated explicitly rather\n"
        "than left as silence."
    )
    add("")
    linear_rows = instances[instances["family"] == "linear"]
    add(
        _table(
            linear_rows,
            [
                "topology",
                "regime",
                "denominator",
                "N_valid",
                "region_1_count",
                "region_2_count",
                "region_3_count",
                "reversal_seed_count",
            ],
        )
    )
    add("")
    for diagnostic in diagnostics:
        add(f"- **{diagnostic.topology} / linear / {diagnostic.regime}** — {diagnostic.statement}")
        if diagnostic.triggered:
            add(
                f"  - class (a) seeds: {diagnostic.class_a_seeds or '—'}\n"
                f"  - class (b) seeds: {diagnostic.class_b_seeds or '—'}"
            )
        for escalation in diagnostic.escalations:
            add(f"  - **ESCALATION:** {escalation}")
    add("")
    if not escalations:
        add(
            "**PS-7 note (f)/(g) escalations: none.** No violating linear seed classified\n"
            "class (b), so the pre-stated structural guarantee (a correctly-recovered\n"
            "skeleton's canonical-order tie-break reproduces the true X–X\n"
            "orientations) was not contradicted on any linear cell."
        )
        add("")

    halts = [d for d in diagnostics if d.halt]
    if halts:
        add("### HARNESS HALT record (PS-8)")
        add("")
        add(
            "The H3 verdict pipeline **stops for the affected component(s)** rather\n"
            "than emitting a label. The fix is a register-logged correction in a\n"
            "follow-up adjudication before any re-run — never a silent retune."
        )
        add("")
        for diagnostic in halts:
            add(
                f"- **HALT** — {diagnostic.topology} / linear / {diagnostic.regime}: "
                f"region-1 count {diagnostic.region_1_count}/{diagnostic.n_valid} < "
                f"{LINEAR_HOLDS_MIN}; violating seeds classify "
                f"{len(diagnostic.class_a_seeds)} class-(a) / "
                f"{len(diagnostic.class_b_seeds)} class-(b) → class-(a) majority."
            )
        add("")
        add("**Propagation — components invalidated by the halt(s):**")
        add("")
        for key, values in propagation.items():
            add(f"- `{key}`: {', '.join(values) if values else '— none —'}")
        add("")
    else:
        add("**No HARNESS HALT fired.**")
        add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3c ----
    orientation = pairs[pairs["topology"] == ORIENTATION_TOPOLOGY]
    orientation_halted = bool(propagation["orientation_pairs"])
    add("## 3c — H3a interaction, orientation channel (PS-8, confirmatory-primary)")
    add("")
    add(
        "The two collider pairs (× additive, × effect-modifying). Per-seed event =\n"
        "`dcost_ld(NLG, s) − dcost_ld(linear, s) > 0` **strictly**, paired by\n"
        "`seed_idx` (CRN); a zero difference is non-firing (PS-8)."
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
            "The collider-linear HARNESS HALT invalidates both collider interaction\n"
            "pairs — the confirmatory-primary orientation channel — per PS-8's\n"
            "halt-scope rule. The seed counts above are recorded for the register;\n"
            "**they are not read as a verdict** and no band is applied to them."
        )
    else:
        passing = int(orientation["gate_pass"].sum())
        add(
            f"### VERDICT: **{orientation_label(passing)}** — {passing}/2 pairs pass "
            f"the {SIGN_GATE_K}/{N_VALID_REQUIRED} seed gate"
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
    add(
        "**Pre-stated predicted pattern (PS-8):** *collider firing and chain/triangle\n"
        "attenuated is the EXPECTED result and is read as Supported on the\n"
        "orientation channel — it may not be reported as Null.*"
    )
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3d ----
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
        "Region-3 seeds are counted in **neither** the numerator **nor** removed from\n"
        "the denominator; their per-instance tally is the `region_3_count` column\n"
        "above. `reversal_seed_count` is the per-instance incidence of `dcost_l0 < 0`\n"
        "— the case where the corrected partition could change a region assignment."
    )
    add("")
    add("**Linear control arm beside the NLG verdict (no band applied):**")
    add("")
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
            + " — invalidated by the HARNESS HALT above. The counts are shown for the\n"
            "register; they do not discharge the control-arm role until a follow-up\n"
            "adjudication."
        )
        add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3e ----
    add("## 3e — CORROBORATING — NON-GATING (PS-8)")
    add("")
    add(
        "Chain and triangle, purely descriptive. **No threshold is applied.** Per the\n"
        "registered symmetric no-consequence clause, whichever way these numbers\n"
        "point they move no label, no gate and no band."
    )
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
    add("**The four corroborating interaction pairs' seed counts:**")
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
    shd = (
        corroborating.pivot_table(
            index=["topology", "regime"], columns="family", values="mean_skeleton_shd"
        )
        .reset_index()
    )
    agrees = [
        f"{row.topology} × {row.regime}: NLG {row.nlg:.4f} "
        f"{'>' if row.nlg > row.linear else ('=' if row.nlg == row.linear else '<')} "
        f"linear {row.linear:.4f}"
        for row in shd.itertuples()
    ]
    n_agree = sum(1 for row in shd.itertuples() if row.nlg > row.linear)
    add(
        "**Descriptive direction vs. 3c.** Pre-stated expectation: NLG skeleton-SHD >\n"
        f"linear skeleton-SHD on chain and triangle. Observed on {n_agree}/"
        f"{len(shd)} corroborating cells:"
    )
    add("")
    for line in agrees:
        add(f"- {line}")
    add("")
    add(
        "Reported as "
        + ("agreement" if n_agree == len(shd) else "a **tension**")
        + " with the pre-stated pattern. Either way **no label moves** — the\n"
        "no-consequence clause is symmetric by registration."
    )
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- 3f ----
    add("## 3f — Descriptive extras")
    add("")
    add(
        "**Normalized interpolation coordinate t = dcost_ld / dcost_l0.** Descriptive\n"
        "only (PS-8); it enters no gate. `t_instability_flag` fires where\n"
        "`|mean dcost_l0|` falls within its own cross-seed SD of zero — a near-zero\n"
        "denominator a reader must not over-read."
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
        add(
            "Not trivially readable from the existing summary CSVs — **deferred to\n"
            "Results**, not derived here."
        )
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

    # ---------------------------------------------------------------- 3g ----
    add("## 3g — H3b NLG-existence fallback tier (PS-8, compute-once seam)")
    add("")
    if orientation_halted:
        add(
            "The orientation channel (3c) emitted **no label** — it is HALTED, not\n"
            "Supported, Mixed or Null. The fallback's invocation condition (3c Mixed\n"
            "or Null) is therefore not satisfiable, and the **fallback-tier control**\n"
            "is itself invalidated by the same collider-linear HARNESS HALT per\n"
            "PS-8's halt-scope rule. **The fallback tier is NOT invoked.**"
        )
    else:
        passing = int(orientation["gate_pass"].sum())
        label = orientation_label(passing)
        if label == "Supported":
            add(
                "3c is **Supported**, so the H3b NLG-existence fallback is **NOT\n"
                "invoked** — stated explicitly rather than omitted."
            )
        else:
            add(
                f"3c is **{label}**, so the fallback tier **is invoked**. It is\n"
                f"discharged by cross-reference to the single banded result at\n"
                f"`{WORSE_THAN_NO_GRAPH_ANCHOR}`; the counts are **not** reprinted here, per "
                f"PS-8's\n"
                "compute-once seam."
            )
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ covars ----
    add("## Diagnostic covariates per instance (PS-8)")
    add("")
    add(
        "`skeleton_shd` on the FULL node set including A (BK constrains A-edge\n"
        "ORIENTATION only, so an A–X adjacency error is a real discovery error).\n"
        "`orientation_wrong_count` on the X–X edges only, marginalized over channel\n"
        "(the strict orientation-error term). `tiebreak_resolved_count` marginalized over\n"
        "correctness (provenance, not error). `wrong∧tiebreak` is the class-(b)\n"
        "diagnostic cell. **nSID** on the ENDOGENOUS sub-graph excluding A, per-topology\n"
        "normalized (d_eff = 3 collider/chain, 2 triangle); descriptive, enters no rule."
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
    violations = int(instances["structural_expectation_violations"].sum())
    if violations:
        flagged = per_seed[per_seed["structural_expectation_violated"]]
        add(
            f"**Structural-expectation violations: {violations}.** Pre-stated (PS-8,\n"
            "per PS-7 note (f)): on a chain/triangle cell with `skeleton_shd = 0`,\n"
            "`tiebreak_resolved_count` = (# X–X edges) and `orientation_wrong_count`\n"
            "= 0. Flagged rows — a readable signal, not noise:"
        )
        add("")
        add(
            _table(
                flagged,
                [
                    "topology",
                    "family",
                    "regime",
                    "seed_idx",
                    "skeleton_shd",
                    "tiebreak_resolved_count",
                    "orientation_wrong_count",
                    "n_xx_true_edges",
                ],
            )
        )
    else:
        add(
            "**Structural-expectation violations: 0.** On every chain/triangle cell\n"
            "with `skeleton_shd = 0`, `tiebreak_resolved_count` equals the number of\n"
            "true X–X edges and `orientation_wrong_count` is 0, as pre-stated\n"
            "(PS-8, per PS-7 note (f))."
        )
    add("")
    total_reversal = int(instances["reversal_seed_count"].sum())
    add(
        f"**Reversal-seed incidence (dcost_l0 < 0): {total_reversal} seed(s) across all\n"
        "12 instances** — the per-instance tally is the `reversal_seed_count` column.\n"
        "This is the incidence at which the corrected partition could change a\n"
        "region assignment relative to the superseded original region glosses (PS-8)."
    )
    add("")
    return "\n".join(out) + "\n"
