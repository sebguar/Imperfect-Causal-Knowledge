"""H4 — group-asymmetric burden degradation: ΔΔB, sign gates, contrast, verdict.

Mechanical reading of H4 / PS-3 under PS-9. The quantities, in difference form
(the correlation form is 0/0 in the control arm, where S(g) has zero
across-group variance by construction):

    ΔS      := S(−1) − S(+1)                    per SCM instance, fit-independent
    ΔB_g(c) := r^CAU_realized(g, c) − r^CAU_realized(g, L2)
    ΔΔB(c)  := ΔB_{−1}(c) − ΔB_{+1}(c)          per instance, per rung, per seed

ΔB_g is read from ``ΔB_g_vs_L2_common_found`` in the per-seed PAIRWISE artifact.
The all-eligible ``ΔB_g_vs_L2`` column in the by-group artifact is a different
population and is never used here; `SOURCE_COLUMN` names the choice in one place
and `load_per_seed_ddb` asserts the column it read. Rungs are c ∈ {L1-oracle,
L0}; L2 is the reference and carries no ΔB by construction (the NaN convention:
self-reference, stored as NaN in the by-group artifact and omitted entirely from
the pairwise artifact), and never reaches an aggregation here.

Verdict procedure per rung (PS-9): per-seed gate
``count(sign(ΔS·ΔΔB) > 0) >= SIGN_GATE_K`` (16 of 20); instance tally >=5/6
Supported, 4/6 Mixed, <=3/6 Null; topology contrast |mean ΔΔB|_treatment >
|mean ΔΔB|_control, failing on >=2 topologies forces Null; the
outlier-discrimination clause; the saturated-topology note. The supporting
correlation enters no threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from icknowledge.analysis.loading import (
    N20_ROOT,
    N_SEEDS_FULL,
    Cell,
    cell_frame,
    grid_cells,
)

#: The common-found PRIMARY population column. Named once so the choice is auditable and
#: cannot drift to the all-eligible column, which is a different population.
SOURCE_COLUMN = "ΔB_g_vs_L2_common_found"

#: Degraded rungs H4 is read at. L2 is the reference and has no ΔB (the NaN convention).
RUNGS = ("L1-oracle", "L0")

#: [PS-9] Per-seed sign-agreement gate: an EM instance
#: shows the predicted sign at a rung iff at least this many of the 20 seeds
#: satisfy sign(ΔS·ΔΔB) > 0. Calibrated at ~1.2% two-tailed under a coin-flip null.
SIGN_GATE_K = 16

#: [PS-9] Instance-level tally thresholds, unchanged.
SUPPORTED_MIN_INSTANCES = 5
MIXED_MIN_INSTANCES = 4

#: [PS-9] Contrast failures on this many topologies force Null.
CONTRAST_FAILURES_FORCING_NULL = 2

#: |ΔS| at or below which an instance counts as LOW-propagation for the
#: outlier-discrimination clause. The six EM instances split cleanly either side:
#: the NLG triangle sits at 0.185 and the NLG collider at 0.299, against 0.342,
#: 0.405, 0.477 and 0.600 above. 0.30 is the gap, not a tuned value — but it is a
#: judgement call and is stated rather than buried.
LOW_DELTA_S_THRESHOLD = 0.30

#: Tolerance on the additive arm's ΔS: flat by construction (PS-3 convention 4),
#: exactly 0 on the linear branch and MC noise on the NLG branch.
ADDITIVE_DELTA_S_ATOL = 0.002

#: [Extended] A control instance with at most this many crossings in a
#: consistent direction reads as an attenuation candidate rather than absence.
CONTROL_ATTENUATION_MAX_CROSSINGS = 4

#: |mean ΔΔB| below which BOTH arms of a topology count as saturated-tiny, so the
#: contrast is reported as directionally-correct-but-attenuated rather than as a
#: strict inequality (PS-9's contrast-on-saturated-topology note).
SATURATED_ABS_DDB = 0.02


# --------------------------------------------------------------------------- #
# Loading — per-seed ΔΔB from the frozen pairwise artifacts
# --------------------------------------------------------------------------- #


def load_per_seed_ddb(
    root: Path = N20_ROOT,
    n_seeds: int = N_SEEDS_FULL,
    cells: list[Cell] | None = None,
) -> pd.DataFrame:
    """Per (cell, seed, rung) ΔΔB from the per-seed pairwise artifacts.

    Defaults to the N=20 tree deliberately: `loading.DEFAULT_ROOT` still points at
    the frozen `cross_seed_N6/` so the N=6 reports keep reproducing off the artifact
    they were written against, and a H4 call that silently picked that up would
    compute a six-seed verdict against a twenty-seed rule.

    ``cells`` narrows the enumeration — used by the tests to exercise the loader on
    a single hand-built cell. Production reads the full grid, and the missing-file
    error below is what stops a partial tree being averaged as if complete.

    Returns columns: topology, family, regime, seed_idx, seed, condition,
    ddB, N_common_found_neg, N_common_found_pos.
    """
    rows = []
    for cell in cells if cells is not None else grid_cells():
        for seed_idx in range(n_seeds):
            path = (
                cell.seed_dir(seed_idx, root)
                / f"aggregate_pairwise_{cell.regime}.csv"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"missing pairwise artifact {path} — H4 reads a COMPLETE grid; "
                    "a partial tree would silently average whichever cells exist."
                )
            frame = pd.read_csv(path, float_precision="round_trip")
            if SOURCE_COLUMN not in frame.columns:
                raise KeyError(
                    f"{path} has no {SOURCE_COLUMN!r} column. H4 reads the common-found population "
                    "COMMON-FOUND population; the all-eligible ΔB_g_vs_L2 column is "
                    "a different population and is not a substitute."
                )
            # The NaN convention: the L2 self-reference row carries no ΔB. It is absent from the
            # pairwise artifact entirely (and NaN in the by-group one). Dropped
            # explicitly so the guard holds whichever artifact shape is handed in.
            frame = frame[frame["condition"] != "L2"]
            for condition in RUNGS:
                sub = frame[frame["condition"] == condition]
                if len(sub) != 2:
                    raise ValueError(
                        f"{path} [{condition}]: expected exactly 2 group rows, got "
                        f"{len(sub)} — ΔΔB is a two-group difference."
                    )
                neg = sub[sub["group"] == -1].iloc[0]
                pos = sub[sub["group"] == 1].iloc[0]
                values = (float(neg[SOURCE_COLUMN]), float(pos[SOURCE_COLUMN]))
                if not all(np.isfinite(values)):
                    # Hard error, never a silent nanmean: a NaN here means an L2
                    # row leaked through or a cell produced no common-found pool.
                    raise ValueError(
                        f"{path} [{condition}]: non-finite {SOURCE_COLUMN} "
                        f"{values} reached the ΔΔB difference. Under the NaN convention, NaN is a "
                        "reference-row marker, not a value to average over."
                    )
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "seed_idx": seed_idx,
                        "seed": int(neg["seed"]),
                        "condition": condition,
                        "ddB": values[0] - values[1],
                        "N_common_found_neg": int(neg["N_common_found_pairwise_vs_L2"]),
                        "N_common_found_pos": int(pos["N_common_found_pairwise_vs_L2"]),
                    }
                )
    return pd.DataFrame(rows)


def load_delta_s(root: Path = N20_ROOT) -> pd.DataFrame:
    """Production ΔS per cell, from the grid's own s_of_g_by_seed.csv.

    Asserts the PS-3 fit-independence invariant operationally: S(g) must be
    identical at every seed, so a value that moved with the seed would mean the
    experiment seed leaked into a descriptor defined to be fit-free.
    """
    frame = pd.read_csv(root / "summary" / "s_of_g_by_seed.csv", float_precision="round_trip")
    rows = []
    for cell in grid_cells():
        sub = cell_frame(frame, cell)
        values = sub["ΔS"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"{cell.label}: no ΔS rows in s_of_g_by_seed.csv")
        if not np.all(values == values[0]):
            raise ValueError(
                f"{cell.label}: ΔS varies across seeds "
                f"({values.min():+.9f} .. {values.max():+.9f}). S(g) is "
                "fit-independent by construction (PS-3) — a seed-varying value "
                "means the run seed leaked into the descriptor."
            )
        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "delta_S": float(values[0]),
            }
        )
    return pd.DataFrame(rows)


def assert_delta_s_preconditions(delta_s: pd.DataFrame) -> None:
    """EM instances carry ΔS < 0; additive instances are flat within MC noise.

    Both are guaranteed by construction (the chain SCM specification's sign convention; PS-3
    convention
    4), so a violation means the frozen coefficients are not the ones the register
    records — checked before pairing rather than trusted, because every sign gate
    below multiplies by sign(ΔS).
    """
    em = delta_s[delta_s["regime"] == "effect_modifying"]
    bad_em = em[em["delta_S"] >= 0]
    if not bad_em.empty:
        raise ValueError(
            "effect-modifying ΔS must be negative on all six instances (the chain-SCM sign "
            f"convention); violations:\n{bad_em.to_string(index=False)}"
        )
    control = delta_s[delta_s["regime"] == "additive"]
    bad_ctrl = control[control["delta_S"].abs() > ADDITIVE_DELTA_S_ATOL]
    if not bad_ctrl.empty:
        raise ValueError(
            f"additive ΔS must be flat within {ADDITIVE_DELTA_S_ATOL} (PS-3 "
            f"convention 4); violations:\n{bad_ctrl.to_string(index=False)}"
        )


# --------------------------------------------------------------------------- #
# Per-cell aggregation + the sign gate
# --------------------------------------------------------------------------- #


def by_cell(per_seed: pd.DataFrame, delta_s: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, rung): mean/SD ΔΔB, seed sign counts, and the PS-9 gate."""
    merged = per_seed.merge(delta_s, on=["topology", "family", "regime"], how="left")
    if merged["delta_S"].isna().any():
        raise ValueError("a cell has per-seed ΔΔB but no ΔS — pairing is incomplete.")

    records = []
    keys = ["topology", "family", "regime", "condition"]
    for key_values, sub in merged.groupby(keys, sort=False):
        record = dict(zip(keys, key_values, strict=True))
        values = sub.sort_values("seed_idx")["ddB"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{record}: non-finite ΔΔB reached aggregation.")
        ds = float(sub["delta_S"].iloc[0])

        n_neg = int((values < 0).sum())
        n_pos = int((values > 0).sum())
        n_zero = int((values == 0).sum())
        # sign(ΔS · ΔΔB) > 0. On the EM arm ΔS < 0, so this counts ΔΔB < 0. On the
        # control arm ΔS ~ 0 and the product is meaningless — the count is reported
        # for the absence-vs-attenuation read, never fed to a gate.
        agree = int((np.sign(ds) * np.sign(values) > 0).sum())

        record.update(
            {
                "delta_S": ds,
                "n_seeds": int(values.size),
                "mean_ddB": float(values.mean()),
                "sd_ddB": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "min_ddB": float(values.min()),
                "max_ddB": float(values.max()),
                "n_seeds_neg": n_neg,
                "n_seeds_pos": n_pos,
                "n_seeds_zero": n_zero,
                "n_seeds_sign_agree": agree,
            }
        )
        if record["regime"] == "effect_modifying":
            record["sign_gate_pass"] = bool(agree >= SIGN_GATE_K)
            record["gate_detail"] = f"{agree}/{values.size} (need ≥{SIGN_GATE_K})"
            record["control_read"] = ""
        else:
            record["sign_gate_pass"] = None
            record["gate_detail"] = ""
            record["control_read"] = _control_read(n_neg, n_pos)
        records.append(record)
    return pd.DataFrame(records)


def _control_read(n_neg: int, n_pos: int) -> str:
    """PS-9's absence-vs-attenuation label for one control instance."""
    crossings = min(n_neg, n_pos)
    if crossings <= CONTROL_ATTENUATION_MAX_CROSSINGS:
        direction = "negative" if n_neg > n_pos else "positive"
        return (
            f"ATTENUATION-CANDIDATE (non-flat): {max(n_neg, n_pos)}/{n_neg + n_pos} "
            f"seeds {direction}, {crossings} crossing(s)"
        )
    return (
        f"ABSENCE (flat within seed noise): {n_neg} negative / {n_pos} positive, "
        f"{crossings} crossing(s)"
    )


# --------------------------------------------------------------------------- #
# Regime contrast (per topology) and the supporting correlation
# --------------------------------------------------------------------------- #


def regime_contrast(cells: pd.DataFrame) -> pd.DataFrame:
    """|mean ΔΔB|_treatment vs |mean ΔΔB|_control, per topology × family and per topology.

    PS-9 reads the contrast requirement PER TOPOLOGY ("across the three
    topologies"), so the per-topology row is the one the verdict consumes; the
    per-family rows are reported alongside because PS-8 permits the cross-family
    reading within a topology and it is where a saturated cell shows itself.
    """
    records = []
    for condition in RUNGS:
        rung = cells[cells["condition"] == condition]
        for topology in rung["topology"].unique():
            topo = rung[rung["topology"] == topology]
            for family in [*sorted(topo["family"].unique()), None]:
                scope = topo if family is None else topo[topo["family"] == family]
                treat = scope[scope["regime"] == "effect_modifying"]["mean_ddB"]
                ctrl = scope[scope["regime"] == "additive"]["mean_ddB"]
                # Mean of |mean ΔΔB| over the cells in scope: on the per-topology
                # row that is the average of the two families' magnitudes, which is
                # the level PS-9's "across the three topologies" phrase reads at.
                t_abs = float(treat.abs().mean())
                c_abs = float(ctrl.abs().mean())
                saturated = max(t_abs, c_abs) < SATURATED_ABS_DDB
                records.append(
                    {
                        "condition": condition,
                        "topology": topology,
                        "family": "ALL" if family is None else family,
                        "abs_mean_ddB_treatment": t_abs,
                        "abs_mean_ddB_control": c_abs,
                        "contrast_holds": bool(t_abs > c_abs),
                        "ratio_treatment_over_control": (
                            t_abs / c_abs if c_abs > 0 else float("inf")
                        ),
                        "saturated_both_arms": bool(saturated),
                    }
                )
    return pd.DataFrame(records)


def supporting_correlation(cells: pd.DataFrame) -> dict[str, float]:
    """the inference stance/PS-8 supporting r over the SIX (ΔS, mean ΔΔB) EM points, per rung.

    Within-topology difference form — six points, one per effect-modifying SCM
    instance — never twelve raw (S(g), ΔB_g) points, because both axes are
    topology-scaled (PS-8). SUPPORTING evidence only (the inference stance): it enters no threshold
    and no verdict branch.
    """
    out = {}
    for condition in RUNGS:
        em = cells[(cells["condition"] == condition) & (cells["regime"] == "effect_modifying")]
        x = em["delta_S"].to_numpy(dtype=float)
        y = em["mean_ddB"].to_numpy(dtype=float)
        out[condition] = (
            float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.ptp(x) > 0 else float("nan")
        )
    return out


# --------------------------------------------------------------------------- #
# The verdict — pure, unit-testable, mechanical
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Verdict:
    """One rung's PS-9 verdict plus every quantity the label was derived from."""

    condition: str
    label: str
    n_instances_passing: int
    n_instances: int
    instance_detail: dict[str, str] = field(default_factory=dict)
    failing_instances: tuple[str, ...] = ()
    outlier_clause: str = ""
    contrast_failures: tuple[str, ...] = ()
    contrast_detail: dict[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def verdict_for_rung(
    cells: pd.DataFrame, contrast: pd.DataFrame, condition: str
) -> Verdict:
    """Apply PS-9 (as extended) mechanically. No narration, no judgement.

    Order of operations is PS-9's own: seed gate -> instance tally -> contrast ->
    outlier-discrimination clause. The contrast can force Null and the outlier
    clause can force Mixed; neither can promote a label upward.
    """
    em = cells[(cells["condition"] == condition) & (cells["regime"] == "effect_modifying")]
    em = em.sort_values(["topology", "family"])
    detail = {
        f"{r.topology}/{r.family}": f"{r.gate_detail} — {'PASS' if r.sign_gate_pass else 'FAIL'}"
        for r in em.itertuples()
    }
    passing = int(em["sign_gate_pass"].sum())
    total = int(len(em))
    failing = tuple(
        f"{r.topology}/{r.family}" for r in em.itertuples() if not r.sign_gate_pass
    )
    reasons: list[str] = []

    if passing >= SUPPORTED_MIN_INSTANCES:
        label = "Supported"
    elif passing >= MIXED_MIN_INSTANCES:
        label = "Mixed"
    else:
        label = "Null"
    reasons.append(f"{passing}/{total} instances pass the {SIGN_GATE_K}/20 seed gate → {label}")

    # --- contrast, read per topology (PS-9: "across the three topologies") -----
    scope = contrast[(contrast["condition"] == condition) & (contrast["family"] == "ALL")]
    contrast_detail = {
        r.topology: (
            f"|ΔΔB|_treat {r.abs_mean_ddB_treatment:.6f} vs |ΔΔB|_ctrl "
            f"{r.abs_mean_ddB_control:.6f} → {'holds' if r.contrast_holds else 'FAILS'}"
            + (" [both arms saturated-tiny]" if r.saturated_both_arms else "")
        )
        for r in scope.itertuples()
    }
    contrast_failures = tuple(
        r.topology
        for r in scope.itertuples()
        if not r.contrast_holds and not r.saturated_both_arms
    )
    if len(contrast_failures) >= CONTRAST_FAILURES_FORCING_NULL:
        label = "Null"
        reasons.append(
            f"contrast fails on {len(contrast_failures)} topologies "
            f"({', '.join(contrast_failures)}) → forced Null (PS-9 Null clause)"
        )
    elif contrast_failures:
        if label == "Supported":
            label = "Mixed"
        reasons.append(
            f"contrast fails on 1 topology ({contrast_failures[0]}) → "
            "holds on some topologies but not others → Mixed (PS-9 Mixed clause)"
        )

    # --- outlier-discrimination clause (anti-tautology, PS-3) ------------------
    outlier = ""
    if len(failing) == 1:
        row = em[em.apply(lambda r: f"{r.topology}/{r.family}" == failing[0], axis=1)].iloc[0]
        magnitude = abs(float(row["delta_S"]))
        low = magnitude <= LOW_DELTA_S_THRESHOLD
        outlier = (
            f"{failing[0]} fails with |ΔS| = {magnitude:.6f} → "
            f"{'LOW' if low else 'HIGH'}-propagation instance"
        )
        if low:
            outlier += (
                "; tolerated — small |ΔΔB| with unstable sign is consistent with "
                "S(g) tracking burden (PS-9 outlier clause)"
            )
            reasons.append(f"outlier clause: {outlier}")
        else:
            outlier += (
                "; counts AGAINST the theory — a high-|ΔS| instance that does not "
                "conform forces Mixed even at 5/6 (PS-9 outlier clause)"
            )
            if label == "Supported":
                label = "Mixed"
            reasons.append(f"outlier clause forces Mixed: {outlier}")

    return Verdict(
        condition=condition,
        label=label,
        n_instances_passing=passing,
        n_instances=total,
        instance_detail=detail,
        failing_instances=failing,
        outlier_clause=outlier,
        contrast_failures=contrast_failures,
        contrast_detail=contrast_detail,
        reasons=tuple(reasons),
    )


def common_found_is_no_op(per_seed: pd.DataFrame, root: Path = N20_ROOT) -> pd.DataFrame:
    """the common-found population watch-for: does common-found still equal all-eligible at N=20?

    The N=6 finding was N_found == N_eligible everywhere, which made the
    common-found population a no-op. That claim is re-checked rather than carried
    forward: the analysis uses common-found regardless (it is the common-found population's
    primary), but
    whether the no-op still holds is reportable.
    """
    rows = []
    for cell in grid_cells():
        for seed_idx in sorted(per_seed["seed_idx"].unique()):
            by_group = pd.read_csv(
                cell.seed_dir(seed_idx, root) / f"aggregate_by_group_{cell.regime}.csv",
                float_precision="round_trip",
            )
            pairwise = pd.read_csv(
                cell.seed_dir(seed_idx, root) / f"aggregate_pairwise_{cell.regime}.csv",
                float_precision="round_trip",
            )
            for condition in RUNGS:
                for group in (-1, 1):
                    bg = by_group[
                        (by_group["condition"] == condition) & (by_group["group"] == group)
                    ]
                    pw = pairwise[
                        (pairwise["condition"] == condition) & (pairwise["group"] == group)
                    ]
                    if bg.empty or pw.empty:
                        continue
                    n_elig = int(bg.iloc[0]["N_eligible"])
                    n_common = int(pw.iloc[0]["N_common_found_pairwise_vs_L2"])
                    if n_elig != n_common:
                        rows.append(
                            {
                                "topology": cell.topology,
                                "family": cell.family,
                                "regime": cell.regime,
                                "seed_idx": seed_idx,
                                "condition": condition,
                                "group": group,
                                "N_eligible": n_elig,
                                "N_common_found": n_common,
                                "difference": n_elig - n_common,
                            }
                        )
    return pd.DataFrame(rows)


def bimodality_flags(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Watch-for: EM instances whose per-seed ΔΔB is plausibly bimodal.

    Flagged by a simple, stated heuristic — a gap in the sorted per-seed values
    wider than the SD, with at least 3 seeds on each side. Deliberately crude and
    descriptive: the point is to say "the mean/SD row understates structure here,
    look at the per-seed points", not to fit a mixture.
    """
    rows = []
    keys = ["topology", "family", "regime", "condition"]
    for key_values, sub in per_seed.groupby(keys, sort=False):
        values = np.sort(sub["ddB"].to_numpy(dtype=float))
        if values.size < 6:
            continue
        sd = float(values.std(ddof=1))
        gaps = np.diff(values)
        idx = int(np.argmax(gaps))
        widest = float(gaps[idx])
        left, right = idx + 1, values.size - idx - 1
        if sd > 0 and widest > sd and min(left, right) >= 3:
            record = dict(zip(keys, key_values, strict=True))
            record.update(
                {
                    "widest_gap": widest,
                    "sd": sd,
                    "gap_over_sd": widest / sd,
                    "n_left": left,
                    "n_right": right,
                    "split_at": float(values[idx]),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Markdown table; pipes escaped because GFM splits rows before parsing inline."""
    pipe = "\\|"
    lines = [
        "| " + " | ".join(str(c).replace("|", pipe) for c in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (bool, np.bool_)):
                cells.append("YES" if value else "**NO**")
            elif value is None:
                cells.append("n/a")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(value) else f"{value:+.6f}")
            else:
                cells.append(str(value).replace("|", pipe))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_verdict_report(
    cells: pd.DataFrame,
    contrast: pd.DataFrame,
    verdicts: dict[str, Verdict],
    correlation: dict[str, float],
    no_op: pd.DataFrame,
    bimodal: pd.DataFrame,
) -> str:
    """The mechanical PS-9 verdict document. Every number traceable, nothing narrated."""
    lines = [
        "# H4 verdict — mechanical PS-9 reading at N=20",
        "",
        "Produced by `icknowledge/analysis/h4.py`. The label below is the OUTPUT of a",
        "pure function applying **PS-9**, as operationalized in its decisions, to",
        "frozen numbers — it is not narrated, and no threshold in it was chosen after",
        "the numbers existed.",
        "",
        "- **ΔΔB(c)** := ΔB_{−1}(c) − ΔB_{+1}(c), on the **common-found**",
        f"  population (`{SOURCE_COLUMN}`).",
        "- **Per-seed gate:** an EM instance shows the predicted sign at rung",
        f"  c iff ≥ **{SIGN_GATE_K}/20** seeds satisfy sign(ΔS·ΔΔB) > 0.",
        "- **Instance tally (PS-9):** ≥5/6 Supported · 4/6 Mixed · ≤3/6 Null.",
        "- **SD is descriptive only**; no SE, no CI (the inference stance).",
        "- L2 is the reference rung and carries no ΔB (the NaN convention); it never enters an",
        "  aggregation here.",
        "",
        "---",
        "",
    ]

    for condition in RUNGS:
        verdict = verdicts[condition]
        lines += [f"## Rung {condition}", "", f"### VERDICT: **{verdict.label}**", ""]
        lines += [f"- {reason}" for reason in verdict.reasons]
        lines += ["", "#### Two-level tally — the six effect-modifying instances", ""]
        em = cells[
            (cells["condition"] == condition) & (cells["regime"] == "effect_modifying")
        ].sort_values(["topology", "family"])
        lines += _table(
            em,
            [
                "topology", "family", "delta_S", "mean_ddB", "sd_ddB",
                "n_seeds_neg", "n_seeds_pos", "n_seeds_sign_agree", "sign_gate_pass",
            ],
        )
        lines += [
            "",
            f"**{verdict.n_instances_passing}/{verdict.n_instances} instances pass** "
            f"the {SIGN_GATE_K}/20 seed gate.",
            "",
        ]
        if verdict.failing_instances:
            lines += [f"Failing instance(s): {', '.join(verdict.failing_instances)}.", ""]
        if verdict.outlier_clause:
            lines += [f"**Outlier-discrimination clause:** {verdict.outlier_clause}", ""]
        else:
            lines += [
                "**Outlier-discrimination clause:** not triggered "
                f"({len(verdict.failing_instances)} instance(s) failing; the clause "
                "reads when exactly one does).",
                "",
            ]

        lines += ["#### Additive-control count read (absence vs. attenuation)", ""]
        control = cells[
            (cells["condition"] == condition) & (cells["regime"] == "additive")
        ].sort_values(["topology", "family"])
        lines += _table(
            control,
            ["topology", "family", "mean_ddB", "sd_ddB", "n_seeds_neg", "n_seeds_pos",
             "control_read"],
        )

        lines += ["", "#### Regime contrast (per topology — the level PS-9 reads at)", ""]
        scope = contrast[contrast["condition"] == condition].sort_values(
            ["topology", "family"]
        )
        lines += _table(
            scope,
            [
                "topology", "family", "abs_mean_ddB_treatment", "abs_mean_ddB_control",
                "ratio_treatment_over_control", "contrast_holds", "saturated_both_arms",
            ],
        )
        lines += [""]
        # State exemptions explicitly. A topology whose BOTH arms are
        # saturated-tiny is exempted from the strict inequality by PS-9's
        # contrast-on-saturated-topology note, so it can show contrast_holds = NO in
        # the table while not counting as a failure. Saying "holds on all three"
        # there would contradict the table directly above it.
        exempt = tuple(
            r.topology
            for r in scope[scope["family"] == "ALL"].itertuples()
            if not r.contrast_holds and r.saturated_both_arms
        )
        if verdict.contrast_failures:
            lines += [f"Contrast FAILS on: {', '.join(verdict.contrast_failures)}.", ""]
        elif exempt:
            lines += [
                f"No contrast failure counted. {', '.join(exempt)}: strict inequality "
                "does NOT hold, but both arms are saturated-tiny "
                f"(|mean ΔΔB| < {SATURATED_ABS_DDB}), so PS-9's "
                "contrast-on-saturated-topology note reads it as "
                "directionally-correct-but-attenuated rather than as a failure. The "
                "remaining topologies satisfy the strict inequality.",
                "",
            ]
        else:
            lines += ["Contrast holds on all three topologies (strict inequality).", ""]

        r = correlation[condition]
        lines += [
            "#### Supporting correlation (the inference stance / PS-8)",
            "",
            f"Pearson r over the **six** (ΔS, mean ΔΔB) effect-modifying points: "
            f"**r = {r:+.4f}**.",
            "",
            "Within-topology difference form — six points, one per effect-modifying SCM",
            "instance — never twelve raw (S(g), ΔB_g) points, because both axes are",
            "topology-scaled (PS-8). **Supporting evidence only (the inference stance):** it "
            "enters no",
            "threshold and no verdict branch.",
            "",
            "---",
            "",
        ]

    lines += ["## Signal-2 discriminator reading (PS-9)", ""]
    non_flat = cells[
        (cells["regime"] == "additive")
        & (cells["control_read"].str.startswith("ATTENUATION"))
    ]
    if non_flat.empty:
        lines += [
            "Every additive-control instance reads as **absence** (flat within seed",
            "noise) at both rungs. Per the discriminator rule the clean contrast",
            "**rules out**",
            "the PS-9 Signal-2 channel (estimation error on the unmodified,",
            "regime-shared downstream edge) as the driver of ΔΔB: a regime-shared",
            "channel would move both arms.",
        ]
    else:
        lines += [
            f"**{len(non_flat)} additive-control (instance, rung) reading(s) are",
            "NON-FLAT** (attenuation candidates). Per the discriminator rule this",
            "makes the",
            "PS-9 Signal-2 channel a **LIVE alternative explanation** for ΔΔB —",
            "estimation error on the unmodified, regime-shared downstream edge",
            "`coef_error[X₂→X₃]` would move BOTH arms, which is what a non-flat control",
            "looks like. It **must be named in the H4 write-up**, not left in the",
            "exploratory appendix.",
            "",
        ]
        lines += _table(
            non_flat.sort_values(["condition", "topology", "family"]),
            ["condition", "topology", "family", "mean_ddB", "control_read"],
        )
    lines += ["", "---", ""]

    lines += ["## Watch-fors", "", "### common-found vs. all-eligible", ""]
    if no_op.empty:
        lines += [
            "`N_common_found == N_eligible` in every (cell, seed, rung, group). The",
            "the common-found population N=6 no-op finding **still holds at N=20** — the "
            "common-found",
            "population changes no number here, though it remains the primary",
            "population by the common-found population regardless.",
        ]
    else:
        lines += [
            f"**The no-op claim NO LONGER HOLDS**: {len(no_op)} (cell, seed, rung,",
            "group) readings have `N_common_found < N_eligible`. The analysis still uses",
            "common-found (the common-found population's primary population regardless); what "
            "stops being",
            "carried forward is the *no-op* claim. Divergent cells:",
            "",
        ]
        summary = (
            no_op.groupby(["topology", "family", "regime"])
            .agg(n_readings=("difference", "size"), max_difference=("difference", "max"))
            .reset_index()
        )
        lines += _table(
            summary, ["topology", "family", "regime", "n_readings", "max_difference"]
        )

    lines += ["", "### Per-seed ΔΔB bimodality", ""]
    if bimodal.empty:
        lines += [
            "No (cell, rung) shows a per-seed ΔΔB gap wider than its own SD with ≥3",
            "seeds either side.",
        ]
    else:
        lines += [
            "The mean/SD rows **understate structure** in the following (cell, rung)",
            "readings — a gap in the sorted per-seed ΔΔB wider than the cell's own SD,",
            "with at least 3 seeds on each side. See the per-seed points in fig4.",
            "",
        ]
        lines += _table(
            bimodal.sort_values(["condition", "topology", "family"]),
            ["topology", "family", "regime", "condition", "split_at", "gap_over_sd",
             "n_left", "n_right"],
        )
    lines.append("")
    return "\n".join(lines)
