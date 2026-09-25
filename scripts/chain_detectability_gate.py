"""PS-2 chain detectability gate — the first-6-seed checkpoint.

    python -m scripts.chain_detectability_gate [--root results/cross_seed_N20]

Evaluates PS-2's detectability gate on the four chain cells, reading the FIRST
SIX seeds of the N=20 grid. HARD CHECKPOINT: the rest of the grid is not
consumed until it passes.

The gate (PS-2, same rule as the collider amendment):

  (i)  sign(Δ_cost(L2)) consistent across all 6 seeds AND
       |mean(Δ_cost(L2))| > per-seed SD of Δ_cost(L2)
  (ii) sign(ΔS) unambiguous under the production S(g)

Condition (ii) is BINDING ON THE EFFECT-MODIFYING ARM ONLY. Under the additive
control ΔS is flat by construction (PS-3 convention 4) — exactly 0 on the linear
branch and within MC noise on the NLG branch — so (ii) checks the PS-3
fit-independence invariance there instead: S(g) must be bit-identical across
seeds. Condition (i) still applies to both arms as a non-degeneracy check.

S(g) VALUES ARE NOT RE-DERIVED HERE. They are read from the grid's own
`summary/s_of_g_by_seed.csv`, cross-checked against the frozen pre-freeze record;
drift is reported as a failure rather than silently preferring one source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icknowledge.analysis.loading import N20_ROOT, N_SEEDS, chain_cells
from icknowledge.analysis.t4_reports import detectability_gate

#: [PS-2] The pre-freeze S(g) record. A drift between the frozen record and the
#: code surfaces here rather than in a thesis table; this check is what feeds the
#: detectability-gate provenance. Keys are (family, regime); values are
#: ΔS := S(−1) − S(+1).
FROZEN_DELTA_S = {
    ("linear", "additive"): 0.000000,
    ("linear", "effect_modifying"): -0.476973,
    ("nlg", "additive"): +0.001771,
    ("nlg", "effect_modifying"): -0.342228,
}

#: Agreement tolerance against the register record. Tight: the linear branch is
#: deterministic and the NLG branch runs at a pinned construction seed, so both
#: should reproduce to the printed precision. Anything looser would let a real
#: coefficient drift pass as rounding.
_RECORD_ATOL = 5e-6


def load_by_cell(root: Path) -> pd.DataFrame:
    path = root / "summary" / "per_seed_by_cell.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run the grid's summary pass first "
            "(scripts/run_cross_seed_grid.py --summary-only)."
        )
    return pd.read_csv(path, float_precision="round_trip")


def load_s_of_g(root: Path) -> pd.DataFrame:
    path = root / "summary" / "s_of_g_by_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run the summary pass first.")
    return pd.read_csv(path, float_precision="round_trip")


def check_against_register(gate: pd.DataFrame) -> list[str]:
    """Cross-check each cell's ΔS against the frozen pre-freeze record."""
    problems = []
    for _, row in gate.iterrows():
        expected = FROZEN_DELTA_S[(row["family"], row["regime"])]
        actual = float(row["delta_S"])
        if abs(actual - expected) > _RECORD_ATOL:
            problems.append(
                f"{row['topology']}/{row['family']}/{row['regime']}: production "
                f"ΔS={actual:+.6f} disagrees with the frozen record {expected:+.6f} "
                f"(|diff|={abs(actual - expected):.2e} > {_RECORD_ATOL:.0e}). Either a "
                "builder coefficient moved after the record was written, or the "
                "record is stale — resolve in the register before using this cell."
            )
    return problems


def format_report(gate: pd.DataFrame, drift: list[str]) -> str:
    lines = [
        "# PS-2 chain detectability gate — first-6-seed checkpoint",
        "",
        "Evaluated on seeds 0-5 of the N=20 grid (`results/cross_seed_N20/`), per",
        "PS-1's staged 6→20 discipline and its prefix property. Same "
        "rule as",
        "the SCM specification's collider amendment.",
        "",
        "- **(i)** `sign(Δ_cost(L2))` consistent across all 6 seeds AND",
        "  `|mean(Δ_cost(L2))| > SD(Δ_cost(L2))`.",
        "- **(ii)** `sign(ΔS)` unambiguous under the production S(g) — binding on the",
        "  **effect-modifying arm only**. Under the additive control ΔS is flat by",
        "  construction (PS-3 convention 4) — exactly 0 on the linear branch,",
        "  within MC noise on the NLG branch — so (ii) checks the PS-3",
        "  fit-independence invariance there instead of a sign.",
        "",
        "| family | regime | mean | sd | \\|mean\\|/sd | cond_i_sign | cond_i_mag | "
        "cond_i | ΔS | ΔS stable | cond_ii | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    yn = {True: "YES", False: "**NO**"}
    for _, r in gate.iterrows():
        lines.append(
            f"| {r['family']} | {r['regime']} | {r['mean']:.6f} | {r['sd']:.6f} | "
            f"{r['abs_mean_over_sd']:.2f} | {yn[r['cond_i_sign_consistent']]} | "
            f"{yn[r['cond_i_magnitude']]} | {yn[r['cond_i_pass']]} | "
            f"{r['delta_S']:+.6f} | {yn[r['delta_S_identical_across_seeds']]} | "
            f"{yn[r['cond_ii_pass']]} | {yn[r['gate_pass']]} |"
        )
    lines += ["", "### Per-seed Δ_cost(L2)", "", "| family | regime | per-seed |", "|---|---|---|"]
    for _, r in gate.iterrows():
        values = ", ".join(f"{v:+.6f}" for v in r["delta_cost_L2_per_seed"])
        lines.append(f"| {r['family']} | {r['regime']} | [{values}] |")

    em = gate[gate["regime"] == "effect_modifying"]
    passed = bool(em["gate_pass"].all()) and not drift
    lines += ["", "### Verdict", ""]
    if drift:
        lines += ["**S(g) DRIFT AGAINST THE FROZEN RECORD:**", ""]
        lines += [f"- {p}" for p in drift] + [""]
    if passed:
        lines += [
            "**GATE PASSES on both chain effect-modifying cells.** PS-2 is discharged",
            "for the chain, joining the triangle and the collider:",
            "all three topologies now have discharged detectability gates.",
            "",
            "The gate is **not re-evaluated at N=20** — PS-1 forecloses re-evaluation",
            "on a growing seed count, which would reintroduce stopping-rule freedom.",
        ]
    else:
        failing = em[~em["gate_pass"]]
        lines += ["**GATE FAILS.** Remaining seeds must NOT be consumed. Failing cells:", ""]
        for _, r in failing.iterrows():
            lines.append(
                f"- chain/{r['family']}/{r['regime']}: cond_i={r['cond_i_pass']}, "
                f"cond_ii={r['cond_ii_pass']}"
            )
        lines += [
            "",
            "the chain SCM specification fallback lever ordering: **(1) γ first** — widens both ΔS "
            "and ΔΔB",
            "through the effect-modified upstream edge, keeping them coupled; **(2) β",
            "or δ next**; **(3) β_{A,2}, β_{A,3} last resort only** — β_A on any",
            "A → X_i edge is NOT in the actionable → classifier propagation sub-block",
            "S(g) measures, so raising it increases Δ_cost(L2) while leaving S(g)",
            "unmoved, decoupling ΔΔB from ΔS. A cell that can only be made detectable",
            "via β_A is a signal that γ is too weak.",
            "",
            "Remediation lands as a follow-up adjudication BEFORE the grid is re-run.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(N20_ROOT))
    parser.add_argument("--out", default=None, help="write the report here as well")
    args = parser.parse_args(argv)

    root = Path(args.root)
    by_cell = load_by_cell(root)
    s_of_g = load_s_of_g(root)

    # n_seeds stays at N_SEEDS (6): PS-2 defines the gate on the first-6 look and
    # PS-1 forbids re-evaluating it on a growing count. The frame is filtered to
    # those seeds rather than the call being widened.
    by_cell = by_cell[by_cell["seed_idx"] < N_SEEDS]
    s_of_g = s_of_g[s_of_g["seed_idx"] < N_SEEDS]

    gate = detectability_gate(by_cell, s_of_g, cells=chain_cells(), n_seeds=N_SEEDS)
    drift = check_against_register(gate)
    report = format_report(gate, drift)
    print(report.encode("ascii", "replace").decode("ascii"))
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nwrote -> {args.out}")

    em = gate[gate["regime"] == "effect_modifying"]
    return 0 if (bool(em["gate_pass"].all()) and not drift) else 1


if __name__ == "__main__":
    raise SystemExit(main())
