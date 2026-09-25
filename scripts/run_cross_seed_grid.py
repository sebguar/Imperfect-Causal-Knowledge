"""Cross-seed grid runner: 12 cells x N seeds.

12 cells = 3 topologies (triangle, collider, chain) x 2 families x 2 regimes,
enumerated with topology OUTERMOST and "chain" appended, so cell positions and
per-seed directory names are stable.

    python -m scripts.run_cross_seed_grid                  # full grid + summary
    python -m scripts.run_cross_seed_grid --dry-run        # ONE cell, dry-run gate
    python -m scripts.run_cross_seed_grid --summary-only   # re-derive summaries
    # [PS-4] four-rung L1-discovered grid, own tree:
    python -m scripts.run_cross_seed_grid --root results/cross_seed_L1d_N20 \\
        --n-seeds 20 --l1-discovered

Run with ``-m`` from the repo root; invoking the file directly fails to import
icknowledge. Set the machine's sleep/standby timeouts to Never before launching.

[PS-1] N pinned at 20, staged 6 -> 20 by appending.
[seeding] ONE spawn(N) from meta-entropy 20260710
    (`utils.seeding.experiment_run_seeds`) shared across cells; `sg_mc` is a
    separate spawn key.
[PS-9] Reporting is mean / SD / min / max, not significance testing; no p-value
is computed here.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from icknowledge.aggregation import (
    AGGREGATION_CODE_VERSION,
    CellKeys,
    aggregate_run,
    pairwise_path,
)
from icknowledge.classifier import classifier_provenance
from icknowledge.descriptor import s_of_g
from icknowledge.estimation import estimation_provenance
from icknowledge.recourse import run_chain, run_collider, run_triangle
from icknowledge.recourse.pipeline import run_regime
from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_classifier_params, write_run_manifest
from icknowledge.utils.seeding import EXPERIMENT_META_ENTROPY, experiment_run_seeds

#: PS-1's early look. Deliberately a default, not a constant baked into the loop:
#: The full grid raises it to 20 and the seed-generation prefix property makes seeds 0-5
#: identical.
N_SEEDS_EARLY_LOOK = 6

# The chain joins triangle and collider as the third topology.
# ORDER IS LOAD-BEARING for backward compatibility: `grid_cells()` iterates
# topologies outermost, so appending "chain" leaves every existing (triangle,
# collider) cell at the same position in the enumeration and every existing per-seed
# directory name untouched. Prepending it would not change any OUTPUT, but it would
# reorder the wall-clock log and the summary CSVs for no reason.
TOPOLOGIES = ("triangle", "collider", "chain")
FAMILIES = ("linear", "nlg")
REGIMES = ("additive", "effect_modifying")

#: (topology, family) -> base config. The config supplies the classifier block,
#: the grid-resolution-pinned grid resolution and the actionable set; this driver overrides
#: ONLY ``seed`` and ``output_dir``, so every cell keeps the exact numerical
#: envelope its single-seed reference runs used.
_CONFIGS = {
    ("triangle", "linear"): "configs/recourse_triangle.yaml",
    ("triangle", "nlg"): "configs/recourse_triangle_nlg.yaml",
    ("collider", "linear"): "configs/recourse_collider.yaml",
    ("collider", "nlg"): "configs/recourse_collider_nlg.yaml",
    ("chain", "linear"): "configs/recourse_chain.yaml",
    ("chain", "nlg"): "configs/recourse_chain_nlg.yaml",
}

#: topology -> the runner module whose family table and condition tuple govern.
_RUNNERS = {"triangle": run_triangle, "collider": run_collider, "chain": run_chain}

#: Summary artifacts live under this subdirectory of the grid root.
SUMMARY_DIR = "summary"

# [PS-4] The four-rung condition set, in `aggregation._CONDITION_ORDER`'s
# canonical ladder order. Defined HERE rather than in the runner modules on
# purpose: the three `_CONDITIONS` tuples are frozen three-rung artifacts that two
# tests assert on, and PS-4 puts the L1-discovered series in a SEPARATE TREE — so
# the fourth rung is a property of the grid INVOCATION (``--l1-discovered``), not
# of a topology's driver. Threaded into `run_cell_seed` as an explicit argument;
# nothing mutates `runner._CONDITIONS`.
L1D_CONDITIONS = ("L0", "L1-oracle", "L1-discovered", "L2")

#: Cross-seed dispersion uses the SAMPLE SD (ddof=1) — the 6 seeds are a sample
#: of the run-to-run distribution, not the population. The inference stance's reporting format is
#: mean/SD/min/max; the PS-2 gate reads |mean| vs SD, so this choice is
#: load-bearing and is stated rather than left to a library default.
_SD_DDOF = 1


def grid_cells() -> list[tuple[str, str, str]]:
    """The 8 cells: topology x family x regime, in a fixed reporting order."""
    return [(t, f, r) for t in TOPOLOGIES for f in FAMILIES for r in REGIMES]


def cell_dir_name(topology: str, family: str, regime: str, seed_idx: int) -> str:
    return f"{topology}_{family}_{regime}_seed_{seed_idx:02d}"


def _build_cfg(topology: str, family: str, seed: int, out_dir: Path) -> DictConfig:
    """Load the cell's base config and override ONLY seed + output_dir."""
    cfg = load_config(_CONFIGS[(topology, family)])
    cfg.seed = int(seed)
    cfg.output_dir = str(out_dir)
    return cfg


def _feature_and_actionable_sets(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """(V_h, S) for S(g): the classifier's features and the actionable set.

    Read from the config rather than hardcoded per topology so the S(g) call
    cannot drift from the cell the run actually used (PS-3 keeps S(g)
    fit-independent, but it is NOT topology-independent).
    """
    return [str(k) for k in cfg.classifier.beta], [str(v) for v in cfg.recourse.actionable]


# --------------------------------------------------------------------------- #
# One (topology, family, regime, seed) run
# --------------------------------------------------------------------------- #


def run_cell_seed(
    topology: str,
    family: str,
    regime: str,
    seed_idx: int,
    seed: int,
    root: Path,
    n_seeds: int = N_SEEDS_EARLY_LOOK,
    conditions: tuple[str, ...] | None = None,
) -> dict:
    """Run one per-seed cell to completion; return its per-seed run record.

    Returns the row for ``summary/wall_clock_log.csv``: cell keys (topology,
    family, regime, seed_idx, seed), start/finish timestamps, timing columns,
    ``n_eligible``, ``eps_scale`` and ``test_accuracy``.

    Writes, into ``root/<cell>_seed_<ii>/``: one per-individual scoring CSV per
    condition, the three aggregate CSVs (by-group, by-cell, pairwise),
    ``classifier_params.json`` and the run manifest.

    ONE CLASSIFIER PER (cell, seed): ``run_regime`` calls ``build_classifier``
    once, above the condition loop; L0, L1-oracle and L2 share that fitted h.

    [PS-1] N=20 is the ceiling; [PS-2-amendment] the detectability gate is not
    re-tuned on new seeds.
    """
    runner = _RUNNERS[topology]
    scm_builder, form_spec_fn = runner._FAMILIES[family]
    # [PS-4] ``conditions=None`` means "whatever this topology's driver
    # declares" — the frozen three-rung tuple, so a grid launched without
    # ``--l1-discovered`` is bit-identical to the sealed three-rung output. An explicit
    # tuple (the ``L1D_CONDITIONS`` four-rung set) OVERRIDES it for this call only;
    # `runner._CONDITIONS` is read, never written.
    if conditions is None:
        conditions = runner._CONDITIONS
    # [G1, the ℓ₂ intervention-cost convention] Anchor gated to LINEAR. The
    # Ehyaei
    # closed form assumes a linear SCM and a linear classifier, so it is
    # inapplicable — not merely loose — on an NLG cell, and run_regime refuses
    # run_anchor=True there outright. Matching each runner's own rule keeps the
    # grid's linear cells numerically identical to their reference runs.
    run_anchor = family == "linear"

    out_dir = root / cell_dir_name(topology, family, regime, seed_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(topology, family, seed, out_dir)

    started = datetime.now(UTC)
    t0, c0 = time.perf_counter(), time.process_time()
    run = run_regime(
        cfg,
        regime,
        conditions=conditions,
        scm_builder=scm_builder,
        form_spec_fn=form_spec_fn,
        run_anchor=run_anchor,
        family=family,
    )
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    finished = datetime.now(UTC)

    for condition in conditions:
        sub = run.table[run.table["condition"] == condition]
        sub.to_csv(out_dir / f"scoring_table_{regime}_{condition}.csv", index=False)

    cell = CellKeys(topology=topology, family=family, regime=regime, seed=int(seed))
    by_group_path, by_cell_path = aggregate_run(out_dir, out_dir, cell)

    # The ONE h this (cell, seed) fit, persisted next to manifest.json. Keyed by
    # regime to match the estimation/aggregation slots below — a per-seed dir holds
    # exactly one regime, but the shape stays parallel so the three provenance
    # records read the same way.
    write_classifier_params(out_dir, {regime: classifier_provenance(run.classifier)})

    write_run_manifest(
        out_dir,
        cfg,
        estimation={regime: estimation_provenance(run.fitted)},
        aggregation={
            regime: {
                "code_version": AGGREGATION_CODE_VERSION,
                "by_group": str(by_group_path),
                "by_cell": str(by_cell_path),
                "pairwise": str(pairwise_path(out_dir, regime)),
            }
        },
        # n_runs materializes the full spawned seed list inline, so each per-seed
        # manifest carries the whole grid's seed provenance. ``n_seeds`` is the
        # grid size THIS run belongs to, threaded from the driver rather than
        # read off N_SEEDS_EARLY_LOOK. Only the manifest is affected: the seeds
        # — and so the scoring tables and aggregates — are identical either way.
        seeding={
            "meta_entropy": int(EXPERIMENT_META_ENTROPY),
            "this_run_seed": int(seed),
            "seed_index": int(seed_idx),
            "n_runs": int(n_seeds),
            "run_seeds": experiment_run_seeds(int(n_seeds)),
            "note": (
                f"PS-1: seed {seed_idx} of the N={int(n_seeds)} staged grid. "
                "The prefix property makes spawn(k) a prefix of spawn(20), so "
                "the first 6 are bit-identical to the N=6 early look."
            ),
        },
        # [PS-4; PS-7] Keyed by regime, exactly like the estimation and
        # aggregation slots. `None` on a three-rung run, and `write_run_manifest`
        # omits the key entirely then — so a three-rung manifest is byte-identical
        # to the sealed three-rung output and the block's PRESENCE is the four-rung marker.
        discovery=(
            {regime: run.discovery.as_dict()} if run.discovery is not None else None
        ),
    )

    spec = run.classifier.label_spec
    return {
        "topology": topology,
        "family": family,
        "regime": regime,
        "seed_idx": seed_idx,
        "seed": int(seed),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "wall_clock_s": round(wall, 3),
        "cpu_time_s": round(cpu, 3),
        # The suspend detector: a suspend inflates wall while CPU stays flat,
        # so a large ratio is the signature. Compute-bound cells sit near or
        # below 1 (numpy/BLAS threads can push CPU above wall).
        "wall_over_cpu": round(wall / cpu, 3) if cpu > 0 else float("nan"),
        "n_eligible": int(len(run.classifier.negative_pool_indices)),
        "eps_scale": float(spec.eps_scale),
        "test_accuracy": float(run.classifier.accuracy_overall),
    }


# --------------------------------------------------------------------------- #
# Cross-seed summaries (the inference stance format: mean / SD / min / max, no significance test)
# --------------------------------------------------------------------------- #


def _load_all(root: Path, kind: str, n_seeds: int) -> pd.DataFrame:
    """Concatenate every per-seed ``aggregate_{kind}_{regime}.csv`` under ``root``."""
    frames = []
    for topology, family, regime in grid_cells():
        for seed_idx in range(n_seeds):
            path = (
                root
                / cell_dir_name(topology, family, regime, seed_idx)
                / f"aggregate_{kind}_{regime}.csv"
            )
            if not path.exists():
                continue
            frame = pd.read_csv(path, float_precision="round_trip")
            frame["seed_idx"] = seed_idx
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no aggregate_{kind}_*.csv found under {root}")
    return pd.concat(frames, ignore_index=True)


def _summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Roll up across seeds: mean / sd / min / max per numeric column, per key.

    # [the inference stance] Effect-size-plus-dispersion, NOT significance testing. Every
    # numeric column is summarized (not a hand-picked subset) so the analysis pass never has to
    # come back to the per-seed CSVs for a column this file forgot to whitelist.
    # NaN-aware: a column that is NaN by design at some conditions (ΔB_g at L2,
    # Δ_dist off L0, the pairwise columns at L2 — all the NaN convention) summarizes
    # over the rows where it is defined, and N_seeds_observed records how many
    # that was, so a thin summary cell cannot masquerade as a full one.
    """
    numeric = [
        c
        for c in frame.columns
        if c not in {*keys, "seed", "seed_idx"}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    records = []
    for key_values, sub in frame.groupby(keys, sort=False):
        values_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        record = dict(zip(keys, values_tuple, strict=True))
        record["N_seeds"] = int(sub["seed_idx"].nunique())
        for col in numeric:
            values = sub[col].to_numpy(dtype=float)
            observed = values[np.isfinite(values)]
            record[f"{col}_mean"] = observed.mean() if observed.size else np.nan
            record[f"{col}_sd"] = (
                observed.std(ddof=_SD_DDOF) if observed.size > _SD_DDOF else np.nan
            )
            record[f"{col}_min"] = observed.min() if observed.size else np.nan
            record[f"{col}_max"] = observed.max() if observed.size else np.nan
            record[f"{col}_N_seeds_observed"] = int(observed.size)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _s_of_g_table(n_seeds: int) -> pd.DataFrame:
    """S(-1), S(+1), ΔS per (topology, family, regime), computed once per seed.

    # [PS-3] S(g) is FIT-INDEPENDENT and DATA-INDEPENDENT: it is a
    # property of the ground-truth SCM alone, and the NLG branch's only
    # randomness is the pinned construction-time `sg_mc` stream — NOT the
    # experiment run seed. So these values MUST be identical across all 6 seeds.
    # The loop recomputes per seed anyway, exactly so that invariant is verified
    # OPERATIONALLY rather than assumed: a value that moved with the seed would
    # mean the experiment seed had leaked into the descriptor.
    """
    records = []
    for topology, family, regime in grid_cells():
        cfg = load_config(_CONFIGS[(topology, family)])
        V_h, S = _feature_and_actionable_sets(cfg)
        scm_builder, _ = _RUNNERS[topology]._FAMILIES[family]
        scm = scm_builder(regime)
        for seed_idx, seed in enumerate(experiment_run_seeds(n_seeds)):
            s_neg = s_of_g(scm, -1.0, V_h=V_h, S=S)
            s_pos = s_of_g(scm, +1.0, V_h=V_h, S=S)
            records.append(
                {
                    "topology": topology,
                    "family": family,
                    "regime": regime,
                    "seed_idx": seed_idx,
                    "seed": int(seed),
                    "S_neg1": s_neg,
                    "S_pos1": s_pos,
                    # ΔS := S(−1) − S(+1), matching the SCM amendment's
                    # orientation (additive ΔS ≈ −0.003, EM ΔS ≈ −0.303).
                    "ΔS": s_neg - s_pos,
                }
            )
    return pd.DataFrame.from_records(records)


def _detectability_markdown(by_cell: pd.DataFrame, s_table: pd.DataFrame) -> str:
    """The analysis-pass hand-off record: Δ_cost(L2) per seed + S(g), NO gate evaluation.

    # [PS-2] The gate is: (i) sign(Δ_cost(L2)) consistent
    # across all 6 seeds AND |mean(Δ_cost(L2))| > per-seed SD; (ii) sign(ΔS)
    # unambiguous under the production S(g) in the effect-modifying regime.
    # This function EMITS the inputs to both conditions and states them; it does
    # NOT evaluate them, and no cell is passed, failed, flagged or retuned here.
    # The analysis pass owns the verdict (the grid run PRODUCES these numbers;
    # the analysis pass EVALUATES).
    """
    lines = [
        "# Detectability-gate inputs (for the analysis pass)",
        "",
        " PS-2 gate, stated for reference — **not evaluated here**:",
        "",
        "1. `sign(Δ_cost(L2))` consistent across all 6 seeds, AND",
        "   `|mean(Δ_cost(L2))|` across seeds `>` per-seed SD of `Δ_cost(L2)`.",
        "2. `sign(ΔS)` unambiguous under the production S(g), effect-modifying regime.",
        "",
        "The grid run produces these numbers; **the analysis pass evaluates the "
        "gate and writes the H1 verdict**.",
        "",
        "Pre-freeze the SCM amendment reference values (NLG collider, n_mc=10,000):",
        "additive `S(−1) ≈ 0.715, S(+1) ≈ 0.718, ΔS ≈ −0.003` at `|ΔS|/MC_SE ≈ 1.4`;",
        "effect-modifying `S(−1) ≈ 0.582, S(+1) ≈ 0.885, ΔS ≈ −0.303` at `|ΔS|/MC_SE ≈ 131`.",
        "",
        "---",
        "",
        "## Δ_cost(L2) across seeds",
        "",
        "| topology | family | regime | per-seed Δ_cost(L2) | mean | SD "
        "| all same sign? | \\|mean\\|/SD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    l2 = by_cell[by_cell["condition"] == "L2"]
    for topology, family, regime in grid_cells():
        sub = l2[
            (l2["topology"] == topology)
            & (l2["family"] == family)
            & (l2["regime"] == regime)
        ].sort_values("seed_idx")
        if sub.empty:
            lines.append(f"| {topology} | {family} | {regime} | (no runs) | | | | |")
            continue
        values = sub["Δ_cost"].to_numpy(dtype=float)
        mean, sd = float(values.mean()), float(values.std(ddof=_SD_DDOF))
        same_sign = bool(np.all(np.sign(values) == np.sign(values[0])))
        ratio = abs(mean) / sd if sd > 0 else float("inf")
        per_seed = ", ".join(f"{v:+.6f}" for v in values)
        lines.append(
            f"| {topology} | {family} | {regime} | {per_seed} | {mean:+.6f} | "
            f"{sd:.6f} | {'YES' if same_sign else 'NO'} | {ratio:.2f} |"
        )

    lines += [
        "",
        "## S(g) under the production descriptor (PS-3 fit-independence check)",
        "",
        "Recomputed once per seed. Under PS-3 S(g) depends only on the",
        "ground-truth SCM (the NLG branch's only randomness is the pinned `sg_mc`",
        "construction stream), so every value below must be **identical across all",
        "seeds** — a seed-varying value would mean the experiment seed leaked in.",
        "",
        "| topology | family | regime | S(−1) | S(+1) | ΔS | identical across seeds? |",
        "|---|---|---|---|---|---|---|",
    ]
    for topology, family, regime in grid_cells():
        sub = s_table[
            (s_table["topology"] == topology)
            & (s_table["family"] == family)
            & (s_table["regime"] == regime)
        ]
        identical = (
            sub["S_neg1"].nunique() == 1
            and sub["S_pos1"].nunique() == 1
        )
        row = sub.iloc[0]
        lines.append(
            f"| {topology} | {family} | {regime} | {row['S_neg1']:.6f} | "
            f"{row['S_pos1']:.6f} | {row['ΔS']:+.6f} | "
            f"{'YES' if identical else 'NO — INVESTIGATE'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_summaries(root: Path, n_seeds: int) -> None:
    """Write the four summary artifacts under ``root/summary/``."""
    summary_dir = root / SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)

    by_group = _load_all(root, "by_group", n_seeds)
    by_cell = _load_all(root, "by_cell", n_seeds)
    pairwise = _load_all(root, "pairwise", n_seeds)

    _summarize(
        by_group, ["topology", "family", "regime", "condition", "group"]
    ).to_csv(summary_dir / "cross_seed_by_group.csv", index=False)
    _summarize(by_cell, ["topology", "family", "regime", "condition"]).to_csv(
        summary_dir / "cross_seed_by_cell.csv", index=False
    )
    # The ΔB_g pairwise-population roll-up (the common-found population comparison set (b)) — a
    # distinct
    # population from the three-way one summarized in by_group, so it gets its
    # own summary rather than being folded in.
    _summarize(
        pairwise, ["topology", "family", "regime", "condition", "group"]
    ).to_csv(summary_dir / "cross_seed_pairwise.csv", index=False)

    # Per-seed long tables, kept alongside the roll-ups: the inference stance reports dispersion,
    # and the PS-2 gate reads the individual seed values, so the raw 6 must
    # survive into the summary directory rather than only their moments.
    by_cell.to_csv(summary_dir / "per_seed_by_cell.csv", index=False)
    by_group.to_csv(summary_dir / "per_seed_by_group.csv", index=False)

    s_table = _s_of_g_table(n_seeds)
    s_table.to_csv(summary_dir / "s_of_g_by_seed.csv", index=False)

    text = _detectability_markdown(by_cell, s_table)
    (summary_dir / "detectability_inputs.md").write_text(text, encoding="utf-8")
    # ASCII-safe console echo: this driver's stdout is a cp1252 Windows console,
    # which cannot encode Δ or − and would raise UnicodeEncodeError AFTER the
    # artifacts are written — the same failure mode the single-seed drivers guard.
    print(text.encode("ascii", "replace").decode("ascii"))
    print(f"wrote summaries -> {summary_dir}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _parse_seed_indices(spec: str | None, n_seeds: int) -> list[int]:
    """Parse ``--seed-indices`` ('0-5', '6-19', '0,3,7') into sorted indices.

    Every index must lie inside the declared grid: a subset that reached past
    ``n_seeds`` would be drawing from a spawn the manifest does not record, which
    is exactly the provenance break the seed-generation mechanics exists to prevent.
    """
    if spec is None:
        return list(range(n_seeds))
    indices: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(part) for part in chunk.split("-", 1))
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(chunk))
    out_of_range = sorted(i for i in indices if not 0 <= i < n_seeds)
    if out_of_range:
        raise ValueError(
            f"seed indices {out_of_range} fall outside the declared grid "
            f"0..{n_seeds - 1}; raise --n-seeds or fix --seed-indices."
        )
    if not indices:
        raise ValueError(f"--seed-indices {spec!r} selected nothing.")
    return sorted(indices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/cross_seed_N6")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS_EARLY_LOOK,
        help="PS-1 pins this at 6 for the early look; 20 for the full grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run ONE cell x ONE seed (the pre-launch well-formedness gate).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="re-derive the summary artifacts from per-seed CSVs already on disk.",
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help=(
            "run the cells but do NOT write summary artifacts. For staged batches: "
            "a roll-up over a half-materialized grid silently averages the cells "
            "that happen to exist, which reads identically to a complete one. "
            "Re-derive with --summary-only once every cell has landed."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "skip per-seed runs whose aggregate CSVs already exist. OFF by "
            "default: a full grid launch must be a clean run, not an "
            "accidental resume. Use only to continue a deliberately "
            "interrupted grid."
        ),
    )
    parser.add_argument(
        "--l1-discovered",
        action="store_true",
        help=(
            "score the FOURTH rung, L1-discovered, alongside L0/L1-oracle/L2 "
            f"(condition set {L1D_CONDITIONS}). [PS-4] Its outputs belong in "
            "a SEPARATE TREE: pass --root results/cross_seed_L1d_N20 as well. Off "
            "by default, so an ordinary grid launch is bit-identical to the "
            "three-rung grid that produced results/cross_seed_N20/."
        ),
    )
    parser.add_argument("--topology", choices=TOPOLOGIES)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument(
        "--seed-indices",
        help=(
            "comma-separated subset of seed indices to run, e.g. '0-5' or '6-19' "
            "or '0,3,7'. Runs the whole 0..n_seeds-1 range when omitted. --n-seeds "
            "still declares which GRID these runs belong to (it sets the manifest's "
            "the seed-generation provenance and the spawn the seeds are drawn from), so staging a "
            "20-seed grid in batches keeps every manifest honest about N=20. Exists "
            "so a gate checkpoint can be evaluated on the first 6 seeds of the N=20 "
            "grid BEFORE the remaining seeds are consumed (PS-1's staged discipline)."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summaries(root, args.n_seeds)
        return 0

    # [the seed-generation mechanics] ONE spawn(N), shared across every cell.
    seeds = experiment_run_seeds(args.n_seeds)
    print(f"=== seed spawn: SeedSequence({EXPERIMENT_META_ENTROPY}).spawn({args.n_seeds}) ===")
    for i, seed in enumerate(seeds):
        print(f"  seed_idx {i}: {seed}")
    print()
    if args.l1_discovered:
        # [PS-4] Loud, because this flag changes WHICH TREE the run belongs
        # to. Writing a four-rung cell into cross_seed_N20/ would break the sealed
        # tree's three-rung denominators; the log is the first place that shows up.
        print(f"=== PS-4 FOUR-RUNG GRID: conditions = {L1D_CONDITIONS} ===")
        print(f"    root = {root}  (must NOT be the sealed three-rung tree)\n")

    cells = [
        (t, f, r)
        for (t, f, r) in grid_cells()
        if (args.topology is None or t == args.topology)
        and (args.family is None or f == args.family)
        and (args.regime is None or r == args.regime)
    ]
    seed_indices = _parse_seed_indices(args.seed_indices, args.n_seeds)
    if args.dry_run:
        cells, seed_indices = cells[:1], [0]
        print(f"DRY RUN: one cell only -> {cells[0]}, seed_idx 0\n")

    records: list[dict] = []
    log_path = root / SUMMARY_DIR / "wall_clock_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()

    for topology, family, regime in cells:
        for seed_idx in seed_indices:
            label = cell_dir_name(topology, family, regime, seed_idx)
            done = (root / label / f"aggregate_by_cell_{regime}.csv").exists()
            if args.skip_completed and done:
                print(f"[skip] {label} (already complete)")
                continue
            print(f"[run ] {label} (seed={seeds[seed_idx]}) ...", flush=True)
            try:
                record = run_cell_seed(
                    topology,
                    family,
                    regime,
                    seed_idx,
                    seeds[seed_idx],
                    root,
                    n_seeds=args.n_seeds,
                    conditions=L1D_CONDITIONS if args.l1_discovered else None,
                )
            except Exception:
                # A failing cell IS the finding. Stop the grid,
                # surface the traceback, and do NOT patch-and-retry silently —
                # The operator decides whether to fix or drop the cell. The partial
                # wall-clock log is flushed first so the runs that did complete
                # are not lost with the failure.
                pd.DataFrame.from_records(records).to_csv(log_path, index=False)
                print(f"\n=== GRID FAILED at cell {label} ===", file=sys.stderr)
                traceback.print_exc()
                print(
                    f"\nPartial wall-clock log written -> {log_path}\n"
                    "No summary artifacts written. Grid ABORTED ("
                    "a failing cell is reported, never silently retried).",
                    file=sys.stderr,
                )
                return 1
            records.append(record)
            pd.DataFrame.from_records(records).to_csv(log_path, index=False)
            print(
                f"       done in {record['wall_clock_s']:.1f}s "
                f"(cpu {record['cpu_time_s']:.1f}s, wall/cpu {record['wall_over_cpu']}) "
                f"n_eligible={record['n_eligible']}",
                flush=True,
            )

    total = time.perf_counter() - total_t0
    print(f"\n=== grid complete: {len(records)} per-seed runs in {total / 3600:.2f} h ===")
    print(f"wall-clock log -> {log_path}")

    if args.dry_run:
        print("DRY RUN: summaries skipped (an incomplete grid has no cross-seed roll-up).")
        return 0
    if args.skip_summaries:
        print(
            "--skip-summaries: no roll-up written. Re-run with --summary-only once "
            "every cell of the grid has landed."
        )
        return 0

    write_summaries(root, args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
