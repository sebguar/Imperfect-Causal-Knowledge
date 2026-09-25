"""Aggregation entry point: per-individual CSVs -> two aggregate CSVs.

Reads the per-individual scoring CSVs for one cell across all conditions found
on disk and produces aggregate_by_group_{regime}.csv and
aggregate_by_cell_{regime}.csv. Idempotent; overwrites outputs. Does NOT run
recourse — it only consumes what's already on disk; the per-individual CSVs
remain the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icknowledge.aggregation.by_cell import build_by_cell_frame
from icknowledge.aggregation.by_group import build_by_group_frame, build_pairwise_frame
from icknowledge.aggregation.common_found import COMMON_FOUND_MIN_GROUP_WARN
from icknowledge.aggregation.schema import (
    CANONICAL_ARITY,
    CellKeys,
    common_found_suffix,
)

#: Version identifier recorded in the run manifest's ``aggregation`` slot — bump
#: on any semantic change to the aggregate schemas or formulas.
AGGREGATION_CODE_VERSION = (
    "aggregation-v3 (PS-4 note (1): common-found denominator "
    "suffix derived from the scored-condition arity via the pinned map "
    "{3: threeway, 4: fourway}; supersedes aggregation-v2 (the common-found "
    "population triple + pairwise artifact) and aggregation-v1 "
    "(PS-6, PS-3 extension). ARITY-3 IDENTITY: at three scored "
    "conditions every v2 column keeps its name, position and value byte-for-byte, "
    "so every sealed three-rung artifact reproduces unchanged; only a four-rung "
    "run emits the _fourway names.)"
)

#: Canonical condition order for aggregate rows (the L-ladder, top down).
_CONDITION_ORDER = ("L0", "L1-oracle", "L1-discovered", "L2")


def _load_condition_tables(per_individual_dir: Path, regime: str) -> dict[str, pd.DataFrame]:
    """Load scoring_table_{regime}_{condition}.csv for every condition on disk."""
    prefix = f"scoring_table_{regime}_"
    found: dict[str, Path] = {
        p.stem.removeprefix(prefix): p
        for p in sorted(per_individual_dir.glob(f"{prefix}*.csv"))
    }
    if not found:
        raise FileNotFoundError(
            f"no per-individual CSVs matching {prefix}*.csv in {per_individual_dir}."
        )
    unknown = sorted(set(found) - set(_CONDITION_ORDER))
    if unknown:
        raise ValueError(
            f"unrecognized condition file suffix(es) {unknown} in {per_individual_dir}."
        )
    tables: dict[str, pd.DataFrame] = {}
    for condition in _CONDITION_ORDER:  # canonical ladder order for output rows
        if condition not in found:
            continue
        # float_precision="round_trip": the default C-parser mode can be 1 ulp off
        # on the last digit; the acceptance gate (test 1) recomputes means to
        # atol=1e-10, so parse exactly what scoring wrote.
        table = pd.read_csv(found[condition], float_precision="round_trip")
        table["acted_set"] = table["acted_set"].fillna("")
        tables[condition] = table
    return tables


def pairwise_path(output_dir: Path, regime: str) -> Path:
    """Path of the pairwise-vs-L2 artifact (the common-found population).

    Exposed as a helper rather than returned from `aggregate_run` so the
    existing two-value unpacking in both run drivers and the aggregation test fixture
    keeps working untouched — an ADDITIVE aggregation change must not force
    churn on its callers.
    """
    return Path(output_dir) / f"aggregate_pairwise_{regime}.csv"


def thin_cell_warnings(
    by_cell: pd.DataFrame,
    cell_keys: CellKeys,
    n_conditions: int = CANONICAL_ARITY,
) -> list[str]:
    """Warning lines for cells whose common-found primary population went thin.

    # [the common-found population] REPORT ONLY — nothing branches on this. The
    # primary population keys on `found` rather than `valid`, so it survives a
    # validity collapse entirely — but if `found` itself thins in either group
    # the H1 primary read is running on a small sample and the analysis layer
    # needs to see it. Deciding what to DO about a thin cell is an
    # analysis-layer call, so this function returns strings and never drops,
    # reweights or flags a row.
    """
    # [PS-4 note (1)] Read the column under its ARITY-CORRECT name — the same
    # name `build_by_cell_frame` wrote — so the warning quotes the denominator the
    # artifact actually carries rather than a hardcoded "threeway".
    column = f"N_common_found_{common_found_suffix(n_conditions)}_min_group"
    lines: list[str] = []
    for _, row in by_cell.iterrows():
        n_min = int(row[column])
        if n_min < COMMON_FOUND_MIN_GROUP_WARN:
            lines.append(
                f"WARNING [the common-found population thin primary population, NOT a gate]: "
                f"{cell_keys.topology}/{cell_keys.family}/{cell_keys.regime}/"
                f"seed={cell_keys.seed} condition={row['condition']}: "
                f"{column} = {n_min} "
                f"(< {COMMON_FOUND_MIN_GROUP_WARN}). The analysis layer decides how to handle this."
            )
    return lines


def aggregate_run(
    per_individual_dir: Path,
    output_dir: Path,
    cell_keys: CellKeys,
) -> tuple[Path, Path]:
    """Aggregate one cell's per-individual CSVs into the aggregate CSVs.

    Returns ``(by_group_path, by_cell_path)``. File naming follows the
    per-individual convention (suffix = regime): aggregate_by_group_{regime}.csv
    and aggregate_by_cell_{regime}.csv under ``output_dir``. The common-found layer additionally
    writes aggregate_pairwise_{regime}.csv (the common-found population); its path comes
    from `pairwise_path`, so this function's return shape is unchanged.
    """
    per_individual_dir = Path(per_individual_dir)
    output_dir = Path(output_dir)

    tables = _load_condition_tables(per_individual_dir, cell_keys.regime)
    by_group = build_by_group_frame(tables, cell_keys)
    by_cell = build_by_cell_frame(tables, by_group, cell_keys)
    pairwise = build_pairwise_frame(tables, cell_keys)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_group_path = output_dir / f"aggregate_by_group_{cell_keys.regime}.csv"
    by_cell_path = output_dir / f"aggregate_by_cell_{cell_keys.regime}.csv"
    by_group.to_csv(by_group_path, index=False)
    by_cell.to_csv(by_cell_path, index=False)
    pairwise.to_csv(pairwise_path(output_dir, cell_keys.regime), index=False)

    for line in thin_cell_warnings(by_cell, cell_keys, len(tables)):
        print(line)
    return by_group_path, by_cell_path
