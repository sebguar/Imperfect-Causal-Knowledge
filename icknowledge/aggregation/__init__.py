"""Aggregation layer: per-individual -> aggregate CSVs.

Rolls the per-individual scoring CSVs up into aggregate_by_group.csv (per
cell × condition × group) and aggregate_by_cell.csv (per cell × condition).
This is the layer H1/H2/H4 read from; the per-individual CSVs remain the source
of truth. Semantics are governed by PS-6 and PS-3.
"""

from icknowledge.aggregation.by_cell import build_by_cell_frame
from icknowledge.aggregation.by_group import build_by_group_frame, build_pairwise_frame
from icknowledge.aggregation.common_found import (
    COMMON_FOUND_MIN_GROUP_WARN,
    CommonFoundPopulations,
    common_found_populations,
)
from icknowledge.aggregation.pipeline import (
    AGGREGATION_CODE_VERSION,
    aggregate_run,
    pairwise_path,
    thin_cell_warnings,
)
from icknowledge.aggregation.schema import (
    BY_CELL_COLUMNS,
    BY_CELL_INT_COLUMNS,
    BY_GROUP_COLUMNS,
    BY_GROUP_INT_COLUMNS,
    CANONICAL_ARITY,
    PAIRWISE_COLUMNS,
    PAIRWISE_INT_COLUMNS,
    ByCellRow,
    ByGroupRow,
    CellKeys,
    PairwiseRow,
    by_cell_columns,
    by_cell_int_columns,
    by_group_columns,
    by_group_int_columns,
    common_found_suffix,
)

__all__ = [
    "AGGREGATION_CODE_VERSION",
    "BY_CELL_COLUMNS",
    "BY_CELL_INT_COLUMNS",
    "BY_GROUP_COLUMNS",
    "BY_GROUP_INT_COLUMNS",
    "CANONICAL_ARITY",
    "COMMON_FOUND_MIN_GROUP_WARN",
    "PAIRWISE_COLUMNS",
    "PAIRWISE_INT_COLUMNS",
    "ByCellRow",
    "ByGroupRow",
    "CellKeys",
    "CommonFoundPopulations",
    "PairwiseRow",
    "aggregate_run",
    "build_by_cell_frame",
    "build_by_group_frame",
    "build_pairwise_frame",
    "by_cell_columns",
    "by_cell_int_columns",
    "by_group_columns",
    "by_group_int_columns",
    "common_found_populations",
    "common_found_suffix",
    "pairwise_path",
    "thin_cell_warnings",
]
