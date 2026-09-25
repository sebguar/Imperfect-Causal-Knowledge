"""Analysis layer: reads the cross-seed grid, runs no experiments.

Everything here is a READER over the ``results/cross_seed_*/`` trees — the per-seed
aggregate CSVs, the per-individual scoring CSVs, and the run manifests that the grid run
already wrote. No module in this package trains a classifier, runs recourse, or
redraws a seed; the only recomputation permitted is the deterministic
ground-truth SCM coefficient lookup (builder signature defaults), which
carries no randomness at all.

Contents:
  loading    — grid enumeration, artifact loaders, true/estimated coefficient map
  t4_reports — H1 monotonicity, ValidityDisp, the detectability gate
  mechanism  — the PS-9 EXPLORATORY mechanism-discrimination diagnostic
"""

from __future__ import annotations

__all__ = ["loading", "mechanism", "t4_reports"]
