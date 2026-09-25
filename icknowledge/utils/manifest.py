"""Run manifest writer to document reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# [PS-7] causal-learn joins the tracked set: it is the causal
# DISCOVERY library, so its version is a first-class input to every L1-discovered
# number, exactly as scikit-learn's is to h. Recorded here IN ADDITION to
# `DiscoveryRecord.library_version` (which reads the same distribution metadata):
# the record rides only on a four-rung run, while this list is written on EVERY
# run, so a three-rung manifest still documents which discovery build was
# installed when the tree was produced. The frozen three-rung trees predate
# causal-learn's addition to this list and do not record the discovery build.
_TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "networkx",
    "omegaconf",
    "causal-learn",
]


def _git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def write_run_manifest(
    output_dir: str | Path,
    config: DictConfig,
    estimation: dict | None = None,
    aggregation: dict | None = None,
    seeding: dict | None = None,
    discovery: dict | None = None,
) -> Path:
    """Write the reproducibility manifest; optionally with an equation-estimation record.

    ``estimation`` — provenance from `estimation.base.estimation_provenance` (or
        regime -> dict): estimator id, serialized FormSpec, fitted coefficients,
        dataset identity, regime, family.
    ``aggregation`` — the aggregation code version and the two aggregate-CSV paths.
    ``seeding`` — from `utils.seeding.seeding_provenance`: the pinned grid
        meta-entropy and this run's master seed, plus the spawned per-run seeds.
        Recording the meta-entropy in every run's manifest is what makes the whole
        replication grid reproducible from one audited integer.
    ``discovery`` (PS-7, PS-4) — the `DiscoveryRecord.as_dict()` payload per regime.

    The presence of ``discovery`` is the four-rung marker: a three-rung run never
    searches, so ``results/cross_seed_L1d_N20/`` manifests carry the key and
    ``results/cross_seed_N20/`` manifests do not. Key on it, not the path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    if estimation is not None:
        manifest["estimation"] = estimation
    if aggregation is not None:
        manifest["aggregation"] = aggregation
    if seeding is not None:
        manifest["seeding"] = seeding
    if discovery is not None:
        manifest["discovery"] = discovery

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


#: Fixed filename so an auditor finds the fitted h next to manifest.json without
#: consulting a path recorded elsewhere. JSON (not .npz) mirrors the manifest and
#: the estimation-provenance record — the repo has no binary-array convention, and
#: an LR's (coef_, intercept_) is small enough that text is the auditable choice.
CLASSIFIER_PARAMS_FILENAME = "classifier_params.json"


def write_classifier_params(output_dir: str | Path, classifier: dict) -> Path:
    """Persist the fitted-classifier parameters beside ``manifest.json``.

    ``classifier`` is the record from
    `icknowledge.classifier.training.classifier_provenance` (or, for multi-regime
    runs, a mapping of regime -> that record). Persisting it makes the ONE
    classifier fit per (cell, seed) a durable artifact rather than an in-memory
    object: h is held constant across L0 / L1-oracle / L2 (only the
    causal model varies), and this file is what lets an auditor confirm that after
    the run, without refitting.

    Floats go through ``json.dumps`` unrounded, so the round-trip is exact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CLASSIFIER_PARAMS_FILENAME
    path.write_text(json.dumps(classifier, indent=2))
    return path
