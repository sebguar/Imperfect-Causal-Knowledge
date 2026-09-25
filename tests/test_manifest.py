"""Tests for icknowledge.utils.manifest."""

import json

from icknowledge.utils.config import load_config
from icknowledge.utils.manifest import write_run_manifest


def test_write_run_manifest_produces_valid_json(tmp_path):
    cfg = load_config("configs/example.yaml")
    manifest_path = write_run_manifest(tmp_path / "run", cfg)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    for key in (
        "timestamp",
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "package_versions",
        "config",
    ):
        assert key in manifest

    assert isinstance(manifest["package_versions"]["numpy"], str)
    assert manifest["package_versions"]["numpy"] != ""
