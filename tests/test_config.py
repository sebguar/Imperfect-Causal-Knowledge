"""Tests for icknowledge.utils.config."""

import pytest

from icknowledge.utils.config import REQUIRED_TOP_LEVEL_KEYS, load_config, require_keys


def test_load_config_example():
    cfg = load_config("configs/example.yaml")
    assert cfg.experiment_name == "example"


def test_require_keys_passes_for_example_config():
    cfg = load_config("configs/example.yaml")
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS)


def test_require_keys_raises_for_missing_key():
    cfg = load_config("configs/example.yaml")
    with pytest.raises(KeyError, match="nonexistent_key"):
        require_keys(cfg, ["seed", "nonexistent_key"])
