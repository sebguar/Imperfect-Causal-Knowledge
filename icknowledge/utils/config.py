"""Config loading (YAML) helpers built on OmegaConf.
require_keys() validates all required keys at once.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REQUIRED_TOP_LEVEL_KEYS = ["seed", "experiment_name", "output_dir"]


def load_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping at the top level of {path}, got {type(cfg).__name__}")
    OmegaConf.resolve(cfg)
    return cfg


def require_keys(cfg: DictConfig, keys: list[str]) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")
