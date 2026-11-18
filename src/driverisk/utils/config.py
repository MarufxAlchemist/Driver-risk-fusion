"""YAML config loading. Hyperparameters and thresholds live in `configs/`, not in code."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from driverisk.utils.paths import configs_dir


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `override` merged in recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping. An empty file gives an empty dict."""
    with Path(path).open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_config(
    name_or_path: str | Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load a config by file path, or by bare name from `configs/` (".yaml" optional).

    Example: `load_config("toy_fusion", {"train": {"epochs": 2}})`.
    """
    path = Path(name_or_path)
    if not path.is_file():
        stem = path.name if path.suffix else f"{path.name}.yaml"
        path = configs_dir() / stem
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {name_or_path}")
    cfg = load_yaml(path)
    return deep_merge(cfg, overrides) if overrides else cfg
