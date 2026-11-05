"""Repository paths. Never hard-code absolute paths or user names (CONTRIBUTING section 5)."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "DRIVERISK_DATA_DIR"
_MARKER = "pyproject.toml"


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) to the folder holding pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"No {_MARKER} found above {here}")


def data_dir() -> Path:
    """Data root: `$DRIVERISK_DATA_DIR` if set, else `<repo>/data`."""
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env).expanduser().resolve() if env else repo_root() / "data"


def processed_dir(dataset: str) -> Path:
    """`data/processed/<dataset>/` for dataset in {uah, dmd, own}."""
    return data_dir() / "processed" / dataset


def configs_dir() -> Path:
    return repo_root() / "configs"


def experiments_dir() -> Path:
    return repo_root() / "experiments"
