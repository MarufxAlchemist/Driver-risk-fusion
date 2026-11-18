"""Shared fixtures: one stub dataset per test session (contract section 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from driverisk.data.stubs import make_stub_dataset
from driverisk.utils.config import load_config


@pytest.fixture(scope="session")
def stub_cfg() -> dict:
    return load_config("stub")


@pytest.fixture(scope="session")
def stub_root(tmp_path_factory: pytest.TempPathFactory, stub_cfg: dict) -> Path:
    root = tmp_path_factory.mktemp("stub_processed")
    make_stub_dataset(root, stub_cfg)
    return root


@pytest.fixture(scope="session")
def stub_dirs(stub_root: Path) -> list[Path]:
    return [stub_root / d for d in ("uah", "dmd", "own")]
