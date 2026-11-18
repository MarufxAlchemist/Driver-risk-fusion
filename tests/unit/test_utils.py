from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from driverisk.utils.config import deep_merge, load_config, load_yaml
from driverisk.utils.paths import DATA_DIR_ENV, data_dir, repo_root
from driverisk.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(3)
    a = (np.random.rand(3), torch.rand(3))  # noqa: NPY002 - tests the global RNG
    set_seed(3)
    b = (np.random.rand(3), torch.rand(3))  # noqa: NPY002
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_set_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_repo_root_has_pyproject() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_data_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path.resolve()
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir() == repo_root() / "data"


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_load_config_by_name_and_override() -> None:
    cfg = load_config("stub", {"seed": 42})
    assert cfg["seed"] == 42 and "datasets" in cfg


def test_load_config_missing() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(p)
