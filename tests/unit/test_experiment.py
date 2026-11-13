"""Experiment logging convention (docs/04 section 8, task P1-D4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from driverisk.utils import experiment as X
from driverisk.utils.paths import repo_root

NOW = datetime(2026, 11, 14, 9, 30, 5, tzinfo=UTC)
CFG = {"model": {"d_model": 32}, "train": {"lr": 0.001, "epochs": 3}}


def test_run_id_has_time_name_seed_and_commit() -> None:
    assert X.make_run_id("toy fusion", 2, "abc1234", NOW) == "20261114-093005_toy-fusion_s2_abc1234"


def test_git_commit_in_this_repo_and_outside_one(tmp_path: Path) -> None:
    commit = X.git_commit(repo_root())
    assert commit.removesuffix("+dirty") != "nogit" and len(commit.removesuffix("+dirty")) >= 7
    assert X.git_commit(tmp_path) == "nogit"


def test_file_sha256_is_stable_and_content_based(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    assert X.file_sha256(a) == X.file_sha256(b) and len(X.file_sha256(a)) == 64
    b.write_text("[]", encoding="utf-8")
    assert X.file_sha256(a) != X.file_sha256(b)


def test_dummy_run_leaves_the_whole_reproducibility_record(tmp_path: Path) -> None:
    split = tmp_path / "splits.json"
    split.write_text('{"train": ["d01"]}', encoding="utf-8")
    with X.start_run(
        "dummy", CFG, seed=1, root=tmp_path / "exp", setting="stub", split_file=split, now=NOW
    ) as run:
        run.log_metrics({"macro_f1": 0.5})
        run.log_metrics({"fa_per_hour": 4.0, "median_lead_s": 1.5})
        folder = run.dir
    assert folder.parent == tmp_path / "exp" / "runs"
    assert yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) == CFG
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"macro_f1": 0.5, "fa_per_hour": 4.0, "median_lead_s": 1.5}
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok" and meta["seed"] == 1 and meta["setting"] == "stub"
    assert meta["split_sha256"] == X.file_sha256(split)
    assert {"commit", "python", "torch", "numpy", "platform", "device", "started_at"} <= set(meta)
    assert meta["duration_s"] >= 0

    table = pd.read_csv(tmp_path / "exp" / "results.csv")
    assert list(table.columns) == list(X.RESULT_COLUMNS)
    row = table.iloc[0]
    assert row["run_id"] == folder.name and row["macro_f1"] == 0.5 and row["status"] == "ok"
    assert row["fa_per_hour"] == 4.0 and row["median_lead_s"] == 1.5


def test_results_csv_accumulates_rows_and_keeps_one_header(tmp_path: Path) -> None:
    for seed in (0, 1, 2):
        with X.start_run("m", CFG, seed=seed, root=tmp_path, now=NOW) as run:
            run.log_metrics({"macro_f1": 0.1 * seed})
    table = pd.read_csv(tmp_path / "results.csv")
    assert table["seed"].tolist() == [0, 1, 2]
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").count("run_id") == 1


def test_failed_run_is_recorded_and_the_error_propagates(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), X.start_run("m", CFG, 0, root=tmp_path) as run:
        run.log_metrics({"macro_f1": 0.2})
        raise RuntimeError("boom")
    meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and "boom" in meta["error"]
    assert pd.read_csv(tmp_path / "results.csv").iloc[0]["status"] == "failed"


def test_unknown_metric_keys_go_to_metrics_json_only(tmp_path: Path) -> None:
    with X.start_run("m", CFG, 0, root=tmp_path) as run:
        run.log_metrics({"custom_thing": 7})
    assert json.loads((run.dir / "metrics.json").read_text(encoding="utf-8")) == {"custom_thing": 7}
    assert list(pd.read_csv(tmp_path / "results.csv").columns) == list(X.RESULT_COLUMNS)


def test_two_runs_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path) -> None:
    a = X.start_run("m", CFG, 0, root=tmp_path, now=NOW)
    b = X.start_run("m", CFG, 0, root=tmp_path, now=NOW)
    assert a.dir != b.dir and a.dir.is_dir() and b.dir.is_dir()
    a.finish()
    b.finish()


def test_numpy_values_are_json_safe(tmp_path: Path) -> None:
    import numpy as np

    with X.start_run("m", CFG, 0, root=tmp_path) as run:
        run.log_metrics({"macro_f1": np.float32(0.25), "lead": np.array([1.0, 2.0])})
    saved = json.loads((run.dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["macro_f1"] == 0.25 and saved["lead"] == [1.0, 2.0]
