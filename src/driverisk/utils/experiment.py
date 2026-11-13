"""Experiment logging: one folder per run plus one row in `experiments/results.csv`.

Convention (docs/04 section 8, CONTRIBUTING section 7):

    experiments/results.csv                 tracked in git, one row per run
    experiments/runs/<run_id>/              git-ignored, large or bulky files
        config.yaml                         the resolved config this run used
        meta.json                           commit, seed, versions, device, split hash, status, time
        metrics.json                        every logged metric

`<run_id>` is `<yyyymmdd-HHMMSS>_<name>_s<seed>_<commit>`. A run started inside a dirty
working tree gets `+dirty` after the commit, so a number can be traced to its code.

Usage::

    with start_run("toy-fusion", cfg, seed=0, setting="stub") as run:
        ...
        run.log_metrics({"macro_f1": 0.71})

A run that raises is recorded as `failed` (with the error text) and the exception propagates.
Writes are not safe for several processes appending at once; run one experiment per process
and let results.csv be written by whoever finishes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from driverisk.utils.paths import experiments_dir

# Columns of results.csv. Everything else a run logs lives only in its metrics.json.
RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "started_at",
    "name",
    "seed",
    "commit",
    "setting",
    "status",
    "macro_f1",
    "fa_per_hour",
    "median_lead_s",
    "split_sha256",
    "duration_s",
)


def git_commit(repo: Path) -> str:
    """Short commit hash of `repo`, with `+dirty` if there are uncommitted changes, or `nogit`."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "--short=7", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (subprocess.SubprocessError, OSError):
        return "nogit"
    return commit + ("+dirty" if dirty else "")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(name: str, seed: int, commit: str, now: datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "run"
    return f"{now.strftime('%Y%m%d-%H%M%S')}_{slug}_s{seed}_{commit}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")


def _device() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


@dataclass
class Run:
    run_id: str
    dir: Path
    results_csv: Path
    meta: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.monotonic, repr=False)
    _done: bool = field(default=False, repr=False)

    def log_metrics(self, values: dict[str, Any]) -> None:
        """Merge `values` into metrics.json (later calls overwrite earlier keys)."""
        self.metrics.update(values)
        _write_json(self.dir / "metrics.json", self.metrics)

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        if self._done:
            return
        self._done = True
        self.meta["status"] = status
        self.meta["duration_s"] = round(time.monotonic() - self._t0, 3)
        if error is not None:
            self.meta["error"] = error
        _write_json(self.dir / "meta.json", self.meta)
        _write_json(self.dir / "metrics.json", self.metrics)
        row = {c: self.meta.get(c, "") for c in RESULT_COLUMNS}
        row["run_id"] = self.run_id
        for key in ("macro_f1", "fa_per_hour", "median_lead_s"):
            row[key] = self.metrics.get(key, "")
        new_file = not self.results_csv.exists()
        with self.results_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.finish("ok")
        else:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")


def start_run(
    name: str,
    config: dict[str, Any],
    seed: int,
    *,
    root: Path | None = None,
    setting: str = "",
    split_file: str | Path | None = None,
    now: datetime | None = None,
) -> Run:
    """Create the run folder, write config.yaml and a provisional meta.json, return the Run."""
    root = Path(root) if root is not None else experiments_dir()
    now = now or datetime.now(UTC)
    commit = git_commit(Path(__file__).resolve().parent)
    run_id = make_run_id(name, seed, commit, now)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    folder, n = runs / run_id, 1
    while True:  # two runs in the same second must not share a folder
        try:
            folder.mkdir()
            break
        except FileExistsError:
            n += 1
            folder = runs / f"{run_id}-{n}"
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    meta: dict[str, Any] = {
        "run_id": folder.name,
        "name": name,
        "seed": seed,
        "setting": setting,
        "commit": commit,
        "started_at": now.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": _device(),
        "split_sha256": file_sha256(split_file) if split_file is not None else "",
        "status": "running",
    }
    _write_json(folder / "meta.json", meta)
    return Run(folder.name, folder, root / "results.csv", meta)
