"""The stub pipeline end to end (the P0 phase gate)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from driverisk.utils.paths import repo_root


def test_demo_stub_runs_end_to_end(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "demo_stub", repo_root() / "scripts" / "demo_stub.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_demo(tmp_path)
    assert result["final_loss"] < result["initial_loss"]
    assert result["n_train_windows"] > 0 and result["n_test_windows"] > 0
    assert result["alert_type"] in {"aggressive", "drowsy", "distracted"}
    assert 0.0 <= result["alert_risk"] <= 1.0
