"""Contract validator: stubs pass, and each rule in contract section 10 catches its violation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data.validate import (
    validate_events,
    validate_manifest,
    validate_processed_dir,
    validate_trip_frame,
)


@pytest.mark.parametrize("dataset", ["uah", "dmd", "own"])
def test_stub_datasets_pass_validator(stub_root: Path, dataset: str) -> None:
    report = validate_processed_dir(stub_root / dataset)
    assert report.ok, report.errors
    assert report.n_trips > 0


@pytest.fixture
def paired_trip(stub_root: Path) -> tuple[pd.DataFrame, dict]:
    root = stub_root / "own"
    tid = pd.read_csv(root / "manifest.csv")["trip_id"].iloc[0]
    meta = json.loads((root / f"{tid}_meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(root / f"{tid}.parquet"), meta


def test_paired_trip_is_valid(paired_trip: tuple[pd.DataFrame, dict]) -> None:
    df, meta = paired_trip
    assert validate_trip_frame(df, meta) == []


def _errors_after(paired_trip, mutate) -> list[str]:
    df, meta = paired_trip
    df = df.copy()
    meta = json.loads(json.dumps(meta))
    result = mutate(df, meta)
    return validate_trip_frame(df if result is None else result, meta)


def test_missing_column(paired_trip) -> None:
    errs = _errors_after(paired_trip, lambda df, m: df.drop(columns=["v_mar"]))
    assert any("missing columns" in e for e in errs)


def test_wrong_dtype(paired_trip) -> None:
    def mutate(df, m):
        df["m_speed"] = df["m_speed"].astype("float64")

    assert any("dtype" in e for e in _errors_after(paired_trip, mutate))


def test_time_gap(paired_trip) -> None:
    def mutate(df, m):
        df.loc[10:, "t"] = df.loc[10:, "t"] + np.float32(0.5)

    assert any("t step" in e for e in _errors_after(paired_trip, mutate))


def test_out_of_range(paired_trip) -> None:
    def mutate(df, m):
        df.loc[5, "v_ear_mean"] = np.float32(1.5)

    assert any("outside" in e and "v_ear_mean" in e for e in _errors_after(paired_trip, mutate))


def test_undeclared_all_nan(paired_trip) -> None:
    def mutate(df, m):
        df["v_mar"] = np.float32(np.nan)

    assert any("all NaN" in e for e in _errors_after(paired_trip, mutate))


def test_declared_all_nan_is_allowed(paired_trip) -> None:
    def mutate(df, m):
        df["v_mar"] = np.float32(np.nan)
        m[S.META_ALL_NAN_KEY] = [*m[S.META_ALL_NAN_KEY], "v_mar"]

    assert _errors_after(paired_trip, mutate) == []


def test_bad_label(paired_trip) -> None:
    def mutate(df, m):
        df.loc[0, "y_cls"] = np.int8(7)

    assert any("y_cls" in e for e in _errors_after(paired_trip, mutate))


def test_contract_version_mismatch(paired_trip) -> None:
    def mutate(df, m):
        df["contract_version"] = "9.9"

    assert any("contract_version" in e for e in _errors_after(paired_trip, mutate))


def test_mask_rule(paired_trip) -> None:
    def mutate(df, m):
        df["mask_v"] = True

    assert any("mask_v disagrees" in e for e in _errors_after(paired_trip, mutate))


def test_manifest_leakage(stub_root: Path) -> None:
    manifest = pd.read_csv(stub_root / "own" / "manifest.csv", dtype={"driver_id": str})
    manifest.loc[0, "split"] = "test" if manifest.loc[0, "split"] != "test" else "train"
    leaky = pd.concat(
        [manifest, manifest.iloc[[0]].assign(trip_id="own_d01_20261101_99", split="val")]
    )
    assert any("more than one split" in e for e in validate_manifest(leaky))


def test_manifest_missing_driver(stub_root: Path) -> None:
    manifest = pd.read_csv(stub_root / "own" / "manifest.csv", dtype={"driver_id": str})
    manifest.loc[0, "driver_id"] = None
    assert any("without driver_id" in e for e in validate_manifest(manifest))


def test_events_rules() -> None:
    assert validate_events([{"onset_s": 1.0, "offset_s": 2.0, "type": "harsh_brake"}], 10.0) == []
    assert validate_events([{"onset_s": 1.0, "offset_s": 2.0, "type": "nap"}], 10.0)
    assert validate_events([{"onset_s": 5.0, "offset_s": 2.0, "type": "harsh_brake"}], 10.0)
    assert validate_events([{"onset_s": 1.0}], 10.0)


def test_parquet_without_manifest_row(stub_root: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(stub_root / "dmd", tmp_path / "dmd")
    extra = next((tmp_path / "dmd").glob("*.parquet"))
    shutil.copy(extra, tmp_path / "dmd" / "dmd_d09_20261101_01.parquet")
    report = validate_processed_dir(tmp_path / "dmd")
    assert any("no manifest row" in e for e in report.errors)
