"""WindowDataset: shapes, masks, NaN handling, labels, normalization, split enforcement."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from driverisk.data import schema as S
from driverisk.data.datasets import WindowDataset, risk_targets, window_label
from driverisk.data.splits import SplitLeakageError, assign_driver_splits, check_driver_disjoint


@pytest.fixture(scope="module")
def train_ds(stub_dirs: list[Path]) -> WindowDataset:
    return WindowDataset(stub_dirs, "train")


def test_batch_matches_contract_section_7(train_ds: WindowDataset) -> None:
    batch = next(iter(DataLoader(train_ds, batch_size=8)))
    T = S.WINDOW_STEPS
    assert batch["x_v"].shape == (8, T, 16) and batch["x_v"].dtype == torch.float32
    assert batch["x_m"].shape == (8, T, 14) and batch["x_m"].dtype == torch.float32
    assert batch["nan_v"].shape == (8, T, 16) and batch["nan_v"].dtype == torch.bool
    assert batch["nan_m"].shape == (8, T, 14) and batch["nan_m"].dtype == torch.bool
    assert batch["mask_v"].shape == (8, T) and batch["mask_v"].dtype == torch.bool
    assert batch["mask_m"].shape == (8, T) and batch["mask_m"].dtype == torch.bool
    assert batch["y_cls"].shape == (8,) and batch["y_cls"].dtype == torch.int64
    assert batch["y_risk"].shape == (8, 3) and batch["y_risk"].dtype == torch.float32
    assert len(batch["driver_id"]) == 8 and len(batch["trip_id"]) == 8
    assert batch["t_start"].shape == (8,)


def test_no_nan_reaches_the_model(train_ds: WindowDataset) -> None:
    for i in range(0, len(train_ds), 7):
        item = train_ds[i]
        assert not torch.isnan(item["x_v"]).any() and not torch.isnan(item["x_m"]).any()
        assert (item["x_v"][item["nan_v"]] == 0).all()
        assert (item["x_m"][item["nan_m"]] == 0).all()


def test_unpaired_trips_have_one_modality(stub_dirs: list[Path]) -> None:
    uah = WindowDataset([stub_dirs[0]], None)
    dmd = WindowDataset([stub_dirs[1]], None)
    assert not any(uah[i]["mask_v"].any() for i in range(len(uah)))
    assert not any(dmd[i]["mask_m"].any() for i in range(len(dmd)))
    assert all(uah[i]["nan_v"].all() for i in range(0, len(uah), 11))


def test_window_count_and_stride(stub_dirs: list[Path]) -> None:
    ds = WindowDataset([stub_dirs[2]], None)
    expected = sum((len(t.t) - S.WINDOW_STEPS) // S.STRIDE_STEPS + 1 for t in ds.trips)
    assert len(ds) == expected
    first, second = ds[0], ds[1]
    assert second["t_start"] - first["t_start"] == pytest.approx(
        S.STRIDE_STEPS * S.STEP_S, abs=1e-4
    )


def test_splits_are_driver_disjoint(stub_dirs: list[Path]) -> None:
    drivers = {
        s: {WindowDataset(stub_dirs, s)[i]["driver_id"] for i in range(0, 50)} for s in S.SPLITS
    }
    assert not (drivers["train"] & drivers["val"]) and not (drivers["train"] & drivers["test"])


def test_leaky_manifest_is_rejected(stub_dirs: list[Path], tmp_path: Path) -> None:
    leaky = tmp_path / "own"
    shutil.copytree(stub_dirs[2], leaky)
    m = pd.read_csv(leaky / "manifest.csv", dtype={"driver_id": str})
    driver = m["driver_id"].iloc[0]
    rows = m.index[m["driver_id"] == driver]
    if len(rows) < 2:
        pytest.skip("stub driver has a single trip")
    m.loc[rows[0], "split"], m.loc[rows[1], "split"] = "train", "test"
    m.to_csv(leaky / "manifest.csv", index=False)
    with pytest.raises(SplitLeakageError):
        WindowDataset([leaky], "train")


def test_same_driver_id_in_two_datasets_is_not_leakage() -> None:
    m = pd.DataFrame(
        {"dataset": ["uah", "dmd"], "driver_id": ["d01", "d01"], "split": ["train", "test"]}
    )
    check_driver_disjoint(m)  # different datasets, different people


def test_assign_driver_splits_covers_every_split() -> None:
    m = pd.DataFrame(
        {"dataset": "own", "driver_id": [f"d{i:02d}" for i in range(1, 5) for _ in range(2)]}
    )
    out = assign_driver_splits(m, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    assert set(out["split"]) == {"train", "val", "test"}
    check_driver_disjoint(out)


def test_norm_stats_only_from_train(stub_dirs: list[Path], train_ds: WindowDataset) -> None:
    stats = train_ds.fit_norm_stats()
    assert stats.mean_v.shape == (16,) and stats.std_m.shape == (14,)
    assert np.all(stats.std_v > 0) and np.all(stats.std_m > 0)
    with pytest.raises(ValueError):
        WindowDataset(stub_dirs, "val").fit_norm_stats()


def test_norm_stats_roundtrip(train_ds: WindowDataset, tmp_path: Path) -> None:
    from driverisk.data.normalize import NormStats

    stats = train_ds.fit_norm_stats()
    stats.to_json(tmp_path / "norm.json")
    back = NormStats.from_json(tmp_path / "norm.json")
    np.testing.assert_allclose(back.mean_m, stats.mean_m, rtol=1e-6)


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        ([2] * 30 + [0] * 20, 2),
        ([-1] * 30 + [1] * 20, -1),  # fewer than half labelled
        ([1] * 25 + [3] * 25, -1),  # tie
        ([-1] * 20 + [3] * 30, 3),
    ],
)
def test_window_label(y: list[int], expected: int) -> None:
    assert window_label(np.array(y, dtype=np.int8)) == expected


def test_risk_targets(train_ds: WindowDataset) -> None:
    trip = train_ds.trips[0]
    trip.onsets, trip.uncertain, trip.events_annotated = np.array([12.5]), [], True
    np.testing.assert_array_equal(risk_targets(trip, 10.0, (2, 3, 5)), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(risk_targets(trip, trip.duration_s - 1.0, (2,)), [-1.0])
    trip.uncertain = [(11.0, 11.5)]
    np.testing.assert_array_equal(risk_targets(trip, 10.0, (2,)), [-1.0])
    trip.events_annotated = False
    np.testing.assert_array_equal(risk_targets(trip, 10.0, (5,)), [-1.0])
