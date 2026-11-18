"""Driver-disjoint split file (P1-C2): `splits.py` record functions and `scripts/make_splits.py`.

Hand-made manifests give exact expectations; the session stub dataset checks the real file
layout. The stub folders are shared by the session, so anything that rewrites a manifest
works on a copy.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data import splits as SP
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.paths import repo_root

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _row(dataset: str, driver: str, date: str = "20261101", nn: int = 1, tag: str = "normal"):
    return {
        "trip_id": f"{dataset}_{driver}_{date}_{nn:02d}",
        "dataset": dataset,
        "driver_id": driver,
        "pairing": "motion_only",
        "duration_s": "120.0",
        "n_steps": "1200",
        "behavior_tag": tag,
        "road_type": "",
        "vehicle_id": "",
        "split": "",
        "source_version": "test",
        "contract_version": S.CONTRACT_VERSION,
    }


def _manifest(dataset: str, n_drivers: int, trips_per_driver: int = 2) -> pd.DataFrame:
    rows = [
        _row(dataset, f"d{d + 1:02d}", nn=k + 1)
        for d in range(n_drivers)
        for k in range(trips_per_driver)
    ]
    return pd.DataFrame(rows, columns=list(S.MANIFEST_COLUMNS))


def _record(manifests: dict[str, pd.DataFrame], **kw) -> dict:
    args = {
        "seed": 0,
        "fractions": FRACTIONS,
        "fractions_per_dataset": {},
        "lodo_datasets": [],
        "personalization": {"k_calibration": 2, "min_trips": 4, "normal_tags": ["normal"]},
    }
    args.update(kw)
    return SP.build_split_record(manifests, **args)


def _script():
    spec = importlib.util.spec_from_file_location(
        "make_splits", repo_root() / "scripts" / "make_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_copy(stub_root: Path, tmp_path: Path) -> list[Path]:
    dst = tmp_path / "processed"
    shutil.copytree(stub_root, dst)
    return [dst / d for d in ("uah", "dmd", "own")]


# --- driver-level assignment ------------------------------------------------------------------


def test_every_driver_assigned_exactly_once_and_no_driver_in_two_splits(
    stub_dirs: list[Path],
) -> None:
    manifests = {
        f"{d.name}/manifest.csv": pd.read_csv(d / "manifest.csv", dtype=str, keep_default_na=False)
        for d in stub_dirs
    }
    record = _record(manifests)
    keys = {
        S.driver_key(ds, did)
        for m in manifests.values()
        for ds, did in zip(m["dataset"], m["driver_id"], strict=True)
    }
    assert set(record["drivers"]) == keys
    assert all(v["split"] in S.SPLITS for v in record["drivers"].values())
    applied = pd.concat([SP.apply_split_record(m, record) for m in manifests.values()])
    SP.check_driver_disjoint(applied)
    assert applied.groupby(["dataset", "driver_id"])["split"].nunique().max() == 1
    # every trip is listed under exactly one driver
    listed = [t for v in record["drivers"].values() for t in v["trips"]]
    assert sorted(listed) == sorted(applied["trip_id"])


def test_same_token_in_two_datasets_is_two_drivers() -> None:
    record = _record({"a": _manifest("uah", 3), "b": _manifest("dmd", 3)})
    assert {"uah:d01", "dmd:d01"} <= set(record["drivers"])
    assert len(record["drivers"]) == 6


def test_fractions_are_respected_per_dataset() -> None:
    record = _record({"m": _manifest("dmd", 20)})
    counts = pd.Series([v["split"] for v in record["drivers"].values()]).value_counts()
    assert counts.to_dict() == {"train": 14, "val": 3, "test": 3}


def test_per_dataset_fractions_override_the_default() -> None:
    record = _record(
        {"a": _manifest("dmd", 20), "b": _manifest("own", 4)},
        fractions_per_dataset={"own": {"train": 0.5, "val": 0.25, "test": 0.25}},
    )
    own = [v["split"] for k, v in record["drivers"].items() if k.startswith("own:")]
    assert sorted(own) == ["test", "train", "train", "val"]


def test_too_few_drivers_for_every_split_gives_a_warning() -> None:
    record = _record({"m": _manifest("own", 2)})
    assert any("own" in w for w in record["warnings"])


def test_bad_fractions_are_refused() -> None:
    with pytest.raises(ValueError, match="fractions"):
        _record({"m": _manifest("dmd", 5)}, fractions={"train": 0.7, "val": 0.2})
    with pytest.raises(ValueError, match="fractions"):
        _record({"m": _manifest("dmd", 5)}, fractions={"train": 0.5, "holdout": 0.5})


# --- determinism ------------------------------------------------------------------------------


def test_same_inputs_and_seed_give_an_identical_file(stub_dirs: list[Path], tmp_path: Path) -> None:
    mod = _script()
    paths = [str(d / "manifest.csv") for d in stub_dirs]
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    assert mod.main([*paths, "--out", str(out_a)]) == 0
    assert mod.main([*reversed(paths), "--out", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    record = json.loads(out_a.read_text(encoding="utf-8"))
    assert record["seed"] == 0 and record["fractions"]["default"]
    assert all(len(i["sha256"]) == 64 for i in record["inputs"])
    # No absolute paths in the file (the repo is public).
    assert str(tmp_path) not in out_a.read_text(encoding="utf-8")


def test_row_order_does_not_matter_and_seed_does() -> None:
    m = _manifest("dmd", 20)
    shuffled = m.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a, b = _record({"m": m}), _record({"m": shuffled})
    assert SP.dump_split_record(a) == SP.dump_split_record(b)
    c = _record({"m": m}, seed=1)
    assert {k: v["split"] for k, v in a["drivers"].items()} != {
        k: v["split"] for k, v in c["drivers"].items()
    }


def test_adding_a_dataset_does_not_reshuffle_another() -> None:
    alone = _record({"m": _manifest("dmd", 20)})
    both = _record({"m": _manifest("dmd", 20), "n": _manifest("uah", 6)})
    dmd = {k: v["split"] for k, v in alone["drivers"].items()}
    assert dmd == {k: v["split"] for k, v in both["drivers"].items() if k.startswith("dmd:")}


# --- refusals ---------------------------------------------------------------------------------


def test_duplicate_trip_ids_are_refused_within_and_across_manifests() -> None:
    m = _manifest("uah", 3)
    dup = pd.concat([m, m.iloc[[0]]], ignore_index=True)
    with pytest.raises(SP.SplitInputError, match="duplicate trip_id"):
        _record({"m": dup})
    with pytest.raises(SP.SplitInputError, match="duplicate trip_id"):
        _record({"a": m, "b": m.copy()})


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_missing_driver_id_is_refused(missing) -> None:
    m = _manifest("uah", 3)
    m.loc[1, "driver_id"] = missing
    with pytest.raises(SP.SplitInputError, match="driver_id"):
        _record({"m": m})


def test_unknown_dataset_and_missing_columns_are_refused() -> None:
    m = _manifest("uah", 3)
    m.loc[0, "dataset"] = "kitti"
    with pytest.raises(SP.SplitInputError, match="dataset"):
        _record({"m": m})
    with pytest.raises(SP.SplitInputError, match="column"):
        _record({"m": _manifest("uah", 3).drop(columns=["driver_id"])})


def test_trip_id_that_names_another_driver_is_refused() -> None:
    # The trip id carries the date used for the chronological personalization split, and a
    # wrong driver part would mean a mislabelled row.
    m = _manifest("uah", 3)
    m.loc[0, "driver_id"] = "d09"
    with pytest.raises(SP.SplitInputError, match="trip_id"):
        _record({"m": m})


def test_script_reports_refusal_and_writes_nothing(tmp_path: Path) -> None:
    m = _manifest("uah", 3)
    m.loc[0, "driver_id"] = ""
    path = tmp_path / "uah" / "manifest.csv"
    path.parent.mkdir()
    m.to_csv(path, index=False)
    out = tmp_path / "splits.json"
    assert _script().main([str(path), "--out", str(out)]) == 2
    assert not out.exists()


# --- leave-one-driver-out with nested validation (F16) ---------------------------------------


def test_lodo_folds_have_nested_rotating_validation() -> None:
    keys = [f"uah:d{i:02d}" for i in range(1, 7)]
    folds = SP.lodo_folds(keys)
    assert len(folds) == 6
    assert sorted(f["test"] for f in folds) == keys  # each driver tested exactly once
    for f in folds:
        assert f["test"] not in f["train"] and f["test"] != f["val"]
        assert f["val"] not in f["train"]
        assert len(f["train"]) == 4
        assert sorted([f["test"], f["val"], *f["train"]]) == keys
    assert sorted(f["val"] for f in folds) == keys  # the validation driver rotates
    assert len({(f["test"], f["val"]) for f in folds}) == 6


def test_lodo_needs_three_drivers() -> None:
    with pytest.raises(ValueError, match="3 drivers"):
        SP.lodo_folds(["uah:d01", "uah:d02"])


def test_lodo_is_stored_in_the_record_for_the_requested_datasets() -> None:
    record = _record({"a": _manifest("uah", 6), "b": _manifest("dmd", 10)}, lodo_datasets=["uah"])
    assert set(record["lodo"]) == {"uah"}
    folds = record["lodo"]["uah"]["folds"]
    assert len(folds) == 6 and all(f["test"].startswith("uah:") for f in folds)
    assert [f["fold"] for f in folds] == list(range(6))
    with pytest.raises(ValueError, match="own"):
        _record({"a": _manifest("uah", 6)}, lodo_datasets=["own"])


# --- personalization split (04 section 2.2) --------------------------------------------------


def test_personalization_calibrates_on_first_normal_trips_chronologically() -> None:
    rows = [
        _row("own", "d01", "20261105", 1, "drowsy"),  # last in time
        _row("own", "d01", "20261101", 1, "normal"),
        _row("own", "d01", "20261101", 2, "aggressive"),
        _row("own", "d01", "20261102", 1, "normal"),
        _row("own", "d01", "20261103", 1, "normal"),
    ]
    m = pd.DataFrame(rows, columns=list(S.MANIFEST_COLUMNS))
    out = SP.personalization_split(m, k_calibration=2, min_trips=4, normal_tags=("normal",))
    d = out["own:d01"]
    assert d["status"] == "ok"
    assert d["calibration"] == ["own_d01_20261101_01", "own_d01_20261102_01"]
    assert d["evaluation"] == ["own_d01_20261103_01", "own_d01_20261105_01"]
    # recorded before the calibration ended, so not "chronologically later": listed, not lost
    assert d["excluded"] == {"own_d01_20261101_02": "before end of calibration"}


def test_personalization_reports_drivers_without_enough_data() -> None:
    rows = [_row("own", "d01", "20261101", k) for k in range(1, 4)]  # 3 trips
    rows += [_row("own", "d02", "20261101", k, "drowsy") for k in range(1, 4)]
    rows += [_row("own", "d02", "20261102", 1, "normal")]  # 4 trips, 1 normal
    rows += [_row("own", "d03", "20261101", k, "drowsy") for k in range(1, 3)]
    rows += [_row("own", "d03", "20261102", k, "normal") for k in range(1, 3)]  # nothing after
    m = pd.DataFrame(rows, columns=list(S.MANIFEST_COLUMNS))
    out = SP.personalization_split(m, k_calibration=2, min_trips=4, normal_tags=("normal",))
    assert set(out) == {"own:d01", "own:d02", "own:d03"}  # nobody silently dropped
    for key in out:
        assert out[key]["status"] == "not enough data" and out[key]["reason"]
    assert "3 trips" in out["own:d01"]["reason"]
    assert "normal" in out["own:d02"]["reason"]
    assert "evaluation" in out["own:d03"]["reason"]


def test_personalization_is_in_the_record() -> None:
    record = _record({"m": _manifest("own", 3, trips_per_driver=5)})
    assert set(record["personalization"]["drivers"]) == {"own:d01", "own:d02", "own:d03"}
    assert record["personalization"]["k_calibration"] == 2


# --- hashes, rewrite, verify ------------------------------------------------------------------


def test_manifest_hash_ignores_the_split_column_and_row_order() -> None:
    m = _manifest("uah", 3)
    h = SP.manifest_sha256(m)
    m2 = m.copy()
    m2["split"] = "train"
    assert SP.manifest_sha256(m2) == h
    assert SP.manifest_sha256(m.iloc[::-1]) == h
    m3 = m.copy()
    m3.loc[0, "duration_s"] = "121.0"
    assert SP.manifest_sha256(m3) != h


def test_write_manifests_rewrites_only_the_split_column(stub_root: Path, tmp_path: Path) -> None:
    dirs = _stub_copy(stub_root, tmp_path)
    before = {d: pd.read_csv(d / "manifest.csv", dtype=str, keep_default_na=False) for d in dirs}
    out = tmp_path / "splits.json"
    paths = [str(d / "manifest.csv") for d in dirs]
    assert _script().main([*paths, "--out", str(out), "--write-manifests"]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    for d in dirs:
        after = pd.read_csv(d / "manifest.csv", dtype=str, keep_default_na=False)
        cols = [c for c in after.columns if c != "split"]
        pd.testing.assert_frame_equal(after[cols], before[d][cols])
        rows = zip(after["dataset"], after["driver_id"], after["split"], strict=True)
        for ds, did, split in rows:
            assert split == record["drivers"][S.driver_key(ds, did)]["split"]
        assert validate_processed_dir(d).ok
    assert SP.verify_split_record(record, {p: pd.read_csv(p, dtype=str) for p in paths}) == []
    # Running again on the rewritten manifests gives the same file.
    out2 = tmp_path / "splits2.json"
    assert _script().main([*paths, "--out", str(out2)]) == 0
    assert out2.read_bytes() == out.read_bytes()


def test_verify_catches_changed_inputs_and_wrong_split_column() -> None:
    m = _manifest("dmd", 10)
    record = _record({"m": m})
    applied = SP.apply_split_record(m, record)
    assert SP.verify_split_record(record, {"m": applied}) == []
    changed = applied.copy()
    changed.loc[0, "n_steps"] = "999"
    assert any("sha256" in e for e in SP.verify_split_record(record, {"m": changed}))
    wrong = applied.copy()
    wrong.loc[0, "split"] = "test" if wrong.loc[0, "split"] != "test" else "train"
    assert any("split" in e for e in SP.verify_split_record(record, {"m": wrong}))


def test_script_verify_mode(stub_root: Path, tmp_path: Path) -> None:
    dirs = _stub_copy(stub_root, tmp_path)
    paths = [str(d / "manifest.csv") for d in dirs]
    out = tmp_path / "splits.json"
    mod = _script()
    assert mod.main([*paths, "--out", str(out), "--write-manifests"]) == 0
    assert mod.main([*paths, "--out", str(out), "--verify"]) == 0
    m = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    m.loc[0, "duration_s"] = "1.0"
    m.to_csv(paths[0], index=False)
    assert mod.main([*paths, "--out", str(out), "--verify"]) == 1


def test_existing_assign_driver_splits_is_unchanged() -> None:
    # The stub generator relies on it; the record functions are additive.
    m = _manifest("uah", 4)
    out = SP.assign_driver_splits(m, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    assert set(out["split"]) == {"train", "val", "test"}
