"""UAH-DriveSet loader (P1-B1) on a synthetic UAH-like tree whose answers are known exactly.

The real download has not been seen, so the tree follows the layout the repo docs record from
the paper (route folder name format, file prefixes, the column map in configs/uah.yaml). These
tests prove the conversion maths and the loud failures; they cannot prove the real format.

Every synthetic route lasts 60 s (raw timestamps start at 1000 s to test the t0 shift):
- accelerometer at 10 Hz, in G: braking of 0.3 G on Z for 20 <= t < 23 s, lateral 0.2 G on Y
  for 30 <= t < 32 s, vertical X = 0; roll 2 deg, pitch -1.5 deg, yaw a +10 deg/s ramp from
  170 deg (crosses 180 deg at t = 1 s and wraps several times)
- GPS at 1 Hz at t = 0.5, 1.5, ...: 90 km/h before 40 s, 30 km/h after; course constant
- lane detection at 20 Hz: offset 0.3 sin(2 pi t / 10) m, detector state 1 (lost) for 10-12 s
- inertial events: braking at 20.5 s (90 km/h), turning at 31 s (90 km/h), acceleration at
  45 s (30 km/h, below the 50 km/h logging limit, so it must be dropped)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data import uah as U
from driverisk.data.validate import validate_processed_dir
from driverisk.features import motion as F
from driverisk.utils.paths import repo_root

G = 9.80665
T0 = 1000.0
DURATION = 60.0
ROUTES = {
    # folder name: (driver id, trip id)
    "20151111123124-25km-D1-NORMAL-MOTORWAY": ("d01", "uah_d01_20151111_01"),
    "20151111134545-16km-D1-AGGRESSIVE-SECONDARY": ("d01", "uah_d01_20151111_02"),
    "20151120160000-25km-D2-NORMAL-MOTORWAY": ("d02", "uah_d02_20151120_01"),
    "20151121093000-25km-D2-AGGRESSIVE-MOTORWAY": ("d02", "uah_d02_20151121_01"),
}
EVENTS_AGGRESSIVE = [(20.5, 1, 2), (31.0, 2, 3), (45.0, 3, 1)]  # (t, type, level)
EVENTS_NORMAL = [(20.5, 1, 1)]


def _wrap180(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def _acc_table(n_cols: int = 11) -> np.ndarray:
    t = np.round(np.arange(0.0, DURATION, 0.1), 6)
    ax = np.zeros_like(t)
    ay = np.where((t >= 30) & (t < 32), 0.2, 0.0)
    az = np.where((t >= 20) & (t < 23), -0.3, 0.0)
    cols = [
        T0 + t, np.ones_like(t), ax, ay, az, ax, ay, az,
        np.full_like(t, 2.0), np.full_like(t, -1.5), _wrap180(170.0 + 10.0 * t),
    ]  # fmt: skip
    return np.column_stack(cols[:n_cols])


def _gps_table() -> np.ndarray:
    t = np.arange(0.5, DURATION, 1.0)
    speed = np.where(t < 40, 90.0, 30.0)
    z = np.zeros_like(t)
    return np.column_stack([T0 + t, speed, z + 40.4, z - 3.6, z + 600, z + 5, z + 5, z + 90, z])


def _lane_table() -> np.ndarray:
    t = np.round(np.arange(0.0, DURATION, 0.05), 6)
    x = 0.3 * np.sin(2 * np.pi * t / 10.0)
    state = np.where((t >= 10) & (t < 12), 1.0, 2.0)
    return np.column_stack([T0 + t, x, np.zeros_like(t), np.full_like(t, 3.5), state])


def _events_table(events: list[tuple[float, int, int]]) -> np.ndarray:
    rows = [[T0 + t, typ, lvl, 40.4, -3.6, 20151111123124] for t, typ, lvl in events]
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def _save(path: Path, table: np.ndarray) -> None:
    np.savetxt(path, table, fmt="%.6f")


def make_route(root: Path, name: str, *, acc_cols: int = 11) -> Path:
    driver = name.split("-")[2]
    folder = root / driver / name
    folder.mkdir(parents=True)
    _save(folder / "RAW_ACCELEROMETERS.txt", _acc_table(acc_cols))
    _save(folder / "RAW_GPS.txt", _gps_table())
    _save(folder / "PROC_LANE_DETECTION.txt", _lane_table())
    events = EVENTS_AGGRESSIVE if "AGGRESSIVE" in name else EVENTS_NORMAL
    _save(folder / "EVENTS_INERTIAL.txt", _events_table(events))
    (folder / "EVENTS_LIST_LANE_CHANGES.txt").write_text("", encoding="utf-8")  # not read
    return folder


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for name in ROUTES:
        make_route(root, name)
    return root


@pytest.fixture
def cfg() -> dict:
    return U.load_uah_config()


@pytest.fixture
def label_map() -> dict:
    return U.load_label_map()


@pytest.fixture
def aggressive(tree: Path, cfg: dict, label_map: dict) -> U.UAHTrip:
    folder = next(tree.rglob("20151111134545*"))
    return U.convert_route(folder, "uah_d01_20151111_02", cfg, label_map)


def _rows(frame: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    t = frame[S.COL_T].to_numpy(np.float64)
    return frame[(t > lo + 1e-6) & (t < hi - 1e-6)]


# --- folder names, ids, labels ---------------------------------------------------------------
def test_parse_route_folder_and_driver_token(tree: Path) -> None:
    r = U.parse_route_folder(next(tree.rglob("20151111134545*")))
    assert (r.driver_raw, r.driver_id, r.behaviour, r.road) == ("D1", "d01", "AGGRESSIVE",
                                                                "SECONDARY")  # fmt: skip
    assert r.day == "20151111" and r.timestamp == "20151111134545" and r.distance_km == 16.0


def test_unparseable_folder_name_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(U.UAHFormatError, match="folder name"):
        U.parse_route_folder(tmp_path / "D1-normal-trip")


def test_trip_ids_number_trips_per_driver_and_day(tree: Path) -> None:
    routes = [U.parse_route_folder(f) for f in U.find_route_folders(tree)]
    ids = U.assign_trip_ids(routes)
    assert {r.folder.name: ids[r.folder] for r in routes} == {k: v[1] for k, v in ROUTES.items()}
    assert all(S.TRIP_ID_RE.match(v) for v in ids.values())


def test_label_map_maps_behaviours_to_contract_ids(label_map: dict) -> None:
    assert label_map == {
        "NORMAL": S.CLASS_TO_ID["normal"],
        "DROWSY": S.CLASS_TO_ID["drowsy"],
        "AGGRESSIVE": S.CLASS_TO_ID["aggressive"],
    }


def test_trip_label_is_trip_level(aggressive: U.UAHTrip, tree: Path, cfg, label_map) -> None:
    assert (aggressive.frame[S.COL_Y_CLS] == S.CLASS_TO_ID["aggressive"]).all()
    assert aggressive.frame[S.COL_Y_CLS].dtype == np.int8
    normal = U.convert_route(next(tree.rglob("20151111123124*")), "uah_d01_20151111_01", cfg,
                             label_map)  # fmt: skip
    assert (normal.frame[S.COL_Y_CLS] == S.CLASS_TO_ID["normal"]).all()
    assert normal.manifest_row["behavior_tag"] == "normal"
    assert normal.manifest_row["road_type"] == "motorway"


def test_unknown_behaviour_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-SLEEPY-MOTORWAY")
    with pytest.raises(U.UAHFormatError, match="SLEEPY"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


# --- unit and axis conversion (contract 4.4) -------------------------------------------------
def test_braking_in_g_becomes_mps2_on_acc_long(aggressive: U.UAHTrip) -> None:
    f = aggressive.frame
    brake = _rows(f, 20.0 - 0.05, 23.0 - 0.05)["m_acc_long"].to_numpy()
    assert brake == pytest.approx(-0.3 * G, abs=1e-5)
    assert _rows(f, 5, 19.9)["m_acc_long"].to_numpy() == pytest.approx(0.0, abs=1e-6)
    turn = _rows(f, 30.0 - 0.05, 32.0 - 0.05)["m_acc_lat"].to_numpy()
    assert turn == pytest.approx(0.2 * G, abs=1e-5)
    assert np.nanmax(np.abs(f["m_acc_vert"].to_numpy())) == pytest.approx(0.0, abs=1e-6)


def test_step_k_covers_the_interval_ending_at_k(aggressive: U.UAHTrip) -> None:
    """Causal grid of motion.py: the raw sample at exactly 20.0 s lands in the step t = 20.0."""
    f = aggressive.frame.set_index(S.COL_T)
    t = f.index.to_numpy(np.float64)
    at = lambda s: f["m_acc_long"].to_numpy()[np.argmin(np.abs(t - s))]  # noqa: E731
    assert at(19.9) == pytest.approx(0.0, abs=1e-6)
    assert at(20.0) == pytest.approx(-0.3 * G, abs=1e-5)


def test_speed_kmh_to_mps_and_held_causally(aggressive: U.UAHTrip) -> None:
    f = aggressive.frame
    assert _rows(f, 1.0, 40.4)["m_speed"].to_numpy() == pytest.approx(25.0, abs=1e-5)
    assert _rows(f, 40.6, 59.9)["m_speed"].to_numpy() == pytest.approx(30 / 3.6, abs=1e-5)
    first = f[S.COL_T].to_numpy() < 0.5 - 1e-6  # before the first fix: no speed
    assert f.loc[first, "m_speed"].isna().all() and (f.loc[first, "m_gps_valid"] == 0).all()
    assert (_rows(f, 0.5, 59.9)["m_gps_valid"] == 1).all()


def test_yaw_rate_from_unwrapped_yaw_across_180(aggressive: U.UAHTrip) -> None:
    rate = aggressive.frame["m_yaw_rate"].to_numpy(np.float64)
    finite = rate[np.isfinite(rate)]
    assert len(finite) >= len(rate) - 1
    assert finite == pytest.approx(10.0, abs=1e-3)  # no +-3600 deg/s spike at the wrap


def test_roll_and_pitch_come_from_the_file(aggressive: U.UAHTrip) -> None:
    assert aggressive.frame["m_roll"].to_numpy() == pytest.approx(2.0, abs=1e-5)
    assert aggressive.frame["m_pitch"].to_numpy() == pytest.approx(-1.5, abs=1e-5)


def test_axis_signs_are_configurable(tree: Path, cfg: dict, label_map: dict) -> None:
    cfg["acceleration"]["axes"]["long"]["sign"] = -1
    cfg["attitude"]["yaw_sign"] = -1
    trip = U.convert_route(next(tree.rglob("20151111134545*")), "uah_d01_20151111_02", cfg,
                           label_map)  # fmt: skip
    brake = _rows(trip.frame, 20.0 - 0.05, 23.0 - 0.05)["m_acc_long"].to_numpy()
    assert brake == pytest.approx(0.3 * G, abs=1e-5)
    assert np.nanmean(trip.frame["m_yaw_rate"]) == pytest.approx(-10.0, abs=1e-3)


def test_lane_offset_and_drift_from_lane_detection(aggressive: U.UAHTrip) -> None:
    f = aggressive.frame
    t = f[S.COL_T].to_numpy(np.float64)
    lane = f["m_lane_offset"].to_numpy(np.float64)
    # detector state != 2 from 10.0 s to 11.95 s; the latest sample of a step decides, so the
    # step t = 10.0 (samples 9.95 valid, 10.0 lost) is already NaN and t = 12.0 is valid again
    lost = (t > 10.0 - 1e-6) & (t < 12.0 - 1e-6)
    assert np.isnan(lane[lost]).all() and np.isfinite(lane[np.isclose(t, 12.0)]).all()
    ok = (t > 0.05) & ~lost & (t < 59.9)
    # the last lane sample of step k is at t_k (20 Hz samples at k*0.1 - 0.05 and k*0.1)
    expected = 0.3 * np.sin(2 * np.pi * t[ok] / 10.0)
    assert lane[ok] == pytest.approx(expected, abs=1e-5)
    drift = pd.Series(lane).rolling(50, min_periods=10).std(ddof=0).to_numpy()
    got = f["m_lateral_drift_5s"].to_numpy(np.float64)
    both = np.isfinite(drift)
    assert np.isnan(got[~both]).all() and got[both] == pytest.approx(drift[both], abs=1e-5)


def test_vehicle_aligned_data_skips_alignment_estimation(aggressive: U.UAHTrip) -> None:
    align = aggressive.meta["alignment"]
    assert align["ok"] and align["R"] == np.eye(3).tolist()
    assert "not estimated" in align["reason"]


# --- vision side, pairing, meta --------------------------------------------------------------
def test_motion_only_trip_declares_all_vision_nan(aggressive: U.UAHTrip) -> None:
    f = aggressive.frame
    assert f[list(S.VISION_COLUMNS)].isna().all().all()
    assert not f[S.COL_MASK_V].any() and f[S.COL_MASK_M].all()
    assert set(S.VISION_COLUMNS) <= set(aggressive.meta[S.META_ALL_NAN_KEY])
    assert aggressive.manifest_row["pairing"] == "motion_only"
    assert aggressive.manifest_row["n_steps"] == len(f) == 600


# --- inertial events and the 50 km/h rule ----------------------------------------------------
def test_inertial_events_mapped_and_gated_at_50_kmh(aggressive: U.UAHTrip) -> None:
    ev = aggressive.events
    assert [(e["onset_s"], e["type"]) for e in ev] == [(20.5, "harsh_brake"), (31.0, "sharp_turn")]
    assert all(e["source"] == "uah_inertial" for e in ev)
    assert [e["uah_level"] for e in ev] == [2, 3]
    m = aggressive.meta["inertial_events"]
    assert m["min_speed_kmh"] == 50.0 and m["n_read"] == 3 and m["n_dropped_low_speed"] == 1
    assert "only above 50 km/h" in aggressive.meta["notes"]
    # where events can exist at all: held GPS speed above 50 km/h, from the first fix to 40.4 s
    assert m["logged_segments_s"] == [[0.5, 40.4]]


def test_min_level_filters_low_events(tree: Path, cfg: dict, label_map: dict) -> None:
    cfg["inertial_events"]["min_level"] = 3
    trip = U.convert_route(next(tree.rglob("20151111134545*")), "uah_d01_20151111_02", cfg,
                           label_map)  # fmt: skip
    assert [e["type"] for e in trip.events] == ["sharp_turn"]


# --- causality --------------------------------------------------------------------------------
def test_future_raw_samples_do_not_change_past_steps(tmp_path: Path, cfg, label_map) -> None:
    a = make_route(tmp_path / "a", "20151111134545-16km-D1-AGGRESSIVE-SECONDARY")
    b = make_route(tmp_path / "b", "20151111134545-16km-D1-AGGRESSIVE-SECONDARY")
    cut = T0 + 30.0
    for name, cols in (("RAW_ACCELEROMETERS.txt", [2, 3, 4, 8, 9, 10]), ("RAW_GPS.txt", [1]),
                       ("PROC_LANE_DETECTION.txt", [1])):  # fmt: skip
        tab = np.loadtxt(b / name)
        future = tab[:, 0] > cut + 1e-6
        tab[np.ix_(future, cols)] = tab[np.ix_(future, cols)] * 0.5 + 0.1
        _save(b / name, tab)
    fa = U.convert_route(a, "uah_d01_20151111_02", cfg, label_map).frame
    fb = U.convert_route(b, "uah_d01_20151111_02", cfg, label_map).frame
    past = fa[S.COL_T].to_numpy() <= 30.0 + 1e-6
    pd.testing.assert_frame_equal(fa[past], fb[past])
    assert not fa[~past].equals(fb[~past])


# --- loud failures ----------------------------------------------------------------------------
def test_wrong_column_count_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY", acc_cols=10)
    with pytest.raises(U.UAHFormatError, match=r"RAW_ACCELEROMETERS.*10 columns.*expected 11"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_non_numeric_row_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "RAW_GPS.txt"
    path.write_text("1.0 2.0 3.0\nfoo bar baz\n", encoding="utf-8")
    with pytest.raises(U.UAHFormatError, match="line 2"):
        U.read_numeric_table(path, 3)


def test_acceleration_not_in_g_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    tab = np.loadtxt(folder / "RAW_ACCELEROMETERS.txt")
    tab[:, 2:8] *= G
    tab[:, 2] += G  # looks like m/s^2 with gravity
    _save(folder / "RAW_ACCELEROMETERS.txt", tab)
    with pytest.raises(U.UAHFormatError, match="not in G"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_unexpected_gravity_on_vertical_fails_loudly(tmp_path: Path, cfg, label_map) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    tab = np.loadtxt(folder / "RAW_ACCELEROMETERS.txt")
    tab[:, [2, 5]] = 1.0  # X reads 1 G at rest: gravity is in the file
    _save(folder / "RAW_ACCELEROMETERS.txt", tab)
    with pytest.raises(U.UAHFormatError, match="gravity"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)
    cfg["acceleration"]["gravity_in_vertical"] = True
    trip = U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)
    assert trip.frame["m_acc_vert"].to_numpy() == pytest.approx(0.0, abs=1e-5)


def test_speed_out_of_range_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    tab = np.loadtxt(folder / "RAW_GPS.txt")
    tab[:, 1] *= 10.0
    _save(folder / "RAW_GPS.txt", tab)
    with pytest.raises(U.UAHFormatError, match="km/h"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_missing_required_file_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    (folder / "RAW_GPS.txt").unlink()
    with pytest.raises(U.UAHFormatError, match="RAW_GPS"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_ambiguous_file_prefix_fails_loudly(tmp_path: Path, cfg: dict, label_map: dict) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    _save(folder / "RAW_GPS_copy.txt", _gps_table())
    with pytest.raises(U.UAHFormatError, match="more than one"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_unknown_event_type_code_fails_loudly(tmp_path: Path, cfg, label_map) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    _save(folder / "EVENTS_INERTIAL.txt", _events_table([(20.5, 7, 1)]))
    with pytest.raises(U.UAHFormatError, match="type code 7"):
        U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)


def test_missing_lane_file_gives_declared_nan_lane_columns(tmp_path, cfg, label_map) -> None:
    folder = make_route(tmp_path, "20151111123124-25km-D1-NORMAL-MOTORWAY")
    (folder / "PROC_LANE_DETECTION.txt").unlink()
    trip = U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)
    assert {"m_lane_offset", "m_lateral_drift_5s"} <= set(trip.meta[S.META_ALL_NAN_KEY])


# --- whole dataset, validator, script ---------------------------------------------------------
def test_process_dataset_passes_the_contract_validator(tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed" / "uah"
    result = U.process_dataset(tree, out)
    assert not result.failures and len(result.trips) == 4
    report = validate_processed_dir(out)
    assert report.ok, report.errors
    assert report.n_trips == 4
    manifest = pd.read_csv(out / "manifest.csv", dtype={"driver_id": str})
    assert list(manifest.columns) == list(S.MANIFEST_COLUMNS)
    assert sorted(manifest["trip_id"]) == sorted(v[1] for v in ROUTES.values())
    assert set(manifest["dataset"]) == {"uah"} and set(manifest["pairing"]) == {"motion_only"}
    assert manifest["split"].isna().all()  # splits are assigned by the splits script
    events = json.loads((out / "uah_d02_20151121_01_events.json").read_text(encoding="utf-8"))
    assert len(events) == 2


def test_process_dataset_refuses_stale_output(tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    U.process_dataset(tree, out)
    with pytest.raises(FileExistsError):
        U.process_dataset(tree, out)
    assert not U.process_dataset(tree, out, overwrite=True).failures


def test_keep_going_collects_failures(tree: Path, tmp_path: Path) -> None:
    bad = next(tree.rglob("20151120160000*"))
    _save(bad / "RAW_ACCELEROMETERS.txt", _acc_table(10))
    with pytest.raises(U.UAHFormatError):
        U.process_dataset(tree, tmp_path / "strict")
    result = U.process_dataset(tree, tmp_path / "lenient", keep_going=True)
    assert len(result.trips) == 3 and [f[0].name for f in result.failures] == [bad.name]
    assert validate_processed_dir(tmp_path / "lenient").ok


def test_summary_per_behaviour(tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    U.process_dataset(tree, out)
    summary = U.summarize_by_behaviour(out)
    assert summary.loc["aggressive", "trips"] == 2 and summary.loc["normal", "trips"] == 2
    assert summary.loc["aggressive", "duration_min"] == pytest.approx(2.0)
    assert summary.loc["aggressive", "harsh_brake"] == 2 and summary.loc["normal", "harsh_brake"]
    assert summary.loc["aggressive", "sharp_turn"] == 2 and summary.loc["normal", "sharp_turn"] == 0
    assert summary.loc["aggressive", "events"] > summary.loc["normal", "events"]


def test_process_uah_script(tree: Path, tmp_path: Path, capsys) -> None:
    spec = importlib.util.spec_from_file_location(
        "process_uah", repo_root() / "scripts" / "process_uah.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    assert module.main([str(tree), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "validator: OK" in text and "aggressive" in text and "UNVERIFIED" in text
    assert module.main([str(tmp_path / "nowhere"), "--out", str(out)]) == 2


# --- the additive helpers in features/motion.py ---------------------------------------------
def test_vehicle_frame_alignment_is_identity() -> None:
    a = F.vehicle_frame_alignment(gravity_included=False)
    assert a.ok and np.array_equal(a.R, np.eye(3)) and np.array_equal(a.g_dev, np.zeros(3))
    b = F.vehicle_frame_alignment(gravity_included=True)
    assert b.g_dev == pytest.approx([0.0, 0.0, G]) and b.up_dev == pytest.approx([0, 0, 1])


def test_resample_last_is_causal() -> None:
    t = np.array([0.0, 0.05, 0.1, 0.15, 0.31])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # step 0 = (-0.1, 0], step 1 = (0, 0.1], step 2 = (0.1, 0.2], step 3 = (0.2, 0.3]
    out = F.resample_last(t, x, 4)
    assert out[:3] == pytest.approx([1.0, 3.0, 4.0]) and np.isnan(out[3])
    x[2] = np.nan  # step 1's latest sample is NaN: skipped by default, kept with keep_nan
    assert F.resample_last(t, x, 4)[1] == pytest.approx(2.0)
    assert np.isnan(F.resample_last(t, x, 4, keep_nan=True)[1])
