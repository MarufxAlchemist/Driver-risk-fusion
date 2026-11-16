"""Own-data pipeline (P1-D1) on a fully synthetic recording whose clocks are known.

The recording: a phone logs IMU (50 Hz, ns-epoch timestamps) and GPS (1 Hz) on the phone clock.
Stationary 0-30 s with a double tap at 10.013 s, accelerate 30-40 s, cruise with a gentle left
turn, brake hard 85-89 s, stationary to the end with a second double tap at 140.013 s. The camera
runs on its own clock: phone = 0.73 s + video * (1 + R), where R gives 40 ms of drift between the
two sync events. Vision features carry markers (an eye closure, a no-face and a low-confidence
stretch) placed at known phone times, so the end-to-end alignment can be measured on the output.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driverisk.data import own
from driverisk.data import schema as S
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.paths import repo_root

G = 9.80665
FINE = 0.02  # 50 Hz IMU
IMU_END = 160.0
VIDEO_END = 150.0
OFFSET = 0.73  # phone time of video t = 0
TAPS = (10.013, 10.513, 140.013, 140.513)  # true tap times on the phone clock
R = 0.04 / (TAPS[2] - TAPS[0])  # 40 ms of drift between the start and the end sync event
EPOCH_NS = 1_764_590_400_000_000_000  # 2025-12-01 12:00 UTC, an arbitrary epoch origin
TRIP = "own_d01_20261201_01"
CLOSURE_PHONE = (120.0, 121.5)
NOFACE_PHONE = (50.0, 52.0)
LOWCONF_PHONE = (70.0, 71.0)
BRAKE_PHONE = 85.0


def video_to_phone(tv, rate: float = R):
    return OFFSET + np.asarray(tv, dtype=np.float64) * (1.0 + rate)


def phone_to_video(tp, rate: float = R):
    return (np.asarray(tp, dtype=np.float64) - OFFSET) / (1.0 + rate)


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


M_TRUE = _rot("z", 30) @ _rot("x", 70) @ _rot("y", -20)  # vehicle frame -> phone frame


def make_phone(taps=TAPS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IMU and GPS logs as a logging app might export them (one CSV each, shared clock)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, IMU_END, FINE)
    a_long = np.zeros_like(t)
    a_long[(t >= 30) & (t < 40)] = 1.5
    a_long[(t >= BRAKE_PHONE) & (t < BRAKE_PHONE + 4)] = -3.75
    v = np.cumsum(a_long) * FINE
    yaw_rate = np.where((t >= 55) & (t < 60), 8.0, 0.0)
    a_lat = v * np.radians(yaw_rate)
    f_veh = np.stack([a_long, a_lat, np.full_like(t, G)], axis=1)
    acc = f_veh @ M_TRUE.T + rng.normal(0, 0.05, f_veh.shape)
    for tap in taps:  # a firm tap: two samples of a sharp spike along the phone's z axis
        i = int(np.ceil(tap / FINE - 1e-9))
        acc[i : i + 2, 2] += 15.0
    gyro = np.stack([np.zeros_like(t), np.zeros_like(t), yaw_rate], axis=1) @ M_TRUE.T
    imu = pd.DataFrame(
        {
            "timestamp": EPOCH_NS + np.round(t * 1e9).astype(np.int64),
            "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2],
            "gx": np.radians(gyro[:, 0]), "gy": np.radians(gyro[:, 1]),
            "gz": np.radians(gyro[:, 2]),
        }
    )  # fmt: skip
    t_gps = np.arange(1.0, IMU_END, 1.0)
    speed = np.clip(np.interp(t_gps, t, v) + rng.normal(0, 0.05, len(t_gps)), 0, None)
    heading = (5.0 - np.interp(t_gps, t, np.cumsum(yaw_rate) * FINE)) % 360.0
    gps = pd.DataFrame(
        {
            "timestamp": EPOCH_NS + np.round(t_gps * 1e9).astype(np.int64),
            "speed": speed,
            "bearing": heading,
        }
    )
    return imu, gps


def _inside(tv: np.ndarray, span: tuple[float, float], rate: float = R) -> np.ndarray:
    p = video_to_phone(tv, rate)
    return (p >= span[0]) & (p <= span[1])


def make_vision_features(rate: float = R) -> pd.DataFrame:
    """A 10 Hz vision-features table on the video clock, as scripts/webcam_features.py writes."""
    tv = np.round(np.arange(0, round(VIDEO_END / S.STEP_S) + 1) * S.STEP_S, 6)
    n = len(tv)
    cols = {name: np.zeros(n) for name in S.VISION_NAMES}
    cols.update(ear_left=np.full(n, 0.3), ear_right=np.full(n, 0.3), ear_mean=np.full(n, 0.3))
    cols["mar"] = np.full(n, 0.2)
    cols["face_conf"] = np.full(n, 0.95)
    cols["face_visible"] = np.ones(n)
    closed = _inside(tv, CLOSURE_PHONE, rate)
    for name in ("ear_left", "ear_right", "ear_mean"):
        cols[name][closed] = 0.1
    cols["eye_closed"][closed] = 1.0
    for name in ("perclos_30s", "perclos_60s"):
        cols[name][:100] = np.nan
    cols["face_conf"][_inside(tv, LOWCONF_PHONE, rate)] = 0.3
    none = _inside(tv, NOFACE_PHONE, rate)
    for name in S.VISION_NAMES:
        if name not in ("face_conf", "face_visible"):
            cols[name][none] = np.nan
    cols["face_conf"][none], cols["face_visible"][none] = 0.0, 0.0
    frame = pd.DataFrame({S.COL_T: tv.astype(np.float32)})
    for name in S.VISION_NAMES:
        frame[S.VISION_PREFIX + name] = cols[name].astype(np.float32)
    frame[S.COL_MASK_V] = (frame["v_face_visible"] == 1) & (frame["v_face_conf"] >= 0.5)
    return frame


VIDEO_SYNC = tuple(round(float(phone_to_video(TAPS[i])), 3) for i in (0, 2))


def trip_sheet(**over) -> dict:
    row = {
        "trip_id": TRIP, "driver_id": "d01", "date": "2026-12-01", "start_time": "10:00",
        "route": "R1", "road_type": "secondary", "weather": "clear", "traffic": "light",
        "lighting": "day", "phone_model": "phoneX", "app_version": "1.0", "mount_position":
        "dash", "glasses": "no", "trip_type": "normal", "planned_events": "brake",
        "observed_events": "one hard brake", "sync_start_ok": "yes", "sync_end_ok": "yes",
        "notes": "synthetic", "recorder": "p01", "video_sync_start_s": VIDEO_SYNC[0],
        "video_sync_end_s": VIDEO_SYNC[1],
    }  # fmt: skip
    row.update(over)
    return row


ANNOTATIONS = pd.DataFrame(
    {
        "trip_id": [TRIP, TRIP, "own_d02_20261201_01"],
        "type": ["micro_sleep", "distraction_episode", "micro_sleep"],
        "onset_s": [119.2, 30.0, 1.0],
        "offset_s": [120.7, 33.0, 2.0],
        "uncertain": ["false", "true", "false"],
    }
)
LABELS = pd.DataFrame(
    {
        "trip_id": [TRIP, TRIP, TRIP],
        "class": ["normal", "aggressive", "drowsy"],
        "onset_s": [0.0, 83.0, 85.0],
        "offset_s": [60.0, 89.0, 86.0],
    }
)


@pytest.fixture(scope="module")
def phone() -> own.PhoneData:
    imu, gps = make_phone()
    return own.phone_data_from_frames(imu, gps, gyro_cols=("gx", "gy", "gz"), gyro_unit="rad/s")


@pytest.fixture(scope="module")
def trip(phone: own.PhoneData) -> own.OwnTrip:
    return own.process_own_trip(
        TRIP, phone, own.VisionFeatures(make_vision_features()), trip_sheet(),
        annotations=ANNOTATIONS, labels=LABELS, source_version="synthetic",
    )  # fmt: skip


def _phone_time(trip: own.OwnTrip, tau: np.ndarray) -> np.ndarray:
    return np.asarray(tau, dtype=np.float64) + trip.meta["trip_start_phone_s"]


def _mapping_error(trip: own.OwnTrip, rate: float = R) -> float:
    """Largest error of the estimated video -> phone map over the whole video, in seconds."""
    tv = np.linspace(0.0, VIDEO_END, 301)
    sync = trip.meta["sync"]
    est = sync["intercept_s"] + sync["scale"] * tv
    return float(np.abs(est - video_to_phone(tv, rate)).max())


# --- names (fix F11) ------------------------------------------------------------------------
def test_trip_id_and_raw_file_names() -> None:
    tid = own.parse_trip_id(TRIP)
    assert (tid.driver_id, tid.date, tid.nn, tid.trip_id) == ("d01", "20261201", "01", TRIP)
    assert own.raw_file_name(TRIP, "imu") == "own_d01_20261201_01_imu"
    assert own.raw_file_name(TRIP, "vision", ".parquet") == "own_d01_20261201_01_vision.parquet"
    for bad in ("uah_d01_20261201_01", "d01_20261201_01", "own_d1_20261201_01", "own_d01_2026_01"):
        with pytest.raises(ValueError, match="own_<driverid>_<yyyymmdd>_<nn>"):
            own.parse_trip_id(bad)
    with pytest.raises(ValueError, match="kind"):
        own.raw_file_name(TRIP, "audio")


# --- clock sync -----------------------------------------------------------------------------
def test_small_drift_uses_one_offset_and_meets_the_residual_target() -> None:
    imu = [TAPS[0], TAPS[2]]
    video = [float(phone_to_video(TAPS[0])), float(phone_to_video(TAPS[2]))]
    sync = own.estimate_clock_sync(imu, video)
    assert sync.offset_start_s == pytest.approx(TAPS[0] - video[0], abs=1e-9)
    assert sync.drift_s == pytest.approx(0.04, abs=1e-3)
    assert not sync.drift_corrected  # contract 8.4: correct only above 50 ms
    assert sync.scale == 1.0
    assert sync.residual_s == pytest.approx(0.02, abs=1e-3) and sync.residual_s < 0.1
    errors = sync.video_to_phone(np.array(video)) - np.array(imu)
    assert np.abs(errors).max() <= sync.residual_s + 1e-9


def test_large_drift_is_corrected_linearly() -> None:
    rate = 0.3 / 130.0  # 300 ms over the trip
    video = np.array([5.0, 135.0])
    imu = 2.0 + video * (1 + rate)
    sync = own.estimate_clock_sync(imu.tolist(), video.tolist())
    assert sync.drift_corrected and sync.drift_s == pytest.approx(0.3, abs=1e-6)
    assert sync.video_to_phone(video) == pytest.approx(imu, abs=1e-9)
    assert sync.video_to_phone(np.array([70.0]))[0] == pytest.approx(2.0 + 70 * (1 + rate))
    assert sync.phone_to_video(sync.video_to_phone(np.array([33.3])))[0] == pytest.approx(33.3)
    assert sync.residual_s == 0.0


def test_missing_or_inconsistent_sync_events_raise_clear_errors() -> None:
    with pytest.raises(own.SyncError, match="start sync event"):
        own.estimate_clock_sync([], [])
    with pytest.raises(own.SyncError, match="end sync event"):
        own.estimate_clock_sync([10.0], [9.0])
    with pytest.raises(own.SyncError, match="same number"):
        own.estimate_clock_sync([10.0, 140.0], [9.0])
    with pytest.raises(own.SyncError, match="do not match"):
        own.estimate_clock_sync([10.0, 140.0], [9.0, 100.0])  # 30% clock-rate error
    with pytest.raises(own.SyncError, match="order"):
        own.estimate_clock_sync([140.0, 10.0], [139.0, 9.0])
    one = own.estimate_clock_sync([10.0], [9.0], require_end=False)
    assert one.offset_start_s == 1.0 and np.isnan(one.drift_s) and not one.drift_corrected
    assert np.isnan(one.residual_s)


def test_imu_sync_events_are_found_while_stationary(phone: own.PhoneData) -> None:
    found = own.find_imu_sync_events(phone.t_acc, phone.acc, phone.t_gps, phone.speed_mps)
    assert len(found) == 2
    assert found[0] == pytest.approx(TAPS[0], abs=FINE)
    assert found[1] == pytest.approx(TAPS[2], abs=FINE)


def test_imu_sync_errors_instead_of_guessing() -> None:
    imu, gps = make_phone(taps=(20.0, 20.5, *TAPS))  # a second stationary tap pair at the start
    p = own.phone_data_from_frames(imu, gps)
    with pytest.raises(own.SyncError, match="2 tap pairs"):
        own.find_imu_sync_events(p.t_acc, p.acc, p.t_gps, p.speed_mps)
    imu, gps = make_phone(taps=TAPS[2:])  # no start taps
    p = own.phone_data_from_frames(imu, gps)
    with pytest.raises(own.SyncError, match="no tap pair"):
        own.find_imu_sync_events(p.t_acc, p.acc, p.t_gps, p.speed_mps)
    imu, gps = make_phone(taps=TAPS[:2])  # no end taps
    p = own.phone_data_from_frames(imu, gps)
    with pytest.raises(own.SyncError, match="end"):
        own.find_imu_sync_events(p.t_acc, p.acc, p.t_gps, p.speed_mps)
    found = own.find_imu_sync_events(p.t_acc, p.acc, p.t_gps, p.speed_mps, require_end=False)
    assert found == [pytest.approx(TAPS[0], abs=FINE)]


# --- phone logs -----------------------------------------------------------------------------
def test_phone_logs_share_one_clock_and_units(phone: own.PhoneData) -> None:
    assert phone.time_unit == "ns epoch"
    assert phone.t_acc[0] == 0.0 and phone.t_gps[0] == pytest.approx(1.0, abs=1e-6)
    assert phone.imu_rate_hz == pytest.approx(50.0, rel=1e-3)
    assert phone.gps_rate_hz == pytest.approx(1.0, rel=1e-3)
    rate = np.linalg.norm(phone.gyro_dps, axis=1).max()
    assert rate == pytest.approx(8.0, rel=1e-6)  # rad/s converted to deg/s
    assert np.median(np.linalg.norm(phone.acc, axis=1)) == pytest.approx(G, abs=0.1)


def test_phone_log_unit_problems_are_reported_not_guessed() -> None:
    imu, gps = make_phone()
    with pytest.raises(ValueError, match="gyro_unit"):
        own.phone_data_from_frames(imu, gps, gyro_cols=("gx", "gy", "gz"))
    in_g = imu.assign(x=imu["x"] / G, y=imu["y"] / G, z=imu["z"] / G)
    with pytest.raises(ValueError, match="acc_unit"):
        own.phone_data_from_frames(in_g, gps)
    p = own.phone_data_from_frames(in_g, gps, acc_unit="g")
    assert np.median(np.linalg.norm(p.acc, axis=1)) == pytest.approx(G, abs=0.1)
    linear = imu.assign(z=imu["z"] - G)  # roughly gravity-free: not supported
    linear = linear.assign(x=linear["x"] * 0, y=linear["y"] * 0, z=linear["z"] * 0 + 0.1)
    with pytest.raises(ValueError, match="gravity"):
        own.phone_data_from_frames(linear, gps)
    backwards = imu.iloc[[0, 2, 1, *range(3, len(imu))]].reset_index(drop=True)
    with pytest.raises(ValueError, match="backwards"):
        own.phone_data_from_frames(backwards, gps)
    sec_gps = gps.assign(timestamp=gps["timestamp"] / 1e9)
    with pytest.raises(ValueError, match="unit"):
        own.phone_data_from_frames(imu, sec_gps)
    with_bad_fix = gps.copy()
    with_bad_fix.loc[5, "speed"] = -1.0
    p = own.phone_data_from_frames(imu, with_bad_fix)
    assert len(p.t_gps) == len(gps) - 1 and (p.speed_mps >= 0).all()


# --- the whole trip -------------------------------------------------------------------------
def test_recovered_sync_is_within_target(trip: own.OwnTrip) -> None:
    sync = trip.meta["sync"]
    true_start = TAPS[0] - float(phone_to_video(TAPS[0]))
    assert abs(sync["offset_start_s"] - true_start) < 0.1
    assert abs(sync["offset_start_s"] - true_start) <= FINE  # one IMU sample
    assert sync["drift_s"] == pytest.approx(0.04, abs=0.01)
    assert sync["residual_s"] < 0.1
    assert _mapping_error(trip) < 0.1  # true residual over the whole video, not only at taps
    assert sync["video_sync_source"] == "trip_sheet" and sync["imu_sync_source"] == "detected"


def test_vision_markers_land_at_the_right_trip_time(trip: own.OwnTrip) -> None:
    df = trip.frame
    tau = df[S.COL_T].to_numpy(np.float64)
    closed = df["v_eye_closed"].to_numpy(np.float64) == 1.0
    onset = _phone_time(trip, tau[closed][0])
    offset = _phone_time(trip, tau[closed][-1])
    assert abs(onset - CLOSURE_PHONE[0]) <= S.STEP_S + 1e-6
    assert abs(offset - CLOSURE_PHONE[1]) <= S.STEP_S + 1e-6
    noface = df["v_face_visible"].to_numpy(np.float64) == 0.0
    assert abs(_phone_time(trip, tau[noface][0]) - NOFACE_PHONE[0]) <= S.STEP_S + 1e-6
    lowconf = (df["v_face_visible"] == 1) & (df["v_face_conf"] < 0.5)
    assert abs(_phone_time(trip, tau[lowconf.to_numpy()][0]) - LOWCONF_PHONE[0]) <= 0.1 + 1e-6


def test_large_drift_is_corrected_end_to_end(phone: own.PhoneData) -> None:
    rate = 0.4 / (TAPS[2] - TAPS[0])  # 400 ms over the trip: without correction, markers miss
    sync_times = [round(float(phone_to_video(TAPS[i], rate)), 3) for i in (0, 2)]
    vision = own.VisionFeatures(make_vision_features(rate))
    trip = own.process_own_trip(TRIP, phone, vision, trip_sheet(), video_sync_s=sync_times)
    sync = trip.meta["sync"]
    assert sync["drift_corrected"] and sync["drift_s"] == pytest.approx(0.4, abs=0.01)
    assert sync["scale"] == pytest.approx(1 + rate, abs=1e-4)
    assert _mapping_error(trip, rate) < 0.02  # residual over the whole video, target 0.1 s
    df = trip.frame
    tau = df[S.COL_T].to_numpy(np.float64)
    closed = df["v_eye_closed"].to_numpy(np.float64) == 1.0
    # the marker sits on the 0.1 s video grid and is resampled by nearest neighbour: 1.5 steps
    for got, want in ((tau[closed][0], CLOSURE_PHONE[0]), (tau[closed][-1], CLOSURE_PHONE[1])):
        assert abs(_phone_time(trip, got) - want) <= 1.5 * S.STEP_S + 0.02


def test_no_face_and_mask_rules_hold(trip: own.OwnTrip) -> None:
    df = trip.frame
    none = df["v_face_visible"] == 0
    assert none.sum() >= 19
    others = [c for c in S.VISION_COLUMNS if c not in ("v_face_conf", "v_face_visible")]
    assert df.loc[none, others].isna().all().all()
    assert (df.loc[none, "v_face_conf"] == 0).all()
    expected = (df["v_face_visible"] == 1) & (df["v_face_conf"] >= S.FACE_CONF_MIN)
    assert (df[S.COL_MASK_V] == expected).all()
    assert not df.loc[(df["v_face_visible"] == 1) & (df["v_face_conf"] < 0.5), "mask_v"].any()
    assert df[S.COL_MASK_M].mean() > 0.99
    for flag in ("v_blink", "v_eye_closed", "v_off_road_gaze", "v_face_visible", "m_gps_valid"):
        vals = df[flag].dropna().to_numpy()
        assert np.isin(vals, (0.0, 1.0)).all(), flag


def test_frame_has_the_contract_columns_and_grid(trip: own.OwnTrip) -> None:
    df = trip.frame
    assert list(df.columns) == list(S.PARQUET_COLUMNS)
    t = df[S.COL_T].to_numpy(np.float64)
    assert t[0] == 0.0 and np.allclose(np.diff(t), S.STEP_S, atol=S.T_STEP_TOLERANCE_S)
    # the trip covers the overlap of both streams
    assert trip.meta["trip_start_phone_s"] == pytest.approx(OFFSET + 0.02, abs=0.03)
    end_phone = _phone_time(trip, t[-1])
    assert end_phone <= float(video_to_phone(VIDEO_END)) + 0.05 + 1e-6
    assert end_phone > float(video_to_phone(VIDEO_END)) - S.STEP_S - 0.05
    assert set(df[S.COL_CONTRACT]) == {S.CONTRACT_VERSION}


def test_labels_and_events(trip: own.OwnTrip) -> None:
    y = trip.frame.set_index(trip.frame[S.COL_T].round(1))[S.COL_Y_CLS]
    assert y.loc[10.0] == S.CLASS_TO_ID["normal"]
    assert y.loc[84.0] == S.CLASS_TO_ID["aggressive"]
    assert y.loc[85.5] == S.UNLABELED  # overlapping different classes
    assert y.loc[70.0] == S.UNLABELED  # not labelled
    detected = [e for e in trip.events if e["source"] == "own_detector"]
    brakes = [e for e in detected if e["type"] == "harsh_brake"]
    assert len(brakes) == 1
    assert _phone_time(trip, brakes[0]["onset_s"]) == pytest.approx(BRAKE_PHONE, abs=0.3)
    annotated = [e for e in trip.events if e["source"] == "own_annotation"]
    assert [(e["type"], e["uncertain"]) for e in annotated] == [
        ("distraction_episode", True),
        ("micro_sleep", False),
    ]
    assert trip.meta["events_annotated"] is True


def test_meta_and_manifest_row(trip: own.OwnTrip) -> None:
    meta = trip.meta
    for key in (*S.META_REQUIRED_KEYS, "all_nan_columns", "ear_threshold", "ear_threshold_source"):
        assert key in meta, key
    assert meta["trip_id"] == TRIP and meta["contract_version"] == S.CONTRACT_VERSION
    assert meta["ear_threshold_source"] == "global_default"
    assert "m_lane_offset" in meta["all_nan_columns"]
    assert meta["sampling_rates"]["imu_hz"] == pytest.approx(50.0, rel=1e-3)
    assert meta["motion_alignment"]["ok"] is True
    assert meta["quality"] == "ok", meta["quality_checks"]
    assert "recorder" not in json.dumps(meta) and "R1" not in json.dumps(meta)
    row = trip.manifest_row
    assert list(row) == list(S.MANIFEST_COLUMNS)
    assert row["trip_id"] == TRIP and row["dataset"] == "own" and row["driver_id"] == "d01"
    assert row["pairing"] == "paired" and row["n_steps"] == len(trip.frame)
    assert row["duration_s"] == pytest.approx(len(trip.frame) * S.STEP_S)
    assert row["behavior_tag"] == "normal" and row["road_type"] == "secondary"
    assert row["source_version"] == "synthetic"


def test_written_trip_passes_the_validator(trip: own.OwnTrip, tmp_path: Path) -> None:
    paths = own.write_own_trip(trip, tmp_path)
    assert sorted(p.name for p in paths.values()) == sorted(
        [f"{TRIP}.parquet", f"{TRIP}_events.json", f"{TRIP}_meta.json", "manifest.csv"]
    )
    report = validate_processed_dir(tmp_path)
    assert report.ok, report.errors
    assert report.n_trips == 1
    own.write_own_trip(trip, tmp_path)  # writing again replaces the manifest row
    manifest = pd.read_csv(tmp_path / "manifest.csv", dtype={"driver_id": str})
    assert manifest["trip_id"].tolist() == [TRIP]
    assert manifest["driver_id"].tolist() == ["d01"]


# --- inputs and errors ----------------------------------------------------------------------
def test_raw_frames_input_is_synced_and_validates(phone: own.PhoneData, tmp_path: Path) -> None:
    tv = np.arange(0, VIDEO_END, 1 / 30.0)
    lm = np.full((len(tv), 478, 3), 0.5)
    lm[:, :, 2] = 0.0
    lm[_inside(tv, NOFACE_PHONE)] = np.nan
    vision = own.VisionFrames(tv, lm, None, image_size=(640, 480))
    trip = own.process_own_trip(TRIP, phone, vision, trip_sheet())
    df = trip.frame
    tau = df[S.COL_T].to_numpy(np.float64)
    noface = df["v_face_visible"].to_numpy(np.float64) == 0.0
    # a step (t - 0.1, t] is no-face only when none of its frames has a face, and it is stamped
    # with its end, so the first such step ends up to two steps after the true onset
    late = _phone_time(trip, tau[noface][0]) - NOFACE_PHONE[0]
    assert -0.02 <= late <= 2 * S.STEP_S + 0.02
    assert abs(_phone_time(trip, tau[noface][-1]) - NOFACE_PHONE[1]) <= S.STEP_S + 1e-6
    assert "v_ear_mean" in trip.meta["all_nan_columns"]  # synthetic face without eye geometry
    assert trip.meta["sampling_rates"]["video_fps"] == pytest.approx(30.0, rel=1e-3)
    assert trip.meta["events_annotated"] is False
    own.write_own_trip(trip, tmp_path)
    report = validate_processed_dir(tmp_path)
    assert report.ok, report.errors


def test_missing_video_sync_times_raise(phone: own.PhoneData) -> None:
    sheet = trip_sheet()
    del sheet["video_sync_start_s"], sheet["video_sync_end_s"]
    vision = own.VisionFeatures(make_vision_features())
    with pytest.raises(own.SyncError, match="video_sync_start_s"):
        own.process_own_trip(TRIP, phone, vision, sheet)
    with pytest.raises(own.SyncError, match="video_sync_end_s"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(video_sync_end_s=""))
    trip = own.process_own_trip(TRIP, phone, vision, sheet, video_sync_s=list(VIDEO_SYNC))
    assert trip.meta["sync"]["video_sync_source"] == "argument"


def test_trip_sheet_and_annotation_errors(phone: own.PhoneData) -> None:
    vision = own.VisionFeatures(make_vision_features())
    with pytest.raises(ValueError, match="driver_id"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(driver_id="d02"))
    with pytest.raises(ValueError, match="date"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(date="2026-12-02"))
    late = ANNOTATIONS.assign(offset_s=[400.0, 33.0, 2.0])
    with pytest.raises(ValueError, match="inside the trip"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(), annotations=late)
    bad_type = ANNOTATIONS.assign(type=["nap", "micro_sleep", "micro_sleep"])
    with pytest.raises(ValueError, match="nap"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(), annotations=bad_type)
    bad_class = LABELS.assign(**{"class": ["normal", "sleepy", "drowsy"]})
    with pytest.raises(ValueError, match="sleepy"):
        own.process_own_trip(TRIP, phone, vision, trip_sheet(), labels=bad_class)


def test_explicit_imu_sync_times_override_detection(phone: own.PhoneData) -> None:
    vision = own.VisionFeatures(make_vision_features())
    trip = own.process_own_trip(TRIP, phone, vision, trip_sheet(), imu_sync_s=[TAPS[0], TAPS[2]])
    assert trip.meta["sync"]["imu_sync_source"] == "argument"
    assert trip.meta["sync"]["offset_start_s"] == pytest.approx(TAPS[0] - VIDEO_SYNC[0], abs=1e-9)


def test_read_trip_sheet(tmp_path: Path) -> None:
    path = tmp_path / "sheet.csv"
    pd.DataFrame([trip_sheet(), trip_sheet(trip_id="own_d01_20261201_02")]).to_csv(
        path, index=False
    )
    row = own.read_trip_sheet(path, TRIP)
    assert row["trip_id"] == TRIP and row["driver_id"] == "d01"
    assert float(row["video_sync_start_s"]) == VIDEO_SYNC[0]
    with pytest.raises(ValueError, match="no row"):
        own.read_trip_sheet(path, "own_d09_20261201_01")


# --- command line ---------------------------------------------------------------------------
def test_cli_processes_a_raw_folder(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    imu, gps = make_phone()
    imu.to_csv(raw / own.raw_file_name(TRIP, "imu", ".csv"), index=False)
    gps.to_csv(raw / own.raw_file_name(TRIP, "gps", ".csv"), index=False)
    make_vision_features().to_parquet(raw / own.raw_file_name(TRIP, "vision", ".parquet"))
    pd.DataFrame([trip_sheet()]).to_csv(raw / "trip_sheet.csv", index=False)
    ANNOTATIONS.to_csv(raw / "annotations.csv", index=False)

    spec = importlib.util.spec_from_file_location(
        "process_own", repo_root() / "scripts" / "process_own.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(
        [
            TRIP, "--raw-dir", str(raw), "--trip-sheet", str(raw / "trip_sheet.csv"),
            "--out", str(out), "--gyro-cols", "gx", "gy", "gz", "--gyro-unit", "rad/s",
            "--annotations", str(raw / "annotations.csv"), "--source-version", "synthetic",
        ]
    )  # fmt: skip
    assert code == 0
    report = validate_processed_dir(out)
    assert report.ok, report.errors
    meta = json.loads((out / f"{TRIP}_meta.json").read_text(encoding="utf-8"))
    assert meta["events_annotated"] is True and meta["sync"]["drift_s"] == pytest.approx(
        0.04, abs=0.01
    )
