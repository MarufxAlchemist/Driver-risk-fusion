"""Motion features (P1-B2): checked against a synthetic drive whose truth is known.

The world: stationary 5 s, accelerate +1.5 m/s^2 for 10 s, cruise with a 5 s left turn at
10 deg/s, brake -3 m/s^2 until stopped, stationary again. The phone sits in an arbitrary
orientation, so recovering `acc_long` / `acc_lat` proves the gravity removal and the axis
alignment, and the sign conventions (x forward, y left, z up, left turn = positive yaw rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data.validate import validate_trip_frame
from driverisk.features import motion as F

G = 9.80665
FINE = 0.02  # 50 Hz raw IMU


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


M_TRUE = _rot("z", 30) @ _rot("x", 70) @ _rot("y", -20)  # vehicle frame -> phone frame


def _world(noise: float = 0.05, seed: int = 0, gps_gap: tuple[float, float] | None = None):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 50.0, FINE)
    a_long = np.zeros_like(t)
    a_long[(t >= 5) & (t < 15)] = 1.5
    a_long[(t >= 35) & (t < 40)] = -3.0
    v = np.cumsum(a_long) * FINE
    yaw_rate = np.where((t >= 20) & (t < 25), 10.0, 0.0)  # deg/s, left turn positive
    a_lat = v * np.radians(yaw_rate)
    f_veh = np.stack([a_long, a_lat, np.full_like(t, G)], axis=1)  # specific force sensed
    acc = f_veh @ M_TRUE.T + rng.normal(0, noise, f_veh.shape)
    gyro = np.stack([np.zeros_like(t), np.zeros_like(t), yaw_rate], axis=1) @ M_TRUE.T
    t_gps = np.arange(1.0, 50.0, 1.0)
    speed = np.interp(t_gps, t, v) + rng.normal(0, 0.05, len(t_gps))
    # compass heading is clockwise; start at 5 deg so the left turn wraps through 0/360
    heading = (5.0 - np.interp(t_gps, t, np.cumsum(yaw_rate) * FINE)) % 360.0
    if gps_gap:
        keep = (t_gps < gps_gap[0]) | (t_gps > gps_gap[1])
        t_gps, speed, heading = t_gps[keep], speed[keep], heading[keep]
    return {
        "t_acc": t, "acc": acc, "gyro": gyro, "t_gps": t_gps, "speed": np.clip(speed, 0, None),
        "heading": heading, "a_long": a_long, "a_lat": a_lat, "yaw_rate": yaw_rate, "v": v,
    }  # fmt: skip


def _run(w: dict, **kw) -> F.MotionResult:
    args = {"gyro_dps": w["gyro"], "heading_deg": w["heading"], **kw}
    return F.extract_motion_features(w["t_acc"], w["acc"], w["t_gps"], w["speed"], **args)


def _col(res: F.MotionResult, name: str) -> np.ndarray:
    return res.frame[S.MOTION_PREFIX + name].to_numpy(np.float64)


def _truth(w: dict, key: str, n: int) -> np.ndarray:
    """Truth averaged into the same 0.1 s steps the extractor uses: (t - 0.1, t]."""
    per_step = round(0.1 / FINE)
    body = w[key][1:]  # step 0 holds only the sample at t = 0; step k covers samples 5k-4 .. 5k
    vals = body[: (len(body) // per_step) * per_step].reshape(-1, per_step).mean(axis=1)
    return np.r_[np.nan, vals][:n]


@pytest.fixture(scope="module")
def world() -> dict:
    return _world()


@pytest.fixture(scope="module")
def result(world: dict) -> F.MotionResult:
    return _run(world)


# --- structure ------------------------------------------------------------------------------
def test_columns_and_grid(result: F.MotionResult) -> None:
    df = result.frame
    assert list(df.columns) == [S.COL_T, *S.MOTION_COLUMNS, S.COL_MASK_M]
    assert df[S.COL_T].iloc[:3].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert np.allclose(np.diff(df[S.COL_T].to_numpy(np.float64)), 0.1, atol=S.T_STEP_TOLERANCE_S)
    assert all(str(df[c].dtype) == "float32" for c in (S.COL_T, *S.MOTION_COLUMNS))
    assert str(df[S.COL_MASK_M].dtype) == "bool"


# --- gravity removal and alignment ---------------------------------------------------------
def test_alignment_is_found_and_reported(result: F.MotionResult) -> None:
    assert result.alignment.ok and result.alignment.up_source == "stationary"
    # the recovered up axis, expressed in the phone frame, is the true one
    assert result.alignment.up_dev @ (M_TRUE @ np.array([0, 0, 1.0])) == pytest.approx(
        1.0, abs=2e-3
    )


def test_longitudinal_and_lateral_acceleration_are_recovered(
    world: dict, result: F.MotionResult
) -> None:
    n = len(result.frame)
    for name, key in (("acc_long", "a_long"), ("acc_lat", "a_lat")):
        est, truth = _col(result, name)[1:], _truth(world, key, n)[1:]
        ok = ~np.isnan(est) & ~np.isnan(truth)
        assert np.corrcoef(est[ok], truth[ok])[0, 1] > 0.97, name
        assert np.sqrt(np.mean((est[ok] - truth[ok]) ** 2)) < 0.25, name
    # signs: accelerating is positive along x, a left turn is positive along y
    acc_long, acc_lat = _col(result, "acc_long"), _col(result, "acc_lat")
    assert acc_long[80:140].mean() > 1.2 and acc_long[360:390].mean() < -2.5
    assert acc_lat[210:240].mean() > 1.5


def test_vertical_acceleration_and_rest_state_are_near_zero(result: F.MotionResult) -> None:
    assert np.nanmean(np.abs(_col(result, "acc_vert"))) < 0.15
    rest = slice(5, 45)  # before the first push at 5 s
    for name in ("acc_long", "acc_lat", "acc_vert"):
        assert np.nanmax(np.abs(_col(result, name)[rest])) < 0.4, name


def test_attitude_is_the_phone_orientation(result: F.MotionResult) -> None:
    up_dev = M_TRUE @ np.array([0, 0, 1.0])
    roll = np.degrees(np.arctan2(up_dev[1], up_dev[2]))
    pitch = np.degrees(np.arctan2(-up_dev[0], np.hypot(up_dev[1], up_dev[2])))
    assert _col(result, "roll")[20:45].mean() == pytest.approx(roll, abs=1.5)
    assert _col(result, "pitch")[20:45].mean() == pytest.approx(pitch, abs=1.5)


def test_jerk_is_the_derivative_of_acceleration() -> None:
    t = np.arange(0.0, 30.0, FINE)
    a = np.clip(0.5 * (t - 10.0), 0, 1.5)  # ramps 0 -> 1.5 m/s^2 over 3 s: jerk 0.5 m/s^3
    v = np.cumsum(a) * FINE
    acc = np.stack([a, 0 * a, np.full_like(a, G)], axis=1) @ M_TRUE.T
    # a stop at the start and end makes the alignment solvable
    t_gps = np.arange(1.0, 30.0, 1.0)
    res = F.extract_motion_features(t, acc, t_gps, np.interp(t_gps, t, v))
    jerk = _col(res, "jerk_long")
    assert np.nanmean(jerk[112:128]) == pytest.approx(0.5, abs=0.12)
    assert np.nanmax(np.abs(jerk[:90])) < 0.3


# --- yaw rate and heading ------------------------------------------------------------------
def test_yaw_rate_from_gyro_is_left_positive(world: dict, result: F.MotionResult) -> None:
    yaw = _col(result, "yaw_rate")
    assert yaw[215:245].mean() == pytest.approx(10.0, abs=0.6)
    assert abs(yaw[100:180].mean()) < 0.5


def test_yaw_rate_from_unwrapped_yaw_angle_when_there_is_no_gyro(world: dict) -> None:
    t10 = np.arange(0.0, 50.0, 0.1)
    yaw_angle = ((np.cumsum(world["yaw_rate"][::5]) * 0.1) + 175.0 + 180.0) % 360.0 - 180.0  # wraps
    res = F.extract_motion_features(
        world["t_acc"], world["acc"], world["t_gps"], world["speed"], yaw_deg=(t10, yaw_angle)
    )
    assert _col(res, "yaw_rate")[215:245].mean() == pytest.approx(10.0, abs=0.8)


def test_yaw_rate_is_nan_without_gyro_or_yaw(world: dict) -> None:
    res = F.extract_motion_features(world["t_acc"], world["acc"], world["t_gps"], world["speed"])
    assert np.isnan(_col(res, "yaw_rate")).all()
    assert np.isnan(_col(res, "heading_rate")).all()  # no heading either


def test_heading_rate_follows_the_same_sign_as_yaw_rate(result: F.MotionResult) -> None:
    hr = _col(result, "heading_rate")
    assert np.nanmean(hr[215:260]) == pytest.approx(10.0, abs=2.5)  # fixes arrive at 1 Hz
    assert abs(np.nanmean(hr[100:180])) < 1.0  # compass wrap 359 -> 1 must not make a spike
    low_speed = _col(result, "speed") < S_MIN_HEADING
    assert np.isnan(hr[low_speed & ~np.isnan(_col(result, "speed"))]).all()


S_MIN_HEADING = F.MotionConfig().min_heading_speed_mps


# --- GPS -----------------------------------------------------------------------------------
def test_speed_is_held_from_the_latest_fix_and_gps_valid_tracks_age() -> None:
    w = _world(gps_gap=(20.0, 26.0))  # no fixes between 19 s and 27 s
    res = _run(w)
    speed, valid = _col(res, "speed"), _col(res, "gps_valid")
    assert valid[100] == 1.0 and valid[195] == 1.0  # fix at 19 s, 1.5 s old at 20.5 s
    assert valid[215] == 0.0 and np.isnan(speed[215])  # 2.1+ s without a fix
    assert valid[300] == 1.0
    assert speed[100] == pytest.approx(w["v"][int(10.0 / FINE)], abs=0.3)  # fix at exactly 10 s
    assert np.nanmax(speed) < 70


def test_speed_variance_is_causal_and_zero_at_constant_speed(result: F.MotionResult) -> None:
    var = _col(result, "speed_var_5s")
    assert np.isnan(var[0]) and var[250] >= 0
    cruise = _col(result, "speed")[170:190]
    assert np.nanmax(np.abs(cruise - cruise[0])) < 0.5
    assert var[199] < 1.0  # cruising: small
    assert var[140] > 2.0  # accelerating 1.5 m/s^2: 5 s of ramp has a variance of several (m/s)^2


# --- masks, lane, drift ---------------------------------------------------------------------
def test_mask_m_goes_false_when_the_imu_is_silent_for_more_than_0_2_s() -> None:
    w = _world()
    keep = (w["t_acc"] < 20.0) | (w["t_acc"] > 21.0)  # 1 s hole
    w2 = {**w, "t_acc": w["t_acc"][keep], "acc": w["acc"][keep], "gyro": w["gyro"][keep]}
    res = _run(w2)
    mask = res.frame[S.COL_MASK_M].to_numpy()
    assert mask[:195].all() and mask[215:].all()
    assert not mask[203:208].any()
    assert np.isnan(_col(res, "acc_long")[204])


def test_lane_offset_is_nan_unless_given_and_drift_uses_it() -> None:
    w = _world()
    plain = _run(w)
    assert np.isnan(_col(plain, "lane_offset")).all()
    n = len(plain.frame)
    t10 = np.arange(n) * 0.1
    lane = 0.5 * np.sin(2 * np.pi * t10 / 4.0)
    res = _run(w, lane_offset=(t10, lane))
    assert np.nanmax(np.abs(_col(res, "lane_offset") - lane)) < 1e-4
    # rolling std over 5 s of a 0.5 m sine = 0.5 / sqrt(2) once the window is full
    assert _col(res, "lateral_drift_5s")[300] == pytest.approx(0.5 / np.sqrt(2), abs=0.08)


def test_drift_proxy_without_lane_offset_is_finite_and_in_range(result: F.MotionResult) -> None:
    drift = _col(result, "lateral_drift_5s")
    assert np.isnan(drift[:49]).all() and np.isfinite(drift[60:]).all()
    assert 0.0 <= np.nanmin(drift) and np.nanmax(drift) <= 5.0


# --- contract and causality -----------------------------------------------------------------
def test_output_passes_the_contract_validator(result: F.MotionResult) -> None:
    df = result.frame.copy()
    n = len(df)
    for c in S.VISION_COLUMNS:
        df[c] = np.float32("nan")
    df[S.VISION_PREFIX + "face_visible"] = np.float32(0.0)
    df[S.VISION_PREFIX + "face_conf"] = np.float32(0.0)
    df[S.COL_MASK_V] = np.zeros(n, dtype=bool)
    df[S.COL_Y_CLS] = np.full(n, -1, dtype=np.int8)
    df[S.COL_CONTRACT] = S.CONTRACT_VERSION
    df = df[list(S.PARQUET_COLUMNS)]
    # no-face rule (contract 4.1): face_visible and face_conf are 0, the other 14 are NaN
    declared = [c for c in S.VISION_COLUMNS if c not in ("v_face_visible", "v_face_conf")]
    meta = {S.META_ALL_NAN_KEY: [*result.all_nan_columns, *declared]}
    assert validate_trip_frame(df, meta) == []


def test_features_are_causal(world: dict) -> None:
    """Dropping the future must not change the past: the same code has to run live on the phone."""
    full = _run(world)
    cut = int(30.0 / FINE)
    short = {
        **world,
        "t_acc": world["t_acc"][:cut],
        "acc": world["acc"][:cut],
        "gyro": world["gyro"][:cut],
        "t_gps": world["t_gps"][world["t_gps"] <= 29.0],
        "speed": world["speed"][world["t_gps"] <= 29.0],
        "heading": world["heading"][world["t_gps"] <= 29.0],
    }  # fmt: skip
    part = F.extract_motion_features(
        short["t_acc"], short["acc"], short["t_gps"], short["speed"],
        gyro_dps=short["gyro"], heading_deg=short["heading"], alignment=full.alignment,
    )  # fmt: skip
    n = len(part.frame) - 5  # the last step is incomplete by construction
    a = full.frame.iloc[:n].drop(columns=[S.MOTION_PREFIX + "lane_offset"])
    b = part.frame.iloc[:n].drop(columns=[S.MOTION_PREFIX + "lane_offset"])
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-4, rtol=1e-4)


def test_no_acceleration_means_no_alignment_and_nan_axes_but_valid_vertical() -> None:
    t = np.arange(0.0, 30.0, FINE)
    acc = np.tile([0.0, 0.0, G], (len(t), 1)) @ M_TRUE.T
    t_gps = np.arange(1.0, 30.0, 1.0)
    res = F.extract_motion_features(t, acc, t_gps, np.full(len(t_gps), 8.0))  # constant speed
    assert not res.alignment.ok and "acceleration" in res.alignment.reason
    for name in ("acc_long", "acc_lat", "jerk_long", "jerk_lat", "lateral_drift_5s"):
        assert np.isnan(_col(res, name)).all(), name
    assert np.nanmax(np.abs(_col(res, "acc_vert"))) < 0.2  # needs only the up axis
    assert np.isfinite(_col(res, "roll")).all()
    assert S.MOTION_PREFIX + "acc_long" in res.all_nan_columns
