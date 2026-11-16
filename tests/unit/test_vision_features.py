"""Vision features (P1-A1/A2) from synthetic MediaPipe-style landmarks with known geometry.

The face helper places the eye, iris and mouth landmarks so that EAR, MAR and the iris offset
are known exactly; head-pose matrices are built from known angles. Nothing here needs MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data.validate import validate_trip_frame
from driverisk.features import vision as V

W_EYE = 0.06  # eye width in normalized image units


def face(
    ear_r: float = 0.30,
    ear_l: float = 0.30,
    mar: float = 0.2,
    iris_dx: float = 0.0,
    iris_dy: float = 0.0,
    n_points: int = 478,
    shift_x: float = 0.0,
) -> np.ndarray:
    """One face, [n_points, 3], normalized image coords (x right, y down)."""
    p = np.zeros((n_points, 3))
    p[:, 0], p[:, 1] = 0.5, 0.5

    def eye(idx: tuple[int, ...], x0: float, ear: float, top: int, bottom: int) -> None:
        a, t1, t2, b, b2, b1 = idx  # corner, upper, upper, corner, lower, lower (EAR order)
        h, y = ear * W_EYE, 0.45
        p[a, :2], p[b, :2] = (x0, y), (x0 + W_EYE, y)
        p[t1, :2], p[t2, :2] = (x0 + 0.02, y - h / 2), (x0 + 0.04, y - h / 2)
        p[b1, :2], p[b2, :2] = (x0 + 0.02, y + h / 2), (x0 + 0.04, y + h / 2)
        p[top, :2], p[bottom, :2] = (x0 + 0.03, y - h / 2), (x0 + 0.03, y + h / 2)

    # subject's right eye appears on the image left (camera image, not mirrored)
    eye(V.RIGHT_EYE, 0.40, ear_r, *V.RIGHT_LIDS)
    eye(V.LEFT_EYE, 0.54, ear_l, *V.LEFT_LIDS)
    if n_points > 468:
        for iris, x0, ear in ((V.RIGHT_IRIS, 0.40, ear_r), (V.LEFT_IRIS, 0.54, ear_l)):
            p[iris, :2] = (x0 + 0.03 + iris_dx * W_EYE / 2, 0.45 + iris_dy * ear * W_EYE)
    top, bottom, left, right = V.MOUTH
    p[left, :2], p[right, :2] = (0.45, 0.65), (0.55, 0.65)
    p[top, :2], p[bottom, :2] = (0.5, 0.65 - mar * 0.05), (0.5, 0.65 + mar * 0.05)
    p[:, 0] += shift_x
    return p


def rot(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """4x4 face-to-camera transform for a head turned to the driver's right / up / tilted right."""
    y, pt, r = np.radians([yaw, pitch, roll])

    def rx(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def ry(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    m = np.eye(4)
    m[:3, :3] = ry(-y) @ rx(-pt) @ rz(r)
    return m


def frames(n: int, fps: float = 30.0, jitter: float = 0.0, seed: int = 0, **face_kw):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps + rng.uniform(-jitter, jitter, n)
    t = np.sort(np.clip(t, 0, None))
    lm = np.repeat(face(**face_kw)[None], n, axis=0)
    mats = np.repeat(rot()[None], n, axis=0)
    return t, lm, mats


def col(res: V.VisionResult, name: str) -> np.ndarray:
    return res.frame[S.VISION_PREFIX + name].to_numpy(np.float64)


# --- per-frame geometry ---------------------------------------------------------------------
def test_ear_and_mar_from_landmarks() -> None:
    f = face(ear_r=0.30, ear_l=0.24, mar=0.5)
    assert V.eye_aspect_ratio(f, V.RIGHT_EYE) == pytest.approx(0.30)
    assert V.eye_aspect_ratio(f, V.LEFT_EYE) == pytest.approx(0.24)
    assert V.mouth_aspect_ratio(f) == pytest.approx(0.5 * 0.1 / 0.1)


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"), [(25, 0, 0), (-30, 0, 0), (0, 15, 0), (0, -20, 0), (0, 0, 12),
                               (20, -10, 8), (-35, 25, -15)],
)  # fmt: skip
def test_head_pose_round_trip(yaw: float, pitch: float, roll: float) -> None:
    got = V.head_pose(rot(yaw, pitch, roll))
    assert got == pytest.approx((yaw, pitch, roll), abs=1e-6)


def test_gaze_follows_the_iris_and_falls_back_to_head_pose() -> None:
    # iris half a half-width toward the image left = toward the driver's right
    g = V.frame_features(face(iris_dx=-0.5), rot(yaw=5.0), V.load_vision_config())
    assert g["gaze_yaw"] == pytest.approx(5.0 + 30.0 * 0.5)
    assert g["head_yaw"] == pytest.approx(5.0)
    up = V.frame_features(face(iris_dy=-0.25), rot(), V.load_vision_config())
    assert up["gaze_pitch"] == pytest.approx(20.0 * 0.25)  # iris higher in the image = looking up
    no_iris = V.frame_features(
        face(n_points=468, iris_dx=-0.5), rot(yaw=7.0), V.load_vision_config()
    )
    assert no_iris["gaze_yaw"] == pytest.approx(7.0)  # no iris landmarks: head pose is the proxy


def test_gaze_ignores_the_iris_while_the_eyes_are_closed() -> None:
    cfg = V.load_vision_config()
    closed = V.frame_features(face(ear_r=0.05, ear_l=0.05, iris_dx=-1.0), rot(yaw=3.0), cfg)
    assert closed["gaze_yaw"] == pytest.approx(3.0)


def test_face_conf_is_the_share_of_mesh_points_inside_the_image() -> None:
    cfg = V.load_vision_config()
    assert V.frame_features(face(), rot(), cfg)["face_conf"] == 1.0
    half_out = face()
    half_out[:234, 0] = 1.2  # 234 of 468 mesh points off the right edge
    assert V.frame_features(half_out, rot(), cfg)["face_conf"] == pytest.approx(0.5)


# --- aggregation to the 10 Hz grid ----------------------------------------------------------
def test_grid_columns_and_step_means() -> None:
    t, lm, mats = frames(90, fps=30.0, jitter=0.004, ear_r=0.30, ear_l=0.26)
    res = V.extract_vision_features(t, lm, mats)
    df = res.frame
    assert list(df.columns) == [S.COL_T, *S.VISION_COLUMNS, S.COL_MASK_V]
    assert all(str(df[c].dtype) == "float32" for c in (S.COL_T, *S.VISION_COLUMNS))
    assert col(res, "ear_mean")[5] == pytest.approx(0.28, abs=1e-5)
    assert col(res, "ear_left")[5] == pytest.approx(0.26, abs=1e-5)
    assert df[S.COL_MASK_V].iloc[1:].all()


def test_no_face_rule_for_a_step_without_a_detected_face() -> None:
    t, lm, mats = frames(90)
    lm[30:60] = np.nan  # frames 1.0-2.0 s: no face
    res = V.extract_vision_features(t, lm, mats)
    k = 15  # step at 1.5 s
    assert col(res, "face_visible")[k] == 0.0 and col(res, "face_conf")[k] == 0.0
    others = [n for n in S.VISION_NAMES if n not in ("face_visible", "face_conf")]
    assert all(np.isnan(col(res, n)[k]) for n in others)
    assert not res.frame[S.COL_MASK_V].iloc[k]


def test_one_empty_step_is_held_two_are_not() -> None:
    t = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.45, 0.5, 0.55])  # nothing in (0.2, 0.4]
    lm = np.repeat(face()[None], len(t), axis=0)
    res = V.extract_vision_features(t, lm, np.repeat(rot()[None], len(t), axis=0))
    vis = col(res, "face_visible")
    assert vis[3] == 1.0  # (0.2, 0.3] empty, held from 0.2
    assert vis[4] == 0.0  # (0.3, 0.4] empty again: no data, no face


def test_low_confidence_face_is_visible_but_masked() -> None:
    t, lm, mats = frames(30)
    lm[:, :300, 0] = 1.3  # most of the mesh off-image
    res = V.extract_vision_features(t, lm, mats)
    assert (col(res, "face_visible")[1:] == 1.0).all()
    assert (col(res, "face_conf")[1:] < 0.5).all() and not res.frame[S.COL_MASK_V].any()


# --- eye closure, blinks, PERCLOS -----------------------------------------------------------
def _closures(duration_s: float, closed: list[tuple[float, float]], fps: float = 30.0):
    t = np.arange(0, duration_s, 1 / fps)
    lm = np.empty((len(t), 478, 3))
    for i, ti in enumerate(t):
        shut = any(a <= ti < b for a, b in closed)
        lm[i] = face(ear_r=0.05 if shut else 0.30, ear_l=0.05 if shut else 0.30)
    return t, lm, np.repeat(rot()[None], len(t), axis=0)


def test_short_closure_is_a_blink_long_one_is_not() -> None:
    res = V.extract_vision_features(*_closures(10.0, [(2.0, 2.2), (5.0, 6.5)]))
    closed, blink = col(res, "eye_closed"), col(res, "blink")
    assert closed[21] == 1.0 and closed[30] == 0.0
    assert blink.sum() == 1.0 and 22 <= int(np.flatnonzero(blink)[0]) <= 23  # on reopening
    assert closed[55:65].all()  # 1.5 s closure: closed, not a blink
    assert np.isnan(blink).sum() == 0


def test_perclos_needs_ten_seconds_then_is_the_closed_share() -> None:
    # closed 0.6 s of every 3 s = 20 %
    closed = [(k * 3.0, k * 3.0 + 0.6) for k in range(30)]
    res = V.extract_vision_features(*_closures(90.0, closed))
    p30, p60 = col(res, "perclos_30s"), col(res, "perclos_60s")
    assert np.isnan(p30[:99]).all() and np.isfinite(p30[100])
    assert p30[400] == pytest.approx(0.2, abs=0.02)
    assert p60[800] == pytest.approx(0.2, abs=0.02)


def test_perclos_is_nan_while_the_face_is_missing() -> None:
    t, lm, mats = _closures(30.0, [])
    lm[(t >= 20.0) & (t < 21.0)] = np.nan
    res = V.extract_vision_features(t, lm, mats)
    assert np.isnan(col(res, "perclos_30s")[205]) and np.isfinite(col(res, "perclos_30s")[250])


def test_threshold_source_is_recorded_and_can_be_overridden() -> None:
    t, lm, mats = frames(30, ear_r=0.23, ear_l=0.23)
    default = V.extract_vision_features(t, lm, mats)
    assert default.meta["ear_threshold"] == 0.21
    assert default.meta["ear_threshold_source"] == "global_default"
    assert (col(default, "eye_closed")[1:] == 0).all()
    personal = V.extract_vision_features(t, lm, mats, ear_threshold=0.25)
    assert personal.meta["ear_threshold_source"] == "driver_calibrated"
    assert (col(personal, "eye_closed")[1:] == 1).all()


def test_off_road_gaze_flag() -> None:
    t, lm, mats = frames(30)
    mats[:] = rot(yaw=30.0)
    assert (col(V.extract_vision_features(t, lm, mats), "off_road_gaze")[1:] == 1).all()
    mats[:] = rot(pitch=-25.0)  # looking down at the lap
    assert (col(V.extract_vision_features(t, lm, mats), "off_road_gaze")[1:] == 1).all()
    mats[:] = rot(yaw=5.0)
    assert (col(V.extract_vision_features(t, lm, mats), "off_road_gaze")[1:] == 0).all()


# --- contract and causality -----------------------------------------------------------------
def test_output_passes_the_contract_validator() -> None:
    t, lm, mats = _closures(20.0, [(5.0, 5.2)])
    lm[(t >= 12.0) & (t < 13.0)] = np.nan
    res = V.extract_vision_features(t, lm, mats)
    df = res.frame.copy()
    n = len(df)
    for c in S.MOTION_COLUMNS:
        df[c] = np.float32("nan")
    df[S.COL_MASK_M] = np.zeros(n, dtype=bool)
    df[S.COL_Y_CLS] = np.full(n, -1, dtype=np.int8)
    df[S.COL_CONTRACT] = S.CONTRACT_VERSION
    df = df[list(S.PARQUET_COLUMNS)]
    meta = {S.META_ALL_NAN_KEY: [*res.all_nan_columns, *S.MOTION_COLUMNS]}
    assert validate_trip_frame(df, meta) == []


def test_features_are_causal() -> None:
    t, lm, mats = _closures(40.0, [(2.0, 2.2), (15.0, 16.0), (30.0, 30.3)])
    full = V.extract_vision_features(t, lm, mats)
    keep = t < 20.0
    part = V.extract_vision_features(t[keep], lm[keep], mats[keep])
    n = len(part.frame) - 1
    pd.testing.assert_frame_equal(full.frame.iloc[:n], part.frame.iloc[:n])


def test_config_matches_the_contract_mask_threshold() -> None:
    assert V.load_vision_config()["face_conf_min"] == S.FACE_CONF_MIN


# --- fixes after the first real-face webcam run (2026-11-16) --------------------------------
def test_ear_does_not_depend_on_the_camera_aspect_ratio() -> None:
    """MediaPipe scales x by the width and y by the height; EAR must use real proportions."""
    cfg = V.load_vision_config()
    square = face(ear_r=0.30, ear_l=0.30)
    raw = square.copy()
    raw[:, 1] = 0.5 + (square[:, 1] - 0.5) * (640 / 480)  # how a 640x480 camera reports it
    wrong = V.frame_features(raw, rot(), cfg)["ear_mean"]
    right = V.frame_features(raw, rot(), cfg, aspect_hw=480 / 640)["ear_mean"]
    assert wrong == pytest.approx(0.40)  # the bug: 4:3 inflates EAR by a third
    assert right == pytest.approx(0.30)


def test_ear_uses_depth_so_a_turned_face_keeps_its_ratio() -> None:
    cfg = V.load_vision_config()
    f = face(ear_r=0.30, ear_l=0.30)
    a = np.radians(30.0)  # turn every point about the vertical axis through the face centre
    x, z = f[:, 0] - 0.5, f[:, 2]
    f[:, 0], f[:, 2] = 0.5 + x * np.cos(a) + z * np.sin(a), -x * np.sin(a) + z * np.cos(a)
    assert V.frame_features(f, rot(), cfg)["ear_mean"] == pytest.approx(0.30, abs=1e-9)


def test_eyes_are_not_read_beyond_the_yaw_limit() -> None:
    cfg = V.load_vision_config()
    limit = cfg["eye_max_abs_yaw_deg"]
    far = V.frame_features(face(iris_dx=-1.0), rot(yaw=limit + 5), cfg)
    assert np.isnan(far["ear_mean"]) and np.isnan(far["ear_left"]) and np.isnan(far["ear_right"])
    assert far["gaze_yaw"] == pytest.approx(limit + 5)  # head pose only
    near = V.frame_features(face(), rot(yaw=limit - 5), cfg)
    assert near["ear_mean"] == pytest.approx(0.30)


def test_turned_head_steps_have_no_eye_state_but_keep_the_face() -> None:
    t, lm, mats = frames(60)
    mats[30:] = rot(yaw=60.0)
    res = V.extract_vision_features(t, lm, mats)
    assert np.isnan(col(res, "eye_closed")[15]) and np.isnan(col(res, "blink")[15])
    assert col(res, "eye_closed")[5] == 0.0
    assert col(res, "face_visible")[15] == 1.0 and res.frame[S.COL_MASK_V].iloc[15]


def test_iris_offset_is_clamped() -> None:
    cfg = V.load_vision_config()
    g = V.frame_features(face(iris_dx=-6.0), rot(), cfg)
    cap = cfg["gaze"]["iris_max_offset"]
    assert g["gaze_yaw"] == pytest.approx(cfg["gaze"]["yaw_gain_deg"] * cap)


def test_extractor_takes_the_image_size() -> None:
    t, lm, mats = frames(30)
    lm[:, :, 1] = 0.5 + (lm[:, :, 1] - 0.5) * (1280 / 720)  # 16:9 camera
    res = V.extract_vision_features(t, lm, mats, image_size=(1280, 720))
    assert col(res, "ear_mean")[5] == pytest.approx(0.30, abs=1e-5)
