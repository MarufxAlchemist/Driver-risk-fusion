"""Vision features X_v (docs/03_data_contract.md section 4.1) from face landmarks.

Input is what MediaPipe Face Landmarker gives per video frame: 478 landmarks (468 mesh + 10
iris) in normalized image coordinates (x right, y down) and a 4x4 face-to-camera transform.
`features/face_landmarker.py` produces that from video; this module needs no MediaPipe, so
everything here runs in CI on synthetic landmarks.

Pipeline: per-frame geometry (EAR, MAR, head pose, gaze proxy, face confidence) -> mean over the
frames of each 0.1 s step -> causal step-level flags and context (eye_closed, blink, PERCLOS,
off_road_gaze). All of it is causal, so the same code runs live.

Conventions and proposals (assumptions A24-A28 in docs/PROGRESS.md; check on the webcam):
* Eye sides are the **subject's**: `ear_left` is the driver's left eye, which appears on the
  right of a camera image that is not mirrored.
* Head pose from the transform R = Ry(-yaw) Rx(-pitch) Rz(roll), with the camera frame x right,
  y up, z towards the camera. Positive yaw = head turned to the driver's right, positive pitch =
  looking up, positive roll = head tilted towards the right shoulder. The signs MediaPipe
  actually produces must be confirmed with a webcam check before real data is processed.
* Gaze is a proxy: head pose plus a gain times the iris offset inside the eye (configurable,
  uncalibrated). With closed eyes or no iris landmarks it falls back to head pose.
* `face_conf` is the share of the 468 mesh landmarks inside the image. The Face Landmarker
  returns no per-face score, so this proxy stands in for "landmark detection confidence".
* Step flags come from the step means: `eye_closed` = step EAR below the threshold; `blink` is
  set on the step where the eye reopens after a closure no longer than `blink_max_s` (only then
  is it known to be a blink); PERCLOS is the closed share of the valid steps in the window,
  NaN until `perclos_min_history_s` of valid steps exist.
* A step without any frame copies the previous step if that one had frames (a frame 0.1-0.2 s
  old is still current, as for the IMU); otherwise it is a no-face step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.utils.config import load_yaml
from driverisk.utils.paths import configs_dir

EXTRACTOR = "driverisk.features.vision 0.1"

# MediaPipe Face Mesh indices. EAR order: corner, upper, upper, corner, lower, lower.
RIGHT_EYE: tuple[int, ...] = (33, 160, 158, 133, 153, 144)  # driver's right eye
LEFT_EYE: tuple[int, ...] = (362, 385, 387, 263, 373, 380)  # driver's left eye
RIGHT_LIDS: tuple[int, int] = (159, 145)  # upper, lower lid centre
LEFT_LIDS: tuple[int, int] = (386, 374)
RIGHT_IRIS, LEFT_IRIS = 468, 473  # iris centres (only in the 478-point model)
MOUTH: tuple[int, int, int, int] = (13, 14, 78, 308)  # inner upper, inner lower, corners
N_MESH = 468

_CONTINUOUS = (
    "ear_left", "ear_right", "ear_mean", "mar", "head_yaw", "head_pitch", "head_roll",
    "gaze_yaw", "gaze_pitch",
)  # fmt: skip


def load_vision_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_dir() / "vision.yaml")


# --- per-frame geometry ---------------------------------------------------------------------
def _dist(p: np.ndarray, a: int, b: int) -> float:
    """3D distance. MediaPipe's z uses about the same scale as x, and including it undoes most
    of the shortening of horizontal distances when the head turns."""
    return float(np.linalg.norm(p[a, :3] - p[b, :3]))


def eye_aspect_ratio(p: np.ndarray, eye: tuple[int, ...]) -> float:
    """(|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|); NaN for a degenerate eye."""
    p1, p2, p3, p4, p5, p6 = eye
    width = _dist(p, p1, p4)
    if width < 1e-9:
        return float("nan")
    return (_dist(p, p2, p6) + _dist(p, p3, p5)) / (2.0 * width)


def mouth_aspect_ratio(p: np.ndarray) -> float:
    top, bottom, left, right = MOUTH
    width = _dist(p, left, right)
    return _dist(p, top, bottom) / width if width >= 1e-9 else float("nan")


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def head_pose(matrix: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 4x4 (or 3x3) face-to-camera transform."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    fwd = r @ np.array([0.0, 0.0, 1.0])
    yaw = np.arctan2(-fwd[0], fwd[2])
    pitch = np.arctan2(fwd[1], np.hypot(fwd[0], fwd[2]))
    up_local = (_ry(-yaw) @ _rx(-pitch)).T @ (r @ np.array([0.0, 1.0, 0.0]))
    roll = np.arctan2(-up_local[0], up_local[1])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def _iris_offsets(p: np.ndarray) -> tuple[float, float]:
    """Mean iris offset over both eyes: horizontal in half eye-widths along image x (left is
    negative), vertical in eye heights along image y (up is negative)."""
    h, v = [], []
    for eye, (top, bottom), iris in ((RIGHT_EYE, RIGHT_LIDS, RIGHT_IRIS),
                                     (LEFT_EYE, LEFT_LIDS, LEFT_IRIS)):  # fmt: skip
        a, b = p[eye[0], :2], p[eye[3], :2]
        half = abs(b[0] - a[0]) / 2.0
        height = abs(p[bottom, 1] - p[top, 1])
        if half < 1e-9 or height < 1e-9:
            return float("nan"), float("nan")
        h.append((p[iris, 0] - (a[0] + b[0]) / 2.0) / half)
        v.append((p[iris, 1] - (p[top, 1] + p[bottom, 1]) / 2.0) / height)
    return float(np.mean(h)), float(np.mean(v))


def frame_features(
    landmarks: np.ndarray,
    matrix: np.ndarray | None,
    cfg: dict[str, Any],
    ear_threshold: float | None = None,
    aspect_hw: float = 1.0,
) -> dict[str, float]:
    """Continuous features of one frame that has a face. `landmarks [468 or 478, 3]`.

    `aspect_hw` = image height / width. MediaPipe divides x by the width and y by the height,
    so y is rescaled to width units before any distance is taken; without this, EAR on a 4:3
    camera is a third too large and changes with the camera.
    """
    raw = np.asarray(landmarks, dtype=np.float64)
    p = raw.copy()
    p[:, 1] *= aspect_hw
    thr = cfg["ear_threshold"] if ear_threshold is None else ear_threshold
    yaw, pitch, roll = head_pose(matrix) if matrix is not None else (np.nan,) * 3
    eyes_readable = not (np.isfinite(yaw) and abs(yaw) > cfg["eye_max_abs_yaw_deg"])
    if eyes_readable:
        ear_r, ear_l = eye_aspect_ratio(p, RIGHT_EYE), eye_aspect_ratio(p, LEFT_EYE)
    else:  # far eye hidden, near eye seen edge-on: no eye state rather than a wrong one
        ear_r = ear_l = float("nan")
    ear = (ear_r + ear_l) / 2.0
    gaze_yaw, gaze_pitch = yaw, pitch
    if len(p) > max(RIGHT_IRIS, LEFT_IRIS) and np.isfinite(ear) and ear >= thr:
        dh, dv = _iris_offsets(p)
        if np.isfinite(dh):
            cap = cfg["gaze"]["iris_max_offset"]
            dh, dv = float(np.clip(dh, -cap, cap)), float(np.clip(dv, -cap, cap))
            gaze_yaw = yaw - cfg["gaze"]["yaw_gain_deg"] * dh  # image-left = driver's right
            gaze_pitch = pitch - cfg["gaze"]["pitch_gain_deg"] * dv  # image-up = looking up
    mesh = raw[:N_MESH, :2]  # inside-the-image test in MediaPipe's own [0, 1] units
    inside = ((mesh >= 0.0) & (mesh <= 1.0)).all(axis=1)
    return {
        "ear_left": ear_l, "ear_right": ear_r, "ear_mean": ear, "mar": mouth_aspect_ratio(p),
        "head_yaw": yaw, "head_pitch": pitch, "head_roll": roll,
        "gaze_yaw": gaze_yaw, "gaze_pitch": gaze_pitch, "face_conf": float(inside.mean()),
    }  # fmt: skip


# --- step-level features --------------------------------------------------------------------
@dataclass
class VisionResult:
    frame: pd.DataFrame  # t, 16 v_* columns, mask_v
    meta: dict[str, Any] = field(default_factory=dict)  # goes into <trip_id>_meta.json
    all_nan_columns: list[str] = field(default_factory=list)


def _blinks(closed: np.ndarray, max_steps: int) -> np.ndarray:
    """1 on the step an eye reopens after a closure of 1..max_steps steps that began after an
    open step; NaN where `closed` is NaN; 0 otherwise. Causal."""
    out = np.where(np.isnan(closed), np.nan, 0.0)
    run, opened_before = 0, False
    for k, c in enumerate(closed):
        if np.isnan(c):
            run, opened_before = 0, False
        elif c == 1.0:
            run += 1
        else:
            if opened_before and 1 <= run <= max_steps:
                out[k] = 1.0
            run, opened_before = 0, True
    return out


def extract_vision_features(
    t_frame: np.ndarray,
    landmarks: np.ndarray,
    matrices: np.ndarray | None,
    *,
    config: dict[str, Any] | None = None,
    ear_threshold: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> VisionResult:
    """Vision features on the 10 Hz grid.

    `t_frame [F]` seconds since trip start (sorted), `landmarks [F, 468|478, 3]` with NaN rows
    for frames where no face was found, `matrices [F, 4, 4]` face transforms (or None).
    `ear_threshold` given = a per-driver calibrated value; otherwise the config default.
    `image_size=(width, height)` of the frames; needed for correct ratios on non-square images.
    """
    cfg = config or load_vision_config()
    dt = S.STEP_S
    thr = float(cfg["ear_threshold"] if ear_threshold is None else ear_threshold)
    source = "global_default" if ear_threshold is None else "driver_calibrated"
    t_frame = np.asarray(t_frame, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    aspect_hw = image_size[1] / image_size[0] if image_size else 1.0
    n = int(np.floor(t_frame[-1] / dt + 1e-9)) + 1
    idx = np.ceil(t_frame / dt - 1e-9).astype(np.int64)

    sums = np.zeros((n, len(_CONTINUOUS)))
    n_face = np.zeros(n, dtype=np.int64)
    n_all = np.zeros(n, dtype=np.int64)
    conf_sum = np.zeros(n)
    for f in range(len(t_frame)):
        k = idx[f]
        if not 0 <= k < n:
            continue
        n_all[k] += 1
        if np.isnan(landmarks[f, 0, 0]):
            continue  # no face in this frame: counts towards the step, adds confidence 0
        mat = None if matrices is None else matrices[f]
        feats = frame_features(landmarks[f], mat, cfg, thr, aspect_hw)
        n_face[k] += 1
        conf_sum[k] += feats["face_conf"]
        sums[k] += [feats[c] for c in _CONTINUOUS]

    cont = np.full((n, len(_CONTINUOUS)), np.nan)
    has = n_face > 0
    cont[has] = sums[has] / n_face[has, None]
    conf = np.where(n_all > 0, conf_sum / np.maximum(n_all, 1), 0.0)
    visible = has.astype(np.float64)
    for k in range(1, n):  # hold one empty step from a step that had frames
        if n_all[k] == 0 and n_all[k - 1] > 0:
            cont[k], conf[k], visible[k] = cont[k - 1], conf[k - 1], visible[k - 1]

    cols = {c: cont[:, i] for i, c in enumerate(_CONTINUOUS)}
    ear = cols["ear_mean"]
    with np.errstate(invalid="ignore"):
        closed = np.where(np.isnan(ear), np.nan, (ear < thr).astype(np.float64))
    cols["eye_closed"] = closed
    cols["blink"] = _blinks(closed, round(cfg["blink_max_s"] / dt))
    min_hist = round(cfg["perclos_min_history_s"] / dt)
    for secs in (30, 60):
        roll = pd.Series(closed).rolling(secs * S.STEP_HZ, min_periods=min_hist).mean()
        cols[f"perclos_{secs}s"] = np.where(np.isnan(closed), np.nan, roll.to_numpy())
    off = cfg["off_road"]
    gy, gp = cols["gaze_yaw"], cols["gaze_pitch"]
    with np.errstate(invalid="ignore"):
        flag = (
            (np.abs(gy) > off["abs_yaw_deg"])
            | (gp < -off["pitch_down_deg"])
            | (gp > off["pitch_up_deg"])
        )
    cols["off_road_gaze"] = np.where(np.isnan(gy) | np.isnan(gp), np.nan, flag.astype(np.float64))
    cols["face_conf"], cols["face_visible"] = conf, visible

    # no-face rule (contract 4.1): visible 0, confidence 0, the other 14 features NaN
    none = visible == 0
    for name in S.VISION_NAMES:
        if name not in ("face_conf", "face_visible"):
            cols[name] = np.where(none, np.nan, cols[name])
    cols["face_conf"] = np.where(none, 0.0, cols["face_conf"])

    frame = pd.DataFrame({S.COL_T: (np.arange(n) * dt).astype(np.float32)})
    for name in S.VISION_NAMES:
        frame[S.VISION_PREFIX + name] = cols[name].astype(np.float32)
    frame[S.COL_MASK_V] = (frame[S.VISION_PREFIX + "face_visible"] == 1.0) & (
        frame[S.VISION_PREFIX + "face_conf"] >= S.FACE_CONF_MIN
    )
    meta = {
        "image_size": list(image_size) if image_size else None,
        "ear_threshold": thr,
        "ear_threshold_source": source,
        "vision_extractor": EXTRACTOR,
        "vision_config_version": cfg.get("version", ""),
    }
    all_nan = [c for c in S.VISION_COLUMNS if frame[c].isna().all()]
    return VisionResult(frame, meta, all_nan)
