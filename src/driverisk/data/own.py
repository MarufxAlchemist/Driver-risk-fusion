"""Own recordings (P1-D1): phone IMU/GPS log + driver-camera vision + trip sheet -> contract files.

One trip in, four things out (docs/03_data_contract.md section 6): `<trip_id>.parquet`,
`<trip_id>_events.json`, `<trip_id>_meta.json` and a row of `manifest.csv`, all passing
`driverisk.data.validate`. Steps follow docs/06 section 8 and the sync rules of docs/03 section 8.

Clocks
* The **phone clock** (IMU and GPS, one logging app, one time column each) is the reference.
  Phone seconds are measured from the first IMU sample.
* The **video clock** is seconds since the start of the camera recording, which is also the `t`
  of a vision-features file from `scripts/webcam_features.py` (it starts at the first frame).
* Sync events (06 section 5): a double tap on the phone at the start (before moving) and at the
  end (after stopping). The IMU side is found with `phone_log.detect_taps`, restricted to the first
  / last `search_s` seconds and to moments when the GPS speed says the car stands (or is unknown).
  More than one candidate, or none, is an error that names the candidates: nothing is guessed.
* The video side comes from the trip sheet (optional columns `video_sync_start_s` and
  `video_sync_end_s`: video time of the **first** tap of each pair, read off the recording by a
  person) or from an argument. The template in docs/human_tasks has no such columns yet.
  The protocol's visual pattern (three slow blinks before the taps) is **not** used: its delay
  to the tap is not defined, so it cannot fix the offset to the 100 ms target.
* Mapping: phone = intercept + scale * video. Offsets are `phone - video` at each sync event and
  drift is their change from start to end. Contract 8.4: when |drift| > `drift_correction_min_s`
  the map goes through both events (linear drift correction, scale != 1); otherwise one constant
  offset, the mean of the two, and the residual is half the drift. With drift correction the
  residual at the two events is zero by construction, so it is not an independent check.

Trip span: the overlap of the IMU log and the mapped video, so `t = 0` is the later of the two
starts. Motion features come from `features.motion.extract_motion_features` (times shifted to the
trip clock). Vision features are either resampled from a 10 Hz features table (linear for
continuous values when both neighbours have a face, nearest neighbour for flags and otherwise;
contract 8.3) or extracted from raw per-frame landmarks whose times were mapped first. MediaPipe
is never imported here.

Labels: `y_cls` from class intervals `trip_id,class,onset_s,offset_s` (class names of
`schema.CLASSES`, trip clock). Steps covered by two different classes and unlabelled steps are -1.
Events: `features.events.detect_events` (source `own_detector`) plus annotated events from the
annotation CSV of docs/human_tasks/05 (source `own_annotation`); both are kept.

Input assumptions (D6, the logging app, is still open; see the module functions):
acceleration in m/s^2 *including gravity* (or `acc_unit="g"`), gyro unit given explicitly, GPS
speed in m/s (or km/h). GPS rows without a usable speed (NaN or negative) are dropped and counted.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.data.phone_log import detect_taps, guess_time_column, to_seconds
from driverisk.features import events as E
from driverisk.features import motion as M
from driverisk.features import vision as V
from driverisk.utils.config import load_yaml
from driverisk.utils.experiment import git_commit
from driverisk.utils.paths import configs_dir, repo_root

log = logging.getLogger(__name__)

EXTRACTOR = "driverisk.data.own 0.1"
DATASET = "own"
RAW_KINDS = ("imu", "gps", "video", "vision", "frames")
OWN_TRIP_RE = re.compile(r"^own_(d\d{2,})_(\d{8})_(\d{2})$")
SHEET_VIDEO_SYNC = ("video_sync_start_s", "video_sync_end_s")
# Trip-sheet fields copied into the meta. `route` and `recorder` stay out (location, a person).
SHEET_META_FIELDS = (
    "start_time", "road_type", "weather", "traffic", "lighting", "phone_model", "app_version",
    "mount_position", "glasses", "trip_type", "planned_events", "observed_events",
    "sync_start_ok", "sync_end_ok",
)  # fmt: skip
NO_EVENTS = {"no events", "no event", "none"}
_TIME_SCALE = {"ns epoch": 1e-9, "ms epoch": 1e-3, "s epoch": 1.0, "s since start": 1.0}
_EPS_STEPS = 1e-6  # float slack when counting whole 0.1 s steps


class SyncError(ValueError):
    """A sync event is missing, ambiguous or inconsistent. Never resolved by guessing."""


def load_own_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_dir() / "own.yaml")


# --- names (fix F11) ------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnTripId:
    driver_id: str
    date: str  # yyyymmdd
    nn: str

    @property
    def trip_id(self) -> str:
        return f"{DATASET}_{self.driver_id}_{self.date}_{self.nn}"


def parse_trip_id(trip_id: str) -> OwnTripId:
    """`own_<driverid>_<yyyymmdd>_<nn>`, for example `own_d03_20261120_02`."""
    m = OWN_TRIP_RE.match(str(trip_id))
    if not m:
        raise ValueError(f"trip_id {trip_id!r} must look like own_<driverid>_<yyyymmdd>_<nn>")
    return OwnTripId(*m.groups())


def raw_file_name(trip_id: str, kind: str, suffix: str = "") -> str:
    """Raw file name `own_<driverid>_<yyyymmdd>_<nn>_<kind><suffix>` (fix F11)."""
    if kind not in RAW_KINDS:
        raise ValueError(f"kind {kind!r} not in {RAW_KINDS}")
    return f"{parse_trip_id(trip_id).trip_id}_{kind}{suffix}"


# --- phone logs -----------------------------------------------------------------------------
@dataclass
class PhoneData:
    """IMU and GPS on the phone clock, seconds since the first IMU sample."""

    t_acc: np.ndarray  # [N]
    acc: np.ndarray  # [N, 3] m/s^2 including gravity, phone frame
    t_gps: np.ndarray  # [G]
    speed_mps: np.ndarray  # [G]
    gyro_dps: np.ndarray | None = None  # [N, 3] deg/s
    heading_deg: np.ndarray | None = None  # [G] compass, clockwise
    time_unit: str = ""
    notes: list[str] = field(default_factory=list)  # rows dropped and why

    @property
    def imu_rate_hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.t_acc)))

    @property
    def gps_rate_hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.t_gps))) if len(self.t_gps) > 1 else math.nan


def _numeric_time(df: pd.DataFrame, col: str, name: str) -> np.ndarray:
    raw = pd.to_numeric(df[col], errors="coerce")
    if raw.isna().any():
        raise ValueError(
            f"{name}: {int(raw.isna().sum())} unreadable timestamps in column {col!r}; "
            "fix or drop those rows first (scripts/check_phone_log.py lists them)"
        )
    values = raw.to_numpy()
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64)
    return values.astype(np.float64)


def _guess(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    return next((lowered[c] for c in candidates if c in lowered), None)


def phone_data_from_frames(
    imu: pd.DataFrame,
    gps: pd.DataFrame,
    *,
    time_col: str | None = None,
    gps_time_col: str | None = None,
    acc_cols: Sequence[str] | None = None,
    gyro_cols: Sequence[str] | None = None,
    gyro_unit: str | None = None,
    acc_unit: str = "m/s^2",
    speed_col: str | None = None,
    heading_col: str | None = None,
    speed_unit: str = "m/s",
) -> PhoneData:
    """IMU and GPS tables (as exported by the logging app) -> `PhoneData` on one clock.

    Time columns are guessed as in `phone_log.guess_time_column`, their unit from the magnitude
    (`phone_log.to_seconds`); both logs must have the same unit, because they are put on one
    clock by subtracting the first IMU timestamp. Accelerometer columns default to `x, y, z`,
    GPS speed to `speed`, heading to `bearing`/`heading`/`course` (optional). `gyro_unit`
    (`deg/s` or `rad/s`) is required when gyro columns are given: apps differ.
    """
    t_col = time_col or guess_time_column(list(imu.columns))
    g_col = gps_time_col or guess_time_column(list(gps.columns))
    raw_imu, raw_gps = _numeric_time(imu, t_col, "IMU"), _numeric_time(gps, g_col, "GPS")
    unit = to_seconds(raw_imu.astype(np.float64))[1]
    gps_unit = to_seconds(raw_gps.astype(np.float64))[1]
    if unit != gps_unit:
        raise ValueError(
            f"IMU time unit looks like {unit!r} but GPS like {gps_unit!r}; both logs must use "
            "one clock and unit (pass the right --time-col / --gps-time-col)"
        )
    if (np.diff(raw_imu) < 0).any() or (np.diff(raw_gps) < 0).any():
        raise ValueError("timestamps go backwards in the IMU or GPS log; not repaired here")
    scale = _TIME_SCALE[unit]
    origin = raw_imu[0]
    if raw_imu.dtype == raw_gps.dtype == np.int64:
        t_acc = (raw_imu - origin).astype(np.float64) * scale
        t_gps = (raw_gps - origin).astype(np.float64) * scale
    else:
        t_acc = (raw_imu.astype(np.float64) - float(origin)) * scale
        t_gps = (raw_gps.astype(np.float64) - float(origin)) * scale
    notes: list[str] = []

    acc_cols = list(acc_cols) if acc_cols else None
    if acc_cols is None:
        guessed = [_guess(list(imu.columns), (a,)) for a in ("x", "y", "z")]
        if None in guessed:
            raise ValueError(f"no accelerometer columns x, y, z in {list(imu.columns)}; pass them")
        acc_cols = [str(c) for c in guessed]
    acc = imu[acc_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    gyro = None
    if gyro_cols:
        if gyro_unit not in ("deg/s", "rad/s"):
            raise ValueError("gyro_unit must be 'deg/s' or 'rad/s' when gyro columns are given")
        gyro = imu[list(gyro_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        if gyro_unit == "rad/s":
            gyro = np.degrees(gyro)
    bad = ~np.isfinite(acc).all(axis=1)
    if gyro is not None:
        bad |= ~np.isfinite(gyro).all(axis=1)
    if bad.any():
        notes.append(f"dropped {int(bad.sum())} IMU rows with missing values")
        t_acc, acc = t_acc[~bad], acc[~bad]
        gyro = gyro[~bad] if gyro is not None else None
    if acc_unit == "g":
        acc = acc * M.G_STD
    elif acc_unit != "m/s^2":
        raise ValueError("acc_unit must be 'm/s^2' or 'g'")
    g = float(np.median(np.linalg.norm(acc, axis=1)))
    if not 7.0 <= g <= 13.0:
        if 0.7 <= g <= 1.3:
            hint = "the values look like units of g: pass acc_unit='g'"
        elif g < 0.7:
            hint = "the log looks like linear acceleration without gravity, which is not supported"
        else:
            hint = "unknown unit"
        raise ValueError(
            f"median accelerometer magnitude {g:.2f} is not about 9.8 m/s^2 ({hint}); "
            "features.motion needs acceleration including gravity"
        )

    s_col = speed_col or _guess(list(gps.columns), ("speed",))
    if s_col is None:
        raise ValueError(f"no GPS speed column among {list(gps.columns)}; pass speed_col")
    speed = pd.to_numeric(gps[s_col], errors="coerce").to_numpy(np.float64)
    if speed_unit == "km/h":
        speed = speed / 3.6
    elif speed_unit != "m/s":
        raise ValueError("speed_unit must be 'm/s' or 'km/h'")
    h_col = heading_col or _guess(list(gps.columns), ("bearing", "heading", "course"))
    heading = None if h_col is None else pd.to_numeric(gps[h_col], errors="coerce").to_numpy()
    usable = np.isfinite(speed) & (speed >= 0)
    if (~usable).any():
        notes.append(f"dropped {int((~usable).sum())} GPS rows without a usable speed")
    t_gps, speed = t_gps[usable], speed[usable]
    if heading is not None:
        heading = heading.astype(np.float64)[usable]
        if not (np.isfinite(heading) & (heading >= 0)).all():
            notes.append("GPS heading has missing or negative values: heading_rate not computed")
            heading = None
    for n in notes:
        log.warning(n)
    return PhoneData(t_acc, acc, t_gps, speed, gyro, heading, unit, notes)


def load_phone_data(imu_csv: str | Path, gps_csv: str | Path, **kwargs: Any) -> PhoneData:
    """Read the IMU and GPS CSV files; keyword arguments as in `phone_data_from_frames`."""
    return phone_data_from_frames(pd.read_csv(imu_csv), pd.read_csv(gps_csv), **kwargs)


# --- clock sync -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockSync:
    """phone = intercept + scale * video (seconds)."""

    offset_start_s: float  # phone - video at the start sync event
    offset_end_s: float  # same at the end event (NaN without one)
    drift_s: float  # offset_end - offset_start (NaN without an end event)
    intercept_s: float
    scale: float
    drift_corrected: bool
    residual_s: float  # largest |mapped video time - IMU time| over the sync events
    imu_sync_s: tuple[float, ...]
    video_sync_s: tuple[float, ...]

    def video_to_phone(self, t_video: np.ndarray) -> np.ndarray:
        return self.intercept_s + self.scale * np.asarray(t_video, dtype=np.float64)

    def phone_to_video(self, t_phone: np.ndarray) -> np.ndarray:
        return (np.asarray(t_phone, dtype=np.float64) - self.intercept_s) / self.scale

    def to_meta(self) -> dict[str, Any]:
        return {
            "offset_start_s": self.offset_start_s,
            "offset_end_s": self.offset_end_s,
            "drift_s": self.drift_s,
            "intercept_s": self.intercept_s,
            "scale": self.scale,
            "drift_corrected": self.drift_corrected,
            "residual_s": self.residual_s,
            "residual_method": (
                "zero by construction (map through both events)"
                if self.drift_corrected
                else "half the start-end offset difference"
                if len(self.imu_sync_s) == 2
                else "not measurable (one sync event)"
            ),
            "imu_sync_s": list(self.imu_sync_s),
            "video_sync_s": list(self.video_sync_s),
            "mapping": "phone_s = intercept_s + scale * video_s",
        }


def estimate_clock_sync(
    imu_sync_s: Sequence[float],
    video_sync_s: Sequence[float],
    config: Mapping[str, Any] | None = None,
    *,
    require_end: bool = True,
) -> ClockSync:
    """Offset (and drift) between the video and the phone clock from matching sync events.

    `imu_sync_s` and `video_sync_s` are the times of the same events (start, then end) on the
    phone and the video clock. With `require_end=False` a start event alone gives an
    offset-only sync (drift unknown); otherwise a missing end event is an error.
    """
    cfg = (config or load_own_config())["sync"]
    imu = [float(x) for x in imu_sync_s]
    vid = [float(x) for x in video_sync_s]
    if len(imu) != len(vid):
        raise SyncError(
            f"need the same number of sync events in both streams: IMU {len(imu)}, video {len(vid)}"
        )
    if not imu:
        raise SyncError("no start sync event: the double tap of 06 section 5 is needed")
    if len(imu) > 2:
        raise SyncError(f"at most two sync events (start, end), got {len(imu)}")
    off0 = imu[0] - vid[0]
    if len(imu) == 1:
        if require_end:
            raise SyncError(
                "no end sync event: 06 section 5 step 7 needs one for drift correction "
                "(require_end=False accepts an offset-only sync and marks the trip low quality)"
            )
        return ClockSync(off0, math.nan, math.nan, off0, 1.0, False, math.nan, (imu[0],), (vid[0],))
    if not (imu[1] > imu[0] and vid[1] > vid[0]):
        raise SyncError(f"sync events out of order: IMU {imu}, video {vid} (start must be first)")
    rate = (imu[1] - imu[0]) / (vid[1] - vid[0])
    if abs(rate - 1.0) > cfg["max_rate_error"]:
        raise SyncError(
            f"start and end sync events do not match: clock rate {rate:.4f} differs from 1 by "
            f"more than {cfg['max_rate_error']}; a wrong tap pair or a wrong video time?"
        )
    off1 = imu[1] - vid[1]
    drift = off1 - off0
    if abs(drift) > cfg["drift_correction_min_s"]:
        intercept, scale, corrected, residual = imu[0] - rate * vid[0], rate, True, 0.0
    else:
        intercept, scale, corrected, residual = (off0 + off1) / 2.0, 1.0, False, abs(drift) / 2.0
    return ClockSync(
        off0, off1, drift, intercept, scale, corrected, residual, tuple(imu), tuple(vid)
    )


def _speed_at(t: float, t_gps: np.ndarray, speed: np.ndarray, max_age_s: float) -> float:
    """Speed of the latest fix at or before `t`; NaN when there is none or it is too old."""
    pos = int(np.searchsorted(t_gps, t + 1e-9, side="right")) - 1
    if pos < 0 or t - t_gps[pos] > max_age_s:
        return math.nan
    return float(speed[pos])


def find_imu_sync_events(
    t_acc: np.ndarray,
    acc: np.ndarray,
    t_gps: np.ndarray | None = None,
    speed_mps: np.ndarray | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    require_end: bool = True,
) -> list[float]:
    """Phone times of the start and end sync events (first tap of each double tap).

    The start pair is searched in the first `search_s` seconds of the log and the end pair in
    the last `search_s` (each at most half the log), using `phone_log.detect_taps` inside that
    window. With GPS, only pairs at a known speed of at most `max_speed_mps` (or an unknown
    speed) count, because the protocol taps before moving and after stopping. Exactly one
    candidate per window is required; zero or several raise `SyncError` naming the times.
    """
    cfg = (config or load_own_config())["sync"]
    t, acc = np.asarray(t_acc, dtype=np.float64), np.asarray(acc, dtype=np.float64)
    start, end = float(t[0]), float(t[-1])
    half = (start + end) / 2.0
    max_age = M.MotionConfig().gps_max_age_s

    def candidates(lo: float, hi: float) -> list[float]:
        sel = (t >= lo) & (t <= hi)
        found = detect_taps(t[sel], acc[sel]) if sel.sum() > 2 else []
        if t_gps is None or speed_mps is None:
            return found
        tg, sp = np.asarray(t_gps, dtype=np.float64), np.asarray(speed_mps, dtype=np.float64)
        return [
            x for x in found if not _speed_at(x, tg, sp, max_age) > cfg["max_speed_mps"]
        ]  # NaN speed (no fix yet) is kept

    def one(found: list[float], where: str, lo: float, hi: float) -> float:
        if len(found) > 1:
            times = ", ".join(f"{x:.2f}" for x in found)
            raise SyncError(
                f"{len(found)} tap pairs found in {where} ({lo:.0f}-{hi:.0f} s) at {times} s; "
                "cannot tell which is the sync event. Check the log and pass the IMU sync "
                "times explicitly"
            )
        return found[0]

    head = (start, min(start + cfg["search_s"], half))
    tail = (max(end - cfg["search_s"], half), end)
    first = candidates(*head)
    if not first:
        raise SyncError(
            f"no tap pair found in the first {head[1] - head[0]:.0f} s of the IMU log while "
            "stationary: the start sync event (06 section 5) is missing or too weak"
        )
    out = [one(first, "the start window", *head)]
    last = candidates(*tail)
    if not last:
        if require_end:
            raise SyncError(
                f"no tap pair found in the last {tail[1] - tail[0]:.0f} s of the IMU log while "
                "stationary: the end sync event (06 section 5 step 7) is missing or too weak"
            )
        return out
    out.append(one(last, "the end window", *tail))
    return out


# --- vision input ---------------------------------------------------------------------------
@dataclass
class VisionFeatures:
    """10 Hz vision features on the video clock: `t` + 16 `v_*` (+ `mask_v`), for example the
    parquet of `scripts/webcam_features.py`. That file does not record the EAR threshold, so
    pass it here; when it is None the config default is assumed and the meta says so."""

    frame: pd.DataFrame
    ear_threshold: float | None = None
    ear_threshold_source: str | None = None


@dataclass
class VisionFrames:
    """Raw per-frame landmarks on the video clock, the input of `vision.extract_vision_features`:
    `t_frame [F]` s, `landmarks [F, 468|478, 3]` (NaN rows = no face), `matrices [F, 4, 4]`."""

    t_frame: np.ndarray
    landmarks: np.ndarray
    matrices: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    ear_threshold: float | None = None  # a per-driver calibrated value, else the config default


def load_vision_features(path: str | Path, **kwargs: Any) -> VisionFeatures:
    return VisionFeatures(pd.read_parquet(path), **kwargs)


def load_vision_frames(path: str | Path) -> VisionFrames:
    """`.npz` with arrays `t`, `landmarks`, optional `matrices` and `image_size` (W, H)."""
    with np.load(path) as z:
        mats = z["matrices"] if "matrices" in z else None
        size = tuple(int(x) for x in z["image_size"]) if "image_size" in z else None
        return VisionFrames(z["t"], z["landmarks"], mats, size)  # type: ignore[arg-type]


def _check_vision_features(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in (S.COL_T, *S.VISION_COLUMNS) if c not in frame.columns]
    if missing:
        raise ValueError(f"vision features table misses columns {missing}")
    t = frame[S.COL_T].to_numpy(np.float64)
    if len(t) < 2 or (np.abs(np.diff(t) - S.STEP_S) > S.T_STEP_TOLERANCE_S).any():
        raise ValueError("vision features must be on a regular 0.1 s grid (column t)")
    return t


def _resample_vision(frame: pd.DataFrame, t_video: np.ndarray) -> pd.DataFrame:
    """Vision features at the video times `t_video` (contract 8.3: linear for continuous values
    when both neighbours have a face and finite values, else nearest neighbour; flags nearest),
    then the no-face rule of contract 4.1."""
    t = _check_vision_features(frame)
    pos = np.clip((t_video - t[0]) / S.STEP_S, 0.0, len(t) - 1.0)
    k0 = np.floor(pos).astype(np.int64)
    k1 = np.minimum(k0 + 1, len(t) - 1)
    w = pos - k0
    near = np.clip(np.rint(pos).astype(np.int64), 0, len(t) - 1)
    visible = frame[S.VISION_PREFIX + "face_visible"].to_numpy(np.float64)
    both = (visible[k0] == 1.0) & (visible[k1] == 1.0)
    out: dict[str, np.ndarray] = {}
    for spec in S.VISION_FEATURES:
        x = frame[S.VISION_PREFIX + spec.name].to_numpy(np.float64)
        value = x[near]
        if spec.kind == "continuous":
            lin = (1.0 - w) * x[k0] + w * x[k1]
            ok = both & np.isfinite(x[k0]) & np.isfinite(x[k1])
            value = np.where(ok, lin, value)
        out[spec.name] = value
    none = out["face_visible"] != 1.0
    for name in S.VISION_NAMES:
        if name not in ("face_conf", "face_visible"):
            out[name] = np.where(none, np.nan, out[name])
    out["face_conf"] = np.where(none, 0.0, out["face_conf"])
    out["face_visible"] = np.where(none, 0.0, 1.0)
    return pd.DataFrame({S.VISION_PREFIX + n: out[n] for n in S.VISION_NAMES})


# --- labels and events ----------------------------------------------------------------------
def _rows_for(table: pd.DataFrame | None, trip_id: str, what: str) -> pd.DataFrame | None:
    if table is None:
        return None
    if "trip_id" not in table.columns:
        raise ValueError(f"{what} table has no trip_id column")
    return table[table["trip_id"].astype(str) == trip_id].reset_index(drop=True)


def _interval(row: Mapping[str, Any], duration_s: float, what: str, i: int) -> tuple[float, float]:
    on, off = float(row["onset_s"]), float(row["offset_s"])
    if not (0.0 <= on <= off <= duration_s + S.T_STEP_TOLERANCE_S):
        raise ValueError(f"{what} row {i}: [{on}, {off}] is not inside the trip [0, {duration_s}]")
    return on, off


def labels_to_y_cls(labels: pd.DataFrame | None, t: np.ndarray, duration_s: float) -> np.ndarray:
    """Per-step class IDs from class intervals (`class`, `onset_s`, `offset_s`, inclusive)."""
    cls = np.full(len(t), S.UNLABELED, dtype=np.int64)
    if labels is None or labels.empty:
        return cls.astype(np.int8)
    conflict = np.zeros(len(t), dtype=bool)
    for i, row in labels.iterrows():
        name = str(row["class"])
        if name not in S.CLASS_TO_ID:
            raise ValueError(f"labels row {i}: unknown class {name!r}, expected one of {S.CLASSES}")
        on, off = _interval(row, duration_s, "labels", int(i))
        inside = (t >= on - 1e-6) & (t <= off + 1e-6)
        cid = S.CLASS_TO_ID[name]
        conflict |= inside & (cls != S.UNLABELED) & (cls != cid)
        cls[inside & (cls == S.UNLABELED)] = cid
    return np.where(conflict, S.UNLABELED, cls).astype(np.int8)


def _parse_bool(value: Any, i: int) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"annotation row {i}: uncertain must be true or false, got {value!r}")


def annotated_events(annotations: pd.DataFrame | None, duration_s: float) -> list[dict[str, Any]]:
    """Rows `type,onset_s,offset_s,uncertain` (docs/human_tasks/05) -> contract events."""
    if annotations is None:
        return []
    out = []
    for i, row in annotations.iterrows():
        kind = str(row["type"])
        if kind not in S.EVENT_TYPES:
            raise ValueError(f"annotation row {i}: unknown event type {kind!r}")
        on, off = _interval(row, duration_s, "annotation", int(i))
        out.append(
            {
                "onset_s": on,
                "offset_s": off,
                "type": kind,
                "uncertain": _parse_bool(row.get("uncertain"), int(i)),
                "source": "own_annotation",
            }
        )
    return out


# --- trip sheet -----------------------------------------------------------------------------
def read_trip_sheet(path: str | Path, trip_id: str) -> dict[str, Any]:
    """The trip sheet row of `trip_id` (docs/06 section 6, plus optional video sync columns)."""
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = sheet[sheet["trip_id"].str.strip() == trip_id]
    if len(rows) != 1:
        raise ValueError(f"trip sheet {path} has no row (or several) for {trip_id}")
    return {k: (v.strip() if isinstance(v, str) else v) for k, v in rows.iloc[0].items()}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _check_sheet(sheet: Mapping[str, Any], tid: OwnTripId) -> None:
    if not _blank(sheet.get("trip_id")) and str(sheet["trip_id"]) != tid.trip_id:
        raise ValueError(f"trip sheet trip_id {sheet['trip_id']!r} != {tid.trip_id}")
    if str(sheet.get("driver_id", "")).strip() != tid.driver_id:
        raise ValueError(f"trip sheet driver_id {sheet.get('driver_id')!r} != {tid.driver_id}")
    date = re.sub(r"\D", "", str(sheet.get("date", "")))
    if date != tid.date:
        raise ValueError(f"trip sheet date {sheet.get('date')!r} != {tid.date} in the trip_id")


def _video_sync_from_sheet(sheet: Mapping[str, Any], require_end: bool) -> list[float]:
    start_col, end_col = SHEET_VIDEO_SYNC
    if _blank(sheet.get(start_col)):
        raise SyncError(
            f"no video time of the start sync event: fill trip sheet column {start_col} (video "
            "seconds of the first tap) or pass it as an argument"
        )
    times = [float(sheet[start_col])]
    if _blank(sheet.get(end_col)):
        if require_end:
            raise SyncError(
                f"no video time of the end sync event: fill trip sheet column {end_col} or pass "
                "it as an argument"
            )
    else:
        times.append(float(sheet[end_col]))
    return times


# --- the trip -------------------------------------------------------------------------------
@dataclass
class OwnTrip:
    trip_id: str
    frame: pd.DataFrame  # contract parquet columns
    events: list[dict[str, Any]]
    meta: dict[str, Any]
    manifest_row: dict[str, Any]


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def _quality(
    frame: pd.DataFrame,
    phone: PhoneData,
    span: tuple[float, float],
    sync: ClockSync,
    events: list[dict[str, Any]],
    sheet: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """The checks of docs/06 section 9. Face visibility is over the whole trip (posed and
    normal segments are not marked yet)."""
    q, residual_max = cfg["quality"], cfg["sync"]["residual_max_s"]
    face = float(frame[S.VISION_PREFIX + "face_visible"].mean())
    sel = (phone.t_acc >= span[0] - 1e-9) & (phone.t_acc <= span[1] + 1e-9)
    gap = float(np.diff(phone.t_acc[sel]).max()) if sel.sum() > 1 else math.inf
    gps = float(frame[S.MOTION_PREFIX + "gps_valid"].mean())
    sync_ok = len(sync.imu_sync_s) == 2 and sync.residual_s < residual_max
    annotated = any(e.get("source") == "own_annotation" for e in events)
    no_events = str(sheet.get("observed_events", "")).strip().lower() in NO_EVENTS
    return {
        "face_visible": {"value": face, "pass": face >= q["face_visible_min"]},
        "imu_max_gap_s": {"value": gap, "pass": gap <= q["imu_max_gap_s"]},
        "gps_valid": {"value": gps, "pass": gps >= q["gps_valid_min"]},
        "sync": {"value": sync.residual_s, "pass": bool(sync_ok)},
        "labels": {"value": annotated or no_events, "pass": annotated or no_events},
    }


def process_own_trip(
    trip_id: str,
    phone: PhoneData,
    vision: VisionFeatures | VisionFrames,
    trip_sheet: Mapping[str, Any],
    *,
    video_sync_s: Sequence[float] | None = None,
    imu_sync_s: Sequence[float] | None = None,
    annotations: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    require_end_sync: bool = True,
    source_version: str | None = None,
    config: Mapping[str, Any] | None = None,
    event_config: Mapping[str, Any] | None = None,
    vision_config: Mapping[str, Any] | None = None,
) -> OwnTrip:
    """One own recording -> contract frame, events, meta and manifest row (not written).

    `video_sync_s` overrides the trip sheet's video sync columns and `imu_sync_s` the tap
    detection. `annotations` (event intervals) and `labels` (class intervals) are tables in the
    trip clock; rows of other trips are ignored. `annotations` given (even with no rows for this
    trip) means the trip was annotated (`events_annotated = true`).
    """
    tid = parse_trip_id(trip_id)
    _check_sheet(trip_sheet, tid)
    cfg = dict(config or load_own_config())
    ev_cfg = dict(event_config or E.load_event_config())
    v_cfg = dict(vision_config or V.load_vision_config())

    if video_sync_s is not None:
        vid_sync, vid_source = [float(x) for x in video_sync_s], "argument"
    else:
        vid_sync, vid_source = _video_sync_from_sheet(trip_sheet, require_end_sync), "trip_sheet"
    if imu_sync_s is not None:
        imu_sync, imu_source = [float(x) for x in imu_sync_s], "argument"
    else:
        imu_sync = find_imu_sync_events(
            phone.t_acc, phone.acc, phone.t_gps, phone.speed_mps, config=cfg,
            require_end=require_end_sync,
        )  # fmt: skip
        imu_source = "detected"
    sync = estimate_clock_sync(imu_sync, vid_sync, cfg, require_end=require_end_sync)

    # trip span: overlap of the IMU log and the mapped video, on the phone clock
    if isinstance(vision, VisionFeatures):
        t_v = _check_vision_features(vision.frame)
    else:
        t_v = np.asarray(vision.t_frame, dtype=np.float64)
    v_start, v_end = sync.video_to_phone(np.array([t_v[0], t_v[-1]]))
    t0, t1 = max(float(phone.t_acc[0]), float(v_start)), min(float(phone.t_acc[-1]), float(v_end))
    if t1 - t0 < S.WINDOW_STEPS * S.STEP_S:
        raise SyncError(f"IMU and video overlap for {t1 - t0:.1f} s only; check the sync times")
    n = int(np.floor((t1 - t0) / S.STEP_S + _EPS_STEPS)) + 1
    grid = np.arange(n) * S.STEP_S

    motion = M.extract_motion_features(
        phone.t_acc - t0, phone.acc, phone.t_gps - t0, phone.speed_mps,
        gyro_dps=phone.gyro_dps, heading_deg=phone.heading_deg,
    )  # fmt: skip
    m_frame = motion.frame.iloc[:n].reset_index(drop=True)
    if len(m_frame) < n:
        raise ValueError(f"motion features cover {len(m_frame)} steps, the trip needs {n}")

    vision_meta: dict[str, Any] = {}
    if isinstance(vision, VisionFeatures):
        v_frame = _resample_vision(vision.frame, sync.phone_to_video(grid + t0))
        thr = vision.ear_threshold
        vision_meta = {
            "ear_threshold": float(v_cfg["ear_threshold"]) if thr is None else float(thr),
            "ear_threshold_source": vision.ear_threshold_source
            or ("global_default" if thr is None else "driver_calibrated"),
            "vision_extractor": "vision features file (extractor version not recorded in it)",
        }
        if thr is None:
            vision_meta["ear_threshold_note"] = (
                "assumed: the vision features file does not record its threshold; "
                "the config default is reported"
            )
        sampling_v = {"vision_features_hz": S.STEP_HZ}
    else:
        tau = sync.video_to_phone(t_v) - t0
        res = V.extract_vision_features(
            tau, vision.landmarks, vision.matrices, config=v_cfg,
            ear_threshold=vision.ear_threshold, image_size=vision.image_size,
        )  # fmt: skip
        if len(res.frame) < n:
            raise ValueError(f"vision features cover {len(res.frame)} steps, the trip needs {n}")
        v_frame = res.frame.iloc[:n][list(S.VISION_COLUMNS)].reset_index(drop=True)
        vision_meta = dict(res.meta)
        fps = (len(t_v) - 1) / (t_v[-1] - t_v[0]) if len(t_v) > 1 else math.nan
        sampling_v = {"video_fps": float(fps)}

    frame = pd.DataFrame({S.COL_T: grid.astype(np.float32)})
    for col in S.VISION_COLUMNS:
        frame[col] = v_frame[col].to_numpy(np.float64).astype(np.float32)
    for col in S.MOTION_COLUMNS:
        frame[col] = m_frame[col].to_numpy(np.float32)
    frame[S.COL_MASK_V] = (frame[S.VISION_PREFIX + "face_visible"] == 1.0) & (
        frame[S.VISION_PREFIX + "face_conf"] >= S.FACE_CONF_MIN
    )
    frame[S.COL_MASK_M] = m_frame[S.COL_MASK_M].to_numpy(bool)
    duration = float(n * S.STEP_S)
    frame[S.COL_Y_CLS] = labels_to_y_cls(
        _rows_for(labels, trip_id, "labels"), grid, duration
    ).astype(np.int8)
    frame[S.COL_CONTRACT] = S.CONTRACT_VERSION

    detected = E.detect_events(frame, ev_cfg)
    ann_rows = _rows_for(annotations, trip_id, "annotations")
    events = sorted(
        detected + annotated_events(ann_rows, duration),
        key=lambda e: (e["onset_s"], e["type"], e["source"]),
    )

    all_nan = [c for c in (*S.VISION_COLUMNS, *S.MOTION_COLUMNS) if frame[c].isna().all()]
    checks = _quality(frame, phone, (t0, t1), sync, events, trip_sheet, cfg)
    meta: dict[str, Any] = {
        "trip_id": trip_id,
        "contract_version": S.CONTRACT_VERSION,
        "sampling_rates": {
            "common_hz": S.STEP_HZ,
            "imu_hz": phone.imu_rate_hz,
            "gps_hz": phone.gps_rate_hz,
            **sampling_v,
        },
        "extractor_versions": {
            "own_pipeline": EXTRACTOR,
            "git_commit": git_commit(repo_root()),
            "motion": "driverisk.features.motion",
            "vision": vision_meta.pop("vision_extractor", ""),
            "events_config": ev_cfg.get("version", ""),
            "vision_config": v_cfg.get("version", ""),
            "own_config": cfg.get("version", ""),
        },
        "notes": str(trip_sheet.get("notes", "") or ""),
        S.META_ALL_NAN_KEY: all_nan,
        "events_annotated": annotations is not None,
        "ear_threshold": vision_meta.pop("ear_threshold"),
        "ear_threshold_source": vision_meta.pop("ear_threshold_source"),
        "vision": vision_meta,
        "sync": {**sync.to_meta(), "imu_sync_source": imu_source, "video_sync_source": vid_source},
        "trip_start_phone_s": t0,
        "trip_end_phone_s": t1,
        "phone_time_unit": phone.time_unit,
        "phone_log_notes": list(phone.notes),
        "motion_alignment": motion.alignment.to_meta(),
        "quality_checks": checks,
        "quality": "ok" if all(c["pass"] for c in checks.values()) else "low",
        "trip_sheet": {k: trip_sheet[k] for k in SHEET_META_FIELDS if k in trip_sheet},
    }
    sheet_value = {k: (None if _blank(trip_sheet.get(k)) else trip_sheet.get(k))
                   for k in ("trip_type", "road_type", "vehicle_id")}  # fmt: skip
    row = {
        "trip_id": trip_id,
        "dataset": DATASET,
        "driver_id": tid.driver_id,
        "pairing": "paired",
        "duration_s": duration,
        "n_steps": n,
        "behavior_tag": sheet_value["trip_type"],
        "road_type": sheet_value["road_type"],
        "vehicle_id": sheet_value["vehicle_id"],
        "split": None,  # assigned by the split script
        "source_version": source_version,
        "contract_version": S.CONTRACT_VERSION,
    }
    return OwnTrip(trip_id, frame, _json_safe(events), _json_safe(meta), row)


def write_own_trip(trip: OwnTrip, out_dir: str | Path) -> dict[str, Path]:
    """Write the three trip files and insert or replace the trip's row in `manifest.csv`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = S.trip_files(trip.trip_id)
    paths = {k: out / v for k, v in names.items()}
    trip.frame.to_parquet(paths["parquet"], index=False)
    paths["events"].write_text(json.dumps(trip.events, indent=2), encoding="utf-8")
    paths["meta"].write_text(json.dumps(trip.meta, indent=2), encoding="utf-8")
    manifest_path = out / "manifest.csv"
    row = pd.DataFrame([trip.manifest_row], columns=list(S.MANIFEST_COLUMNS))
    if manifest_path.is_file():
        old = pd.read_csv(manifest_path, dtype={"trip_id": str, "driver_id": str})
        old = old[old["trip_id"] != trip.trip_id]
        row = pd.concat([old, row], ignore_index=True) if len(old) else row
    row.sort_values("trip_id").to_csv(manifest_path, index=False)
    paths["manifest"] = manifest_path
    log.info("wrote %s (%d steps) to %s", trip.trip_id, len(trip.frame), out)
    return paths
