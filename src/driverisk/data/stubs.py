"""Synthetic stub trips in the exact contract format (docs/03_data_contract.md section 9).

Stub data exists so every workstream can build before real data arrives. It is NOT a
model of real driving. The behaviour signals follow a deliberately simple toy rule:

- drowsy:     low EAR, eye closures, head nods; lane drift on motion-only trips
- aggressive: harsh braking/acceleration pulses and sharp turns
- distracted: sustained head/gaze yaw away from the road

Availability gaps (NaN + mask False) and low face confidence are injected at random.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.data.splits import assign_driver_splits

log = logging.getLogger(__name__)

# Toy signal constants: only used to make stub behaviour learnable, not physical claims.
_EAR_OPEN, _EAR_DROWSY, _EAR_CLOSED_BELOW = 0.30, 0.17, 0.20
_YAW_DISTRACTED_DEG, _OFF_ROAD_YAW_DEG = 35.0, 20.0
_BRAKE_MS2, _ACCEL_MS2, _TURN_MS2 = -6.0, 4.0, 5.0
_PERCLOS_MIN_HISTORY_S = 10.0  # assumption A4: PERCLOS needs at least 10 s of history


@dataclass
class StubTrip:
    frame: pd.DataFrame
    events: list[dict[str, Any]]
    meta: dict[str, Any]
    manifest_row: dict[str, Any]


def _gap_mask(
    rng: np.random.Generator, n: int, start_prob: float, length_steps: tuple[int, int]
) -> np.ndarray:
    """Boolean [n], True inside random availability gaps."""
    gap = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(rng.random(n) < start_prob)
    for s in starts:
        gap[s : s + int(rng.integers(length_steps[0], length_steps[1] + 1))] = True
    return gap


def _segments(
    rng: np.random.Generator, n: int, class_probs: list[float], seg_steps: tuple[int, int]
) -> list[tuple[int, int, int]]:
    """Cut [0, n) into (start, end, class_id) behaviour segments."""
    probs = np.asarray(class_probs, dtype=np.float64)
    probs = probs / probs.sum()
    out, start = [], 0
    while start < n:
        end = min(n, start + int(rng.integers(seg_steps[0], seg_steps[1] + 1)))
        out.append((start, end, int(rng.choice(S.K_CLASSES, p=probs))))
        start = end
    return out


def _causal_rolling_mean(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=min_periods).mean().to_numpy()


def _vision(rng: np.random.Generator, n: int, cls: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    drowsy, distracted = cls == S.CLASS_TO_ID["drowsy"], cls == S.CLASS_TO_ID["distracted"]
    ear = np.where(drowsy, _EAR_DROWSY, _EAR_OPEN) + rng.normal(0, 0.02, n)
    # Eye closures: occasional 1-2 s closures when drowsy, short blinks otherwise.
    blink = (rng.random(n) < 0.03).astype(np.float64)
    closures = _gap_mask(rng, n, 0.01, (10, 20)) & drowsy
    ear = np.where(closures | (blink > 0), 0.08 + rng.normal(0, 0.01, n), ear)
    ear = np.clip(ear, 0.0, 1.0)
    ear_l = np.clip(ear + rng.normal(0, 0.01, n), 0.0, 1.0)
    ear_r = np.clip(ear + rng.normal(0, 0.01, n), 0.0, 1.0)
    eye_closed = (ear < _EAR_CLOSED_BELOW).astype(np.float64)
    min_hist = int(_PERCLOS_MIN_HISTORY_S * S.STEP_HZ)
    yawn = _gap_mask(rng, n, 0.003, (20, 40)) & drowsy
    mar = np.clip(np.where(yawn, 0.8, 0.1) + rng.normal(0, 0.03, n), 0.0, 2.0)
    head_yaw = rng.normal(0, 3, n) + np.where(
        distracted, _YAW_DISTRACTED_DEG * rng.choice([-1, 1]), 0
    )
    nods = _gap_mask(rng, n, 0.005, (10, 20)) & drowsy
    head_pitch = rng.normal(-2, 3, n) + np.where(nods, -15.0, 0.0)
    head_roll = rng.normal(0, 2, n)
    gaze_yaw = head_yaw + rng.normal(0, 2, n)
    gaze_pitch = head_pitch + rng.normal(0, 2, n)
    face_conf = rng.uniform(0.8, 1.0, n)
    face_visible = np.ones(n)

    low_conf = rng.random(n) < cfg["low_conf_prob"]
    face_conf[low_conf] = rng.uniform(0.2, 0.49, int(low_conf.sum()))
    gap_len = tuple(int(s * S.STEP_HZ) for s in cfg["dropout"]["length_s"])
    gap = _gap_mask(rng, n, cfg["dropout"]["start_prob"], gap_len)

    feats = {
        "ear_left": ear_l,
        "ear_right": ear_r,
        "ear_mean": (ear_l + ear_r) / 2,
        "mar": mar,
        "blink": blink,
        "eye_closed": eye_closed,
        "perclos_30s": _causal_rolling_mean(eye_closed, 30 * S.STEP_HZ, min_hist),
        "perclos_60s": _causal_rolling_mean(eye_closed, 60 * S.STEP_HZ, min_hist),
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "head_roll": head_roll,
        "gaze_yaw": gaze_yaw,
        "gaze_pitch": gaze_pitch,
        "off_road_gaze": (np.abs(gaze_yaw) > _OFF_ROAD_YAW_DEG).astype(np.float64),
        "face_conf": face_conf,
        "face_visible": face_visible,
    }
    # No face (assumption A3): face_visible = 0, face_conf = 0, the other 14 features NaN.
    for name in S.VISION_NAMES:
        feats[name] = feats[name].copy()
        feats[name][gap] = np.nan
    feats["face_visible"][gap] = 0.0
    feats["face_conf"][gap] = 0.0
    return feats


def _motion(
    rng: np.random.Generator, n: int, cls: np.ndarray, dataset: str, cfg: dict
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Returns motion features and a bool [n] marking injected harsh pulses."""
    aggressive, drowsy = cls == S.CLASS_TO_ID["aggressive"], cls == S.CLASS_TO_ID["drowsy"]
    speed = np.clip(15 + np.cumsum(rng.normal(0, 0.05, n)), 5, 30)
    acc_long = np.gradient(speed, S.STEP_S) + rng.normal(0, 0.2, n)
    acc_lat = rng.normal(0, 0.3, n)
    pulses = _gap_mask(rng, n, 0.02, (10, 20)) & aggressive
    sign = rng.choice([-1.0, 1.0], n)
    acc_long = np.where(pulses, np.where(sign < 0, _BRAKE_MS2, _ACCEL_MS2), acc_long)
    turns = _gap_mask(rng, n, 0.01, (10, 20)) & aggressive
    acc_lat = np.where(turns, _TURN_MS2 * sign, acc_lat)
    yaw_rate = rng.normal(0, 2, n) + np.where(turns, 25.0 * sign, 0.0)

    if dataset == "uah":
        drift = np.where(drowsy, 0.6 * np.sin(np.arange(n) / 30.0), 0.0)
        lane_offset = np.clip(drift + rng.normal(0, 0.1, n), -5, 5)
        lane_offset[rng.random(n) > cfg["lane_detect_prob"]] = np.nan
        lateral = pd.Series(lane_offset).rolling(50, min_periods=10).std().to_numpy()
    else:  # no lane detection on own data; drift proxy from lateral acceleration
        lane_offset = np.full(n, np.nan)
        lateral = pd.Series(acc_lat * 0.1).rolling(50, min_periods=10).std().to_numpy()

    feats = {
        "acc_long": np.clip(acc_long, -50, 50),
        "acc_lat": np.clip(acc_lat, -50, 50),
        "acc_vert": rng.normal(0, 0.1, n),
        "jerk_long": np.clip(np.gradient(acc_long, S.STEP_S), -500, 500),
        "jerk_lat": np.clip(np.gradient(acc_lat, S.STEP_S), -500, 500),
        "yaw_rate": np.clip(yaw_rate, -360, 360),
        "roll": rng.normal(1, 0.5, n),
        "pitch": rng.normal(-3, 0.5, n),
        "speed": speed,
        "speed_var_5s": pd.Series(speed).rolling(50, min_periods=1).var().fillna(0).to_numpy(),
        "heading_rate": np.clip(yaw_rate + rng.normal(0, 1, n), -360, 360),
        "lane_offset": lane_offset,
        "lateral_drift_5s": np.clip(lateral, 0, 5),
        "gps_valid": (rng.random(n) > 0.02).astype(np.float64),
    }
    gap_len = tuple(int(s * S.STEP_HZ) for s in cfg["dropout"]["length_s"])
    gap = _gap_mask(rng, n, cfg["dropout"]["start_prob"], gap_len)
    for name in S.MOTION_NAMES:
        feats[name] = feats[name].copy()
        feats[name][gap] = np.nan
    return feats, pulses & ~gap


def _events(
    segments: list[tuple[int, int, int]], has_vision: bool, pulses: np.ndarray | None
) -> list[dict[str, Any]]:
    """One event per non-normal segment, typed by what the trip can observe."""
    events = []
    for start, end, c in segments:
        name = S.CLASSES[c]
        if name == "normal":
            continue
        onset, offset = start * S.STEP_S, end * S.STEP_S
        if name == "aggressive":
            hits = np.flatnonzero(pulses[start:end]) if pulses is not None else np.array([])
            if hits.size == 0:
                continue
            onset, offset = (start + hits[0]) * S.STEP_S, (start + hits[-1] + 1) * S.STEP_S
            etype = "harsh_brake"
        elif name == "drowsy":
            etype = "micro_sleep" if has_vision else "lane_weave"
        else:
            etype = "distraction_episode"
        events.append({"onset_s": round(onset, 3), "offset_s": round(offset, 3), "type": etype})
    return events


def make_stub_trip(
    rng: np.random.Generator, dataset: str, driver_id: str, trip_no: int, cfg: dict
) -> StubTrip:
    """Build one contract-valid stub trip."""
    pairing = cfg["datasets"][dataset]["pairing"]
    has_v, has_m = pairing in ("paired", "vision_only"), pairing in ("paired", "motion_only")
    duration_s = int(rng.integers(cfg["duration_s"][0], cfg["duration_s"][1] + 1))
    n = duration_s * S.STEP_HZ
    seg_steps = (cfg["segment_s"][0] * S.STEP_HZ, cfg["segment_s"][1] * S.STEP_HZ)
    segments = _segments(rng, n, cfg["class_probs"][dataset], seg_steps)
    cls = np.empty(n, dtype=np.int64)
    y = np.empty(n, dtype=np.int8)
    for start, end, c in segments:
        cls[start:end] = c
        y[start:end] = S.UNLABELED if rng.random() < cfg["unlabeled_prob"] else c

    cols: dict[str, np.ndarray] = {S.COL_T: (np.arange(n) * S.STEP_S).astype(np.float32)}
    vis = _vision(rng, n, cls, cfg) if has_v else {k: np.full(n, np.nan) for k in S.VISION_NAMES}
    mot, pulses = (
        _motion(rng, n, cls, dataset, cfg)
        if has_m
        else (
            {k: np.full(n, np.nan) for k in S.MOTION_NAMES},
            None,
        )
    )
    for name in S.VISION_NAMES:
        cols[S.VISION_PREFIX + name] = vis[name].astype(np.float32)
    for name in S.MOTION_NAMES:
        cols[S.MOTION_PREFIX + name] = mot[name].astype(np.float32)
    with np.errstate(invalid="ignore"):
        cols[S.COL_MASK_V] = (vis["face_visible"] == 1.0) & (vis["face_conf"] >= S.FACE_CONF_MIN)
    cols[S.COL_MASK_M] = ~np.isnan(mot["acc_long"])
    cols[S.COL_Y_CLS] = y
    frame = pd.DataFrame(cols)
    frame[S.COL_CONTRACT] = S.CONTRACT_VERSION

    all_nan = [c for c in (*S.VISION_COLUMNS, *S.MOTION_COLUMNS) if frame[c].isna().all()]
    trip_id = f"{dataset}_{driver_id}_{cfg['start_date']}_{trip_no:02d}"
    meta = {
        "trip_id": trip_id,
        "contract_version": S.CONTRACT_VERSION,
        "sampling_rates": {"common_clock_hz": S.STEP_HZ},
        "extractor_versions": {"stub": "driverisk.data.stubs"},
        "notes": "SYNTHETIC STUB DATA, not real driving.",
        S.META_ALL_NAN_KEY: all_nan,
    }
    counts = np.bincount(cls, minlength=S.K_CLASSES)
    row = {
        "trip_id": trip_id,
        "dataset": dataset,
        "driver_id": driver_id,
        "pairing": pairing,
        "duration_s": float(n * S.STEP_S),
        "n_steps": n,
        "behavior_tag": S.CLASSES[int(counts.argmax())],
        "road_type": "stub",
        "vehicle_id": "stub",
        "split": None,
        "source_version": "stub-v1",
        "contract_version": S.CONTRACT_VERSION,
    }
    return StubTrip(frame, _events(segments, has_v, pulses), meta, row)


def write_trip(trip: StubTrip, out_dir: Path) -> None:
    files = S.trip_files(trip.manifest_row["trip_id"])
    trip.frame.to_parquet(out_dir / files["parquet"], index=False)
    (out_dir / files["events"]).write_text(json.dumps(trip.events, indent=1), encoding="utf-8")
    (out_dir / files["meta"]).write_text(json.dumps(trip.meta, indent=1), encoding="utf-8")


def make_stub_dataset(
    out_root: str | Path, cfg: dict, datasets: tuple[str, ...] | None = None
) -> list[Path]:
    """Write stub trips to `<out_root>/<dataset>/` with a manifest and driver-disjoint splits.

    Returns the dataset directories written.
    """
    out_root = Path(out_root)
    rng = np.random.default_rng(cfg["seed"])
    written = []
    for dataset in datasets or tuple(cfg["datasets"]):
        spec = cfg["datasets"][dataset]
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        drivers = [f"d{i + 1:02d}" for i in range(spec["n_drivers"])]
        rows, per_driver = [], dict.fromkeys(drivers, 0)
        for k in range(spec["n_trips"]):
            driver = drivers[k % len(drivers)]
            per_driver[driver] += 1
            trip = make_stub_trip(rng, dataset, driver, per_driver[driver], cfg)
            write_trip(trip, out_dir)
            rows.append(trip.manifest_row)
        manifest = assign_driver_splits(pd.DataFrame(rows), cfg["split_fractions"], cfg["seed"])
        manifest.to_csv(out_dir / "manifest.csv", index=False)
        log.info("wrote %d stub trips to %s", len(rows), out_dir)
        written.append(out_dir)
    return written
