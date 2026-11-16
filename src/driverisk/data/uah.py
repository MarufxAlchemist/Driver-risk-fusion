"""UAH-DriveSet loader (P1-B1): route folders -> contract trips (docs/03_data_contract.md).

STATUS: built against the documented format only. The dataset has NOT been downloaded, so every
layout detail below is an assumption from the UAH paper (R11) as recorded in docs/03 section 4.4
and docs/11 section 3.1, with column orders in `configs/uah.yaml` marked UNVERIFIED. Each
assumption is checked when a file is read, and a mismatch raises `UAHFormatError` with the file
and the expected value; nothing is guessed or repaired silently.

Layout (from `uah_inspect`, reused here): one folder per route named
"Date(YYYYMMDDhhmmss)-Distance(km)-Driver-Behavior-Road" (`uah_inspect.FOLDER_RE`), with text
files found by name prefix (`uah_inspect.FILE_PREFIXES`): RAW accelerometer, RAW GPS, PROC lane
detection, EVENTS inertial. First column = timestamp in seconds.

Conversion (contract 4.4):
* acceleration G -> m/s^2 (x 9.80665); axis map Z -> acc_long, Y -> acc_lat, X -> acc_vert,
  each with a configurable sign. The data is vehicle-aligned, so the phone alignment estimation
  of `features.motion` is NOT run (`motion.vehicle_frame_alignment`).
* GPS speed km/h -> m/s; speed and heading rate are held from the latest fix (assumption A19).
* yaw_rate = d(unwrap(yaw))/dt; roll and pitch are taken from the file (latest sample per step).
* lane_offset = lane-detector offset where the detector state is "detected", else NaN (the latest
  sample per step decides); lateral_drift_5s = its rolling std over 5 s.
* inertial events (braking / turning / acceleration) become harsh_brake / sharp_turn /
  harsh_accel with `source: "uah_inertial"`. They are logged only above 50 km/h, so events at a
  lower held GPS speed are dropped and the stretches where events can exist are written to meta.
* 10 Hz causal grid of `features.motion`: step k covers (k*0.1 - 0.1, k*0.1]; t = 0 is the first
  accelerometer sample. Vision columns are all NaN (pairing `motion_only`) and declared.
* y_cls = the trip-level behaviour mapped through `configs/label_map_uah.yaml`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.data.uah_inspect import FILE_PREFIXES, FOLDER_RE
from driverisk.features import motion as M
from driverisk.utils.config import load_yaml
from driverisk.utils.paths import configs_dir

log = logging.getLogger(__name__)

DATASET = "uah"
EVENT_SOURCE = "uah_inertial"
_EXTRACTOR = "driverisk.data.uah"

# Assumptions this loader makes that nobody has checked on the real files. Written into every
# trip's meta so the processed data carries its own caveat.
UNVERIFIED = (
    "folder name format Date-Distance(km)-Driver-Behavior-Road (uah_inspect.FOLDER_RE)",
    "file name prefixes RAW_ACC, RAW_GPS, PROC_LANE, EVENTS_INERTIAL",
    "column counts and orders in configs/uah.yaml",
    "accelerations in G, vehicle-aligned; Z longitudinal, Y lateral, X vertical; axis signs",
    "whether the vertical channel contains gravity",
    "GPS speed in km/h; course in degrees clockwise",
    "roll/pitch/yaw in degrees; yaw sign (must be left-positive)",
    "lane detector offset in metres and state code 2 = detected",
    "inertial event codes (1 brake, 2 turn, 3 accel) and levels (1-3)",
    "inertial events logged only above 50 km/h",
    "timestamps in seconds, shared by all files of a route",
)


class UAHFormatError(ValueError):
    """A UAH file or folder does not match the documented (assumed) format."""


@dataclass(frozen=True)
class RouteInfo:
    folder: Path
    timestamp: str  # YYYYMMDDhhmmss from the folder name
    distance_km: float
    driver_raw: str  # "D1"
    behaviour: str  # upper case, e.g. "NORMAL"
    road: str  # upper case, e.g. "MOTORWAY"

    @property
    def driver_id(self) -> str:
        return f"d{int(self.driver_raw[1:]):02d}"

    @property
    def day(self) -> str:
        return self.timestamp[:8]


@dataclass
class UAHTrip:
    frame: pd.DataFrame
    events: list[dict[str, Any]]
    meta: dict[str, Any]
    manifest_row: dict[str, Any]


@dataclass
class ProcessResult:
    trips: list[UAHTrip] = field(default_factory=list)
    failures: list[tuple[Path, str]] = field(default_factory=list)  # (route folder, error)
    manifest: pd.DataFrame | None = None


# --- configuration --------------------------------------------------------------------------
def load_uah_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_dir() / "uah.yaml")


def load_label_map(path: str | Path | None = None) -> dict[str, int]:
    """UAH behaviour (upper case) -> y_cls id, from `configs/label_map_uah.yaml`."""
    raw = load_yaml(path or configs_dir() / "label_map_uah.yaml")
    mapping = raw.get("behaviour_to_class")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{path or 'label_map_uah.yaml'} needs a behaviour_to_class mapping")
    out: dict[str, int] = {}
    for behaviour, cls in mapping.items():
        if cls is None:
            out[str(behaviour).upper()] = S.UNLABELED
        elif cls in S.CLASS_TO_ID:
            out[str(behaviour).upper()] = S.CLASS_TO_ID[cls]
        else:
            raise ValueError(f"label map: {behaviour} -> {cls!r} is not one of {S.CLASSES}")
    return out


# --- folders and files ----------------------------------------------------------------------
def parse_route_folder(folder: Path) -> RouteInfo:
    m = FOLDER_RE.match(folder.name)
    if m is None:
        raise UAHFormatError(
            f"{folder}: folder name does not match Date-Distance(km)-Driver-Behavior-Road "
            f"({FOLDER_RE.pattern}); the format is from the paper and UNVERIFIED"
        )
    return RouteInfo(folder, m[1], float(m[2]), m[3].upper(), m[4].upper(), m[5].upper())


def find_route_folders(root: str | Path) -> list[Path]:
    """Every folder below `root` holding a RAW accelerometer file (as `uah_inspect` does)."""
    root = Path(root)
    prefix = FILE_PREFIXES["acc"]
    folders = {
        p.parent for p in root.rglob("*") if p.is_file() and p.name.upper().startswith(prefix)
    }
    return sorted(folders)


def find_route_file(folder: Path, kind: str, cfg: dict[str, Any]) -> Path | None:
    """The one file of `kind` in `folder` by name prefix; error if several match."""
    spec = cfg["files"][kind]
    prefix = spec["prefix"].upper()
    hits = sorted(p for p in folder.iterdir() if p.is_file() and p.name.upper().startswith(prefix))
    if len(hits) > 1:
        names = [p.name for p in hits]
        raise UAHFormatError(f"{folder}: more than one file starts with {prefix}: {names}")
    if not hits:
        if spec.get("required", False):
            raise UAHFormatError(f"{folder}: no file starting with {prefix} (required)")
        return None
    return hits[0]


def read_numeric_table(path: Path, n_cols: int, skip_header_lines: int = 0) -> np.ndarray:
    """Whitespace- or comma-separated numbers, exactly `n_cols` per row. Errors name the line."""
    rows: list[list[float]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if lineno <= skip_header_lines or not line.strip():
            continue
        parts = line.replace(",", " ").split()
        try:
            values = [float(x) for x in parts]
        except ValueError:
            raise UAHFormatError(
                f"{path.name} line {lineno}: not all numeric: {line.strip()[:80]!r}"
            ) from None
        if len(values) != n_cols:
            raise UAHFormatError(
                f"{path.name} line {lineno}: {len(values)} columns, expected {n_cols} "
                "(configs/uah.yaml, UNVERIFIED column map)"
            )
        rows.append(values)
    if not rows:
        raise UAHFormatError(f"{path.name}: no data rows")
    table = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(table).all():
        raise UAHFormatError(f"{path.name}: contains NaN or inf")
    return table


def _load(folder: Path, kind: str, cfg: dict[str, Any]) -> tuple[Path, dict[str, np.ndarray]]:
    path = find_route_file(folder, kind, cfg)
    if path is None:
        return Path(), {}
    spec = cfg["files"][kind]
    table = read_numeric_table(path, int(spec["n_cols"]), int(cfg.get("skip_header_lines", 0)))
    cols = {name: table[:, int(i)] for name, i in spec["columns"].items()}
    d = np.diff(cols["t"])
    strict = kind in ("acc", "gps")  # these are differenced and searched; the others are binned
    bad = (d <= 0) if strict else (d < 0)
    if bad.any():
        first = int(np.argmax(bad)) + 2
        order = "strictly increasing" if strict else "non-decreasing"
        raise UAHFormatError(f"{path.name}: timestamps not {order} (first at data row {first})")
    return path, cols


# --- conversion -----------------------------------------------------------------------------
def _vehicle_acceleration(acc: dict[str, np.ndarray], cfg: dict[str, Any], name: str):
    a_cfg = cfg["acceleration"]
    suffix = {"raw": "raw", "kf": "kf"}.get(a_cfg["source"])
    if suffix is None:
        raise ValueError(f"acceleration.source must be raw or kf, got {a_cfg['source']!r}")
    g_cols = [acc[f"a{ax}_{s}"] for ax in "xyz" for s in ("raw", "kf")]
    peak = float(np.max(np.abs(np.stack(g_cols))))
    if peak > a_cfg["max_abs_g"]:
        raise UAHFormatError(
            f"{name}: |acceleration| reaches {peak:.2f}, above {a_cfg['max_abs_g']} G: the "
            "values look not in G (contract 4.4 assumes G, UNVERIFIED)"
        )
    out = []
    for axis in ("long", "lat", "vert"):
        spec = a_cfg["axes"][axis]
        if spec["src"] not in ("x", "y", "z") or spec["sign"] not in (1, -1):
            raise ValueError(f"acceleration.axes.{axis} needs src x/y/z and sign +1/-1: {spec}")
        out.append(spec["sign"] * acc[f"a{spec['src']}_{suffix}"] * a_cfg["g_to_mps2"])
    veh = np.stack(out, axis=1)
    gravity = bool(a_cfg["gravity_in_vertical"])
    vert = veh[:, 2] - (M.G_STD if gravity else 0.0)
    med = float(np.median(vert))
    if abs(med) > 0.5 * M.G_STD:
        raise UAHFormatError(
            f"{name}: median vertical acceleration after gravity handling is {med:.2f} m/s^2; "
            f"gravity_in_vertical={gravity} or the vertical axis/sign in configs/uah.yaml is "
            "wrong for this file (UNVERIFIED)"
        )
    return veh, gravity


def _clean_course(course: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Course outside [0, 360] (iOS reports -1 when unknown) is held from the last valid one.

    None when no course is valid (heading_rate is then NaN, never a made-up 0).
    """
    bad = ~((course >= 0) & (course <= 360))
    if bad.all():
        return None, int(bad.sum())
    s = pd.Series(np.where(bad, np.nan, course)).ffill()
    return s.fillna(s.dropna().iloc[0]).to_numpy(), int(bad.sum())


def _runs_s(flag: np.ndarray, t: np.ndarray) -> list[list[float]]:
    padded = np.r_[False, flag, False].astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [
        [round(float(t[a]), 3), round(float(t[b]), 3)] for a, b in zip(starts, ends, strict=True)
    ]


def _held_speed(t: np.ndarray, t_fix: np.ndarray, kmh: np.ndarray, max_age: float) -> np.ndarray:
    """Speed of the latest fix at or before each time (NaN when none or older than `max_age`)."""
    pos = np.searchsorted(t_fix, t + 1e-9, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = pos >= 0
    ok[ok] = t[ok] - t_fix[pos[ok]] <= max_age + 1e-9
    out[ok] = kmh[pos[ok]]
    return out


def _inertial_events(
    ev: dict[str, np.ndarray], t0: float, t_gps: np.ndarray, speed_kmh: np.ndarray,
    duration_s: float, cfg: dict[str, Any], name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # fmt: skip
    e_cfg = cfg["inertial_events"]
    type_map = {int(k): v for k, v in e_cfg["type_map"].items()}
    stats = {"n_read": 0, "n_dropped_low_speed": 0, "n_dropped_level": 0,
             "n_dropped_outside_trip": 0}  # fmt: skip
    if not ev:
        return [], stats
    t_rel = ev["t"] - t0
    held = _held_speed(t_rel, t_gps, speed_kmh, M.MotionConfig().gps_max_age_s)
    events = []
    for i in range(len(t_rel)):
        stats["n_read"] += 1
        code, level = ev["type"][i], ev["level"][i]
        if code != int(code) or int(code) not in type_map:
            raise UAHFormatError(f"{name}: inertial event type code {code:g} not in {type_map}")
        if level != int(level) or not 1 <= level <= 3:
            raise UAHFormatError(f"{name}: inertial event level {level:g} not in 1-3")
        if level < e_cfg["min_level"]:
            stats["n_dropped_level"] += 1
            continue
        if not held[i] > e_cfg["min_speed_kmh"]:  # NaN (no fix) fails too
            stats["n_dropped_low_speed"] += 1
            continue
        onset = float(t_rel[i])
        offset = onset + float(e_cfg["duration_s"])
        if onset < 0 or offset > duration_s + S.T_STEP_TOLERANCE_S:
            stats["n_dropped_outside_trip"] += 1
            continue
        events.append({
            "onset_s": round(onset, 3), "offset_s": round(offset, 3),
            "type": type_map[int(code)], "source": EVENT_SOURCE,
            "uah_level": int(level), "speed_kmh": round(float(held[i]), 1),
        })  # fmt: skip
    if stats["n_dropped_low_speed"]:
        log.warning("%s: %d inertial events at or below %s km/h dropped", name,
                    stats["n_dropped_low_speed"], e_cfg["min_speed_kmh"])  # fmt: skip
    return events, stats


def _median_rate(t: np.ndarray) -> float | None:
    return round(1.0 / float(np.median(np.diff(t))), 2) if len(t) > 1 else None


def convert_route(
    folder: Path, trip_id: str, cfg: dict[str, Any], label_map: dict[str, int]
) -> UAHTrip:
    """Convert one UAH route folder into one contract trip (not written to disk)."""
    route = parse_route_folder(folder)
    if route.behaviour not in label_map:
        raise UAHFormatError(
            f"{folder.name}: behaviour {route.behaviour} not in configs/label_map_uah.yaml "
            f"({sorted(label_map)})"
        )
    acc_path, acc = _load(folder, "acc", cfg)
    gps_path, gps = _load(folder, "gps", cfg)
    lane_path, lane = _load(folder, "lane", cfg)
    ev_path, ev = _load(folder, "inertial", cfg)

    t0 = float(acc["t"][0])
    t_acc, t_gps = acc["t"] - t0, gps["t"] - t0
    veh, gravity = _vehicle_acceleration(acc, cfg, acc_path.name)

    speed_kmh = gps["speed_kmh"]
    s_cfg = cfg["speed"]
    if speed_kmh.max() > s_cfg["max_kmh"]:
        raise UAHFormatError(
            f"{gps_path.name}: speed reaches {speed_kmh.max():.1f}, above {s_cfg['max_kmh']} "
            "km/h: not km/h? (UNVERIFIED unit)"
        )
    valid_fix = speed_kmh >= 0  # iOS reports a negative speed when it has none
    course, n_bad_course = _clean_course(gps["course"][valid_fix])
    t_fix, kmh = t_gps[valid_fix], speed_kmh[valid_fix]

    att = cfg["attitude"]
    result = M.extract_motion_features(
        t_acc, veh, t_fix, kmh * s_cfg["kmh_to_mps"],
        heading_deg=course,
        yaw_deg=(t_acc, att["yaw_sign"] * acc["yaw"]),
        lane_offset=(np.array([]), np.array([])),  # filled below with the detector rule
        alignment=M.vehicle_frame_alignment(gravity_included=gravity),
    )  # fmt: skip
    frame = result.frame
    n = len(frame)
    mask_m = frame[S.COL_MASK_M].to_numpy(bool)
    roll = M.resample_last(t_acc, att["roll_sign"] * acc["roll"], n)
    pitch = M.resample_last(t_acc, att["pitch_sign"] * acc["pitch"], n)
    frame["m_roll"] = np.where(mask_m, roll, np.nan).astype(np.float32)
    frame["m_pitch"] = np.where(mask_m, pitch, np.nan).astype(np.float32)

    l_cfg = cfg["lane"]
    n_lane_too_far = 0
    if lane:
        x = np.where(lane["state"] == l_cfg["valid_state"], lane["x_offset_m"], np.nan)
        too_far = np.abs(x) > l_cfg["max_abs_offset_m"]
        n_lane_too_far = int(too_far.sum())
        x[too_far] = np.nan
        offset = M.resample_last(lane["t"] - t0, x, n, keep_nan=True)
    else:
        offset = np.full(n, np.nan)
    mc = M.MotionConfig()
    window = round(mc.window_s / S.STEP_S)
    drift = pd.Series(offset).rolling(window, min_periods=mc.min_window_steps).std(ddof=0)
    frame["m_lane_offset"] = offset.astype(np.float32)
    frame["m_lateral_drift_5s"] = drift.to_numpy().astype(np.float32)

    for col in S.VISION_COLUMNS:
        frame[col] = np.full(n, np.nan, dtype=np.float32)
    frame[S.COL_MASK_V] = np.zeros(n, dtype=bool)
    frame[S.COL_Y_CLS] = np.full(n, label_map[route.behaviour], dtype=np.int8)
    frame[S.COL_CONTRACT] = S.CONTRACT_VERSION
    frame = frame[list(S.PARQUET_COLUMNS)]

    duration_s = float(n * S.STEP_S)
    events, ev_stats = _inertial_events(
        ev, t0, t_fix, kmh, duration_s, cfg, f"{folder.name}/{ev_path.name}"
    )
    grid = np.arange(n) * S.STEP_S
    above = frame["m_speed"].to_numpy(np.float64) * 3.6 > cfg["inertial_events"]["min_speed_kmh"]
    ev_stats.update(
        file_present=bool(ev),
        min_speed_kmh=float(cfg["inertial_events"]["min_speed_kmh"]),
        min_level=int(cfg["inertial_events"]["min_level"]),
        logged_segments_s=_runs_s(above, grid),
    )

    all_nan = [c for c in (*S.VISION_COLUMNS, *S.MOTION_COLUMNS) if frame[c].isna().all()]
    meta = {
        "trip_id": trip_id,
        "contract_version": S.CONTRACT_VERSION,
        "sampling_rates": {
            "common_clock_hz": S.STEP_HZ,
            "acc_median_hz": _median_rate(t_acc),
            "gps_median_hz": _median_rate(t_fix),
            "lane_median_hz": _median_rate(lane["t"]) if lane else None,
        },
        "extractor_versions": {"uah_loader": _EXTRACTOR, "uah_config": cfg.get("version")},
        "notes": (
            "UAH-DriveSet route converted by driverisk.data.uah against the documented format; "
            "the format is UNVERIFIED until scripts/inspect_uah.py has run on the download. "
            "Motion only (no driver camera). Inertial events are logged only above 50 km/h: "
            "outside inertial_events.logged_segments_s their absence means 'not logged'. "
            "y_cls is the trip-level behaviour."
        ),
        S.META_ALL_NAN_KEY: all_nan,
        "events_annotated": bool(ev),
        "pairing": "motion_only",
        "uah": {
            "folder": folder.name,
            "driver": route.driver_raw,
            "behaviour": route.behaviour,
            "road": route.road,
            "distance_km": route.distance_km,
            "start": route.timestamp,
            "files": {
                k: p.name
                for k, p in (
                    ("acc", acc_path),
                    ("gps", gps_path),
                    ("lane", lane_path),
                    ("inertial", ev_path),
                )
                if p.name
            },
            "t0_source_s": t0,
        },  # fmt: skip
        "conversion": {
            "acceleration": cfg["acceleration"],
            "attitude": att,
            "speed_kmh_to_mps": s_cfg["kmh_to_mps"],
            "lane": l_cfg,
            "gps_rows_negative_speed_dropped": int((~valid_fix).sum()),
            "gps_course_invalid_held": n_bad_course,
            "lane_samples_beyond_max_offset": n_lane_too_far,
        },
        "alignment": result.alignment.to_meta(),
        "inertial_events": ev_stats,
        "unverified_assumptions": list(UNVERIFIED),
    }
    row = {
        "trip_id": trip_id,
        "dataset": DATASET,
        "driver_id": route.driver_id,
        "pairing": "motion_only",
        "duration_s": duration_s,
        "n_steps": n,
        "behavior_tag": route.behaviour.lower(),
        "road_type": route.road.lower(),
        "vehicle_id": cfg.get("vehicle_id", "unknown"),
        "split": None,
        "source_version": cfg.get("source_version", "unknown"),
        "contract_version": S.CONTRACT_VERSION,
    }
    return UAHTrip(frame, events, meta, row)


# --- whole dataset --------------------------------------------------------------------------
def assign_trip_ids(routes: list[RouteInfo]) -> dict[Path, str]:
    """`uah_<dNN>_<yyyymmdd>_<nn>`, nn counting a driver's routes of one day in time order."""
    ids: dict[Path, str] = {}
    counter: dict[tuple[str, str], int] = {}
    for r in sorted(routes, key=lambda r: (r.driver_id, r.timestamp, r.folder.name)):
        key = (r.driver_id, r.day)
        counter[key] = counter.get(key, 0) + 1
        ids[r.folder] = f"{DATASET}_{r.driver_id}_{r.day}_{counter[key]:02d}"
    return ids


def write_trip(trip: UAHTrip, out_dir: Path) -> None:
    files = S.trip_files(trip.manifest_row["trip_id"])
    trip.frame.to_parquet(out_dir / files["parquet"], index=False)
    (out_dir / files["events"]).write_text(json.dumps(trip.events, indent=1), encoding="utf-8")
    (out_dir / files["meta"]).write_text(json.dumps(trip.meta, indent=1), encoding="utf-8")


def _existing_outputs(out_dir: Path) -> list[Path]:
    return sorted(
        [*out_dir.glob(f"{DATASET}_*.parquet"), *out_dir.glob(f"{DATASET}_*_events.json"),
         *out_dir.glob(f"{DATASET}_*_meta.json"), *out_dir.glob("manifest.csv")]
    )  # fmt: skip


def process_dataset(
    root: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
    label_map: dict[str, int] | None = None,
    *,
    keep_going: bool = False,
    overwrite: bool = False,
) -> ProcessResult:
    """Convert every route under `root` into `out_dir` and write `manifest.csv`.

    Stops at the first `UAHFormatError` unless `keep_going`, which records it and continues.
    Refuses to write into a folder that already holds UAH outputs unless `overwrite`.
    """
    root, out_dir = Path(root), Path(out_dir)
    cfg = cfg or load_uah_config()
    label_map = label_map or load_label_map()
    folders = find_route_folders(root)
    if not folders:
        raise UAHFormatError(f"{root}: no folder with a {FILE_PREFIXES['acc']}* file")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_outputs(out_dir)
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {len(existing)} UAH output files")
    for p in existing:
        p.unlink()

    result = ProcessResult()
    routes = []
    for folder in folders:
        try:
            routes.append(parse_route_folder(folder))
        except UAHFormatError as exc:
            if not keep_going:
                raise
            result.failures.append((folder, str(exc)))
    ids = assign_trip_ids(routes)
    for r in sorted(routes, key=lambda r: ids[r.folder]):
        try:
            trip = convert_route(r.folder, ids[r.folder], cfg, label_map)
        except UAHFormatError as exc:
            if not keep_going:
                raise
            log.error("%s", exc)
            result.failures.append((r.folder, str(exc)))
            continue
        write_trip(trip, out_dir)
        result.trips.append(trip)
    manifest = pd.DataFrame(
        [t.manifest_row for t in result.trips], columns=list(S.MANIFEST_COLUMNS)
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    result.manifest = manifest
    log.info("wrote %d UAH trips to %s (%d failed)", len(result.trips), out_dir,
             len(result.failures))  # fmt: skip
    return result


def summarize_by_behaviour(out_dir: str | Path) -> pd.DataFrame:
    """Per trip-level behaviour: trips, minutes, minutes above 50 km/h, events per type.

    `events_per_h_logged` divides by the time above the logging speed, where events can exist.
    """
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "manifest.csv", dtype={"driver_id": str})
    types = ("harsh_accel", "harsh_brake", "sharp_turn")
    rows = []
    for _, m in manifest.iterrows():
        files = S.trip_files(str(m["trip_id"]))
        events = json.loads((out_dir / files["events"]).read_text(encoding="utf-8"))
        meta = json.loads((out_dir / files["meta"]).read_text(encoding="utf-8"))
        segs = meta.get("inertial_events", {}).get("logged_segments_s", [])
        logged_s = sum(b - a + S.STEP_S for a, b in segs)
        row = {"behaviour": m["behavior_tag"], "trips": 1, "duration_min": m["duration_s"] / 60,
               "logged_min": logged_s / 60, "events": len(events)}  # fmt: skip
        for k in types:
            row[k] = sum(e["type"] == k for e in events)
        rows.append(row)
    cols = ["trips", "duration_min", "logged_min", "events", *types]
    if not rows:
        return pd.DataFrame(columns=["events_per_h_logged", *cols])
    out = pd.DataFrame(rows).groupby("behaviour")[cols].sum()
    hours = out["logged_min"] / 60
    out["events_per_h_logged"] = out["events"] / hours.where(hours > 0)
    return out
