"""DMD sessions -> contract trips: face-video features + annotation labels (P1-A3).

One DMD recording session (a driver, one protocol session such as s5 = drowsiness) becomes one
contract trip: vision features from the face video (via `features.vision`), the class and event
labels from the OpenLABEL annotation (via `data.dmd`), all motion columns NaN (`pairing =
vision_only`), written as `dmd_<dNN>_<yyyymmdd>_<nn>.parquet` + events + meta + a manifest row.

Real naming (checked 2026-11-18): `gA_1_s5_2019-03-14T14;26;17+01;00_rgb_face.mp4` and
`..._rgb_ann_drowsiness.json`: group, subject number, session, date, time, camera/annotation.
Subject numbers run across groups (gA 1-5, gB 6-10: assumption), so `d<subject>` is the driver id;
the same person in the distraction and drowsiness sets therefore gets the same driver key, and
splits keep them together.

MediaPipe is slow (about 35 frames/s here), so landmarks are cached per session as `.npz`
(`save_landmark_cache`); `process_dataset` reads only the cache. Filling the cache needs the
optional `vision` dependency group and is done by `scripts/process_dmd.py`.

Licence (CC BY-NC-ND 4.0): the processed files are derived material and must not be shared or
committed; every trip's meta says so.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import dmd as D
from driverisk.data import schema as S
from driverisk.features import vision as V

log = logging.getLogger(__name__)
DATASET = "dmd"
_EXTRACTOR = "driverisk.data.dmd_trips 0.1"
NOTE = (
    "DMD (Vicomtech) trip built from the face video and the OpenLABEL annotation. Vision only: "
    "all motion columns are NaN. y_cls is -1 wherever the annotation gives no class (the "
    "drowsiness files have no 'normal' label). Derived from CC BY-NC-ND 4.0 material: academic use "
    "only, do not share or commit."
)

_ANN_RE = re.compile(r"^(?P<stem>.+)_rgb_ann_(?P<kind>[A-Za-z]+)\.json$")
_STEM_RE = re.compile(
    r"^(?P<group>g[A-Z])_(?P<subject>\d+)_s(?P<session>\d+)_(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T"
)


class DmdSessionError(ValueError):
    """A DMD file name or file the loader cannot place; the message names it."""


@dataclass(frozen=True)
class SessionInfo:
    stem: str  # everything before `_rgb_`
    group: str
    subject: int
    session: int
    date: str  # yyyymmdd
    annotation: Path
    video: Path | None  # the `_rgb_face.mp4`, if present
    annotation_type: str  # drowsiness | distraction | gaze ...


@dataclass
class DmdTrip:
    frame: pd.DataFrame
    events: list[dict[str, Any]]
    meta: dict[str, Any]
    manifest_row: dict[str, Any]


@dataclass
class ProcessResult:
    trips: list[DmdTrip] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)  # (session stem, error)
    manifest: pd.DataFrame | None = None


@dataclass(frozen=True)
class LandmarkCache:
    t: np.ndarray  # [F] seconds, frame_index / fps
    landmarks: np.ndarray  # [F, 478, 3], NaN where no face
    matrices: np.ndarray  # [F, 4, 4]
    fps: float
    size: tuple[int, int]  # (width, height)


# --- sessions and ids ------------------------------------------------------------------------
def find_sessions(root: str | Path) -> list[SessionInfo]:
    """Every annotation file under `root`, paired with the face video of the same session."""
    root = Path(root)
    sessions: list[SessionInfo] = []
    for ann in sorted(root.rglob("*_rgb_ann_*.json")):
        m = _ANN_RE.match(ann.name)
        s = _STEM_RE.match(m["stem"]) if m else None
        if m is None or s is None:
            raise DmdSessionError(
                f"{ann}: name does not look like <group>_<subject>_s<session>_<date>T..._rgb_ann_"
                "<type>.json (the DMD naming checked on 2026-11-18)"
            )
        video = ann.with_name(f"{m['stem']}_rgb_face.mp4")
        sessions.append(
            SessionInfo(
                stem=m["stem"], group=s["group"], subject=int(s["subject"]),
                session=int(s["session"]), date=f"{s['y']}{s['m']}{s['d']}",
                annotation=ann, video=video if video.is_file() else None,
                annotation_type=m["kind"].lower(),
            )
        )  # fmt: skip
    return sorted(sessions, key=lambda x: x.stem)


def assign_ids(sessions: list[SessionInfo]) -> dict[str, tuple[str, str]]:
    """`stem -> (driver_id, trip_id)`; nn counts a driver's sessions per day by session no."""
    out: dict[str, tuple[str, str]] = {}
    counter: dict[tuple[str, str], int] = {}
    for s in sorted(sessions, key=lambda s: (s.subject, s.date, s.session, s.stem)):
        driver = f"d{s.subject:02d}"
        counter[(driver, s.date)] = counter.get((driver, s.date), 0) + 1
        out[s.stem] = (driver, f"{DATASET}_{driver}_{s.date}_{counter[(driver, s.date)]:02d}")
    return out


# --- landmark cache --------------------------------------------------------------------------
def cache_path(cache_dir: str | Path, stem: str) -> Path:
    """`;` and `+` of the session time stamp are replaced so the name is safe on every OS."""
    return Path(cache_dir) / (stem.replace(";", "-").replace("+", "p") + ".npz")


def save_landmark_cache(
    path: Path, landmarks: np.ndarray, matrices: np.ndarray, fps: float, size: tuple[int, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, landmarks=np.asarray(landmarks, dtype=np.float32),
        matrices=np.asarray(matrices, dtype=np.float32), fps=float(fps),
        width=int(size[0]), height=int(size[1]),
    )  # fmt: skip


def load_landmark_cache(path: Path) -> LandmarkCache:
    with np.load(path) as z:
        lm, mats = z["landmarks"].astype(np.float64), z["matrices"].astype(np.float64)
        fps, size = float(z["fps"]), (int(z["width"]), int(z["height"]))
    return LandmarkCache(np.arange(len(lm)) / fps, lm, mats, fps, size)


# --- one trip --------------------------------------------------------------------------------
def build_dmd_trip(
    session: SessionInfo,
    t: np.ndarray,
    landmarks: np.ndarray,
    matrices: np.ndarray | None,
    image_size: tuple[int, int],
    fps: float,
    annotation: dict[str, Any],
    label_map: D.DmdLabelMap,
    trip_id: str,
    driver_id: str,
    *,
    vision_config: dict[str, Any] | None = None,
    ear_threshold: float | None = None,
    frame_shift: int = 0,
) -> DmdTrip:
    """Vision features of the face video + labels of the annotation -> one contract trip."""
    cfg = vision_config or V.load_vision_config()
    vis = V.extract_vision_features(
        t, landmarks, matrices, config=cfg, ear_threshold=ear_threshold, image_size=image_size
    )
    frame = vis.frame.copy()
    n = len(frame)
    labels = D.dmd_labels(annotation, label_map, fps, n_steps=n, frame_shift=frame_shift)

    for col in S.MOTION_COLUMNS:
        frame[col] = np.float32("nan")
    frame[S.COL_MASK_M] = np.zeros(n, dtype=bool)
    frame[S.COL_Y_CLS] = labels.y_cls.astype(np.int8)
    frame[S.COL_CONTRACT] = S.CONTRACT_VERSION
    frame = frame[list(S.PARQUET_COLUMNS)]

    r = labels.report
    meta: dict[str, Any] = {
        "trip_id": trip_id,
        "contract_version": S.CONTRACT_VERSION,
        "sampling_rates": {"common_clock_hz": S.STEP_HZ, "video_fps": round(float(fps), 3)},
        "extractor_versions": {
            "dmd_trips": _EXTRACTOR, "vision": V.EXTRACTOR, "vision_config": cfg.get("version"),
            "label_map": label_map.status,
        },  # fmt: skip
        "notes": NOTE,
        S.META_ALL_NAN_KEY: [*S.MOTION_COLUMNS, *vis.all_nan_columns],
        "events_annotated": True,
        "pairing": "vision_only",
        **vis.meta,
        "dmd": {
            "group": session.group, "subject": session.subject, "session": session.session,
            "date": session.date, "annotation": session.annotation.name,
            "annotation_type": session.annotation_type, "video_fps": round(float(fps), 3),
            "n_frames_video": int(len(t)), "n_frames_annotation": int(r.n_frames),
            "n_frames_annotation_source": r.n_frames_source, "frame_shift": int(frame_shift),
            "label_report": {
                "unknown_labels": r.unknown_labels, "ignored_labels": r.ignored_labels,
                "rejected_events": len(r.rejected_events), "conflict_frames": r.conflict_frames,
                "clipped_intervals": r.clipped_intervals, "problems": r.problems,
            },
        },
    }  # fmt: skip
    row = {
        "trip_id": trip_id, "dataset": DATASET, "driver_id": driver_id,
        "pairing": "vision_only", "duration_s": round(n * S.STEP_S, 3), "n_steps": n,
        "behavior_tag": f"{session.annotation_type}_s{session.session}",
        "road_type": "unknown", "vehicle_id": "unknown", "split": "",
        "source_version": "DMD RGB release 2026-05-14 (readme date)",
        "contract_version": S.CONTRACT_VERSION,
    }  # fmt: skip
    return DmdTrip(frame, labels.events, meta, row)


# --- dataset ---------------------------------------------------------------------------------
def write_trip(trip: DmdTrip, out_dir: Path) -> None:
    files = S.trip_files(trip.manifest_row["trip_id"])
    trip.frame.to_parquet(out_dir / files["parquet"], index=False)
    (out_dir / files["events"]).write_text(json.dumps(trip.events, indent=1), encoding="utf-8")
    (out_dir / files["meta"]).write_text(json.dumps(trip.meta, indent=1), encoding="utf-8")


def process_dataset(
    root: str | Path,
    out_dir: str | Path,
    cache_dir: str | Path,
    label_map: D.DmdLabelMap | None = None,
    *,
    vision_config: dict[str, Any] | None = None,
    frame_shift: int = 0,
    keep_going: bool = False,
    overwrite: bool = False,
) -> ProcessResult:
    """Convert every session under `root` that has cached landmarks; write the manifest."""
    root, out_dir, cache_dir = Path(root), Path(out_dir), Path(cache_dir)
    label_map = label_map or D.load_label_map()
    sessions = find_sessions(root)
    if not sessions:
        raise DmdSessionError(f"{root}: no *_rgb_ann_*.json annotation file found")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [*out_dir.glob(f"{DATASET}_*"), *out_dir.glob("manifest.csv")]
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {len(existing)} DMD output files")
    ids = assign_ids(sessions)
    result = ProcessResult()
    for s in sessions:
        driver_id, trip_id = ids[s.stem]
        try:
            path = cache_path(cache_dir, s.stem)
            if not path.is_file():
                raise DmdSessionError(
                    f"{s.stem}: no cached landmarks at {path.name}; run scripts/process_dmd.py "
                    "(needs the optional vision group) to extract them from the face video"
                )
            c = load_landmark_cache(path)
            doc = D.load_openlabel(s.annotation)
            trip = build_dmd_trip(
                s, c.t, c.landmarks, c.matrices, c.size, c.fps, doc, label_map, trip_id,
                driver_id, vision_config=vision_config, frame_shift=frame_shift,
            )  # fmt: skip
        except (DmdSessionError, D.UnknownLabelError, ValueError) as exc:
            if not keep_going:
                raise
            log.error("%s failed: %s", s.stem, exc)
            result.failures.append((s.stem, str(exc)))
            continue
        write_trip(trip, out_dir)
        result.trips.append(trip)
    manifest = pd.DataFrame(
        [t.manifest_row for t in result.trips], columns=list(S.MANIFEST_COLUMNS)
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    result.manifest = manifest
    log.info(
        "wrote %d DMD trips to %s (%d failed)", len(result.trips), out_dir, len(result.failures)
    )
    return result
