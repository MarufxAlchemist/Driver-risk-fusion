"""Inspect a downloaded UAH-DriveSet folder to fill the "Verified" column (03 section 3.1).

The folder layout comes from the dataset paper (R11 section III): one folder per driver, one
sub-folder per route named "Date(YYYYMMDDhhmmss)-Distance(Km)Driver-Behavior-Road", each with
files whose names start with RAW / PROC / EVENTS / SEMANTIC and whose first column is the
timestamp in seconds. Exact file names are NOT known to this code, so files are found by
prefix and anything that does not parse is reported, not guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Prefixes taken from the paper's file descriptions; matched case-insensitively.
FILE_PREFIXES = {
    "acc": "RAW_ACC",
    "gps": "RAW_GPS",
    "lane": "PROC_LANE",
    "vehicle": "PROC_VEHICLE",
    "osm": "PROC_OPEN",
    "lane_changes": "EVENTS_LIST_LANE",
    "inertial": "EVENTS_INERTIAL",
    "sem_final": "SEMANTIC_FINAL",
    "sem_online": "SEMANTIC_ONLINE",
}
FOLDER_RE = re.compile(r"^(\d{14})-([\d.]+)km-(D\d+)-([A-Za-z]+)-([A-Za-z]+)$")


@dataclass
class FileInfo:
    path: Path
    n_rows: int
    n_cols: int
    duration_s: float
    median_rate_hz: float
    col_min: list[float]
    col_max: list[float]


@dataclass
class TripInfo:
    folder: str
    parsed: bool
    date: str = ""
    distance_km: float = float("nan")
    driver: str = ""
    behavior: str = ""
    road: str = ""
    files: dict[str, FileInfo] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    has_video: bool = False


def read_table(path: Path) -> FileInfo:
    """Whitespace- or comma-separated numeric table, first column = timestamp (s)."""
    text = path.read_text(encoding="utf-8", errors="replace").replace(",", " ")
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    data = np.array(
        [[float(x) for x in r] for r in rows if all(_is_num(x) for x in r)], dtype=np.float64
    )
    if data.size == 0:
        return FileInfo(path, 0, 0, 0.0, float("nan"), [], [])
    t = data[:, 0]
    dt = np.diff(t)
    rate = 1.0 / float(np.median(dt[dt > 0])) if (dt > 0).any() else float("nan")
    return FileInfo(
        path, len(data), data.shape[1], float(t[-1] - t[0]), rate,
        np.nanmin(data, 0).round(3).tolist(), np.nanmax(data, 0).round(3).tolist(),
    )  # fmt: skip


def _is_num(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def find_file(folder: Path, prefix: str) -> Path | None:
    hits = sorted(p for p in folder.iterdir() if p.is_file() and p.name.upper().startswith(prefix))
    return hits[0] if hits else None


def inspect_trip(folder: Path) -> TripInfo:
    m = FOLDER_RE.match(folder.name)
    info = TripInfo(folder.name, parsed=m is not None)
    if m:
        info.date, info.distance_km = m[1], float(m[2])
        info.driver, info.behavior, info.road = m[3].upper(), m[4].upper(), m[5].upper()
    info.has_video = any(p.suffix.lower() in {".mp4", ".avi", ".mov"} for p in folder.iterdir())
    for key, prefix in FILE_PREFIXES.items():
        path = find_file(folder, prefix)
        if path is None:
            info.missing.append(key)
        else:
            info.files[key] = read_table(path)
    return info


def inspect_dataset(root: Path) -> list[TripInfo]:
    """Every folder (at any depth below `root`) that contains a RAW accelerometer file."""
    trips = []
    for d in sorted({p.parent for p in root.rglob("*") if p.is_file()}):
        if find_file(d, FILE_PREFIXES["acc"]) is not None:
            trips.append(inspect_trip(d))
    return trips
