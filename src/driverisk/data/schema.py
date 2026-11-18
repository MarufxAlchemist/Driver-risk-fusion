"""Data contract constants (docs/03_data_contract.md, contract version 0.1).

This module is the single code source of truth for feature names, counts, units,
physical ranges, label IDs, column names and tensor shapes. Everything else
imports from here. Changing anything in this file is a contract change: bump
`CONTRACT_VERSION` and update docs/03_data_contract.md in the same PR.

Ranges are the generous physical bounds used by the validator (contract section 10),
not typical values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CONTRACT_VERSION = "0.1"

# --- Clock and windows (contract sections 1-2, decision D1) -------------------------------
STEP_HZ = 10
STEP_S = 1.0 / STEP_HZ
WINDOW_STEPS = 50  # T: 5 s
STRIDE_STEPS = 10  # 1 s
T_STEP_TOLERANCE_S = 1e-3  # contract section 10
HORIZONS_S: tuple[int, ...] = (2, 3, 5)  # y_risk columns (contract section 7)

FeatureKind = Literal["continuous", "flag"]


@dataclass(frozen=True)
class FeatureSpec:
    """One feature column. `lo`/`hi` are inclusive validator bounds."""

    index: int
    name: str
    unit: str
    kind: FeatureKind
    lo: float
    hi: float


def _flag(index: int, name: str) -> FeatureSpec:
    return FeatureSpec(index, name, "{0,1}", "flag", 0.0, 1.0)


# --- Vision features X_v, Dv = 16 (contract section 4.1) ------------------------------------
VISION_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(0, "ear_left", "ratio", "continuous", 0.0, 1.0),
    FeatureSpec(1, "ear_right", "ratio", "continuous", 0.0, 1.0),
    FeatureSpec(2, "ear_mean", "ratio", "continuous", 0.0, 1.0),
    FeatureSpec(3, "mar", "ratio", "continuous", 0.0, 2.0),
    _flag(4, "blink"),
    _flag(5, "eye_closed"),
    FeatureSpec(6, "perclos_30s", "fraction", "continuous", 0.0, 1.0),
    FeatureSpec(7, "perclos_60s", "fraction", "continuous", 0.0, 1.0),
    FeatureSpec(8, "head_yaw", "deg", "continuous", -180.0, 180.0),
    FeatureSpec(9, "head_pitch", "deg", "continuous", -90.0, 90.0),
    FeatureSpec(10, "head_roll", "deg", "continuous", -180.0, 180.0),
    FeatureSpec(11, "gaze_yaw", "deg", "continuous", -180.0, 180.0),
    FeatureSpec(12, "gaze_pitch", "deg", "continuous", -90.0, 90.0),
    _flag(13, "off_road_gaze"),
    FeatureSpec(14, "face_conf", "[0,1]", "continuous", 0.0, 1.0),
    _flag(15, "face_visible"),
)

# --- Motion features X_m, Dm = 14 (contract section 4.2) ------------------------------------
MOTION_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(0, "acc_long", "m/s^2", "continuous", -50.0, 50.0),
    FeatureSpec(1, "acc_lat", "m/s^2", "continuous", -50.0, 50.0),
    FeatureSpec(2, "acc_vert", "m/s^2", "continuous", -50.0, 50.0),
    FeatureSpec(3, "jerk_long", "m/s^3", "continuous", -500.0, 500.0),
    FeatureSpec(4, "jerk_lat", "m/s^3", "continuous", -500.0, 500.0),
    FeatureSpec(5, "yaw_rate", "deg/s", "continuous", -360.0, 360.0),
    FeatureSpec(6, "roll", "deg", "continuous", -180.0, 180.0),
    FeatureSpec(7, "pitch", "deg", "continuous", -180.0, 180.0),
    FeatureSpec(8, "speed", "m/s", "continuous", 0.0, 70.0),
    FeatureSpec(9, "speed_var_5s", "(m/s)^2", "continuous", 0.0, 500.0),
    FeatureSpec(10, "heading_rate", "deg/s", "continuous", -360.0, 360.0),
    FeatureSpec(11, "lane_offset", "m", "continuous", -5.0, 5.0),
    FeatureSpec(12, "lateral_drift_5s", "m", "continuous", 0.0, 5.0),
    _flag(13, "gps_valid"),
)

VISION_NAMES: tuple[str, ...] = tuple(f.name for f in VISION_FEATURES)
MOTION_NAMES: tuple[str, ...] = tuple(f.name for f in MOTION_FEATURES)
DV = len(VISION_FEATURES)
DM = len(MOTION_FEATURES)

# Vision mask rule (contract section 4.1): face visible and confidence >= threshold.
FACE_CONF_MIN = 0.5
# Motion features that may be NaN while motion is available (contract section 4.2).
MOTION_NAN_ALLOWED: tuple[str, ...] = ("lane_offset", "lateral_drift_5s")

# --- Parquet columns (contract section 6.2) --------------------------------------------------
VISION_PREFIX = "v_"
MOTION_PREFIX = "m_"
VISION_COLUMNS: tuple[str, ...] = tuple(VISION_PREFIX + n for n in VISION_NAMES)
MOTION_COLUMNS: tuple[str, ...] = tuple(MOTION_PREFIX + n for n in MOTION_NAMES)
COL_T = "t"
COL_MASK_V = "mask_v"
COL_MASK_M = "mask_m"
COL_Y_CLS = "y_cls"
COL_CONTRACT = "contract_version"

PARQUET_COLUMNS: tuple[str, ...] = (
    COL_T,
    *VISION_COLUMNS,
    *MOTION_COLUMNS,
    COL_MASK_V,
    COL_MASK_M,
    COL_Y_CLS,
    COL_CONTRACT,
)
# Expected pandas dtypes as strings, so the validator can compare `str(df[c].dtype)`.
PARQUET_DTYPES: dict[str, str] = {
    COL_T: "float32",
    **{c: "float32" for c in VISION_COLUMNS},
    **{c: "float32" for c in MOTION_COLUMNS},
    COL_MASK_V: "bool",
    COL_MASK_M: "bool",
    COL_Y_CLS: "int8",
}

# --- Labels (contract section 5) -------------------------------------------------------------
UNLABELED = -1
CLASSES: tuple[str, ...] = ("normal", "aggressive", "drowsy", "distracted")
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
K_CLASSES = len(CLASSES)
VALID_Y_CLS: frozenset[int] = frozenset({UNLABELED, *range(K_CLASSES)})

EVENT_TYPES: tuple[str, ...] = (
    "harsh_accel",
    "harsh_brake",
    "sharp_turn",
    "lane_weave",
    "micro_sleep",
    "distraction_episode",
)

# --- Datasets, pairing, manifest (contract sections 3.4, 6.1, 6.3) ---------------------------
DATASETS: tuple[str, ...] = ("uah", "dmd", "own")
PAIRINGS: tuple[str, ...] = ("paired", "vision_only", "motion_only")
SPLITS: tuple[str, ...] = ("train", "val", "test")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "dataset",
    "driver_id",
    "pairing",
    "duration_s",
    "n_steps",
    "behavior_tag",
    "road_type",
    "vehicle_id",
    "split",
    "source_version",
    "contract_version",
)

DRIVER_ID_RE = re.compile(r"^d\d{2,}$")
TRIP_ID_RE = re.compile(r"^(uah|dmd|own)_d\d{2,}_\d{8}_\d{2}$")

# Required keys of <trip_id>_meta.json (contract section 6.4). `all_nan_columns` is how a
# trip declares feature columns that are legitimately all NaN (contract section 10).
META_REQUIRED_KEYS: tuple[str, ...] = (
    "trip_id",
    "contract_version",
    "sampling_rates",
    "extractor_versions",
    "notes",
)
META_ALL_NAN_KEY = "all_nan_columns"


def trip_files(trip_id: str) -> dict[str, str]:
    """File names for one trip inside data/processed/<dataset>/ (contract section 6.1)."""
    return {
        "parquet": f"{trip_id}.parquet",
        "events": f"{trip_id}_events.json",
        "meta": f"{trip_id}_meta.json",
    }


def driver_key(dataset: str, driver_id: str) -> str:
    """Globally unique driver key. `d03` may exist in several datasets (assumption A6)."""
    return f"{dataset}:{driver_id}"
