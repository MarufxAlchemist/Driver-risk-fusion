"""Contract validator (docs/03_data_contract.md section 10).

Every check returns human-readable error strings instead of raising, so one run
reports all problems at once. `validate_processed_dir` is what CI runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.data.splits import SplitLeakageError, check_driver_disjoint

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    n_trips: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, prefix: str, errors: list[str]) -> None:
        self.errors.extend(f"{prefix}: {e}" for e in errors)


def validate_trip_frame(df: pd.DataFrame, meta: dict | None = None) -> list[str]:
    """Check one trip's parquet table against the contract."""
    errors: list[str] = []
    missing = [c for c in S.PARQUET_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    extra = [c for c in df.columns if c not in S.PARQUET_COLUMNS]
    if extra:
        errors.append(f"unexpected columns {extra}")
    if len(df) == 0:
        return [*errors, "no rows"]

    for col, dtype in S.PARQUET_DTYPES.items():
        if str(df[col].dtype) != dtype:
            errors.append(f"column {col} has dtype {df[col].dtype}, expected {dtype}")

    versions = set(df[S.COL_CONTRACT].astype(str).unique())
    if versions != {S.CONTRACT_VERSION}:
        errors.append(f"contract_version {sorted(versions)} != {S.CONTRACT_VERSION}")

    t = df[S.COL_T].to_numpy(dtype=np.float64)
    if t[0] < 0:
        errors.append(f"t starts at {t[0]} < 0")
    if len(t) > 1:
        dt = np.diff(t)
        bad = np.abs(dt - S.STEP_S) > S.T_STEP_TOLERANCE_S
        if bad.any():
            first = int(np.argmax(bad)) + 1
            errors.append(f"t step != {S.STEP_S}s at {int(bad.sum())} rows (first row {first})")

    declared_nan = set((meta or {}).get(S.META_ALL_NAN_KEY, []))
    specs = [(S.VISION_PREFIX + f.name, f) for f in S.VISION_FEATURES]
    specs += [(S.MOTION_PREFIX + f.name, f) for f in S.MOTION_FEATURES]
    for col, spec in specs:
        values = df[col].to_numpy(dtype=np.float64)
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            if col not in declared_nan:
                errors.append(
                    f"column {col} is all NaN but not declared in meta {S.META_ALL_NAN_KEY}"
                )
            continue
        if np.isinf(finite).any():
            errors.append(f"column {col} contains inf")
            continue
        out = (finite < spec.lo) | (finite > spec.hi)
        if out.any():
            bounds = f"[{spec.lo}, {spec.hi}] {spec.unit}"
            errors.append(f"column {col} has {int(out.sum())} values outside {bounds}")
        if spec.kind == "flag" and not np.isin(finite, (0.0, 1.0)).all():
            errors.append(f"flag column {col} has values other than 0/1")

    y = df[S.COL_Y_CLS].to_numpy()
    bad_y = sorted(set(np.unique(y).tolist()) - S.VALID_Y_CLS)
    if bad_y:
        errors.append(f"y_cls has invalid values {bad_y}")

    errors.extend(_check_vision_mask(df))
    return errors


def _check_vision_mask(df: pd.DataFrame) -> list[str]:
    """mask_v[t] == (face_visible == 1 and face_conf >= 0.5) (contract section 4.1)."""
    vis = df[S.VISION_PREFIX + "face_visible"].to_numpy(dtype=np.float64)
    conf = df[S.VISION_PREFIX + "face_conf"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        expected = (vis == 1.0) & (conf >= S.FACE_CONF_MIN)
    actual = df[S.COL_MASK_V].to_numpy(dtype=bool)
    n_bad = int((expected != actual).sum())
    return [f"mask_v disagrees with face_visible/face_conf rule at {n_bad} rows"] if n_bad else []


def validate_meta(meta: dict, trip_id: str) -> list[str]:
    errors = [f"meta missing key {k}" for k in S.META_REQUIRED_KEYS if k not in meta]
    if meta.get("trip_id") not in (None, trip_id):
        errors.append(f"meta trip_id {meta.get('trip_id')} != {trip_id}")
    if meta.get("contract_version") not in (None, S.CONTRACT_VERSION):
        errors.append(
            f"meta contract_version {meta.get('contract_version')} != {S.CONTRACT_VERSION}"
        )
    unknown = set(meta.get(S.META_ALL_NAN_KEY, [])) - set(S.VISION_COLUMNS) - set(S.MOTION_COLUMNS)
    if unknown:
        errors.append(f"meta {S.META_ALL_NAN_KEY} names unknown columns {sorted(unknown)}")
    return errors


def validate_events(events: list, duration_s: float) -> list[str]:
    """Events are `{onset_s, offset_s, type[, uncertain]}` (contract section 5.2)."""
    errors: list[str] = []
    if not isinstance(events, list):
        return ["events file must contain a JSON list"]
    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or not {"onset_s", "offset_s", "type"} <= set(ev):
            errors.append(f"event {i} must have onset_s, offset_s, type")
            continue
        if ev["type"] not in S.EVENT_TYPES:
            errors.append(f"event {i} has unknown type {ev['type']!r}")
        on, off = float(ev["onset_s"]), float(ev["offset_s"])
        if not 0.0 <= on <= off <= duration_s + S.T_STEP_TOLERANCE_S:
            errors.append(f"event {i} interval [{on}, {off}] not inside trip [0, {duration_s}]")
    return errors


def validate_manifest(manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [c for c in S.MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        return [f"manifest missing columns {missing}"]
    if manifest["trip_id"].duplicated().any():
        errors.append("manifest has duplicate trip_id rows")
    if manifest["driver_id"].isna().any() or (manifest["driver_id"].astype(str) == "").any():
        errors.append("manifest has rows without driver_id")
    for col, allowed in (("dataset", S.DATASETS), ("pairing", S.PAIRINGS)):
        bad = sorted(set(manifest[col].dropna()) - set(allowed))
        if bad:
            errors.append(f"manifest {col} has invalid values {bad}")
    splits = manifest["split"].dropna()
    bad_split = sorted(set(splits) - set(S.SPLITS))
    if bad_split:
        errors.append(f"manifest split has invalid values {bad_split}")
    for tid in manifest["trip_id"]:
        if not S.TRIP_ID_RE.match(str(tid)):
            errors.append(f"trip_id {tid!r} does not match <dataset>_<driverid>_<yyyymmdd>_<nn>")
    for did in manifest["driver_id"].dropna():
        if not S.DRIVER_ID_RE.match(str(did)):
            errors.append(f"driver_id {did!r} is not an anonymous token like d03")
    errors.extend(driver_split_overlap(manifest))
    return errors


def driver_split_overlap(manifest: pd.DataFrame) -> list[str]:
    """No driver may appear in more than one split (ADR-0005)."""
    try:
        check_driver_disjoint(manifest)
    except SplitLeakageError as exc:
        return [str(exc)]
    return []


def validate_processed_dir(path: str | Path) -> ValidationReport:
    """Validate `data/processed/<dataset>/`: manifest plus every trip's files."""
    root = Path(path)
    report = ValidationReport()
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        report.errors.append(f"{manifest_path} not found")
        return report
    manifest = pd.read_csv(manifest_path, dtype={"driver_id": str, "trip_id": str})
    report.extend("manifest", validate_manifest(manifest))

    listed = set(manifest["trip_id"].astype(str))
    on_disk = {p.stem for p in root.glob("*.parquet")}
    for tid in sorted(on_disk - listed):
        report.errors.append(f"{tid}.parquet has no manifest row")
    for tid in sorted(listed - on_disk):
        report.errors.append(f"manifest row {tid} has no parquet file")

    for _, row in manifest.iterrows():
        tid = str(row["trip_id"])
        files = S.trip_files(tid)
        if tid not in on_disk:
            continue
        meta_path, events_path = root / files["meta"], root / files["events"]
        meta: dict = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            report.extend(tid, validate_meta(meta, tid))
        else:
            report.errors.append(f"{tid}: meta file missing")
        df = pd.read_parquet(root / files["parquet"])
        report.extend(tid, validate_trip_frame(df, meta))
        if int(row["n_steps"]) != len(df):
            report.errors.append(f"{tid}: manifest n_steps {row['n_steps']} != {len(df)} rows")
        if events_path.is_file():
            events = json.loads(events_path.read_text(encoding="utf-8"))
            report.extend(tid, validate_events(events, float(row["duration_s"])))
        else:
            report.errors.append(f"{tid}: events file missing")
        report.n_trips += 1
    log.info("validated %d trips in %s: %d errors", report.n_trips, root, len(report.errors))
    return report
