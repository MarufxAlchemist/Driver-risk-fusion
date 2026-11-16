"""Driver-disjoint split assignment and enforcement (ADR-0005).

Two layers:

- `assign_driver_splits` / `check_driver_disjoint`: the P0 helpers used by the stub
  generator and `WindowDataset`. Unchanged.
- The **split record** (P1-C2, `scripts/make_splits.py`): one frozen JSON file holding the
  per-driver holdout assignment, leave-one-driver-out folds with nested validation (fix F16)
  and the within-driver personalization split (`04` section 2.2). Built from manifests only,
  so it is reproducible: same manifests + same seed -> byte-identical file.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S

SPLIT_RECORD_FORMAT = 1
PERSONALIZATION_OK = "ok"
NOT_ENOUGH_DATA = "not enough data"  # 04 section 2.2: reported, never dropped silently


class SplitLeakageError(ValueError):
    """Raised when one driver appears in more than one split."""


def assign_driver_splits(
    manifest: pd.DataFrame, fractions: dict[str, float], seed: int
) -> pd.DataFrame:
    """Assign every trip the split of its driver. Drivers are shuffled per dataset.

    With at least as many drivers as splits, every split gets at least one driver.
    Returns a copy with the `split` column filled.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6 or set(fractions) - set(S.SPLITS):
        raise ValueError(f"fractions must cover {S.SPLITS} and sum to 1, got {fractions}")
    names = [s for s in S.SPLITS if fractions.get(s, 0) > 0]
    rng = np.random.default_rng(seed)
    out = manifest.copy()
    for dataset, rows in out.groupby("dataset"):
        drivers = sorted(rows["driver_id"].astype(str).unique())
        rng.shuffle(drivers)
        n = len(drivers)
        counts = [int(round(fractions[s] * n)) for s in names]
        if n >= len(names):
            counts = [max(1, c) for c in counts]
        while sum(counts) > n:  # trim the largest split first
            counts[int(np.argmax(counts))] -= 1
        counts[0] += n - sum(counts)
        bounds = np.cumsum([0, *counts])
        mapping = {
            d: names[i] for i in range(len(names)) for d in drivers[bounds[i] : bounds[i + 1]]
        }
        sel = out["dataset"] == dataset
        out.loc[sel, "split"] = out.loc[sel, "driver_id"].astype(str).map(mapping)
    return out


def check_driver_disjoint(manifest: pd.DataFrame) -> None:
    """Raise `SplitLeakageError` if a driver key appears in more than one split."""
    rows = manifest.dropna(subset=["split"])
    keys = [
        S.driver_key(d, i)
        for d, i in zip(rows["dataset"], rows["driver_id"].astype(str), strict=True)
    ]
    n_splits = pd.Series(rows["split"].to_numpy(), index=keys).groupby(level=0).nunique()
    leaking = sorted(n_splits[n_splits > 1].index)
    if leaking:
        raise SplitLeakageError(f"drivers in more than one split: {leaking}")


# --- Split record (P1-C2) ----------------------------------------------------------------------


class SplitInputError(ValueError):
    """The manifests cannot be split safely (duplicate trips, missing drivers, bad values)."""


_REQUIRED_FOR_SPLITS = ("trip_id", "dataset", "driver_id")
_LODO_RULE = (
    "drivers sorted by key; fold i tests driver i, validates on driver i+1 (wrapping), "
    "trains on the rest"
)
_PERSONALIZATION_RULE = (
    "per driver with at least min_trips trips, ordered by the date and number in trip_id: "
    "calibration = the first k_calibration trips whose behavior_tag is in normal_tags; "
    "evaluation = every trip after the last calibration trip; earlier non-calibration trips "
    "are listed under excluded"
)
_TRIP_ORDER_RE = re.compile(r"_(\d{8})_(\d{2})$")


def _as_text(manifest: pd.DataFrame) -> pd.DataFrame:
    """Every cell as a string, NaN/None as "" (callers may read with or without `dtype=str`)."""
    return manifest.astype(object).where(manifest.notna(), "").astype(str)


def check_split_inputs(manifest: pd.DataFrame) -> None:
    """Refuse manifests that cannot be split by driver without guessing.

    Raises `SplitInputError` for missing columns, rows without `driver_id`, duplicate
    `trip_id`, unknown `dataset` values, and trip ids that do not follow contract 6.1 or name
    another dataset/driver than their row.
    """
    missing = [c for c in _REQUIRED_FOR_SPLITS if c not in manifest.columns]
    if missing:
        raise SplitInputError(f"manifest missing column(s) {missing}")
    text = _as_text(manifest[list(_REQUIRED_FOR_SPLITS)])
    no_driver = text["driver_id"].str.strip() == ""
    if no_driver.any():
        raise SplitInputError(
            f"{int(no_driver.sum())} row(s) without driver_id: {sorted(text['trip_id'][no_driver])}"
        )
    dup = text["trip_id"][text["trip_id"].duplicated(keep=False)]
    if not dup.empty:
        raise SplitInputError(f"duplicate trip_id {sorted(set(dup))}")
    bad = sorted(set(text["dataset"]) - set(S.DATASETS))
    if bad:
        raise SplitInputError(f"unknown dataset value(s) {bad}, expected one of {S.DATASETS}")
    for tid, ds, did in zip(text["trip_id"], text["dataset"], text["driver_id"], strict=True):
        if not S.TRIP_ID_RE.match(tid) or not tid.startswith(f"{ds}_{did}_"):
            raise SplitInputError(
                f"trip_id {tid!r} does not follow <dataset>_<driverid>_<yyyymmdd>_<nn> "
                f"for dataset {ds!r}, driver_id {did!r}"
            )


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """SHA-256 of a manifest's content **without** the `split` column, rows sorted by trip_id.

    Leaving `split` out means writing the assignment back into the manifest does not change
    the hash, so a re-run on rewritten manifests reproduces the same split file.
    """
    text = _as_text(manifest.drop(columns=["split"], errors="ignore"))
    text = text.sort_values("trip_id", kind="mergesort").reset_index(drop=True)
    payload = text.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    ok = (
        isinstance(fractions, Mapping)
        and not set(fractions) - set(S.SPLITS)
        and all(float(v) >= 0 for v in fractions.values())
        and abs(sum(float(v) for v in fractions.values()) - 1.0) <= 1e-6
    )
    if not ok:
        raise ValueError(f"fractions must use {S.SPLITS}, be >= 0 and sum to 1, got {fractions}")
    return {s: float(fractions.get(s, 0.0)) for s in S.SPLITS}


def _split_counts(n: int, fractions: dict[str, float]) -> list[tuple[str, int]]:
    """Same rounding as `assign_driver_splits`: every active split gets a driver if n allows."""
    names = [s for s in S.SPLITS if fractions[s] > 0]
    counts = [int(round(fractions[s] * n)) for s in names]
    if n >= len(names):
        counts = [max(1, c) for c in counts]
    while sum(counts) > n:
        counts[int(np.argmax(counts))] -= 1
    counts[0] += n - sum(counts)
    return list(zip(names, counts, strict=True))


def assign_splits_by_driver(
    manifest: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    fractions_per_dataset: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Holdout split per dataset, by global driver key. Returns (key -> split, warnings).

    Each dataset gets its own random stream seeded by (seed, CRC32 of the dataset name), so
    adding or changing one dataset never reshuffles the others. Drivers are sorted before the
    shuffle, so row order and manifest order do not matter.
    """
    check_split_inputs(manifest)
    _check_fractions(fractions)
    per = dict(fractions_per_dataset or {})
    unknown = sorted(set(per) - set(S.DATASETS))
    if unknown:
        raise ValueError(f"fractions_per_dataset names unknown datasets {unknown}")
    text = _as_text(manifest)
    out: dict[str, str] = {}
    warnings: list[str] = []
    for dataset in sorted(set(text["dataset"])):
        fr = _check_fractions(per.get(dataset, fractions))
        drivers = sorted(set(text.loc[text["dataset"] == dataset, "driver_id"]))
        rng = np.random.default_rng([int(seed), zlib.crc32(dataset.encode("utf-8"))])
        order = [drivers[i] for i in rng.permutation(len(drivers))]
        pos = 0
        counts = _split_counts(len(drivers), fr)
        for name, count in counts:
            for driver in order[pos : pos + count]:
                out[S.driver_key(dataset, driver)] = name
            pos += count
        empty = [name for name, count in counts if count == 0]
        if empty:
            warnings.append(
                f"{dataset}: {len(drivers)} driver(s) for splits {[n for n, _ in counts]}; "
                f"empty: {empty}"
            )
    return out, warnings


def lodo_folds(driver_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Leave-one-driver-out folds with nested validation (fix F16).

    Fold i: test = driver i, validation = driver i+1 (wrapping), training = all others, with
    drivers in sorted order. Every driver is tested once and used for validation once.
    """
    keys = sorted(set(driver_keys))
    if len(keys) < 3:
        raise ValueError(
            f"leave-one-driver-out with nested validation needs at least 3 drivers, got {len(keys)}"
        )
    folds = []
    for i, test in enumerate(keys):
        val = keys[(i + 1) % len(keys)]
        train = [k for k in keys if k not in (test, val)]
        folds.append({"fold": i, "test": test, "val": val, "train": train})
    return folds


def _trip_order(trip_id: str) -> tuple[str, int]:
    match = _TRIP_ORDER_RE.search(trip_id)
    if match is None:  # check_split_inputs already refuses these
        raise SplitInputError(f"cannot order trip {trip_id!r} by date")
    return match.group(1), int(match.group(2))


def personalization_split(
    manifest: pd.DataFrame,
    k_calibration: int = 2,
    min_trips: int = 4,
    normal_tags: Sequence[str] = ("normal",),
) -> dict[str, dict[str, Any]]:
    """Within-driver calibration / evaluation split (`04` section 2.2, ADR-0005).

    Chronology comes from the `<yyyymmdd>_<nn>` part of the trip id (the contract has no trip
    start time). Every driver appears in the result; drivers that cannot be split get
    `status = "not enough data"` and a reason instead of being dropped.
    """
    check_split_inputs(manifest)
    if "behavior_tag" not in manifest.columns:
        raise SplitInputError("manifest missing column 'behavior_tag' (needed for normal trips)")
    tags = set(normal_tags)
    text = _as_text(manifest)
    out: dict[str, dict[str, Any]] = {}
    for (dataset, driver), rows in text.groupby(["dataset", "driver_id"], sort=True):
        trips = sorted(rows["trip_id"], key=_trip_order)
        tag = dict(zip(rows["trip_id"], rows["behavior_tag"], strict=True))
        entry: dict[str, Any] = {
            "n_trips": len(trips),
            "status": NOT_ENOUGH_DATA,
            "reason": "",
            "calibration": [],
            "evaluation": [],
            "excluded": {},
        }
        normal = [t for t in trips if tag[t] in tags]
        if len(trips) < min_trips:
            entry["reason"] = f"{len(trips)} trips, needs at least {min_trips}"
        elif len(normal) < k_calibration:
            entry["reason"] = (
                f"{len(normal)} normal trip(s) (behavior_tag in {sorted(tags)}), "
                f"needs {k_calibration} for calibration"
            )
        else:
            calibration = normal[:k_calibration]
            end = trips.index(calibration[-1])
            evaluation = trips[end + 1 :]
            if not evaluation:
                entry["reason"] = f"no evaluation trip after the {k_calibration} calibration trips"
            else:
                entry.update(
                    status=PERSONALIZATION_OK,
                    calibration=calibration,
                    evaluation=evaluation,
                    excluded={
                        t: "before end of calibration" for t in trips[:end] if t not in calibration
                    },
                )
        out[S.driver_key(str(dataset), str(driver))] = entry
    return out


def build_split_record(
    manifests: Mapping[str, pd.DataFrame],
    *,
    seed: int,
    fractions: Mapping[str, float],
    fractions_per_dataset: Mapping[str, Mapping[str, float]] | None = None,
    lodo_datasets: Sequence[str] = (),
    personalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything `splits.json` holds, from manifests keyed by a short, non-absolute name."""
    if not manifests:
        raise SplitInputError("no manifests given")
    inputs, frames = [], []
    for name, manifest in manifests.items():
        check_split_inputs(manifest)
        text = _as_text(manifest)
        inputs.append(
            {
                "name": str(name),
                "sha256": manifest_sha256(manifest),
                "n_trips": len(text),
                "datasets": sorted(set(text["dataset"])),
            }
        )
        frames.append(text)
    combined = pd.concat(frames, ignore_index=True)
    check_split_inputs(combined)  # duplicates across manifests
    per = dict(fractions_per_dataset or {})
    assignment, warnings = assign_splits_by_driver(combined, fractions, seed, per)

    drivers: dict[str, dict[str, Any]] = {}
    for (dataset, driver), rows in combined.groupby(["dataset", "driver_id"], sort=True):
        key = S.driver_key(str(dataset), str(driver))
        drivers[key] = {
            "dataset": str(dataset),
            "split": assignment[key],
            "n_trips": len(rows),
            "trips": sorted(rows["trip_id"]),
        }

    lodo: dict[str, Any] = {}
    for dataset in sorted(set(lodo_datasets)):
        keys = [k for k, v in drivers.items() if v["dataset"] == dataset]
        if not keys:
            raise ValueError(f"leave-one-driver-out requested for {dataset!r}, which has no trips")
        lodo[dataset] = {"rule": _LODO_RULE, "drivers": sorted(keys), "folds": lodo_folds(keys)}

    p = {"k_calibration": 2, "min_trips": 4, "normal_tags": ["normal"], **(personalization or {})}
    pers = personalization_split(
        combined, int(p["k_calibration"]), int(p["min_trips"]), tuple(p["normal_tags"])
    )
    return {
        "format": SPLIT_RECORD_FORMAT,
        "contract_version": S.CONTRACT_VERSION,
        "seed": int(seed),
        "fractions": {
            "default": _check_fractions(fractions),
            "per_dataset": {k: _check_fractions(v) for k, v in sorted(per.items())},
        },
        "inputs": sorted(inputs, key=lambda i: (i["name"], i["sha256"])),
        "drivers": drivers,
        "lodo": lodo,
        "personalization": {
            "rule": _PERSONALIZATION_RULE,
            "k_calibration": int(p["k_calibration"]),
            "min_trips": int(p["min_trips"]),
            "normal_tags": sorted(p["normal_tags"]),
            "drivers": pers,
        },
        "warnings": warnings,
    }


def dump_split_record(record: Mapping[str, Any]) -> str:
    """Canonical JSON text (sorted keys, LF, trailing newline): equal records -> equal bytes."""
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def apply_split_record(manifest: pd.DataFrame, record: Mapping[str, Any]) -> pd.DataFrame:
    """A copy of `manifest` with `split` filled from the record; other columns untouched."""
    out = manifest.copy()
    keys = [
        S.driver_key(str(d), str(i)) for d, i in zip(out["dataset"], out["driver_id"], strict=True)
    ]
    missing = sorted({k for k in keys if k not in record["drivers"]})
    if missing:
        raise SplitInputError(f"drivers not in the split record: {missing}")
    out["split"] = [record["drivers"][k]["split"] for k in keys]
    return out


def verify_split_record(
    record: Mapping[str, Any],
    manifests: Mapping[str, pd.DataFrame],
    require_all: bool = True,
) -> list[str]:
    """Errors if the manifests are not the ones the record was made from, or disagree with it.

    Manifests are matched to record inputs by content hash (`manifest_sha256`), not by name.
    """
    errors: list[str] = []
    known = {i["sha256"]: i["name"] for i in record["inputs"]}
    seen: set[str] = set()
    for name, manifest in manifests.items():
        digest = manifest_sha256(manifest)
        if digest in known:
            seen.add(digest)
        else:
            errors.append(f"{name}: sha256 {digest[:12]} matches no input of the split record")
        text = _as_text(manifest)
        split_col = text["split"] if "split" in text.columns else [""] * len(text)
        for tid, ds, did, split in zip(
            text["trip_id"], text["dataset"], text["driver_id"], split_col, strict=True
        ):
            entry = record["drivers"].get(S.driver_key(ds, did))
            if entry is None:
                errors.append(f"{name}: driver {S.driver_key(ds, did)} is not in the record")
            elif split != entry["split"]:
                errors.append(f"{name}: trip {tid} has split {split!r}, record: {entry['split']!r}")
    if require_all:
        errors.extend(
            f"record input {n} was not given" for h, n in sorted(known.items()) if h not in seen
        )
    return errors
