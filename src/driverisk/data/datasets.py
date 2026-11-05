"""`WindowDataset`: windows cut on the fly from per-trip parquet (03 sections 1.4 and 7)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from driverisk.data import schema as S
from driverisk.data.normalize import NormStats, compute_norm_stats
from driverisk.data.splits import check_driver_disjoint

log = logging.getLogger(__name__)

# Assumption A5: a window gets a class label only if at least this share of its steps is labelled.
MIN_LABELLED_FRACTION = 0.5


@dataclass
class _Trip:
    trip_id: str
    driver_key: str
    t: np.ndarray  # [n] float32
    x_v: np.ndarray  # [n, 16] float32 (raw, may hold NaN)
    x_m: np.ndarray  # [n, 14] float32
    mask_v: np.ndarray  # [n] bool
    mask_m: np.ndarray  # [n] bool
    y_cls: np.ndarray  # [n] int8
    onsets: np.ndarray  # [E] event onset times (certain events)
    uncertain: list[tuple[float, float]]  # (onset, offset) of uncertain events
    events_annotated: bool
    duration_s: float


def load_manifests(dataset_dirs: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for d in dataset_dirs:
        m = pd.read_csv(Path(d) / "manifest.csv", dtype={"driver_id": str, "trip_id": str})
        m["_dir"] = str(d)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def _load_trip(row: pd.Series) -> _Trip:
    root, tid = Path(row["_dir"]), str(row["trip_id"])
    files = S.trip_files(tid)
    df = pd.read_parquet(root / files["parquet"])
    events = json.loads((root / files["events"]).read_text(encoding="utf-8"))
    meta_path = root / files["meta"]
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    certain = [e for e in events if not e.get("uncertain", False)]
    return _Trip(
        trip_id=tid,
        driver_key=S.driver_key(str(row["dataset"]), str(row["driver_id"])),
        t=df[S.COL_T].to_numpy(np.float32),
        x_v=df[list(S.VISION_COLUMNS)].to_numpy(np.float32),
        x_m=df[list(S.MOTION_COLUMNS)].to_numpy(np.float32),
        mask_v=df[S.COL_MASK_V].to_numpy(bool),
        mask_m=df[S.COL_MASK_M].to_numpy(bool),
        y_cls=df[S.COL_Y_CLS].to_numpy(np.int8),
        onsets=np.array([float(e["onset_s"]) for e in certain], dtype=np.float64),
        uncertain=[
            (float(e["onset_s"]), float(e["offset_s"])) for e in events if e.get("uncertain", False)
        ],
        events_annotated=bool(meta.get("events_annotated", True)),
        duration_s=float(row["duration_s"]),
    )


def window_label(y: np.ndarray) -> int:
    """Majority class over labelled steps; -1 if under-labelled or tied (assumption A5)."""
    labelled = y[y != S.UNLABELED]
    if labelled.size < MIN_LABELLED_FRACTION * y.size:
        return S.UNLABELED
    counts = np.bincount(labelled.astype(np.int64), minlength=S.K_CLASSES)
    top = np.flatnonzero(counts == counts.max())
    return int(top[0]) if top.size == 1 else S.UNLABELED


def risk_targets(trip: _Trip, t_end: float, horizons: Sequence[int]) -> np.ndarray:
    """y_risk[h] = 1 if a certain event starts in (t_end, t_end + H] (assumption A7).

    -1 when unknown: trip not annotated, horizon past trip end, or an uncertain event
    starts inside the horizon.
    """
    out = np.zeros(len(horizons), dtype=np.float32)
    for i, h in enumerate(horizons):
        end = t_end + h
        if not trip.events_annotated or end > trip.duration_s + S.T_STEP_TOLERANCE_S:
            out[i] = -1.0
        elif any(t_end < on <= end for on, _ in trip.uncertain):
            out[i] = -1.0
        else:
            out[i] = float(((trip.onsets > t_end) & (trip.onsets <= end)).any())
    return out


class WindowDataset(Dataset):
    """Windows of `WINDOW_STEPS` steps with stride `STRIDE_STEPS` over one split.

    Each item is a dict matching contract section 7 (without the batch dimension):
    `x_v [50, 16]`, `x_m [50, 14]`, `nan_v`, `nan_m`, `mask_v [50]`, `mask_m [50]`,
    `y_cls []` int64, `y_risk [3]` float32, `driver_id` str, `trip_id` str, `t_start` float.

    Constructing the dataset checks that no driver appears in more than one split of the
    combined manifests, so leaky splits fail loudly.
    """

    def __init__(
        self,
        dataset_dirs: Sequence[str | Path],
        split: str | None,
        norm_stats: NormStats | None = None,
        window_steps: int = S.WINDOW_STEPS,
        stride_steps: int = S.STRIDE_STEPS,
        horizons: Sequence[int] = S.HORIZONS_S,
    ) -> None:
        if split is not None and split not in S.SPLITS:
            raise ValueError(f"split must be one of {S.SPLITS} or None, got {split!r}")
        manifest = load_manifests(dataset_dirs)
        check_driver_disjoint(manifest)
        rows = manifest if split is None else manifest[manifest["split"] == split]
        self.split = split
        self.window_steps, self.stride_steps = window_steps, stride_steps
        self.horizons = tuple(horizons)
        self.trips = [_load_trip(r) for _, r in rows.iterrows()]
        self.norm = norm_stats or NormStats.identity()
        self.index = [
            (k, s)
            for k, trip in enumerate(self.trips)
            for s in range(0, len(trip.t) - window_steps + 1, stride_steps)
        ]
        log.info(
            "WindowDataset split=%s trips=%d windows=%d", split, len(self.trips), len(self.index)
        )

    def fit_norm_stats(self) -> NormStats:
        """Statistics of THIS dataset; call it only on the training split."""
        if self.split != "train":
            raise ValueError("normalization statistics must come from the training split only")
        return compute_norm_stats(
            [t.x_v for t in self.trips],
            [t.mask_v for t in self.trips],
            [t.x_m for t in self.trips],
            [t.mask_m for t in self.trips],
            source="train",
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        k, s = self.index[i]
        trip, e = self.trips[k], s + self.window_steps
        x_v, x_m = trip.x_v[s:e], trip.x_m[s:e]
        nan_v, nan_m = np.isnan(x_v), np.isnan(x_m)
        x_v = np.where(nan_v, 0.0, (x_v - self.norm.mean_v) / self.norm.std_v).astype(np.float32)
        x_m = np.where(nan_m, 0.0, (x_m - self.norm.mean_m) / self.norm.std_m).astype(np.float32)
        return {
            "x_v": torch.from_numpy(x_v),
            "x_m": torch.from_numpy(x_m),
            "nan_v": torch.from_numpy(nan_v),
            "nan_m": torch.from_numpy(nan_m),
            "mask_v": torch.from_numpy(trip.mask_v[s:e].copy()),
            "mask_m": torch.from_numpy(trip.mask_m[s:e].copy()),
            "y_cls": torch.tensor(window_label(trip.y_cls[s:e]), dtype=torch.int64),
            "y_risk": torch.from_numpy(risk_targets(trip, float(trip.t[e - 1]), self.horizons)),
            "driver_id": trip.driver_key,
            "trip_id": trip.trip_id,
            "t_start": float(trip.t[s]),
        }
