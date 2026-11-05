"""Per-feature normalization statistics, computed on the training split only (04 section 1.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driverisk.data import schema as S

_STD_FLOOR = 1e-6  # avoids division by zero for constant features (e.g. stub flags)


@dataclass(frozen=True)
class NormStats:
    mean_v: np.ndarray  # [16] float32
    std_v: np.ndarray  # [16] float32
    mean_m: np.ndarray  # [14] float32
    std_m: np.ndarray  # [14] float32
    source: str = ""  # which split/run produced the stats, for traceability

    @staticmethod
    def identity() -> NormStats:
        return NormStats(
            np.zeros(S.DV, np.float32),
            np.ones(S.DV, np.float32),
            np.zeros(S.DM, np.float32),
            np.ones(S.DM, np.float32),
            "identity",
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "contract_version": S.CONTRACT_VERSION,
            "source": self.source,
            "vision": dict(
                zip(
                    S.VISION_NAMES,
                    zip(self.mean_v.tolist(), self.std_v.tolist(), strict=True),
                    strict=True,
                )
            ),
            "motion": dict(
                zip(
                    S.MOTION_NAMES,
                    zip(self.mean_m.tolist(), self.std_m.tolist(), strict=True),
                    strict=True,
                )
            ),
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> NormStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        v = np.array([payload["vision"][n] for n in S.VISION_NAMES], np.float32)
        m = np.array([payload["motion"][n] for n in S.MOTION_NAMES], np.float32)
        return NormStats(v[:, 0], v[:, 1], m[:, 0], m[:, 1], payload.get("source", ""))


def _moments(
    rows: list[np.ndarray], masks: list[np.ndarray], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean/std over steps where the modality is available."""
    valid = [x[m] for x, m in zip(rows, masks, strict=True) if m.any()]
    if not valid:
        return np.zeros(dim, np.float32), np.ones(dim, np.float32)
    stacked = np.concatenate(valid, axis=0).astype(np.float64)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)  # all-NaN feature (e.g. lane_offset on own data)
    std = np.where(np.isnan(std) | (std < _STD_FLOOR), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_norm_stats(
    x_v: list[np.ndarray],
    mask_v: list[np.ndarray],
    x_m: list[np.ndarray],
    mask_m: list[np.ndarray],
    source: str,
) -> NormStats:
    """Inputs are per-trip arrays `[n, 16]`, `[n]`, `[n, 14]`, `[n]` from the TRAIN split only."""
    mean_v, std_v = _moments(x_v, mask_v, S.DV)
    mean_m, std_m = _moments(x_m, mask_m, S.DM)
    return NormStats(mean_v, std_v, mean_m, std_m, source)
