"""Training-time augmentation of one window (docs/04 section 6). Never used for validation or test.

Works on a sample dict as produced by `WindowDataset` (normalized features `x_v [T, 16]`,
`x_m [T, 14]`, NaN flags, masks). Rules shared by every operation:

* only available steps (`mask`) and non-NaN entries are changed;
* 0/1 flags, `face_conf` and the long-term PERCLOS context are never jittered;
* labels, ids and shapes are untouched, and the input dict is not modified;
* scaling and rotation act in physical units (they are undone and redone with the
  normalization statistics), because rotating a normalized pair would not be a rotation.

Not implemented: flipping clips (it would need the head-yaw label features flipped too).
Raw-IMU effects such as "phone moved" belong before feature extraction (features/motion.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from driverisk.data import schema as S
from driverisk.data.normalize import NormStats

_MI = {n: i for i, n in enumerate(S.MOTION_NAMES)}
_VI = {n: i for i, n in enumerate(S.VISION_NAMES)}

_NO_JITTER_VISION = {"face_conf", "perclos_30s", "perclos_60s"}
_MOTION_JITTER = [i for i, f in enumerate(S.MOTION_FEATURES) if f.kind == "continuous"]
_VISION_JITTER = [
    i
    for i, f in enumerate(S.VISION_FEATURES)
    if f.kind == "continuous" and f.name not in _NO_JITTER_VISION
]
_SCALED = [_MI[n] for n in ("acc_long", "acc_lat", "acc_vert", "jerk_long", "jerk_lat")]
_ROTATED = [(_MI["acc_long"], _MI["acc_lat"]), (_MI["jerk_long"], _MI["jerk_lat"])]


@dataclass(frozen=True)
class AugmentConfig:
    """Magnitudes are initial proposals; tune on validation only."""

    motion_jitter_sigma: float = 0.05  # in normalized units
    motion_scale_range: tuple[float, float] = (0.9, 1.1)  # one factor per window
    motion_rotation_deg: float = 5.0  # max axis misalignment of the (long, lat) plane
    vision_jitter_sigma: float = 0.02  # landmark jitter, normalized units
    face_drop_prob: float = 0.05  # share of steps turned into "face not found"

    def __post_init__(self) -> None:
        lo, hi = self.motion_scale_range
        if not 0 < lo <= hi:
            raise ValueError(f"motion_scale_range must satisfy 0 < lo <= hi, got {(lo, hi)}")
        if min(self.motion_jitter_sigma, self.vision_jitter_sigma, self.motion_rotation_deg) < 0:
            raise ValueError("jitter sigmas and the rotation angle must be non-negative")
        if not 0.0 <= self.face_drop_prob <= 1.0:
            raise ValueError(f"face_drop_prob must be in [0, 1], got {self.face_drop_prob}")


class Augmenter:
    """Callable `sample -> sample`; pass it as `WindowDataset(..., transform=...)`.

    Each DataLoader worker draws from its own stream (seed + worker id), so workers do not
    repeat each other's augmentations; runs stay reproducible for a fixed worker count.
    """

    def __init__(self, norm: NormStats, config: AugmentConfig | None = None, seed: int = 0) -> None:
        self.norm, self.cfg, self.seed = norm, config or AugmentConfig(), seed
        self._gen: torch.Generator | None = None
        self._mean_m = torch.from_numpy(np.asarray(norm.mean_m, dtype=np.float32))
        self._std_m = torch.from_numpy(np.asarray(norm.std_m, dtype=np.float32))

    def _generator(self) -> torch.Generator:
        if self._gen is None:
            info = torch.utils.data.get_worker_info()
            self._gen = torch.Generator().manual_seed(self.seed + (info.id if info else 0))
        return self._gen

    def _uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=self._generator()))

    def __call__(self, sample: dict) -> dict:
        out = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
        gen, cfg = self._generator(), self.cfg
        x_m, x_v = out["x_m"], out["x_v"]
        live_m = out["mask_m"].unsqueeze(-1) & ~out["nan_m"]
        live_v = out["mask_v"].unsqueeze(-1) & ~out["nan_v"]

        # physical-unit operations on the motion window
        lo, hi = self.cfg.motion_scale_range
        scale = self._uniform(lo, hi)
        angle = np.radians(self._uniform(-cfg.motion_rotation_deg, cfg.motion_rotation_deg))
        raw = x_m * self._std_m + self._mean_m
        scaled = raw.clone()
        for j in _SCALED:
            scaled[:, j] = torch.where(live_m[:, j], raw[:, j] * scale, raw[:, j])
        c, s = float(np.cos(angle)), float(np.sin(angle))
        for lon, lat in _ROTATED:
            both = live_m[:, lon] & live_m[:, lat]
            a, b = scaled[:, lon].clone(), scaled[:, lat].clone()  # clones: the writes below alias
            scaled[:, lon] = torch.where(both, c * a - s * b, a)
            scaled[:, lat] = torch.where(both, s * a + c * b, b)
        # Apply the change as a delta in normalized units: exactly zero where nothing changed,
        # so an all-off configuration returns the window bit for bit.
        x_m = x_m + torch.where(live_m, (scaled - raw) / self._std_m, torch.zeros_like(x_m))

        # jitter in normalized units
        if cfg.motion_jitter_sigma > 0:
            noise = torch.randn(x_m.shape, generator=gen) * cfg.motion_jitter_sigma
            keep = torch.zeros_like(live_m)
            keep[:, _MOTION_JITTER] = live_m[:, _MOTION_JITTER]
            x_m = x_m + noise * keep
        if cfg.vision_jitter_sigma > 0:
            noise = torch.randn(x_v.shape, generator=gen) * cfg.vision_jitter_sigma
            keep = torch.zeros_like(live_v)
            keep[:, _VISION_JITTER] = live_v[:, _VISION_JITTER]
            x_v = x_v + noise * keep
        out["x_m"], out["x_v"] = x_m.to(sample["x_m"].dtype), x_v.to(sample["x_v"].dtype)

        # random "face not found" steps
        if cfg.face_drop_prob > 0:
            drop = torch.rand(out["mask_v"].shape, generator=gen) < cfg.face_drop_prob
            out["mask_v"] = out["mask_v"] & ~drop
            out["x_v"] = out["x_v"].masked_fill(drop.unsqueeze(-1), 0.0)
            out["nan_v"] = out["nan_v"] | drop.unsqueeze(-1)
        return out
