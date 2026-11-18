"""Degradations for the robustness sweep (docs/04 section 3.4) and time-offset tests (F24).

All functions take a batch as produced by `WindowDataset` / `DataLoader` and return a **new**
batch; the input is never modified. A modality is "camera" (`x_v`, `nan_v`, `mask_v`) or
"motion" (`x_m`, `nan_m`, `mask_m`). A masked step is stored the way the dataset stores a
missing one: mask False, feature values 0, NaN flag True.

Not here yet: "phone moved" (rotating the raw IMU axes) must run on raw IMU before feature
extraction, so it belongs with the motion feature code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import torch

from driverisk.evaluation.metrics import macro_f1

Modality = Literal["camera", "motion"]
_KEYS: dict[str, tuple[str, str, str]] = {
    "camera": ("x_v", "nan_v", "mask_v"),
    "motion": ("x_m", "nan_m", "mask_m"),
}


def _keys(modality: str) -> tuple[str, str, str]:
    if modality not in _KEYS:
        raise ValueError(f"modality must be 'camera' or 'motion', got {modality!r}")
    return _KEYS[modality]


def _copy(batch: dict) -> dict:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _blank(out: dict, keys: tuple[str, str, str], steps: torch.Tensor) -> None:
    """Mark the `[B, T]` boolean `steps` as unavailable in `out` (in place)."""
    x, nan, mask = keys
    out[mask] = out[mask] & ~steps
    out[x] = out[x].masked_fill(steps.unsqueeze(-1), 0.0)
    out[nan] = out[nan] | steps.unsqueeze(-1)


def mask_modality(
    batch: dict,
    modality: Modality,
    fraction: float,
    generator: torch.Generator,
    *,
    contiguous: bool = False,
) -> dict:
    """Make `fraction` of each window's steps unavailable for one modality.

    Exactly `round(fraction * T)` steps per sample. By default the steps are random; with
    `contiguous=True` they form one run (a face lost for a few seconds), which is closer to
    real dropouts.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    keys = _keys(modality)
    out = _copy(batch)
    n_batch, n_steps = out[keys[2]].shape
    k = round(fraction * n_steps)
    if k == 0:
        return out
    steps = torch.zeros(n_batch, n_steps, dtype=torch.bool)
    for i in range(n_batch):
        if contiguous:
            start = int(torch.randint(0, n_steps - k + 1, (1,), generator=generator))
            steps[i, start : start + k] = True
        else:
            steps[i, torch.randperm(n_steps, generator=generator)[:k]] = True
    _blank(out, keys, steps)
    return out


def add_noise(batch: dict, modality: Modality, snr_db: float, generator: torch.Generator) -> dict:
    """Add Gaussian noise at `snr_db` to the available steps of one modality.

    Features are normalized to unit variance, so the noise standard deviation is
    `10 ** (-snr_db / 20)`. Masked steps stay exactly as they were.
    """
    x, _, mask = _keys(modality)
    out = _copy(batch)
    sigma = 10.0 ** (-snr_db / 20.0)
    noise = torch.randn(out[x].shape, generator=generator) * sigma
    out[x] = out[x] + noise * out[mask].unsqueeze(-1)
    return out


def shift_modality(batch: dict, modality: Modality, steps: int) -> dict:
    """Delay one stream by `steps` 0.1 s steps (negative = it arrives early).

    The values move, the other modality does not, and the steps that have nothing to show
    are marked unavailable. This is the time-offset perturbation (+-100, 250, 500 ms).
    """
    keys = _keys(modality)
    x, nan, mask = keys
    out = _copy(batch)
    n_steps = out[mask].shape[1]
    if steps == 0:
        return out
    if abs(steps) >= n_steps:
        _blank(out, keys, torch.ones_like(out[mask]))
        return out
    for key in (x, nan, mask):
        moved = torch.roll(out[key], shifts=steps, dims=1)
        out[key] = moved
    gap = torch.zeros_like(out[mask])
    if steps > 0:
        gap[:, :steps] = True
    else:
        gap[:, steps:] = True
    # `mask` was rolled, so the wrapped-around steps must be removed explicitly.
    _blank(out, keys, gap)
    return out


def robustness_curve(
    predict: Callable[[dict], np.ndarray],
    batches: Iterable[dict],
    modality: Modality,
    fractions: Sequence[float],
    seed: int = 0,
    *,
    contiguous: bool = False,
) -> dict[float, float]:
    """Macro-F1 of `predict(batch) -> class ids` as one modality is lost for `fraction` of steps.

    The same seed is used for every fraction, so the curve is reproducible. Batches are kept
    in memory once, then degraded per fraction.
    """
    batches = list(batches)
    curve: dict[float, float] = {}
    for r in fractions:
        gen = torch.Generator().manual_seed(seed)
        truth, pred = [], []
        for batch in batches:
            degraded = mask_modality(batch, modality, r, gen, contiguous=contiguous)
            pred.append(np.asarray(predict(degraded)))
            truth.append(batch["y_cls"].numpy())
        curve[float(r)] = macro_f1(np.concatenate(truth), np.concatenate(pred))
    return curve
