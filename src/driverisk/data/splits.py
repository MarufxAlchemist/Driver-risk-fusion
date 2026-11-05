"""Driver-disjoint split assignment and enforcement (ADR-0005)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from driverisk.data import schema as S


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
