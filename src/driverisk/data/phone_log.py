"""Checks for a phone sensor log before it is used for own recordings (06 section 3, P0-D5).

The export format of the chosen logging app is not known yet (decision D6), so column names
are passed in or guessed from common spellings. Nothing here assumes one app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIME_CANDIDATES = ("seconds_elapsed", "time", "timestamp", "time (s)", "t", "epoch")
GAP_LIMIT_S = 0.5  # 06 section 9: IMU continuity, no gaps over 0.5 s
MIN_RATE_HZ = 10.0  # contract common clock is 10 Hz; a slower log cannot fill it
JITTER_WARN = 0.25  # std(dt) / median(dt)
TAP_PAIR_MAX_S = 1.5  # two firm taps within this interval form a sync event
TAP_MAD_FACTOR = 8.0  # a tap is a peak this many MADs above the median magnitude
TAP_REFRACTORY_S = 0.15  # one tap rings for a few samples; count it once


@dataclass
class LogReport:
    n_rows: int
    time_column: str
    time_unit: str
    duration_s: float
    median_rate_hz: float
    jitter: float
    max_gap_s: float
    n_gaps: int
    n_duplicate_times: int
    nan_fraction: float
    taps: list[float] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def guess_time_column(columns: list[str]) -> str:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in TIME_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    raise ValueError(f"no time column found among {columns}; pass --time-col")


def to_seconds(values: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert a time column to seconds from its first sample. Unit is guessed from magnitude.

    Magnitude of the first value: >1e17 ns epoch, >1e11 ms epoch, >1e9 s epoch; smaller values
    are taken as seconds since start (an app that logs `seconds_elapsed`).
    """
    first = float(np.nanmedian(values[: min(len(values), 5)]))
    if first > 1e17:
        scale, unit = 1e-9, "ns epoch"
    elif first > 1e11:
        scale, unit = 1e-3, "ms epoch"
    elif first > 1e9:
        scale, unit = 1.0, "s epoch"
    else:
        scale, unit = 1.0, "s since start"
    return (values - values[0]) * scale, unit


def detect_taps(t: np.ndarray, acc: np.ndarray) -> list[float]:
    """Times (s) of sync events: two firm taps within `TAP_PAIR_MAX_S` (06 section 5).

    Args:
        t: [N] seconds, increasing.
        acc: [N, 3] accelerometer (any unit); the tap shows as a spike in the magnitude.

    Returns the time of the first tap of each pair.
    """
    mag = np.linalg.norm(acc, axis=1)
    dev = np.abs(mag - np.median(mag))
    mad = np.median(dev) + 1e-9
    peaks: list[float] = []
    for i in np.flatnonzero(dev > TAP_MAD_FACTOR * mad):
        if not peaks or t[i] - peaks[-1] > TAP_REFRACTORY_S:
            peaks.append(float(t[i]))
    pairs, i = [], 0
    while i < len(peaks) - 1:
        if peaks[i + 1] - peaks[i] <= TAP_PAIR_MAX_S:
            pairs.append(peaks[i])
            i += 2
        else:
            i += 1
    return pairs


def analyze_log(
    df: pd.DataFrame, time_col: str | None = None, acc_cols: tuple[str, str, str] | None = None
) -> LogReport:
    """Timing quality of a sensor log; finds sync taps when accelerometer columns are given."""
    time_col = time_col or guess_time_column(list(df.columns))
    raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    problems: list[str] = []
    if np.isnan(raw).any():
        problems.append(f"{int(np.isnan(raw).sum())} unreadable timestamps")
        df, raw = df[~np.isnan(raw)], raw[~np.isnan(raw)]
    t, unit = to_seconds(raw)
    dt = np.diff(t)
    if (dt < 0).any():
        problems.append(f"timestamps go backwards at {int((dt < 0).sum())} samples")
    median_dt = float(np.median(dt[dt > 0])) if (dt > 0).any() else float("nan")
    rate = 1.0 / median_dt if median_dt > 0 else float("nan")
    jitter = float(np.std(dt) / median_dt) if median_dt > 0 else float("nan")
    gaps = dt[dt > GAP_LIMIT_S]
    n_dup = int((dt == 0).sum())
    if not rate >= MIN_RATE_HZ:
        problems.append(f"median rate {rate:.1f} Hz is below {MIN_RATE_HZ} Hz")
    if jitter > JITTER_WARN:
        problems.append(f"timestamp jitter {jitter:.2f} exceeds {JITTER_WARN}")
    if gaps.size:
        problems.append(f"{gaps.size} gaps longer than {GAP_LIMIT_S}s (max {gaps.max():.2f}s)")
    numeric = df.select_dtypes("number")
    nan_fraction = float(numeric.isna().to_numpy().mean()) if numeric.size else 0.0
    if nan_fraction > 0.01:
        problems.append(f"{nan_fraction:.1%} of numeric values are missing")

    taps: list[float] = []
    if acc_cols is not None:
        acc = df[list(acc_cols)].to_numpy(dtype=np.float64)
        taps = detect_taps(t, acc)
        if len(taps) < 1:
            problems.append("no tap pair found (06 section 5 needs two firm taps at the start)")
    return LogReport(
        n_rows=len(df), time_column=time_col, time_unit=unit,
        duration_s=float(t[-1] - t[0]) if len(t) else 0.0, median_rate_hz=rate, jitter=jitter,
        max_gap_s=float(gaps.max()) if gaps.size else float(dt.max()) if dt.size else 0.0,
        n_gaps=int(gaps.size), n_duplicate_times=n_dup, nan_fraction=nan_fraction,
        taps=taps, problems=problems,
    )  # fmt: skip
