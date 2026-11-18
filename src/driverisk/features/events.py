"""Event detector for motion events (P1-B3, docs/03_data_contract.md section 5.2).

Finds `harsh_accel`, `harsh_brake`, `sharp_turn` and `lane_weave` in a 10 Hz motion frame
(the output of `features.motion`, or any frame with the contract's `m_*` columns and `mask_m`).
`micro_sleep` and `distraction_episode` come from vision and annotation, not from here.

Thresholds are in `configs/events.yaml` and are proposals until agreed with the guide. Events are
returned in the `<trip_id>_events.json` format: `{onset_s, offset_s, type, source}`; `onset_s` is
the time of the first step over the threshold and `offset_s` the time of the last one.

A step counts only if `mask_m` is True and the value is finite; a gap ends the run it interrupts.
Speed gates use the held GPS speed: when a gate is above zero, an unknown speed fails it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driverisk.data import schema as S
from driverisk.utils.config import load_yaml
from driverisk.utils.paths import configs_dir

SOURCE = "own_detector"
Span = tuple[int, int]  # first and last step index, inclusive


def load_event_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_dir() / "events.yaml")


def _runs(flag: np.ndarray) -> list[Span]:
    """Maximal runs of True as (first, last) inclusive index pairs."""
    padded = np.r_[False, flag, False].astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _threshold_spans(
    flag: np.ndarray, t: np.ndarray, min_duration_s: float, merge_gap_s: float
) -> list[Span]:
    """Runs of `flag`, joined when closer than `merge_gap_s`, dropped when too short."""
    merged: list[list[int]] = []
    for first, last in _runs(flag):
        if merged and t[first] - t[merged[-1][1]] <= merge_gap_s + 1e-9:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else S.STEP_S
    return [
        (a, b) for a, b in merged if (b - a + 1) * dt >= min_duration_s - 1e-9
    ]  # whole steps: three 0.1 s steps are 0.3 s


def _event(t: np.ndarray, span: Span, kind: str) -> dict[str, Any]:
    return {
        "onset_s": float(t[span[0]]),
        "offset_s": float(t[span[1]]),
        "type": kind,
        "source": SOURCE,
    }


def _lane_weaves(
    t: np.ndarray, offset: np.ndarray, usable: np.ndarray, cfg: dict[str, Any]
) -> list[Span]:
    """Chains of >= `min_excursions` lane excursions alternating sides in quick succession."""
    enter, leave = cfg["enter_abs_offset_m"], cfg["exit_abs_offset_m"]
    excursions: list[tuple[int, int, int]] = []  # side, first step, last step
    side, first = 0, 0
    for i in range(len(t)):
        x = offset[i] if usable[i] else np.nan
        new_side = side
        if np.isnan(x):
            new_side = 0
        elif side == 0:
            new_side = int(np.sign(x)) if abs(x) >= enter else 0
        elif abs(x) < leave:
            new_side = 0
        elif abs(x) >= enter and np.sign(x) != side:
            new_side = int(np.sign(x))  # straight across to the other side
        if new_side != side:
            if side != 0:
                excursions.append((side, first, i - 1))
            side, first = new_side, i
    if side != 0:
        excursions.append((side, first, len(t) - 1))

    spans: list[Span] = []
    chain: list[tuple[int, int, int]] = []

    def close() -> None:
        if len(chain) >= cfg["min_excursions"]:
            spans.append((chain[0][1], chain[-1][2]))

    for exc in excursions:
        if chain and exc[0] != chain[-1][0] and t[exc[1]] - t[chain[-1][2]] <= cfg["max_gap_s"]:
            chain.append(exc)
        else:
            close()
            chain = [exc]
    close()
    return spans


def detect_events(
    frame: pd.DataFrame, cfg: dict[str, Any], *, source: str = SOURCE
) -> list[dict[str, Any]]:
    """Detect motion events in a 10 Hz motion frame; returns events sorted by onset."""
    p = S.MOTION_PREFIX
    t = frame[S.COL_T].to_numpy(np.float64)
    avail = frame[S.COL_MASK_M].to_numpy(bool)
    acc_long = frame[p + "acc_long"].to_numpy(np.float64)
    acc_lat = frame[p + "acc_lat"].to_numpy(np.float64)
    speed = frame[p + "speed"].to_numpy(np.float64)
    lane = frame[p + "lane_offset"].to_numpy(np.float64)

    def gate(block: dict[str, Any]) -> np.ndarray:
        floor = block["min_speed_mps"]
        if floor <= 0:
            return np.ones(len(t), dtype=bool)
        with np.errstate(invalid="ignore"):
            return speed >= floor  # NaN compares False: unknown speed fails the gate

    events: list[dict[str, Any]] = []

    def add(kind: str, flag: np.ndarray, block: dict[str, Any]) -> None:
        flag = flag & avail & gate(block)
        for span in _threshold_spans(flag, t, block["min_duration_s"], block["merge_gap_s"]):
            events.append(_event(t, span, kind))

    with np.errstate(invalid="ignore"):  # comparisons with NaN are False, which is what we want
        add("harsh_accel", acc_long >= cfg["harsh_accel"]["acc_long_min_mps2"], cfg["harsh_accel"])
        add("harsh_brake", acc_long <= cfg["harsh_brake"]["acc_long_max_mps2"], cfg["harsh_brake"])
        add("sharp_turn", np.abs(acc_lat) >= cfg["sharp_turn"]["abs_acc_lat_min_mps2"],
            cfg["sharp_turn"])  # fmt: skip
    weave = cfg["lane_weave"]
    usable = avail & gate(weave) & np.isfinite(lane)
    for span in _lane_weaves(t, lane, usable, weave):
        events.append(_event(t, span, "lane_weave"))

    for e in events:
        e["source"] = source
    return sorted(events, key=lambda e: (e["onset_s"], e["type"]))
