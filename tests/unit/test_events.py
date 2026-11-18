"""Motion event detector (P1-B3). Frames are built directly at 10 Hz with the contract columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driverisk.data import schema as S
from driverisk.data.validate import validate_events
from driverisk.features import events as E

DT = 0.1


def _frame(n: int = 300, **cols) -> pd.DataFrame:
    """n steps; unspecified features are 0, speed defaults to 15 m/s, mask_m True."""
    df = pd.DataFrame({S.COL_T: (np.arange(n) * DT).astype(np.float32)})
    for name in S.MOTION_NAMES:
        df[S.MOTION_PREFIX + name] = np.float32(0.0)
    df[S.MOTION_PREFIX + "speed"] = np.float32(15.0)
    df[S.MOTION_PREFIX + "lane_offset"] = np.float32("nan")
    df[S.COL_MASK_M] = True
    for name, values in cols.items():
        df[S.MOTION_PREFIX + name] = np.asarray(values, dtype=np.float32)
    return df


def _pulse(n: int, start_s: float, dur_s: float, value: float, base: float = 0.0) -> np.ndarray:
    x = np.full(n, base)
    x[round(start_s / DT) : round((start_s + dur_s) / DT)] = value
    return x


def _detect(df: pd.DataFrame, **kw) -> list[dict]:
    return E.detect_events(df, E.load_event_config(), **kw)


def test_config_file_is_complete_and_sane() -> None:
    cfg = E.load_event_config()
    assert cfg["harsh_brake"]["acc_long_max_mps2"] < 0 < cfg["harsh_accel"]["acc_long_min_mps2"]
    assert cfg["sharp_turn"]["abs_acc_lat_min_mps2"] > 0
    assert cfg["lane_weave"]["exit_abs_offset_m"] < cfg["lane_weave"]["enter_abs_offset_m"]
    for block in ("harsh_accel", "harsh_brake", "sharp_turn"):
        assert cfg[block]["min_duration_s"] > 0 and cfg[block]["merge_gap_s"] >= 0


def test_harsh_brake_onset_and_offset_are_the_threshold_crossings() -> None:
    ev = _detect(_frame(acc_long=_pulse(300, 10.0, 1.0, -4.0)))
    assert ev == [
        {"onset_s": pytest.approx(10.0), "offset_s": pytest.approx(10.9), "type": "harsh_brake",
         "source": "own_detector"}
    ]  # fmt: skip


def test_harsh_accel_and_sharp_turn_either_direction() -> None:
    a = _detect(_frame(acc_long=_pulse(300, 5.0, 1.0, 3.0)))
    assert [e["type"] for e in a] == ["harsh_accel"]
    for sign in (+1, -1):
        t = _detect(_frame(acc_lat=_pulse(300, 5.0, 1.5, sign * 3.0)))
        assert [e["type"] for e in t] == ["sharp_turn"]


def test_below_threshold_or_too_short_is_ignored() -> None:
    assert _detect(_frame(acc_long=_pulse(300, 10.0, 2.0, -2.9))) == []  # not hard enough
    assert _detect(_frame(acc_long=_pulse(300, 10.0, 0.2, -6.0))) == []  # 2 steps < 0.3 s
    assert _detect(_frame(acc_long=_pulse(300, 10.0, 0.3, -6.0)))  # 3 steps = 0.3 s is enough


def test_close_bursts_merge_and_far_ones_do_not() -> None:
    near = _pulse(300, 10.0, 0.5, -4.0) + _pulse(300, 11.0, 0.5, -4.0)  # 0.5 s apart
    assert len(_detect(_frame(acc_long=near))) == 1
    far = _pulse(300, 10.0, 0.5, -4.0) + _pulse(300, 13.0, 0.5, -4.0)  # 2.5 s apart
    assert len(_detect(_frame(acc_long=far))) == 2


def test_speed_gate_applies_to_turns_only_as_configured() -> None:
    slow = _frame(acc_lat=_pulse(300, 5.0, 1.5, 3.0), speed=np.full(300, 1.5))
    assert _detect(slow) == []  # below the 3 m/s turn gate
    nan_speed = _frame(acc_lat=_pulse(300, 5.0, 1.5, 3.0), speed=np.full(300, np.nan))
    assert _detect(nan_speed) == []  # unknown speed cannot pass a speed gate
    brake_slow = _frame(acc_long=_pulse(300, 5.0, 1.0, -4.0), speed=np.full(300, 1.5))
    assert [e["type"] for e in _detect(brake_slow)] == ["harsh_brake"]  # no gate for braking


def test_masked_and_nan_steps_never_trigger_and_break_a_burst() -> None:
    df = _frame(acc_long=_pulse(300, 10.0, 2.0, -4.0))
    df.loc[df[S.COL_T] > 10.95, S.COL_MASK_M] = False  # the IMU drops out half way
    ev = _detect(df)
    assert len(ev) == 1 and ev[0]["offset_s"] < 11.0
    df = _frame(acc_long=np.full(300, np.nan))
    assert _detect(df) == []


def _swing(n: int, start_s: float, periods: float, amp: float, period_s: float = 4.0):
    x = np.full(n, np.nan)
    i0, i1 = round(start_s / DT), round((start_s + periods * period_s) / DT)
    t = np.arange(i1 - i0) * DT
    x[i0:i1] = amp * np.sin(2 * np.pi * t / period_s)
    x[:i0] = 0.0
    x[i1:] = 0.0
    return x


def test_lane_weave_needs_repeated_excursions() -> None:
    weave = _detect(_frame(lane_offset=_swing(300, 5.0, 2.0, 0.8)))  # +,-,+,- over 8 s
    assert [e["type"] for e in weave] == ["lane_weave"]
    assert 5.0 <= weave[0]["onset_s"] < 8.0 and weave[0]["offset_s"] > 10.0
    one = np.zeros(300)
    one[100:140] = 0.9  # a single lane departure is not a weave
    assert _detect(_frame(lane_offset=one)) == []
    assert _detect(_frame(lane_offset=_swing(300, 5.0, 2.0, 0.3))) == []  # amplitude too small


def test_lane_weave_is_off_when_there_is_no_lane_data() -> None:
    assert _detect(_frame()) == []  # lane_offset is NaN everywhere


def test_hysteresis_stops_a_noisy_edge_from_counting_as_many_excursions() -> None:
    x = np.zeros(300)
    x[100:200] = 0.52 + 0.05 * np.sin(np.arange(100) * 3.0)  # hovers around the entry level
    assert _detect(_frame(lane_offset=x)) == []


def test_output_is_sorted_valid_and_uses_contract_event_types() -> None:
    df = _frame(
        acc_long=_pulse(300, 20.0, 1.0, -4.0) + _pulse(300, 5.0, 1.0, 3.0),
        acc_lat=_pulse(300, 12.0, 1.0, -3.0),
    )
    ev = _detect(df)
    assert [e["onset_s"] for e in ev] == sorted(e["onset_s"] for e in ev)
    assert {e["type"] for e in ev} == {"harsh_accel", "sharp_turn", "harsh_brake"}
    assert {e["type"] for e in ev} <= set(S.EVENT_TYPES)
    assert validate_events(ev, duration_s=float(df[S.COL_T].iloc[-1])) == []


def test_events_that_touch_the_end_of_the_trip_are_closed_at_the_last_step() -> None:
    df = _frame(120, acc_long=_pulse(120, 11.0, 5.0, -4.0))  # still braking when data ends
    ev = _detect(df)
    assert len(ev) == 1 and ev[0]["offset_s"] == pytest.approx(float(df[S.COL_T].iloc[-1]))
