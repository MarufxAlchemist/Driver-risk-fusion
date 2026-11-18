"""Phone-log checker, on synthetic logs (no real app export exists yet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driverisk.data.phone_log import analyze_log, detect_taps, guess_time_column, to_seconds


def _log(rate: float = 50.0, seconds: float = 20.0, taps: tuple[float, ...] = (3.0, 3.4)):
    rng = np.random.default_rng(0)
    t = np.arange(0, seconds, 1 / rate)
    acc = rng.normal(0, 0.05, (len(t), 3)) + np.array([0.0, 0.0, 9.81])
    for tap in taps:
        acc[np.searchsorted(t, tap)] += [4.0, 3.0, 8.0]
    return t, acc


def _df(t: np.ndarray, acc: np.ndarray, **extra) -> pd.DataFrame:
    return pd.DataFrame(
        {"seconds_elapsed": t, "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2], **extra}
    )


def test_clean_log_with_taps_is_ok() -> None:
    r = analyze_log(_df(*_log()), acc_cols=("x", "y", "z"))
    assert r.ok, r.problems
    assert r.median_rate_hz == pytest.approx(50.0, rel=0.01)
    assert r.taps == [pytest.approx(3.0, abs=0.05)]


def test_single_tap_is_not_a_pair() -> None:
    t, acc = _log(taps=(3.0,))
    assert detect_taps(t, acc) == []


def test_taps_too_far_apart_are_not_a_pair() -> None:
    t, acc = _log(taps=(3.0, 6.0))
    assert detect_taps(t, acc) == []


def test_gap_is_reported() -> None:
    t, acc = _log()
    keep = (t < 8.0) | (t > 9.5)
    r = analyze_log(_df(t[keep], acc[keep]))
    assert r.n_gaps == 1 and any("gaps" in p for p in r.problems)


def test_slow_log_is_rejected() -> None:
    r = analyze_log(_df(*_log(rate=5.0)))
    assert any("below" in p for p in r.problems)


def test_backwards_time_is_reported() -> None:
    t, acc = _log()
    t[100:110] = t[100:110][::-1]
    assert any("backwards" in p for p in analyze_log(_df(t, acc)).problems)


def test_epoch_units_are_detected() -> None:
    base = np.array([0.0, 0.1, 0.2])
    assert to_seconds(base + 1.7e18)[1] == "ns epoch"
    assert to_seconds(base * 1e3 + 1.7e12)[1] == "ms epoch"
    assert to_seconds(base + 1.7e9)[1] == "s epoch"
    secs, unit = to_seconds(base * 1e9 + 1.7e18)
    assert unit == "ns epoch" and secs[-1] == pytest.approx(0.2, abs=1e-6)


def test_guess_time_column() -> None:
    assert guess_time_column(["Time", "x"]) == "Time"
    with pytest.raises(ValueError):
        guess_time_column(["a", "b"])
