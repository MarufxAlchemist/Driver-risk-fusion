from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from driverisk.evaluation.agreement import cohens_kappa, compare, load_intervals, match_intervals


def _ann(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["trip_id", "type", "onset_s", "offset_s", "uncertain"])


def test_kappa_known_values() -> None:
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0], bool)
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert cohens_kappa(a, ~a) == pytest.approx(-1.0)
    b = np.array([1, 0, 0, 0, 1, 0, 1, 1], bool)  # po=0.75, pe=0.5 -> kappa=0.5
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_undefined_without_variation() -> None:
    assert math.isnan(cohens_kappa(np.zeros(10, bool), np.zeros(10, bool)))


def test_match_intervals_is_one_to_one_and_thresholded() -> None:
    a = [(0.0, 2.0), (10.0, 12.0)]
    b = [(0.1, 2.1), (11.9, 14.0), (30.0, 31.0)]
    ious = match_intervals(a, b)
    assert len(ious) == 1 and ious[0] == pytest.approx(1.9 / 2.1)
    # one annotator interval cannot match two of the other's
    assert len(match_intervals([(0.0, 4.0)], [(0.0, 2.2), (1.8, 4.0)])) == 1


def test_compare_perfect_and_disjoint() -> None:
    rows = [("t1", "micro_sleep", 5.0, 7.0, False), ("t1", "harsh_brake", 20.0, 21.0, False)]
    res, unc = compare(_ann(rows), _ann(rows), {"t1": 60.0})
    by = {r.event_type: r for r in res}
    assert by["micro_sleep"].kappa == pytest.approx(1.0) and by["micro_sleep"].f1 == 1.0
    assert unc == {"a": 0, "b": 0}
    moved = [("t1", "micro_sleep", 40.0, 42.0, False), ("t1", "harsh_brake", 20.0, 21.0, False)]
    res, _ = compare(_ann(rows), _ann(moved), {"t1": 60.0})
    by = {r.event_type: r for r in res}
    assert by["micro_sleep"].f1 == 0.0 and by["micro_sleep"].kappa < 0.1
    assert by["harsh_brake"].f1 == 1.0


def test_uncertain_rows_are_excluded_but_counted() -> None:
    a = _ann([("t1", "micro_sleep", 5.0, 7.0, False), ("t1", "micro_sleep", 30.0, 31.0, True)])
    b = _ann([("t1", "micro_sleep", 5.0, 7.0, False)])
    res, unc = compare(a, b, {"t1": 60.0})
    ms = next(r for r in res if r.event_type == "micro_sleep")
    assert (ms.n_a, ms.n_b, ms.f1) == (1, 1, 1.0)
    assert unc == {"a": 1, "b": 0}


def test_only_shared_trips_are_compared() -> None:
    a = _ann([("t1", "micro_sleep", 5.0, 7.0, False), ("t2", "micro_sleep", 5.0, 7.0, False)])
    b = _ann([("t1", "micro_sleep", 5.0, 7.0, False)])
    res, _ = compare(a, b, {"t1": 30.0})
    assert next(r for r in res if r.event_type == "micro_sleep").n_a == 1
    with pytest.raises(ValueError, match="no trips"):
        compare(a, _ann([("t9", "micro_sleep", 1.0, 2.0, False)]))


def test_bad_annotations_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown event types"):
        load_intervals(_ann([("t1", "nap", 1.0, 2.0, False)]))
    with pytest.raises(ValueError, match="ends before"):
        load_intervals(_ann([("t1", "micro_sleep", 3.0, 2.0, False)]))
    with pytest.raises(ValueError, match="missing columns"):
        load_intervals(pd.DataFrame({"trip_id": ["t1"]}))
