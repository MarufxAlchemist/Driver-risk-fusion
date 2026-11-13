"""Metrics library (docs/04 section 3). Expected numbers are worked out by hand in comments."""

from __future__ import annotations

import numpy as np
import pytest

from driverisk.evaluation import metrics as M

# --- classification -------------------------------------------------------------------------
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])
# class 0: TP1 FP1 FN1 -> F1 0.5
# class 1: TP2 FP1 FN0 -> P 2/3, R 1, F1 0.8
# class 2: TP1 FN1     -> P 1, R 0.5, F1 2/3


def test_confusion_matrix_rows_are_truth() -> None:
    cm = M.confusion_matrix(Y_TRUE, Y_PRED, classes=[0, 1, 2])
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_macro_f1_and_per_class() -> None:
    assert M.macro_f1(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    per = M.per_class_prf(Y_TRUE, Y_PRED)
    assert per[1]["precision"] == pytest.approx(2 / 3)
    assert per[1]["recall"] == pytest.approx(1.0)
    assert per[2]["f1"] == pytest.approx(2 / 3)
    assert M.balanced_accuracy(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 1.0 + 0.5) / 3)


def test_unlabelled_truth_is_ignored() -> None:
    y_true = np.concatenate([Y_TRUE, [-1, -1]])
    y_pred = np.concatenate([Y_PRED, [2, 0]])
    assert M.macro_f1(y_true, y_pred) == pytest.approx(M.macro_f1(Y_TRUE, Y_PRED))


def test_default_classes_are_those_present_in_truth() -> None:
    # Class 3 never occurs in the truth (like 'distracted' in UAH): it must not add a zero-F1
    # term to the macro average, but predicting it is still an error for the true class.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 3, 1, 1])
    assert M.macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1.0) / 2)  # class 0: P1 R.5
    assert M.macro_f1(y_true, y_pred, classes=[0, 1, 2, 3]) == pytest.approx((2 / 3 + 1.0) / 4)


def test_perfect_and_empty_cases() -> None:
    assert M.macro_f1(Y_TRUE, Y_TRUE) == 1.0
    assert np.isnan(M.macro_f1(np.array([-1, -1]), np.array([0, 1])))


# --- calibration ----------------------------------------------------------------------------
def test_ece_hand_computed() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y = np.array([0, 1, 1, 0])
    # confidences .9 .8 .7 .6, correct 1 0 1 1, one sample per bin: (.1 + .8 + .3 + .4) / 4
    assert M.expected_calibration_error(probs, y) == pytest.approx(0.4)


def test_ece_is_zero_for_perfectly_calibrated_bins() -> None:
    # 10 samples at confidence 0.7 with exactly 7 correct
    probs = np.tile([0.7, 0.3], (10, 1))
    y = np.array([0] * 7 + [1] * 3)
    assert M.expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_brier_ignores_unknown() -> None:
    assert M.brier_score(np.array([0.9, 0.2, 0.5]), np.array([1, 0, -1])) == pytest.approx(0.025)


def test_ece_binary_matches_manual() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    # bin (0,.1]: mean p .1, freq .5 -> .4 ; bin (.8,.9]: mean p .9, freq 1 -> .1 ; weights .5 each
    assert M.ece_binary(p, y) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)


# --- ranking / forecasting ------------------------------------------------------------------
def test_average_precision() -> None:
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert M.average_precision(y, s) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert M.average_precision(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(M.average_precision(np.array([0, 0]), np.array([0.1, 0.2])))
    assert M.average_precision(np.array([1, -1, 0]), np.array([0.9, 0.99, 0.1])) == 1.0


def test_pr_auc_per_horizon() -> None:
    y = np.array([[1, 0, -1], [0, 1, 0], [1, 1, 1]])
    s = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3], [0.7, 0.6, 0.9]])
    out = M.pr_auc_per_horizon(y, s, horizons=(2, 3, 5))
    assert set(out) == {2, 3, 5}
    assert out[2] == pytest.approx(1.0)  # positives scored .9 and .7 above the negative .2


# --- alerting -------------------------------------------------------------------------------
E1, E2 = M.EventSpan(10.0, 12.0), M.EventSpan(30.0, 31.0)


def _run(alerts, events=(E1, E2), duration=3600.0, **kw):
    return M.alerting_metrics(np.array(alerts, dtype=float), list(events), duration, **kw)


def test_fa_per_hour_recall_and_lead_time() -> None:
    # windows [onset - 5, offset + 2]: E1 [5, 14] (9 s), E2 [25, 33] (8 s) -> 3583 s normal
    r = _run([8.0, 11.5, 12.5, 20.0, 40.0], h_max_s=5.0, post_s=2.0)
    assert r.n_false_alarms == 2  # 20 and 40
    assert r.normal_hours == pytest.approx(3583.0 / 3600.0)
    assert r.fa_per_hour == pytest.approx(2 / (3583.0 / 3600.0))
    assert (r.n_events, r.n_detected) == (2, 1) and r.recall == pytest.approx(0.5)
    assert r.lead_times.tolist() == [2.0]  # earliest alert 8.0 for onset 10.0
    assert r.median_lead == 2.0 and r.early_warning_rate == pytest.approx(0.5)


def test_late_alert_has_negative_lead_and_no_early_credit() -> None:
    r = _run([11.0], events=[E1])
    assert r.lead_times.tolist() == [-1.0] and r.n_detected == 1 and r.early_warning_rate == 0.0


def test_alert_after_offset_does_not_detect_but_is_not_a_false_alarm_within_post() -> None:
    r = _run([13.0], events=[E1])  # offset 12, post 2 -> inside the FA exclusion window
    assert r.n_detected == 0 and r.n_false_alarms == 0


def test_repeated_alerts_for_one_event_are_not_false_alarms() -> None:
    r = _run([9.0, 9.5, 10.0, 10.5, 11.0], events=[E1])
    assert r.n_false_alarms == 0 and r.n_detected == 1 and r.lead_times.tolist() == [1.0]


def test_boundaries_are_inclusive() -> None:
    assert _run([5.0, 14.0], events=[E1]).n_false_alarms == 0  # edges of [5, 14]


def test_uncertain_events_are_neither_target_nor_false_alarm() -> None:
    unc = M.EventSpan(10.0, 12.0, uncertain=True)
    r = _run([8.0, 20.0], events=[unc])
    assert r.n_events == 0 and r.n_false_alarms == 1  # only the alert at 20
    assert r.normal_hours == pytest.approx((3600.0 - 9.0) / 3600.0)
    assert np.isnan(r.recall)


def test_excluded_and_include_only_ranges() -> None:
    # posed segment [100, 200]: alerts inside are dropped and the time leaves the denominator
    r = _run([150.0, 300.0], events=[], exclude=[(100.0, 200.0)])
    assert r.n_false_alarms == 1 and r.normal_hours == pytest.approx(3500.0 / 3600.0)
    # UAH rule: only count time/alerts where speed > 50 km/h
    r = _run([150.0, 300.0], events=[], include_only=[(250.0, 350.0)])
    assert r.n_false_alarms == 1 and r.normal_hours == pytest.approx(100.0 / 3600.0)


def test_no_normal_time_gives_nan_rate() -> None:
    r = _run([], events=[], duration=0.0)
    assert np.isnan(r.fa_per_hour)


def test_pooling_sums_counts_over_trips() -> None:
    a = _run([20.0], events=[E1])
    b = _run([8.0], events=[E1])
    pooled = M.pool_alerting([a, b])
    assert pooled.n_false_alarms == 1 and pooled.n_events == 2 and pooled.n_detected == 1
    assert pooled.normal_hours == pytest.approx(a.normal_hours + b.normal_hours)
    assert pooled.lead_times.tolist() == [2.0]


def test_alerts_from_scores_uses_rising_edge_and_refractory_period() -> None:
    t = np.arange(0, 10, 1.0)
    s = np.array([0.1, 0.9, 0.9, 0.2, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    assert M.alerts_from_scores(t, s, tau=0.5).tolist() == [1.0, 4.0, 7.0]
    assert M.alerts_from_scores(t, s, tau=0.5, refractory_s=4.0).tolist() == [1.0, 7.0]
    assert M.alerts_from_scores(t, np.full(10, 0.9), tau=0.5).tolist() == [0.0]  # starts high


def test_select_threshold_picks_lowest_tau_within_budget() -> None:
    taus = np.array([0.3, 0.5, 0.7, 0.9])
    fa = np.array([20.0, 8.0, 5.0, 1.0])
    assert M.select_threshold(taus, fa, budget=6.0) == 0.7
    assert np.isnan(M.select_threshold(taus, np.array([9.0, 9.0, 9.0, 9.0]), budget=6.0))
