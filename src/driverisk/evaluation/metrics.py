"""Metrics library (docs/04_experiment_protocol.md section 3). NumPy only, no sklearn.

Conventions
* Labels equal to `schema.UNLABELED` (-1) in the truth are ignored everywhere.
* Macro-F1 averages over the classes that occur in the truth (assumption A16, fix F15): UAH has
  no `distracted` and DMD no `aggressive`, so a fixed K = 4 would add zero-F1 terms for
  classes that cannot occur. Pass `classes=` to override.
* Alerting follows fix F15 (assumption A17): one alert stream per trip, matched to events
  with explicit windows, pooled over trips by summing counts (never by averaging rates).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from driverisk.data import schema as S

# --- classification -------------------------------------------------------------------------


def _valid_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    keep = y_true != S.UNLABELED
    return y_true[keep].astype(np.int64), y_pred[keep].astype(np.int64)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None
) -> np.ndarray:
    """`cm[i, j]` = number of samples of true class `classes[i]` predicted as `classes[j]`.

    Predictions outside `classes` are dropped from the columns (they are still errors for the
    precision-recall functions below, which count them against the true class).
    """
    t, p = _valid_pairs(y_true, y_pred)
    labels = list(classes) if classes is not None else sorted(np.unique(t).tolist())
    index = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(t.tolist(), p.tolist(), strict=True):
        if a in index and b in index:
            cm[index[a], index[b]] += 1
    return cm


def per_class_prf(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None
) -> dict[int, dict[str, float]]:
    """Precision, recall, F1 and support per class. Undefined ratios are 0.0 (F1 convention)."""
    t, p = _valid_pairs(y_true, y_pred)
    labels = list(classes) if classes is not None else sorted(np.unique(t).tolist())
    out: dict[int, dict[str, float]] = {}
    for c in labels:
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[int(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(tp + fn),
        }
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None) -> float:
    per = per_class_prf(y_true, y_pred, classes)
    return float(np.mean([v["f1"] for v in per.values()])) if per else float("nan")


def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] | None = None
) -> float:
    per = per_class_prf(y_true, y_pred, classes)
    return float(np.mean([v["recall"] for v in per.values()])) if per else float("nan")


# --- calibration ----------------------------------------------------------------------------


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Right-closed equal-width bins (lo, hi]; the epsilon keeps 0.7 out of the 0.7-0.8 bin."""
    return np.clip(np.ceil(conf * n_bins - 1e-9).astype(np.int64) - 1, 0, n_bins - 1)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE with `n_bins` equal-width confidence bins (docs/04 section 3.1)."""
    probs, y_true = np.asarray(probs, dtype=np.float64), np.asarray(y_true)
    keep = y_true != S.UNLABELED
    probs, y_true = probs[keep], y_true[keep]
    if len(y_true) == 0:
        return float("nan")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    return _ece(conf, correct, n_bins)


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE of event probabilities: bin by `p`, compare mean `p` with the observed frequency."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y)
    keep = (y == 0) | (y == 1)
    p, y = p[keep], y[keep].astype(np.float64)
    if len(y) == 0:
        return float("nan")
    return _ece(p, y, n_bins)


def _ece(conf: np.ndarray, outcome: np.ndarray, n_bins: int) -> float:
    idx = _bin_index(conf, n_bins)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(outcome[sel].mean() - conf[sel].mean())
    return float(total)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of event probabilities; samples with y not in {0, 1} are skipped."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y)
    keep = (y == 0) | (y == 1)
    if not keep.any():
        return float("nan")
    return float(np.mean((p[keep] - y[keep]) ** 2))


# --- ranking / forecasting ------------------------------------------------------------------


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve as a step sum (the sklearn definition).

    Tied scores form one threshold. Samples with y not in {0, 1} are skipped. NaN when the
    truth has no positives.
    """
    y, score = np.asarray(y), np.asarray(score, dtype=np.float64)
    keep = (y == 0) | (y == 1)
    y, score = y[keep].astype(np.float64), score[keep]
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    last_of_tie = np.r_[score[1:] != score[:-1], True]
    tp, fp = tp[last_of_tie], fp[last_of_tie]
    precision, recall = tp / (tp + fp), tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def pr_auc_per_horizon(
    y_risk: np.ndarray, score: np.ndarray, horizons: Sequence[int] = S.HORIZONS_S
) -> dict[int, float]:
    """PR-AUC for each horizon column of `y_risk [N, len(horizons)]` (docs/04 section 3.3)."""
    y_risk, score = np.asarray(y_risk), np.asarray(score)
    return {int(h): average_precision(y_risk[:, i], score[:, i]) for i, h in enumerate(horizons)}


# --- alerting (docs/04 section 3.2, fix F15) -------------------------------------------------

Interval = tuple[float, float]


@dataclass(frozen=True)
class EventSpan:
    onset_s: float
    offset_s: float
    uncertain: bool = False


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for lo, hi in sorted((float(a), float(b)) for a, b in intervals if b >= a):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def _length(intervals: Iterable[Interval]) -> float:
    return float(sum(b - a for a, b in intervals))


def _inside(times: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for lo, hi in intervals:
        mask |= (times >= lo) & (times <= hi)
    return mask


def _subtract(base: list[Interval], holes: list[Interval]) -> list[Interval]:
    """`base` minus `holes`; both must already be merged (sorted, disjoint)."""
    out: list[Interval] = []
    for lo, hi in base:
        cur = lo
        for h_lo, h_hi in holes:
            if h_hi <= cur or h_lo >= hi:
                continue
            if h_lo > cur:
                out.append((cur, h_lo))
            cur = max(cur, h_hi)
        if cur < hi:
            out.append((cur, hi))
    return out


@dataclass(frozen=True)
class AlertingResult:
    """Counts first, rates derived, so results from several trips pool by addition."""

    n_alerts: int  # alerts that were counted (inside the evaluated range, not near uncertain)
    n_false_alarms: int
    normal_hours: float
    n_events: int  # certain events only
    n_detected: int
    lead_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    early_s: float = 1.0

    @property
    def fa_per_hour(self) -> float:
        return self.n_false_alarms / self.normal_hours if self.normal_hours > 0 else float("nan")

    @property
    def recall(self) -> float:
        return self.n_detected / self.n_events if self.n_events else float("nan")

    @property
    def median_lead(self) -> float:
        return float(np.median(self.lead_times)) if len(self.lead_times) else float("nan")

    @property
    def lead_iqr(self) -> tuple[float, float]:
        if not len(self.lead_times):
            return (float("nan"), float("nan"))
        q1, q3 = np.percentile(self.lead_times, [25, 75])
        return (float(q1), float(q3))

    @property
    def early_warning_rate(self) -> float:
        """Share of all certain events warned at least `early_s` before onset (missed count 0)."""
        if not self.n_events:
            return float("nan")
        return float((self.lead_times >= self.early_s).sum() / self.n_events)


def alerting_metrics(
    alert_times: np.ndarray,
    events: Sequence[EventSpan],
    duration_s: float,
    *,
    h_max_s: float = float(max(S.HORIZONS_S)),
    post_s: float = 2.0,
    exclude: Sequence[Interval] = (),
    include_only: Sequence[Interval] | None = None,
    early_s: float = 1.0,
) -> AlertingResult:
    """Alerting metrics for one trip.

    * An event's **evaluation window** is `[onset - h_max_s, offset + post_s]`. An alert inside
      the window of any event, certain or uncertain, is never a false alarm.
    * **False alarm:** a counted alert outside every event window.
    * **Detected:** a certain event with at least one alert in `[onset - h_max_s, offset]`;
      its **lead time** is `onset - earliest such alert` (negative = late). One-to-one, earliest
      alert wins; later alerts for the same event are repeats, not false alarms.
    * **Normal time** (FA/h denominator): the evaluated range minus all event windows. Time in
      `exclude` (posed or parked segments) is removed, and alerts there are dropped. With
      `include_only` (for example UAH speed > 50 km/h) only time and alerts inside it count.
    * Uncertain events are not targets (not in recall or lead time) and cover their window.
    """
    alerts = np.sort(np.asarray(alert_times, dtype=np.float64))
    exclude_m = _merge(exclude)
    evaluated = _merge(include_only) if include_only is not None else ([(0.0, float(duration_s))])
    evaluated = _subtract(evaluated, exclude_m)

    counted = alerts[_inside(alerts, evaluated)] if evaluated else alerts[:0]
    windows = _merge((e.onset_s - h_max_s, e.offset_s + post_s) for e in events)
    near_event = _inside(counted, windows)
    false_alarms = int((~near_event).sum())
    normal_s = _length(_subtract(evaluated, windows))

    leads: list[float] = []
    targets = [e for e in events if not e.uncertain]
    for e in targets:
        hits = counted[(counted >= e.onset_s - h_max_s) & (counted <= e.offset_s)]
        if len(hits):
            leads.append(e.onset_s - float(hits[0]))
    return AlertingResult(
        n_alerts=len(counted),
        n_false_alarms=false_alarms,
        normal_hours=normal_s / 3600.0,
        n_events=len(targets),
        n_detected=len(leads),
        lead_times=np.array(leads, dtype=np.float64),
        early_s=early_s,
    )


def pool_alerting(results: Sequence[AlertingResult]) -> AlertingResult:
    """Combine trips: FA/h = total false alarms / total normal hours, and so on."""
    if not results:
        raise ValueError("nothing to pool")
    return AlertingResult(
        n_alerts=sum(r.n_alerts for r in results),
        n_false_alarms=sum(r.n_false_alarms for r in results),
        normal_hours=sum(r.normal_hours for r in results),
        n_events=sum(r.n_events for r in results),
        n_detected=sum(r.n_detected for r in results),
        lead_times=np.concatenate([r.lead_times for r in results]),
        early_s=results[0].early_s,
    )


def alerts_from_scores(
    t: np.ndarray, score: np.ndarray, tau: float, refractory_s: float = 0.0
) -> np.ndarray:
    """Alert times: moments where `score` rises to >= `tau` (the first sample counts if high).

    After an alert, further rising edges within `refractory_s` are suppressed.
    """
    t, score = np.asarray(t, dtype=np.float64), np.asarray(score, dtype=np.float64)
    high = score >= tau
    rising = high & ~np.r_[False, high[:-1]]
    out: list[float] = []
    for ti in t[rising]:
        if not out or ti - out[-1] >= refractory_s:
            out.append(float(ti))
    return np.array(out, dtype=np.float64)


def select_threshold(taus: np.ndarray, fa_per_hour: np.ndarray, budget: float) -> float:
    """Lowest threshold (most sensitive) whose validation FA/h is within `budget`; NaN if none."""
    taus, fa = np.asarray(taus, dtype=np.float64), np.asarray(fa_per_hour, dtype=np.float64)
    ok = np.flatnonzero(fa <= budget)
    return float(taus[ok].min()) if len(ok) else float("nan")
