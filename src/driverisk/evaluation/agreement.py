"""Inter-annotator agreement for event intervals (06 section 7).

Two annotators label the same trips as interval lists `(onset_s, offset_s, type)`. Two
measures are reported per event type:

* Cohen's kappa on 10 Hz step labels (event present / absent at each step);
* interval matching: greedy one-to-one matching at IoU >= `MATCH_IOU`, giving the share of
  intervals both annotators found (F1) and the mean IoU of matched pairs.

Intervals flagged `uncertain` are excluded from headline numbers and counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driverisk.data import schema as S

MATCH_IOU = 0.5  # an interval counts as found if the other annotator overlaps it this much
COLUMNS = ("trip_id", "type", "onset_s", "offset_s", "uncertain")


@dataclass
class TypeAgreement:
    event_type: str
    kappa: float  # nan when neither annotator marks the type (undefined)
    f1: float
    mean_iou: float  # nan when nothing matched
    n_a: int
    n_b: int
    n_matched: int


def load_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an annotation table and split off uncertain rows (returns certain rows)."""
    missing = [c for c in COLUMNS[:4] if c not in df.columns]
    if missing:
        raise ValueError(f"annotation file missing columns {missing}; expected {COLUMNS}")
    out = df.copy()
    out["uncertain"] = out["uncertain"].fillna(False).astype(bool) if "uncertain" in out else False
    bad = sorted(set(out["type"]) - set(S.EVENT_TYPES))
    if bad:
        raise ValueError(f"unknown event types {bad}; allowed {S.EVENT_TYPES}")
    if (out["offset_s"] < out["onset_s"]).any():
        raise ValueError("an interval ends before it starts")
    return out


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two boolean arrays; nan if chance agreement is 1 (no variation)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def _raster(rows: pd.DataFrame, n_steps: int) -> np.ndarray:
    x = np.zeros(n_steps, dtype=bool)
    for on, off in zip(rows["onset_s"], rows["offset_s"], strict=True):
        x[int(round(on * S.STEP_HZ)) : int(round(off * S.STEP_HZ))] = True
    return x


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_intervals(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[float]:
    """Greedy one-to-one matching, best IoU first; returns the IoU of each matched pair."""
    pairs = sorted(
        ((_iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True
    )
    used_a, used_b, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append(iou)
    return matched


def compare(
    ann_a: pd.DataFrame, ann_b: pd.DataFrame, durations: dict[str, float] | None = None
) -> tuple[list[TypeAgreement], dict[str, int]]:
    """Agreement per event type over the trips both annotators cover.

    `durations` maps trip_id to length in seconds; missing trips use the latest offset
    either annotator gave, plus 1 s. Returns (per-type results, counts of uncertain rows).
    """
    a_all, b_all = load_intervals(ann_a), load_intervals(ann_b)
    uncertain = {"a": int(a_all["uncertain"].sum()), "b": int(b_all["uncertain"].sum())}
    a, b = a_all[~a_all["uncertain"]], b_all[~b_all["uncertain"]]
    trips = sorted(set(a_all["trip_id"]) & set(b_all["trip_id"]))
    if not trips:
        raise ValueError("annotators share no trips; nothing to compare")
    durations = durations or {}
    results = []
    for etype in S.EVENT_TYPES:
        ka, kb, n_a, n_b, ious = [], [], 0, 0, []
        for tid in trips:
            ra, rb = (
                a[(a.trip_id == tid) & (a.type == etype)],
                b[(b.trip_id == tid) & (b.type == etype)],
            )
            last = max([0.0, *ra.offset_s, *rb.offset_s])
            n_steps = int(np.ceil(durations.get(tid, last + 1.0) * S.STEP_HZ))
            ka.append(_raster(ra, n_steps))
            kb.append(_raster(rb, n_steps))
            ia, ib = (
                list(zip(ra.onset_s, ra.offset_s, strict=True)),
                list(zip(rb.onset_s, rb.offset_s, strict=True)),
            )
            n_a, n_b = n_a + len(ia), n_b + len(ib)
            ious += match_intervals(ia, ib)
        n_m = len(ious)
        f1 = 2 * n_m / (n_a + n_b) if (n_a + n_b) else float("nan")
        results.append(
            TypeAgreement(
                etype, cohens_kappa(np.concatenate(ka), np.concatenate(kb)), f1,
                float(np.mean(ious)) if ious else float("nan"), n_a, n_b, n_m,
            )
        )  # fmt: skip
    return results, uncertain
