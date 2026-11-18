"""DMD annotation labels -> contract labels (P1-C1 label map, label part of P1-A3). No video.

DMD is NOT downloaded yet (human task H1). What this module assumes about the files is
UNVERIFIED and comes from DMD being documented as using OpenLABEL / VCD JSON:

- the document root is `{"openlabel": {...}}` (or `{"vcd": {...}}` in older VCD files);
- `root["actions"]` maps an id to `{"type": "<label name>", "frame_intervals":
  [{"frame_start": int, "frame_end": int}, ...]}`, with `frame_end` INCLUSIVE;
- `root["frame_intervals"]` gives the annotated frame range of the whole sequence;
- frame indices count frames of the annotated video at a constant `fps`.

Any of these can be wrong; `scripts/inspect_dmd.py` (after the download) must confirm them.
Nothing here guesses: unknown labels and malformed actions are reported, and `strict=True`
turns unknown labels into an error.

Label semantics (`configs/label_map_dmd.yaml`, a proposal):

- a label may be class evidence (`class`), may form events (`event`), both or neither;
- per frame, any non-normal class beats `normal`; two different non-normal classes give -1
  (conflict, counted in the report); no class evidence gives -1;
- event frames of one type (from all its labels) are merged into runs (gaps up to
  `merge_gap_s` are bridged) and kept if `min_s <= duration < max_s`; rejected runs are
  reported. A kept event may add class evidence for its frames (`event_classes`);
- per 10 Hz step, the class is the majority over the step's frames with the window rule of
  assumption A5 (`datasets.window_label`). Frames go to steps as in `features/vision.py`:
  frame time t -> step ceil(t / 0.1), so step k aggregates frames in (t_k - 0.1, t_k].
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from driverisk.data import schema as S
from driverisk.data.datasets import window_label
from driverisk.utils.config import load_yaml
from driverisk.utils.paths import configs_dir

log = logging.getLogger(__name__)

UNVERIFIED = "UNVERIFIED"
VERIFIED = "verified"
EVENT_SOURCE = "dmd_label"
_ROOT_KEYS = ("openlabel", "vcd")
_EPS = 1e-9


class UnknownLabelError(ValueError):
    """The annotation uses labels that the label map does not list."""


@dataclass(frozen=True)
class LabelRule:
    name: str
    cls: str | None
    event: str | None
    status: str
    note: str = ""


@dataclass(frozen=True)
class EventRule:
    min_s: float
    max_s: float | None  # exclusive upper bound; None = no upper bound
    merge_gap_s: float


@dataclass(frozen=True)
class DmdLabelMap:
    status: str
    labels: dict[str, LabelRule]
    events: dict[str, EventRule]
    event_classes: dict[str, str]


@dataclass(frozen=True)
class ActionInterval:
    label: str
    frame_start: int
    frame_end: int  # inclusive (UNVERIFIED for DMD)


@dataclass
class ParsedAnnotation:
    intervals: list[ActionInterval]
    n_frames_hint: int | None
    problems: list[str]


@dataclass
class DmdLabelReport:
    n_frames: int
    n_frames_source: str
    unknown_labels: dict[str, int] = field(default_factory=dict)  # label -> intervals
    ignored_labels: dict[str, int] = field(default_factory=dict)
    rejected_events: list[dict[str, Any]] = field(default_factory=list)
    conflict_frames: int = 0
    clipped_intervals: int = 0  # intervals cut by the video / step range
    problems: list[str] = field(default_factory=list)


@dataclass
class DmdLabels:
    y_cls: np.ndarray  # [n_steps] int8, -1 = unlabelled
    events: list[dict[str, Any]]  # contract section 5.2, source "dmd_label"
    n_steps: int
    report: DmdLabelReport


# --- label map --------------------------------------------------------------------------------


def label_map_from_dict(raw: dict[str, Any]) -> DmdLabelMap:
    """Validate a label map mapping (the YAML content). Raises ValueError on anything odd."""
    labels: dict[str, LabelRule] = {}
    for name, spec in (raw.get("labels") or {}).items():
        spec = spec or {}
        cls, event, status = spec.get("class"), spec.get("event"), spec.get("status")
        if cls is not None and cls not in S.CLASSES:
            raise ValueError(f"label {name!r}: class {cls!r} is not one of {S.CLASSES}")
        if event is not None and event not in S.EVENT_TYPES:
            raise ValueError(f"label {name!r}: event {event!r} is not one of {S.EVENT_TYPES}")
        if status not in (UNVERIFIED, VERIFIED):
            raise ValueError(f"label {name!r}: status must be {UNVERIFIED!r} or {VERIFIED!r}")
        labels[str(name)] = LabelRule(str(name), cls, event, status, str(spec.get("note", "")))

    events: dict[str, EventRule] = {}
    for etype, spec in (raw.get("events") or {}).items():
        if etype not in S.EVENT_TYPES:
            raise ValueError(f"event rule {etype!r} is not one of {S.EVENT_TYPES}")
        max_s = spec.get("max_s")
        events[etype] = EventRule(
            min_s=float(spec.get("min_s", 0.0)),
            max_s=None if max_s is None else float(max_s),
            merge_gap_s=float(spec.get("merge_gap_s", 0.0)),
        )
    needed = sorted({r.event for r in labels.values() if r.event} - set(events))
    if needed:
        raise ValueError(f"labels produce events {needed} but the map has no rule for them")

    event_classes: dict[str, str] = {}
    for etype, cls in (raw.get("event_classes") or {}).items():
        if etype not in S.EVENT_TYPES:
            raise ValueError(f"event_classes: {etype!r} is not one of {S.EVENT_TYPES}")
        if cls not in S.CLASSES:
            raise ValueError(f"event_classes: class {cls!r} is not one of {S.CLASSES}")
        event_classes[etype] = cls
    return DmdLabelMap(str(raw.get("status", "")), labels, events, event_classes)


def load_label_map(path: str | Path | None = None) -> DmdLabelMap:
    """Load `configs/label_map_dmd.yaml` (or `path`)."""
    return label_map_from_dict(load_yaml(path or configs_dir() / "label_map_dmd.yaml"))


# --- parsing ----------------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value) if float(value).is_integer() else None


def _max_frame_end(intervals: Any) -> int | None:
    ends = [_int_or_none(fi.get("frame_end")) for fi in intervals or [] if isinstance(fi, dict)]
    ends = [e for e in ends if e is not None]
    return max(ends) + 1 if ends else None


def parse_openlabel(doc: dict[str, Any], label_field: str = "type") -> ParsedAnnotation:
    """Action intervals of an OpenLABEL / VCD document (structure UNVERIFIED for DMD)."""
    root = next((doc[k] for k in _ROOT_KEYS if isinstance(doc, dict) and k in doc), None)
    if not isinstance(root, dict):
        raise ValueError(f"not an OpenLABEL/VCD document: no root key out of {_ROOT_KEYS}")
    problems: list[str] = []
    intervals: list[ActionInterval] = []
    actions = root.get("actions") or {}
    if not isinstance(actions, dict):
        raise ValueError("'actions' must be an object mapping ids to actions")
    for uid in sorted(actions, key=str):
        action = actions[uid] if isinstance(actions[uid], dict) else {}
        label = action.get(label_field)
        if not isinstance(label, str) or not label:
            problems.append(f"action {uid}: no {label_field!r}")
            continue
        fis = action.get("frame_intervals")
        if not isinstance(fis, list) or not fis:
            problems.append(f"action {uid} ({label}): no frame_intervals")
            continue
        for fi in fis:
            start = _int_or_none(fi.get("frame_start")) if isinstance(fi, dict) else None
            end = _int_or_none(fi.get("frame_end")) if isinstance(fi, dict) else None
            if start is None or end is None or start < 0 or end < start:
                problems.append(f"action {uid} ({label}): bad frame interval {fi!r}")
                continue
            intervals.append(ActionInterval(label, start, end))
    return ParsedAnnotation(intervals, _max_frame_end(root.get("frame_intervals")), problems)


def load_openlabel(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- labels on the 10 Hz grid -----------------------------------------------------------------


def _runs(mask: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    """(start, end inclusive) of True runs, bridging gaps of up to `max_gap_frames` frames."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap_frames + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def dmd_labels(
    annotation: dict[str, Any] | ParsedAnnotation,
    label_map: DmdLabelMap,
    fps: float,
    *,
    n_frames: int | None = None,
    n_steps: int | None = None,
    frame_shift: int = 0,
    strict: bool = False,
) -> DmdLabels:
    """Per-step `y_cls` and the event list of one DMD sequence.

    `fps`: frame rate of the annotated video. `n_frames`: frames in that video (else taken from
    the document, then from the last annotated frame; the report says which). `n_steps`: force
    the step count, for example to match the vision parquet; frames past it are ignored and
    counted. `frame_shift`: video frame = annotation frame + shift (stream sync, UNVERIFIED).
    """
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    parsed = annotation if isinstance(annotation, ParsedAnnotation) else parse_openlabel(annotation)
    if n_frames is not None:
        source = "argument"
    elif parsed.n_frames_hint is not None:
        n_frames, source = parsed.n_frames_hint, "document frame_intervals"
    else:
        n_frames = max((iv.frame_end + 1 for iv in parsed.intervals), default=0)
        source = "last annotated frame"
    if n_frames <= 0:
        raise ValueError("no frames: give n_frames or an annotation with frame intervals")

    frame_step = np.ceil(np.arange(n_frames) / fps / S.STEP_S - _EPS).astype(np.int64)
    forced = n_steps is not None
    if n_steps is None:
        n_steps = int(np.floor((n_frames - 1) / fps / S.STEP_S + _EPS)) + 1
    n_used = int(np.searchsorted(frame_step, n_steps))  # frames whose step is on the grid
    # "Clipped" = cut by the video range or by a forced step count; the last frames of a video
    # that fall past the final full step are dropped silently, exactly as in features/vision.py.
    clip_limit = n_used if forced else n_frames
    report = DmdLabelReport(n_frames=n_frames, n_frames_source=source, problems=parsed.problems)

    unknown: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    evidence = np.zeros((S.K_CLASSES, n_used), dtype=bool)
    event_frames = {e: np.zeros(n_used, dtype=bool) for e in label_map.events}
    for iv in parsed.intervals:
        rule = label_map.labels.get(iv.label)
        if rule is None:
            unknown[iv.label] += 1
            continue
        if rule.cls is None and rule.event is None:
            ignored[iv.label] += 1
            continue
        start, end = iv.frame_start + frame_shift, iv.frame_end + frame_shift
        if start < 0 or end > clip_limit - 1:
            report.clipped_intervals += 1
        lo, hi = max(start, 0), min(end, n_used - 1)
        if lo > hi:
            continue
        if rule.cls is not None:
            evidence[S.CLASS_TO_ID[rule.cls], lo : hi + 1] = True
        if rule.event is not None:
            event_frames[rule.event][lo : hi + 1] = True
    report.unknown_labels, report.ignored_labels = dict(unknown), dict(ignored)
    if unknown:
        msg = f"labels not in the DMD label map: {dict(sorted(unknown.items()))}"
        if strict:
            raise UnknownLabelError(msg)
        log.warning(msg)

    t_end = n_steps * S.STEP_S
    events: list[dict[str, Any]] = []
    for etype in sorted(event_frames):
        rule = label_map.events[etype]
        gap = int(np.floor(rule.merge_gap_s * fps + _EPS))
        for start, end in _runs(event_frames[etype], gap):
            onset = round(start / fps, 3)
            offset = round(min((end + 1) / fps, t_end), 3)
            duration = (end + 1 - start) / fps
            item = {"onset_s": onset, "offset_s": offset, "type": etype}
            if duration < rule.min_s - _EPS:
                report.rejected_events.append(
                    {**item, "reason": f"shorter than min_s {rule.min_s}"}
                )
            elif rule.max_s is not None and duration >= rule.max_s - _EPS:
                report.rejected_events.append({**item, "reason": f"not below max_s {rule.max_s}"})
            else:
                events.append({**item, "uncertain": False, "source": EVENT_SOURCE})
                cls = label_map.event_classes.get(etype)
                if cls is not None:
                    evidence[S.CLASS_TO_ID[cls], start : end + 1] = True
    events.sort(key=lambda e: (e["onset_s"], e["type"]))

    abnormal = evidence[1:]
    n_abnormal = abnormal.sum(axis=0)
    frame_cls = np.where(
        n_abnormal > 1,
        S.UNLABELED,
        np.where(n_abnormal == 1, 1 + abnormal.argmax(axis=0), np.where(evidence[0], 0, -1)),
    ).astype(np.int8)
    report.conflict_frames = int((n_abnormal > 1).sum())

    y_cls = np.full(n_steps, S.UNLABELED, dtype=np.int8)
    steps = frame_step[:n_used]
    bounds = np.searchsorted(steps, np.arange(n_steps + 1))
    for k in range(n_steps):
        if bounds[k + 1] > bounds[k]:
            y_cls[k] = window_label(frame_cls[bounds[k] : bounds[k + 1]])
    return DmdLabels(y_cls=y_cls, events=events, n_steps=n_steps, report=report)
