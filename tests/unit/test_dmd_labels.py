"""DMD labels (P1-C1 label map, label part of P1-A3) on small synthetic OpenLABEL-like JSON.

DMD is not downloaded. The synthetic documents follow the structure DMD is documented to use
(OpenLABEL / VCD: `actions` with `type` and `frame_intervals`), which itself is unverified.
The behaviour tests use their own small label map, so they do not depend on the proposal in
`configs/label_map_dmd.yaml`; that file only gets structural checks here.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from driverisk.data import dmd as D
from driverisk.data import schema as S
from driverisk.data.validate import validate_events
from driverisk.utils.paths import configs_dir

FPS = 30.0
N_FRAMES = 300  # 10 s -> 100 steps (same step rule as features/vision.py)

MAP = {
    "status": "test map",
    "labels": {
        "driver_actions/safe_drive": {"class": "normal", "event": None, "status": "UNVERIFIED"},
        "driver_actions/texting_right": {
            "class": "distracted",
            "event": "distraction_episode",
            "status": "UNVERIFIED",
        },
        "driver_actions/phonecall_right": {
            "class": "distracted",
            "event": "distraction_episode",
            "status": "UNVERIFIED",
        },
        "eyes_state/close": {"class": None, "event": "micro_sleep", "status": "UNVERIFIED"},
        "eyes_state/open": {"class": None, "event": None, "status": "UNVERIFIED"},
        "yawning/yawning": {"class": "drowsy", "event": None, "status": "UNVERIFIED"},
    },
    "events": {
        "micro_sleep": {"min_s": 1.0, "max_s": 3.0, "merge_gap_s": 0.0},
        "distraction_episode": {"min_s": 2.0, "max_s": None, "merge_gap_s": 0.0},
    },
    "event_classes": {"micro_sleep": "drowsy", "distraction_episode": "distracted"},
}


def _map(**changes) -> D.DmdLabelMap:
    raw = copy.deepcopy(MAP)
    raw.update(changes)
    return D.label_map_from_dict(raw)


def _doc(actions: list[tuple[str, list[tuple[int, int]]]], root: str = "openlabel") -> dict:
    return {
        root: {
            "metadata": {"schema_version": "synthetic"},
            "frame_intervals": [{"frame_start": 0, "frame_end": N_FRAMES - 1}],
            "actions": {
                str(i): {
                    "name": label,
                    "type": label,
                    "frame_intervals": [{"frame_start": s, "frame_end": e} for s, e in ivs],
                }
                for i, (label, ivs) in enumerate(actions)
            },
        }
    }


SCENE = [
    ("driver_actions/safe_drive", [(0, 299)]),
    ("driver_actions/texting_right", [(60, 149)]),  # 2.0-5.0 s
    ("eyes_state/close", [(180, 218), (240, 245)]),  # 1.3 s closure, then a 0.2 s blink
    ("eyes_state/open", [(0, 179)]),
    ("driver_actions/something_new", [(0, 10)]),  # not in the map
]


# --- the shipped proposal ---------------------------------------------------------------------


# Names seen in three real DMD drowsiness files on 2026-11-18 (gA/1, gA/5, gB/6, session s5).
CHECKED_ON_REAL_FILES = {
    "eyes_state/open", "eyes_state/close", "eyes_state/opening", "eyes_state/closing",
    "eyes_state/undefined", "blinks/blinking", "yawning/Yawning with hand",
    "yawning/Yawning without hand",
}  # fmt: skip


def test_shipped_map_is_a_proposal_and_only_the_checked_labels_are_verified() -> None:
    path = configs_dir() / "label_map_dmd.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "proposal" in raw["status"].lower() and "guide" in raw["status"].lower()
    lm = D.load_label_map(path)
    assert lm.labels, "the proposal lists labels"
    verified = {n for n, r in lm.labels.items() if r.status == D.VERIFIED}
    assert verified == CHECKED_ON_REAL_FILES  # the distraction/gaze names are still from memory
    assert all(r.cls in (None, *S.CLASSES) for r in lm.labels.values())
    assert all(r.cls != "aggressive" for r in lm.labels.values())  # DMD has no vehicle motion
    assert all(r.event in (None, *S.EVENT_TYPES) for r in lm.labels.values())
    assert lm.events["micro_sleep"].min_s == 1.0 and lm.events["micro_sleep"].max_s == 3.0


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("labels", "eyes_state/open", "class"), "sleepy", "class"),
        (("labels", "eyes_state/open", "event"), "nap", "event"),
        (("labels", "eyes_state/open", "status"), "checked", "status"),
        (("event_classes", "micro_sleep"), "tired", "class"),
    ],
)
def test_label_map_refuses_values_outside_the_contract(path, value, match) -> None:
    raw = copy.deepcopy(MAP)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        D.label_map_from_dict(raw)


def test_label_map_refuses_missing_status_and_unknown_event_rules() -> None:
    raw = copy.deepcopy(MAP)
    del raw["labels"]["eyes_state/open"]["status"]
    with pytest.raises(ValueError, match="status"):
        D.label_map_from_dict(raw)
    raw = copy.deepcopy(MAP)
    raw["events"]["nap"] = {"min_s": 1.0}
    with pytest.raises(ValueError, match="nap"):
        D.label_map_from_dict(raw)
    raw = copy.deepcopy(MAP)
    del raw["events"]["micro_sleep"]  # a label produces micro_sleep, so it needs a rule
    with pytest.raises(ValueError, match="micro_sleep"):
        D.label_map_from_dict(raw)


# --- parsing ----------------------------------------------------------------------------------


@pytest.mark.parametrize("root", ["openlabel", "vcd"])
def test_parse_reads_actions_and_frame_intervals(root: str) -> None:
    parsed = D.parse_openlabel(_doc(SCENE, root=root))
    got = sorted((i.label, i.frame_start, i.frame_end) for i in parsed.intervals)
    assert ("eyes_state/close", 180, 218) in got and ("eyes_state/close", 240, 245) in got
    assert len(got) == 6
    assert parsed.n_frames_hint == N_FRAMES
    assert parsed.problems == []


def test_parse_reports_malformed_actions_instead_of_guessing() -> None:
    doc = _doc([("driver_actions/safe_drive", [(10, 5)])])
    actions = doc["openlabel"]["actions"]
    actions["7"] = {"name": "x", "frame_intervals": [{"frame_start": 0, "frame_end": 3}]}  # no type
    actions["8"] = {"type": "driver_actions/radio"}  # no intervals
    actions["9"] = {"type": "driver_actions/radio", "frame_intervals": [{"frame_start": "a"}]}
    parsed = D.parse_openlabel(doc)
    assert parsed.intervals == []
    assert len(parsed.problems) == 4
    with pytest.raises(ValueError, match="openlabel"):
        D.parse_openlabel({"something": {}})


# --- labels on the 10 Hz grid -----------------------------------------------------------------


def test_step_classes_and_events_on_the_10hz_grid() -> None:
    out = D.dmd_labels(_doc(SCENE), _map(), fps=FPS, n_frames=N_FRAMES)
    y = out.y_cls
    assert y.dtype == np.int8 and len(y) == 100 and out.n_steps == 100
    assert (y[1:19] == S.CLASS_TO_ID["normal"]).all()
    assert (y[21:50] == S.CLASS_TO_ID["distracted"]).all()  # texting 2.0-5.0 s
    assert (y[61:73] == S.CLASS_TO_ID["drowsy"]).all()  # closure 6.0-7.3 s via micro_sleep
    assert (y[76:] == S.CLASS_TO_ID["normal"]).all()  # 0.2 s blink is not drowsiness
    assert out.events == [
        {
            "onset_s": 2.0,
            "offset_s": 5.0,
            "type": "distraction_episode",
            "uncertain": False,
            "source": "dmd_label",
        },
        {
            "onset_s": 6.0,
            "offset_s": 7.3,
            "type": "micro_sleep",
            "uncertain": False,
            "source": "dmd_label",
        },
    ]
    assert validate_events(out.events, out.n_steps * S.STEP_S) == []


def test_unknown_labels_are_reported_and_strict_mode_refuses() -> None:
    out = D.dmd_labels(_doc(SCENE), _map(), fps=FPS, n_frames=N_FRAMES)
    assert out.report.unknown_labels == {"driver_actions/something_new": 1}
    assert out.report.ignored_labels == {"eyes_state/open": 1}
    with pytest.raises(D.UnknownLabelError, match="something_new"):
        D.dmd_labels(_doc(SCENE), _map(), fps=FPS, n_frames=N_FRAMES, strict=True)


def test_rejected_event_intervals_are_reported() -> None:
    scene = [
        ("driver_actions/safe_drive", [(0, 299)]),
        ("eyes_state/close", [(30, 35), (120, 239)]),  # 0.2 s and 4.0 s
    ]
    out = D.dmd_labels(_doc(scene), _map(), fps=FPS, n_frames=N_FRAMES)
    assert out.events == []
    reasons = sorted(r["reason"] for r in out.report.rejected_events)
    assert len(reasons) == 2
    assert any("min_s" in r for r in reasons) and any("max_s" in r for r in reasons)
    long_one = next(r for r in out.report.rejected_events if "max_s" in r["reason"])
    assert long_one["type"] == "micro_sleep" and long_one["onset_s"] == 4.0
    assert long_one["offset_s"] == 8.0
    # the 4 s closure is not labelled drowsy either: that would be a guess (gap in 03 5.2)
    assert (out.y_cls[45:75] == S.CLASS_TO_ID["normal"]).all()


def test_adjacent_labels_of_one_event_type_merge() -> None:
    scene = [
        ("driver_actions/safe_drive", [(0, 299)]),
        ("driver_actions/texting_right", [(60, 89)]),
        ("driver_actions/phonecall_right", [(90, 149)]),
    ]
    out = D.dmd_labels(_doc(scene), _map(), fps=FPS, n_frames=N_FRAMES)
    assert [(e["onset_s"], e["offset_s"]) for e in out.events] == [(2.0, 5.0)]


def test_short_gap_merges_only_within_merge_gap() -> None:
    scene = [("driver_actions/texting_right", [(60, 89), (96, 149)])]  # 0.2 s gap
    events = {"distraction_episode": {"min_s": 2.0, "max_s": None, "merge_gap_s": 0.25}}
    merged = D.dmd_labels(
        _doc(scene), _map(events={**MAP["events"], **events}), fps=FPS, n_frames=N_FRAMES
    )
    assert [(e["onset_s"], e["offset_s"]) for e in merged.events] == [(2.0, 5.0)]
    split = D.dmd_labels(_doc(scene), _map(), fps=FPS, n_frames=N_FRAMES)
    assert split.events == []  # without merging: 1.0 s and 1.8 s, both below min_s = 2 s
    assert len(split.report.rejected_events) == 2


def test_conflicting_classes_give_unlabelled_steps() -> None:
    scene = [
        ("driver_actions/texting_right", [(0, 299)]),
        ("yawning/yawning", [(150, 299)]),
    ]
    out = D.dmd_labels(_doc(scene), _map(), fps=FPS, n_frames=N_FRAMES)
    assert (out.y_cls[5:45] == S.CLASS_TO_ID["distracted"]).all()
    assert (out.y_cls[55:] == S.UNLABELED).all()
    # frames 150-297: frames 298 and 299 fall into step 100, past the grid (as in vision.py)
    assert out.report.conflict_frames == 148
    assert out.report.clipped_intervals == 0  # the unused tail frames are not "clipping"


def test_frames_without_class_evidence_stay_unlabelled() -> None:
    scene = [("eyes_state/open", [(0, 299)]), ("driver_actions/safe_drive", [(150, 299)])]
    out = D.dmd_labels(_doc(scene), _map(), fps=FPS, n_frames=N_FRAMES)
    assert (out.y_cls[:49] == S.UNLABELED).all()
    assert (out.y_cls[52:] == S.CLASS_TO_ID["normal"]).all()


def test_frame_count_sources_shift_and_clipping() -> None:
    doc = _doc(SCENE)
    hinted = D.dmd_labels(doc, _map(), fps=FPS)
    assert hinted.n_steps == 100 and hinted.report.n_frames_source == "document frame_intervals"
    shifted = D.dmd_labels(doc, _map(), fps=FPS, n_frames=N_FRAMES, frame_shift=30)
    assert [e["onset_s"] for e in shifted.events] == [3.0, 7.0]
    short = D.dmd_labels(doc, _map(), fps=FPS, n_frames=N_FRAMES, n_steps=60)
    assert len(short.y_cls) == 60
    assert short.events[0]["offset_s"] == 5.0
    assert short.report.clipped_intervals >= 1  # frames beyond step 59 ignored, and counted
    assert validate_events(short.events, 60 * S.STEP_S) == []


def test_bad_fps_is_refused() -> None:
    with pytest.raises(ValueError, match="fps"):
        D.dmd_labels(_doc(SCENE), _map(), fps=0.0, n_frames=N_FRAMES)
