"""DMD sessions -> contract trips (video features + labels), on synthetic landmarks.

File names copy the real DMD naming, e.g. `gA_1_s5_2019-03-14T14;26;17+01;00_rgb_face.mp4`.
The landmarks are synthetic faces with known eye openness; no video and no MediaPipe are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driverisk.data import dmd as D
from driverisk.data import dmd_trips as T
from driverisk.data import schema as S
from driverisk.data.validate import (
    validate_events,
    validate_meta,
    validate_processed_dir,
    validate_trip_frame,
)
from driverisk.features import vision as V

FPS = 29.76
SECONDS = 20.0
N_FRAMES = int(SECONDS * FPS)
STEM_A = "gA_1_s5_2019-03-14T14;26;17+01;00"
STEM_B = "gA_1_s2_2019-03-14T16;00;00+01;00"
STEM_C = "gB_6_s5_2019-03-13T13;37;11+01;00"
W_EYE = 0.06


def _face(ear: float) -> np.ndarray:
    p = np.full((478, 3), 0.5)

    def eye(idx, x0, top, bottom):
        a, t1, t2, b, b2, b1 = idx
        h, y = ear * W_EYE, 0.45
        p[a, :2], p[b, :2] = (x0, y), (x0 + W_EYE, y)
        p[t1, :2], p[t2, :2] = (x0 + 0.02, y - h / 2), (x0 + 0.04, y - h / 2)
        p[b1, :2], p[b2, :2] = (x0 + 0.02, y + h / 2), (x0 + 0.04, y + h / 2)
        p[top, :2], p[bottom, :2] = (x0 + 0.03, y - h / 2), (x0 + 0.03, y + h / 2)

    eye(V.RIGHT_EYE, 0.40, *V.RIGHT_LIDS)
    eye(V.LEFT_EYE, 0.54, *V.LEFT_LIDS)
    for iris, x0 in ((V.RIGHT_IRIS, 0.40), (V.LEFT_IRIS, 0.54)):
        p[iris, :2] = (x0 + 0.03, 0.45)
    top, bottom, left, right = V.MOUTH
    p[left, :2], p[right, :2] = (0.45, 0.65), (0.55, 0.65)
    p[top, :2], p[bottom, :2] = (0.5, 0.64), (0.5, 0.66)
    return p


def _landmarks(closed: list[tuple[float, float]] | None = None, n: int = N_FRAMES):
    """Frames at FPS; eyes closed (EAR 0.05) inside the given (start_s, end_s) spans."""
    t = np.arange(n) / FPS
    lm = np.empty((n, 478, 3))
    for i, ti in enumerate(t):
        shut = any(a <= ti < b for a, b in (closed or []))
        lm[i] = _face(0.05 if shut else 0.30)
    # the geometry above is in square units; MediaPipe reports y / height on a 16:9 frame
    lm[:, :, 1] = 0.5 + (lm[:, :, 1] - 0.5) / (720 / 1280)
    mats = np.repeat(np.eye(4)[None], n, axis=0)
    return t, lm, mats


def _annotation(spans: dict[str, list[tuple[float, float]]], n_frames: int = N_FRAMES) -> dict:
    """OpenLABEL-like document; `spans` are seconds, converted to inclusive frame intervals."""
    actions = {
        str(k): {
            "name": "",
            "type": typ,
            "frame_intervals": [
                {"frame_start": int(a * FPS), "frame_end": int(b * FPS) - 1} for a, b in ivs
            ],
        }
        for k, (typ, ivs) in enumerate(spans.items())
    }
    return {
        "openlabel": {
            "metadata": {"schema_version": "1.0.0"},
            "frame_intervals": [{"frame_start": 0, "frame_end": n_frames - 1}],
            "actions": actions,
        }
    }


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "dmd"
    for stem, kind in ((STEM_A, "drowsiness"), (STEM_B, "distraction"), (STEM_C, "drowsiness")):
        group, subject, session = stem.split("_")[:3]
        folder = root / "dmd" / group / subject / session
        _touch(folder / f"{stem}_rgb_face.mp4")
        _touch(folder / f"{stem}_rgb_body.mp4")
        _touch(folder / f"{stem}_rgb_mosaic.avi")
        (folder / f"{stem}_rgb_ann_{kind}.json").write_text(
            json.dumps(_annotation({"eyes_state/close": [(5.0, 6.5)]})), encoding="utf-8"
        )
    return root


# --- sessions and ids ----------------------------------------------------------------------
def test_sessions_are_found_with_their_annotation_and_face_video(tree: Path) -> None:
    sessions = T.find_sessions(tree)
    assert [s.stem for s in sessions] == sorted([STEM_A, STEM_B, STEM_C])
    s = next(s for s in sessions if s.stem == STEM_A)
    assert (s.group, s.subject, s.session, s.date) == ("gA", 1, 5, "20190314")
    assert (
        s.annotation_type == "drowsiness"
        and s.video is not None
        and s.video.name.endswith("face.mp4")
    )


def test_session_without_a_face_video_is_still_listed(tree: Path) -> None:
    next(tree.rglob("*gB_6_s5*_rgb_face.mp4")).unlink()
    s = next(s for s in T.find_sessions(tree) if s.stem == STEM_C)
    assert s.video is None


def test_ids_follow_the_contract_and_count_sessions_per_driver_and_day(tree: Path) -> None:
    ids = T.assign_ids(T.find_sessions(tree))
    assert ids[STEM_B] == ("d01", "dmd_d01_20190314_01")  # session s2 comes before s5
    assert ids[STEM_A] == ("d01", "dmd_d01_20190314_02")
    assert ids[STEM_C] == ("d06", "dmd_d06_20190313_01")
    assert all(S.TRIP_ID_RE.match(v[1]) and S.DRIVER_ID_RE.match(v[0]) for v in ids.values())


def test_unparseable_session_name_is_reported_not_guessed(tmp_path: Path) -> None:
    _touch(tmp_path / "odd_rgb_ann_drowsiness.json")
    with pytest.raises(T.DmdSessionError, match="odd_rgb_ann_drowsiness"):
        T.find_sessions(tmp_path)


# --- building one trip -----------------------------------------------------------------------
def _build(spans, closed, **kw):
    t, lm, mats = _landmarks(closed)
    session = T.SessionInfo(STEM_A, "gA", 1, 5, "20190314", Path("x.json"), None, "drowsiness")
    return T.build_dmd_trip(
        session, t, lm, mats, (1280, 720), FPS, _annotation(spans), D.load_label_map(),
        "dmd_d01_20190314_01", "d01", **kw,
    )  # fmt: skip


@pytest.fixture(scope="module")
def trip() -> T.DmdTrip:
    spans = {"eyes_state/close": [(5.0, 6.5)], "yawning/Yawning with hand": [(10.0, 12.0)]}
    return _build(spans, closed=[(5.0, 6.5)])


def test_trip_has_exactly_the_contract_columns_and_passes_the_validator(trip: T.DmdTrip) -> None:
    assert list(trip.frame.columns) == list(S.PARQUET_COLUMNS)
    assert validate_trip_frame(trip.frame, trip.meta) == []
    assert validate_meta(trip.meta, "dmd_d01_20190314_01") == []
    assert validate_events(trip.events, float(trip.frame[S.COL_T].iloc[-1])) == []
    assert (trip.frame[S.COL_Y_CLS].dtype, trip.frame[S.COL_T].dtype) == ("int8", "float32")


def test_vision_only_motion_is_declared_missing(trip: T.DmdTrip) -> None:
    f = trip.frame
    assert f[list(S.MOTION_COLUMNS)].isna().all().all() and not f[S.COL_MASK_M].any()
    assert set(S.MOTION_COLUMNS) <= set(trip.meta[S.META_ALL_NAN_KEY])
    assert trip.meta["pairing"] == "vision_only" and trip.manifest_row["pairing"] == "vision_only"


def test_eye_closure_in_the_video_matches_the_annotation(trip: T.DmdTrip) -> None:
    f = trip.frame
    t = f[S.COL_T].to_numpy(np.float64)
    closed = f["v_eye_closed"].to_numpy(np.float64)
    assert (closed[(t > 5.3) & (t < 6.4)] == 1.0).all() and (closed[t < 4.8] == 0.0).all()
    drowsy = f[S.COL_Y_CLS].to_numpy() == S.CLASS_TO_ID["drowsy"]
    assert drowsy[(t > 5.3) & (t < 6.4)].all() and drowsy[(t > 10.5) & (t < 11.8)].all()
    assert (f.loc[t < 4.5, S.COL_Y_CLS] == S.UNLABELED).all()  # no normal label in these files


def test_micro_sleep_event_comes_from_the_annotation(trip: T.DmdTrip) -> None:
    ev = [e for e in trip.events if e["type"] == "micro_sleep"]
    assert len(ev) == 1 and ev[0]["source"] == "dmd_label"
    assert ev[0]["onset_s"] == pytest.approx(5.0, abs=0.15)
    assert ev[0]["offset_s"] == pytest.approx(6.4, abs=0.15)


def test_meta_carries_provenance_and_the_label_report(trip: T.DmdTrip) -> None:
    m = trip.meta
    assert m["ear_threshold"] == V.load_vision_config()["ear_threshold"]
    assert m["ear_threshold_source"] == "global_default"
    d = m["dmd"]
    assert (d["group"], d["subject"], d["session"]) == ("gA", 1, 5)
    assert d["annotation_type"] == "drowsiness" and d["video_fps"] == pytest.approx(FPS)
    assert d["label_report"]["unknown_labels"] == {} and d["frame_shift"] == 0
    assert "CC BY-NC-ND" in m["notes"]  # licence reminder travels with the data


def test_manifest_row(trip: T.DmdTrip) -> None:
    row = trip.manifest_row
    assert row["trip_id"] == "dmd_d01_20190314_01" and row["dataset"] == "dmd"
    assert row["driver_id"] == "d01" and row["behavior_tag"] == "drowsiness_s5"
    assert row["n_steps"] == len(trip.frame)
    assert set(row) == set(S.MANIFEST_COLUMNS)  # the file-level check is in the dataset test


def test_unknown_annotation_label_is_reported_in_the_meta() -> None:
    trip = _build({"eyes_state/close": [(5.0, 6.5)], "mystery/label": [(1.0, 2.0)]}, [(5.0, 6.5)])
    assert trip.meta["dmd"]["label_report"]["unknown_labels"] == {"mystery/label": 1}


def test_frame_shift_moves_the_labels_not_the_video() -> None:
    plain = _build({"eyes_state/close": [(5.0, 6.5)]}, [(5.0, 6.5)])
    shifted = _build({"eyes_state/close": [(5.0, 6.5)]}, [(5.0, 6.5)], frame_shift=30)
    ev = lambda tr: [e["onset_s"] for e in tr.events if e["type"] == "micro_sleep"]  # noqa: E731
    assert ev(shifted)[0] == pytest.approx(ev(plain)[0] + 30 / FPS, abs=0.11)
    assert shifted.meta["dmd"]["frame_shift"] == 30
    pd.testing.assert_series_equal(plain.frame["v_eye_closed"], shifted.frame["v_eye_closed"])


def test_annotation_longer_than_the_video_is_noted_not_hidden() -> None:
    t, lm, mats = _landmarks(None, n=N_FRAMES - 1)  # video one frame shorter, as in real gA_1
    session = T.SessionInfo(STEM_A, "gA", 1, 5, "20190314", Path("x.json"), None, "drowsiness")
    trip = T.build_dmd_trip(
        session, t, lm, mats, (1280, 720), FPS, _annotation({}, N_FRAMES), D.load_label_map(),
        "dmd_d01_20190314_01", "d01",
    )  # fmt: skip
    assert trip.meta["dmd"]["n_frames_annotation"] == N_FRAMES
    assert trip.meta["dmd"]["n_frames_video"] == N_FRAMES - 1


# --- cache and dataset -----------------------------------------------------------------------
def test_landmark_cache_round_trip(tmp_path: Path) -> None:
    t, lm, mats = _landmarks([(5.0, 6.0)])
    path = T.cache_path(tmp_path, STEM_A)
    assert path.name == "gA_1_s5_2019-03-14T14-26-17p01-00.npz"  # no ';' or '+' in cache names
    T.save_landmark_cache(path, lm, mats, FPS, (1280, 720))
    got = T.load_landmark_cache(path)
    assert got.fps == pytest.approx(FPS) and got.size == (1280, 720)
    assert got.landmarks.shape == lm.shape and np.allclose(got.landmarks, lm, atol=1e-6)
    assert np.allclose(got.t, t)


def test_process_dataset_writes_validated_trips_and_a_manifest(tree: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for stem in (STEM_A, STEM_B, STEM_C):
        t, lm, mats = _landmarks([(5.0, 6.5)])
        T.save_landmark_cache(T.cache_path(cache, stem), lm, mats, FPS, (1280, 720))
    out = tmp_path / "processed"
    result = T.process_dataset(tree, out, cache)
    assert len(result.trips) == 3 and not result.failures
    assert validate_processed_dir(out).ok
    manifest = pd.read_csv(out / "manifest.csv", dtype=str)
    assert sorted(manifest["driver_id"]) == ["d01", "d01", "d06"]
    with pytest.raises(FileExistsError):
        T.process_dataset(tree, out, cache)


def test_missing_cache_and_no_extractor_gives_a_clear_failure(tree: Path, tmp_path: Path) -> None:
    result = T.process_dataset(tree, tmp_path / "o", tmp_path / "empty_cache", keep_going=True)
    assert len(result.failures) == 3 and "landmark" in result.failures[0][1].lower()
