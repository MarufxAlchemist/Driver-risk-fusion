"""Regression tests on the REAL UAH-DriveSet and DMD files. Skipped unless the data is on disk.

The datasets are not in git (licences), so CI skips these; they run on a machine that has
`data/raw/uah_driveset` and `data/raw/dmd`. Every expected number below was measured on the real
files on 2026-11-18 (docs/PROGRESS.md) and guards the quirks the loaders had to learn:
NORMAL1/NORMAL2 folder names, 12-column GPS, lane states 1 and 2, a yaw column that is phone
attitude, repeated start-up timestamps, one exact duplicated block, unsorted and empty event files,
editor backup files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from driverisk.data import dmd as D
from driverisk.data import schema as S
from driverisk.data import uah as U
from driverisk.utils.paths import data_dir

UAH_ROOT = data_dir() / "raw" / "uah_driveset"
DMD_ROOT = data_dir() / "raw" / "dmd"

needs_uah = pytest.mark.skipif(not UAH_ROOT.is_dir(), reason="data/raw/uah_driveset not present")
needs_dmd = pytest.mark.skipif(not DMD_ROOT.is_dir(), reason="data/raw/dmd not present")


def _route(fragment: str) -> Path:
    hits = [p for p in U.find_route_folders(UAH_ROOT) if fragment in p.name]
    assert len(hits) == 1, (fragment, hits)
    return hits[0]


@needs_uah
def test_all_forty_routes_are_found_and_parse() -> None:
    folders = U.find_route_folders(UAH_ROOT)
    assert len(folders) == 40
    routes = [U.parse_route_folder(f) for f in folders]
    assert {r.behaviour for r in routes} == {"NORMAL", "DROWSY", "AGGRESSIVE"}
    assert sum(r.behaviour_raw in ("NORMAL1", "NORMAL2") for r in routes) == 10
    assert len({r.driver_id for r in routes}) == 6


@needs_uah
@pytest.mark.parametrize(
    ("fragment", "check"),
    [
        ("D1-NORMAL1-SECONDARY", "digit behaviour"),
        ("D6-DROWSY-MOTORWAY", "duplicated block"),
        ("D4-NORMAL2-SECONDARY", "empty events file"),
        ("D1-DROWSY-MOTORWAY", "unsorted events"),
        ("D5-DROWSY-MOTORWAY", "backup file"),
    ],
)
def test_the_quirky_real_routes_convert_and_validate(fragment: str, check: str) -> None:
    cfg, label_map = U.load_uah_config(), U.load_label_map()
    folder = _route(fragment)
    trip = U.convert_route(folder, "uah_d01_20151111_01", cfg, label_map)
    from driverisk.data.validate import validate_trip_frame

    assert validate_trip_frame(trip.frame, trip.meta) == [], check
    assert trip.frame["m_yaw_rate"].isna().all()  # yaw column is phone attitude, never used
    assert np.isfinite(trip.frame["m_speed"]).mean() > 0.95
    assert trip.frame["m_acc_long"].abs().max() > 0.1  # not all zero: converted, not dropped


@needs_uah
def test_real_units_and_the_event_type_semantics() -> None:
    """Braking events must show negative acc_long, accelerations positive (Z axis, + forward)."""
    cfg, label_map = U.load_uah_config(), U.load_label_map()
    trip = U.convert_route(_route("D1-AGGRESSIVE-MOTORWAY"), "uah_d01_20151111_01", cfg, label_map)
    f = trip.frame
    t = f[S.COL_T].to_numpy(np.float64)
    peaks = {"harsh_brake": [], "harsh_accel": []}
    for e in trip.events:
        if e["type"] in peaks:
            w = (t >= e["onset_s"] - 1.5) & (t <= e["onset_s"] + 1.5)
            x = f["m_acc_long"].to_numpy(np.float64)[w]
            peaks[e["type"]].append(x[np.nanargmax(np.abs(x))])
    assert len(peaks["harsh_brake"]) >= 3
    assert np.mean(np.array(peaks["harsh_brake"]) < 0) > 0.9
    assert np.nanmax(np.abs(f["m_acc_long"])) < 4.0 * 9.80665  # sane m/s^2
    assert np.nanmedian(np.abs(f["m_acc_vert"])) < 0.5  # no gravity left in the vertical axis


@needs_uah
def test_aggressive_trips_have_more_events_than_normal_ones() -> None:
    cfg, label_map = U.load_uah_config(), U.load_label_map()
    rate = {}
    for key in ("D2-AGGRESSIVE-MOTORWAY", "D2-NORMAL-MOTORWAY"):
        trip = U.convert_route(_route(key), "uah_d02_20151120_01", cfg, label_map)
        segs = trip.meta["inertial_events"]["logged_segments_s"]
        hours = sum(b - a + S.STEP_S for a, b in segs) / 3600  # only time above 50 km/h counts
        rate[key] = len(trip.events) / hours
    assert rate["D2-AGGRESSIVE-MOTORWAY"] > 2 * rate["D2-NORMAL-MOTORWAY"]


DMD_EXPECTED = {  # session -> (frames in annotation, kept micro_sleep events), measured 2026-11-18
    "gA_1_s5": (5481, 6),
    "gA_5_s5": (5287, 4),
    "gB_6_s5": (5410, 1),
}


@needs_dmd
@pytest.mark.parametrize("session", sorted(DMD_EXPECTED))
def test_dmd_drowsiness_annotations_parse_with_no_unknown_label(session: str) -> None:
    path = next(DMD_ROOT.rglob(f"{session}_*_rgb_ann_drowsiness.json"))
    doc = D.load_openlabel(path)
    res = D.dmd_labels(doc, D.load_label_map(), fps=29.76)
    n_frames, n_micro = DMD_EXPECTED[session]
    assert res.report.n_frames == n_frames
    assert res.report.unknown_labels == {} and res.report.problems == []
    assert res.report.conflict_frames == 0
    assert sum(e["type"] == "micro_sleep" for e in res.events) == n_micro
    labelled = res.y_cls[res.y_cls != S.UNLABELED]
    assert set(labelled.tolist()) <= {S.CLASS_TO_ID["drowsy"]}  # these files carry no normal label
