"""UAH inspector on a tiny synthetic tree that follows the paper's description of the layout.

The real file names and name format are unverified (that is what the inspector is for), so
this only tests the parsing and statistics logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from driverisk.data.uah_inspect import FOLDER_RE, inspect_dataset, inspect_trip, read_table


def _write(path: Path, rate_hz: float, seconds: float, n_cols: int) -> None:
    t = np.arange(0, seconds, 1 / rate_hz)
    cols = [t] + [np.full_like(t, 0.1 * (i + 1)) for i in range(n_cols - 1)]
    np.savetxt(path, np.column_stack(cols), fmt="%.4f")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    trip = tmp_path / "D1" / "20151111123124-25km-D1-NORMAL-MOTORWAY"
    trip.mkdir(parents=True)
    _write(trip / "RAW_ACCELEROMETERS.txt", 10.0, 60.0, 12)
    _write(trip / "RAW_GPS.txt", 1.0, 60.0, 9)
    (trip / "video.mp4").write_bytes(b"")
    odd = tmp_path / "D2" / "some-other-name"
    odd.mkdir(parents=True)
    _write(odd / "raw_accelerometers.txt", 10.0, 10.0, 12)
    return tmp_path


def test_read_table_rate_and_columns(tree: Path) -> None:
    info = read_table(next(tree.rglob("RAW_ACCELEROMETERS.txt")))
    assert info.n_cols == 12 and info.n_rows == 600
    assert info.median_rate_hz == pytest.approx(10.0)
    assert info.duration_s == pytest.approx(59.9)


def test_inspect_trip_parses_name_and_finds_files(tree: Path) -> None:
    t = inspect_trip(next(tree.rglob("20151111123124*")))
    assert t.parsed and (t.driver, t.behavior, t.road) == ("D1", "NORMAL", "MOTORWAY")
    assert t.distance_km == 25.0 and t.has_video
    assert set(t.files) == {"acc", "gps"} and "lane" in t.missing


def test_unparsed_folders_are_reported_not_guessed(tree: Path) -> None:
    trips = inspect_dataset(tree)
    assert len(trips) == 2
    odd = next(t for t in trips if not t.parsed)
    assert odd.folder == "some-other-name" and odd.driver == ""  # case-insensitive file match


def test_folder_regex_rejects_garbage() -> None:
    assert FOLDER_RE.match("20151111123124-25km-D1-NORMAL-MOTORWAY")
    assert not FOLDER_RE.match("D1")
