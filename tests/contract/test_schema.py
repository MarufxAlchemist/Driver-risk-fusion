"""schema.py must agree with docs/03_data_contract.md."""

from __future__ import annotations

import re

import pytest

from driverisk.data import schema as S
from driverisk.utils.paths import repo_root

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*`([a-z0-9_]+)`\s*\|")


def _doc_table(section_title: str) -> list[tuple[int, str]]:
    """Rows `| idx | `name` |` of the feature table under a heading."""
    text = (repo_root() / "docs" / "03_data_contract.md").read_text(encoding="utf-8")
    start = text.index(section_title)
    end = text.index("\n### ", start + 1)
    return [(int(m[1]), m[2]) for line in text[start:end].splitlines() if (m := ROW.match(line))]


def test_feature_counts() -> None:
    assert S.DV == 16
    assert S.DM == 14


@pytest.mark.parametrize(
    ("title", "specs"),
    [
        ("### 4.1 Vision features", S.VISION_FEATURES),
        ("### 4.2 Motion features", S.MOTION_FEATURES),
    ],
)
def test_features_match_contract_doc(title: str, specs: tuple[S.FeatureSpec, ...]) -> None:
    assert _doc_table(title) == [(f.index, f.name) for f in specs]


def test_indices_sequential_and_names_unique() -> None:
    for specs in (S.VISION_FEATURES, S.MOTION_FEATURES):
        assert [f.index for f in specs] == list(range(len(specs)))
        assert len({f.name for f in specs}) == len(specs)
        assert all(f.lo < f.hi for f in specs)


def test_parquet_columns() -> None:
    assert len(S.PARQUET_COLUMNS) == 1 + 16 + 14 + 4
    assert len(set(S.PARQUET_COLUMNS)) == len(S.PARQUET_COLUMNS)
    assert S.PARQUET_COLUMNS[0] == "t"


def test_labels() -> None:
    assert S.CLASSES == ("normal", "aggressive", "drowsy", "distracted")
    assert S.VALID_Y_CLS == {-1, 0, 1, 2, 3}
    assert len(S.EVENT_TYPES) == 6


def test_window_constants() -> None:
    assert S.WINDOW_STEPS * S.STEP_S == pytest.approx(5.0)
    assert S.STRIDE_STEPS * S.STEP_S == pytest.approx(1.0)
    assert S.HORIZONS_S == (2, 3, 5)


@pytest.mark.parametrize(
    ("value", "ok"),
    [("own_d03_20261120_02", True), ("uah_d1_20261120_02", False), ("xyz_d03_20261120_02", False)],
)
def test_trip_id_pattern(value: str, ok: bool) -> None:
    assert bool(S.TRIP_ID_RE.match(value)) is ok
