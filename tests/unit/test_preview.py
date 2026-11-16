"""Live preview (optional OpenCV). Rendering and the blink counter are tested without a window."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from driverisk.features import vision as V  # noqa: E402
from driverisk.features.preview import LivePreview  # noqa: E402


def _landmarks() -> np.ndarray:
    lm = np.full((478, 3), 0.5)
    lm[:, 0] += np.linspace(-0.1, 0.1, 478)  # spread the points so geometry is not degenerate
    lm[:, 1] += np.linspace(-0.1, 0.1, 478)[::-1] * 0.5
    return lm


def test_render_draws_on_a_copy_and_keeps_the_size() -> None:
    p = LivePreview(show=False)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    lm = _landmarks()
    feats = V.frame_features(lm, np.eye(4), p.cfg)
    img = p.render(rgb, lm, feats, 1.0)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8 and img.any()
    assert not rgb.any()  # the input frame is untouched


def test_render_without_a_face_says_so() -> None:
    p = LivePreview(show=False)
    img = p.render(np.zeros((120, 200, 3), dtype=np.uint8), np.full((478, 3), np.nan), None, 0.0)
    assert img.any()


def test_blink_counter_counts_short_closures_only() -> None:
    p = LivePreview(show=False)
    thr = p.thr
    ear = {True: thr - 0.1, False: thr + 0.1}
    seq = [
        (0.0, False),
        (0.1, True),
        (0.3, False),  # 0.2 s closure: blink
        (1.0, True),
        (2.5, False),  # 1.5 s closure: not a blink
        (3.0, True),
        (3.2, float("nan")),
        (3.3, False),
    ]  # face lost mid-closure: not counted
    for t, closed in seq:
        p.update_blinks(t, float("nan") if closed != closed else ear[closed])  # noqa: PLR0124
    assert p.blinks == 1


def test_callback_without_window_keeps_running() -> None:
    p = LivePreview(show=False)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    assert p(0.0, rgb, _landmarks(), np.eye(4)) is True
    assert p(0.1, rgb, np.full((478, 3), np.nan), np.full((4, 4), np.nan)) is True
