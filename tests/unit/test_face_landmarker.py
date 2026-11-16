"""MediaPipe adapter smoke test. Skipped unless the optional `vision` group and the model exist."""

from __future__ import annotations

import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")

from driverisk.features import face_landmarker as FL  # noqa: E402
from driverisk.features import vision as V  # noqa: E402

pytestmark = pytest.mark.skipif(
    not FL.default_model_path().is_file(), reason="models/face_landmarker.task not downloaded"
)


def test_frames_without_a_face_give_nan_and_feed_the_extractor() -> None:
    blank = np.full((240, 320, 3), 127, dtype=np.uint8)
    frames = ((k / 15.0, blank) for k in range(15))
    t, lm, mats, size = FL.run_on_frames(frames)
    assert size == (320, 240)
    assert lm.shape == (15, FL.N_LANDMARKS, 3) and mats.shape == (15, 4, 4)
    assert np.isnan(lm).all()
    res = V.extract_vision_features(t, lm, mats, image_size=size)
    assert (res.frame["v_face_visible"] == 0).all() and not res.frame["mask_v"].any()


def test_timestamps_are_forced_to_increase() -> None:
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    with FL.FaceLandmarkerAdapter() as adapter:
        adapter.process(blank, 1.0)
        adapter.process(blank, 1.0)  # same time again must not raise
