"""MediaPipe Face Landmarker adapter: video frames -> landmarks + face transforms.

Produces exactly the inputs of `features.vision.extract_vision_features`:
`landmarks [478, 3]` in normalized image coordinates (NaN when no face) and the 4x4
face-to-camera transform. MediaPipe is an optional dependency (`uv sync --group vision`), so it is
imported only when this module is used. Pinned version and model: ADR-0006.

Frames must be RGB `uint8 [H, W, 3]` and **not mirrored** (a webcam preview is often mirrored;
the left/right conventions in `vision.py` assume the raw camera image).
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from driverisk.utils.paths import repo_root

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)
N_LANDMARKS = 478


def default_model_path() -> Path:
    return repo_root() / "models" / "face_landmarker.task"  # git-ignored


def ensure_model(path: Path | None = None) -> Path:
    """Download the Face Landmarker model once (about 4 MB) if it is not there yet."""
    path = path or default_model_path()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    return path


class FaceLandmarkerAdapter:
    """One driver face per frame, VIDEO mode (timestamps must increase)."""

    def __init__(
        self,
        model_path: Path | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        options = vision.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, rgb: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Landmarks `[478, 3]` and transform `[4, 4]` of one frame; NaN arrays when no face."""
        import mediapipe as mp

        ms = max(int(round(t_s * 1000.0)), self._last_ms + 1)  # strictly increasing
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return np.full((N_LANDMARKS, 3), np.nan), np.full((4, 4), np.nan)
        lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float64)
        if result.facial_transformation_matrixes:
            mat = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        else:
            mat = np.full((4, 4), np.nan)
        return lm, mat

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def run_on_frames(
    frames: Iterator[tuple[float, np.ndarray]], model_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(t_s, rgb)` pairs -> `(t [F], landmarks [F, 478, 3], matrices [F, 4, 4])`."""
    ts, lms, mats = [], [], []
    with FaceLandmarkerAdapter(model_path) as adapter:
        for t_s, rgb in frames:
            lm, mat = adapter.process(rgb, t_s)
            ts.append(t_s)
            lms.append(lm)
            mats.append(mat)
    if not ts:
        raise ValueError("no frames")
    return np.array(ts), np.stack(lms), np.stack(mats)


def video_frames(
    source: str | int, max_seconds: float | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Frames from a video file (its own timestamps) or a camera index (wall-clock time)."""
    import time

    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"cannot open video source {source!r}")
    live = isinstance(source, int)
    t0 = time.monotonic()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_s = time.monotonic() - t0 if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if max_seconds is not None and t_s > max_seconds:
                break
            yield t_s, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
