"""Live preview window for the webcam check: see yourself and what the extractor reads.

Shows the camera image **mirrored** (like a selfie, easier to use) with the eye, iris and mouth
landmarks drawn on it and a panel with the live per-frame values: EAR and open/closed, head
yaw/pitch/roll, gaze, blink count, face confidence. All processing uses the raw, un-mirrored
frame; only the picture is flipped. Press `q` or `Esc` to stop early.

The blink counter here is a per-frame approximation for feedback; the stored features use the
10 Hz step logic in `features/vision.py`. OpenCV comes with the optional `vision` group and is
imported only when a preview is created.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from driverisk.features import vision as V

_GREEN, _RED, _YELLOW, _WHITE, _CYAN = (
    (60, 200, 60),
    (40, 40, 230),
    (0, 210, 255),
    (255, 255, 255),
    (230, 200, 0),
)
_EYE_POINTS = (*V.RIGHT_EYE, *V.LEFT_EYE, *V.RIGHT_LIDS, *V.LEFT_LIDS)


class LivePreview:
    """Callable for `face_landmarker.run_on_frames(on_frame=...)`; returns False to stop."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        ear_threshold: float | None = None,
        window: str = "driverisk vision check (q / Esc to stop)",
        show: bool = True,
    ) -> None:
        self.cfg = cfg or V.load_vision_config()
        self.thr = float(self.cfg["ear_threshold"] if ear_threshold is None else ear_threshold)
        self.window, self.show = window, show
        self.blinks = 0
        self._closed_since: float | None = None
        self._last_t: float | None = None
        self._fps = 0.0

    # --- state -----------------------------------------------------------------------------
    def update_blinks(self, t: float, ear: float) -> None:
        """Count a blink when the eye reopens after a closure no longer than `blink_max_s`."""
        if not np.isfinite(ear):
            self._closed_since = None
            return
        if ear < self.thr:
            if self._closed_since is None:
                self._closed_since = t
        elif self._closed_since is not None:
            if t - self._closed_since <= self.cfg["blink_max_s"]:
                self.blinks += 1
            self._closed_since = None

    # --- drawing ---------------------------------------------------------------------------
    def render(self, rgb: np.ndarray, lm: np.ndarray, feats: dict[str, float] | None, t: float):
        """BGR image to display: mirrored frame + landmarks + value panel. Pure, no window."""
        import cv2

        h, w = rgb.shape[:2]
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if feats is not None:
            for i in _EYE_POINTS:
                cv2.circle(img, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 2, _CYAN, -1)
            for i in V.MOUTH:
                cv2.circle(img, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 2, _YELLOW, -1)
            if len(lm) > max(V.RIGHT_IRIS, V.LEFT_IRIS):
                for i in (V.RIGHT_IRIS, V.LEFT_IRIS):
                    cv2.circle(img, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 4, _GREEN, 1)
        img = cv2.flip(img, 1)  # mirror for the viewer only

        lines: list[tuple[str, tuple[int, int, int]]] = []
        if feats is None:
            lines.append(("NO FACE", _RED))
        else:
            ear = feats["ear_mean"]
            if np.isfinite(ear):
                closed = ear < self.thr
                state, colour = ("CLOSED", _RED) if closed else ("OPEN", _GREEN)
                lines.append((f"EAR {ear:.2f} (thr {self.thr:.2f})  eyes {state}", colour))
            else:
                lines.append(("eyes not read (head turned too far)", _YELLOW))
            lines.append((
                f"head yaw {feats['head_yaw']:+6.1f}  pitch {feats['head_pitch']:+6.1f}  "
                f"roll {feats['head_roll']:+6.1f} deg", _WHITE,
            ))  # fmt: skip
            lines.append((
                f"gaze yaw {feats['gaze_yaw']:+6.1f}  pitch {feats['gaze_pitch']:+6.1f} deg", _WHITE
            ))  # fmt: skip
            lines.append((f"face_conf {feats['face_conf']:.2f}   MAR {feats['mar']:.2f}", _WHITE))
        lines.append((f"blinks {self.blinks}   t {t:5.1f} s   {self._fps:4.1f} fps", _WHITE))

        pad, step = 8, 22
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (min(w, 470), pad + step * len(lines) + 4), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for k, (text, colour) in enumerate(lines):
            cv2.putText(img, text, (pad, pad + step * (k + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, colour, 1, cv2.LINE_AA)  # fmt: skip
        hint = "check: turn to YOUR right -> yaw +   look up -> pitch +   (mirror view)"
        cv2.putText(img, hint, (pad, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _YELLOW, 1,
                    cv2.LINE_AA)  # fmt: skip
        return img

    # --- callback --------------------------------------------------------------------------
    def __call__(self, t: float, rgb: np.ndarray, lm: np.ndarray, mat: np.ndarray) -> bool:
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = t
        h, w = rgb.shape[:2]
        feats = None
        if not np.isnan(lm[0, 0]):
            matrix = None if np.isnan(mat).any() else mat
            feats = V.frame_features(lm, matrix, self.cfg, self.thr, aspect_hw=h / w)
            self.update_blinks(t, feats["ear_mean"])
        else:
            self.update_blinks(t, float("nan"))
        if not self.show:
            return True
        import cv2

        cv2.imshow(self.window, self.render(rgb, lm, feats, t))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        try:  # the user closed the window with the mouse
            return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        if self.show:
            import cv2

            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
