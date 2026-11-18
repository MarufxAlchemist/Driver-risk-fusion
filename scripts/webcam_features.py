"""Run the vision feature extractor on the webcam or a video file and sanity-check the output.

    uv sync --group vision
    uv run python scripts/webcam_features.py --seconds 60            # webcam 0
    uv run python scripts/webcam_features.py --video clip.mp4

A live preview window shows you (mirrored) with the eye/iris/mouth points and the values the
extractor reads; press q or Esc to stop early. `--no-preview` runs without a window.

Writes `<out>/vision_features.parquet` (git-ignored) and `<out>/vision_features.png`, and prints
a short summary. Nothing is recorded: frames are processed in memory and discarded.

Suggested check (P1-A1 "done when"): look straight ahead, then turn your head right, then look up,
then blink a few times, then close your eyes for 2 s. The printout and plot should show positive
head yaw for "right", positive pitch for "up", a few blinks and one long closure. If a sign is
wrong, tell Claude: the convention lives in features/vision.py (assumption A25).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from driverisk.data import schema as S
from driverisk.features import face_landmarker as FL
from driverisk.features import vision as V
from driverisk.utils.paths import repo_root

log = logging.getLogger("webcam_features")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", help="video file; default is the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=repo_root() / "demo_out" / "vision")
    ap.add_argument("--ear-threshold", type=float, default=None)
    ap.add_argument("--no-preview", action="store_true", help="run without the live window")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = args.video if args.video else args.camera
    preview = None
    if not args.no_preview:
        from driverisk.features.preview import LivePreview

        preview = LivePreview(ear_threshold=args.ear_threshold)
        print("Preview open: turn to your right, look up, blink, close your eyes 2 s. q/Esc stops.")
    try:
        t, lm, mats, size = FL.run_on_frames(
            FL.video_frames(source, args.seconds), on_frame=preview
        )
    finally:
        if preview is not None:
            preview.close()
    t = t - t[0]
    res = V.extract_vision_features(t, lm, mats, ear_threshold=args.ear_threshold, image_size=size)
    args.out.mkdir(parents=True, exist_ok=True)
    res.frame.to_parquet(args.out / "vision_features.parquet", index=False)

    df, p = res.frame, S.VISION_PREFIX
    fps = len(t) / max(t[-1], 1e-9)
    print(
        f"frames {len(t)} over {t[-1]:.1f} s ({fps:.1f} fps), {size[0]}x{size[1]}, steps {len(df)}"
    )
    visible, usable = df[p + "face_visible"].mean(), df[S.COL_MASK_V].mean()
    print(f"face visible {visible:.0%}, usable (mask_v) {usable:.0%}")
    for name in (
        "ear_mean",
        "mar",
        "head_yaw",
        "head_pitch",
        "head_roll",
        "gaze_yaw",
        "gaze_pitch",
    ):
        x = df[p + name].to_numpy(np.float64)
        if np.isfinite(x).any():
            lo, mid, hi = np.nanmin(x), np.nanmedian(x), np.nanmax(x)
            print(f"  {name:11s} min {lo:7.2f}  median {mid:7.2f}  max {hi:7.2f}")
    blinks, closed = int(np.nansum(df[p + "blink"])), int(np.nansum(df[p + "eye_closed"]))
    print(f"  blinks {blinks}, closed steps {closed}")
    print(f"  EAR threshold {res.meta['ear_threshold']} ({res.meta['ear_threshold_source']})")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        tt = df[S.COL_T]
        ax[0].plot(tt, df[p + "ear_mean"], label="EAR")
        ax[0].axhline(res.meta["ear_threshold"], ls="--", c="grey", label="threshold")
        ax[0].plot(tt, df[p + "eye_closed"] * 0.05, label="eye closed (x0.05)")
        ax[0].set_ylabel("ratio")
        ax[0].legend(loc="upper right")
        for name in ("head_yaw", "head_pitch", "head_roll"):
            ax[1].plot(tt, df[p + name], label=name)
        ax[1].set_ylabel("deg")
        ax[1].legend(loc="upper right")
        ax[2].plot(tt, df[p + "perclos_30s"], label="PERCLOS 30 s")
        ax[2].plot(tt, df[p + "face_conf"], label="face_conf")
        ax[2].set_xlabel("time (s)")
        ax[2].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.out / "vision_features.png", dpi=110)
        print(f"plot: {args.out / 'vision_features.png'}")
    except ImportError:
        log.info("matplotlib not installed; skipped the plot")


if __name__ == "__main__":
    main()
