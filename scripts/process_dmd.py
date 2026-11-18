"""Convert downloaded DMD sessions into contract trips (face-video features + annotation labels).

    uv run --group vision python scripts/process_dmd.py data/raw/dmd --out data/processed/dmd
    uv run python scripts/process_dmd.py data/raw/dmd --out data/processed/dmd --no-extract

Step 1 (slow, needs the optional `vision` group): MediaPipe over every face video, cached as
`data/interim/dmd/<session>.npz` (about 35 frames/s; a 3-minute session takes about 3 minutes).
Sessions that already have a cache are not run again. Step 2 (fast): features + labels -> parquet,
events, meta and `manifest.csv`, checked by the contract validator.

The outputs are derived from CC BY-NC-ND 4.0 material: they stay in git-ignored `data/` and are
never shared. Exit codes: 0 ok, 1 some session failed, 2 no input.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from driverisk.data import dmd_trips as T
from driverisk.data import schema as S
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.paths import data_dir


def extract_landmarks(session: T.SessionInfo, cache_dir: Path) -> None:
    """Run MediaPipe over one face video and cache the result (needs the `vision` group)."""
    import cv2

    from driverisk.features import face_landmarker as FL

    assert session.video is not None
    cap = cv2.VideoCapture(str(session.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    lms = np.full((n, FL.N_LANDMARKS, 3), np.nan, dtype=np.float32)
    mats = np.full((n, 4, 4), np.nan, dtype=np.float32)
    t0, done = time.time(), n
    print(f"  {session.stem}: {n} frames, {fps:.2f} fps, {size[0]}x{size[1]}", flush=True)
    with FL.FaceLandmarkerAdapter() as adapter:
        for i in range(n):
            ok, bgr = cap.read()
            if not ok:
                done = i
                break
            lms[i], mats[i] = adapter.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i / fps)
            if i and i % 1000 == 0:
                print(f"    frame {i}/{n} ({time.time() - t0:.0f} s)", flush=True)
    cap.release()
    T.save_landmark_cache(T.cache_path(cache_dir, session.stem), lms[:done], mats[:done], fps, size)
    found = float(np.mean(~np.isnan(lms[:done, 0, 0])))
    print(f"    done in {time.time() - t0:.0f} s, face found in {found:.1%} of frames", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="folder with the extracted DMD sessions")
    ap.add_argument("--out", type=Path, default=data_dir() / "processed" / "dmd")
    ap.add_argument("--cache", type=Path, default=data_dir() / "interim" / "dmd")
    ap.add_argument("--no-extract", action="store_true", help="use cached landmarks only")
    ap.add_argument(
        "--frame-shift", type=int, default=0, help="annotation frame + shift = video frame"
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.root.is_dir():
        print(f"{args.root} is not a folder")
        return 2
    sessions = T.find_sessions(args.root)
    if not sessions:
        print("no *_rgb_ann_*.json annotation files found")
        return 2
    todo = [s for s in sessions if not T.cache_path(args.cache, s.stem).is_file()]
    print(f"{len(sessions)} sessions, {len(todo)} without cached landmarks")
    if todo and not args.no_extract:
        for s in todo:
            if s.video is None:
                print(f"  {s.stem}: no face video, cannot extract")
                continue
            extract_landmarks(s, args.cache)

    result = T.process_dataset(
        args.root, args.out, args.cache, frame_shift=args.frame_shift,
        keep_going=args.keep_going, overwrite=args.overwrite,
    )  # fmt: skip
    print(f"sessions converted: {len(result.trips)}  failed: {len(result.failures)} -> {args.out}")
    for stem, err in result.failures:
        print(f"  FAILED {stem}: {err}")
    report = validate_processed_dir(args.out)
    print("validator:", "OK" if report.ok else "FAILED")
    for line in getattr(report, "errors", [])[:10]:
        print("  ", line)
    for trip in result.trips:
        d = trip.meta["dmd"]
        y = trip.frame[S.COL_Y_CLS].to_numpy()
        counts = {S.CLASSES[k]: int(v) for k, v in Counter(y[y >= 0].tolist()).items()}
        ev = Counter(e["type"] for e in trip.events)
        print(
            f"  {trip.manifest_row['trip_id']}  {trip.manifest_row['behavior_tag']:<14} "
            f"{trip.manifest_row['duration_s'] / 60:4.1f} min  face visible "
            f"{trip.frame['v_face_visible'].mean():.0%}  steps/class {counts}  events {dict(ev)}"
            f"  (annotation {d['n_frames_annotation']} frames, video {d['n_frames_video']})"
        )
    return 0 if report.ok and not result.failures else 1


if __name__ == "__main__":
    sys.exit(main())
