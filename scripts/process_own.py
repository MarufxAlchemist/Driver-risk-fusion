"""Process one own recording into the contract format (P1-D1, docs/06 section 8).

    uv run python scripts/process_own.py own_d01_20261201_01 --raw-dir PATH --trip-sheet sheet.csv

Raw files in `--raw-dir` follow fix F11: `<trip_id>_imu.csv`, `<trip_id>_gps.csv` and either
`<trip_id>_vision.parquet` (10 Hz vision features on the video clock, for example from
`scripts/webcam_features.py --video <trip_id>_video.mp4`) or `<trip_id>_frames.npz` (arrays `t`,
`landmarks`, optional `matrices`, `image_size`). The video sync times come from the trip sheet
columns `video_sync_start_s` / `video_sync_end_s` or from `--video-sync`. Output goes to
`data/processed/own/` unless `--out` is given; the folder is validated afterwards.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from driverisk.data import own
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.paths import processed_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trip_id", help="own_<driverid>_<yyyymmdd>_<nn>")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--trip-sheet", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: data/processed/own")
    p.add_argument("--annotations", type=Path, help="event CSV: trip_id,type,onset_s,offset_s,...")
    p.add_argument("--labels", type=Path, help="class CSV: trip_id,class,onset_s,offset_s")
    p.add_argument("--video-sync", nargs="+", type=float, metavar="S",
                   help="video seconds of the first tap at start [and end]")  # fmt: skip
    p.add_argument("--imu-sync", nargs="+", type=float, metavar="S",
                   help="phone seconds of the start [and end] tap; default: detected")  # fmt: skip
    p.add_argument("--allow-missing-end-sync", action="store_true",
                   help="accept an offset-only sync (trip becomes quality low)")  # fmt: skip
    p.add_argument("--time-col")
    p.add_argument("--gps-time-col")
    p.add_argument("--acc-cols", nargs=3)
    p.add_argument("--acc-unit", choices=("m/s^2", "g"), default="m/s^2")
    p.add_argument("--gyro-cols", nargs=3)
    p.add_argument("--gyro-unit", choices=("deg/s", "rad/s"))
    p.add_argument("--speed-col")
    p.add_argument("--speed-unit", choices=("m/s", "km/h"), default="m/s")
    p.add_argument("--heading-col")
    p.add_argument("--ear-threshold", type=float, help="per-driver calibrated EAR threshold")
    p.add_argument("--source-version", help="recording batch, for the manifest")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tid = args.trip_id
    raw = args.raw_dir
    phone = own.load_phone_data(
        raw / own.raw_file_name(tid, "imu", ".csv"),
        raw / own.raw_file_name(tid, "gps", ".csv"),
        time_col=args.time_col, gps_time_col=args.gps_time_col, acc_cols=args.acc_cols,
        acc_unit=args.acc_unit, gyro_cols=args.gyro_cols, gyro_unit=args.gyro_unit,
        speed_col=args.speed_col, speed_unit=args.speed_unit, heading_col=args.heading_col,
    )  # fmt: skip
    features = raw / own.raw_file_name(tid, "vision", ".parquet")
    frames = raw / own.raw_file_name(tid, "frames", ".npz")
    if features.is_file():
        source = "driver_calibrated" if args.ear_threshold is not None else None
        vision = own.load_vision_features(
            features, ear_threshold=args.ear_threshold, ear_threshold_source=source
        )
    elif frames.is_file():
        vision = own.load_vision_frames(frames)
        vision.ear_threshold = args.ear_threshold
    else:
        print(f"no vision input: expected {features.name} or {frames.name} in {raw}")
        return 2
    sheet = own.read_trip_sheet(args.trip_sheet, tid)
    trip = own.process_own_trip(
        tid, phone, vision, sheet,
        video_sync_s=args.video_sync, imu_sync_s=args.imu_sync,
        annotations=pd.read_csv(args.annotations) if args.annotations else None,
        labels=pd.read_csv(args.labels) if args.labels else None,
        require_end_sync=not args.allow_missing_end_sync, source_version=args.source_version,
    )  # fmt: skip
    out = args.out or processed_dir("own")
    own.write_own_trip(trip, out)

    sync = trip.meta["sync"]
    print(f"{tid}: {len(trip.frame)} steps ({trip.manifest_row['duration_s']:.1f} s)")
    print(
        f"sync: offset {sync['offset_start_s']:.3f} s, drift {sync['drift_s']} s, "
        f"corrected {sync['drift_corrected']}, residual {sync['residual_s']} s"
    )
    print(f"events: {len(trip.events)}, quality: {trip.meta['quality']}")
    for name, check in trip.meta["quality_checks"].items():
        print(f"  {name:14s} {'pass' if check['pass'] else 'FAIL'}  ({check['value']})")
    report = validate_processed_dir(out)
    for err in report.errors:
        print(f"INVALID: {err}")
    print("VALIDATION:", "OK" if report.ok else "FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
