"""Inspect a downloaded UAH-DriveSet and print what 03_data_contract.md section 3.1 needs.

Usage: uv run python scripts/inspect_uah.py data/raw/uah_driveset
Read-only; copies nothing. Paste the output into the "Verified" column notes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from driverisk.data.uah_inspect import inspect_dataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    if not args.root.is_dir():
        print(f"{args.root} is not a folder. Download UAH-DriveSet first (docs/human_tasks).")
        return 2

    trips = inspect_dataset(args.root)
    if not trips:
        print("No folder with a RAW accelerometer file found; layout differs from the paper.")
        return 1
    parsed = [t for t in trips if t.parsed]
    print(f"trips found: {len(trips)}  (folder name parsed: {len(parsed)})")
    for t in trips:
        if not t.parsed:
            print(f"  UNPARSED folder name: {t.folder}")
    print("drivers   :", sorted({t.driver for t in parsed}))
    print("behaviours:", dict(Counter(t.behavior for t in parsed)))
    print("roads     :", dict(Counter(t.road for t in parsed)))
    per = Counter((t.driver, t.behavior, t.road) for t in parsed)
    print("trips per (driver, behaviour, road):")
    for k in sorted(per):
        print(f"  {k}: {per[k]}")

    for key in ("acc", "gps", "lane", "vehicle", "osm", "inertial"):
        files = [t.files[key] for t in trips if key in t.files]
        if not files:
            print(f"{key:<9}: no files")
            continue
        rates = [f.median_rate_hz for f in files if np.isfinite(f.median_rate_hz)]
        cols = Counter(f.n_cols for f in files)
        print(
            f"{key:<9}: {len(files)} files, columns {dict(cols)}, median rate "
            f"{np.median(rates):.1f} Hz (min {min(rates):.1f}, max {max(rates):.1f})"
        )
    total_min = sum(t.files["acc"].duration_s for t in trips if "acc" in t.files) / 60
    print(f"total duration (RAW accelerometer): {total_min:.0f} min")
    print(f"trips with video file: {sum(t.has_video for t in trips)}")
    missing = Counter(m for t in trips for m in t.missing)
    print("missing file kinds (count of trips):", dict(missing) or "none")
    print("\nAcceleration columns 3-5 min/max of the first trip (paper says units are G):")
    first = trips[0].files["acc"]
    print("  min", first.col_min[2:8], "\n  max", first.col_max[2:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
