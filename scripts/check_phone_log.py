"""Check a phone sensor-log CSV (P0-D5): timestamps, rate, jitter, gaps, sync taps.

Example:
  uv run python scripts/check_phone_log.py path/to/Accelerometer.csv --acc-cols x y z
The CSV is read from where you saved it; nothing is uploaded or copied into the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from driverisk.data.phone_log import analyze_log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--time-col", default=None, help="timestamp column (guessed if omitted)")
    p.add_argument("--acc-cols", nargs=3, metavar=("X", "Y", "Z"), help="accelerometer columns")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{args.csv.name}: {len(df)} rows, columns = {list(df.columns)}")
    r = analyze_log(df, args.time_col, tuple(args.acc_cols) if args.acc_cols else None)
    print(f"time column      : {r.time_column} ({r.time_unit})")
    print(f"duration         : {r.duration_s:.1f} s")
    print(f"median rate      : {r.median_rate_hz:.1f} Hz   jitter {r.jitter:.3f}")
    print(f"gaps > 0.5 s     : {r.n_gaps}   (longest step {r.max_gap_s:.2f} s)")
    print(f"duplicate times  : {r.n_duplicate_times}   missing values {r.nan_fraction:.2%}")
    if args.acc_cols:
        print(f"sync tap pairs at: {[round(x, 2) for x in r.taps]} s")
    for prob in r.problems:
        print(f"PROBLEM: {prob}")
    print("RESULT:", "OK" if r.ok else "NOT OK")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
