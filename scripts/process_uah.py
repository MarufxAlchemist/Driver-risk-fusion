"""Convert a downloaded UAH-DriveSet into contract trips, validate them, print a summary.

Usage: uv run python scripts/process_uah.py data/raw/uah_driveset [--out data/processed/uah]

Run scripts/inspect_uah.py first: the file format this loader expects (configs/uah.yaml) comes
from the paper and is UNVERIFIED. Any mismatch stops with an error naming the file; nothing is
repaired. The per-behaviour summary is the input of the P1-B3 check (aggressive trips should
have more inertial events than normal ones).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from driverisk.data.uah import (
    UAHFormatError,
    load_label_map,
    load_uah_config,
    process_dataset,
    summarize_by_behaviour,
)
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.paths import processed_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("root", type=Path, help="UAH-DriveSet download folder")
    p.add_argument("--out", type=Path, default=None, help="output (default data/processed/uah)")
    p.add_argument("--config", type=Path, default=None, help="default configs/uah.yaml")
    p.add_argument("--label-map", type=Path, default=None, help="default label_map_uah.yaml")
    p.add_argument("--keep-going", action="store_true", help="skip failing routes, report them")
    p.add_argument("--overwrite", action="store_true", help="replace existing UAH outputs")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.root.is_dir():
        print(f"{args.root} is not a folder. Download UAH-DriveSet first (docs/human_tasks).")
        return 2
    out = args.out or processed_dir("uah")
    print(
        "NOTE: configs/uah.yaml was checked on the real download on 2026-11-18; see the "
        "UNVERIFIED list in each trip's meta for what is still open."
    )
    try:
        result = process_dataset(
            args.root, out, load_uah_config(args.config), load_label_map(args.label_map),
            keep_going=args.keep_going, overwrite=args.overwrite,
        )  # fmt: skip
    except (UAHFormatError, FileExistsError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"routes converted: {len(result.trips)}  failed: {len(result.failures)}  -> {out}")
    for folder, err in result.failures:
        print(f"  FAILED {folder.name}: {err}")

    report = validate_processed_dir(out)
    if report.ok:
        print(f"validator: OK ({report.n_trips} trips)")
    else:
        print(f"validator: {len(report.errors)} errors")
        for e in report.errors[:50]:
            print("  ", e)

    print("\nper behaviour (events = UAH inertial events kept, logged only above 50 km/h):")
    with pd.option_context("display.width", 120, "display.float_format", "{:.1f}".format):
        print(summarize_by_behaviour(out).to_string())
    return 0 if report.ok and not result.failures else 1


if __name__ == "__main__":
    sys.exit(main())
