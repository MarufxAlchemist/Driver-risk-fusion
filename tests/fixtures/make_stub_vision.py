"""P0-A2: write 5 synthetic vision-only (DMD-like) stub trips (contract section 9).

Usage: uv run python tests/fixtures/make_stub_vision.py [out_root]
Default out_root: tests/fixtures/generated (git-ignored).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from driverisk.data.stubs import make_stub_dataset
from driverisk.data.validate import validate_processed_dir
from driverisk.utils.config import load_config

DEFAULT_OUT = Path(__file__).parent / "generated"


def main(out_root: Path, dataset: str = "dmd") -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    (out_dir,) = make_stub_dataset(out_root, load_config("stub"), datasets=(dataset,))
    report = validate_processed_dir(out_dir)
    for err in report.errors:
        logging.error(err)
    status = "OK" if report.ok else "FAILED"
    logging.info("%s: %d trips, validator %s", out_dir, report.n_trips, status)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT))
