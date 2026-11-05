"""P0-B2: write 5 synthetic motion-only (UAH-like) stub trips (contract section 9).

Usage: uv run python tests/fixtures/make_stub_motion.py [out_root]
Default out_root: tests/fixtures/generated (git-ignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

from make_stub_vision import DEFAULT_OUT, main

if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT, dataset="uah"))
