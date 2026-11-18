"""Agreement between two annotators' interval CSVs (06 section 7, task P2-D2).

CSV columns: trip_id,type,onset_s,offset_s,uncertain  (see docs/human_tasks/templates/).
Example:
  uv run python scripts/annotation_agreement.py a.csv b.csv \
      --manifest data/processed/own/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from driverisk.evaluation.agreement import compare


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotator_a", type=Path)
    p.add_argument("annotator_b", type=Path)
    p.add_argument("--manifest", type=Path, help="manifest.csv with trip_id,duration_s (optional)")
    args = p.parse_args()

    durations = None
    if args.manifest:
        m = pd.read_csv(args.manifest)
        durations = dict(zip(m["trip_id"], m["duration_s"], strict=True))
    results, unc = compare(pd.read_csv(args.annotator_a), pd.read_csv(args.annotator_b), durations)

    print(f"{'type':<22}{'kappa':>7}{'F1':>7}{'mean IoU':>10}{'n_A':>6}{'n_B':>6}{'matched':>9}")
    for r in results:
        if r.n_a + r.n_b == 0:
            continue
        print(
            f"{r.event_type:<22}{r.kappa:>7.2f}{r.f1:>7.2f}{r.mean_iou:>10.2f}"
            f"{r.n_a:>6}{r.n_b:>6}{r.n_matched:>9}"
        )
    print(f"uncertain intervals excluded: A={unc['a']} B={unc['b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
