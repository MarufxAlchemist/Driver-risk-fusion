"""Make the frozen driver-disjoint split file `splits.json` from one or more manifests (P1-C2).

Usage:
  uv run python scripts/make_splits.py data/processed/uah/manifest.csv \
      data/processed/dmd/manifest.csv --out data/processed/splits.json [--write-manifests]
  uv run python scripts/make_splits.py <manifests...> --out <splits.json> --verify

The file holds the seed, fractions, the split of every driver key `<dataset>:<driver_id>`,
leave-one-driver-out folds with nested validation, the personalization split and the SHA-256
of every input manifest (without its `split` column). Same manifests + same seed -> the same
bytes. `--write-manifests` writes the assignment into each manifest's `split` column and
changes nothing else. `--verify` checks manifests against an existing file and writes nothing.

Exit codes: 0 ok, 1 verification failed, 2 inputs refused (nothing written).
Settings: `configs/splits.yaml` (or `--config`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from driverisk.data import splits as SP
from driverisk.utils.config import load_config
from driverisk.utils.paths import data_dir


def _input_name(path: Path) -> str:
    """`<folder>/<file>`, never an absolute path (the repo is public)."""
    return f"{path.parent.name}/{path.name}"


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_manifest(path: Path, frame: pd.DataFrame) -> None:
    """Rewrite with the line endings the file already had."""
    newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    frame.to_csv(path, index=False, lineterminator=newline)


def _summary(record: dict) -> list[str]:
    lines = []
    per = Counter((v["dataset"], v["split"]) for v in record["drivers"].values())
    for dataset in sorted({d for d, _ in per}):
        counts = {s: per[(dataset, s)] for s in ("train", "val", "test")}
        lines.append(f"{dataset}: drivers per split {counts}")
    for dataset, lodo in record["lodo"].items():
        lines.append(f"{dataset}: {len(lodo['folds'])} leave-one-driver-out folds")
    status = Counter(v["status"] for v in record["personalization"]["drivers"].values())
    lines.append(f"personalization: {dict(status)}")
    lines.extend(f"WARNING: {w}" for w in record["warnings"])
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("manifests", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: data/processed/splits.json")
    p.add_argument("--config", default="splits", help="config name or path (configs/splits.yaml)")
    p.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    p.add_argument("--write-manifests", action="store_true", help="fill each manifest's split")
    p.add_argument("--verify", action="store_true", help="check manifests against --out")
    args = p.parse_args(argv)
    out = args.out or data_dir() / "processed" / "splits.json"

    missing = [str(m) for m in args.manifests if not m.is_file()]
    if missing:
        print(f"refused: manifest(s) not found: {missing}")
        return 2
    names = [_input_name(m) for m in args.manifests]
    if len(set(names)) != len(names):
        print(f"refused: two inputs have the same <folder>/<file> name: {names}")
        return 2
    frames = {n: _read_manifest(m) for n, m in zip(names, args.manifests, strict=True)}

    if args.verify:
        if not out.is_file():
            print(f"{out} not found")
            return 1
        errors = SP.verify_split_record(json.loads(out.read_text(encoding="utf-8")), frames)
        for e in errors:
            print(f"MISMATCH: {e}")
        print("verify:", "ok" if not errors else f"{len(errors)} problem(s)")
        return 1 if errors else 0

    cfg = load_config(args.config)
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    try:
        record = SP.build_split_record(
            frames,
            seed=seed,
            fractions=cfg["fractions"],
            fractions_per_dataset=cfg.get("fractions_per_dataset") or {},
            lodo_datasets=cfg.get("lodo_datasets") or [],
            personalization=cfg.get("personalization") or {},
        )
    except ValueError as exc:  # SplitInputError is a ValueError
        print(f"refused: {exc}")
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(SP.dump_split_record(record).encode("utf-8"))
    print(f"wrote {out} (seed {seed}, {len(record['drivers'])} drivers)")
    for line in _summary(record):
        print(" ", line)

    if args.write_manifests:
        for name, path in zip(names, args.manifests, strict=True):
            before = frames[name]
            after = SP.apply_split_record(before, record)
            changed = int((after["split"] != before.get("split", "")).sum())
            _write_manifest(path, after)
            print(f"  rewrote split column of {name}: {changed} row(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
