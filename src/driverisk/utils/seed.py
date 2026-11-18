"""Seeding helpers so every run is reproducible (CONTRIBUTING section 5)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Args:
        seed: non-negative integer seed.
        deterministic: also ask PyTorch for deterministic kernels where available.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code still uses the global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """`DataLoader(worker_init_fn=seed_worker)`: derive each worker's seed from torch's."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)  # noqa: NPY002 - global RNG per worker
    random.seed(worker_seed + worker_id)


def make_generator(seed: int) -> torch.Generator:
    """Generator for `DataLoader(generator=...)` so shuffling order is seeded."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
