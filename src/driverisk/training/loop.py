"""Minimal supervised training loop (P0). P2-C3 adds checkpointing, early stopping, schedules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from driverisk.data import schema as S
from driverisk.models.fusion import CrossAttentionFusion, forward_batch

log = logging.getLogger(__name__)


@dataclass
class History:
    epoch_loss: list[float] = field(default_factory=list)
    initial_loss: float = float("nan")


def mean_loss(model: CrossAttentionFusion, loader: DataLoader, device: str = "cpu") -> float:
    """Mean cross-entropy over labelled windows, model in eval mode."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=S.UNLABELED, reduction="sum")
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            y = batch["y_cls"].to(device)
            total += float(loss_fn(forward_batch(model, batch, device)["logits"], y))
            n += int((y != S.UNLABELED).sum())
    return total / max(n, 1)


def train_classifier(
    model: CrossAttentionFusion,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str = "cpu",
) -> History:
    """Train with AdamW on window labels; unlabelled windows (-1) are ignored."""
    model.to(device)
    hist = History(initial_loss=mean_loss(model, loader, device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=S.UNLABELED)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            y = batch["y_cls"].to(device)
            if (y == S.UNLABELED).all():
                continue
            loss = loss_fn(forward_batch(model, batch, device)["logits"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        hist.epoch_loss.append(float(np.mean(losses)) if losses else float("nan"))
        log.info("epoch %d loss %.4f", epoch, hist.epoch_loss[-1])
    return hist


def predict(
    model: CrossAttentionFusion, loader: DataLoader, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Class probabilities and labels for every window in `loader`."""
    model.eval()
    probs, labels, gates = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = forward_batch(model, batch, device)
            probs.append(torch.softmax(out["logits"], -1).cpu().numpy())
            gates.append(out["gates"].cpu().numpy())
            labels.append(batch["y_cls"].numpy())
    return {
        "probs": np.concatenate(probs),
        "y_cls": np.concatenate(labels),
        "gates": np.concatenate(gates),
    }
