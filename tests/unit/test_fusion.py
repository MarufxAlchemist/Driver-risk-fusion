"""Toy cross-attention fusion (P0-C4): shapes, masking contract, modality dropout, learning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from driverisk.data import schema as S
from driverisk.data.datasets import WindowDataset
from driverisk.models.fusion import CrossAttentionFusion, forward_batch
from driverisk.training.loop import mean_loss, train_classifier
from driverisk.utils.config import load_config
from driverisk.utils.seed import make_generator, set_seed

B, T = 4, S.WINDOW_STEPS


def _inputs(mask_v: torch.Tensor | None = None, mask_m: torch.Tensor | None = None) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "x_v": torch.randn(B, T, S.DV, generator=g),
        "x_m": torch.randn(B, T, S.DM, generator=g),
        "nan_v": torch.zeros(B, T, S.DV, dtype=torch.bool),
        "nan_m": torch.zeros(B, T, S.DM, dtype=torch.bool),
        "mask_v": torch.ones(B, T, dtype=torch.bool) if mask_v is None else mask_v,
        "mask_m": torch.ones(B, T, dtype=torch.bool) if mask_m is None else mask_m,
    }


@pytest.fixture
def model() -> CrossAttentionFusion:
    set_seed(0)
    return CrossAttentionFusion(d_model=32, n_heads=4).eval()


def test_output_shapes(model: CrossAttentionFusion) -> None:
    out = forward_batch(model, _inputs())
    assert out["logits"].shape == (B, S.K_CLASSES)
    assert out["z"].shape == (B, 32)
    assert out["gates"].shape == (B, 2)


def test_masked_values_do_not_change_output(model: CrossAttentionFusion) -> None:
    mask_v = torch.ones(B, T, dtype=torch.bool)
    mask_v[:, 10:30] = False
    a = _inputs(mask_v=mask_v)
    b = {k: v.clone() for k, v in a.items()}
    b["x_v"][:, 10:30] = 1e3  # garbage where the camera is unavailable
    torch.testing.assert_close(forward_batch(model, a)["logits"], forward_batch(model, b)["logits"])


def test_fully_missing_stream_gets_zero_gate_and_no_nan(model: CrossAttentionFusion) -> None:
    inp = _inputs(mask_v=torch.zeros(B, T, dtype=torch.bool))
    out = forward_batch(model, inp)
    assert torch.isfinite(out["logits"]).all()
    assert (out["gates"][:, 0] == 0).all() and (out["gates"][:, 1] > 0).all()


def test_both_streams_missing_is_finite(model: CrossAttentionFusion) -> None:
    none = torch.zeros(B, T, dtype=torch.bool)
    out = forward_batch(model, _inputs(mask_v=none, mask_m=none))
    assert torch.isfinite(out["logits"]).all()


def test_modality_dropout_never_drops_both_or_the_only_stream() -> None:
    m = CrossAttentionFusion(d_model=16, n_heads=2, modality_dropout_p=1.0).train()
    full = torch.ones(256, T, dtype=torch.bool)
    v, mo = m._modality_dropout(full, full)
    assert (v.any(1) | mo.any(1)).all()
    v, mo = m._modality_dropout(full, torch.zeros_like(full))  # vision-only samples
    assert v.all()
    assert m.eval()._modality_dropout(full, full)[0].all()  # no dropout at eval time


def test_loss_drops_on_stub_toy_rule(stub_dirs: list[Path]) -> None:
    cfg = load_config("toy_fusion")
    set_seed(cfg["seed"])
    ds = WindowDataset(stub_dirs, "train")
    ds.norm = ds.fit_norm_stats()
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=make_generator(0)
    )
    model = CrossAttentionFusion(**cfg["model"])
    hist = train_classifier(
        model, loader, epochs=3, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    final = mean_loss(model, loader)
    assert final < 0.7 * hist.initial_loss, (hist.initial_loss, hist.epoch_loss, final)
