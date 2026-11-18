"""Degradation helpers for the robustness sweep (docs/04 section 3.4 and fix F24)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from driverisk.data import schema as S
from driverisk.evaluation import robustness as R
from driverisk.models.fusion import CrossAttentionFusion, forward_batch
from driverisk.utils.seed import set_seed

B, T = 6, S.WINDOW_STEPS


def _batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "x_v": torch.randn(B, T, S.DV, generator=g),
        "x_m": torch.randn(B, T, S.DM, generator=g),
        "nan_v": torch.zeros(B, T, S.DV, dtype=torch.bool),
        "nan_m": torch.zeros(B, T, S.DM, dtype=torch.bool),
        "mask_v": torch.ones(B, T, dtype=torch.bool),
        "mask_m": torch.ones(B, T, dtype=torch.bool),
        "y_cls": torch.tensor([0, 1, 2, 3, 1, 2]),
        "trip_id": ["t"] * B,
    }


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_fraction_zero_is_identity_and_input_is_never_mutated() -> None:
    b = _batch()
    before = {k: v.clone() for k, v in b.items() if torch.is_tensor(v)}
    out = R.mask_modality(b, "camera", 0.0, _gen())
    for k, v in before.items():
        assert torch.equal(b[k], v)  # input untouched
        assert torch.equal(out[k], v)
    assert out["x_v"] is not b["x_v"]
    assert out["trip_id"] == b["trip_id"]  # non-tensor entries pass through


def test_full_loss_masks_zeroes_and_flags_only_that_modality() -> None:
    b = _batch()
    out = R.mask_modality(b, "camera", 1.0, _gen())
    assert not out["mask_v"].any() and (out["x_v"] == 0).all() and out["nan_v"].all()
    assert torch.equal(out["x_m"], b["x_m"]) and out["mask_m"].all() and not out["nan_m"].any()
    out = R.mask_modality(b, "motion", 1.0, _gen())
    assert not out["mask_m"].any() and out["mask_v"].all()


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_exact_number_of_steps_is_masked_per_sample(fraction: float) -> None:
    out = R.mask_modality(_batch(), "camera", fraction, _gen())
    expected = round(fraction * T)
    assert (~out["mask_v"]).sum(dim=1).tolist() == [expected] * B
    # masked steps are zeroed and flagged, available steps are untouched
    masked = ~out["mask_v"]
    assert (out["x_v"][masked] == 0).all() and out["nan_v"][masked].all()
    assert torch.equal(out["x_v"][~masked], _batch()["x_v"][~masked])


def test_masking_is_reproducible_by_seed() -> None:
    a = R.mask_modality(_batch(), "camera", 0.5, _gen(1))["mask_v"]
    b = R.mask_modality(_batch(), "camera", 0.5, _gen(1))["mask_v"]
    c = R.mask_modality(_batch(), "camera", 0.5, _gen(2))["mask_v"]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_contiguous_mode_masks_one_run_per_sample() -> None:
    out = R.mask_modality(_batch(), "camera", 0.25, _gen(), contiguous=True)
    for row in (~out["mask_v"]).numpy():
        idx = np.flatnonzero(row)
        assert len(idx) == round(0.25 * T) and idx[-1] - idx[0] + 1 == len(idx)


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        R.mask_modality(_batch(), "radar", 0.5, _gen())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        R.mask_modality(_batch(), "camera", 1.5, _gen())


def test_fusion_model_ignores_whatever_was_under_the_mask() -> None:
    set_seed(0)
    model = CrossAttentionFusion(d_model=32, n_heads=4).eval()
    out = R.mask_modality(_batch(), "camera", 1.0, _gen())
    junk = {**out, "x_v": torch.randn_like(out["x_v"]) * 50}
    a = forward_batch(model, out)["logits"]
    b = forward_batch(model, junk)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_gaussian_noise_hits_requested_snr_and_spares_masked_steps() -> None:
    b = _batch()
    b["x_m"] = torch.zeros(B, T, S.DM)
    b["mask_m"][:, :10] = False
    out = R.add_noise(b, "motion", snr_db=20.0, generator=_gen(3))
    avail = out["mask_m"]
    # features are normalized to unit variance, so 20 dB means noise std 0.1
    assert out["x_m"][avail].std().item() == pytest.approx(0.1, rel=0.1)
    assert (out["x_m"][~avail] == 0).all()
    assert torch.equal(out["x_v"], b["x_v"]) and torch.equal(out["mask_m"], b["mask_m"])


def test_time_shift_delays_one_stream_and_masks_the_gap() -> None:
    b = _batch()
    out = R.shift_modality(b, "motion", steps=3)  # motion arrives 3 steps (0.3 s) late
    assert torch.equal(out["x_m"][:, 3:], b["x_m"][:, :-3])
    assert not out["mask_m"][:, :3].any() and out["mask_m"][:, 3:].all()
    assert (out["x_m"][:, :3] == 0).all() and out["nan_m"][:, :3].all()
    assert torch.equal(out["x_v"], b["x_v"])
    early = R.shift_modality(b, "camera", steps=-2)
    assert torch.equal(early["x_v"][:, :-2], b["x_v"][:, 2:])
    assert not early["mask_v"][:, -2:].any()
    assert torch.equal(R.shift_modality(b, "camera", 0)["x_v"], b["x_v"])
    assert not R.shift_modality(b, "camera", T + 5)["mask_v"].any()


def test_robustness_curve_runs_every_fraction_with_a_fixed_seed() -> None:
    batches = [_batch(0), _batch(1)]

    def predict(batch: dict) -> np.ndarray:
        # a 'model' that is right whenever the camera is on for the whole window, else says 0
        truth = batch["y_cls"].numpy()
        ok = batch["mask_v"].all(dim=1).numpy()
        return np.where(ok, truth, 0)

    curve = R.robustness_curve(predict, batches, "camera", [0.0, 0.1, 1.0], seed=0)
    assert list(curve) == [0.0, 0.1, 1.0]
    assert curve[0.0] == pytest.approx(1.0)
    assert curve[1.0] < curve[0.0]
    again = R.robustness_curve(predict, batches, "camera", [0.0, 0.1, 1.0], seed=0)
    assert curve == again
