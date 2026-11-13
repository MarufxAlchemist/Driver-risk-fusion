"""Training augmentation (docs/04 section 6): jitter, scale, rotation and face drop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from driverisk.data import schema as S
from driverisk.data.augment import AugmentConfig, Augmenter
from driverisk.data.datasets import WindowDataset
from driverisk.data.normalize import NormStats

T = 2000
MI = {n: i for i, n in enumerate(S.MOTION_NAMES)}
VI = {n: i for i, n in enumerate(S.VISION_NAMES)}


def _norm() -> NormStats:
    return NormStats(
        np.full(S.DV, 0.5, np.float32),
        np.full(S.DV, 2.0, np.float32),
        np.full(S.DM, 1.0, np.float32),
        np.full(S.DM, 3.0, np.float32),
        "test",
    )


def _sample(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_v, x_m = torch.randn(T, S.DV, generator=g), torch.randn(T, S.DM, generator=g)
    nan_v, nan_m = torch.zeros(T, S.DV, dtype=torch.bool), torch.zeros(T, S.DM, dtype=torch.bool)
    nan_m[:, MI["lane_offset"]] = True  # a declared-NaN column
    x_m[nan_m] = 0.0
    return {
        "x_v": x_v, "x_m": x_m, "nan_v": nan_v, "nan_m": nan_m,
        "mask_v": torch.ones(T, dtype=torch.bool), "mask_m": torch.ones(T, dtype=torch.bool),
        "y_cls": torch.tensor(2), "y_risk": torch.tensor([0.0, 1.0, -1.0]),
        "driver_id": "own:d01", "trip_id": "own_d01_20261101_01", "t_start": 3.0,
    }  # fmt: skip


def _only(**kw) -> AugmentConfig:
    base = {
        "motion_jitter_sigma": 0.0, "motion_scale_range": (1.0, 1.0), "motion_rotation_deg": 0.0,
        "vision_jitter_sigma": 0.0, "face_drop_prob": 0.0,
    }  # fmt: skip
    return AugmentConfig(**{**base, **kw})


def _raw(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return x * torch.from_numpy(std) + torch.from_numpy(mean)


def test_all_zero_config_is_the_identity() -> None:
    s = _sample()
    out = Augmenter(_norm(), _only(), seed=0)(s)
    for k in ("x_v", "x_m", "nan_v", "nan_m", "mask_v", "mask_m"):
        assert torch.equal(out[k], s[k]), k


def test_labels_ids_and_shapes_are_never_touched() -> None:
    s = _sample()
    out = Augmenter(_norm(), AugmentConfig(), seed=0)(s)
    assert out["y_cls"] == s["y_cls"] and torch.equal(out["y_risk"], s["y_risk"])
    assert (out["driver_id"], out["trip_id"], out["t_start"]) == (
        s["driver_id"], s["trip_id"], s["t_start"],
    )  # fmt: skip
    assert out["x_v"].shape == s["x_v"].shape and out["x_m"].dtype == torch.float32
    assert out is not s and not torch.equal(out["x_m"], s["x_m"])  # input dict is not modified


def test_input_sample_is_not_mutated() -> None:
    s = _sample()
    before = {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
    Augmenter(_norm(), AugmentConfig(), seed=3)(s)
    for k, v in before.items():
        assert torch.equal(s[k], v), k


def test_same_seed_same_result_other_seed_different() -> None:
    a = Augmenter(_norm(), AugmentConfig(), seed=1)(_sample())["x_m"]
    b = Augmenter(_norm(), AugmentConfig(), seed=1)(_sample())["x_m"]
    c = Augmenter(_norm(), AugmentConfig(), seed=2)(_sample())["x_m"]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_jitter_hits_sigma_and_spares_flags_nan_and_masked_steps() -> None:
    s = _sample()
    s["mask_m"][:500] = False
    out = Augmenter(_norm(), _only(motion_jitter_sigma=0.1), seed=0)(s)
    diff = out["x_m"] - s["x_m"]
    cont = [i for i, f in enumerate(S.MOTION_FEATURES) if f.kind == "continuous"]
    cont = [i for i in cont if i != MI["lane_offset"]]
    assert diff[500:, cont].std().item() == pytest.approx(0.1, rel=0.05)
    assert (diff[:500] == 0).all()  # masked steps
    assert (diff[:, MI["gps_valid"]] == 0).all()  # flag
    assert (diff[:, MI["lane_offset"]] == 0).all()  # NaN-flagged column stays 0


def test_vision_jitter_leaves_flags_and_long_term_context_alone() -> None:
    s = _sample()
    out = Augmenter(_norm(), _only(vision_jitter_sigma=0.1), seed=0)(s)
    diff = out["x_v"] - s["x_v"]
    for name in ("blink", "eye_closed", "off_road_gaze", "face_visible", "face_conf"):
        assert (diff[:, VI[name]] == 0).all(), name
    for name in ("perclos_30s", "perclos_60s"):
        assert (diff[:, VI[name]] == 0).all(), name
    assert diff[:, VI["ear_mean"]].std().item() == pytest.approx(0.1, rel=0.1)


def test_scale_applies_one_factor_per_sample_to_acceleration_features_only() -> None:
    norm = _norm()
    s = _sample()
    out = Augmenter(norm, _only(motion_scale_range=(1.2, 1.2)), seed=0)(s)
    raw_in, raw_out = (
        _raw(s["x_m"], norm.mean_m, norm.std_m),
        _raw(out["x_m"], norm.mean_m, norm.std_m),
    )
    for name in ("acc_long", "acc_lat", "acc_vert", "jerk_long", "jerk_lat"):
        assert torch.allclose(raw_out[:, MI[name]], raw_in[:, MI[name]] * 1.2, atol=1e-4), name
    for name in ("speed", "yaw_rate", "roll", "gps_valid"):
        assert torch.equal(out["x_m"][:, MI[name]], s["x_m"][:, MI[name]]), name


def test_rotation_keeps_the_horizontal_acceleration_magnitude() -> None:
    norm = _norm()
    s = _sample()
    out = Augmenter(norm, _only(motion_rotation_deg=10.0), seed=5)(s)
    a, b = _raw(s["x_m"], norm.mean_m, norm.std_m), _raw(out["x_m"], norm.mean_m, norm.std_m)
    for lon, lat in (("acc_long", "acc_lat"), ("jerk_long", "jerk_lat")):
        before = torch.hypot(a[:, MI[lon]], a[:, MI[lat]])
        after = torch.hypot(b[:, MI[lon]], b[:, MI[lat]])
        assert torch.allclose(before, after, atol=1e-3)
    angle = torch.atan2(b[:, MI["acc_lat"]], b[:, MI["acc_long"]]) - torch.atan2(
        a[:, MI["acc_lat"]], a[:, MI["acc_long"]]
    )
    angle = torch.remainder(angle + np.pi, 2 * np.pi) - np.pi
    assert angle.abs().max().item() <= np.radians(10.0) + 1e-3
    assert angle.abs().max().item() > np.radians(0.5)  # it did rotate
    assert torch.equal(out["x_m"][:, MI["speed"]], s["x_m"][:, MI["speed"]])


def test_face_drop_masks_the_requested_share_of_steps() -> None:
    s = _sample()
    out = Augmenter(_norm(), _only(face_drop_prob=0.2), seed=0)(s)
    dropped = ~out["mask_v"]
    assert dropped.float().mean().item() == pytest.approx(0.2, abs=0.03)
    assert (out["x_v"][dropped] == 0).all() and out["nan_v"][dropped].all()
    assert torch.equal(out["x_v"][~dropped], s["x_v"][~dropped])
    assert torch.equal(out["mask_m"], s["mask_m"])


def test_bad_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        AugmentConfig(face_drop_prob=1.5)
    with pytest.raises(ValueError):
        AugmentConfig(motion_scale_range=(1.2, 0.8))
    with pytest.raises(ValueError):
        AugmentConfig(motion_jitter_sigma=-0.1)


def test_window_dataset_applies_the_transform_and_defaults_to_none(stub_dirs: list[Path]) -> None:
    plain = WindowDataset(stub_dirs, "train")
    aug = WindowDataset(
        stub_dirs, "train", transform=Augmenter(plain.norm, AugmentConfig(), seed=0)
    )
    a, b = plain[0], aug[0]
    assert a["trip_id"] == b["trip_id"] and torch.equal(a["y_cls"], b["y_cls"])
    assert not torch.equal(a["x_m"], b["x_m"]) or not torch.equal(a["x_v"], b["x_v"])
    assert torch.equal(plain[0]["x_m"], a["x_m"])  # the plain dataset stays deterministic
