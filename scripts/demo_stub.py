"""`make demo-stub`: stub data -> validator -> windows -> toy fusion -> mock API (P0 gate).

Everything here is synthetic. The numbers prove the plumbing works, not that the model works.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader

from driverisk.data import schema as S
from driverisk.data.datasets import WindowDataset
from driverisk.data.stubs import make_stub_dataset
from driverisk.data.validate import validate_processed_dir
from driverisk.models.fusion import CrossAttentionFusion
from driverisk.serving import schemas as M
from driverisk.serving.mock_app import create_app
from driverisk.training.loop import mean_loss, predict, train_classifier
from driverisk.utils.config import load_config
from driverisk.utils.paths import repo_root
from driverisk.utils.seed import make_generator, set_seed

log = logging.getLogger("demo_stub")
API_KEY = "demo-key"

# ADR-0007 part 2: a class decision is a state alert and uses the class name.
CLASS_TO_ALERT: dict[int, M.AlertType] = {1: "aggressive", 2: "drowsy", 3: "distracted"}


def _alert_from_window(prob: np.ndarray, gates: np.ndarray, t: float, trip_id: str, driver: str):
    cls = int(np.argmax(prob[1:])) + 1  # best non-normal class
    camera, motion = float(gates[0]), float(gates[1])
    total = (camera + motion) or 1.0
    used: list[M.Modality] = [m for m, g in (("camera", camera), ("motion", motion)) if g > 0]
    return M.Alert(
        alert_id="alt_demo",
        trip_id=trip_id,
        driver_id=driver,
        ts=datetime.now(UTC),
        t=t,
        type=CLASS_TO_ALERT[cls],
        kind="detected",
        severity="info",
        risk_score=float(prob[cls]),
        horizon_s=0,
        personalized=False,
        confidence=float(prob[cls]),
        modalities_used=used or ["motion"],
        reason=M.Reason(
            summary=f"Stub model flagged class '{S.CLASSES[cls]}' (synthetic data).",
            template_id="stub",
            evidence=[],
            modality_weights={"camera": camera / total, "motion": motion / total},
        ),
        degraded=len(used) < 2,
    )


def run_demo(out_dir: Path) -> dict:
    cfg = load_config("toy_fusion")
    set_seed(cfg["seed"])

    log.info("1/5 generating stub trips in %s", out_dir)
    dirs = make_stub_dataset(out_dir, load_config("stub"))

    log.info("2/5 validating against the data contract")
    for d in dirs:
        report = validate_processed_dir(d)
        if not report.ok:
            raise RuntimeError(f"contract validation failed for {d}: {report.errors[:5]}")

    log.info("3/5 building driver-disjoint windows")
    train = WindowDataset(dirs, "train")
    train.norm = train.fit_norm_stats()
    test = WindowDataset(dirs, "test", norm_stats=train.norm)
    tcfg = cfg["train"]
    loader = DataLoader(
        train, batch_size=tcfg["batch_size"], shuffle=True, generator=make_generator(cfg["seed"])
    )

    log.info("4/5 training toy cross-attention fusion")
    model = CrossAttentionFusion(**cfg["model"])
    hist = train_classifier(model, loader, tcfg["epochs"], tcfg["lr"], tcfg["weight_decay"])
    final = mean_loss(model, loader)
    if not final < hist.initial_loss:
        raise RuntimeError(f"loss did not drop: {hist.initial_loss:.3f} -> {final:.3f}")

    log.info("5/5 sending a model-derived alert through the mock API")
    out = predict(model, DataLoader(test, batch_size=128))
    best = int(np.argmax(out["probs"][:, 1:].max(axis=1)))  # most confident non-normal window
    item = test[best]
    client = TestClient(create_app(api_key=API_KEY, stream_delay_s=0.0))
    headers = {"X-API-Key": API_KEY}
    trip = client.post(
        "/v1/trips",
        headers=headers,
        json={
            "driver_id": "drv_demo",
            "device": {"type": "stub", "model": "demo", "mount": "none"},
            "contract_version": S.CONTRACT_VERSION,
            "sensors": {"camera": True, "imu": True, "gps": False},
        },
    ).json()
    alert = _alert_from_window(
        out["probs"][best], out["gates"][best], item["t_start"] + 5.0, trip["trip_id"], "drv_demo"
    )
    posted = client.post(
        f"/v1/trips/{trip['trip_id']}/alerts",
        headers=headers,
        json={"alert": alert.model_dump(mode="json")},
    )
    posted.raise_for_status()
    listed = client.get(f"/v1/trips/{trip['trip_id']}/alerts", headers=headers).json()
    return {
        "n_train_windows": len(train),
        "n_test_windows": len(test),
        "initial_loss": hist.initial_loss,
        "final_loss": final,
        "alert_type": listed[0]["type"],
        "alert_risk": listed[0]["risk_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root() / "demo_out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_demo(args.out)
    print("\nstub pipeline OK (synthetic data, plumbing check only)")
    for key, value in result.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
