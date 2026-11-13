"""Mock API: every endpoint in docs/05_api_contract.md, the error envelope, the scripted
WebSocket sequence, and the OpenAPI snapshot."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from driverisk.data import schema as S
from driverisk.serving import schemas as M
from driverisk.serving.mock_app import create_app
from driverisk.serving.scripted import example_alert, scripted_messages
from driverisk.utils.paths import repo_root

KEY = "test-key"
H = {"X-API-Key": KEY}
TRIP_BODY = {
    "driver_id": "drv_demo",
    "device": {"type": "laptop_webcam", "model": "dev", "mount": "windshield_center"},
    "contract_version": S.CONTRACT_VERSION,
    "sensors": {"camera": True, "imu": True, "gps": True},
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(api_key=KEY, stream_delay_s=0.0))


@pytest.fixture
def trip_id(client: TestClient) -> str:
    return client.post("/v1/trips", json=TRIP_BODY, headers=H).json()["trip_id"]


def _err(resp, status: int, code: str) -> None:
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code


def test_openapi_matches_snapshot() -> None:
    snapshot = (repo_root() / "docs" / "openapi_snapshot.json").read_text(encoding="utf-8")
    current = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    assert current == snapshot, "API schema changed: run scripts/update_openapi_snapshot.py"


def test_health_needs_no_key(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok" and body["contract_version"] == S.CONTRACT_VERSION


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/v1/trips"), ("post", "/v1/drivers"), ("get", "/v1/drivers/drv_demo")],
)
def test_bad_key_is_401(client: TestClient, method: str, path: str) -> None:
    _err(getattr(client, method)(path), 401, "BAD_API_KEY")
    _err(getattr(client, method)(path, headers={"X-API-Key": "nope"}), 401, "BAD_API_KEY")


def test_driver_lifecycle(client: TestClient) -> None:
    created = client.post("/v1/drivers", headers=H)
    assert created.status_code == 201
    driver = M.Driver.model_validate(created.json())
    assert driver.calibration.status == "calibrating" and not driver.calibration.personalized
    assert client.get(f"/v1/drivers/{driver.driver_id}", headers=H).status_code == 200
    base = M.BaselineSummary.model_validate(
        client.get(f"/v1/drivers/{driver.driver_id}/baseline", headers=H).json()
    )
    assert base.stats
    again = client.post(f"/v1/drivers/{driver.driver_id}/calibrate", headers=H).json()
    assert again["calibration"]["windows_collected"] == 0
    _err(client.get("/v1/drivers/drv_nope", headers=H), 404, "DRIVER_NOT_FOUND")


def test_trip_lifecycle(client: TestClient) -> None:
    started = client.post("/v1/trips", json=TRIP_BODY, headers=H)
    assert started.status_code == 201
    out = M.TripStartResponse.model_validate(started.json())
    assert out.stream_url == f"/v1/trips/{out.trip_id}/stream"
    assert client.get(f"/v1/trips/{out.trip_id}", headers=H).json()["ended_at"] is None
    assert [t["trip_id"] for t in client.get("/v1/trips?driver_id=drv_demo", headers=H).json()] == [
        out.trip_id
    ]
    assert client.get("/v1/trips?driver_id=drv_other", headers=H).json() == []
    M.TripSummary.model_validate(client.get(f"/v1/trips/{out.trip_id}/summary", headers=H).json())
    ended = client.post(f"/v1/trips/{out.trip_id}/end", headers=H)
    M.TripSummary.model_validate(ended.json())
    _err(client.post(f"/v1/trips/{out.trip_id}/end", headers=H), 409, "TRIP_ALREADY_ENDED")
    _err(client.get("/v1/trips/trp_nope", headers=H), 404, "TRIP_NOT_FOUND")


def test_trip_start_validation(client: TestClient) -> None:
    _err(
        client.post("/v1/trips", json={**TRIP_BODY, "contract_version": "9.9"}, headers=H),
        400,
        "CONTRACT_VERSION_MISMATCH",
    )
    _err(
        client.post("/v1/trips", json={**TRIP_BODY, "driver_id": "drv_nope"}, headers=H),
        404,
        "DRIVER_NOT_FOUND",
    )
    resp = client.post("/v1/trips", json={"driver_id": "drv_demo"}, headers=H)
    _err(resp, 422, "VALIDATION_ERROR")


def _step(n_v: int = S.DV, n_m: int = S.DM) -> dict:
    return {"x_v": [0.1] * n_v, "x_m": [None] * n_m, "mask_v": True, "mask_m": False}


def test_steps_batch(client: TestClient, trip_id: str) -> None:
    ok = client.post(
        f"/v1/trips/{trip_id}/steps",
        json={"contract_version": S.CONTRACT_VERSION, "t0": 120.0, "steps": [_step()] * 5},
        headers=H,
    )
    assert ok.json() == {"accepted": 5, "t_next": pytest.approx(120.5)}


def test_steps_wrong_lengths_are_422(client: TestClient, trip_id: str) -> None:
    for step in (_step(n_v=15), _step(n_m=13)):
        resp = client.post(
            f"/v1/trips/{trip_id}/steps",
            json={"contract_version": S.CONTRACT_VERSION, "t0": 0, "steps": [step]},
            headers=H,
        )
        _err(resp, 422, "VALIDATION_ERROR")


def test_steps_too_many_is_invalid_window(client: TestClient, trip_id: str) -> None:
    body = {
        "contract_version": S.CONTRACT_VERSION,
        "t0": 0,
        "steps": [_step()] * (M.MAX_STEPS_PER_BATCH + 1),
    }
    _err(client.post(f"/v1/trips/{trip_id}/steps", json=body, headers=H), 400, "INVALID_WINDOW")


def test_alert_report_and_list(client: TestClient, trip_id: str) -> None:
    alert = example_alert(trip_id, "drv_demo", "alt_0001")
    posted = client.post(
        f"/v1/trips/{trip_id}/alerts", json={"alert": alert.model_dump(mode="json")}, headers=H
    )
    assert posted.status_code == 201
    listed = client.get(f"/v1/trips/{trip_id}/alerts", headers=H).json()
    assert [a["alert_id"] for a in listed] == ["alt_0001"]


def test_replay(client: TestClient) -> None:
    resp = client.post(
        "/v1/replay",
        json={"trip_id": "own_d03_20261120_02", "driver_id": "drv_demo"},
        headers=H,
    )
    assert resp.status_code == 201
    assert resp.json()["stream_url"].endswith("/stream")


def test_alert_field_rules() -> None:
    base = example_alert("trp_1", "drv_1").model_dump()
    with pytest.raises(ValueError, match="horizon_s"):
        M.Alert.model_validate({**base, "kind": "detected"})  # detected needs horizon 0
    with pytest.raises(ValueError, match="horizon_s"):
        M.Alert.model_validate({**base, "horizon_s": 4})
    with pytest.raises(ValueError):
        M.Alert.model_validate({**base, "risk_score": 1.2})
    with pytest.raises(ValueError):
        M.Alert.model_validate({**base, "type": "texting"})
    ok = M.Alert.model_validate({**base, "kind": "detected", "horizon_s": 0})
    assert ok.kind == "detected"


def test_contract_example_numbers() -> None:
    ev = example_alert("trp_1", "drv_1").reason.evidence
    for e in ev:
        assert e.z == pytest.approx((e.value - e.baseline_mean) / e.baseline_std, abs=0.05)


def test_scripted_sequence_order() -> None:
    kinds = [m.type for m in scripted_messages("trp_1", "drv_1")]
    assert kinds[0] == "status" and kinds[-1] == "trip_ended"
    order = [kinds.index(k) for k in ("alert", "degraded", "recovered")]
    assert order == sorted(order)
    calib = [m.calibration for m in scripted_messages("trp_1", "drv_1") if m.type == "status"]
    assert calib[0] == "calibrating" and calib[-1] == "ready"
    # while the camera is degraded, status reports motion only
    degraded = kinds.index("degraded")
    assert scripted_messages("trp_1", "drv_1")[degraded + 1].modalities_used == ["motion"]


def test_websocket_streams_scripted_sequence(client: TestClient, trip_id: str) -> None:
    adapter = TypeAdapter(M.WsMessage)
    with client.websocket_connect(f"/v1/trips/{trip_id}/stream?token={KEY}") as ws:
        received = []
        while True:
            try:
                received.append(adapter.validate_json(ws.receive_text()))
            except Exception:  # server closes after trip_ended
                break
    assert [m.type for m in received] == [m.type for m in scripted_messages(trip_id, "drv_demo")]


def test_websocket_rejects_bad_token_and_unknown_trip(client: TestClient, trip_id: str) -> None:
    from starlette.websockets import WebSocketDisconnect

    for url in (f"/v1/trips/{trip_id}/stream?token=bad", f"/v1/trips/trp_nope/stream?token={KEY}"):
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(url):
            pass


def test_ws_messages_reject_garbage() -> None:
    adapter = TypeAdapter(M.WsMessage)
    with pytest.raises(ValueError):
        adapter.validate_python({"type": "status", "t": 1.0})
    with pytest.raises(ValueError):
        adapter.validate_python({"type": "unknown"})


def test_dashboard_shell_is_served(client: TestClient) -> None:
    page = client.get("/dashboard/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for needle in ('id="gauge"', 'id="alerts"', "/v1/trips", "?token="):
        assert needle in page.text
    # Alert text comes from the server; it must never be inserted as markup.
    assert "innerHTML" not in page.text
