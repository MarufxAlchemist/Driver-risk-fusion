"""Mock FastAPI server returning canned data for every endpoint (05 section 5).

Everything is in memory and nothing comes from a model: it exists so the dashboard can be
built before real models exist. Auth: header `X-API-Key` on REST; WebSocket clients cannot set
headers in browsers, so `/stream` takes `?token=` instead (assumption A10, pending approval).
"""

from __future__ import annotations

import asyncio
import os
import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from driverisk.data import schema as S
from driverisk.serving import schemas as M
from driverisk.serving.scripted import example_alert, example_summary, scripted_messages
from driverisk.utils.paths import repo_root

API_KEY_ENV = "DRIVERISK_API_KEY"
DEFAULT_API_KEY = "dev-token"  # demo only; production auth is Future Scope
MODEL_VERSION = "mock-0"
WINDOWS_REQUIRED = 600  # 10 min of 1 s windows; mock value, tuned in P4 (assumption A8)
DEMO_DRIVER_ID = "drv_demo"

_ERR = {"model": M.ErrorEnvelope}
ERROR_RESPONSES: dict[int | str, dict] = {
    400: _ERR, 401: _ERR, 404: _ERR, 409: _ERR,
}  # fmt: skip


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, **details: object) -> None:
        self.status_code, self.code, self.message, self.details = (
            status_code, code, message, details,
        )  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str, taken: dict) -> str:
    while (value := f"{prefix}_{secrets.token_hex(2)}") in taken:
        pass
    return value


def _calibration(collected: int) -> M.Calibration:
    ready = collected >= WINDOWS_REQUIRED
    return M.Calibration(
        status="ready" if ready else "calibrating",
        windows_collected=collected,
        windows_required=WINDOWS_REQUIRED,
        personalized=ready,
    )


class _State:
    """In-memory store, one per app instance so tests stay independent."""

    def __init__(self) -> None:
        now = _now()
        self.drivers: dict[str, M.Driver] = {
            DEMO_DRIVER_ID: M.Driver(
                driver_id=DEMO_DRIVER_ID, created_at=now, calibration=_calibration(143)
            )
        }
        self.trips: dict[str, M.TripInfo] = {}
        self.alerts: dict[str, list[M.Alert]] = {}
        self.stream_delay_s = 1.0  # pause between scripted WebSocket messages


def create_app(api_key: str | None = None, stream_delay_s: float = 1.0) -> FastAPI:
    key = api_key or os.environ.get(API_KEY_ENV, DEFAULT_API_KEY)
    state = _State()
    state.stream_delay_s = stream_delay_s
    app = FastAPI(
        title="Driver Risk Fusion API (mock)",
        version=S.CONTRACT_VERSION,
        description="Canned responses for dashboard development. See docs/05_api_contract.md.",
    )
    app.state.store = state
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(value: str | None = Depends(header)) -> None:
        if value != key:
            raise ApiError(401, "BAD_API_KEY", "Missing or invalid X-API-Key")

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        body = M.ErrorEnvelope(
            error=M.ErrorBody(code=exc.code, message=exc.message, details=exc.details)
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep location and message only: `input` could echo request data back.
        errors = [
            {"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        body = M.ErrorEnvelope(
            error=M.ErrorBody(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"errors": errors},
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    def driver_or_404(driver_id: str) -> M.Driver:
        if driver_id not in state.drivers:
            raise ApiError(404, "DRIVER_NOT_FOUND", f"Driver {driver_id} does not exist")
        return state.drivers[driver_id]

    def trip_or_404(trip_id: str) -> M.TripInfo:
        if trip_id not in state.trips:
            raise ApiError(404, "TRIP_NOT_FOUND", f"Trip {trip_id} does not exist")
        return state.trips[trip_id]

    def open_trip_or_error(trip_id: str) -> M.TripInfo:
        trip = trip_or_404(trip_id)
        if trip.ended_at is not None:
            raise ApiError(409, "TRIP_ALREADY_ENDED", f"Trip {trip_id} has ended")
        return trip

    def check_version(version: str) -> None:
        if version != S.CONTRACT_VERSION:
            raise ApiError(
                400, "CONTRACT_VERSION_MISMATCH", "Unsupported contract_version",
                expected=S.CONTRACT_VERSION, got=version,
            )  # fmt: skip

    def start_trip(req: M.TripStartRequest) -> M.TripStartResponse:
        check_version(req.contract_version)
        driver_or_404(req.driver_id)
        trip_id, started = _new_id("trp", state.trips), _now()
        state.trips[trip_id] = M.TripInfo(
            trip_id=trip_id, driver_id=req.driver_id, started_at=started,
            device=req.device, sensors=req.sensors, contract_version=req.contract_version,
        )  # fmt: skip
        state.alerts[trip_id] = []
        return M.TripStartResponse(
            trip_id=trip_id, started_at=started, stream_url=f"/v1/trips/{trip_id}/stream"
        )

    # --- REST ------------------------------------------------------------------------------
    @app.get("/v1/health", response_model=M.Health, tags=["system"])
    def health() -> M.Health:
        return M.Health(model_version=MODEL_VERSION)

    api = dict(dependencies=[Depends(require_key)], responses=ERROR_RESPONSES)

    @app.post("/v1/drivers", response_model=M.Driver, status_code=201, tags=["drivers"], **api)
    def create_driver() -> M.Driver:
        driver_id = _new_id("drv", state.drivers)
        state.drivers[driver_id] = M.Driver(
            driver_id=driver_id, created_at=_now(), calibration=_calibration(0)
        )
        return state.drivers[driver_id]

    @app.get("/v1/drivers/{driver_id}", response_model=M.Driver, tags=["drivers"], **api)
    def get_driver(driver_id: str) -> M.Driver:
        return driver_or_404(driver_id)

    @app.get(
        "/v1/drivers/{driver_id}/baseline",
        response_model=M.BaselineSummary,
        tags=["drivers"],
        **api,
    )
    def get_baseline(driver_id: str) -> M.BaselineSummary:
        driver = driver_or_404(driver_id)
        return M.BaselineSummary(
            driver_id=driver_id,
            personalized=driver.calibration.personalized,
            n_windows=driver.calibration.windows_collected,
            stats=[
                M.BaselineStat(feature="perclos_60s", mean=0.06, std=0.03, modality="camera"),
                M.BaselineStat(feature="head_pitch", mean=-2.0, std=4.0, modality="camera"),
                M.BaselineStat(feature="acc_long", mean=0.1, std=0.9, modality="motion"),
            ],
        )

    @app.post("/v1/drivers/{driver_id}/calibrate", response_model=M.Driver, tags=["drivers"], **api)
    def calibrate(driver_id: str) -> M.Driver:
        driver_or_404(driver_id)
        state.drivers[driver_id] = state.drivers[driver_id].model_copy(
            update={"calibration": _calibration(0)}
        )
        return state.drivers[driver_id]

    @app.post(
        "/v1/trips", response_model=M.TripStartResponse, status_code=201, tags=["trips"], **api
    )
    def post_trip(req: M.TripStartRequest) -> M.TripStartResponse:
        return start_trip(req)

    @app.get("/v1/trips", response_model=list[M.TripInfo], tags=["trips"], **api)
    def list_trips(
        driver_id: str | None = None,
        from_: Annotated[datetime | None, Query(alias="from")] = None,
        to: datetime | None = None,
    ) -> list[M.TripInfo]:
        return [
            t for t in state.trips.values()
            if (driver_id is None or t.driver_id == driver_id)
            and (from_ is None or t.started_at >= from_)
            and (to is None or t.started_at <= to)
        ]  # fmt: skip

    @app.get("/v1/trips/{trip_id}", response_model=M.TripInfo, tags=["trips"], **api)
    def get_trip(trip_id: str) -> M.TripInfo:
        return trip_or_404(trip_id)

    @app.post("/v1/trips/{trip_id}/steps", response_model=M.StepsAck, tags=["trips"], **api)
    def post_steps(trip_id: str, batch: M.StepBatch) -> M.StepsAck:
        open_trip_or_error(trip_id)
        check_version(batch.contract_version)
        if len(batch.steps) > M.MAX_STEPS_PER_BATCH:
            raise ApiError(
                400, "INVALID_WINDOW", "Too many steps in one batch",
                max_steps=M.MAX_STEPS_PER_BATCH, got=len(batch.steps),
            )  # fmt: skip
        return M.StepsAck(accepted=len(batch.steps), t_next=batch.t0 + len(batch.steps) * S.STEP_S)

    @app.post(
        "/v1/trips/{trip_id}/alerts", response_model=M.Alert, status_code=201, tags=["trips"], **api
    )
    def report_alert(trip_id: str, body: M.AlertReport) -> M.Alert:
        open_trip_or_error(trip_id)
        state.alerts[trip_id].append(body.alert)
        return body.alert

    @app.get("/v1/trips/{trip_id}/alerts", response_model=list[M.Alert], tags=["trips"], **api)
    def list_alerts(trip_id: str) -> list[M.Alert]:
        trip = trip_or_404(trip_id)
        return state.alerts.get(trip_id) or [example_alert(trip_id, trip.driver_id)]

    @app.post("/v1/trips/{trip_id}/end", response_model=M.TripSummary, tags=["trips"], **api)
    def end_trip(trip_id: str) -> M.TripSummary:
        trip = open_trip_or_error(trip_id)
        state.trips[trip_id] = trip.model_copy(update={"ended_at": _now()})
        return example_summary(trip_id)

    @app.get("/v1/trips/{trip_id}/summary", response_model=M.TripSummary, tags=["trips"], **api)
    def get_summary(trip_id: str) -> M.TripSummary:
        trip_or_404(trip_id)
        return example_summary(trip_id)

    @app.post("/v1/replay", response_model=M.ReplayResponse, status_code=201, tags=["dev"], **api)
    def replay(req: M.ReplayRequest) -> M.ReplayResponse:
        driver_or_404(req.driver_id)
        started = start_trip(
            M.TripStartRequest(
                driver_id=req.driver_id,
                device=M.Device(type="replay", model=req.trip_id, mount="none"),
                contract_version=S.CONTRACT_VERSION,
                sensors=M.Sensors(camera=True, imu=True, gps=True),
            )
        )
        return M.ReplayResponse(trip_id=started.trip_id, stream_url=started.stream_url)

    # --- WebSocket -------------------------------------------------------------------------
    @app.websocket("/v1/trips/{trip_id}/stream")
    async def stream(ws: WebSocket, trip_id: str, token: str | None = None) -> None:
        # Browsers cannot set headers on a WebSocket, so the key travels as ?token= (A10).
        if token != key or trip_id not in state.trips:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await ws.accept()
        trip = state.trips[trip_id]
        try:
            for message in scripted_messages(trip_id, trip.driver_id):
                await ws.send_text(message.model_dump_json())
                await asyncio.sleep(state.stream_delay_s)
            await ws.close()
        except WebSocketDisconnect:
            return

    # Dashboard shell; mounts are not part of the OpenAPI document.
    dashboard_dir = repo_root() / "dashboard"
    if dashboard_dir.is_dir():
        app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    return app


app = create_app()
