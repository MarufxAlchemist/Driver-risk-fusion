"""Pydantic models: the source of truth for docs/05_api_contract.md (contract version 0.1).

A CI test dumps the OpenAPI document and diffs it against docs/openapi_snapshot.json, so
any change here is visible in review. Field rules mirror section 3 of the API contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from driverisk.data import schema as S

Modality = Literal["camera", "motion"]
# ADR-0007 part 2: event alerts use the event names of the data contract, state alerts use the
# window-class names. A test pins this Literal to schema.EVENT_TYPES + schema.CLASSES[1:].
AlertType = Literal[
    "harsh_accel", "harsh_brake", "sharp_turn", "lane_weave", "micro_sleep", "distraction_episode",
    "aggressive", "drowsy", "distracted",
]  # fmt: skip
EVENT_ALERT_TYPES: tuple[str, ...] = S.EVENT_TYPES
STATE_ALERT_TYPES: tuple[str, ...] = S.CLASSES[1:]
AlertKind = Literal["predictive", "detected"]
Severity = Literal["info", "medium", "high"]
CalibrationStatus = Literal["calibrating", "ready", "stale"]

MAX_STEPS_PER_BATCH = 600  # 05 section 6: 60 s at 10 Hz


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Errors (05 section 1) -------------------------------------------------------------------
class ErrorBody(_Model):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(_Model):
    error: ErrorBody


# --- Health ----------------------------------------------------------------------------------
class Health(_Model):
    status: Literal["ok"] = "ok"
    model_version: str
    contract_version: str = S.CONTRACT_VERSION


# --- Drivers (3.1) ---------------------------------------------------------------------------
class Calibration(_Model):
    status: CalibrationStatus
    windows_collected: int = Field(ge=0)
    windows_required: int = Field(gt=0)
    personalized: bool

    @model_validator(mode="after")
    def _personalized_only_when_ready(self) -> Calibration:
        if self.personalized != (self.windows_collected >= self.windows_required):
            raise ValueError("personalized must equal windows_collected >= windows_required")
        return self


class Driver(_Model):
    driver_id: str = Field(pattern=r"^drv_[0-9a-z]+$")
    created_at: datetime
    calibration: Calibration


class BaselineStat(_Model):
    feature: str
    mean: float
    std: float = Field(ge=0)
    modality: Modality


class BaselineSummary(_Model):
    driver_id: str
    personalized: bool
    n_windows: int = Field(ge=0)
    stats: list[BaselineStat]


# --- Trips (3.2) -----------------------------------------------------------------------------
class Device(_Model):
    type: str
    model: str
    mount: str


class Sensors(_Model):
    camera: bool
    imu: bool
    gps: bool


class TripStartRequest(_Model):
    driver_id: str
    device: Device
    contract_version: str
    sensors: Sensors


class TripStartResponse(_Model):
    trip_id: str = Field(pattern=r"^trp_[0-9a-z]+$")
    started_at: datetime
    stream_url: str


class TripInfo(_Model):
    trip_id: str
    driver_id: str
    started_at: datetime
    ended_at: datetime | None = None
    device: Device
    sensors: Sensors
    contract_version: str


# --- Step batch (3.3) ------------------------------------------------------------------------
class Step(_Model):
    x_v: list[float | None] = Field(min_length=S.DV, max_length=S.DV)
    x_m: list[float | None] = Field(min_length=S.DM, max_length=S.DM)
    mask_v: bool
    mask_m: bool


class StepBatch(_Model):
    contract_version: str
    t0: float = Field(ge=0)
    steps: list[Step] = Field(min_length=1)


class StepsAck(_Model):
    accepted: int = Field(ge=0)
    t_next: float


# --- Alerts (3.4) ----------------------------------------------------------------------------
class Evidence(_Model):
    feature: str
    value: float
    baseline_mean: float
    baseline_std: float = Field(ge=0)
    z: float
    modality: Modality


class Reason(_Model):
    summary: str
    template_id: str
    evidence: list[Evidence]
    modality_weights: dict[Modality, float]


class Alert(_Model):
    alert_id: str = Field(pattern=r"^alt_[0-9a-z]+$")
    trip_id: str
    driver_id: str
    ts: datetime
    t: float = Field(ge=0)
    type: AlertType
    kind: AlertKind
    severity: Severity
    risk_score: float = Field(ge=0, le=1)
    horizon_s: Literal[0, 2, 3, 5]
    personalized: bool
    confidence: float = Field(ge=0, le=1)
    modalities_used: list[Modality] = Field(min_length=1)
    reason: Reason
    degraded: bool

    @model_validator(mode="after")
    def _horizon_matches_kind(self) -> Alert:
        if self.kind == "predictive" and self.horizon_s not in (2, 3, 5):
            raise ValueError("predictive alerts need horizon_s in {2, 3, 5}")
        if self.kind == "predictive" and self.type not in EVENT_ALERT_TYPES:
            raise ValueError("predictive alerts are only allowed for event types (forecast onsets)")
        if self.kind == "detected" and self.horizon_s != 0:
            raise ValueError("detected alerts need horizon_s == 0")
        return self


class AlertReport(_Model):
    """Body of POST /trips/{id}/alerts: the alert the edge raised (server fills nothing in)."""

    alert: Alert


# --- WebSocket messages, server -> client (3.6) ----------------------------------------------
class StatusMessage(_Model):
    type: Literal["status"] = "status"
    t: float
    risk_score: float = Field(ge=0, le=1)
    modalities_used: list[Modality]
    calibration: CalibrationStatus


class AlertMessage(_Model):
    type: Literal["alert"] = "alert"
    alert: Alert


class DegradedMessage(_Model):
    type: Literal["degraded"] = "degraded"
    modality: Modality
    reason: str
    t: float


class RecoveredMessage(_Model):
    """Closes a `degraded` message for the same modality (docs/05 section 3.6)."""

    type: Literal["recovered"] = "recovered"
    modality: Modality
    t: float


class TripEndedMessage(_Model):
    type: Literal["trip_ended"] = "trip_ended"
    trip_id: str


WsMessage = Annotated[
    StatusMessage | AlertMessage | DegradedMessage | RecoveredMessage | TripEndedMessage,
    Field(discriminator="type"),
]


# --- Trip summary (3.7) ----------------------------------------------------------------------
class RiskySegment(_Model):
    t_start: float
    t_end: float
    peak_risk: float = Field(ge=0, le=1)
    types: list[AlertType]


class RoutePoint(_Model):
    t: float
    lat: float
    lon: float


class BaselineComparison(_Model):
    personalized: bool
    alerts_if_global: int = Field(ge=0)
    alerts_personalized: int = Field(ge=0)


class TripSummary(_Model):
    trip_id: str
    duration_s: float = Field(ge=0)
    mean_risk: float = Field(ge=0, le=1)
    max_risk: float = Field(ge=0, le=1)
    alerts_by_type: dict[AlertType, int]
    risky_segments: list[RiskySegment]
    degraded_fraction: dict[Modality, float]
    route: list[RoutePoint] | None = None  # only with GPS consent
    baseline_comparison: BaselineComparison


# --- Replay (dev only) -----------------------------------------------------------------------
class ReplayRequest(_Model):
    trip_id: str = Field(description="Processed trip id, e.g. own_d03_20261120_02")
    driver_id: str
    speed: float = Field(default=1.0, gt=0)


class ReplayResponse(_Model):
    trip_id: str = Field(pattern=r"^trp_[0-9a-z]+$")
    stream_url: str
