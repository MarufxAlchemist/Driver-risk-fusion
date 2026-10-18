# 05. API Contract

Purpose: let workstream D build the FastAPI server and dashboard against mock data while workstreams A-C build the models. The server never receives raw video.

Status: DRAFT v0.1. Base path `/v1`. Freeze target: end of week 3.

---

## 1. Conventions

- JSON over HTTP; live alerts over WebSocket.
- Timestamps: ISO 8601 UTC strings for wall clock (`ts`) and float seconds for trip time (`t`).
- IDs are opaque strings: `drv_xxxx` (driver), `trp_xxxx` (trip), `alt_xxxx` (alert).
- Auth (demo): header `X-API-Key: <token>` from environment variable. Production auth is Future Scope.
- Errors: standard envelope below.
- Pydantic models live in `src/driverisk/serving/schemas.py` and are the source of truth; this file must match them. A CI test dumps the OpenAPI schema and diffs it against `docs/openapi_snapshot.json`.

Error envelope:

```json
{ "error": { "code": "TRIP_NOT_FOUND", "message": "Trip trp_01 does not exist", "details": {} } }
```

| HTTP | Code examples |
|---|---|
| 400 | `INVALID_WINDOW`, `CONTRACT_VERSION_MISMATCH` |
| 401 | `BAD_API_KEY` |
| 404 | `TRIP_NOT_FOUND`, `DRIVER_NOT_FOUND` |
| 409 | `TRIP_ALREADY_ENDED` |
| 422 | Pydantic validation errors |

---

## 2. Endpoints

| Method | Path | Purpose | Used by |
|---|---|---|---|
| GET | `/v1/health` | Liveness, model version, contract version | Everyone |
| POST | `/v1/drivers` | Register an anonymous driver | Edge app |
| GET | `/v1/drivers/{driver_id}` | Driver profile and calibration status | Dashboard |
| GET | `/v1/drivers/{driver_id}/baseline` | Personal baseline summary | Dashboard |
| POST | `/v1/drivers/{driver_id}/calibrate` | Start / reset calibration | Edge app |
| POST | `/v1/trips` | Start a trip | Edge app |
| POST | `/v1/trips/{trip_id}/steps` | Upload a batch of feature steps (optional, for server-side inference or logging) | Edge app |
| POST | `/v1/trips/{trip_id}/alerts` | Edge reports an alert it raised | Edge app |
| WS | `/v1/trips/{trip_id}/stream` | Live status and alerts (server -> dashboard) | Dashboard |
| POST | `/v1/trips/{trip_id}/end` | End the trip, trigger summary | Edge app |
| GET | `/v1/trips/{trip_id}` | Trip metadata | Dashboard |
| GET | `/v1/trips/{trip_id}/alerts` | Alerts list | Dashboard |
| GET | `/v1/trips/{trip_id}/summary` | Trip analytics (scores, risky segments) | Dashboard |
| GET | `/v1/trips` | List trips (filters: `driver_id`, `from`, `to`) | Dashboard |
| POST | `/v1/replay` | Replay a processed trip through the pipeline (demo / testing) | Dev |

---

## 3. Schemas

### 3.1 Driver

```json
{
  "driver_id": "drv_a1b2",
  "created_at": "2026-11-02T09:15:00Z",
  "calibration": {
    "status": "calibrating",
    "windows_collected": 143,
    "windows_required": 600,
    "personalized": false
  }
}
```

`status` in `calibrating | ready | stale`. `personalized` is false until `windows_collected >= windows_required`.

### 3.2 Trip start

Request:

```json
{
  "driver_id": "drv_a1b2",
  "device": { "type": "laptop_webcam", "model": "dev", "mount": "windshield_center" },
  "contract_version": "0.1",
  "sensors": { "camera": true, "imu": true, "gps": true }
}
```

Response: `{ "trip_id": "trp_77ef", "started_at": "...", "stream_url": "/v1/trips/trp_77ef/stream" }`

### 3.3 Step batch (optional upload)

```json
{
  "contract_version": "0.1",
  "t0": 120.0,
  "steps": [
    {
      "x_v": [0.31,0.30,0.305,0.12,0,0,0.04,0.06,2.1,-3.5,0.4,1.0,-2.0,0,0.93,1],
      "x_m": [0.2,0.1,0.0,0.5,0.1,1.2,0.5,-0.3,12.4,0.2,0.8,null,null,1],
      "mask_v": true,
      "mask_m": true
    }
  ]
}
```

Lengths of `x_v` and `x_m` must equal 16 and 14 (`03_data_contract.md`). `null` means NaN.

### 3.4 Alert (central object)

```json
{
  "alert_id": "alt_9c01",
  "trip_id": "trp_77ef",
  "driver_id": "drv_a1b2",
  "ts": "2026-11-02T09:31:12.340Z",
  "t": 745.3,
  "type": "drowsy",
  "kind": "predictive",
  "severity": "high",
  "risk_score": 0.82,
  "horizon_s": 3,
  "personalized": true,
  "confidence": 0.74,
  "modalities_used": ["camera", "motion"],
  "reason": {
    "summary": "Eyes closed 38% of the last minute (your usual: 6%). Camera is the main source.",
    "template_id": "drowsy_perclos",
    "evidence": [
      { "feature": "perclos_60s", "value": 0.38, "baseline_mean": 0.06, "baseline_std": 0.03, "z": 10.7, "modality": "camera" },
      { "feature": "head_pitch", "value": -14.0, "baseline_mean": -2.0, "baseline_std": 4.0, "z": -3.0, "modality": "camera" }
    ],
    "modality_weights": { "camera": 0.71, "motion": 0.29 }
  },
  "degraded": false
}
```

Field rules:

| Field | Allowed values / rule |
|---|---|
| `type` | `aggressive_accel`, `harsh_brake`, `sharp_turn`, `lane_weave`, `drowsy`, `micro_sleep`, `distracted` |
| `kind` | `predictive` (risk forecast over `horizon_s`) or `detected` (already happening) |
| `severity` | `info`, `medium`, `high` |
| `risk_score` | 0-1, calibrated probability from the forecaster |
| `horizon_s` | 2, 3 or 5 for predictive; 0 for detected |
| `personalized` | True if the baseline was ready for this driver |
| `modalities_used` | Subset of `camera`, `motion`; one entry means degraded mode |
| `degraded` | True when a modality was masked during the window |
| `reason.evidence[].z` | Deviation from the driver's own baseline (or global if not personalized) |

### 3.5 Reason templates

| `template_id` | Example summary |
|---|---|
| `drowsy_perclos` | "Eyes closed {pct}% of the last minute (your usual: {base}%)." |
| `micro_sleep` | "Eyes closed for {sec} s." |
| `yawning` | "Frequent yawning ({n} in 2 min)." |
| `distracted_gaze` | "Looking away from the road for {sec} s (your usual: {base} s)." |
| `harsh_brake` | "Braking {val} m/s^2, well above your usual {base} m/s^2 at this speed." |
| `aggressive_accel` | "Acceleration {val} m/s^2 vs your usual {base} m/s^2." |
| `lane_weave` | "Repeated lane drifting ({n} drifts in {sec} s)." |
| `degraded_mode` | "Camera unavailable, using motion only." (status message, not an alert) |

Templates live in `src/driverisk/explain/templates.yaml` so wording can change without code changes. Every template must degrade gracefully when no baseline is available (omit the "your usual" part).

### 3.6 WebSocket messages (server -> client)

Each message has `type`:

```json
{ "type": "status", "t": 745.0, "risk_score": 0.31, "modalities_used": ["camera","motion"], "calibration": "ready" }
{ "type": "alert", "alert": { "...": "see 3.4" } }
{ "type": "degraded", "modality": "camera", "reason": "face_not_visible", "t": 750.2 }
{ "type": "trip_ended", "trip_id": "trp_77ef" }
```

Status messages at 1 Hz. The dashboard must tolerate missing or reordered messages.

### 3.7 Trip summary

```json
{
  "trip_id": "trp_77ef",
  "duration_s": 1803,
  "mean_risk": 0.18,
  "max_risk": 0.91,
  "alerts_by_type": { "drowsy": 2, "harsh_brake": 3 },
  "risky_segments": [ { "t_start": 700.0, "t_end": 760.0, "peak_risk": 0.91, "types": ["drowsy"] } ],
  "degraded_fraction": { "camera": 0.04, "motion": 0.0 },
  "route": [ { "t": 0, "lat": 18.52, "lon": 73.85 } ],
  "baseline_comparison": { "personalized": true, "alerts_if_global": 7, "alerts_personalized": 5 }
}
```

`route` is included only if the driver consents to GPS storage.

---

## 4. Edge <-> server responsibilities

| Concern | Edge | Server |
|---|---|---|
| Raw video | Processes and discards (or stores locally if recording mode) | Never receives |
| Feature extraction, fusion, alerts | Yes | No (optional replay mode for testing) |
| Personal baseline | Stored on device | Optional backup of summary statistics only |
| Trip history, dashboard | No | Yes |

---

## 5. Mock server requirement (week 2 stub)

Workstream D ships `scripts/mock_server.py` returning canned responses for every endpoint above and emitting a scripted alert sequence on the WebSocket (calibrating -> ready -> predictive drowsy alert -> degraded camera -> recovered). Dashboard development uses it until real models exist.

---

## 6. Non-functional targets

| Item | Target |
|---|---|
| Alert delivery over WebSocket | < 100 ms after the edge reports it |
| Server availability during demo | Runs locally with `make serve` |
| Max payload per step batch | 600 steps (60 s) |
| Logging | Structured JSON logs, no PII |
