"""Canned data for the mock server (docs/05_api_contract.md sections 3.4, 3.7 and 5).

Nothing here comes from a model. Values are the examples written in the API contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

from driverisk.serving import schemas as M

# Wording of the evidence summary is the contract example; real text comes from the
# explanation module (P4-C5) and templates.yaml.
_SUMMARY = "Eyes closed 38% of the last minute (your usual: 6%). Camera is the main source."


def example_alert(trip_id: str, driver_id: str, alert_id: str = "alt_9c01") -> M.Alert:
    return M.Alert(
        alert_id=alert_id,
        trip_id=trip_id,
        driver_id=driver_id,
        ts=datetime(2026, 11, 2, 9, 31, 12, 340000, tzinfo=UTC),
        t=745.3,
        type="micro_sleep",  # predictive alerts name an event onset (ADR-0007), not the state
        kind="predictive",
        severity="high",
        risk_score=0.82,
        horizon_s=3,
        personalized=True,
        confidence=0.74,
        modalities_used=["camera", "motion"],
        reason=M.Reason(
            summary=_SUMMARY,
            template_id="drowsy_perclos",
            evidence=[
                M.Evidence(
                    feature="perclos_60s", value=0.38, baseline_mean=0.06, baseline_std=0.03,
                    z=10.7, modality="camera",
                ),
                M.Evidence(
                    feature="head_pitch", value=-14.0, baseline_mean=-2.0, baseline_std=4.0,
                    z=-3.0, modality="camera",
                ),
            ],
            modality_weights={"camera": 0.71, "motion": 0.29},
        ),
        degraded=False,
    )  # fmt: skip


def scripted_messages(trip_id: str, driver_id: str) -> list[M.WsMessage]:
    """calibrating, ready, predictive micro_sleep alert, degraded camera, recovered, end."""
    both: list[M.Modality] = ["camera", "motion"]
    return [
        M.StatusMessage(t=0.0, risk_score=0.05, modalities_used=both, calibration="calibrating"),
        M.StatusMessage(t=1.0, risk_score=0.07, modalities_used=both, calibration="calibrating"),
        M.StatusMessage(t=2.0, risk_score=0.08, modalities_used=both, calibration="ready"),
        M.StatusMessage(t=744.0, risk_score=0.61, modalities_used=both, calibration="ready"),
        M.AlertMessage(alert=example_alert(trip_id, driver_id)),
        M.DegradedMessage(modality="camera", reason="face_not_visible", t=750.2),
        M.StatusMessage(t=751.0, risk_score=0.30, modalities_used=["motion"], calibration="ready"),
        M.RecoveredMessage(modality="camera", t=760.0),
        M.StatusMessage(t=761.0, risk_score=0.20, modalities_used=both, calibration="ready"),
        M.TripEndedMessage(trip_id=trip_id),
    ]


def example_summary(trip_id: str) -> M.TripSummary:
    return M.TripSummary(
        trip_id=trip_id,
        duration_s=1803.0,
        mean_risk=0.18,
        max_risk=0.91,
        alerts_by_type={"drowsy": 2, "harsh_brake": 3},
        risky_segments=[
            M.RiskySegment(t_start=700.0, t_end=760.0, peak_risk=0.91, types=["drowsy"])
        ],
        degraded_fraction={"camera": 0.04, "motion": 0.0},
        route=None,  # GPS route only with consent
        baseline_comparison=M.BaselineComparison(
            personalized=True, alerts_if_global=7, alerts_personalized=5
        ),
    )
