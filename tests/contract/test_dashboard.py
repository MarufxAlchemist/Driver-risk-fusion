"""Static checks on the dashboard page (P1-D3).

No browser runs in CI, so these tests read the served HTML and check the things a browser test
would otherwise catch first: the page is served, the elements the script needs exist, server
text never becomes markup, the WebSocket uses `?token=`, every message type of the contract is
handled, and the summary endpoint is called. Message and alert types are read from
`schemas.py`, so a contract change that the dashboard does not follow fails here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from driverisk.serving import schemas as M
from driverisk.serving.mock_app import create_app
from driverisk.utils.paths import repo_root

REQUIRED_IDS = (
    # connection
    "key-form", "key", "save-key", "state",
    # trip list
    "start", "refresh", "trips", "trips-empty",
    # per-trip view
    "trip-view", "trip-title", "trip-meta", "conn", "reconnect", "banner",
    "gauge", "risk-value", "t", "calib", "mods", "stale",
    # alert feed and summary
    "alerts", "alerts-empty", "summary", "load-summary",
)  # fmt: skip


@pytest.fixture(scope="module")
def page() -> str:
    client = TestClient(create_app(api_key="test-key", stream_delay_s=0.0))
    resp = client.get("/dashboard/")
    assert resp.status_code == 200 and "text/html" in resp.headers["content-type"]
    return resp.text


@pytest.fixture(scope="module")
def script(page: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", page, flags=re.S)
    assert len(blocks) == 1, "expected exactly one inline script"
    return blocks[0]


def test_served_page_is_the_file_on_disk(page: str) -> None:
    on_disk = (repo_root() / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert page.replace("\r\n", "\n") == on_disk.replace("\r\n", "\n")


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_required_element_ids(page: str, element_id: str) -> None:
    assert f'id="{element_id}"' in page


def test_ids_used_by_script_exist(page: str, script: str) -> None:
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script))
    missing = sorted(i for i in used if f'id="{i}"' not in page)
    assert not missing, f"script looks up ids that the page does not define: {missing}"


@pytest.mark.parametrize(
    "forbidden",
    ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"],
)
def test_no_markup_injection_or_eval(page: str, forbidden: str) -> None:
    assert forbidden not in page


def test_works_offline_no_external_resources(page: str) -> None:
    # No CDN scripts, web fonts or remote images: the page must work from the mock server alone.
    assert "http://" not in page and "https://" not in page
    assert "@import" not in page
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=", page)


def test_websocket_uses_token_query(script: str) -> None:
    assert "new WebSocket(" in script
    assert '"/stream?token="' in script
    assert "encodeURIComponent(apiKey())" in script


def test_rest_calls_send_api_key_header(script: str) -> None:
    assert '"X-API-Key"' in script


def test_api_key_kept_in_session_storage_only(script: str) -> None:
    assert "localStorage" not in script
    assert "document.cookie" not in script
    # every sessionStorage access sits inside a try block
    for line in script.splitlines():
        if "sessionStorage" in line:
            assert line.strip().startswith("try {"), line


def _ws_message_types() -> list[str]:
    union = get_args(M.WsMessage)[0]
    return [model.model_fields["type"].default for model in get_args(union)]


def test_ws_message_types_read_from_schema() -> None:
    assert _ws_message_types() == ["status", "alert", "degraded", "recovered", "trip_ended"]


@pytest.mark.parametrize("msg_type", _ws_message_types())
def test_handles_every_ws_message_type(script: str, msg_type: str) -> None:
    assert f'case "{msg_type}":' in script


def test_unknown_ws_messages_are_ignored(script: str) -> None:
    switch = script[script.index("switch (msg.type)") :]
    assert "default:" in switch[: switch.index("\n    }\n")]
    assert "JSON.parse(e.data)" in script and "catch (err)" in script


@pytest.mark.parametrize("alert_type", get_args(M.AlertType))
def test_every_alert_type_has_a_label(script: str, alert_type: str) -> None:
    assert f'"{alert_type}":' in script


def test_state_alert_types_match_schema(script: str) -> None:
    found = re.search(r"var STATE_TYPES = \[(.*?)\];", script)
    assert found
    assert tuple(re.findall(r'"([a-z_]+)"', found.group(1))) == M.STATE_ALERT_TYPES


def test_severities_match_schema(script: str) -> None:
    found = re.search(r"var SEVERITIES = \[(.*?)\];", script)
    assert found
    assert tuple(re.findall(r'"([a-z_]+)"', found.group(1))) == get_args(M.Severity)


def test_calls_trip_list_alerts_and_summary_endpoints(script: str) -> None:
    assert 'api("/v1/trips")' in script
    assert '"/alerts"' in script
    assert '"/summary"' in script


def test_shows_alert_and_summary_fields(script: str) -> None:
    # Fields of Alert, Reason, Evidence and TripSummary the P1-D3 view must show.
    for name in (
        "severity", "kind", "horizon_s", "risk_score", "personalized", "summary", "evidence",
        "baseline_mean", "baseline_std", "modality_weights", "alerts_by_type", "risky_segments",
        "baseline_comparison", "alerts_if_global", "alerts_personalized", "degraded_fraction",
    ):  # fmt: skip
        assert f".{name}" in script, name


def test_inputs_are_labelled(page: str) -> None:
    for input_id in re.findall(r'<(?:input|meter)[^>]*\sid="([^"]+)"', page):
        assert f'for="{input_id}"' in page, input_id


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_script_parses_with_node(script: str, tmp_path: Path) -> None:
    js = tmp_path / "dashboard.js"
    js.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(js)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
