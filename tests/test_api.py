"""
Integration tests for the Aegis AI FastAPI backend (Task 6.6).

Tests exercise all 5 endpoints using FastAPI's built-in TestClient
(no running server required). A temporary SQLite database is created
per test session and cleaned up on teardown.
"""
import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient

# Point the API to a throwaway temp DB before import
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ["LOG_DB_PATH"] = _TMP_DB.name

# Now import the app (which picks up LOG_DB_PATH)
import src.api.main as api_module
from src.api.main import app

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    """Re-initialise incidents table between test functions."""
    api_module._init_incidents_table()
    yield
    # Clean up rows between tests (not the file, just the data)
    import sqlite3
    conn = sqlite3.connect(_TMP_DB.name)
    conn.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sample_report(**overrides):
    payload = {
        "system_id": "test-sys-01",
        "anomaly_description": "Accuracy dropped below threshold",
        "failure_category": "model_issue",
        "raw_metrics": {"accuracy": 0.70, "latency_ms": 480},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "total_incidents" in data
        assert "pending_approvals" in data

    def test_health_uptime_is_positive(self):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# POST /report-anomaly
# ---------------------------------------------------------------------------

class TestReportAnomaly:
    def test_creates_incident_201(self):
        resp = client.post("/report-anomaly", json=_sample_report())
        assert resp.status_code == 201

    def test_response_contains_id(self):
        data = client.post("/report-anomaly", json=_sample_report()).json()
        assert "id" in data
        assert isinstance(data["id"], int)
        assert data["id"] >= 1

    def test_response_echoes_system_id(self):
        data = client.post("/report-anomaly", json=_sample_report(system_id="echo-sys")).json()
        assert data["system_id"] == "echo-sys"

    def test_response_has_fix_status(self):
        data = client.post("/report-anomaly", json=_sample_report()).json()
        assert data["fix_status"] in ("pending", "suggested", "pending_approval", "success", "failed")

    def test_missing_system_id_422(self):
        payload = _sample_report()
        del payload["system_id"]
        resp = client.post("/report-anomaly", json=payload)
        assert resp.status_code == 422

    def test_missing_failure_category_422(self):
        payload = _sample_report()
        del payload["failure_category"]
        resp = client.post("/report-anomaly", json=payload)
        assert resp.status_code == 422

    def test_empty_raw_metrics_accepted(self):
        payload = _sample_report(raw_metrics={})
        resp = client.post("/report-anomaly", json=payload)
        assert resp.status_code == 201

    def test_incident_count_increments(self):
        before = client.get("/health").json()["total_incidents"]
        client.post("/report-anomaly", json=_sample_report())
        after = client.get("/health").json()["total_incidents"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# GET /incidents
# ---------------------------------------------------------------------------

class TestListIncidents:
    def test_returns_200(self):
        resp = client.get("/incidents")
        assert resp.status_code == 200

    def test_empty_list_initially(self):
        data = client.get("/incidents").json()
        assert data["incidents"] == []
        assert data["total"] == 0

    def test_incident_appears_after_report(self):
        client.post("/report-anomaly", json=_sample_report())
        data = client.get("/incidents").json()
        assert data["total"] == 1
        assert len(data["incidents"]) == 1

    def test_pagination_fields_present(self):
        data = client.get("/incidents").json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "incidents" in data

    def test_filter_by_system_id(self):
        client.post("/report-anomaly", json=_sample_report(system_id="sys-A"))
        client.post("/report-anomaly", json=_sample_report(system_id="sys-B"))
        data = client.get("/incidents", params={"system_id": "sys-A"}).json()
        assert data["total"] == 1
        assert data["incidents"][0]["system_id"] == "sys-A"

    def test_filter_by_fix_status(self):
        client.post("/report-anomaly", json=_sample_report())
        # All new incidents will have status suggested or pending_approval
        data = client.get("/incidents").json()
        status = data["incidents"][0]["fix_status"]
        filtered = client.get("/incidents", params={"fix_status": status}).json()
        assert filtered["total"] >= 1


# ---------------------------------------------------------------------------
# GET /incidents/{id}
# ---------------------------------------------------------------------------

class TestGetIncident:
    def test_returns_correct_incident(self):
        created = client.post("/report-anomaly", json=_sample_report()).json()
        inc_id = created["id"]
        fetched = client.get(f"/incidents/{inc_id}").json()
        assert fetched["id"] == inc_id
        assert fetched["system_id"] == created["system_id"]

    def test_404_for_missing_id(self):
        resp = client.get("/incidents/999999")
        assert resp.status_code == 404

    def test_response_has_root_cause(self):
        created = client.post("/report-anomaly", json=_sample_report()).json()
        fetched = client.get(f"/incidents/{created['id']}").json()
        assert "root_cause" in fetched


# ---------------------------------------------------------------------------
# POST /incidents/{id}/approve
# ---------------------------------------------------------------------------

class TestApproveIncident:
    def test_approve_returns_200(self):
        created = client.post("/report-anomaly", json=_sample_report()).json()
        resp = client.post(
            f"/incidents/{created['id']}/approve",
            json={"approved": True, "approver_note": "Looks good."},
        )
        assert resp.status_code == 200

    def test_approve_sets_approved_flag(self):
        created = client.post("/report-anomaly", json=_sample_report()).json()
        result = client.post(
            f"/incidents/{created['id']}/approve",
            json={"approved": True},
        ).json()
        assert result["approved"] is True

    def test_decline_sets_suggested_status(self):
        created = client.post("/report-anomaly", json=_sample_report()).json()
        result = client.post(
            f"/incidents/{created['id']}/approve",
            json={"approved": False},
        ).json()
        assert result["fix_status"] == "suggested"
        assert result["approved"] is False

    def test_approve_404_for_missing_id(self):
        resp = client.post(
            "/incidents/999999/approve",
            json={"approved": True},
        )
        assert resp.status_code == 404
