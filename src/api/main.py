"""
Aegis AI — FastAPI Backend
==========================
Provides 5 REST endpoints to expose the self-healing pipeline:

  POST /report-anomaly          — accept raw metrics; run agent; store incident
  GET  /incidents               — paginated list of all incidents
  GET  /incidents/{id}          — single incident detail
  POST /incidents/{id}/approve  — approve pending HITL fix; re-executes fix
  GET  /health                  — service liveness + stats

Incidents are stored in the same SQLite database used by AuditLogger,
in a dedicated `incidents` table.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.models import (
    AnomalyReportRequest,
    ApprovalRequest,
    HealthResponse,
    IncidentListResponse,
    IncidentResponse,
)
from src.healing.audit_logger import AuditLogger
from src.healing.fix_executor import FixExecutor

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DB_PATH: str = os.getenv("LOG_DB_PATH", "data/logs/events.db")
_START_TIME: float = time.time()


def _get_conn() -> sqlite3.Connection:
    """Return a configured SQLite connection."""
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_incidents_table() -> None:
    """Create the `incidents` table if it does not exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id TEXT NOT NULL,
            anomaly_description TEXT NOT NULL,
            failure_category TEXT NOT NULL,
            raw_metrics TEXT,
            root_cause TEXT,
            fix_recommendation TEXT,
            confidence_score REAL,
            requires_human_review INTEGER DEFAULT 0,
            fix_status TEXT DEFAULT 'pending',
            approved INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Incidents table ready.")


def _row_to_incident(row: sqlite3.Row) -> IncidentResponse:
    """Convert a SQLite Row to an IncidentResponse model."""
    raw = json.loads(row["raw_metrics"]) if row["raw_metrics"] else {}
    return IncidentResponse(
        id=row["id"],
        system_id=row["system_id"],
        anomaly_description=row["anomaly_description"],
        failure_category=row["failure_category"],
        raw_metrics=raw,
        root_cause=row["root_cause"],
        fix_recommendation=row["fix_recommendation"],
        confidence_score=row["confidence_score"],
        requires_human_review=bool(row["requires_human_review"]),
        fix_status=row["fix_status"] or "pending",
        approved=bool(row["approved"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Initialize DB table on startup."""
    _init_incidents_table()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aegis AI Self-Healing API",
    description=(
        "REST interface for the Aegis AI LLM monitoring and self-healing system. "
        "Submit anomaly reports, review agent diagnoses, and approve autonomous fixes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared audit logger (writes to same DB)
_audit_logger = AuditLogger(db_path=_DB_PATH)


# ---------------------------------------------------------------------------
# Agent runner helper (lazy import to avoid heavy startup cost)
# ---------------------------------------------------------------------------

def _run_agent_diagnosis(
    system_id: str,
    anomaly_description: str,
    failure_category: str,
    raw_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Invoke the LangGraph debugging agent graph and return the final state.
    Falls back to a rule-based stub when OPENAI_API_KEY is not set, so the
    API remains functional in offline / test environments.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-placeholder"):
        logger.warning("OPENAI_API_KEY not set — returning stub diagnosis.")
        confidence = 0.6
        return {
            "root_cause": f"[Stub] Detected {failure_category} in system {system_id}.",
            "fix_recommendation": "Review system logs and apply corrective action.",
            "confidence_score": confidence,
            "requires_human_review": confidence < 0.75,
        }

    try:
        from langchain_openai import ChatOpenAI
        from src.agent.graph import build_graph

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        graph = build_graph(llm=llm)
        initial_state = {
            "system_id": system_id,
            "anomaly_description": anomaly_description,
            "failure_category": failure_category,
            "raw_metrics": raw_metrics,
            "retrieved_context": [],
            "analysis_steps": [],
            "root_cause": "",
            "fix_recommendation": "",
            "confidence_score": 0.0,
            "requires_human_review": False,
            "iteration_count": 0,
            "messages": [],
        }
        config = {"configurable": {"thread_id": f"{system_id}-{int(time.time())}"}}
        final_state = graph.invoke(initial_state, config=config)
        return {
            "root_cause": final_state.get("root_cause", ""),
            "fix_recommendation": final_state.get("fix_recommendation", ""),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "requires_human_review": final_state.get("requires_human_review", False),
        }
    except Exception as exc:
        logger.error(f"Agent graph invocation failed: {exc}")
        return {
            "root_cause": f"Agent error: {exc}",
            "fix_recommendation": "Manual investigation required.",
            "confidence_score": 0.0,
            "requires_human_review": True,
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/report-anomaly", response_model=IncidentResponse, status_code=201, tags=["Incidents"])
def report_anomaly(payload: AnomalyReportRequest) -> IncidentResponse:
    """
    Accept an anomaly report, run the Aegis debugging agent, persist the
    incident, and return the full diagnosis result.

    - **system_id**: ID of the monitored LLM system
    - **anomaly_description**: Human-readable description of what went wrong
    - **failure_category**: One of `data_issue | prompt_issue | model_issue | system_issue | incident_issue`
    - **raw_metrics**: Key/value metric readings at time of anomaly
    """
    logger.info(f"Received anomaly report for system '{payload.system_id}'.")

    # Run agent diagnosis
    diagnosis = _run_agent_diagnosis(
        system_id=payload.system_id,
        anomaly_description=payload.anomaly_description,
        failure_category=payload.failure_category,
        raw_metrics=payload.raw_metrics,
    )

    # Determine initial fix status
    fix_status = "pending_approval" if diagnosis["requires_human_review"] else "suggested"

    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO incidents
            (system_id, anomaly_description, failure_category, raw_metrics,
             root_cause, fix_recommendation, confidence_score,
             requires_human_review, fix_status, approved, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            payload.system_id,
            payload.anomaly_description,
            payload.failure_category,
            json.dumps(payload.raw_metrics),
            diagnosis["root_cause"],
            diagnosis["fix_recommendation"],
            diagnosis["confidence_score"],
            int(diagnosis["requires_human_review"]),
            fix_status,
            now,
        ),
    )
    conn.commit()
    incident_id = cursor.lastrowid
    conn.close()

    # Audit log
    _audit_logger.log_action(
        agent_id="API",
        action="report_anomaly",
        outcome="incident_created",
        details={"incident_id": incident_id, "system_id": payload.system_id, "fix_status": fix_status},
    )

    # Return created incident
    return IncidentResponse(
        id=incident_id,
        system_id=payload.system_id,
        anomaly_description=payload.anomaly_description,
        failure_category=payload.failure_category,
        raw_metrics=payload.raw_metrics,
        root_cause=diagnosis["root_cause"],
        fix_recommendation=diagnosis["fix_recommendation"],
        confidence_score=diagnosis["confidence_score"],
        requires_human_review=diagnosis["requires_human_review"],
        fix_status=fix_status,
        approved=False,
        created_at=now,
    )


@app.get("/incidents", response_model=IncidentListResponse, tags=["Incidents"])
def list_incidents(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    system_id: Optional[str] = Query(None, description="Filter by system_id"),
    fix_status: Optional[str] = Query(None, description="Filter by fix_status"),
) -> IncidentListResponse:
    """
    Return a paginated list of all incidents, newest first.

    Optionally filter by **system_id** or **fix_status**.
    """
    conn = _get_conn()
    offset = (page - 1) * page_size

    where_clauses = []
    params: list = []
    if system_id:
        where_clauses.append("system_id = ?")
        params.append(system_id)
    if fix_status:
        where_clauses.append("fix_status = ?")
        params.append(fix_status)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    total_row = conn.execute(f"SELECT COUNT(*) FROM incidents {where_sql}", params).fetchone()
    total = total_row[0] if total_row else 0

    rows = conn.execute(
        f"SELECT * FROM incidents {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()
    conn.close()

    return IncidentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        incidents=[_row_to_incident(r) for r in rows],
    )


@app.get("/incidents/{incident_id}", response_model=IncidentResponse, tags=["Incidents"])
def get_incident(incident_id: int) -> IncidentResponse:
    """
    Retrieve a single incident by its integer **id**.

    Returns **404** if not found.
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    return _row_to_incident(row)


@app.post("/incidents/{incident_id}/approve", response_model=IncidentResponse, tags=["Incidents"])
def approve_incident(incident_id: int, payload: ApprovalRequest) -> IncidentResponse:
    """
    Approve (or reject) a pending HITL fix for an incident.

    When **approved=true** the `FixExecutor` is re-run in `semi-auto` mode
    with the approval flag set, executing the recommended fix action.
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    fix_status = row["fix_status"]
    now = datetime.utcnow().isoformat()

    if payload.approved:
        # Build fix plan from the stored recommendation
        fix_plan = {
            "action": "alert_retrain",  # default safe action; real action comes from recommendation
            "system_id": row["system_id"],
            "metric_details": json.loads(row["raw_metrics"]) if row["raw_metrics"] else {},
            "fix_recommendation": row["fix_recommendation"],
        }
        executor = FixExecutor(mode="semi-auto", audit_logger=_audit_logger)
        result = executor.execute(fix_plan, approved=True)
        new_fix_status = "success" if result.get("status") == "success" else "failed"
    else:
        new_fix_status = "suggested"  # human declined; revert to suggestion-only

    conn.execute(
        "UPDATE incidents SET approved = ?, fix_status = ?, updated_at = ? WHERE id = ?",
        (int(payload.approved), new_fix_status, now, incident_id),
    )
    conn.commit()

    # Re-fetch updated row
    updated_row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()

    _audit_logger.log_action(
        agent_id="API",
        action="approve_incident",
        outcome=new_fix_status,
        details={
            "incident_id": incident_id,
            "approved": payload.approved,
            "note": payload.approver_note,
        },
    )

    return _row_to_incident(updated_row)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Return service liveness status, uptime, and incident statistics.
    """
    conn = _get_conn()
    total_row = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()
    pending_row = conn.execute(
        "SELECT COUNT(*) FROM incidents WHERE fix_status = 'pending_approval' AND approved = 0"
    ).fetchone()
    conn.close()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 2),
        total_incidents=total_row[0] if total_row else 0,
        pending_approvals=pending_row[0] if pending_row else 0,
    )
