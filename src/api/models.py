"""
Pydantic request/response schemas for the Aegis AI FastAPI backend.
All models use strict typing and include field-level documentation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnomalyReportRequest(BaseModel):
    """Payload for POST /report-anomaly."""

    system_id: str = Field(..., description="Unique identifier of the monitored system.")
    anomaly_description: str = Field(..., description="Human-readable description of the anomaly.")
    failure_category: str = Field(
        ...,
        description="Category of the failure: data_issue | prompt_issue | model_issue | system_issue | incident_issue",
    )
    raw_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw metric readings at the time of the anomaly.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "system_id": "llm-prod-01",
            "anomaly_description": "Model accuracy dropped below 0.75 threshold",
            "failure_category": "model_issue",
            "raw_metrics": {"accuracy": 0.71, "latency_ms": 520},
        }
    }}


class ApprovalRequest(BaseModel):
    """Payload for POST /incidents/{id}/approve."""

    approved: bool = Field(..., description="Whether the fix is approved for execution.")
    approver_note: Optional[str] = Field(None, description="Optional note from the human reviewer.")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IncidentResponse(BaseModel):
    """Single incident record returned by the API."""

    id: int = Field(..., description="Auto-incremented incident ID.")
    system_id: str
    anomaly_description: str
    failure_category: str
    raw_metrics: Dict[str, Any]

    # Agent diagnosis outputs
    root_cause: Optional[str] = None
    fix_recommendation: Optional[str] = None
    confidence_score: Optional[float] = None
    requires_human_review: bool = False

    # Fix execution tracking
    fix_status: str = Field("pending", description="pending | suggested | pending_approval | success | failed")
    approved: bool = False

    # Timestamps
    created_at: str
    updated_at: Optional[str] = None


class IncidentListResponse(BaseModel):
    """Paginated list of incidents."""

    total: int
    page: int
    page_size: int
    incidents: List[IncidentResponse]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: float
    total_incidents: int
    pending_approvals: int
