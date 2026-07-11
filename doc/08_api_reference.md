# Aegis AI — API Reference

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) | `http://localhost:8000/redoc`

---

## Overview

The Aegis AI REST API exposes the self-healing pipeline as a set of JSON endpoints. It accepts anomaly reports, runs the LangGraph debugging agent, persists incidents in SQLite, and allows human reviewers to approve or decline recommended fixes.

---

## Authentication

No authentication is required in the default configuration. For production deployments, add an API gateway / reverse proxy with token-based authentication in front of the service.

---

## Endpoints

### 1. `POST /report-anomaly`

Submit a new anomaly report. The agent runs synchronously and returns the full diagnosis.

**Request Body**

```json
{
  "system_id":           "string (required) — ID of the monitored system",
  "anomaly_description": "string (required) — what went wrong",
  "failure_category":    "string (required) — one of: data_issue | prompt_issue | model_issue | system_issue | incident_issue",
  "raw_metrics":         "object (optional) — key/value metric readings"
}
```

**Example Request**

```bash
curl -X POST http://localhost:8000/report-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "llm-prod-01",
    "anomaly_description": "Model accuracy dropped below 0.75 threshold",
    "failure_category": "model_issue",
    "raw_metrics": {"accuracy": 0.71, "latency_ms": 520}
  }'
```

**Response** — `201 Created`

```json
{
  "id": 1,
  "system_id": "llm-prod-01",
  "anomaly_description": "Model accuracy dropped below 0.75 threshold",
  "failure_category": "model_issue",
  "raw_metrics": {"accuracy": 0.71, "latency_ms": 520},
  "root_cause": "Accuracy degradation detected in model output layer...",
  "fix_recommendation": "Trigger retraining pipeline with fresh data...",
  "confidence_score": 0.82,
  "requires_human_review": false,
  "fix_status": "suggested",
  "approved": false,
  "created_at": "2026-07-11T15:00:00",
  "updated_at": null
}
```

**Error Responses**

| Status | Reason |
|--------|--------|
| `422 Unprocessable Entity` | Missing required fields or invalid types |

---

### 2. `GET /incidents`

Return a paginated list of incidents, newest first.

**Query Parameters**

| Parameter   | Type    | Default | Description |
|-------------|---------|---------|-------------|
| `page`      | integer | `1`     | Page number (1-indexed) |
| `page_size` | integer | `20`    | Items per page (max 100) |
| `system_id` | string  | –       | Filter by system ID |
| `fix_status`| string  | –       | Filter by status: `pending`, `pending_approval`, `suggested`, `success`, `failed` |

**Example Request**

```bash
# All incidents
curl http://localhost:8000/incidents

# Filter by system + status
curl "http://localhost:8000/incidents?system_id=llm-prod-01&fix_status=pending_approval&page=1&page_size=10"
```

**Response** — `200 OK`

```json
{
  "total": 42,
  "page": 1,
  "page_size": 10,
  "incidents": [ /* array of IncidentResponse objects */ ]
}
```

---

### 3. `GET /incidents/{id}`

Retrieve a single incident by its integer ID.

**Path Parameter**

| Parameter | Type    | Description |
|-----------|---------|-------------|
| `id`      | integer | Incident ID |

**Example Request**

```bash
curl http://localhost:8000/incidents/1
```

**Response** — `200 OK` — Full `IncidentResponse` object (same schema as above).

**Error Responses**

| Status | Reason |
|--------|--------|
| `404 Not Found` | No incident with the given ID exists |

---

### 4. `POST /incidents/{id}/approve`

Approve or decline a pending HITL fix for an incident.

When `approved = true`, the `FixExecutor` is called in `semi-auto` mode with the approval flag, executing the recommended fix action and logging the outcome to the audit trail.

**Path Parameter**

| Parameter | Type    | Description |
|-----------|---------|-------------|
| `id`      | integer | Incident ID |

**Request Body**

```json
{
  "approved":      "boolean (required)",
  "approver_note": "string (optional) — reason for approval/rejection"
}
```

**Example Request**

```bash
# Approve
curl -X POST http://localhost:8000/incidents/1/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "approver_note": "Reviewed and confirmed root cause."}'

# Decline
curl -X POST http://localhost:8000/incidents/1/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": false, "approver_note": "Needs more investigation."}'
```

**Response** — `200 OK` — Updated `IncidentResponse` with new `fix_status` and `approved` fields.

**Fix Status Transitions**

| Approval | Resulting `fix_status` |
|----------|------------------------|
| `true`   | `success` or `failed` (from executor outcome) |
| `false`  | `suggested` (reverted to suggestion-only) |

**Error Responses**

| Status | Reason |
|--------|--------|
| `404 Not Found` | No incident with the given ID exists |

---

### 5. `GET /health`

Return service liveness status, version, uptime, and incident statistics.

**Example Request**

```bash
curl http://localhost:8000/health
```

**Response** — `200 OK`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3720.5,
  "total_incidents": 42,
  "pending_approvals": 3
}
```

---

## Data Schemas

### `IncidentResponse`

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Auto-incremented incident ID |
| `system_id` | string | Monitored system identifier |
| `anomaly_description` | string | Description of the anomaly |
| `failure_category` | string | Category of failure |
| `raw_metrics` | object | Raw metric readings at anomaly time |
| `root_cause` | string \| null | Agent-diagnosed root cause |
| `fix_recommendation` | string \| null | Agent-recommended corrective action |
| `confidence_score` | float \| null | Agent confidence in diagnosis (0.0–1.0) |
| `requires_human_review` | boolean | Whether HITL approval is needed |
| `fix_status` | string | `pending` \| `suggested` \| `pending_approval` \| `success` \| `failed` |
| `approved` | boolean | Whether fix was approved by a human |
| `created_at` | string (ISO 8601) | Incident creation timestamp |
| `updated_at` | string \| null | Last update timestamp |

---

## Fix Status State Machine

```
                 [report-anomaly]
                       │
              ┌────────▼────────┐
              │    pending      │   (initial, pre-agent-run)
              └────────┬────────┘
                       │ agent run complete
          ┌────────────┴────────────┐
          │ confidence ≥ 0.75       │ confidence < 0.75
          ▼                         ▼
    [suggested]            [pending_approval]
          │                         │
          │                    human reviews
          │               ┌─────────┴──────────┐
          │           approved              declined
          │               ▼                    ▼
          └──────────► [success]          [suggested]
                           │
                      executor error
                           ▼
                        [failed]
```

---

## Interactive API Explorer

The FastAPI backend automatically serves interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json
