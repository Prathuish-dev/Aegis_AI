# Aegis AI — Deployment Guide

**Version:** 1.0.0

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Development Setup (no Docker)](#2-local-development-setup)
3. [Docker Compose Setup](#3-docker-compose-setup)
4. [Environment Variable Reference](#4-environment-variable-reference)
5. [Verifying the Deployment](#5-verifying-the-deployment)
6. [Common Issues & Troubleshooting](#6-common-issues--troubleshooting)
7. [Production Considerations](#7-production-considerations)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.11+ | Use `python --version` to check |
| pip | 23+ | `pip install --upgrade pip` |
| Docker | 24+ | For containerised deployment |
| Docker Compose | 2.x (`compose` plugin) | Comes bundled with Docker Desktop |
| Git | 2.x | For cloning the repo |

**Optional but recommended:**
- An OpenAI API key for live LLM agent diagnosis (the system works in stub mode without one)

---

## 2. Local Development Setup

### 2.1 Clone the Repository

```bash
git clone https://github.com/Prathuish-dev/Aegis_AI.git
cd Aegis_AI
```

### 2.2 Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install uvicorn[standard] fastapi streamlit
```

### 2.4 Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY and other values
```

### 2.5 Run the FastAPI Backend

```bash
# From the project root
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### 2.6 Run the Streamlit Dashboard

Open a second terminal (with the virtual env activated):

```bash
streamlit run ui/dashboard.py --server.port 8501
```

The dashboard will open at: http://localhost:8501

### 2.7 Run the Test Suite

```bash
# All tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_api.py -v
```

---

## 3. Docker Compose Setup

This is the **recommended** way to run Aegis AI in a production-like environment.

### 3.1 Configure Environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

Key variables to set in `.env`:

```dotenv
OPENAI_API_KEY=sk-...your-key...
FIX_MODE=suggest          # or semi-auto, auto
LOG_DB_PATH=data/logs/events.db
CHROMA_PERSIST_DIR=data/chroma_db
```

### 3.2 Build and Start All Services

```bash
docker compose up --build
```

Or to run in the background (detached mode):

```bash
docker compose up --build -d
```

### 3.3 Verify Services Are Running

```bash
docker compose ps
```

Expected output:
```
NAME         STATUS           PORTS
aegis-api    Up (healthy)     0.0.0.0:8000->8000/tcp
aegis-ui     Up               0.0.0.0:8501->8501/tcp
```

### 3.4 Access the Services

| Service | URL |
|---------|-----|
| FastAPI Backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

### 3.5 View Logs

```bash
# All services
docker compose logs -f

# API only
docker compose logs -f aegis-api

# UI only
docker compose logs -f aegis-ui
```

### 3.6 Stop Services

```bash
docker compose down
```

To also remove named volumes (⚠️ deletes all stored incidents and ChromaDB data):

```bash
docker compose down -v
```

---

## 4. Environment Variable Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OPENAI_API_KEY` | – | Optional* | OpenAI API key for LLM diagnosis. Without it, the API uses a rule-based stub. |
| `LOG_DB_PATH` | `data/logs/events.db` | No | Path to the SQLite database file |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | No | Directory for ChromaDB vector store persistence |
| `FIX_MODE` | `suggest` | No | Healing mode: `suggest`, `semi-auto`, or `auto` |
| `API_BASE_URL` | `http://localhost:8000` | No | URL the Streamlit dashboard uses to reach the API (use `http://aegis-api:8000` in Docker) |
| `GROQ_API_KEY` | – | Optional | Groq API key (alternative to OpenAI) |
| `LANGCHAIN_API_KEY` | – | Optional | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | `false` | No | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | `aegis-ai` | No | LangSmith project name |

> **\* Stub Mode:** When `OPENAI_API_KEY` is not set or is set to a placeholder value, the `/report-anomaly` endpoint returns a deterministic stub diagnosis. All other API functionality works normally. This is useful for UI development and testing.

---

## 5. Verifying the Deployment

### 5.1 Health Check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "ok", "version": "1.0.0", "uptime_seconds": 12.3, "total_incidents": 0, "pending_approvals": 0}
```

### 5.2 Submit a Test Anomaly

```bash
curl -X POST http://localhost:8000/report-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "test-system",
    "anomaly_description": "Accuracy dropped below threshold",
    "failure_category": "model_issue",
    "raw_metrics": {"accuracy": 0.72, "latency_ms": 450}
  }'
```

Expected: `201 Created` with a diagnosis in the response body.

### 5.3 Verify Incident in Dashboard

1. Open http://localhost:8501
2. Navigate to **🏠 Live Incident Feed**
3. The incident submitted above should appear in the table.

### 5.4 Test HITL Approval

1. In the dashboard, navigate to **✅ HITL Approval Panel**
2. If the submitted incident's confidence score < 0.75, it will appear here
3. Click **✅ Approve** or **❌ Decline**

---

## 6. Common Issues & Troubleshooting

### API fails to start — `ModuleNotFoundError`

**Cause:** `PYTHONPATH` is not set to the project root.

**Fix (local dev):**
```bash
# Windows PowerShell
$env:PYTHONPATH = "."
# macOS / Linux
export PYTHONPATH=.
```

Or run with:
```bash
python -m uvicorn src.api.main:app --reload
```

---

### Streamlit shows "Cannot connect to Aegis API"

**Cause:** The FastAPI backend is not running or `API_BASE_URL` is wrong.

**Fix:**
1. Confirm the API is running: `curl http://localhost:8000/health`
2. Check `API_BASE_URL` in your `.env` file matches the actual API address.
3. In Docker Compose, the UI container uses `http://aegis-api:8000` (the internal Docker network name). Do not set `API_BASE_URL=http://localhost:8000` inside Docker.

---

### Port conflicts

**Symptoms:** `Error: bind: address already in use`

**Fix:**
```bash
# Find what is using port 8000
# Windows
netstat -ano | findstr :8000
# macOS/Linux
lsof -i :8000

# Change ports in docker-compose.yml if needed
ports:
  - "8001:8000"   # host_port:container_port
```

---

### Docker build fails — pip install timeout

**Cause:** Slow network or large dependencies (sentence-transformers, torch).

**Fix:**
```bash
# Increase Docker build timeout
DOCKER_BUILDKIT=1 docker compose build --progress=plain
```

---

### ChromaDB not persisting between restarts

**Cause:** The `chroma_data` volume was deleted or not mounted.

**Fix:**
```bash
# List volumes
docker volume ls | grep aegis

# Re-create with volume
docker compose up --build
```

---

### OpenAI API returns 401 Unauthorized

**Cause:** Invalid or expired `OPENAI_API_KEY`.

**Fix:**
1. Generate a new key at https://platform.openai.com/api-keys
2. Update `.env` and restart: `docker compose restart aegis-api`

---

## 7. Production Considerations

| Concern | Recommendation |
|---------|---------------|
| **Authentication** | Add OAuth2 / API token middleware to FastAPI |
| **Database** | Migrate from SQLite to PostgreSQL for concurrent write loads |
| **Secrets** | Use Docker secrets or a vault (e.g. HashiCorp Vault, AWS Secrets Manager) instead of `.env` files |
| **TLS/HTTPS** | Deploy behind a reverse proxy (nginx, Traefik) with a TLS certificate |
| **Scaling** | Run multiple `aegis-api` replicas behind a load balancer (stateless except for SQLite — move to Postgres first) |
| **Monitoring** | Enable `LANGCHAIN_TRACING_V2=true` for LangSmith observability; add Prometheus metrics via `prometheus-fastapi-instrumentator` |
| **Log rotation** | Mount SQLite data volume and configure OS-level log rotation on the container host |
