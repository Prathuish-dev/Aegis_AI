# 🛡️ Aegis AI — v1.0.0

### *Autonomous Self-Healing System for ML & LLM Pipelines*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-black?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)
[![Tests](https://img.shields.io/badge/Tests-121%20passed-2ECC71?logo=pytest)](./tests)

---

## 🚀 Overview

**Aegis AI** is a production-grade autonomous self-healing system that monitors, diagnoses, and repairs LLM and ML pipelines in real time — without human intervention (or with a human-in-the-loop when confidence is low).

> 💡 Think of it as an **AI Engineer in software form** — always watching, always learning, always fixing.

---

## 🔥 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Debugging Agent** | LangGraph state machine with Chain-of-Thought reasoning |
| 📚 **RAG Knowledge Engine** | ChromaDB + RAGAS-evaluated retrieval over 10+ knowledge docs |
| 🔍 **Anomaly Detection** | Detects drift (PSI), accuracy drops, latency spikes, error rate surges |
| 🔧 **Self-Healing Engine** | FixExecutor with `suggest` / `semi-auto` / `auto` modes |
| 🤖 **LLM-as-a-Judge** | Faithfulness evaluation without human labels |
| ✅ **HITL Approval** | Human-in-the-loop approval panel with audit trail |
| 🌐 **REST API** | FastAPI with 5 endpoints + Swagger UI |
| 🖥️ **Live Dashboard** | Streamlit dashboard with auto-refresh, incident feed, HITL panel |
| 🐳 **Docker Ready** | Multi-container Compose deployment with persistent volumes |

---

## 🏗️ Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                  Monitored LLM System                    │
  └────────────────────────┬─────────────────────────────────┘
                           │ Metrics / Logs
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Phase 2 — Anomaly Detection Layer                       │
  │  MetricTracker · DriftDetector · LogCollector            │
  └────────────────────────┬─────────────────────────────────┘
                           │ AnomalyEvent
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Phase 3 — RAG Knowledge Engine                          │
  │  ChromaDB · KnowledgeRetriever · RAGEvaluator (RAGAS)    │
  └────────────────────────┬─────────────────────────────────┘
                           │ Retrieved Context
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Phase 4 — LangGraph Debugging Agent                     │
  │  CoT Reasoning · Root Cause · Fix Recommendation         │
  └────────────────────────┬─────────────────────────────────┘
                           │ Diagnosis (confidence ≥ 0.75 → auto)
             ┌─────────────┴────────────┐
             │                          │ confidence < 0.75
             ▼                          ▼
  ┌──────────────────┐     ┌────────────────────────────────┐
  │  Phase 5 —       │     │  HITL Approval Panel           │
  │  Self-Healing    │     │  (Streamlit Dashboard)         │
  │  FixExecutor     │     └───────────────┬────────────────┘
  │  Verifier        │                     │ approved
  │  AuditLogger     │◄────────────────────┘
  └──────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Phase 6 — API & Dashboard                               │
  │  FastAPI (port 8000) · Streamlit (port 8501)             │
  └──────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start (Docker Compose — Recommended)

```bash
# 1. Clone
git clone https://github.com/Prathuish-dev/Aegis_AI.git
cd Aegis_AI

# 2. Configure
cp .env.example .env
# Set OPENAI_API_KEY in .env (optional — works in stub mode without it)

# 3. Launch
docker compose up --build

# 4. Open
#    API:       http://localhost:8000
#    Docs:      http://localhost:8000/docs
#    Dashboard: http://localhost:8501
```

---

## 🛠️ Local Development Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install uvicorn[standard] fastapi streamlit

# Run API
uvicorn src.api.main:app --reload --port 8000

# Run Dashboard (new terminal)
streamlit run ui/dashboard.py

# Run tests
pytest --cov=src --cov-report=term-missing
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/report-anomaly` | Submit anomaly → get AI diagnosis |
| `GET` | `/incidents` | List all incidents (paginated, filterable) |
| `GET` | `/incidents/{id}` | Get single incident detail |
| `POST` | `/incidents/{id}/approve` | Approve/decline HITL fix |
| `GET` | `/health` | Service liveness + stats |

**Test the API:**
```bash
curl -X POST http://localhost:8000/report-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": "llm-prod-01",
    "anomaly_description": "Accuracy dropped below 0.75",
    "failure_category": "model_issue",
    "raw_metrics": {"accuracy": 0.71}
  }'
```

Full API reference: [`doc/08_api_reference.md`](doc/08_api_reference.md)

---

## 📁 Project Structure

```
Aegis_AI/
├── src/
│   ├── api/              # FastAPI backend (main.py, models.py)
│   ├── agent/            # LangGraph agent (graph.py, state.py, prompts.py, nodes.py)
│   ├── detection/        # Anomaly detectors (drift, retrieval quality)
│   ├── healing/          # Self-healing engine (fix_executor, verifier, audit_logger)
│   ├── monitoring/       # MetricTracker, LogCollector, models
│   └── rag/              # RAG retriever + RAGAS evaluator
├── ui/
│   └── dashboard.py      # Streamlit dashboard (5 pages)
├── tests/                # 121 tests across all modules
├── data/
│   ├── knowledge_base/   # 10+ troubleshooting guides
│   └── knowledge_base/past_incidents/  # Mock incident records
├── config/
│   └── settings.yaml     # Detection thresholds, DB path
├── doc/                  # Architecture, API reference, deployment guide
├── scripts/              # Spot-check and simulation scripts
├── Dockerfile            # FastAPI backend image
├── Dockerfile.streamlit  # Streamlit UI image
├── docker-compose.yml    # Multi-container deployment
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

---

## 🧪 Test Coverage

```
121 tests — 0 failures

Module                                   Coverage
─────────────────────────────────────────────────
src/api/main.py                            92%
src/api/models.py                         100%
src/healing/fix_executor.py                88%
src/healing/audit_logger.py                95%
src/healing/verifier.py                    86%
src/healing/llm_quality_detector.py        84%
src/detection/retrieval_quality_detector.py 90%
src/monitoring/metric_tracker.py           89%
src/rag/evaluator.py                       82%
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11 |
| **LLM** | OpenAI GPT-4o-mini / Groq / Ollama |
| **Agent** | LangGraph + LangChain |
| **RAG** | ChromaDB + sentence-transformers + RAGAS |
| **API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit |
| **Storage** | SQLite (incidents + audit log) + ChromaDB (vectors) |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest + pytest-cov |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [`doc/08_api_reference.md`](doc/08_api_reference.md) | Full REST API reference with curl examples |
| [`doc/09_deployment_guide.md`](doc/09_deployment_guide.md) | Setup, Docker Compose, env vars, troubleshooting |
| [`TASK_SHEET.md`](TASK_SHEET.md) | Project task breakdown and progress tracking |

---

## 🧑‍💻 Author

**PRATHUISH SANJEEVAN**  
AI Engineer | Machine Learning | LLM Systems

---

## ⭐ If you find this interesting

Give it a star ⭐ and feel free to contribute!
