# ⚡ Quick Start Guide — Aegis AI Development

---

## Prerequisites

Before starting development, ensure you have:

```
✅ Python 3.11+
✅ Git
✅ Docker Desktop
✅ VS Code (recommended)
✅ API key: OpenAI OR Groq (free) OR Ollama (local)
```

---

## Environment Setup (Day 1)

### Step 1: Clone & Navigate
```bash
git clone https://github.com/Prathuish-dev/Aegis_AI.git
cd Aegis_AI
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
# OR
GROQ_API_KEY=gsk_...

# LangSmith (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=aegis-ai

# Database
LOG_DB_PATH=data/logs/events.db
CHROMA_DB_PATH=./chroma_db

# Detection Thresholds
ACCURACY_DROP_THRESHOLD=0.05
LATENCY_P95_THRESHOLD_MS=2000
ERROR_RATE_THRESHOLD=0.02
RELEVANCE_SCORE_THRESHOLD=0.5

# Self-Healing Mode
FIX_MODE=suggest  # suggest | semi-auto | auto
```

### Step 5: Initialize Knowledge Base
```bash
python scripts/init_knowledge_base.py
```

### Step 6: Run Tests
```bash
pytest tests/ -v
```

---

## First Run: Test the System

### Simulate an Anomaly
```bash
python scripts/simulate_anomaly.py --type accuracy_drop
```

### Expected Output
```
🔍 Aegis AI — Anomaly Detected
══════════════════════════════
📊 System: test-classifier-v1
⚠️  Type: Model Performance Drop
📉 Accuracy: 87% → 61% (-26%)

🔎 Retrieving knowledge base context...
✅ Found 4 relevant documents

🧠 Running LLM debugging agent...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Root Cause:       Data drift in input features
Failure Category: model_issue (data drift)
Confidence:       82%

🔧 Fix Recommendation:
  Action: Retrain model with recent data
  Steps:
    1. Run feature distribution analysis on last 7 days
    2. Identify shifted features (focus on top 5 by importance)
    3. Trigger retraining pipeline with 30-day window
    4. Monitor accuracy for 24 hours post-retrain
  Urgency: scheduled
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 References: data_drift_guide.md, incident_INC-2024-001.json
```

---

## Development Workflow

### Daily Workflow
```
1. Pull latest changes: git pull
2. Activate venv: venv\Scripts\activate
3. Run tests: pytest tests/ -v
4. Develop feature
5. Test manually: python scripts/simulate_anomaly.py
6. Commit: git add . && git commit -m "feat: ..."
7. Push: git push
```

### Testing a Specific Component
```bash
# Test detection only
python -m pytest tests/test_detection.py -v

# Test RAG retrieval
python -m pytest tests/test_rag.py -v

# Test full pipeline
python -m pytest tests/test_integration.py -v
```

---

## Project Structure (When Built)

```
Aegis_AI/
├── doc/                        ← You are here
│   ├── 01_project_overview.md
│   ├── 02_architecture_deep_dive.md
│   ├── 03_tech_stack_research.md
│   ├── 04_development_lifecycle.md
│   ├── 05_rag_system_guide.md
│   ├── 06_failure_detection_research.md
│   ├── 07_quickstart_guide.md
│   └── 08_api_reference.md
│
├── src/
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── log_collector.py      ← Collects logs from AI systems
│   │   ├── metric_tracker.py     ← Tracks numerical metrics
│   │   └── models.py             ← Pydantic data models
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── rule_engine.py        ← Threshold-based rules
│   │   ├── drift_detector.py     ← KS test, PSI drift detection
│   │   ├── relevance_scorer.py   ← LLM output quality scoring
│   │   └── anomaly_detector.py   ← Unified detector (combines all)
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py          ← Document loading + embedding
│   │   ├── retrieval.py          ← Query + retrieve from vector DB
│   │   └── evaluator.py          ← RAGAS quality evaluation
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py              ← LangGraph AgentState definition
│   │   ├── graph.py              ← LangGraph workflow definition
│   │   ├── nodes.py              ← Individual agent node functions
│   │   └── prompts.py            ← Prompt templates
│   │
│   ├── healing/
│   │   ├── __init__.py
│   │   ├── fix_executor.py       ← Applies or suggests fixes
│   │   ├── prompt_optimizer.py   ← Auto-rewrites failing prompts
│   │   └── verifier.py           ← Post-fix verification
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py               ← FastAPI application
│
├── ui/
│   └── dashboard.py              ← Streamlit dashboard
│
├── data/
│   ├── knowledge_base/
│   │   ├── llm_issues/
│   │   ├── model_issues/
│   │   ├── past_incidents/
│   │   └── system_issues/
│   └── logs/
│       └── events.db
│
├── tests/
│   ├── test_detection.py
│   ├── test_rag.py
│   ├── test_agent.py
│   └── test_integration.py
│
├── scripts/
│   ├── init_knowledge_base.py    ← Populate ChromaDB from docs
│   └── simulate_anomaly.py       ← Send fake anomalies for testing
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── config/
│   └── settings.yaml
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Key Commands Reference

| Task | Command |
|---|---|
| Install deps | `pip install -r requirements.txt` |
| Build knowledge base | `python scripts/init_knowledge_base.py` |
| Run all tests | `pytest tests/ -v` |
| Test drift detection | `python -c "from src.detection import DriftDetector; ..."` |
| Start API server | `uvicorn src.api.main:app --reload` |
| Start UI | `streamlit run ui/dashboard.py` |
| Run Docker | `docker-compose up --build` |
| Simulate anomaly | `python scripts/simulate_anomaly.py --type [type]` |

### Anomaly Types for Simulation
```bash
python scripts/simulate_anomaly.py --type accuracy_drop
python scripts/simulate_anomaly.py --type llm_hallucination
python scripts/simulate_anomaly.py --type data_drift
python scripts/simulate_anomaly.py --type api_timeout
python scripts/simulate_anomaly.py --type prompt_failure
```

---

## Common Issues & Fixes

### Issue: ChromaDB "Collection already exists"
```python
# Fix: Use get_or_create instead of create
collection = client.get_or_create_collection(name="aegis_knowledge")
```

### Issue: OpenAI rate limits during development
```
Fix: Use Groq API instead (free, fast)
pip install groq
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")
```

### Issue: FAISS not finding similar documents
```
Fix: Verify your embeddings are normalized before indexing
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### Issue: LangGraph state not persisting between nodes
```python
# Fix: Use checkpointer
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
# Always pass thread_id for persistence
config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke(state, config=config)
```

---

## Learning Resources

### Must-Read Documentation
| Resource | URL | Priority |
|---|---|---|
| LangGraph Docs | https://langchain-ai.github.io/langgraph/ | 🔴 Critical |
| LangChain Docs | https://python.langchain.com/docs/ | 🔴 Critical |
| ChromaDB Docs | https://docs.trychroma.com/ | 🟡 Important |
| RAGAS Docs | https://docs.ragas.io/ | 🟡 Important |
| Evidently AI | https://docs.evidentlyai.com/ | 🟢 Useful |
| FastAPI Docs | https://fastapi.tiangolo.com/ | 🟡 Important |
| Streamlit Docs | https://docs.streamlit.io/ | 🟢 Useful |

### Recommended Tutorials
1. **LangGraph Quickstart** — build your first agentic workflow
2. **RAG from Scratch** (LangChain YouTube series)
3. **Evidently AI crash course** — drift detection in 30 minutes
4. **RAGAS tutorial** — evaluating RAG quality

### Papers Worth Reading
- "Corrective RAG (CRAG)" — adaptive retrieval correction
- "Self-RAG" — LLM decides when to retrieve
- "RAGAS: Automated Evaluation of RAG" — evaluation framework
- "Monitoring ML Models in Production" — MLOps fundamentals
