# 🗓️ Development Lifecycle — Aegis AI

> A complete, phase-by-phase guide to building Aegis AI from scratch to production.

---

## Overview: Agentic AI SDLC vs Traditional SDLC

| Aspect | Traditional SDLC | Agentic AI SDLC (Aegis AI) |
|---|---|---|
| **Driver** | Human-coded logic | LLM-orchestrated reasoning |
| **Primary Output** | Static code | Adaptive behaviors + self-improving system |
| **Testing** | Unit + integration tests | EvalOps + adversarial testing + LLM evals |
| **Maintenance** | Periodic updates | Continuous self-optimization |
| **Failure Mode** | Explicit errors | Silent failures + model drift |

---

## 📅 Complete Development Roadmap

### Timeline Summary

```
Week 1-2:   Phase 1 - Foundation & Core Monitoring
Week 3-4:   Phase 2 - Anomaly Detection Engine
Week 5-6:   Phase 3 - RAG Knowledge Engine
Week 7-8:   Phase 4 - LLM Debugging Agent
Week 9-10:  Phase 5 - Self-Healing Engine
Week 11-12: Phase 6 - UI, Docker, Polish
```

---

## Phase 1: Foundation Setup (Week 1-2)

### Goals
- Set up project structure
- Build log ingestion system
- Create data models
- Establish config management

### Tasks

#### 1.1 Project Scaffolding
```bash
Aegis_AI/
├── src/
│   ├── __init__.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── log_collector.py
│   │   ├── metric_tracker.py
│   │   └── models.py
│   ├── detection/
│   ├── rag/
│   ├── agent/
│   ├── healing/
│   └── api/                  ← FastAPI backend (Phase 6)
│       ├── __init__.py
│       └── main.py
├── data/
│   ├── knowledge_base/
│   └── logs/
├── tests/
├── config/
│   └── settings.yaml
├── requirements.txt
├── .env.example
└── docker/
```

#### 1.2 Data Models (Pydantic)
```python
# src/monitoring/models.py
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    LLM_RELEVANCE = "llm_relevance"

class AnomalyEvent(BaseModel):
    event_id: str
    system_id: str
    metric_type: MetricType
    current_value: float
    baseline_value: float
    timestamp: datetime
    raw_context: dict
    severity: str  # "low", "medium", "high", "critical"
```

#### 1.3 Log Ingestion
```python
# src/monitoring/log_collector.py
import json
import sqlite3
import os
from pathlib import Path

class LogCollector:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("LOG_DB_PATH", "data/logs/events.db")
        # Ensure parent directories exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the events table if it does not exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                system_id TEXT,
                metric_type TEXT,
                log_entry TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def ingest_log(self, log_entry: dict) -> None:
        """Accept a structured log and persist it."""
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?)",
            (log_entry["event_id"], log_entry["system_id"],
             log_entry["metric_type"], json.dumps(log_entry), 
             log_entry["timestamp"])
        )
        self.conn.commit()
```

### Deliverables
- [ ] Project structure created
- [ ] Pydantic models for all data types
- [ ] SQLite log storage working
- [ ] Config loading from `.env` and `settings.yaml`
- [ ] Sample log generator (for testing)

---

## Phase 2: Anomaly Detection Engine (Week 3-4)

### Goals
- Build rule-based detection
- Implement statistical drift detection
- Create failure classifier

### Tasks

#### 2.1 Rule Engine
```python
# src/detection/rule_engine.py
from dataclasses import dataclass

@dataclass
class DetectionRule:
    name: str
    metric_type: str
    operator: str       # "lt", "gt", "delta_pct"
    threshold: float
    severity: str

DEFAULT_RULES = [
    DetectionRule("accuracy_drop", "accuracy", "lt", 0.75, "high"),
    DetectionRule("high_latency", "latency_p95", "gt", 2000, "medium"),
    DetectionRule("error_spike", "error_rate", "gt", 0.05, "critical"),
]

class RuleEngine:
    def __init__(self, rules=DEFAULT_RULES):
        self.rules = rules

    def evaluate(self, metric_type, value) -> list[DetectionRule]:
        """Return all triggered rules for a given metric."""
        triggered = []
        for rule in self.rules:
            if rule.metric_type == metric_type:
                if self._check(rule, value):
                    triggered.append(rule)
        return triggered

    def _check(self, rule: DetectionRule, value: float) -> bool:
        """Evaluate a single rule against the observed metric value."""
        if rule.operator == "lt":
            return value < rule.threshold
        elif rule.operator == "gt":
            return value > rule.threshold
        elif rule.operator == "delta_pct":
            # threshold is the allowed % change (negative = drop)
            return value < rule.threshold
        return False
```

#### 2.2 Statistical Drift Detector
```python
# src/detection/drift_detector.py
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def detect_feature_drift(self, reference: np.ndarray, 
                              production: np.ndarray) -> dict:
        """KS test for continuous feature drift."""
        ks_stat, p_value = stats.ks_2samp(reference, production)
        return {
            "drift_detected": p_value < self.alpha,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "severity": self._severity(p_value)
        }

    def compute_psi(self, reference, production, bins=10) -> dict:
        """Population Stability Index for distribution shift.

        IMPORTANT — two correctness requirements:
        1. Bin edges must be computed on the reference data and then
           REUSED for the production data. Computing edges independently
           on each dataset produces incomparable bins (different ranges).
        2. We need the fraction of samples per bin (sums to 1.0),
           NOT the probability density (density=True makes the integral
           equal 1, not the sum — incorrect for PSI).
        """
        # Step 1: derive bin edges from the reference distribution
        ref_counts, bin_edges = np.histogram(reference, bins=bins)
        # Step 2: apply the SAME bin edges to production data
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # Step 3: convert counts to proportions (sum = 1.0)
        ref_pct = ref_counts / len(reference)
        prod_pct = prod_counts / len(production)

        # Step 4: avoid log(0) with a small epsilon
        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        prod_pct = np.where(prod_pct == 0, 1e-4, prod_pct)

        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

        return {
            "psi": float(psi),
            "severity": "none" if psi < 0.1 else "moderate" if psi < 0.25 else "severe"
        }
```

#### 2.3 Semantic Relevance Scorer (For LLMs)
```python
# src/detection/relevance_scorer.py
from sentence_transformers import SentenceTransformer, util

class RelevanceScorer:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, query: str, response: str) -> float:
        """Return semantic similarity between query and LLM response."""
        embeddings = self.model.encode([query, response])
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def is_hallucinating(self, query: str, response: str, 
                          threshold=0.5) -> bool:
        return self.score(query, response) < threshold
```

### Deliverables
- [ ] Rule engine with configurable rules
- [ ] KS test and PSI drift detection
- [ ] Semantic relevance scorer
- [ ] Unified `AnomalyDetector` class that combines all three
- [ ] Unit tests for all detectors

---

## Phase 3: RAG Knowledge Engine (Week 5-6)

### Goals
- Build vector knowledge base
- Implement document ingestion pipeline
- Create retrieval chain

### Tasks

#### 3.1 Knowledge Base Population
```
data/knowledge_base/
├── llm_issues/
│   ├── hallucination_guide.md
│   ├── prompt_engineering_rules.md
│   └── context_injection_patterns.md
├── model_issues/
│   ├── accuracy_drop_playbook.md
│   ├── data_drift_guide.md
│   └── retraining_checklist.md
├── past_incidents/
│   ├── incident_001.json    ← error + fix pair
│   └── incident_002.json
└── system_issues/
    ├── api_failure_guide.md
    └── timeout_handling.md
```

#### 3.2 Document Ingestion
```python
# src/rag/ingestion.py
#
# ⚠️  EMBEDDING MODEL CONSISTENCY RULE:
# The embedding model used here during ingestion MUST be identical to
# the one used in AegisRAG (retrieval.py). Mixing models (e.g., ingesting
# with Chroma's default all-MiniLM-L6-v2 and querying with OpenAI embeddings)
# produces vectors in incompatible spaces, making retrieval completely
# meaningless. Both classes accept the same `embedding_function` argument.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma

class KnowledgeBaseIngester:
    def __init__(self, embedding_function, db_path="./chroma_db"):
        # Pass the SAME embedding_function instance that AegisRAG will use.
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def ingest_directory(self, path: str, category: str):
        """Load, embed and persist all documents from a directory."""
        loader = DirectoryLoader(path, glob="**/*.md")
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        # Use the LangChain Chroma wrapper so the same embedding_function
        # is applied consistently for both ingestion and retrieval.
        vectorstore = Chroma(
            collection_name="aegis_knowledge",
            embedding_function=self.embedding_function,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # Attach category metadata to every chunk for filtered retrieval
        for chunk in chunks:
            chunk.metadata["category"] = category

        vectorstore.add_documents(chunks)
        print(f"✅ Ingested {len(chunks)} chunks from {path}")
```

#### 3.3 RAG Retrieval Chain
```python
# src/rag/retrieval.py
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class AegisRAG:
    def __init__(self, embedding_model, llm, db_path="./chroma_db"):
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",          # Maximal Marginal Relevance
            search_kwargs={"k": 5, "fetch_k": 20}
        )

    def retrieve_context(self, query: str, category_filter: str = None) -> list:
        """Retrieve relevant documents for a given anomaly description."""
        if category_filter:
            docs = self.retriever.invoke(
                query,
                filter={"category": category_filter}
            )
        else:
            docs = self.retriever.invoke(query)
        return docs
```

### Deliverables
- [ ] Knowledge base populated with 20+ documents
- [ ] ChromaDB ingestion pipeline working
- [ ] Retrieval with MMR (diversity-aware) search
- [ ] Category-based filtering
- [ ] Test retrieval quality manually

---

## Phase 4: LLM Debugging Agent (Week 7-8)

### Goals
- Build LangGraph agent
- Implement CoT reasoning
- Create structured output
- Add confidence scoring

### Tasks

#### 4.1 Agent State Definition
```python
# src/agent/state.py
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AegisAgentState(TypedDict):
    # Input
    anomaly_description: str
    system_id: str
    failure_category: str
    raw_metrics: dict

    # Processing
    retrieved_context: list[str]
    analysis_steps: list[str]

    # Output
    root_cause: str
    fix_recommendation: str
    confidence_score: float
    requires_human_review: bool

    # Control flow
    iteration_count: int
    messages: Annotated[list[BaseMessage], operator.add]
```

#### 4.2 Agent Graph
```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_debugging_agent(llm, rag):
    graph = StateGraph(AegisAgentState)

    # --- Add all nodes ---
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("analyze_anomaly", analyze_anomaly_node)
    graph.add_node("classify_failure", classify_failure_node)
    graph.add_node("generate_fix", generate_fix_node)
    graph.add_node("human_review", human_review_node)
    # ✅ apply_fix must be a declared node so edges into it are valid
    graph.add_node("apply_fix", apply_fix_node)

    # --- Define edges (control flow) ---
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "analyze_anomaly")
    graph.add_edge("analyze_anomaly", "classify_failure")
    graph.add_edge("classify_failure", "generate_fix")

    # Route to human_review if confidence is low; otherwise skip to apply_fix
    graph.add_conditional_edges(
        "generate_fix",
        route_after_fix,   # returns "human_review" or "apply_fix"
        {
            "human_review": "human_review",
            "apply_fix": "apply_fix",
        }
    )

    # ✅ human_review must have an outgoing edge; without it the graph
    # is a dead end — once a human reviews, execution cannot continue.
    graph.add_edge("human_review", "apply_fix")
    graph.add_edge("apply_fix", END)

    checkpointer = MemorySaver()
    # interrupt_before belongs on compile(), NOT on add_node()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]   # pauses here for human input
    )
```

### Deliverables
- [ ] LangGraph state machine working end-to-end
- [ ] Structured JSON output from agent
- [ ] Confidence score calculation
- [ ] Human-in-the-loop interrupt working
- [ ] Integration tests with mock anomalies

---

## Phase 5: Self-Healing Engine (Week 9-10)

### Goals
- Implement fix application logic
- Build prompt optimizer
- Create post-fix verification

### Tasks

#### 5.1 Fix Executor
```python
# src/healing/fix_executor.py

class FixExecutor:
    def __init__(self, mode="suggest"):
        """
        mode: 'suggest' | 'semi-auto' | 'auto'
        """
        self.mode = mode

    def execute(self, fix_plan: dict, approved: bool = False) -> dict:
        if self.mode == "suggest":
            return self._format_suggestion(fix_plan)
        elif self.mode == "semi-auto" and approved:
            return self._apply_fix(fix_plan)
        elif self.mode == "auto":
            return self._apply_fix(fix_plan)
        else:
            return {"status": "pending_approval", "plan": fix_plan}

    def _apply_fix(self, fix_plan: dict) -> dict:
        action = fix_plan.get("action")
        if action == "rewrite_prompt":
            return self._rewrite_prompt(fix_plan)
        elif action == "update_config":
            return self._update_config(fix_plan)
        elif action == "alert_retrain":
            return self._send_retrain_alert(fix_plan)
```

#### 5.2 Prompt Auto-Optimizer
```python
# src/healing/prompt_optimizer.py

class PromptOptimizer:
    def __init__(self, llm):
        self.llm = llm

    def optimize(self, original_prompt: str, 
                 failure_reason: str) -> str:
        """Rewrite a failing prompt based on failure analysis."""
        optimization_prompt = f"""
        ORIGINAL PROMPT (failing): {original_prompt}
        FAILURE REASON: {failure_reason}

        Rewrite this prompt to:
        1. Be more specific and clear
        2. Include output format instructions
        3. Add guardrails against the identified failure
        4. Keep similar intent

        Return ONLY the new prompt, nothing else.
        """
        return self.llm.invoke(optimization_prompt).content
```

### Deliverables
- [ ] Fix executor with three modes (suggest/semi-auto/auto)
- [ ] Prompt optimizer working
- [ ] Post-fix monitoring trigger (to verify fix worked)
- [ ] Audit log of all actions taken

---

## Phase 6: UI, API, Docker & Polish (Week 11-12)

### Goals
- Build Streamlit dashboard
- Create FastAPI endpoints
- Dockerize everything
- Write documentation

### Streamlit Dashboard Features
- [ ] Live incident feed
- [ ] Incident detail view (anomaly + diagnosis + fix)
- [ ] Manual trigger for diagnosis
- [ ] Human approval interface for pending fixes
- [ ] System health metrics overview

### FastAPI Endpoints
```
POST /api/report-anomaly       ← Receive anomaly event
GET  /api/incidents            ← List all incidents
GET  /api/incidents/{id}       ← Get specific incident
POST /api/incidents/{id}/approve ← Approve auto-fix
GET  /api/health               ← System health check
```

### Docker Compose Setup
- [ ] `aegis-api` container (FastAPI backend)
- [ ] `aegis-ui` container (Streamlit frontend)
- [ ] `chroma-db` volume for persistent vector store
- [ ] `.env` file for API keys

---

## Testing Strategy

### Unit Tests
- Test each detector independently
- Mock LLM calls in agent tests
- Test RAG retrieval relevance

### Integration Tests
- End-to-end: fake anomaly → diagnosis → fix suggestion
- Test with each failure category

### Eval Tests (AI-Specific)
```python
# Use RAGAS to evaluate RAG quality
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Test cases: anomaly + expected diagnosis category
test_cases = [
    {
        "anomaly": "Model accuracy dropped from 85% to 62%",
        "expected_category": "model_issue",
        "expected_cause_keywords": ["drift", "distribution", "retrain"]
    }
]
```

---

## Definition of Done (DoD)

A phase is "done" when:
- [ ] Core functionality works end-to-end
- [ ] Unit tests pass (>80% coverage)
- [ ] Code is documented with docstrings
- [ ] No hard-coded secrets or API keys
- [ ] README for that component is updated
