# 🏗️ Aegis AI — Architecture Deep Dive

---

## Overview: Closed-Loop Architecture

Aegis AI follows a **closed-loop agentic architecture** — the most modern paradigm for self-healing AI systems (2025-2026). Unlike simple linear pipelines, every stage has feedback loops.

```
                    ┌─────────────────────────────────┐
                    │         AI/ML System Under      │
                    │            Observation          │
                    └──────────────┬──────────────────┘
                                   │ logs, metrics, outputs
                                   ▼
         ┌─────────────────────────────────────────────┐
         │            MONITORING LAYER                 │
         │  • Log Parser & Collector                   │
         │  • Metric Tracker (accuracy, latency, etc.) │
         │  • Output Sampler                           │
         └──────────────────┬──────────────────────────┘
                            │ structured events
                            ▼
         ┌─────────────────────────────────────────────┐
         │           ANOMALY DETECTION LAYER           │
         │  • Rule-based triggers (threshold crossing) │
         │  • Statistical drift detectors (KS, PSI)   │
         │  • Semantic relevance scoring               │
         └──────────────────┬──────────────────────────┘
                            │ anomaly event + context
                            ▼
         ┌─────────────────────────────────────────────┐
         │           RAG KNOWLEDGE ENGINE              │
         │  • Vector DB (ChromaDB/FAISS)               │
         │  • Past error logs + solutions              │
         │  • Debugging guides & prompt rules          │
         │  • Hybrid search (semantic + keyword)       │
         └──────────────────┬──────────────────────────┘
                            │ retrieved context
                            ▼
         ┌─────────────────────────────────────────────┐
         │        LLM DEBUGGING AGENT (Brain)          │
         │  • Root cause analysis                      │
         │  • Failure classification                   │
         │  • Fix strategy generation                  │
         │  • Reasoning chain (Chain-of-Thought)       │
         └──────────────────┬──────────────────────────┘
                            │ diagnosis + fix plan
                            ▼
         ┌─────────────────────────────────────────────┐
         │           FIX / HEALING ENGINE              │
         │  • Suggest fix (v1 - safe)                  │
         │  • Auto-apply fix (v2 - advanced)           │
         │  • Human-in-the-loop approval               │
         │  • Post-fix verification                    │
         └─────────────────────────────────────────────┘
                            │ feedback loop
                            └───────────────────────────▶ Monitoring Layer
```

---

## Component 1: Monitoring Layer

### Role
Acts as the **sensory system** of Aegis AI. Collects raw data from monitored AI/ML systems.

### What It Collects
| Data Type | Examples | Collection Method |
|---|---|---|
| **Model Metrics** | Accuracy, F1, precision, AUC | Polling / webhook |
| **LLM Outputs** | Generated text samples | Log interception |
| **Latency Metrics** | P50, P95, P99 response times | Middleware |
| **Error Logs** | Exceptions, stack traces | Log streaming |
| **Input Features** | Feature distributions over time | Batch sampling |
| **API Health** | Status codes, rate limit hits | Health checks |

### Key Design Decisions
- **Structured Log Format**: All logs are normalized to JSON with `timestamp`, `system_id`, `metric_type`, `value`, `metadata`
- **Configurable Polling Interval**: Default 60s for batch metrics, real-time for error logs
- **Persistent Storage**: SQLite for MVP, PostgreSQL for production

---

## Component 2: Anomaly Detection Layer

### Role
Decides **whether something is wrong** and classifies the failure category.

### Detection Strategies

#### 2a. Rule-Based Detection (Fast, Deterministic)
```python
# Example rules
RULES = {
    "accuracy_drop": {"metric": "accuracy", "threshold": -0.05, "window": "1h"},
    "high_latency":  {"metric": "p95_latency", "threshold": 2000, "unit": "ms"},
    "error_rate":    {"metric": "error_rate", "threshold": 0.02},
}
```

#### 2b. Statistical Drift Detection
Uses **Kolmogorov-Smirnov (KS) Test** for continuous features:
```python
from scipy import stats

def detect_drift(reference_data, production_data, alpha=0.05):
    ks_stat, p_value = stats.ks_2samp(reference_data, production_data)
    return p_value < alpha  # True = drift detected
```

**Population Stability Index (PSI)** for distribution stability:
- PSI < 0.1 → No significant drift ✅
- 0.1 ≤ PSI < 0.25 → Moderate drift ⚠️
- PSI ≥ 0.25 → Significant drift 🚨

#### 2c. Semantic Relevance Scoring (For LLM Outputs)
```python
from sentence_transformers import SentenceTransformer, util

# ✅ Model is instantiated ONCE at module level, not on every call.
# Loading SentenceTransformer inside a function causes it to be reloaded
# from disk on every invocation, causing OOM and severe latency spikes.
_RELEVANCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def score_relevance(query, response, threshold=0.5):
    """Returns True if the response is semantically relevant to the query."""
    similarity = util.cos_sim(
        _RELEVANCE_MODEL.encode(query),
        _RELEVANCE_MODEL.encode(response)
    )
    return float(similarity) > threshold
```

### Failure Classification Taxonomy

| Category | Subcategory | Detection Method |
|---|---|---|
| **Data Issue** | Feature drift, schema change | KS test, PSI, schema validator |
| **Model Issue** | Accuracy drop, bias drift | Threshold rules, statistical tests |
| **Prompt Issue** | Hallucination, vagueness | Semantic similarity, faithfulness score |
| **Retrieval Issue** | Low relevance, empty results | Similarity threshold, result count |
| **System Issue** | API failure, timeout, OOM | Error log parsing, status codes |

---

## Component 3: RAG Knowledge Engine

### Role
Before the LLM reasons, it **retrieves relevant historical knowledge** to ground its analysis.

### Knowledge Base Contents
```
knowledge_base/
├── past_incidents/         ← Historical error + fix pairs
├── debugging_guides/       ← How to fix known ML issues
├── prompt_engineering/     ← Prompt best practices & templates
├── drift_detection/        ← Data drift examples and resolutions
└── system_patterns/        ← Common failure signatures
```

### RAG Pipeline Architecture

```
Query (anomaly description)
        │
        ▼
  [Query Embedding]  ←── sentence-transformers
        │
        ▼
  [Vector Search]    ←── ChromaDB / FAISS
        │
        ▼
  [Top-K Documents]  (k=3 to 5)
        │
        ▼
  [Context Builder]  ←── formats retrieved docs + anomaly
        │
        ▼
  [LLM Prompt]       ←── injected with context
```

### Vector Store Choice: ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Persistence | ✅ Built-in | ❌ Manual |
| Metadata filtering | ✅ Easy | ❌ Complex |
| Scale | Medium | ✅ Millions of vectors |
| Setup complexity | Low | Medium |
| **Recommendation** | **Use for MVP** | Use for production scale |

### Hybrid Search (Advanced)
Combine **dense vector search** (semantic) with **BM25 keyword search** for higher precision:
```python
# ChromaDB + BM25 hybrid
results = vector_db.query(query_texts=[query], n_results=5)
bm25_results = bm25.get_scores(query.split())
# Merge and re-rank
```

---

## Component 4: LLM Debugging Agent

### Role
The **brain of Aegis AI**. Takes anomaly context + retrieved knowledge and reasons about root cause and fix.

### Agent Architecture: LangGraph State Machine

```python
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    anomaly: str          # What went wrong
    context: list         # Retrieved RAG documents
    analysis: str         # Root cause analysis
    fix_plan: str         # Generated fix
    confidence: float     # Agent's confidence score
    approved: bool        # Human approval status

# Graph nodes
graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_anomaly)
graph.add_node("classify", classify_failure)
graph.add_node("generate_fix", generate_fix_plan)
graph.add_node("human_review", human_in_the_loop)
graph.add_node("apply_fix", apply_fix)
```

### Reasoning Pattern: Chain-of-Thought (CoT)

The agent uses structured reasoning:
```
Step 1: Describe what the anomaly looks like
Step 2: List possible root causes
Step 3: Cross-reference with retrieved knowledge
Step 4: Identify most likely cause
Step 5: Generate specific fix recommendation
Step 6: Estimate confidence level
```

### Prompt Template Structure
```
SYSTEM: You are an expert AI system debugger. Analyze failures precisely.

CONTEXT FROM KNOWLEDGE BASE:
{retrieved_documents}

ANOMALY DETECTED:
- System: {system_name}
- Type: {failure_type}
- Details: {anomaly_details}
- Metrics: {relevant_metrics}

INSTRUCTIONS:
1. Identify the root cause
2. Classify the failure type
3. Provide a specific, actionable fix
4. Rate your confidence (0-100%)

OUTPUT FORMAT (JSON):
{
  "root_cause": "...",
  "failure_category": "...",
  "fix_recommendation": "...",
  "confidence": 85,
  "references": [...]
}
```

---

## Component 5: Fix / Healing Engine

### Role
Executes or presents the fix recommendation.

### Fix Strategy Levels

| Level | Mode | Action | Safety |
|---|---|---|---|
| **L1** | Suggest | Display fix to engineer | 🟢 Safe |
| **L2** | Semi-auto | Apply after human approval | 🟡 Moderate |
| **L3** | Auto-heal | Apply automatically | 🔴 Requires testing |

### Self-Healing Actions by Category

| Failure | L1 Suggestion | L2/L3 Auto-Fix |
|---|---|---|
| Prompt failure | Rewrite prompt recommendation | Auto-replace prompt in config |
| Low accuracy | Recommend retraining | Trigger retraining pipeline |
| Data drift | Flag affected features | Alert + log for review |
| API timeout | Retry recommendation | Implement exponential backoff |
| Hallucination | Suggest adding context | Inject RAG context to prompt |

---

## Cross-Cutting Concerns

### Observability
- All agent steps are logged with timestamps, inputs, outputs
- LangSmith integration for tracing LLM reasoning chains
- Structured JSON logs for every component

### Human-in-the-Loop (HITL)
```python
# ✅ Correct LangGraph HITL pattern.
# interrupt_before is NOT a parameter of add_node().
# It must be passed to graph.compile() as a list of node names.
from langgraph.checkpoint.memory import MemorySaver

graph.add_node("human_review", human_review_node)
graph.add_node("apply_fix", apply_fix_node)
graph.add_edge("human_review", "apply_fix")

memory = MemorySaver()
# The graph will pause execution BEFORE entering "human_review",
# allowing the engineer to inspect state and resume or modify.
app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])
# Engineer reviews → resumes app.invoke() with updated state → apply_fix runs
```

### Security
- No production secrets in logs
- Fix actions require confirmation for irreversible changes
- Audit trail for every autonomous action taken
