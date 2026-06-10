# 🛠️ Tech Stack Research — Aegis AI

---

## Technology Decision Matrix

Every technology choice in Aegis AI was researched to justify selection based on the project's needs (autonomous AI debugging, RAG, monitoring).

---

## 1. Programming Language: Python 3.11+

**Why Python:**
- Dominant language in AI/ML ecosystem
- All required libraries (LangChain, LangGraph, FAISS, ChromaDB, scikit-learn) have first-class Python support
- Fastest iteration speed for AI prototyping

**Version:** Python 3.11+ (for better performance and type hint support)

---

## 2. LLM Frameworks

### Primary Choice: LangChain + LangGraph

| Framework | Purpose | When to Use |
|---|---|---|
| **LangChain** | RAG chains, prompt templates, tool calling | All RAG and single-step chains |
| **LangGraph** | Stateful multi-step agent workflows | The debugging agent (cyclic logic) |
| **LlamaIndex** | Advanced document indexing & retrieval | Complex multi-document RAG |
| **CrewAI** | Role-based multi-agent teams | Phase 4 multi-agent upgrade |

**Decision: LangChain + LangGraph**
- LangChain handles RAG pipeline and LLM calls
- LangGraph handles the agent state machine (loops, conditionals, HITL)
- Together they cover the full system

#### Why LangGraph Over Basic Agents?
```
Basic LangChain Agent:  Linear → Execute → Done
LangGraph Agent:        Analyze → Fix → Test → Loop Back If Failed → Done
```

LangGraph supports:
- **Cyclic graphs** (retry loops)
- **Persistent state** (resume after crash)
- **Human-in-the-loop interrupts**
- **Time-travel debugging** via LangGraph Studio

---

## 3. LLM Providers

### Option A: OpenAI (GPT-4o / GPT-4o-mini)
- ✅ Best reasoning quality
- ✅ Structured output (JSON mode)
- ✅ Function calling support
- ❌ API cost (use GPT-4o-mini for dev, GPT-4o for production)

### Option B: Local Models (Ollama)
- ✅ Free, private, no API costs
- ✅ Works offline
- ❌ Lower reasoning quality than GPT-4o
- **Models to try:** `llama3.2`, `mistral-nemo`, `deepseek-r1`

### Option C: Groq (LLaMA 3.3 70B)
- ✅ Free tier, extremely fast inference
- ✅ High quality reasoning
- ✅ Best for development & demos

**Recommended Setup:**
```
Development:  Groq (free, fast) or Ollama (offline)
Production:   OpenAI GPT-4o-mini (cost-effective, high quality)
```

---

## 4. Vector Databases

### ChromaDB (Primary Choice)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(
    name="aegis_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["LLM hallucination occurs when..."],
    metadatas=[{"category": "llm_issue", "severity": "high"}],
    ids=["doc_001"]
)

# Query
results = collection.query(
    query_texts=["model accuracy dropped suddenly"],
    n_results=5,
    where={"category": "model_issue"}  # metadata filtering
)
```

**Why ChromaDB for MVP:**
- Zero-configuration setup
- Built-in persistence (saves to disk)
- Metadata filtering (filter by failure category)
- Integrates seamlessly with LangChain

### FAISS (Production Scale)

```python
import faiss
import numpy as np

# Build flat L2 index
dimension = 1536  # OpenAI embedding size
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.array(embeddings, dtype=np.float32)
index.add(vectors)

# Search
D, I = index.search(query_vector, k=5)
```

**Use FAISS when:** Knowledge base exceeds 100K+ documents

---

## 5. Embedding Models

### Option A: OpenAI text-embedding-3-small
- 1536 dimensions, very high quality
- Cost: ~$0.02 per 1M tokens

### Option B: sentence-transformers (Local, Free)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
# 384 dimensions, fast, free
embeddings = model.encode(["text to embed"])
```

**Recommendation:** Use `sentence-transformers` for development, switch to OpenAI embeddings for production quality.

---

## 6. Data Drift Detection Libraries

### scipy (Built-in statistical tests)
```python
from scipy import stats

# Kolmogorov-Smirnov test (continuous features)
ks_stat, p_value = stats.ks_2samp(reference, production)

# Chi-squared test for categorical features.
# ⚠️  Do NOT use scipy.stats.chisquare() on raw label arrays.
# chisquare() compares observed frequencies against a pre-defined expected
# distribution. For drift detection we compare two empirical distributions,
# which requires a contingency table approach:
import pandas as pd
contingency_table = pd.crosstab(reference_series, production_series)
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
```

### Evidently AI (Full drift reports)
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")
```

### Alibi Detect (Advanced)
```python
from alibi_detect.cd import KSDrift

cd = KSDrift(reference_data, p_val=0.05)
result = cd.predict(production_data)
```

**Recommendation for Aegis AI:**
- `scipy` for simple per-feature tests (low dependency)
- `evidently` for generating human-readable drift reports

---

## 7. Hallucination Detection

### RAGAS (RAG Evaluation)
```python
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

# ✅ RAGAS requires `contexts` to be list[list[str]] — a list of lists
# where each inner list contains the plain text strings of the retrieved
# documents for that question. Passing LangChain Document objects directly
# will cause a TypeError. Extract .page_content from each document.
data = {
    "question": [query],
    "answer": [llm_response],
    "contexts": [[doc.page_content for doc in retrieved_docs]],
}
dataset = Dataset.from_dict(data)

# ⚠️  RAGAS defaults to OpenAI for evaluation. If using Groq or Ollama,
# pass explicit llm/embeddings wrappers; otherwise evaluate() will fail.
# Example with a custom LLM:
# from ragas.llms import LangchainLLMWrapper
# from langchain_groq import ChatGroq
# ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile"))
# results = evaluate(dataset, metrics=[faithfulness], llm=ragas_llm)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

**Key RAGAS Metrics:**
| Metric | Meaning | Target |
|---|---|---|
| `faithfulness` | Is the answer grounded in context? | > 0.8 |
| `answer_relevancy` | Is the answer relevant to the question? | > 0.7 |
| `context_precision` | Did we retrieve the right context? | > 0.7 |
| `context_recall` | Did we retrieve all needed context? | > 0.6 |

### LLM-as-Judge (Simple, No Library)
```python
def check_faithfulness(query, response, context, llm):
    prompt = f"""
    Context: {context}
    Response: {response}

    Is the response factually supported by the context?
    Answer with JSON: {{"faithful": true/false, "reason": "..."}}
    """
    return llm.invoke(prompt)
```

---

## 8. Monitoring & Logging

### Structured Logging
```python
import logging
import json
from datetime import datetime

def log_event(event_type, system_id, details):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "system_id": system_id,
        "details": details
    }
    logging.info(json.dumps(log_entry))
```

### LangSmith (LLM Call Tracing)
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "aegis-ai"
# All LangChain calls are automatically traced
```

### OpenLIT (Auto-instrumentation)
```python
import openlit
openlit.init()  # Automatically instruments OpenAI, LangChain, etc.
```

---

## 9. Web Framework & UI

### FastAPI (Backend API)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Aegis AI API")

class AnomalyEvent(BaseModel):
    system_id: str
    metric_type: str
    value: float
    timestamp: str

@app.post("/report-anomaly")
async def report_anomaly(event: AnomalyEvent):
    # Trigger diagnosis pipeline
    return {"status": "analyzing", "event_id": "..."}
```

### Streamlit (Dashboard UI)
```python
import streamlit as st

st.title("🛡️ Aegis AI Dashboard")
st.metric("Active Issues", 3, delta=-2)
st.dataframe(recent_incidents_df)
if st.button("Run Diagnosis"):
    with st.spinner("Analyzing..."):
        result = run_diagnosis(selected_incident)
    st.json(result)
```

---

## 10. Containerization: Docker

### Multi-stage Dockerfile
```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS production
COPY src/ ./src/
COPY data/ ./data/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: "3.8"
services:
  aegis-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data

  aegis-ui:
    build:
      context: .
      dockerfile: docker/Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - aegis-api
```

---

## Complete Requirements File

```
# Core LLM & Agent
langchain==0.3.x
langchain-openai==0.2.x
langchain-community==0.3.x
langgraph==0.2.x
langsmith==0.1.x

# Vector Databases
chromadb==0.5.x
faiss-cpu==1.8.x

# Embeddings
sentence-transformers==3.x

# Statistical Testing
scipy==1.13.x
evidently==0.4.x
numpy==1.26.x
pandas==2.2.x

# Evaluation
ragas==0.1.x

# API & UI
fastapi==0.115.x
uvicorn==0.30.x
streamlit==1.39.x
pydantic==2.8.x

# Utilities
python-dotenv==1.0.x
loguru==0.7.x
rich==13.x
httpx==0.27.x

# Optional: Local Models
# ollama==0.3.x
```
