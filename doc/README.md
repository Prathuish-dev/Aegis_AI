# 📖 doc/ — Aegis AI Research & Documentation Index

> This folder contains all research, architecture decisions, and development guides for the **Aegis AI** project.

---

## 📂 Document Index

| # | File | Description | Priority |
|---|---|---|---|
| 01 | [01_project_overview.md](./01_project_overview.md) | Core concept, problem statement, value proposition | 🔴 Start Here |
| 02 | [02_architecture_deep_dive.md](./02_architecture_deep_dive.md) | Detailed system architecture with code examples | 🔴 Critical |
| 03 | [03_tech_stack_research.md](./03_tech_stack_research.md) | Technology decisions: LangChain, ChromaDB, FAISS, etc. | 🔴 Critical |
| 04 | [04_development_lifecycle.md](./04_development_lifecycle.md) | Phase-by-phase development plan (12-week roadmap) | 🟡 Reference |
| 05 | [05_rag_system_guide.md](./05_rag_system_guide.md) | Deep dive into RAG pipeline, embeddings, retrieval strategies | 🟡 Reference |
| 06 | [06_failure_detection_research.md](./06_failure_detection_research.md) | All 5 failure categories with detection code | 🟡 Reference |
| 07 | [07_quickstart_guide.md](./07_quickstart_guide.md) | Setup instructions, commands, project structure | 🟢 Operational |

---

## 🧠 Suggested Reading Order

### If you're new to the project:
1. → `01_project_overview.md` — understand WHY this exists
2. → `02_architecture_deep_dive.md` — understand HOW it works
3. → `03_tech_stack_research.md` — understand WHAT tools to use
4. → `07_quickstart_guide.md` — get it running

### If you're starting development:
1. → `04_development_lifecycle.md` — follow the 12-week plan
2. → `06_failure_detection_research.md` — implement detectors first
3. → `05_rag_system_guide.md` — build the knowledge engine

---

## 🗺️ System at a Glance

```
AI/ML System Under Observation
         │
         ▼
  [Monitoring Layer]          ← Collects logs, metrics, outputs
         │ anomaly detected
         ▼
  [Detection Engine]          ← Rules + KS Test + Semantic Scoring
         │ structured anomaly event
         ▼
  [RAG Knowledge Engine]      ← Retrieves relevant past incidents & guides
         │ context + anomaly
         ▼
  [LLM Debugging Agent]       ← Reasons about root cause (LangGraph)
         │ diagnosis + fix plan
         ▼
  [Self-Healing Engine]       ← Suggests or applies fix
         │ result
         ▼
  [Monitoring Layer]          ← Verifies fix worked (feedback loop)
```

---

## 🔑 Key Concepts Quick Reference

| Concept | What It Means | Where Used |
|---|---|---|
| **Data Drift** | Input data distribution changed from training | Detection Layer |
| **KS Test** | Statistical test for distribution comparison | Drift Detector |
| **PSI** | Population Stability Index (drift severity score) | Drift Detector |
| **RAG** | Retrieval-Augmented Generation — grounding LLM with knowledge | RAG Engine |
| **ChromaDB** | Vector database for storing/searching embeddings | RAG Engine |
| **FAISS** | Fast similarity search library (production scale) | RAG Engine (advanced) |
| **MMR** | Maximal Marginal Relevance — diverse retrieval results | RAG Retrieval |
| **LangGraph** | Framework for stateful, cyclic LLM agent workflows | Debugging Agent |
| **RAGAS** | RAG evaluation framework (faithfulness, relevance) | Evaluation |
| **HITL** | Human-in-the-Loop — human approves before action | Fix Engine |
| **CoT** | Chain-of-Thought — step-by-step LLM reasoning | Agent Prompts |

---

## 📊 Research Sources

The documentation in this folder is based on research from:

- **LangChain / LangGraph official documentation** (2025)
- **Academic papers**: Corrective RAG (CRAG), Self-RAG, RAGAS evaluation
- **Industry reports**: MLOps monitoring best practices 2025-2026
- **Tools**: Evidently AI, Alibi Detect, OpenLIT, Arize Phoenix
- **Community**: LangChain Discord, Hugging Face forums

---

## ✅ Research Completion Status

| Topic | Status | Document |
|---|---|---|
| Project concept & scope | ✅ Complete | 01_project_overview.md |
| System architecture | ✅ Complete | 02_architecture_deep_dive.md |
| Technology selection | ✅ Complete | 03_tech_stack_research.md |
| Development roadmap | ✅ Complete | 04_development_lifecycle.md |
| RAG pipeline design | ✅ Complete | 05_rag_system_guide.md |
| Failure detection methods | ✅ Complete | 06_failure_detection_research.md |
| Environment & quickstart | ✅ Complete | 07_quickstart_guide.md |
| API reference | 🔲 Pending (build Phase 6) | 08_api_reference.md |
| Deployment guide | 🔲 Pending (build Phase 6) | 09_deployment_guide.md |
| Multi-agent upgrade | 🔲 Pending (advanced phase) | 10_multiagent_design.md |
