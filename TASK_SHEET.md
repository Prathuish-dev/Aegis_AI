# 🛡️ Aegis AI — Multi-Agent Collaborative Task Sheet

**Project Name:** Aegis AI — Autonomous Self-Healing System for ML/LLM Pipelines  
**Primary Goal:** Build a production-ready, closed-loop autonomous system that monitors, diagnoses, and self-heals ML/LLM pipeline failures using RAG + LangGraph agents  
**Deadline:** 12 Weeks from Kickoff  
**Document Version:** v1.0  
**Last Updated:** 2026-06-13  

---

## 👥 Agent Roster

| Agent ID | Role | Scope |
|---|---|---|
| **Agent-A** | Backend & Infrastructure Engineer | Project scaffolding, data models, SQLite/DB, config, Docker |
| **Agent-B** | ML/AI Engineer | Anomaly detection, drift detection, statistical tests, embeddings |
| **Agent-C** | LLM & RAG Engineer | RAG pipeline, ChromaDB, LangChain chains, knowledge base ingestion |
| **Agent-D** | Agentic Systems Engineer | LangGraph agent, state machine, HITL workflow, self-healing engine |
| **Agent-E** | QA, UI & DevOps Engineer | Unit tests, integration tests, Streamlit dashboard, FastAPI, Docker Compose |

---

---

# SECTION 1 — Operational Protocols

---

## 1.1 Concurrency Rules

- **No two agents may modify the same file simultaneously.** Each file is "claimed" when an agent begins work on it. A claim is logged in the hand-off notes column.
- **Agents working on parallel tasks must operate on separate modules.** Example: Agent-B works in `src/detection/`, Agent-C works in `src/rag/` — their directories are fully isolated.
- **Shared interfaces (Pydantic models in `src/monitoring/models.py`) are frozen once Agent-A completes Task 1.3.** No agent may change model schemas without a team-wide notification and a version bump.
- **Only one agent merges to `main` at a time.** All agents work on feature branches (`feature/agent-X-task-Y`). Merge requests require no conflicting in-progress claims.
- **A task is only "In Progress 🔄" for one agent at a time.** If a task requires two agents, it is split into sub-tasks with distinct IDs.

---

## 1.2 Hand-off Procedures

When an agent completes a task that another task depends on, they must:

1. **Mark the task `Completed ✅`** in the task sheet immediately.
2. **Write a hand-off note** in the `Notes/Hand-off` column containing:
   - Module/file path of the deliverable
   - The exact class/function name exposed for the next agent
   - Any known limitations or edge cases
   - Branch name or commit SHA
3. **Notify the receiving agent** by tagging them in the notes: `→ READY FOR [Agent-X]`.
4. **Do not begin a subsequent task that touches the same interface** until the receiving agent acknowledges.

> **Example Hand-off Note:**  
> `src/monitoring/models.py → AnomalyEvent, MetricType exported. Branch: feature/agent-a-phase1. → READY FOR Agent-B`

---

## 1.3 Conflict / Blocker Resolution Protocol

| Situation | Protocol |
|---|---|
| **Agent is blocked on a dependency** | Mark task `Blocked 🛑`. Pick next available `Pending ⏳` task with no unsatisfied dependencies. Log blocker: `BLOCKED ON [Task ID] — assigned to [Agent]`. |
| **Two agents need to edit the same file** | The later agent waits. First agent adds note: `FILE CLAIMED by [Agent-X] until [Task ID] complete`. |
| **Dependency task is delayed > 2 days** | Escalate: the delayed agent provides a partial stub/interface so the blocked agent can continue with mocked input. |
| **Agent disagrees on implementation** | Raise a `DESIGN CONFLICT` note. Default to the decision documented in `doc/02_architecture_deep_dive.md`. If not covered, the more senior-scoped agent (Agent-D for agent logic, Agent-C for RAG) has final say. |
| **Discovered a bug in a completed task** | Open a new task with ID `[original-ID].fix` and assign it. Do not revert the original task's `Completed ✅` status. |

---

## 1.4 Communication & Logging Standards

All task note entries follow this format:

```
[YYYY-MM-DD HH:MM] [Agent-X] [STATUS_CHANGE | NOTE | BLOCKER | HANDOFF]
Message here.
```

> **Example:**  
> `[2026-06-14 09:30] [Agent-A] [HANDOFF] AnomalyEvent model finalized. src/monitoring/models.py. → READY FOR Agent-B, Agent-C`

- **Status must be updated the same day work starts or completes** — no retroactive updates.
- **Branch naming convention:** `feature/agent-[A-E]-[task-id]` (e.g., `feature/agent-c-3.2`)
- **Commit message format:** `[Phase X] short description — Agent-[X]`
- **All work is pushed to GitHub daily** at end of session, even if incomplete (push to feature branch, not `main`).

---

---

# SECTION 2 — The Master Checklist

---

## 📦 Phase 1 — Foundation Setup (Week 1–2)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **1.1** | Create full directory scaffold (`src/`, `doc/`, `data/`, `tests/`, `docker/`, `config/`) | Agent-A | None | ✅ Completed | Structure defined in `doc/04_development_lifecycle.md` |
| **1.2** | Set up Python 3.11 virtual environment, `requirements.txt`, `.env.example` | Agent-A | 1.1 | ✅ Completed | All deps documented in `doc/03_tech_stack_research.md` |
| **1.3** | Define all Pydantic data models: `AnomalyEvent`, `MetricType`, `FailureCategory` in `src/monitoring/models.py` | Agent-A | 1.2 | ✅ Completed | **SCHEMA FROZEN** after this. → READY FOR Agent-B, Agent-C, Agent-D |
| **1.4** | Implement `LogCollector` class with SQLite persistence and `_init_db()` in `src/monitoring/log_collector.py` | Agent-A | 1.3 | ✅ Completed | `_init_db()` creates events table; env-driven db_path via `LOG_DB_PATH` |
| **1.5** | Implement `MetricTracker` class for polling numerical metrics in `src/monitoring/metric_tracker.py` | Agent-A | 1.3 | 🔄 In Progress | Tracks accuracy, latency, error_rate with sliding window<br>[2026-06-13 21:03] [Agent-A] [STATUS_CHANGE] Task started. |
| **1.6** | Write sample log generator script `scripts/generate_sample_logs.py` for dev/testing | Agent-E | 1.4 | ⏳ Pending | Generates 5 failure scenario logs for each category |
| **1.7** | Set up `config/settings.yaml` with all detection thresholds and system settings | Agent-A | 1.2 | 🔄 In Progress | Thresholds: accuracy drop ≥5%, latency P95 ≥2000ms, error rate ≥2%<br>[2026-06-13 21:03] [Agent-A] [STATUS_CHANGE] Task started. |
| **1.8** | Write unit tests for `LogCollector` and `MetricTracker` in `tests/test_monitoring.py` | Agent-E | 1.4, 1.5 | ⏳ Pending | Test: ingest, retrieve, schema validation |

---

## 🔍 Phase 2 — Anomaly Detection Engine (Week 3–4)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **2.1** | Implement `DetectionRule` dataclass and `RuleEngine` with `_check()` for `lt`, `gt`, `delta_pct` operators in `src/detection/rule_engine.py` | Agent-B | 1.3 | ⏳ Pending | Rules configurable from `settings.yaml` |
| **2.2** | Implement `DriftDetector` with KS test (`detect_feature_drift`) and corrected PSI (`compute_psi`) in `src/detection/drift_detector.py` | Agent-B | 1.3 | ⏳ Pending | Use shared `bin_edges` for PSI; proportions not density |
| **2.3** | Implement `DataIssueDetector` with `detect_covariate_drift` (KS + `chi2_contingency`) and `detect_schema_drift` in `src/detection/data_issue_detector.py` | Agent-B | 2.2 | ⏳ Pending | `chi2_contingency` on `pd.crosstab` — see doc/06 |
| **2.4** | Implement `ModelPerformanceDetector` with sliding window accuracy/F1/AUC monitoring in `src/detection/model_performance_detector.py` | Agent-B | 1.3 | ⏳ Pending | Configurable window_hours and drop_threshold |
| **2.5** | Implement module-level `RelevanceScorer` singleton (SentenceTransformer loaded once) in `src/detection/relevance_scorer.py` | Agent-B | 1.3 | ⏳ Pending | Model: `BAAI/bge-small-en-v1.5`; never instantiate inside function |
| **2.6** | Implement unified `AnomalyDetector` facade combining 2.1–2.5, outputting `AnomalyEvent` schema | Agent-B | 2.1, 2.2, 2.3, 2.4, 2.5 | ⏳ Pending | → READY FOR Agent-D after this |
| **2.7** | Implement `SystemIssueDetector` with regex-based log pattern matching in `src/detection/system_issue_detector.py` | Agent-B | 1.4 | ⏳ Pending | Patterns: api_timeout, rate_limit, OOM, auth_error, model_not_found |
| **2.8** | Write unit tests for all detectors in `tests/test_detection.py` | Agent-E | 2.1–2.7 | ⏳ Pending | Cover: no drift, drift detected, edge cases (empty arrays, single element) |

---

## 📚 Phase 3 — RAG Knowledge Engine (Week 5–6)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **3.1** | Populate knowledge base documents: min 5 guides covering LLM issues, model issues, data drift, system issues, prompt engineering under `data/knowledge_base/` | Agent-C | 1.1 | ✅ Completed | Use Markdown format; min 400 words per guide<br>[2026-06-13 20:55] [Agent-C] [STATUS_CHANGE] Task started.<br>[2026-06-13 21:00] [Agent-C] [STATUS_CHANGE] Completed. All 5 RAG markdown guides created. data/knowledge_base/ → READY FOR Agent-C (Task 3.3) |
| **3.2** | Populate 5+ past incident JSON records under `data/knowledge_base/past_incidents/` | Agent-C | 1.1 | ✅ Completed | Schema: incident_id, failure_type, root_cause, fix_applied, outcome, tags<br>[2026-06-13 20:55] [Agent-C] [STATUS_CHANGE] Task started.<br>[2026-06-13 21:00] [Agent-C] [STATUS_CHANGE] Completed. All 5 past incident records created. data/knowledge_base/past_incidents/ → READY FOR Agent-C (Task 3.3) |
| **3.3** | Implement `KnowledgeBaseIngester` using LangChain `Chroma` wrapper with injected `embedding_function` in `src/rag/ingestion.py` | Agent-C | 3.1, 3.2 | ⏳ Pending | Must accept same `embedding_function` as `AegisRAG` — model consistency rule |
| **3.4** | Implement `AegisRAG` retriever with MMR search and category filtering in `src/rag/retrieval.py` | Agent-C | 3.3 | ⏳ Pending | `search_type="mmr"`, k=5, fetch_k=20 |
| **3.5** | Implement `RAGEvaluator` using corrected RAGAS setup (list[list[str]] contexts, non-OpenAI LLM config) in `src/rag/evaluator.py` | Agent-C | 3.4 | ⏳ Pending | Targets: faithfulness ≥0.85, answer_relevancy ≥0.80 |
| **3.6** | Implement `rebuild_knowledge_base.py` script to re-ingest all docs into ChromaDB | Agent-C | 3.3 | ⏳ Pending | Supports `--reset` flag to wipe and re-index |
| **3.7** | Write unit tests for RAG retrieval quality in `tests/test_rag.py` | Agent-E | 3.4, 3.5 | ⏳ Pending | Golden test queries for each failure category; assert top-3 docs |
| **3.8** | Run manual RAG quality spot-check: all 5 golden test queries must return expected docs in top-3 | Agent-C, Agent-E | 3.7 | ⏳ Pending | Document retrieval scores in test notes |

---

## 🤖 Phase 4 — LLM Debugging Agent (Week 7–8)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **4.1** | Define `AegisAgentState` TypedDict in `src/agent/state.py` | Agent-D | 1.3 | ⏳ Pending | Fields: anomaly_description, retrieved_context, root_cause, fix_recommendation, confidence_score, requires_human_review, iteration_count |
| **4.2** | Implement all agent node functions (`retrieve_context_node`, `analyze_anomaly_node`, `classify_failure_node`, `generate_fix_node`, `human_review_node`, `apply_fix_node`) in `src/agent/nodes.py` | Agent-D | 4.1, 3.4 | ⏳ Pending | Each node must update state and log via structured logger |
| **4.3** | Implement `route_after_fix()` conditional router (returns `"human_review"` if confidence < 80, else `"apply_fix"`) | Agent-D | 4.2 | ⏳ Pending | Threshold configurable in `settings.yaml` |
| **4.4** | Build LangGraph state machine using corrected topology: all nodes declared, `human_review → apply_fix → END`, `interrupt_before=["human_review"]` in `src/agent/graph.py` | Agent-D | 4.2, 4.3 | ⏳ Pending | Use `MemorySaver` checkpointer; `interrupt_before` on `compile()` only |
| **4.5** | Define Chain-of-Thought prompt templates in `src/agent/prompts.py` | Agent-D | 4.1 | ⏳ Pending | Structured JSON output: root_cause, failure_category, fix_recommendation, confidence_score, knowledge_references |
| **4.6** | End-to-end integration test: simulate each of 5 failure types → verify agent produces valid JSON diagnosis | Agent-D, Agent-E | 4.4, 4.5, 2.6, 3.4 | ⏳ Pending | Use mocked LLM responses to avoid API cost in CI |
| **4.7** | Write unit tests for graph routing logic and state transitions in `tests/test_agent.py` | Agent-E | 4.4 | ⏳ Pending | Test: low-confidence routes to human_review; high-confidence skips it |

---

## 🔧 Phase 5 — Self-Healing Engine (Week 9–10)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **5.1** | Implement `FixExecutor` with three modes (`suggest`, `semi-auto`, `auto`) in `src/healing/fix_executor.py` | Agent-D | 4.4 | ⏳ Pending | Mode driven by `FIX_MODE` env variable |
| **5.2** | Implement `PromptOptimizer` that rewrites failing prompts using LLM in `src/healing/prompt_optimizer.py` | Agent-D | 4.5 | ⏳ Pending | Maps issue_type → rewrite strategy |
| **5.3** | Implement `LLMQualityDetector` with `_llm_faithfulness_check()` (LLM-as-Judge replacing embedding similarity) in `src/healing/llm_quality_detector.py` | Agent-B | 2.5 | ⏳ Pending | JSON output: `{"faithful": bool, "reason": str}` with parse-error fallback |
| **5.4** | Implement `RetrievalQualityDetector` with pairwise document embedding comparison for duplicate detection in `src/detection/retrieval_quality_detector.py` | Agent-B | 2.5 | ⏳ Pending | Compare doc embeddings against each other, not against query scores |
| **5.5** | Implement post-fix verifier that re-triggers monitoring on healed system to confirm recovery in `src/healing/verifier.py` | Agent-D | 5.1, 2.6 | ⏳ Pending | Polls metric for 10 min post-fix; logs `healed` or `unresolved` |
| **5.6** | Implement structured audit logger: record every autonomous action with timestamp, agent, action, outcome in `src/healing/audit_logger.py` | Agent-A | 1.4 | ⏳ Pending | Append to SQLite `audit_log` table |
| **5.7** | Write integration tests: simulate accuracy drop → auto-fix suggestion → verify audit log entry | Agent-E | 5.1–5.6 | ⏳ Pending | Assert fix suggestion contains required fields |

---

## 🖥️ Phase 6 — UI, API & Docker (Week 11–12)

| Task ID | Task Description | Assigned Agent(s) | Dependencies | Status | Notes / Hand-off |
|---|---|---|---|---|---|
| **6.1** | Implement FastAPI app with 5 endpoints in `src/api/main.py`: `POST /report-anomaly`, `GET /incidents`, `GET /incidents/{id}`, `POST /incidents/{id}/approve`, `GET /health` | Agent-A | 4.4, 5.1 | ⏳ Pending | Use Pydantic request/response schemas from models.py |
| **6.2** | Build Streamlit dashboard in `ui/dashboard.py`: live incident feed, detail view, manual trigger, HITL approval panel, health metrics | Agent-E | 6.1 | ⏳ Pending | Real-time polling of FastAPI `/incidents` endpoint |
| **6.3** | Write multi-stage `Dockerfile` for FastAPI backend | Agent-A | 6.1 | ⏳ Pending | `python:3.11-slim`; expose port 8000 |
| **6.4** | Write `Dockerfile.streamlit` for UI container | Agent-E | 6.2 | ⏳ Pending | Expose port 8501 |
| **6.5** | Write `docker-compose.yml` with `aegis-api`, `aegis-ui` services and ChromaDB volume | Agent-A | 6.3, 6.4 | ⏳ Pending | Env vars from `.env` file; `depends_on` ordering |
| **6.6** | Full end-to-end system test: `docker-compose up` → submit anomaly via API → verify dashboard shows diagnosis | Agent-E | 6.5 | ⏳ Pending | Document any port conflicts or container startup order issues |
| **6.7** | Write API reference documentation in `doc/08_api_reference.md` | Agent-A | 6.1 | ⏳ Pending | Include request/response schemas and example `curl` commands |
| **6.8** | Write deployment guide in `doc/09_deployment_guide.md` | Agent-E | 6.5, 6.6 | ⏳ Pending | Covers local Docker, env setup, common issues |
| **6.9** | Final code review: all modules have docstrings, no hard-coded secrets, coverage ≥80% | Agent-E | All Phase 1–6 | ⏳ Pending | Run `pytest --cov=src` and attach report |
| **6.10** | Tag release `v1.0.0` and push to GitHub | Agent-A | 6.6, 6.9 | ⏳ Pending | Update main `README.md` with setup instructions and demo GIF |

---

## 📊 Progress Summary

| Phase | Total Tasks | Completed ✅ | In Progress 🔄 | Blocked 🛑 | Pending ⏳ |
|---|---|---|---|---|---|
| Phase 1 — Foundation | 8 | 4 | 2 | 0 | 2 |
| Phase 2 — Detection | 8 | 0 | 0 | 0 | 8 |
| Phase 3 — RAG | 8 | 2 | 0 | 0 | 6 |
| Phase 4 — Agent | 7 | 0 | 0 | 0 | 7 |
| Phase 5 — Healing | 7 | 0 | 0 | 0 | 7 |
| Phase 6 — UI/API | 10 | 0 | 0 | 0 | 10 |
| **TOTAL** | **48** | **6** | 2 | **0** | **40** |

---

---

# SECTION 3 — Immediate Action Plan

> Tasks that can start **RIGHT NOW** in parallel, without waiting for any dependency.

---

## 🚀 Parallel Kickoff Tasks (Start Simultaneously)

The following 5 tasks have **no unsatisfied dependencies** and can be started immediately by all 5 agents working concurrently:

| Priority | Task ID | Agent | Action to Take Now |
|---|---|---|---|
| 🔴 **Critical Path** | **1.5** | **Agent-A** | Implement `MetricTracker` class. This unblocks 1.8 and feeds into Phase 2 detectors. |
| 🔴 **Critical Path** | **2.1** | **Agent-B** | Implement `RuleEngine` with `_check()`. Task 1.3 (models) is complete — unblocks the entire Phase 2 chain. |
| 🔴 **Critical Path** | **3.1** | **Agent-C** | Write all 5+ knowledge base markdown guides in `data/knowledge_base/`. Task 1.1 is done. This is pure content work with zero code dependencies. |
| 🔴 **Critical Path** | **4.1** | **Agent-D** | Define `AegisAgentState` TypedDict. Task 1.3 (models) is complete — this is prerequisite for all Phase 4 nodes. |
| 🟡 **Parallel** | **1.6** | **Agent-E** | Write sample log generator script. Task 1.4 is complete — needed for all downstream testing. |

---

## ⛓️ Critical Path (Longest Dependency Chain)

The most blocking sequence that determines the overall project timeline:

```
1.3 (Models) ✅
  └─▶ 2.6 (Unified AnomalyDetector)
        └─▶ 4.2 (Agent Nodes)
              └─▶ 4.4 (LangGraph Graph)
                    └─▶ 5.1 (Fix Executor)
                          └─▶ 6.1 (FastAPI)
                                └─▶ 6.5 (Docker Compose)
                                      └─▶ 6.6 (E2E Test) ← PROJECT DONE
```

> **Agent-D is the critical path bottleneck.** Prioritise clearing Phase 4 tasks before moving Agent-D to Phase 5 work.

---

## 🗓️ Week-by-Week Sprint Allocation

| Week | Agent-A | Agent-B | Agent-C | Agent-D | Agent-E |
|---|---|---|---|---|---|
| **1–2** | 1.5, 1.7 | 2.1, 2.2 | 3.1, 3.2 | 4.1, 4.5 | 1.6, 1.8 |
| **3–4** | — | 2.3, 2.4, 2.5, 2.6, 2.7 | 3.3, 3.4 | — | 2.8 |
| **5–6** | — | 5.3, 5.4 | 3.5, 3.6 | — | 3.7, 3.8 |
| **7–8** | — | — | — | 4.2, 4.3, 4.4 | 4.6, 4.7 |
| **9–10** | 5.6 | — | — | 5.1, 5.2, 5.5 | 5.7 |
| **11–12** | 6.1, 6.3, 6.5, 6.7, 6.10 | — | — | — | 6.2, 6.4, 6.6, 6.8, 6.9 |

---

## ⚠️ Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Agent-D blocked if 2.6 (AnomalyDetector) is late | Medium | High | Agent-D works on 4.1 and 4.5 (state + prompts) independently while waiting |
| ChromaDB embedding model mismatch regression | Low | Critical | Enforce: `KnowledgeBaseIngester` and `AegisRAG` always use the same `embedding_function` instance — documented in 3.3 |
| RAGAS evaluation fails with non-OpenAI LLM | Medium | Medium | Configure `LangchainLLMWrapper` for Groq/Ollama before running 3.5 |
| LangGraph graph topology bug (dead-end node) | Low | High | All graphs reviewed against doc/04 corrected template before merge |
| API key costs in CI testing | High | Low | All agent tests use mocked LLM responses via `unittest.mock` |
