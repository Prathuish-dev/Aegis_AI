# 🔍 Failure Detection & Classification — Research Notes

---

## The Failure Taxonomy of AI/ML Systems

Aegis AI must detect and classify **5 distinct failure categories**. Each requires different detection methods and produces different fixes.

---

## Category 1: Data Issues

### What It Is
Changes in the statistical distribution of input data that the model was not trained on.

### Types of Data Drift

| Type | Description | Detection |
|---|---|---|
| **Covariate Drift** | Input feature distribution changes | KS test per feature |
| **Label Drift** | Target variable distribution changes | KS test on labels |
| **Concept Drift** | Relationship between X and Y changes | Model performance monitoring |
| **Schema Drift** | New/missing columns, type changes | Schema validator |

### Detection Implementation

```python
import scipy.stats        # ✅ explicit import required; bare `scipy.stats` raises NameError
import pandas as pd
import numpy as np

class DataIssueDetector:

    def detect_covariate_drift(self, reference_df, production_df):
        """Check each feature for distribution shift."""
        results = {}
        for col in reference_df.columns:
            if reference_df[col].dtype in ['float64', 'int64']:
                # Numerical features: KS two-sample test
                stat, p = scipy.stats.ks_2samp(
                    reference_df[col].dropna(),
                    production_df[col].dropna()
                )
                results[col] = {
                    "test": "KS",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "drift": p < 0.05
                }
            else:
                # Categorical features: chi-squared contingency test.
                # scipy.stats.chisquare() requires expected frequencies as input
                # and is NOT suitable for raw label arrays. Instead, build a
                # contingency table and use chi2_contingency.
                contingency_table = pd.crosstab(
                    reference_df[col], production_df[col]
                )
                chi2, p, dof, _ = scipy.stats.chi2_contingency(contingency_table)
                results[col] = {
                    "test": "chi2_contingency",
                    "statistic": float(chi2),
                    "p_value": float(p),
                    "dof": int(dof),
                    "drift": p < 0.05
                }
        return results

    def detect_schema_drift(self, reference_schema, production_schema):
        """Detect missing or new columns."""
        ref_cols = set(reference_schema.keys())
        prod_cols = set(production_schema.keys())
        return {
            "missing_columns": ref_cols - prod_cols,
            "new_columns": prod_cols - ref_cols,
            "type_changes": {
                col: (reference_schema[col], production_schema[col])
                for col in ref_cols & prod_cols
                if reference_schema[col] != production_schema[col]
            }
        }
```

### Recommended Fixes
| Data Issue | Fix |
|---|---|
| Feature drift detected | Trigger model retraining with recent data |
| Schema change | Update data pipeline schema validator |
| Missing values spike | Check upstream data source |
| Label distribution shift | Alert for potential concept drift |

---

## Category 2: Model Performance Issues

### What It Is
The ML model's predictive performance degrades in production.

### Key Metrics to Monitor

| Metric | Good Range | Alert Threshold |
|---|---|---|
| Accuracy | >80% | Drop > 5% in 24h |
| F1 Score | >0.75 | Drop > 0.05 in 24h |
| AUC-ROC | >0.85 | Drop > 0.05 in 24h |
| Precision | >0.80 | Drop > 0.05 |
| Recall | >0.80 | Drop > 0.05 |

### Detection: Sliding Window Approach

```python
class ModelPerformanceDetector:
    def __init__(self, window_hours=24, drop_threshold=0.05):
        self.window = window_hours
        self.threshold = drop_threshold

    def check_performance_drop(self, metrics_history: list) -> dict:
        """
        metrics_history: list of {timestamp, accuracy, f1, ...}
        Returns: anomaly dict if drop detected
        """
        if len(metrics_history) < 2:
            return None

        # Compare current to baseline (first in window)
        baseline = metrics_history[0]
        current = metrics_history[-1]

        drops = {}
        for metric in ["accuracy", "f1", "auc"]:
            if metric in baseline and metric in current:
                delta = current[metric] - baseline[metric]
                if delta < -self.threshold:
                    drops[metric] = {
                        "baseline": baseline[metric],
                        "current": current[metric],
                        "drop": abs(delta),
                        "severity": "high" if abs(delta) > 0.15 else "medium"
                    }

        return drops if drops else None
```

### Root Cause Analysis Decision Tree

```
Performance Drop Detected
│
├─ Is input data distribution changed?
│   ├─ YES → Data Drift Issue (→ Retrain)
│   └─ NO → Continue
│
├─ Did model get updated recently?
│   ├─ YES → Regression from new model (→ Rollback)
│   └─ NO → Continue
│
├─ Is ground truth label distribution stable?
│   ├─ NO → Concept Drift (→ Retrain with fresh labels)
│   └─ YES → Continue
│
└─ Is feature importance changed?
    ├─ YES → Feature Importance Shift (→ Feature Engineering)
    └─ NO → Unexplained Degradation (→ Human Investigation)
```

---

## Category 3: LLM Issues (Hallucination & Quality)

### What It Is
The LLM generates text that is irrelevant, incorrect, vague, or harmful.

### Types of LLM Failures

| Failure | Description | Detection Method |
|---|---|---|
| **Hallucination** | Confident but factually wrong | RAGAS faithfulness score |
| **Irrelevance** | Response doesn't answer the query | Semantic similarity score |
| **Vagueness** | Answer is too generic to be useful | Length + specificity check |
| **Toxicity** | Harmful or inappropriate content | Content moderation API |
| **Refusal** | Over-refuses legitimate requests | Response pattern check |
| **Language Confusion** | Responds in wrong language | Language detection |

### Detection Implementation

```python
from sentence_transformers import SentenceTransformer, util

class LLMQualityDetector:
    def __init__(self, llm, relevance_threshold=0.5, min_length=50):
        # ✅ Model is loaded once at construction time, not per call.
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.llm = llm                          # required for hallucination check
        self.relevance_threshold = relevance_threshold
        self.min_length = min_length

    def detect_issues(self, query: str, response: str,
                       context: str = None) -> list[str]:
        issues = []

        # 1. Check query↔response relevance via embedding similarity.
        #    This is a valid use: comparing two SHORT texts (query vs response)
        #    of similar length gives a meaningful similarity signal.
        similarity = self._semantic_similarity(query, response)
        if similarity < self.relevance_threshold:
            issues.append("low_relevance")

        # 2. Check response length (vagueness proxy)
        if len(response.split()) < self.min_length:
            issues.append("too_short")

        # 3. Hallucination detection via LLM-as-a-Judge.
        #    ⚠️  Do NOT use embedding similarity between context and response
        #    to detect hallucinations. A long context and a short response
        #    are semantically diluted (different lengths), and topically
        #    similar hallucinations still score high. Use an LLM judge instead.
        if context:
            verdict = self._llm_faithfulness_check(query, response, context)
            if not verdict.get("faithful", True):
                issues.append("potential_hallucination")

        # 4. Check for uncertainty signals
        uncertainty_phrases = [
            "I'm not sure", "I don't know", "I cannot confirm",
            "may not be accurate", "I apologize"
        ]
        if any(p.lower() in response.lower() for p in uncertainty_phrases):
            issues.append("uncertainty_detected")

        return issues

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.encoder.encode([text1, text2])
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def _llm_faithfulness_check(self, query: str, response: str,
                                  context: str) -> dict:
        """Ask the LLM to judge whether the response is grounded in context.

        This is the only reliable automated method for hallucination detection
        without deploying a dedicated NLI model (e.g., True-NLI).
        """
        prompt = f"""You are a factual accuracy evaluator.

CONTEXT (ground truth):
{context}

RESPONSE TO EVALUATE:
{response}

Is every factual claim in the RESPONSE directly supported by the CONTEXT?
Reply ONLY with valid JSON: {{"faithful": true/false, "reason": "brief explanation"}}"""
        import json
        raw = self.llm.invoke(prompt).content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fail safe: treat parse errors as uncertain (not a hallucination flag)
            return {"faithful": True, "reason": "parse_error"}
```

### Automatic Fix Strategies

```python
class LLMFixer:
    def __init__(self, llm):
        self.llm = llm

    def optimize_prompt(self, original_prompt: str, 
                         issue_type: str) -> str:
        """Automatically rewrite a failing prompt."""
        strategies = {
            "low_relevance": "Add explicit instructions to stay on topic",
            "potential_hallucination": "Add: Answer ONLY from the provided context",
            "too_short": "Add: Provide a detailed, comprehensive answer",
            "uncertainty_detected": "Add: Use only information you are certain about"
        }

        strategy = strategies.get(issue_type, "Improve clarity")

        rewrite_prompt = f"""
        Original prompt: {original_prompt}
        Issue: {issue_type}
        Strategy to apply: {strategy}

        Rewrite the prompt to fix this issue. Return ONLY the new prompt.
        """
        return self.llm.invoke(rewrite_prompt).content
```

---

## Category 4: Retrieval Issues (RAG Failures)

### What It Is
The vector search returns irrelevant, empty, or low-quality documents.

### Detection Metrics

| Metric | Definition | Alert When |
|---|---|---|
| **Top-1 Similarity** | Cosine sim of best result | < 0.6 |
| **Result Count** | Number of documents returned | 0 (empty results) |
| **Context Diversity** | Similarity between returned docs | > 0.95 (duplicates) |
| **Retrieval Latency** | Time to retrieve | > 500ms |

### Detection Implementation

```python
from sentence_transformers import SentenceTransformer, util

class RetrievalQualityDetector:
    def __init__(self, min_similarity=0.6, min_results=2,
                 duplicate_threshold=0.95):
        self.min_similarity = min_similarity
        self.min_results = min_results
        self.duplicate_threshold = duplicate_threshold
        # Encoder used for pairwise document similarity (duplicate check)
        self._encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def evaluate_retrieval(self, query: str,
                            results: list,
                            scores: list) -> dict:
        """Assess retrieval quality and flag detected issues.

        Args:
            results: list of retrieved document strings.
            scores:  cosine similarity of each document to the QUERY.
        """
        issues = []

        # 1. Empty results
        if len(results) == 0:
            issues.append("no_results")
            return {"issues": issues, "severity": "critical"}

        # 2. Low top-1 similarity to query
        if scores[0] < self.min_similarity:
            issues.append("low_relevance")

        # 3. Too few results
        if len(results) < self.min_results:
            issues.append("insufficient_results")

        # 4. Duplicate content detection.
        #    ⚠️  Do NOT use coefficient of variation of query-similarity scores.
        #    If three relevant but different docs all score 0.91, 0.90, 0.89,
        #    their low CV incorrectly triggers a duplicate flag.
        #    Instead, compare the document embeddings against EACH OTHER.
        if len(results) > 1:
            doc_embeddings = self._encoder.encode(results)
            for i in range(len(doc_embeddings)):
                for j in range(i + 1, len(doc_embeddings)):
                    pairwise_sim = float(
                        util.cos_sim(doc_embeddings[i], doc_embeddings[j])
                    )
                    if pairwise_sim > self.duplicate_threshold:
                        issues.append("duplicate_results")
                        break  # one duplicate pair is enough to flag
                if "duplicate_results" in issues:
                    break

        return {
            "issues": issues,
            "top_score": scores[0] if scores else 0,
            "num_results": len(results),
            "severity": "high" if "no_results" in issues else "medium"
        }
```

### Fix Strategies

| Retrieval Issue | Fix |
|---|---|
| No results | Expand query (HyDE or query expansion) |
| Low similarity | Re-index with better chunking strategy |
| Duplicate results | Switch to MMR retrieval |
| Stale knowledge | Trigger knowledge base re-ingestion |

---

## Category 5: System / Infrastructure Issues

### What It Is
The AI pipeline fails due to infrastructure, API, or configuration problems.

### Monitored System Events

```python
SYSTEM_FAILURE_PATTERNS = {
    "api_timeout": {
        "pattern": "HTTPTimeoutError|TimeoutError|ReadTimeout",
        "severity": "high",
        "fix": "exponential_backoff"
    },
    "rate_limit": {
        "pattern": "RateLimitError|429",
        "severity": "medium",
        "fix": "implement_retry_with_backoff"
    },
    "out_of_memory": {
        "pattern": "OOMError|MemoryError|CUDA out of memory",
        "severity": "critical",
        "fix": "reduce_batch_size_or_scale_up"
    },
    "invalid_api_key": {
        "pattern": "AuthenticationError|401|Invalid API key",
        "severity": "critical",
        "fix": "rotate_api_key"
    },
    "model_not_found": {
        "pattern": "ModelNotFoundError|404|model_not_found",
        "severity": "high",
        "fix": "check_model_name_or_fallback"
    }
}

class SystemIssueDetector:
    def detect_from_log(self, log_message: str) -> dict | None:
        for issue_type, config in SYSTEM_FAILURE_PATTERNS.items():
            import re
            if re.search(config["pattern"], log_message):
                return {
                    "type": issue_type,
                    "severity": config["severity"],
                    "suggested_fix": config["fix"],
                    "log_snippet": log_message[:200]
                }
        return None
```

---

## Unified Anomaly Event Schema

All detected anomalies produce a standardized event:

```python
{
    "event_id": "uuid4",
    "detected_at": "2024-10-15T09:23:11Z",
    "system_id": "product-recommender-v2",

    # Classification
    "failure_category": "model_issue",  # data/model/llm/retrieval/system
    "failure_subcategory": "accuracy_drop",
    "severity": "high",  # low/medium/high/critical

    # Evidence
    "evidence": {
        "metric_name": "accuracy",
        "current_value": 0.61,
        "baseline_value": 0.87,
        "change_pct": -29.9
    },

    # Context for RAG
    "context_for_rag": "Accuracy dropped from 87% to 61% in 24 hours. Feature drift detected in 'user_age' column (KS p-value: 0.003).",

    # Status
    "status": "pending_diagnosis",  # pending/diagnosing/diagnosed/fixing/resolved
    "diagnosis": null,
    "fix_applied": null
}
```
