# 📚 RAG System Deep Dive — Aegis AI

---

## Why RAG Is Central to Aegis AI

The LLM debugging agent, if used without RAG, would:
1. **Hallucinate** fix recommendations
2. **Forget** past incidents that were already solved
3. **Miss** domain-specific knowledge (your specific system's quirks)

**RAG grounds the LLM's reasoning in real, verified knowledge.**

---

## What Gets Stored in the Knowledge Base

### Category 1: Past Incident Logs (Gold Standard)

Every time Aegis AI successfully diagnoses and fixes an issue, the incident is stored as a **structured memory**:

```json
{
    "incident_id": "INC-2024-001",
    "timestamp": "2024-10-15T09:23:00Z",
    "system_id": "product-recommender-v2",
    "failure_type": "accuracy_drop",
    "anomaly_description": "Model accuracy dropped from 87% to 61% over 3 days",
    "root_cause_analysis": "Feature 'user_age' distribution shifted significantly. New data source (mobile app) brought younger demographic not seen in training data.",
    "fix_applied": "Triggered retraining pipeline with last 30-day data window. Updated feature scaling for 'user_age'.",
    "outcome": "Accuracy recovered to 84% within 2 retraining cycles.",
    "tags": ["data_drift", "feature_shift", "retrain", "age_feature"]
}
```

**Effect:** When the next accuracy drop occurs, the RAG system retrieves this past incident, and the agent immediately knows to look at feature distributions.

---

### Category 2: Debugging Guides

Curated knowledge documents about common ML failure patterns:

```markdown
# LLM Hallucination: Causes and Fixes

## What It Is
Hallucination occurs when an LLM generates confident, plausible-sounding
but factually incorrect information not supported by its training data
or provided context.

## Common Causes
1. **Insufficient context**: The prompt lacks grounding facts
2. **Ambiguous query**: Multiple valid interpretations exist
3. **Knowledge cutoff**: LLM doesn't know recent events
4. **High temperature**: Too much randomness in generation

## Detection Methods
- Semantic similarity between response and known facts < 0.5
- RAGAS faithfulness score < 0.7
- LLM-as-judge rates response as "not grounded"

## Fix Strategies
1. **Add RAG context**: Inject relevant retrieved documents into prompt
2. **Lower temperature**: Reduce to 0.1-0.3 for factual tasks
3. **Use structured output**: Force JSON with specific fields
4. **Add self-check**: Ask LLM to verify its own claims
5. **Cite sources**: Instruct LLM to reference provided documents only

## Prompt Template (Anti-Hallucination)
Answer ONLY based on the context below. If you don't know, say "I don't have enough information."

Context: {context}
Question: {question}
```

---

### Category 3: Prompt Engineering Rules

```markdown
# Prompt Engineering Best Practices for Reliability

## Rule 1: Be Specific About Output Format
❌ Bad: "Summarize this article"
✅ Good: "Summarize in exactly 3 bullet points, each under 20 words"

## Rule 2: Use Role Prompting
✅ "You are an expert ML engineer diagnosing a production failure."

## Rule 3: Provide Examples (Few-shot)
Show 2-3 examples of the expected input/output format.

## Rule 4: Chain-of-Thought for Reasoning
Add: "Think step by step before giving your final answer."

## Rule 5: Set Explicit Boundaries
"Answer only about [topic]. If asked about anything else, say you can't help."

## Rule 6: Test with Adversarial Inputs
Try edge cases: empty inputs, very long inputs, inputs in other languages.
```

---

### Category 4: System-Specific Patterns

Templates for your monitored systems:

```json
{
    "system_id": "rag-chatbot-v1",
    "known_failure_patterns": [
        {
            "pattern": "Response contains 'I don't know' more than 30% of queries",
            "likely_cause": "Knowledge base is outdated or retrieval quality is poor",
            "first_check": "Run vector DB query directly and inspect top-5 results"
        },
        {
            "pattern": "Latency > 5 seconds consistently",
            "likely_cause": "Embedding model or vector DB overloaded",
            "first_check": "Check ChromaDB collection size and query time separately"
        }
    ]
}
```

---

## RAG Pipeline Implementation Details

### Step 1: Document Preprocessing

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optimal chunk settings for debugging knowledge
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # Smaller chunks = more precise retrieval
    chunk_overlap=100,       # Overlap preserves context at boundaries
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    # Split on headings first, then paragraphs, then sentences
)
```

**Why these settings:**
- 800 tokens is small enough for precise retrieval but large enough for context
- Heading-aware splitting keeps semantically related content together
- Overlap prevents losing context when a concept spans two chunks

---

### Step 2: Embedding Strategy

#### Option A: OpenAI Embeddings (High Quality)
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dims, good quality/cost ratio
    dimensions=512  # Can reduce for faster search with slight quality loss
)
```

#### Option B: Local Embeddings (Free)
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # Better than MiniLM for retrieval
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

**Recommendation:** Use `BAAI/bge-small-en-v1.5` for development (free, good quality). It outperforms `all-MiniLM-L6-v2` on retrieval benchmarks.

---

### Step 3: Advanced Retrieval Strategies

#### MMR (Maximal Marginal Relevance) — Prevents Duplicate Results
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,        # Return 5 documents
        "fetch_k": 20, # Consider 20 candidates
        "lambda_mult": 0.7  # 0=max diversity, 1=max relevance
    }
)
```

#### Contextual Compression — Extract Only Relevant Parts
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
# Returns only the relevant sentences from each document
```

#### Self-Query — LLM Generates Filter Conditions
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

# LLM automatically generates: {"category": "llm_issue", "severity": "high"}
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Debugging guides for ML/LLM issues",
    metadata_field_info=metadata_field_info
)
```

---

### Step 4: Retrieval-Augmented Prompt Construction

```python
def build_diagnosis_prompt(anomaly: dict, retrieved_docs: list) -> str:
    context_text = "\n\n---\n\n".join([
        f"SOURCE: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    return f"""
You are Aegis, an expert autonomous AI debugging system.

## RETRIEVED KNOWLEDGE BASE CONTEXT:
{context_text}

## ANOMALY DETECTED:
- System ID: {anomaly['system_id']}
- Metric: {anomaly['metric_type']}
- Current Value: {anomaly['current_value']}
- Baseline: {anomaly['baseline_value']}
- Change: {anomaly['change_pct']}%
- Timestamp: {anomaly['timestamp']}

## ANALYSIS INSTRUCTIONS:
1. Review the knowledge base context above carefully
2. Identify the most likely root cause based on patterns
3. Classify the failure type (data/model/prompt/retrieval/system)
4. Generate a specific, actionable fix recommendation
5. Reference the knowledge source for your recommendation
6. Estimate your confidence (0-100%)

## OUTPUT FORMAT (JSON):
{{
    "root_cause": "...",
    "failure_category": "model_issue|data_issue|prompt_issue|retrieval_issue|system_issue",
    "detailed_analysis": "...",
    "fix_recommendation": {{
        "action": "...",
        "steps": ["step 1", "step 2"],
        "urgency": "immediate|scheduled|monitor"
    }},
    "confidence_score": 85,
    "knowledge_references": ["source 1", "source 2"]
}}
"""
```

---

## RAG Quality Evaluation

### Using RAGAS Framework

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

def evaluate_rag_quality(test_cases):
    """
    Evaluate RAG pipeline quality using RAGAS.

    Each test case must have the form:
      {
        "question":    str,
        "answer":      str,        # LLM-generated response
        "contexts":    list[str],  # ✅ plain-text strings, NOT Document objects
        "ground_truth": str        # (required by context_recall)
      }

    ⚠️  RAGAS defaults to OpenAI APIs for evaluation. If your project uses
    Groq or Ollama, configure a custom LLM before calling evaluate():

        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile"))
        # Then pass: evaluate(dataset, metrics=[...], llm=ragas_llm)
    """
    # ✅ Extract page_content strings if callers pass LangChain Document objects
    normalised = []
    for tc in test_cases:
        normalised.append({
            **tc,
            "contexts": [
                doc.page_content if hasattr(doc, "page_content") else doc
                for doc in tc["contexts"]
            ]
        })

    dataset = Dataset.from_list(normalised)
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,       # Is answer grounded in context?
            answer_relevancy,   # Is answer relevant to question?
            context_precision,  # Did we retrieve the right context?
            context_recall      # Did we retrieve ALL needed context?
        ]
    )
    return results

# Target scores for Aegis AI
QUALITY_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70
}
```

### Manual Spot-Check Protocol

For each failure category, maintain a set of **golden test queries**:

```python
GOLDEN_TESTS = {
    "accuracy_drop": {
        "query": "ML model accuracy suddenly dropped from 85% to 60%",
        "expected_in_top_3": ["data_drift_guide.md", "retrain_checklist.md"],
        "expected_keywords_in_answer": ["drift", "retrain", "distribution"]
    },
    "llm_hallucination": {
        "query": "LLM is giving confident but wrong answers",
        "expected_in_top_3": ["hallucination_guide.md"],
        "expected_keywords_in_answer": ["rag", "context", "temperature", "grounding"]
    }
}
```

---

## Knowledge Base Maintenance

### When to Update
1. **After every resolved incident** — add the incident record
2. **When new failure patterns discovered** — add debugging guide
3. **When a fix didn't work** — update with what DID work
4. **Quarterly review** — verify guides are still current

### Version Control
- Store all knowledge base documents in Git
- Use semantic versioning for guides (v1.0, v1.1, etc.)
- Re-index ChromaDB after any knowledge base update

### Ingestion Pipeline (Automated)
```python
# Run this script when knowledge base is updated
python scripts/rebuild_knowledge_base.py --path data/knowledge_base/ --reset
```
