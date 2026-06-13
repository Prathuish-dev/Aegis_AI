# Prompt Engineering Guidelines for Reliability

## 1. Introduction and Objectives
In an autonomous system like Aegis AI, prompts are not mere conversation starters—they are **functional programming interfaces** for Large Language Models. If a prompt is poorly structured, the model might produce unstructured text when JSON is required, miss critical context, or output incorrect repair code.

To ensure consistent output across diverse LLM providers (OpenAI, Groq, Ollama), all prompt templates in Aegis AI must adhere to the formatting rules, structure guidelines, and anti-patterns detailed in this document.

---

## 2. Core Prompting Principles

### Rule 1: Enforce Structured Output (JSON Schema)
Never ask the LLM to output a format without specifying the exact JSON schema. Always request a raw JSON block, and specify how the model should behave if fields are unknown.
- **Good:** "Return a JSON object containing the keys `root_cause` (string) and `confidence` (float between 0 and 1). Output nothing but the raw JSON object."
- **Bad:** "Output your analysis as JSON."

### Rule 2: Grounding and Boundary Setting
To prevent hallucinations, explicitly constrain the LLM's knowledge search space to the provided context. Define a fallback response when the answer cannot be found in the context.
- **Good:** "Answer the user question using ONLY the retrieved facts below. If the facts do not contain the answer, output: 'INSUFFICIENT_CONTEXT'. Do not assume or guess."
- **Bad:** "Use the context to answer the question, but feel free to add your own knowledge if needed."

### Rule 3: Use Role Prompting
Establish clear system personas to set the domain context. This optimizes the model's token attention weights for professional debugging.
- **Good:** "You are Aegis, a senior site reliability engineer and autonomous self-healing agent. Your goal is to analyze metric failures and propose surgical fixes."

### Rule 4: Chain-of-Thought (CoT) Reasoning
For complex diagnosis tasks, instruct the LLM to write down its reasoning steps before outputting the final JSON properties. This prevents the model from choosing a category or action before processing the evidence.
- **Implementation:** Include a `reasoning` or `detailed_analysis` field as the **first** key in the expected JSON object, forcing the model to generate its thought tokens first.

### Rule 5: Few-Shot Prompting for Complex Rules
Provide 2-3 examples of the input-output mapping within the prompt structure. Few-shot examples are highly effective for correcting edge-case formatting issues.

---

## 3. Production-Ready Prompt Template Example

This template demonstrates the layout of a diagnosis prompt:

```markdown
You are Aegis, an expert autonomous AI debugging system.

## RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

## ANOMALY DETECTED:
- System ID: {system_id}
- Metric: {metric_type}
- Current Value: {current_value}
- Baseline: {baseline_value}
- Change: {change_pct}%
- Timestamp: {timestamp}

## ANALYSIS INSTRUCTIONS:
1. Review the knowledge base context above carefully.
2. Identify the most likely root cause based on historical patterns.
3. Classify the failure category.
4. Generate a specific, actionable fix recommendation.
5. Estimate your confidence (0-100%).
6. Output your response as a single, valid JSON block.

## OUTPUT FORMAT (JSON):
{{
    "detailed_analysis": "Step-by-step reasoning explaining the analysis of the anomaly.",
    "root_cause": "A short sentence explaining the physical root cause.",
    "failure_category": "model_issue | data_issue | prompt_issue | retrieval_issue | system_issue",
    "fix_recommendation": {{
        "action": "The command or setting change to apply.",
        "steps": ["Step 1 to perform", "Step 2 to perform"],
        "urgency": "immediate | scheduled | monitor"
    }},
    "confidence_score": 85
}}
```

---

## 4. Common Prompt Anti-Patterns

1. **Negative Constraints:** Telling the model what *not* to do (e.g., "Do not include any explanation") is less effective than telling it exactly what *to* do (e.g., "Output only the raw JSON").
2. **Cognitive Overload:** Mixing model debugging instructions, RAG retrieval instructions, and database commands in a single prompt. Split these tasks into separate nodes within the LangGraph agent.
3. **Implicit Defaults:** Relying on the model to infer defaults. Always specify the default behavior for missing inputs or low-confidence outcomes.
4. **Context Ingestion Leakage:** Allowing unstructured user input or logs to run directly in the prompt without separator tags, which can trigger prompt injection vulnerabilities. Always wrap logs in XML-like tags (e.g., `<log_payload>...</log_payload>`).
