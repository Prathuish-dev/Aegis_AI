import os
import json
import re
from typing import Any, List, Dict
from loguru import logger

class LLMQualityDetector:
    """
    LLMQualityDetector acts as an LLM-as-a-Judge to evaluate the quality (specifically faithfulness)
    of an LLM response relative to retrieved RAG context.
    """
    def __init__(self, llm: Any = None):
        """
        Initialize the LLMQualityDetector.
        
        Args:
            llm: A LangChain-compatible LLM instance that supports .invoke().
        """
        if llm is None:
            llm = self._get_default_llm()
        self.llm = llm
        logger.info("LLMQualityDetector initialized.")

    def _get_default_llm(self) -> Any:
        """Load default chat LLM based on environment variables or fallback to a dummy."""
        if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            except ImportError:
                pass
                
        if os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
            except ImportError:
                pass
                
        # Create a mock chat model for evaluation testing
        class DummyChatModel:
            def invoke(self, prompt, *args, **kwargs):
                class DummyMessage:
                    content = '{"faithful": true, "reason": "Mocked response shows that the output is grounded."}'
                return DummyMessage()
        return DummyChatModel()

    def detect_quality(self, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Runs the LLM-as-a-Judge faithfulness check.
        
        Args:
            question: The user input query.
            answer: The generated LLM response.
            contexts: List of retrieved context string snippets.
            
        Returns:
            Dict containing:
            - faithful: bool (True if the answer is grounded in the contexts, False otherwise)
            - reason: str (Explanation of the decision)
        """
        if not contexts:
            return {
                "faithful": False,
                "reason": "No retrieved contexts provided to ground the answer."
            }
            
        context_text = "\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are an expert quality assurance evaluator acting as an LLM-as-a-Judge.
Your task is to determine whether the GENERATED ANSWER is FAITHFUL to the RETRIEVED CONTEXTS.
An answer is faithful if and only if all claims, facts, and statements in it can be directly inferred from or are supported by the retrieved contexts.
If the answer contains any information, assumptions, or claims not mentioned in the context (hallucinations), it is NOT faithful.

RETRIEVED CONTEXTS:
{context_text}

USER QUESTION:
{question}

GENERATED ANSWER:
{answer}

Evaluate carefully and output your response in JSON format.
Your output must contain exactly two fields:
- "faithful": true or false
- "reason": A detailed explanation of your evaluation, referencing specific contexts or pointing out hallucinations.

Response MUST be valid JSON only. Do not include markdown code blocks or any explanation outside the JSON.
"""

        try:
            response = self.llm.invoke(prompt)
            
            # Extract raw text
            if hasattr(response, "content"):
                raw_text = str(response.content).strip()
            else:
                raw_text = str(response).strip()
                
            return self._parse_json_response(raw_text)
            
        except Exception as e:
            logger.error(f"Error executing LLMQualityDetector: {e}")
            return {
                "faithful": False,
                "reason": f"Execution error: {str(e)}"
            }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Helper to parse JSON output and fallback safely on formatting errors."""
        # Remove markdown code fences if present
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = cleaned.strip()
        
        try:
            data = json.loads(cleaned)
            # Ensure keys exist and are of correct types
            faithful = bool(data.get("faithful", False))
            reason = str(data.get("reason", "No reason provided."))
            return {
                "faithful": faithful,
                "reason": reason
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLMQualityDetector JSON output: {e}. Attempting recovery.")
            
            # Simple fallback heuristic parsers
            faithful_match = re.search(r'"faithful"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
            
            if faithful_match:
                faithful_val = faithful_match.group(1).lower() == "true"
                reason_val = reason_match.group(1) if reason_match else f"Recovered from raw text: {cleaned[:100]}"
                return {
                    "faithful": faithful_val,
                    "reason": reason_val
                }
                
            return {
                "faithful": False,
                "reason": f"Failed to parse LLM-as-a-Judge response as JSON. Raw response: {text[:200]}"
            }
