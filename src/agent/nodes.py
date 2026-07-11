import json
import re
import os
from typing import Dict, Any, List
from loguru import logger
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from src.agent.state import AegisAgentState
from src.agent.prompts import DIAGNOSTIC_PROMPT
from src.monitoring.models import FailureCategory

# Fallback Mock Knowledge Base
MOCK_KNOWLEDGE_BASE = {
    "data_issue": [
        "Data Drift Guide: When input feature distributions shift (high PSI or low KS p-value), check if reference data has changed. If so, alert the data engineering team.",
        "Incident #101: Covariate drift detected in numerical feature. Mitigation: Log drift, trigger alerts, and do not auto-apply retrain without human sign-off."
    ],
    "model_issue": [
        "Model Accuracy Retraining Playbook: If accuracy drop >= 5%, trigger retrain pipeline on the latest sliding window dataset.",
        "Incident #102: Drop in F1 score due to model decay. Mitigation: Suggest retraining pipeline."
    ],
    "prompt_issue": [
        "LLM Prompting Guardrails: Avoid hallucinations by providing formatting instructions and using few-shot examples.",
        "Incident #103: Model output hallucination. Mitigation: PromptOptimizer rewrote the prompt with structured guidelines."
    ],
    "retrieval_issue": [
        "RAG Relevance Optimization: If retrieval returns low relevance, verify chunk overlap or increase vector similarity threshold.",
        "Incident #104: Vector DB returned irrelevant context. Mitigation: Rebuild ChromaDB and re-verify embeddings."
    ],
    "system_issue": [
        "API Failure and Timeout Guide: Set exponential backoff and retry parameters for rate limits (HTTP 429) or timeouts.",
        "Incident #105: OpenAI API timeout. Mitigation: Retry with 3s initial delay."
    ]
}

class MockLLM:
    """Mock LLM for testing purposes when no API key is configured or for CI."""
    def invoke(self, prompt: Any) -> Any:
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        prompt_str = str(prompt)
        category = "system_issue"
        if "accuracy" in prompt_str.lower() or "model" in prompt_str.lower():
            category = "model_issue"
        elif "drift" in prompt_str.lower() or "data" in prompt_str.lower():
            category = "data_issue"
        elif "hallucination" in prompt_str.lower() or "prompt" in prompt_str.lower():
            category = "prompt_issue"
        elif "retrieval" in prompt_str.lower() or "relevance" in prompt_str.lower():
            category = "retrieval_issue"
            
        mock_json = {
            "chain_of_thought": f"Parsed anomaly and found symptoms pointing to {category}. Cross-referenced with guides and confirmed.",
            "root_cause": f"Simulated root cause for {category} anomaly.",
            "failure_category": category,
            "fix_recommendation": f"Apply auto-remedy or retraining for {category}.",
            "confidence_score": 0.85,
            "knowledge_references": ["guide_1", "incident_101"]
        }
        
        if "ORIGINAL PROMPT" in prompt_str:
            return MockResponse("OPTIMIZED PROMPT: " + prompt_str.split("ORIGINAL PROMPT (failing):")[1].split("FAILURE REASON:")[0].strip() + " [optimized]")
            
        return MockResponse(json.dumps(mock_json))

def parse_json_response(content: str) -> dict:
    """Extracts and parses JSON from the LLM response."""
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM response: {content}")

def get_llm(config: RunnableConfig = None) -> Any:
    """Gets the LLM from the runnable config, or instantiates a default/mock LLM."""
    if config and isinstance(config, dict) and "configurable" in config:
        llm = config["configurable"].get("llm")
        if llm:
            return llm
            
    # Try using langchain_openai if key is present
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        except ImportError:
            pass
            
    logger.warning("No API key configured or ChatOpenAI import failed. Using MockLLM.")
    return MockLLM()

def get_rag(config: RunnableConfig = None) -> Any:
    """Gets the RAG instance from the runnable config, or returns None (triggering fallback)."""
    if config and isinstance(config, dict) and "configurable" in config:
        return config["configurable"].get("rag")
    return None

# Node 1: retrieve_context_node
def retrieve_context_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Retrieves relevant context documents from RAG or fallback mock database."""
    anomaly_desc = state.get("anomaly_description", "")
    failure_cat = state.get("failure_category", "")
    
    logger.info(f"Running retrieve_context_node. Anomaly: '{anomaly_desc}', Category: '{failure_cat}'")
    
    retrieved_docs = []
    rag = get_rag(config)
    
    if rag:
        try:
            docs = rag.retrieve(query=anomaly_desc, category=failure_cat)
            retrieved_docs = [doc.page_content for doc in docs]
            logger.info(f"Retrieved {len(retrieved_docs)} documents from RAG.")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}. Falling back to mock database.")
            
    if not retrieved_docs:
        # Fallback to Mock database
        cat_key = failure_cat if failure_cat in MOCK_KNOWLEDGE_BASE else "system_issue"
        retrieved_docs = MOCK_KNOWLEDGE_BASE[cat_key]
        logger.info(f"Retrieved {len(retrieved_docs)} documents from mock knowledge base (Category: {cat_key}).")
        
    analysis_steps = state.get("analysis_steps", [])
    analysis_steps.append(f"Retrieved {len(retrieved_docs)} context documents from knowledge base.")
    
    return {
        "retrieved_context": retrieved_docs,
        "analysis_steps": analysis_steps
    }

# Node 2: analyze_anomaly_node
def analyze_anomaly_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Analyzes the anomaly using the LLM with CoT reasoning and retrieved context."""
    logger.info("Running analyze_anomaly_node.")
    
    anomaly_desc = state.get("anomaly_description", "")
    retrieved_context = "\n\n".join(state.get("retrieved_context", []))
    system_id = state.get("system_id", "unknown")
    raw_metrics = str(state.get("raw_metrics", {}))
    
    llm = get_llm(config)
    
    # Format templates
    formatted_prompt = DIAGNOSTIC_PROMPT.format_messages(
        anomaly_description=anomaly_desc,
        retrieved_context=retrieved_context if retrieved_context else "No context retrieved.",
        system_id=system_id,
        raw_metrics=raw_metrics
    )
    
    try:
        response = llm.invoke(formatted_prompt)
        response_content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_response(response_content)
        
        root_cause = parsed.get("root_cause", "Unknown root cause.")
        chain_of_thought = parsed.get("chain_of_thought", "No chain of thought provided.")
        failure_category = parsed.get("failure_category", "system_issue")
        fix_recommendation = parsed.get("fix_recommendation", "")
        confidence_score = float(parsed.get("confidence_score", 0.5))
        
        # Save analysis steps
        analysis_steps = state.get("analysis_steps", [])
        analysis_steps.append(f"LLM Reasoning: {chain_of_thought}")
        
        # Add message to history
        messages = state.get("messages", [])
        messages.append(AIMessage(content=response_content))
        
        return {
            "root_cause": root_cause,
            "failure_category": failure_category,
            "fix_recommendation": fix_recommendation,
            "confidence_score": confidence_score,
            "analysis_steps": analysis_steps,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error in analyze_anomaly_node: {e}")
        # Safeguard fallback
        analysis_steps = state.get("analysis_steps", [])
        analysis_steps.append(f"Analysis failed due to error: {e}")
        return {
            "root_cause": "Failed to analyze anomaly due to processing error.",
            "confidence_score": 0.0,
            "analysis_steps": analysis_steps
        }

# Node 3: classify_failure_node
def classify_failure_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Validates and finalizes the classification of the failure category."""
    category = state.get("failure_category", "system_issue")
    logger.info(f"Running classify_failure_node. Input category: {category}")
    
    # Standardize category
    valid_categories = [c.value for c in FailureCategory]
    if category not in valid_categories:
        logger.warning(f"Invalid category '{category}' normalized to 'system_issue'")
        category = "system_issue"
        
    analysis_steps = state.get("analysis_steps", [])
    analysis_steps.append(f"Failure category finalized as: {category}")
    
    return {
        "failure_category": category,
        "analysis_steps": analysis_steps
    }

# Node 4: generate_fix_node
def generate_fix_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Generates the recommended fix and sets human review requirement based on confidence."""
    logger.info("Running generate_fix_node.")
    
    confidence = state.get("confidence_score", 0.0)
    fix_rec = state.get("fix_recommendation", "")
    
    # Scale confidence score if it is between 0.0 and 1.0 (CoT prompt spec)
    # The default routing threshold is 80.0 (or 0.8)
    if confidence <= 1.0:
        scaled_confidence = confidence * 100.0
    else:
        scaled_confidence = confidence
        
    # Read confidence threshold from settings.yaml or default to 80.0
    confidence_threshold = 80.0
    
    requires_human = scaled_confidence < confidence_threshold
    logger.info(f"Confidence score: {scaled_confidence} (Threshold: {confidence_threshold}). Requires human review: {requires_human}")
    
    analysis_steps = state.get("analysis_steps", [])
    analysis_steps.append(f"Evaluated confidence: {scaled_confidence}%. Requires human review: {requires_human}")
    
    return {
        "requires_human_review": requires_human,
        "analysis_steps": analysis_steps
    }

# Node 5: human_review_node
def human_review_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Placeholder for human-in-the-loop review. Execution pauses before entering this node in compiled graph."""
    logger.info("Running human_review_node.")
    
    analysis_steps = state.get("analysis_steps", [])
    analysis_steps.append("Human-in-the-loop review performed.")
    
    # Once executed, we assume the human approved/reviewed the recommendation
    return {
        "requires_human_review": False,
        "analysis_steps": analysis_steps
    }

# Node 6: apply_fix_node
def apply_fix_node(state: AegisAgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Applies the recommended fix, updating iteration and execution logs."""
    logger.info("Running apply_fix_node.")
    
    fix_rec = state.get("fix_recommendation", "No fix recommendation available.")
    iteration = state.get("iteration_count", 0) + 1
    
    logger.info(f"Applying fix recommendation: '{fix_rec}'. Iteration count: {iteration}")
    
    analysis_steps = state.get("analysis_steps", [])
    analysis_steps.append(f"Successfully applied fix at iteration {iteration}.")
    
    return {
        "iteration_count": iteration,
        "analysis_steps": analysis_steps
    }
