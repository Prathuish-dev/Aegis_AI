import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.agent.graph import build_debugging_agent
from src.agent.state import AegisAgentState
from src.agent.nodes import MockLLM
from src.monitoring.models import FailureCategory

def run_simulation(anomaly_desc: str, failure_category: str, system_id: str, raw_metrics: dict, mock_confidence: float) -> dict:
    """Simulates agent graph execution for a specific failure scenario."""
    # Define custom Mock LLM to return specific confidence and failure type
    class ScenarioMockLLM:
        def invoke(self, prompt: str, *args, **kwargs):
            class Content:
                content = json.dumps({
                    "chain_of_thought": f"Reasoning about {anomaly_desc} under {failure_category} category.",
                    "root_cause": f"Simulated root cause for {anomaly_desc}.",
                    "failure_category": failure_category,
                    "fix_recommendation": f"Auto-heal or retrain action for {failure_category}.",
                    "confidence_score": mock_confidence,
                    "knowledge_references": ["golden_guide_doc", "incident_ref"]
                })
            return Content()

    # Compile the graph
    app = build_debugging_agent()
    
    # Configure the custom mock LLM for this run
    config = {
        "configurable": {
            "thread_id": f"thread_{failure_category}",
            "llm": ScenarioMockLLM()
        }
    }
    
    initial_state = AegisAgentState(
        anomaly_description=anomaly_desc,
        system_id=system_id,
        failure_category="",
        raw_metrics=raw_metrics,
        retrieved_context=[],
        analysis_steps=[],
        root_cause="",
        fix_recommendation="",
        confidence_score=0.0,
        requires_human_review=False,
        iteration_count=0,
        messages=[]
    )
    
    # Execute the graph
    state = app.invoke(initial_state, config=config)
    
    # Check if the graph paused before human_review
    next_step = app.get_state(config).next
    if next_step == ("human_review",):
        # Resume the graph (simulating human approval)
        state = app.invoke(None, config=config)
        state["was_paused"] = True
    else:
        state["was_paused"] = False
        
    return state

def main():
    print("=" * 80)
    print(" Aegis AI LLM Debugging Agent End-to-End Simulation")
    print("=" * 80)
    
    scenarios = [
        {
            "name": "Data Drift Scenario (Low Confidence)",
            "desc": "Covariate drift detected in numerical feature 'age' (PSI = 0.35)",
            "category": "data_issue",
            "system_id": "churn_predictor",
            "metrics": {"feature": "age", "psi": 0.35},
            "confidence": 0.65  # Will trigger Human-in-the-Loop pause (< 0.80)
        },
        {
            "name": "Model Performance Scenario (High Confidence)",
            "desc": "Accuracy dropped by 8% over the past 24 hours",
            "category": "model_issue",
            "system_id": "recommendation_engine",
            "metrics": {"accuracy_drop": 0.08},
            "confidence": 0.85  # Will bypass Human-in-the-Loop pause (>= 0.80)
        },
        {
            "name": "LLM Hallucination Scenario (Low Confidence)",
            "desc": "Faithfulness score fell below threshold of 0.85",
            "category": "prompt_issue",
            "system_id": "customer_support_llm",
            "metrics": {"faithfulness_score": 0.68},
            "confidence": 0.72  # Will trigger Human-in-the-Loop pause (< 0.80)
        },
        {
            "name": "RAG Context Mismatch Scenario (High Confidence)",
            "desc": "Low similarity score between retrieved documents and query",
            "category": "retrieval_issue",
            "system_id": "search_assistant",
            "metrics": {"relevance_score": 0.42},
            "confidence": 0.90  # Will bypass Human-in-the-Loop pause (>= 0.80)
        },
        {
            "name": "System Failure Scenario (Low Confidence)",
            "desc": "CUDA out of memory error during batch processing",
            "category": "system_issue",
            "system_id": "image_classifier",
            "metrics": {"error": "OutOfMemory"},
            "confidence": 0.55  # Will trigger Human-in-the-Loop pause (< 0.80)
        }
    ]
    
    all_passed = True
    
    for i, sc in enumerate(scenarios, 1):
        print(f"\n--- Running Scenario {i}: {sc['name']} ---")
        print(f"Anomaly: {sc['desc']}")
        print(f"System:  {sc['system_id']}")
        
        try:
            state = run_simulation(
                anomaly_desc=sc["desc"],
                failure_category=sc["category"],
                system_id=sc["system_id"],
                raw_metrics=sc["metrics"],
                mock_confidence=sc["confidence"]
            )
            
            # Print simulation outcomes
            print(f"Result Category:  {state['failure_category']}")
            print(f"Root Cause:       {state['root_cause']}")
            print(f"Fix Rec:          {state['fix_recommendation']}")
            print(f"Confidence:       {state['confidence_score']*100}%")
            print(f"HITL Pause:       {state['was_paused']}")
            print(f"Iteration Count:  {state['iteration_count']}")
            
            # Basic assertions/validations
            assert state["failure_category"] == sc["category"]
            assert state["iteration_count"] == 1
            if sc["confidence"] < 0.80:
                assert state["was_paused"] is True
            else:
                assert state["was_paused"] is False
                
            print("Status:           SUCCESS")
            
        except Exception as e:
            print(f"Status:           FAILED due to error: {e}")
            all_passed = False
            
    print("\n" + "=" * 80)
    if all_passed:
        print("E2E Simulation Completed Successfully! All 5 scenarios behaved correctly.")
        sys.exit(0)
    else:
        print("E2E Simulation Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
