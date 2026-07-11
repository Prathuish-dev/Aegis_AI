import unittest
from typing import Any
from src.agent.state import AegisAgentState
from src.agent.nodes import (
    retrieve_context_node,
    analyze_anomaly_node,
    classify_failure_node,
    generate_fix_node,
    human_review_node,
    apply_fix_node,
    MockLLM,
    parse_json_response
)
from src.healing.prompt_optimizer import PromptOptimizer
from src.monitoring.models import FailureCategory

class MockRAG:
    """Mock RAG class for testing retrieve_context_node RAG path."""
    def __init__(self, expected_docs):
        self.expected_docs = expected_docs
        
    def retrieve(self, query: str, category: str = None) -> list:
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
        return [MockDoc(doc) for doc in self.expected_docs]

class TestAgentNodesAndOptimizer(unittest.TestCase):
    def test_parse_json_response(self):
        # Raw json
        raw = '{"key": "val"}'
        self.assertEqual(parse_json_response(raw), {"key": "val"})
        
        # Wrapped in markdown
        wrapped = '```json\n{"key": "val"}\n```'
        self.assertEqual(parse_json_response(wrapped), {"key": "val"})
        
        # Wrapped in normal block
        wrapped_normal = '```\n{"key": "val"}\n```'
        self.assertEqual(parse_json_response(wrapped_normal), {"key": "val"})
        
        # Regex fallback
        fallback = 'Some text before {"key": "val"} some text after'
        self.assertEqual(parse_json_response(fallback), {"key": "val"})

    def test_prompt_optimizer(self):
        mock_llm = MockLLM()
        optimizer = PromptOptimizer(llm=mock_llm)
        original_prompt = "What is 2+2?"
        failure_reason = "Generated 5 instead of 4."
        
        optimized = optimizer.optimize(original_prompt, failure_reason)
        self.assertIn("What is 2+2?", optimized)
        self.assertIn("[optimized]", optimized)

    def test_retrieve_context_node_fallback(self):
        state = AegisAgentState(
            anomaly_description="accuracy drop",
            system_id="sys_1",
            failure_category="model_issue",
            raw_metrics={},
            retrieved_context=[],
            analysis_steps=[],
            root_cause="",
            fix_recommendation="",
            confidence_score=0.0,
            requires_human_review=False,
            iteration_count=0,
            messages=[]
        )
        
        # Running retrieve_context_node with no RAG provided in config
        result = retrieve_context_node(state, config={})
        self.assertTrue(len(result["retrieved_context"]) > 0)
        self.assertIn("Model Accuracy Retraining Playbook", result["retrieved_context"][0])
        self.assertEqual(len(result["analysis_steps"]), 1)

    def test_retrieve_context_node_rag(self):
        state = AegisAgentState(
            anomaly_description="accuracy drop",
            system_id="sys_1",
            failure_category="model_issue",
            raw_metrics={},
            retrieved_context=[],
            analysis_steps=[],
            root_cause="",
            fix_recommendation="",
            confidence_score=0.0,
            requires_human_review=False,
            iteration_count=0,
            messages=[]
        )
        
        # Providing MockRAG in config
        mock_rag = MockRAG(["Retrieved Guide doc 1", "Retrieved Guide doc 2"])
        config = {
            "configurable": {
                "rag": mock_rag
            }
        }
        
        result = retrieve_context_node(state, config=config)
        self.assertEqual(len(result["retrieved_context"]), 2)
        self.assertEqual(result["retrieved_context"][0], "Retrieved Guide doc 1")

    def test_analyze_anomaly_node(self):
        state = AegisAgentState(
            anomaly_description="Model accuracy drops",
            system_id="sys_1",
            failure_category="",
            raw_metrics={},
            retrieved_context=["Some guide on model training"],
            analysis_steps=[],
            root_cause="",
            fix_recommendation="",
            confidence_score=0.0,
            requires_human_review=False,
            iteration_count=0,
            messages=[]
        )
        
        # Running with MockLLM
        config = {
            "configurable": {
                "llm": MockLLM()
            }
        }
        
        result = analyze_anomaly_node(state, config=config)
        self.assertEqual(result["failure_category"], "model_issue")
        self.assertAlmostEqual(result["confidence_score"], 0.85)
        self.assertTrue(len(result["root_cause"]) > 0)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(len(result["analysis_steps"]), 1)

    def test_classify_failure_node(self):
        # Valid category
        state = AegisAgentState(
            failure_category="data_issue",
            analysis_steps=[]
        )
        result = classify_failure_node(state)
        self.assertEqual(result["failure_category"], "data_issue")
        
        # Invalid category (should fallback to system_issue)
        state_invalid = AegisAgentState(
            failure_category="some_random_invalid_category",
            analysis_steps=[]
        )
        result_invalid = classify_failure_node(state_invalid)
        self.assertEqual(result_invalid["failure_category"], "system_issue")

    def test_generate_fix_node(self):
        # High confidence (no human review)
        state_high_conf = AegisAgentState(
            confidence_score=0.9,
            fix_recommendation="retrain",
            analysis_steps=[]
        )
        result_high = generate_fix_node(state_high_conf)
        self.assertFalse(result_high["requires_human_review"])
        
        # High confidence scaled (no human review)
        state_high_conf_scaled = AegisAgentState(
            confidence_score=90.0,
            fix_recommendation="retrain",
            analysis_steps=[]
        )
        result_high_scaled = generate_fix_node(state_high_conf_scaled)
        self.assertFalse(result_high_scaled["requires_human_review"])
        
        # Low confidence (requires human review)
        state_low_conf = AegisAgentState(
            confidence_score=0.7,
            fix_recommendation="retrain",
            analysis_steps=[]
        )
        result_low = generate_fix_node(state_low_conf)
        self.assertTrue(result_low["requires_human_review"])

    def test_human_review_node(self):
        state = AegisAgentState(
            requires_human_review=True,
            analysis_steps=[]
        )
        result = human_review_node(state)
        self.assertFalse(result["requires_human_review"])
        self.assertEqual(len(result["analysis_steps"]), 1)

    def test_apply_fix_node(self):
        state = AegisAgentState(
            iteration_count=2,
            analysis_steps=[]
        )
        result = apply_fix_node(state)
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(len(result["analysis_steps"]), 1)

if __name__ == "__main__":
    unittest.main()
