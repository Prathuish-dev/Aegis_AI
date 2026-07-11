import unittest
from typing import get_type_hints
from src.agent.state import AegisAgentState
from src.agent.prompts import DIAGNOSTIC_PROMPT
from src.agent.graph import build_debugging_agent, route_after_fix
from src.agent.nodes import MockLLM
from langchain_core.messages import AIMessage, HumanMessage

class TestAgentComponents(unittest.TestCase):
    def test_agent_state_keys(self):
        # Verify that all required keys are defined in AegisAgentState
        expected_keys = {
            "anomaly_description",
            "system_id",
            "failure_category",
            "raw_metrics",
            "retrieved_context",
            "analysis_steps",
            "root_cause",
            "fix_recommendation",
            "confidence_score",
            "requires_human_review",
            "iteration_count",
            "messages"
        }
        
        # In Python, TypedDict keys can be retrieved via __annotations__
        annotations = AegisAgentState.__annotations__
        for key in expected_keys:
            self.assertIn(key, annotations, f"Key '{key}' is missing from AegisAgentState")
            
    def test_diagnostic_prompt_variables(self):
        # Verify that the diagnostic prompt templates contain the expected placeholders
        input_variables = DIAGNOSTIC_PROMPT.input_variables
        expected_variables = {"anomaly_description", "retrieved_context", "system_id", "raw_metrics"}
        for var in expected_variables:
            self.assertIn(var, input_variables, f"Input variable '{var}' is missing from DIAGNOSTIC_PROMPT")

    def test_prompt_formatting(self):
        # Verify formatting the prompt with dummy values succeeds
        formatted = DIAGNOSTIC_PROMPT.format_prompt(
            anomaly_description="Test anomaly",
            retrieved_context="Test context",
            system_id="test_system",
            raw_metrics="{}"
        )
        messages = formatted.to_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].type, "human")
        self.assertIn("Test anomaly", messages[1].content)
        self.assertIn("Test context", messages[1].content)
        self.assertIn("test_system", messages[1].content)

    def test_route_after_fix(self):
        # If requires_human_review is True, route to human_review
        state_low = AegisAgentState(requires_human_review=True)
        self.assertEqual(route_after_fix(state_low), "human_review")
        
        # If requires_human_review is False, route to apply_fix
        state_high = AegisAgentState(requires_human_review=False)
        self.assertEqual(route_after_fix(state_high), "apply_fix")
        
        # Default fallback should be human_review
        state_missing = AegisAgentState()
        self.assertEqual(route_after_fix(state_missing), "human_review")

    def test_build_debugging_agent(self):
        app = build_debugging_agent()
        self.assertIsNotNone(app)
        
    def test_graph_execution_high_confidence(self):
        # Setup high confidence LLM mock
        class HighConfidenceLLM:
            def invoke(self, *args, **kwargs):
                class Content:
                    content = '{"root_cause": "test", "failure_category": "system_issue", "fix_recommendation": "restart", "confidence_score": 0.95}'
                return Content()
                
        app = build_debugging_agent()
        config = {
            "configurable": {
                "thread_id": "test_high_thread",
                "llm": HighConfidenceLLM()
            }
        }
        
        initial_state = AegisAgentState(
            anomaly_description="high latency in database",
            system_id="database_01",
            failure_category="",
            raw_metrics={"latency": 2500},
            retrieved_context=[],
            analysis_steps=[],
            root_cause="",
            fix_recommendation="",
            confidence_score=0.0,
            requires_human_review=False,
            iteration_count=0,
            messages=[]
        )
        
        # Run graph to the end
        final_state = app.invoke(initial_state, config=config)
        
        # Verify it went through all nodes and reached apply_fix
        self.assertEqual(final_state["failure_category"], "system_issue")
        self.assertFalse(final_state["requires_human_review"])
        self.assertEqual(final_state["iteration_count"], 1)
        self.assertIn("Successfully applied fix at iteration 1.", final_state["analysis_steps"][-1])

    def test_graph_execution_low_confidence_interrupt(self):
        # Setup low confidence LLM mock (confidence score 0.40 < 0.80)
        class LowConfidenceLLM:
            def invoke(self, *args, **kwargs):
                class Content:
                    content = '{"root_cause": "unknown", "failure_category": "system_issue", "fix_recommendation": "investigate", "confidence_score": 0.40}'
                return Content()
                
        app = build_debugging_agent()
        config = {
            "configurable": {
                "thread_id": "test_low_thread",
                "llm": LowConfidenceLLM()
            }
        }
        
        initial_state = AegisAgentState(
            anomaly_description="intermittent failures",
            system_id="web_server_01",
            failure_category="",
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
        
        # Run the graph; it should halt/pause at "human_review" due to low confidence
        paused_state = app.invoke(initial_state, config=config)
        
        # Get current state from graph execution
        next_step = app.get_state(config).next
        
        # Assert that the graph is indeed paused before "human_review" node
        self.assertEqual(next_step, ("human_review",))
        self.assertTrue(paused_state["requires_human_review"])
        self.assertEqual(paused_state["iteration_count"], 0) # apply_fix has not run
        
        # Resume the graph by passing None as input (human approves the recommendation)
        final_state = app.invoke(None, config=config)
        
        # Verify it resumed, ran human_review, apply_fix, and reached END
        self.assertEqual(final_state["iteration_count"], 1)
        self.assertFalse(final_state["requires_human_review"]) # human_review clears flag
        self.assertIn("Human-in-the-loop review performed.", final_state["analysis_steps"][-2])
        self.assertIn("Successfully applied fix at iteration 1.", final_state["analysis_steps"][-1])

if __name__ == "__main__":
    unittest.main()
