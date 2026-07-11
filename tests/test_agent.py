import unittest
from typing import get_type_hints
from src.agent.state import AegisAgentState
from src.agent.prompts import DIAGNOSTIC_PROMPT, DIAGNOSTIC_SYSTEM_PROMPT, DIAGNOSTIC_USER_PROMPT

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

if __name__ == "__main__":
    unittest.main()
