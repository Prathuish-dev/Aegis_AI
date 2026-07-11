import unittest
from unittest.mock import MagicMock
from src.healing.llm_quality_detector import LLMQualityDetector

class TestLLMQualityDetector(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.detector = LLMQualityDetector(llm=self.mock_llm)

    def test_detect_quality_empty_contexts(self):
        # Empty contexts should instantly fail faithfulness
        result = self.detector.detect_quality(
            question="What causes OOM?",
            answer="CUDA is out of memory.",
            contexts=[]
        )
        self.assertFalse(result["faithful"])
        self.assertIn("No retrieved contexts", result["reason"])

    def test_detect_quality_faithful_json(self):
        class MockResponse:
            content = '{"faithful": true, "reason": "Everything is grounded in context."}'
            
        self.mock_llm.invoke.return_value = MockResponse()
        
        result = self.detector.detect_quality(
            question="Q",
            answer="A",
            contexts=["C"]
        )
        
        self.assertTrue(result["faithful"])
        self.assertEqual(result["reason"], "Everything is grounded in context.")
        self.mock_llm.invoke.assert_called_once()

    def test_detect_quality_unfaithful_json(self):
        class MockResponse:
            content = '{"faithful": false, "reason": "The answer mentions facts not supported by context."}'
            
        self.mock_llm.invoke.return_value = MockResponse()
        
        result = self.detector.detect_quality(
            question="Q",
            answer="A",
            contexts=["C"]
        )
        
        self.assertFalse(result["faithful"])
        self.assertEqual(result["reason"], "The answer mentions facts not supported by context.")

    def test_parse_json_markdown_blocks(self):
        class MockResponse:
            # Markdown block formatting
            content = """```json
{"faithful": true, "reason": "Wrapped in markdown blocks."}
```"""
            
        self.mock_llm.invoke.return_value = MockResponse()
        
        result = self.detector.detect_quality(
            question="Q",
            answer="A",
            contexts=["C"]
        )
        
        self.assertTrue(result["faithful"])
        self.assertEqual(result["reason"], "Wrapped in markdown blocks.")

    def test_parse_json_regex_recovery(self):
        class MockResponse:
            # Malformed JSON with extra preamble text that would fail standard json.loads
            content = """Here is the evaluation result:
{"faithful": true, "reason": "Clean parse recovery."}"""
            
        self.mock_llm.invoke.return_value = MockResponse()
        
        result = self.detector.detect_quality(
            question="Q",
            answer="A",
            contexts=["C"]
        )
        
        self.assertTrue(result["faithful"])
        self.assertEqual(result["reason"], "Clean parse recovery.")

    def test_parse_json_failure_fallback(self):
        class MockResponse:
            content = "This is a plain text response with no JSON structure at all."
            
        self.mock_llm.invoke.return_value = MockResponse()
        
        result = self.detector.detect_quality(
            question="Q",
            answer="A",
            contexts=["C"]
        )
        
        self.assertFalse(result["faithful"])
        self.assertIn("Failed to parse LLM-as-a-Judge response", result["reason"])

if __name__ == "__main__":
    unittest.main()
