import unittest
from unittest.mock import patch, MagicMock
from src.rag.evaluator import RAGEvaluator, QUALITY_TARGETS

class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        # Create mock LLM and embeddings to avoid loading heavy models or hitting APIs
        self.mock_llm = MagicMock()
        self.mock_embeddings = MagicMock()
        
        # Initialize evaluator with mocks
        self.evaluator = RAGEvaluator(llm=self.mock_llm, embeddings=self.mock_embeddings)

    @patch("src.rag.evaluator.evaluate")
    def test_evaluate_quality_success(self, mock_ragas_evaluate):
        # Configure mock return value for RAGAS evaluate
        mock_results = {
            "faithfulness": 0.88,
            "answer_relevancy": 0.82,
            "context_precision": 0.78,
            "context_recall": 0.72
        }
        mock_ragas_evaluate.return_value = mock_results
        
        test_cases = [
            {
                "question": "What causes data drift?",
                "answer": "Changes in feature distributions over time.",
                "contexts": ["Data drift occurs when feature distributions shift."],
                "ground_truth": "Feature distribution shifts over time."
            }
        ]
        
        scores = self.evaluator.evaluate_quality(test_cases)
        
        # Verify result format
        self.assertEqual(scores["faithfulness"], 0.88)
        self.assertEqual(scores["answer_relevancy"], 0.82)
        self.assertEqual(scores["context_precision"], 0.78)
        self.assertEqual(scores["context_recall"], 0.72)
        
        # Verify that mock_ragas_evaluate was called
        mock_ragas_evaluate.assert_called_once()

    def test_verify_targets_all_passing(self):
        scores = {
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.75
        }
        
        verification = self.evaluator.verify_targets(scores)
        
        self.assertTrue(verification["passed"])
        for metric in QUALITY_TARGETS:
            self.assertTrue(verification["details"][metric]["passed"])
            self.assertEqual(verification["details"][metric]["score"], scores[metric])

    def test_verify_targets_failing(self):
        # Faithfulness (0.70) is below target (0.85)
        scores = {
            "faithfulness": 0.70,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.75
        }
        
        verification = self.evaluator.verify_targets(scores)
        
        self.assertFalse(verification["passed"])
        self.assertFalse(verification["details"]["faithfulness"]["passed"])
        self.assertTrue(verification["details"]["answer_relevancy"]["passed"])

    def test_context_normalization(self):
        # Test case with LangChain Document objects in contexts
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
                
        test_cases = [
            {
                "question": "Q",
                "answer": "A",
                "contexts": [MockDoc("Document text content"), "Plain text content"],
                "ground_truth": "GT"
            }
        ]
        
        # We patch evaluate to inspect how dataset was built
        with patch("src.rag.evaluator.evaluate") as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {}
            self.evaluator.evaluate_quality(test_cases)
            
            # Extract dataset passed to evaluate
            called_dataset = mock_ragas_evaluate.call_args[1]["dataset"]
            
            # Verify contexts were normalized to plain text list
            self.assertEqual(called_dataset[0]["contexts"], ["Document text content", "Plain text content"])

if __name__ == "__main__":
    unittest.main()
