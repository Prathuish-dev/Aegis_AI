import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.detection.retrieval_quality_detector import RetrievalQualityDetector

class TestRetrievalQualityDetector(unittest.TestCase):
    def setUp(self):
        self.mock_embeddings = MagicMock()
        self.detector = RetrievalQualityDetector(embedding_function=self.mock_embeddings)

    def test_detect_duplicates_less_than_two_docs(self):
        # 0 docs
        res_0 = self.detector.detect_duplicates([])
        self.assertFalse(res_0["has_duplicates"])
        
        # 1 doc
        res_1 = self.detector.detect_duplicates([Document(page_content="Only one doc")])
        self.assertFalse(res_1["has_duplicates"])

    def test_detect_duplicates_success(self):
        # We mock embed_documents to return unit vectors
        # Doc 0 and Doc 1 are identical (similarity 1.0)
        # Doc 2 is completely different (orthogonal vector)
        self.mock_embeddings.embed_documents.return_value = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        
        docs = [
            Document(page_content="First duplicated text.", metadata={"source": "doc1.md"}),
            Document(page_content="First duplicated text.", metadata={"source": "doc2.md"}),
            Document(page_content="Completely different orthogonal statement.", metadata={"source": "doc3.md"})
        ]
        
        res = self.detector.detect_duplicates(docs, threshold=0.85)
        
        self.assertTrue(res["has_duplicates"])
        self.assertEqual(len(res["duplicate_pairs"]), 1)
        
        pair = res["duplicate_pairs"][0]
        self.assertEqual(pair["doc1_index"], 0)
        self.assertEqual(pair["doc2_index"], 1)
        self.assertAlmostEqual(pair["similarity"], 1.0)
        self.assertEqual(pair["doc1_source"], "doc1.md")
        self.assertEqual(pair["doc2_source"], "doc2.md")

    def test_detect_duplicates_below_threshold(self):
        # Similarity is 0.707 (45 degrees), threshold is 0.80
        self.mock_embeddings.embed_documents.return_value = [
            [1.0, 0.0],
            [0.707, 0.707]
        ]
        
        docs = [
            Document(page_content="Doc 1", metadata={"source": "1.md"}),
            Document(page_content="Doc 2", metadata={"source": "2.md"})
        ]
        
        res = self.detector.detect_duplicates(docs, threshold=0.80)
        self.assertFalse(res["has_duplicates"])
        self.assertEqual(len(res["duplicate_pairs"]), 0)

    def test_text_fallback_on_embedding_exception(self):
        # embed_documents raises an exception
        self.mock_embeddings.embed_documents.side_effect = Exception("Model service offline")
        
        # Test fallback jaccard word overlap
        # Identical content
        docs = [
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"source": "fox.md"}),
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"source": "dog.md"})
        ]
        
        res = self.detector.detect_duplicates(docs, threshold=0.85)
        self.assertTrue(res["has_duplicates"])
        self.assertEqual(len(res["duplicate_pairs"]), 1)
        self.assertAlmostEqual(res["duplicate_pairs"][0]["similarity"], 1.0)

if __name__ == "__main__":
    unittest.main()
