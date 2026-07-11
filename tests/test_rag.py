import unittest
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.rag.retrieval import AegisRAG

class TestRAGRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load local embedding model
        cls.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        cls.persist_dir = "data/chroma_db"
        cls.rag = AegisRAG(
            embedding_function=cls.embeddings,
            persist_directory=cls.persist_dir
        )

    def test_basic_retrieval(self):
        # Query about drift detection
        query = "How do we detect data drift in production?"
        results = self.rag.retrieve(query, k=3)
        self.assertGreater(len(results), 0, "No results retrieved")
        
        # Verify that the retrieved documents contain drift-related content
        content_found = False
        for doc in results:
            if "drift" in doc.page_content.lower() or "covariate" in doc.page_content.lower():
                content_found = True
                break
        self.assertTrue(content_found, "Should retrieve drift-related context")

    def test_category_filtering(self):
        # Query about system issues
        query = "Out of memory error rate limit exceptions"
        
        # Retrieval with system_issue filter
        system_results = self.rag.retrieve(query, category="system_issue", k=3)
        self.assertGreater(len(system_results), 0, "No results retrieved for system_issue")
        for doc in system_results:
            self.assertEqual(doc.metadata.get("category"), "system_issue", "Metadata category mismatch")
            
        # Retrieval with data_issue filter
        data_results = self.rag.retrieve(query, category="data_issue", k=3)
        self.assertGreater(len(data_results), 0, "No results retrieved for data_issue")
        for doc in data_results:
            self.assertEqual(doc.metadata.get("category"), "data_issue", "Metadata category mismatch")

    def test_incident_retrieval(self):
        # Query that should match a past incident
        query = "INC-2026-004 memory leak in embedding service"
        results = self.rag.retrieve(query, k=2)
        self.assertGreater(len(results), 0, "No results retrieved")
        
        # Verify if the incident id metadata is present in one of the documents
        incident_found = False
        for doc in results:
            if doc.metadata.get("type") == "incident" and doc.metadata.get("incident_id") == "INC-2026-004":
                incident_found = True
                break
        self.assertTrue(incident_found, "Should retrieve the specific incident record INC-2026-004")

if __name__ == "__main__":
    unittest.main()
