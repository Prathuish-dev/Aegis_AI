import sys
import os
from typing import List, Dict, Any, Union
from loguru import logger

# --- Runtime Monkeypatch to Bypass RAGAS VertexAI Import Crash ---
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.embeddings.vertexai'] = MagicMock()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.documents import Document

QUALITY_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70
}

class RAGEvaluator:
    def __init__(self, llm: Any = None, embeddings: Any = None):
        """
        Initializes the RAGEvaluator with custom or default LLM and embeddings.
        
        Args:
            llm: LangChain ChatModel instance.
            embeddings: LangChain Embeddings instance.
        """
        # Determine LLM
        if llm is None:
            llm = self._get_default_llm()
        self.llm = llm
        
        # Determine Embeddings
        if embeddings is None:
            embeddings = self._get_default_embeddings()
        self.embeddings = embeddings
        
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        logger.info("RAGEvaluator initialized successfully.")

    def _get_default_llm(self) -> Any:
        """Load default chat LLM based on environment variables."""
        if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here":
            try:
                from langchain_openai import ChatOpenAI
                logger.info("RAGEvaluator using OpenAI ChatOpenAI (gpt-4o-mini).")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            except ImportError:
                pass
                
        if os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                logger.info("RAGEvaluator using ChatGroq.")
                return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
            except ImportError:
                pass
                
        logger.warning("No API key configured for OpenAI/Groq. Falling back to a dummy ChatModel for testing.")
        # Create a mock chat model for evaluation testing
        class DummyChatModel:
            def invoke(self, messages, *args, **kwargs):
                class DummyMessage:
                    content = "Mock response grounding detail: This is a faithful and relevant answer."
                return DummyMessage()
        return DummyChatModel()

    def _get_default_embeddings(self) -> Any:
        """Load default HuggingFaceEmbeddings BAAI/bge-small-en-v1.5."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("RAGEvaluator loading BAAI/bge-small-en-v1.5 embeddings.")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Failed to load default embeddings: {e}")
            raise

    def evaluate_quality(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate RAG pipeline quality metrics.
        
        Args:
            test_cases: List of dictionaries of the form:
                {
                    "question": str,
                    "answer": str,
                    "contexts": List[Union[str, Document]],
                    "ground_truth": str
                }
                
        Returns:
            Dict containing average metric scores for:
            - faithfulness
            - answer_relevancy
            - context_precision
            - context_recall
        """
        if not test_cases:
            logger.warning("No test cases provided for evaluation.")
            return {}

        logger.info(f"Evaluating {len(test_cases)} RAG test cases...")
        
        # Normalize contexts (extract page_content from LangChain Document objects if present)
        normalized_cases = []
        for i, tc in enumerate(test_cases):
            raw_contexts = tc.get("contexts", [])
            contexts = [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in raw_contexts
            ]
            normalized_cases.append({
                "question": tc["question"],
                "answer": tc["answer"],
                "contexts": contexts,
                "ground_truth": tc["ground_truth"]
            })
            
        dataset = Dataset.from_list(normalized_cases)
        
        try:
            # Perform RAGAS evaluation
            results = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            evaluation_scores = {
                "faithfulness": float(results.get("faithfulness", 0.0)),
                "answer_relevancy": float(results.get("answer_relevancy", 0.0)),
                "context_precision": float(results.get("context_precision", 0.0)),
                "context_recall": float(results.get("context_recall", 0.0))
            }
            logger.info(f"RAGAS evaluation scores: {evaluation_scores}")
            return evaluation_scores
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation execution: {e}")
            raise

    def verify_targets(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Compares evaluation scores against the project quality targets.
        
        Args:
            scores: Dict containing metric scores.
            
        Returns:
            Dict containing:
            - passed: bool (True if all metrics meet/exceed target scores)
            - details: Dict mapping metric -> {"score": float, "target": float, "passed": bool}
        """
        details = {}
        all_passed = True
        
        for metric, target in QUALITY_TARGETS.items():
            score = scores.get(metric, 0.0)
            passed = score >= target
            if not passed:
                all_passed = False
            details[metric] = {
                "score": score,
                "target": target,
                "passed": passed
            }
            
        return {
            "passed": all_passed,
            "details": details
        }
