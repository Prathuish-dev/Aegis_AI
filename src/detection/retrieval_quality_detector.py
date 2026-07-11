import numpy as np
from typing import Any, List, Dict, Tuple
from loguru import logger
from langchain_core.documents import Document

class RetrievalQualityDetector:
    """
    RetrievalQualityDetector performs quality checks on retrieved documents,
    specifically detecting near-duplicates using pairwise embedding comparisons.
    """
    def __init__(self, embedding_function: Any = None):
        """
        Initialize the RetrievalQualityDetector.
        
        Args:
            embedding_function: LangChain-compatible embedding model. If None, loads default.
        """
        if embedding_function is None:
            embedding_function = self._get_default_embeddings()
        self.embedding_function = embedding_function
        logger.info("RetrievalQualityDetector initialized.")

    def _get_default_embeddings(self) -> Any:
        """Loads the default HuggingFace BAAI/bge-small-en-v1.5 embeddings model."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Failed to load default embeddings: {e}")
            raise

    def detect_duplicates(self, documents: List[Document], threshold: float = 0.85) -> Dict[str, Any]:
        """
        Detects near-duplicate documents within the retrieved list using pairwise cosine similarity.
        
        Args:
            documents: List of LangChain Document objects.
            threshold: Cosine similarity threshold above which docs are flagged as duplicates.
            
        Returns:
            Dict containing:
            - has_duplicates: bool
            - duplicate_pairs: List of Dict detailing matching documents and their similarity scores.
        """
        if len(documents) < 2:
            return {
                "has_duplicates": False,
                "duplicate_pairs": []
            }
            
        logger.info(f"Checking for duplicates among {len(documents)} retrieved documents (threshold: {threshold}).")
        
        # Extract page contents
        texts = [doc.page_content for doc in documents]
        
        # Embed all texts
        try:
            embeddings = self.embedding_function.embed_documents(texts)
            embeddings_array = np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating document embeddings for duplicate checks: {e}")
            # Fallback simple string matching if embeddings fail
            return self._detect_duplicates_text_fallback(documents, threshold)
            
        duplicate_pairs = []
        n = len(documents)
        
        for i in range(n):
            for j in range(i + 1, n):
                vec1 = embeddings_array[i]
                vec2 = embeddings_array[j]
                
                # Compute cosine similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
                    
                if similarity >= threshold:
                    logger.warning(f"Detected duplicate docs at index {i} and {j} (similarity: {similarity:.4f}).")
                    duplicate_pairs.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "similarity": similarity,
                        "doc1_source": documents[i].metadata.get("source", "unknown"),
                        "doc2_source": documents[j].metadata.get("source", "unknown"),
                        "doc1_preview": texts[i][:100] + "...",
                        "doc2_preview": texts[j][:100] + "..."
                    })
                    
        return {
            "has_duplicates": len(duplicate_pairs) > 0,
            "duplicate_pairs": duplicate_pairs
        }

    def _detect_duplicates_text_fallback(self, documents: List[Document], threshold: float) -> Dict[str, Any]:
        """Fallback character-level overlap duplicate checker in case embedding fails."""
        duplicate_pairs = []
        n = len(documents)
        
        for i in range(n):
            for j in range(i + 1, n):
                text1 = documents[i].page_content.strip().lower()
                text2 = documents[j].page_content.strip().lower()
                
                # Jaccard similarity of words
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                if not words1 or not words2:
                    similarity = 0.0
                else:
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    similarity = len(intersection) / len(union)
                    
                if similarity >= threshold:
                    duplicate_pairs.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "similarity": similarity,
                        "doc1_source": documents[i].metadata.get("source", "unknown"),
                        "doc2_source": documents[j].metadata.get("source", "unknown"),
                        "doc1_preview": documents[i].page_content[:100] + "...",
                        "doc2_preview": documents[j].page_content[:100] + "..."
                    })
                    
        return {
            "has_duplicates": len(duplicate_pairs) > 0,
            "duplicate_pairs": duplicate_pairs
        }
