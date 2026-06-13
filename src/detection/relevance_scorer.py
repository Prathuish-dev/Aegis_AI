from sentence_transformers import SentenceTransformer, util
from typing import Union

# Initialize the model once at the module level (singleton pattern)
# Model is loaded once when the module is imported.
_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

class RelevanceScorer:
    """Semantic relevance scorer utilizing SentenceTransformer embeddings to evaluate text similarity."""

    def __init__(self):
        """Initializes the RelevanceScorer using the pre-loaded singleton model."""
        self.model = _MODEL

    def score(self, query: str, response: str) -> float:
        """Computes the cosine similarity between the query and response embeddings.
        
        Args:
            query: The input query string.
            response: The generated response string.
            
        Returns:
            Cosine similarity score as a float (typically between 0.0 and 1.0).
        """
        if not query or not response:
            return 0.0
            
        embeddings = self.model.encode([query, response], convert_to_numpy=True)
        # util.cos_sim returns a similarity matrix; extract the float value
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity)

    def is_hallucinating(self, query: str, response: str, threshold: float = 0.5) -> bool:
        """Determines if the response similarity is below the acceptable threshold.
        
        Args:
            query: The input query string.
            response: The generated response string.
            threshold: Similarity threshold below which a response is flagged (default: 0.5).
            
        Returns:
            True if the score is below the threshold, otherwise False.
        """
        return self.score(query, response) < threshold
