from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model once at the module level (singleton pattern)
_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

class RelevanceScorer:
    """Scorer to determine the semantic relevance of LLM responses to a query using SentenceTransformers."""

    def __init__(self):
        self.model = _MODEL

    def score(self, query: str, response: str) -> float:
        """Return the semantic cosine similarity between the query and the response."""
        if not query or not response:
            return 0.0
        try:
            embeddings = self.model.encode([query, response], convert_to_numpy=True)
            return float(util.cos_sim(embeddings[0], embeddings[1]))
        except Exception:
            return 0.0

    def is_hallucinating(self, query: str, response: str, threshold: float = 0.5) -> bool:
        """Determine if the response is likely irrelevant or hallucinated based on similarity threshold."""
        return self.score(query, response) < threshold
