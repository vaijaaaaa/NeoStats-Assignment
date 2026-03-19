"""
Embeddings Module: Converts text into vectors for semantic search.
"""

from sentence_transformers import SentenceTransformer


_model = None


def _get_model() -> SentenceTransformer:
    """Load embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of texts into embedding vectors.

    Args:
        texts (list[str]): Input text list.

    Returns:
        list[list[float]]: Embedding vectors.
    """
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()
