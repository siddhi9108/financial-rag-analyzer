"""
embeddings.py
-------------
Generates vector embeddings using sentence-transformers (free, local).
Model: all-MiniLM-L6-v2 — fast, lightweight, great for semantic search.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


# Singleton pattern — load model once, reuse it
_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the embedding model (cached after first call).
    Downloads automatically on first use (~90MB).
    """
    global _model
    if _model is None:
        print(f"📥 Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        print("✅ Embedding model ready")
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Convert a list of strings into embedding vectors.
    
    Args:
        texts: List of text strings to embed
        batch_size: Process in batches to avoid memory issues
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    model = get_embedding_model()
    
    # Encode in batches for memory efficiency
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine similarity works better normalized
    )
    
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single search query.
    Normalized for cosine similarity.
    """
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding[0].tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns value between -1 and 1 (higher = more similar).
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Test embeddings
    texts = [
        "Total revenue increased by 15% year over year.",
        "Operating expenses rose due to higher R&D investment.",
        "The company declared a quarterly dividend of $0.50 per share."
    ]
    query = "What happened to revenue?"
    
    print("Embedding test texts...")
    embeddings = embed_texts(texts)
    query_emb = embed_query(query)
    
    print(f"\nQuery: '{query}'")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        sim = cosine_similarity(query_emb, emb)
        print(f"  [{sim:.3f}] {text}")
