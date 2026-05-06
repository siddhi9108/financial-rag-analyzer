"""
retriever.py
------------
Manages the ChromaDB vector store for document storage and retrieval.
ChromaDB stores embeddings locally — no external service needed.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os

from ingestion import DocumentChunk
from embeddings import embed_texts, embed_query


CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "financial_documents"


def get_chroma_client() -> chromadb.Client:
    """
    Create or connect to a persistent ChromaDB instance.
    Data is saved to disk so you don't re-embed on every restart.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(client: chromadb.Client) -> chromadb.Collection:
    """
    Get existing collection or create a new one.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection


def add_chunks_to_store(chunks: List[DocumentChunk]) -> None:
    """
    Embed chunks and store them in ChromaDB.
    Skips chunks that are already stored (by chunk_id).
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Get existing IDs to avoid duplicates
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        print("ℹ️  All chunks already in vector store. Skipping embedding.")
        return

    print(f"🔢 Embedding {len(new_chunks)} new chunks...")

    texts = [c.text for c in new_chunks]
    embeddings = embed_texts(texts)

    # Build metadata for each chunk
    metadatas = [
        {
            "source": c.source,
            "page": c.page,
            "chunk_id": c.chunk_id
        }
        for c in new_chunks
    ]

    ids = [c.chunk_id for c in new_chunks]

    # Add to ChromaDB in batches of 500
    batch_size = 500
    for i in range(0, len(new_chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"✅ Stored {len(new_chunks)} chunks in ChromaDB")


def retrieve(
    query: str,
    top_k: int = 5,
    source_filter: str = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a query.
    
    Args:
        query: User's question
        top_k: Number of chunks to return
        source_filter: Optional — filter results by document filename
    
    Returns:
        List of dicts with 'text', 'source', 'page', 'score'
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    # Optional metadata filter
    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        # ChromaDB returns distance (lower = better), convert to similarity score
        distance = results["distances"][0][i]
        similarity = 1 - distance  # cosine: 1 = identical, 0 = unrelated

        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "score": round(similarity, 4)
        })

    # Sort by score descending
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    return retrieved


def list_documents() -> List[str]:
    """
    Return list of unique document filenames in the store.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []

    all_items = collection.get(include=["metadatas"])
    sources = list(set(m["source"] for m in all_items["metadatas"]))
    return sorted(sources)


def delete_document(filename: str) -> int:
    """
    Remove all chunks belonging to a specific document.
    Returns number of chunks deleted.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.get(
        where={"source": filename},
        include=[]
    )
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    print(f"🗑️  Deleted {len(ids_to_delete)} chunks for '{filename}'")
    return len(ids_to_delete)


def get_store_stats() -> Dict[str, Any]:
    """
    Return statistics about the vector store.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    docs = list_documents()

    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "documents": docs
    }


if __name__ == "__main__":
    stats = get_store_stats()
    print(f"Vector Store Stats: {stats}")
