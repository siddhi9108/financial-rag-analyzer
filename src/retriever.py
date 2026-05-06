import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import DocumentChunk
from embeddings import embed_texts, embed_query

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "financial_documents"

def get_chroma_client():
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

def get_or_create_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks_to_store(chunks: List[DocumentChunk]) -> None:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    if not new_chunks:
        print("All chunks already in vector store.")
        return
    texts = [c.text for c in new_chunks]
    embeddings = embed_texts(texts)
    metadatas = [{"source": c.source, "page": c.page, "chunk_id": c.chunk_id} for c in new_chunks]
    ids = [c.chunk_id for c in new_chunks]
    batch_size = 500
    for i in range(0, len(new_chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    print(f"Stored {len(new_chunks)} chunks in ChromaDB")

def retrieve(query: str, top_k: int = 5, source_filter: str = None) -> List[Dict[str, Any]]:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    if collection.count() == 0:
        return []
    query_embedding = embed_query(query)
    where = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "score": round(similarity, 4)
        })
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    return retrieved

def list_documents() -> List[str]:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    if collection.count() == 0:
        return []
    all_items = collection.get(include=["metadatas"])
    sources = list(set(m["source"] for m in all_items["metadatas"]))
    return sorted(sources)

def delete_document(filename: str) -> int:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    results = collection.get(where={"source": filename}, include=[])
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)

def get_store_stats() -> Dict[str, Any]:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    docs = list_documents()
    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "documents": docs
    }