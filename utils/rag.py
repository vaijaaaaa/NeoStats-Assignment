"""
RAG Utilities: Load docs, chunk text, build FAISS index, retrieve context.
"""

import os
import faiss
import numpy as np

from config.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from models.embeddings import get_embeddings


def load_documents(data_dir: str = "data") -> list[dict]:
    """
    Load all .txt files from data directory.

    Returns:
        list[dict]: Each item has {"source": filename, "text": content}
    """
    documents = []

    if not os.path.exists(data_dir):
        return documents

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                if text:
                    documents.append({"source": file_name, "text": text})
        except Exception:
            continue

    return documents


def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    """
    Split long text into overlapping chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += max(chunk_size - overlap, 1)

    return chunks


def build_faiss_index(chunks: list[str]):
    """
    Create FAISS index from text chunks.

    Returns:
        tuple: (index, chunks) where index can be None if no data.
    """
    if not chunks:
        return None, []

    vectors = get_embeddings(chunks)
    if not vectors:
        return None, []

    matrix = np.array(vectors, dtype=np.float32)
    dimension = matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(matrix)

    return index, chunks


def setup_rag(data_dir: str = "data"):
    """
    End-to-end setup: load docs -> chunk -> build index.

    Returns:
        tuple: (index, chunk_records)
        chunk_records format: [{"source": ..., "chunk": ...}, ...]
    """
    documents = load_documents(data_dir)
    if not documents:
        return None, []

    chunk_records = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"])
        for chunk in doc_chunks:
            chunk_records.append({"source": doc["source"], "chunk": chunk})

    chunks_only = [item["chunk"] for item in chunk_records]
    index, chunks = build_faiss_index(chunks_only)

    if index is None:
        return None, []

    final_records = []
    for i, chunk in enumerate(chunks):
        source = chunk_records[i]["source"] if i < len(chunk_records) else "unknown"
        final_records.append({"source": source, "chunk": chunk})

    return index, final_records


def retrieve_relevant_chunks(query: str, index, chunk_records: list[dict], top_k: int = 3) -> list[dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    """
    if not query or index is None or not chunk_records:
        return []

    query_vector = get_embeddings([query])
    if not query_vector:
        return []

    query_matrix = np.array(query_vector, dtype=np.float32)
    k = min(top_k, len(chunk_records))

    distances, indices = index.search(query_matrix, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunk_records):
            continue
        results.append(
            {
                "source": chunk_records[idx]["source"],
                "chunk": chunk_records[idx]["chunk"],
                "distance": float(distances[0][rank]),
            }
        )

    return results


def build_context_from_chunks(chunks: list[dict]) -> str:
    """
    Build a single context string from retrieved chunks.
    """
    if not chunks:
        return ""

    context_parts = []
    for item in chunks:
        source = item.get("source", "unknown")
        text = item.get("chunk", "")
        context_parts.append(f"[Source: {source}]\n{text}")

    return "\n\n".join(context_parts)
