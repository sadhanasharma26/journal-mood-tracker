from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model locally."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embedding(text: str) -> np.ndarray:
    """Generate a normalized 384-dim float32 embedding for the provided text."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension {vector.shape[0]}; expected {EMBEDDING_DIM}."
        )

    return vector


def serialize_embedding(vector: np.ndarray) -> bytes:
    """Serialize a float32 embedding vector to bytes for SQLite storage."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension {arr.shape[0]}; expected {EMBEDDING_DIM}."
        )
    return arr.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize bytes from SQLite back into a float32 embedding vector."""
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension {arr.shape[0]}; expected {EMBEDDING_DIM}."
        )
    return arr


def build_faiss_index(entries: list[dict]) -> tuple[faiss.IndexFlatIP, list[int]]:
    """
    Build an in-memory FAISS cosine-similarity index from journal entries.

    Each entry dict must include:
    - id: int
    - embedding: bytes (serialized float32 vector)
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    id_map: list[int] = []

    if not entries:
        return index, id_map

    vectors: list[np.ndarray] = []
    for item in entries:
        vectors.append(deserialize_embedding(item["embedding"]))
        id_map.append(int(item["id"]))

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    index.add(matrix)

    return index, id_map


def semantic_search(
    query: str,
    index: faiss.IndexFlatIP,
    id_map: list[int],
    top_k: int = 5,
) -> list[dict]:
    """Run semantic search and return matching entry IDs with similarity scores."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if index.ntotal == 0 or not id_map:
        return []

    query_vec = generate_embedding(query).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    k = min(top_k, len(id_map))
    scores, indices = index.search(query_vec, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "entry_id": id_map[idx],
                "score": float(score),
            }
        )

    return results
