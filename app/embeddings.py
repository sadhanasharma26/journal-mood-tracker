from functools import lru_cache
import hashlib
import os
import platform
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _is_light_mode_enabled() -> bool:
    value = os.getenv("JMT_LIGHT_MODE", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    """Load and cache the sentence-transformer model locally."""
    # Import lazily so API startup does not depend on ML runtime initialization.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embedding(text: str) -> np.ndarray:
    """Generate a normalized 384-dim float32 embedding for the provided text."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    if _is_light_mode_enabled():
        digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

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


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row and keep float32 for cosine similarity math."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def build_faiss_index(entries: list[dict]) -> tuple[Any, list[int]]:
    """
    Build an in-memory FAISS cosine-similarity index from journal entries.

    Each entry dict must include:
    - id: int
    - embedding: bytes (serialized float32 vector)
    """
    id_map: list[int] = []
    vectors: list[np.ndarray] = []

    for item in entries:
        vectors.append(deserialize_embedding(item["embedding"]))
        id_map.append(int(item["id"]))

    matrix = (
        _normalize_rows(np.vstack(vectors).astype(np.float32))
        if vectors
        else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    )

    # Import FAISS lazily to avoid loading native OpenMP libs during startup.
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if matrix.shape[0] > 0:
            index.add(matrix)
        return index, id_map
    except Exception:
        # Portable fallback for environments where FAISS/native runtime is unstable.
        return {"backend": "numpy", "matrix": matrix}, id_map


def semantic_search(
    query: str,
    index: Any,
    id_map: list[int],
    top_k: int = 5,
) -> list[dict]:
    """Run semantic search and return matching entry IDs with similarity scores."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if not id_map:
        return []

    query_vec = generate_embedding(query).astype(np.float32)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm > 0:
        query_vec = query_vec / query_norm

    k = min(top_k, len(id_map))

    if isinstance(index, dict) and index.get("backend") == "numpy":
        matrix = index.get("matrix")
        if matrix is None or matrix.shape[0] == 0:
            return []

        similarity = matrix @ query_vec
        top_indices = np.argsort(-similarity)[:k]
        scores_arr = similarity[top_indices]
        pairs = list(zip(scores_arr.tolist(), top_indices.tolist()))
    else:
        if getattr(index, "ntotal", 0) == 0:
            return []

        scores, indices = index.search(query_vec.reshape(1, -1), k)
        pairs = list(zip(scores[0].tolist(), indices[0].tolist()))

    results: list[dict] = []
    for score, idx in pairs:
        if idx == -1:
            continue
        results.append(
            {
                "entry_id": id_map[idx],
                "score": float(score),
            }
        )

    return results
