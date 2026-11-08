import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore:
    """
    An in-memory, numpy-based vector store.
    Stores text passages, their embedding vectors, and metadata.
    Performs nearest-neighbor semantic search using cosine similarity,
    with support for configurable similarity thresholds and optional
    lightweight hybrid (semantic + keyword) reranking.
    No external vector database libraries (like FAISS or Chroma) are used.
    """

    def __init__(self):
        """Initialize an empty in-memory vector store."""
        self.texts: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadatas: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self.texts)

    def add(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a single text item and its corresponding vector to the store.

        :param text: Raw text content of the chunk.
        :param vector: 1D numpy array representing the embedding vector.
        :param metadata: Optional dictionary containing additional information.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        self.texts.append(text)
        self.metadatas.append(metadata or {})

        if self.vectors is None:
            self.vectors = vector
        else:
            self.vectors = np.vstack([self.vectors, vector])

    def add_batch(
        self,
        texts: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add multiple text chunks and their embedding matrix simultaneously.

        :param texts: List of chunk text strings.
        :param vectors: 2D numpy array of shape (N, embedding_dim).
        :param metadatas: Optional list of N metadata dictionaries.
        """
        if len(texts) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        if len(texts) != vectors.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of vectors ({vectors.shape[0]})."
            )

        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a 1D query vector 'a' and a 2D matrix 'b'.
        Formula: (a · b) / (||a|| * ||b||)
        Optimized with unit vector dot product for pre-normalized embeddings.

        :param a: 1D query vector of shape (D,).
        :param b: 2D matrix of shape (N, D).
        :return: 1D array of shape (N,) containing similarity scores in [-1.0, 1.0].
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32)

        norm_a = float(np.linalg.norm(a))
        if norm_a > 0:
            a = a / norm_a

        dot_products = np.dot(b, a)
        norm_b = np.linalg.norm(b, axis=1)
        denominators = np.maximum(norm_b, 1e-10)
        return dot_products / denominators

    @staticmethod
    def _compute_keyword_overlap(query_text: str, passage: str) -> float:
        """
        Compute lightweight term overlap between query and passage.
        Extracts words > 2 chars, ignoring common punctuation.
        """
        q_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query_text.lower()))
        if not q_words:
            return 0.0
        p_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", passage.lower()))
        overlap = len(q_words & p_words)
        return float(overlap / len(q_words))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 2,
        min_score: Optional[float] = None,
        query_text: Optional[str] = None,
        hybrid_weight: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k most semantically similar chunks to the query vector.
        Supports optional similarity thresholding and hybrid scoring.

        :param query_vector: 1D embedding vector of the user query.
        :param top_k: Maximum number of results to return.
        :param min_score: Optional minimum score threshold (filters out irrelevant chunks).
        :param query_text: Optional raw query string for hybrid keyword matching.
        :param hybrid_weight: Weight for semantic similarity vs keyword overlap (default: 0.85).
        :return: List of dicts:
                 [{"score": float, "text": str, "metadata": dict}]
                 Sorted in descending order of score.
        """
        if self.vectors is None or len(self.texts) == 0:
            return []

        # 1. Compute cosine similarities via BLAS dot product
        cosine_scores = self.cosine_similarity(query_vector, self.vectors)

        # 2. Combine with keyword overlap if query_text is provided
        if query_text and hybrid_weight < 1.0:
            # Optimize by scoring candidates that have non-trivial semantic similarity
            # or are within top candidates, rather than running regex across all chunks
            candidate_indices = np.where(cosine_scores >= 0.05)[0]
            if len(candidate_indices) < top_k * 3:
                # If few chunks pass 0.05, consider top 20 by cosine
                candidate_indices = np.argsort(cosine_scores)[-20:][::-1]

            combined_scores = cosine_scores.copy()
            for idx in candidate_indices:
                text = self.texts[idx]
                kw_score = self._compute_keyword_overlap(query_text, text)
                cos_s = max(float(cosine_scores[idx]), 0.0)
                combined_scores[idx] = hybrid_weight * cos_s + (1.0 - hybrid_weight) * kw_score

            scores_to_rank = combined_scores
        else:
            scores_to_rank = cosine_scores

        # 3. Determine indices sorted descending
        sorted_indices = np.argsort(scores_to_rank)[::-1]

        results = []
        for idx in sorted_indices:
            score = float(round(float(scores_to_rank[idx]), 4))
            # Apply similarity threshold filtering if requested
            if min_score is not None and score < min_score:
                continue

            results.append({
                "score": score,
                "text": self.texts[idx],
                "metadata": self.metadatas[idx]
            })

            if len(results) >= top_k:
                break

        return results

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all stored chunks belonging to a specific document ID.
        Preserves original document order.
        """
        chunks = []
        for i, meta in enumerate(self.metadatas):
            if meta.get("document_id") == document_id:
                chunks.append({
                    "text": self.texts[i],
                    "metadata": meta,
                    "index": i
                })
        return chunks

    def save(self, index_dir: Any):
        """
        Persist in-memory vectors, texts, and metadata to disk.
        Uses NumPy binary format (.npy) for vectors and JSON for metadata.
        """
        import json

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        embeddings_file = path / "embeddings.npy"
        metadata_file = path / "metadata.json"

        if self.vectors is not None and len(self.texts) > 0:
            np.save(str(embeddings_file), self.vectors)
        else:
            if embeddings_file.exists():
                embeddings_file.unlink()

        payload = {
            "texts": self.texts,
            "metadatas": self.metadatas
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self, index_dir: Any) -> bool:
        """
        Load persisted vectors, texts, and metadata from disk into memory.
        Validates index integrity and performs auto-repair if mismatched.
        """
        import json

        path = Path(index_dir)
        embeddings_file = path / "embeddings.npy"
        metadata_file = path / "metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            return False

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = data.get("texts", [])
            metadatas = data.get("metadatas", [])
            vectors = np.load(str(embeddings_file))

            # Integrity check: vector row count must equal text count
            if len(texts) != vectors.shape[0]:
                print(
                    f"[VectorStore Warning] Mismatch between texts ({len(texts)}) "
                    f"and vectors ({vectors.shape[0]}). Truncating to valid range."
                )
                valid_count = min(len(texts), vectors.shape[0])
                texts = texts[:valid_count]
                metadatas = metadatas[:valid_count]
                vectors = vectors[:valid_count]

            self.texts = texts
            self.metadatas = metadatas
            self.vectors = vectors
            return True

        except Exception as err:
            print(f"[VectorStore Error] Failed to load index from '{index_dir}': {err}")
            return False

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Remove all stored chunks associated with a specific document ID.

        :param document_id: Unique ID of the document whose chunks should be removed.
        :return: Number of deleted chunks.
        """
        if self.vectors is None or len(self.texts) == 0:
            return 0

        keep_indices = []
        for i, meta in enumerate(self.metadatas):
            if meta.get("document_id") != document_id:
                keep_indices.append(i)

        deleted_count = len(self.texts) - len(keep_indices)
        if deleted_count == 0:
            return 0

        if len(keep_indices) == 0:
            self.texts = []
            self.metadatas = []
            self.vectors = None
        else:
            self.texts = [self.texts[i] for i in keep_indices]
            self.metadatas = [self.metadatas[i] for i in keep_indices]
            self.vectors = self.vectors[keep_indices]

        return deleted_count

    def clear(self, index_dir: Optional[Any] = None):
        """
        Reset the vector store in memory, and optionally purge saved disk files.
        """
        self.texts = []
        self.vectors = None
        self.metadatas = []

        if index_dir:
            path = Path(index_dir)
            embeddings_file = path / "embeddings.npy"
            metadata_file = path / "metadata.json"
            if embeddings_file.exists():
                embeddings_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
