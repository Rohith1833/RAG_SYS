from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore:
    """
    An in-memory, numpy-based vector store.
    Stores text passages, their embedding vectors, and optional metadata.
    Performs nearest-neighbor semantic search using cosine similarity.
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

        :param a: 1D query vector of shape (D,).
        :param b: 2D matrix of shape (N, D).
        :return: 1D array of shape (N,) containing similarity scores in [-1.0, 1.0].
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32)

        dot_products = np.dot(b, a)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b, axis=1)

        # Avoid division by zero
        denominators = np.maximum(norm_a * norm_b, 1e-10)
        return dot_products / denominators

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k most semantically similar chunks to the query vector.

        :param query_vector: 1D embedding vector of the user query.
        :param top_k: Maximum number of results to return.
        :return: List of dicts:
                 [{"score": float, "text": str, "metadata": dict}]
                 Sorted in descending order of similarity score.
        """
        if self.vectors is None or len(self.texts) == 0:
            return []

        # Compute cosine similarity across all stored passage vectors
        similarities = self.cosine_similarity(query_vector, self.vectors)

        # Determine indices of top-k highest similarity scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(round(float(similarities[idx]), 4)),
                "text": self.texts[idx],
                "metadata": self.metadatas[idx]
            })

        return results

    def save(self, index_dir: Any):
        """
        Persist the in-memory vectors, texts, and metadata to disk.
        Uses NumPy binary format (.npy) for vectors and JSON for metadata.
        No external vector database is used.

        :param index_dir: Path or directory string where index files are saved.
        """
        import json
        from pathlib import Path

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

        :param index_dir: Directory containing 'embeddings.npy' and 'metadata.json'.
        :return: True if loaded successfully, False if index files do not exist.
        """
        import json
        from pathlib import Path

        path = Path(index_dir)
        embeddings_file = path / "embeddings.npy"
        metadata_file = path / "metadata.json"

        if not embeddings_file.exists() or not metadata_file.exists():
            return False

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.texts = data.get("texts", [])
            self.metadatas = data.get("metadatas", [])
            self.vectors = np.load(str(embeddings_file))

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

        :param index_dir: Optional index directory to clean on disk.
        """
        from pathlib import Path

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
