import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add local directory to path for clean imports
V2_DIR = Path(__file__).resolve().parent
sys.path.append(str(V2_DIR))

from embedder import Embedder
from vector_store import VectorStore


class SemanticRetriever:
    """
    Coordinates semantic vector retrieval by composing an Embedder
    and an in-memory VectorStore.
    Maps text queries to dense vectors and finds the closest document chunks.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the semantic retriever.

        :param embedder: An Embedder instance. If None, instantiates a default all-MiniLM-L6-v2 embedder.
        :param vector_store: A VectorStore instance. If None, creates a new in-memory store.
        """
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = vector_store if vector_store is not None else VectorStore()

    def index_chunks(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Embed and store a collection of text chunks.

        :param chunks: List of chunk text passages.
        :param metadatas: Optional list of metadata dicts corresponding to chunks.
        :return: Number of indexed passages.
        """
        if not chunks:
            return 0

        # Step 1: Generate embeddings for all text chunks in a batch
        vectors = self.embedder.embed_texts(chunks)

        # Step 2: Store chunks and their vectors
        self.vector_store.add_batch(
            texts=chunks,
            vectors=vectors,
            metadatas=metadatas
        )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Embed a user's natural language query and retrieve the top-k most similar chunks.

        :param query: User query text.
        :param top_k: Number of highest-similarity passages to return.
        :return: List of dicts with {"score": float, "text": str, "metadata": dict}.
        """
        # Step 1: Embed the search query into the same semantic space
        query_vector = self.embedder.embed_text(query)

        # Step 2: Execute similarity search against stored document vectors
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        return results
