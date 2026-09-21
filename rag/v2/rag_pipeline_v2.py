import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# --------------------------------------------------
# Import Project Dependencies
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))

from ultron_backend import UltronBackend
from document_loader import DocumentLoader
from chunker import TextChunker
from embedder import Embedder
from vector_store import VectorStore
from semantic_retriever import SemanticRetriever


class RAGPipelineV2:
    """
    RAGPipelineV2 coordinates the end-to-end Semantic Retrieval-Augmented Generation pipeline:
    1. Document Ingestion (DocumentLoader)
    2. Text Chunking (TextChunker)
    3. Dense Embedding (Embedder with all-MiniLM-L6-v2)
    4. In-Memory Vector Store (VectorStore)
    5. Semantic Retrieval (SemanticRetriever with Cosine Similarity)
    6. Grounded Context Construction
    7. Answer Generation via UltronBackend (Ollama / llama3:latest)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        backend: Optional[UltronBackend] = None
    ):
        """
        Initialize the RAG V2 pipeline.

        :param chunk_size: Number of characters per chunk.
        :param overlap: Number of overlapping characters.
        :param embedding_model: HuggingFace sentence-transformer model identifier.
        :param backend: Existing UltronBackend instance. If None, instantiates a new one.
        """
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore()
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        self.backend = backend or UltronBackend(model="llama3:latest")
        self.chunks: List[str] = []

    def load_document(self, file_path: str | Path) -> int:
        """
        Ingest, chunk, embed, and index a document.

        :param file_path: Path to the source text document.
        :return: Number of indexed chunks.
        """
        raw_text = DocumentLoader.load_text(file_path)
        self.chunks = self.chunker.chunk_text(raw_text)

        # Index chunks in the semantic retriever
        metadatas = [{"source": str(file_path), "chunk_index": i} for i in range(len(self.chunks))]
        self.retriever.index_chunks(self.chunks, metadatas=metadatas)

        return len(self.chunks)

    def retrieve_chunks(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Find top-k most semantically relevant passages for a query.

        :param query: User's search query or question.
        :param top_k: Number of chunks to retrieve.
        :return: List of scored chunk dictionaries.
        """
        return self.retriever.retrieve(query=query, top_k=top_k)

    def build_rag_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Assemble the grounded prompt enforcing answer grounding in retrieved passages.

        :param query: User's question.
        :param retrieved_chunks: List of retrieved passages with similarity scores.
        :return: Complete prompt string for UltronBackend.
        """
        if not retrieved_chunks:
            context_body = "(No relevant documents or passages were found for this query.)"
        else:
            context_body = "\n\n---\n\n".join([
                f"[Passage {i+1} | Semantic Similarity: {c['score']:.4f}]:\n{c['text']}"
                for i, c in enumerate(retrieved_chunks)
            ])

        prompt = (
            "Answer the user's question using only the provided context.\n\n"
            "If the answer is not present in the context, say that the provided documents do not contain enough information.\n\n"
            "Context:\n"
            f"{context_body}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        return prompt

    def query(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Run the complete RAG V2 pipeline on a question.

        :param question: User's input question.
        :param top_k: Number of chunks to retrieve for grounding.
        :return: Dict containing the question, retrieved chunks, retrieval latency, prompt, and answer.
        """
        t0 = time.perf_counter()
        retrieved = self.retrieve_chunks(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        rag_prompt = self.build_rag_prompt(question, retrieved)
        answer = self.backend.respond(rag_prompt)

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "retrieval_time_ms": round(retrieval_ms, 2),
            "prompt": rag_prompt,
            "answer": answer
        }
