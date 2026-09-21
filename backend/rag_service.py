import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Import Project Dependencies
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from ultron_backend import UltronBackend
from chunker import TextChunker
from embedder import Embedder
from vector_store import VectorStore
from semantic_retriever import SemanticRetriever
from document_parser import DocumentParser
from document_manager import DocumentManager


class RAGService:
    """
    Central orchestration service for Ultron's end-to-end Local RAG system.

    Architectural Flow:
        DocumentParser (Parse TXT / MD / PDF)
            ↓
        TextChunker (Split with 500-char window and 100-char overlap)
            ↓
        Embedder (Dense 384-d vectors with all-MiniLM-L6-v2 on CUDA/CPU)
            ↓
        VectorStore (In-memory NumPy matrix + cosine similarity + local persistence)
            ↓
        SemanticRetriever (Dense vector query and ranking)
            ↓
        RAGService (Context assembly + anti-hallucination prompt enforcement)
            ↓
        UltronBackend (llama3:latest via Ollama)
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        document_manager: Optional[DocumentManager] = None,
        backend: Optional[UltronBackend] = None,
        chunk_size: int = 500,
        overlap: int = 100,
        index_dir: Optional[Path] = None,
        uploads_dir: Optional[Path] = None
    ):
        self.project_root = PROJECT_ROOT
        self.index_dir = index_dir or (self.project_root / "rag" / "index")
        self.uploads_dir = uploads_dir or (self.project_root / "rag" / "uploads")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # 1. Embedder: loaded once at startup (owns SentenceTransformer)
        self.embedder = embedder if embedder is not None else Embedder(model_name="all-MiniLM-L6-v2")

        # 2. VectorStore: NumPy in-memory storage (owns vectors & persistence)
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        loaded = self.vector_store.load(self.index_dir)
        if loaded:
            print(f"[RAGService] Loaded existing vector store: {len(self.vector_store)} chunk(s).")
        else:
            print("[RAGService] Initialized fresh vector store in memory.")

        # 3. SemanticRetriever: connects Embedder and VectorStore
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )

        # 4. Document Ingestion & Management
        self.document_manager = document_manager if document_manager is not None else DocumentManager(
            uploads_dir=self.uploads_dir,
            registry_path=self.index_dir / "documents.json"
        )
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

        # 5. LLM Reasoning Engine (UltronBackend -> Ollama -> llama3:latest)
        self.backend = backend if backend is not None else UltronBackend(model="llama3:latest")


    def upload_document(
        self,
        filename: str,
        file_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Validate, save, parse, chunk, embed, and index an uploaded document.

        :param filename: Original name of the uploaded file.
        :param file_bytes: Binary contents of the file.
        :return: Summary dict with status, doc_id, filename, chunks_created, and latency.
        """
        t_start = time.perf_counter()

        if not filename or not filename.strip():
            raise ValueError("Uploaded filename cannot be empty.")

        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

        safe_name = secure_filename(filename)
        if not safe_name:
            safe_name = f"upload_{uuid.uuid4().hex[:8]}.txt"

        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        stored_filename = f"{doc_id}_{safe_name}"
        saved_path = self.uploads_dir / stored_filename

        # Write uploaded file to disk
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        try:
            # Step 1: Parse document into page segments
            segments = DocumentParser.parse(saved_path)

            all_chunks: List[str] = []
            all_metadatas: List[Dict[str, Any]] = []

            # Step 2: Chunk each page segment while preserving page & document metadata
            chunk_global_id = 1
            for seg in segments:
                page_num = seg.get("page")
                page_text = seg.get("text", "")
                page_chunks = self.chunker.chunk_text(page_text)

                for c_text in page_chunks:
                    all_chunks.append(c_text)
                    all_metadatas.append({
                        "document_id": doc_id,
                        "filename": safe_name,
                        "chunk_id": chunk_global_id,
                        "page": page_num,
                        "char_count": len(c_text)
                    })
                    chunk_global_id += 1

            if not all_chunks:
                raise ValueError(f"No extractable text chunks could be generated from '{safe_name}'.")

            # Step 3: Generate dense semantic embeddings via Embedder
            vectors = self.embedder.embed_texts(all_chunks)

            # Step 4: Add vectors and chunks to VectorStore
            self.vector_store.add_batch(
                texts=all_chunks,
                vectors=vectors,
                metadatas=all_metadatas
            )

            # Step 5: Persist updated index to disk
            self.vector_store.save(self.index_dir)

            # Step 6: Register document record in DocumentManager
            record = self.document_manager.register_document(
                filename=safe_name,
                stored_path=saved_path,
                file_type=saved_path.suffix.lstrip(".").lower(),
                number_of_chunks=len(all_chunks),
                file_size=len(file_bytes),
                doc_id=doc_id
            )

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            return {
                "status": "success",
                "document": record,
                "chunks_indexed": len(all_chunks),
                "total_chunks_in_store": len(self.vector_store),
                "elapsed_ms": round(elapsed_ms, 2)
            }

        except Exception as err:
            # Clean up uploaded file if indexing fails
            if saved_path.exists():
                saved_path.unlink()
            raise err

    def ask_question(
        self,
        message: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Perform semantic retrieval and grounded answer generation.

        :param message: User's question or search query.
        :param top_k: Maximum number of semantic chunks to retrieve (default: 3).
        :return: Dict containing answer, sources list, retrieval time, and metrics.
        """
        clean_query = message.strip() if message else ""
        if not clean_query:
            raise ValueError("Question cannot be empty.")

        # Handle empty knowledge base gracefully
        if len(self.vector_store) == 0:
            return {
                "answer": (
                    "Your local knowledge base is currently empty. "
                    "Please upload a document (.txt, .md, or .pdf) using the Upload panel "
                    "to provide context for answering questions."
                ),
                "sources": [],
                "retrieval_ms": 0.0,
                "retrieved_count": 0,
                "empty_store": True
            }

        t_start = time.perf_counter()

        # Step 1: Semantic Retrieval via SemanticRetriever
        retrieved = self.retriever.retrieve(query=clean_query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_start) * 1000

        # Step 2: Format sources for UI inspection
        sources = []
        context_blocks = []

        for i, item in enumerate(retrieved, start=1):
            meta = item.get("metadata", {})
            filename = meta.get("filename", "unknown")
            page = meta.get("page")
            chunk_id = meta.get("chunk_id", i)
            score = item.get("score", 0.0)
            text = item.get("text", "")

            sources.append({
                "source_index": i,
                "filename": filename,
                "chunk_id": chunk_id,
                "page": page,
                "score": score,
                "preview": text.replace("\n", " ")[:180] + ("..." if len(text) > 180 else "")
            })

            page_info = f", Page {page}" if page is not None else ""
            context_blocks.append(
                f"[Source {i}: {filename}{page_info} | Cosine Similarity: {score:.4f}]\n{text}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        # Step 3: Construct strict grounding prompt
        grounded_prompt = (
            "You are Ultron, a helpful and precise local AI assistant.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question using ONLY the provided retrieved context below.\n"
            "2. Do NOT fabricate, extrapolate, or invent facts not present in the context.\n"
            "3. If the context does NOT contain enough information to answer the question, "
            "explicitly state that the provided documents do not contain the answer.\n"
            "4. Keep your response clear, concise, and grounded in the source passages.\n\n"
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{clean_query}\n\n"
            "GROUNDED ANSWER:"
        )

        # Step 4: Query UltronBackend (Ollama / llama3:latest)
        ollama_available = True
        generation_available = True
        try:
            answer = self.backend.respond(grounded_prompt)
        except ConnectionError:
            ollama_available = False
            generation_available = False
            answer = (
                "Semantic retrieval succeeded, but Ollama is currently offline, "
                "so llama3:latest cannot generate the final answer."
            )
        except TimeoutError:
            ollama_available = False
            generation_available = False
            answer = (
                "Semantic retrieval succeeded, but request to Ollama timed out."
            )
        except Exception as ex:
            ollama_available = False
            generation_available = False
            answer = (
                f"Semantic retrieval succeeded, but llama3:latest generation was unavailable: {str(ex)}"
            )

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "retrieved_count": len(retrieved),
            "empty_store": False,
            "ollama_available": ollama_available,
            "generation_available": generation_available
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return self.document_manager.list_documents()

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document, its physical file, and its vectors from the store.

        :param document_id: Unique ID of the document to delete.
        :return: Summary of deletion.
        """
        record = self.document_manager.remove_document(document_id)
        if not record:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        deleted_chunks = self.vector_store.delete_by_document_id(document_id)
        self.vector_store.save(self.index_dir)

        return {
            "status": "success",
            "deleted_document": record,
            "deleted_chunks": deleted_chunks,
            "remaining_chunks": len(self.vector_store)
        }

    def clear_index(self) -> Dict[str, Any]:
        """Purge all indexed documents and vectors."""
        self.document_manager.clear()
        self.vector_store.clear(self.index_dir)

        return {
            "status": "success",
            "message": "Knowledge base and vector index cleared."
        }

    def get_health(self) -> Dict[str, Any]:
        """
        Check health of Ollama backend, device metrics, and knowledge base stats.
        Distinguishes subsystem states (ingestion/embedding/retrieval vs generation).
        """
        ollama_ok = False
        ollama_error = None
        try:
            ollama_ok = self.backend.check_connection(timeout=1.5)
        except Exception as err:
            ollama_error = str(err)

        docs = self.document_manager.list_documents()

        return {
            "status": "ok" if ollama_ok else "degraded",
            "ollama": ollama_ok,
            "generation_ready": ollama_ok,
            "embedding_ready": True,
            "ingestion_ready": True,
            "retrieval_ready": True,
            "ollama_error": ollama_error,
            "model": self.backend.model,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dim,
            "device": self.embedder.device,
            "documents": len(docs),
            "chunks": len(self.vector_store)
        }
