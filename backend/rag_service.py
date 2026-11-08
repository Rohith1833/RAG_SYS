import hashlib
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
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
from query_reformulator import QueryReformulator
from summarizer import DocumentSummarizer

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


class RAGService:
    """
    Central orchestration service for Ultron's dual-mode AI system.

    Capabilities:
    1. General Chatbot: Direct multi-turn dialogue with llama3:latest via Ollama.
    2. Document-Grounded RAG: Semantic retrieval + dense embeddings + cosine/hybrid search + strict context grounding.
    3. Conversation-Aware RAG: Rewrites ambiguous follow-up questions referencing dialogue history.
    4. Complete Document Summarization: Hierarchical map-reduce summarization of full documents.
    5. Knowledge Base Management: Ingestion, deduplication, deletion, and NumPy vector persistence.
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

        # 6. Conversation Reformulator & Document Summarizer
        self.reformulator = QueryReformulator(backend=self.backend)
        self.summarizer = DocumentSummarizer(backend=self.backend)

    def upload_document(
        self,
        filename: str,
        file_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Validate, save, parse, chunk, embed, and index an uploaded document.
        Detects duplicates and handles TXT, MD, PDF, and DOCX files.

        :param filename: Original name of the uploaded file.
        :param file_bytes: Binary contents of the file.
        :return: Summary dict with status, doc_id, filename, chunks_created, and latency.
        """
        t_start = time.perf_counter()

        if not filename or not filename.strip():
            raise ValueError("Uploaded filename cannot be empty.")

        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

        if len(file_bytes) > MAX_UPLOAD_SIZE:
            raise ValueError(
                f"File '{filename}' exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
            )

        safe_name = secure_filename(filename)
        if not safe_name:
            safe_name = f"upload_{uuid.uuid4().hex[:8]}.txt"

        ext = Path(safe_name).suffix.lower()
        if not DocumentParser.is_supported(safe_name):
            supported = ", ".join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file format '{ext}'. Ultron supports: {supported}")

        # Check for duplicate document content via SHA-256
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing_doc = self.document_manager.find_by_hash(file_hash)
        if existing_doc:
            return {
                "status": "already_indexed",
                "document": existing_doc,
                "chunks_indexed": 0,
                "total_chunks_in_store": len(self.vector_store),
                "message": f"Document '{existing_doc['filename']}' with identical content is already indexed in the knowledge base.",
                "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2)
            }

        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        stored_filename = f"{doc_id}_{safe_name}"
        saved_path = (self.uploads_dir / stored_filename).resolve()

        # Security: Enforce path traversal defense
        try:
            saved_path.relative_to(self.uploads_dir.resolve())
        except ValueError:
            raise ValueError(f"Illegal filename '{filename}' attempting path traversal.")

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
                file_type=ext.lstrip(".").lower(),
                number_of_chunks=len(all_chunks),
                file_size=len(file_bytes),
                doc_id=doc_id,
                file_hash=file_hash
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
                try:
                    saved_path.unlink()
                except Exception:
                    pass
            raise err

    def chat(
        self,
        message: str,
        mode: str = "rag",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        min_score: Optional[float] = 0.20
    ) -> Dict[str, Any]:
        """
        Unified conversational endpoint supporting both General Chat and Document RAG.

        :param message: User message / prompt.
        :param mode: 'general' (no document retrieval) or 'rag' (grounded document Q&A).
        :param conversation_history: Prior conversation turns: [{'role': 'user'|'assistant', 'content': '...'}]
        :param top_k: Maximum chunks to retrieve in RAG mode.
        :param min_score: Minimum similarity threshold in RAG mode.
        :return: Dict containing answer, sources, timing, and readiness metadata.
        """
        clean_msg = message.strip() if message else ""
        if not clean_msg:
            raise ValueError("Message cannot be empty.")

        mode_clean = mode.lower().strip()
        if mode_clean not in {"general", "rag"}:
            mode_clean = "rag"

        # ----------------------------------------------------
        # MODE 1: GENERAL CHAT (No Document Retrieval)
        # ----------------------------------------------------
        if mode_clean == "general":
            t0 = time.perf_counter()
            ollama_ok = True
            try:
                answer = self.backend.respond(clean_msg, conversation=conversation_history)
            except ConnectionError:
                ollama_ok = False
                answer = (
                    "Ollama is currently offline, so Ultron's language engine (llama3:latest) "
                    "cannot respond. Please start Ollama using 'ollama serve' in your terminal."
                )
            except Exception as ex:
                ollama_ok = False
                answer = f"Error during generation: {str(ex)}"

            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "answer": answer,
                "mode": "general",
                "sources": [],
                "retrieval_ms": 0.0,
                "latency_ms": round(latency_ms, 2),
                "retrieved_count": 0,
                "empty_store": False,
                "ollama_available": ollama_ok,
                "generation_available": ollama_ok
            }

        # ----------------------------------------------------
        # MODE 2: DOCUMENT-GROUNDED RAG
        # ----------------------------------------------------
        # Check if vector store has documents
        if len(self.vector_store) == 0:
            return {
                "answer": (
                    "Your local knowledge base is currently empty. "
                    "Please upload a document (.txt, .md, .pdf, or .docx) using the Document Panel "
                    "to use Document RAG mode, or switch to General Chat mode to talk with Ultron directly."
                ),
                "mode": "rag",
                "sources": [],
                "retrieval_ms": 0.0,
                "retrieved_count": 0,
                "empty_store": True,
                "ollama_available": True,
                "generation_available": True
            }

        t_start = time.perf_counter()

        # Step 1: Follow-Up Question Reformulation (Conversation-Aware)
        search_query = clean_msg
        if conversation_history:
            search_query = self.reformulator.reformulate(clean_msg, conversation_history)

        # Step 2: Semantic & Hybrid Retrieval
        retrieved = self.retriever.retrieve(
            query=search_query,
            top_k=top_k,
            min_score=min_score,
            hybrid=True
        )
        retrieval_ms = (time.perf_counter() - t_start) * 1000

        # If no chunks met the threshold, report absence of grounded information
        if not retrieved:
            # Fallback retrieve top 1 without min_score to check highest relevance
            fallback = self.retriever.retrieve(query=search_query, top_k=1, min_score=None, hybrid=True)
            best_score = fallback[0]["score"] if fallback else 0.0

            return {
                "answer": (
                    "I searched the indexed documents, but could not find information relevant to your question "
                    f"(highest match similarity was {best_score:.4f}, below threshold). "
                    "The uploaded documents do not appear to contain the answer. "
                    "You can switch to General Chat mode if you want me to answer using general knowledge."
                ),
                "mode": "rag",
                "sources": [],
                "retrieval_ms": round(retrieval_ms, 2),
                "retrieved_count": 0,
                "empty_store": False,
                "ollama_available": True,
                "generation_available": True
            }

        # Step 3: Format sources for UI
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
                "preview": text.replace("\n", " ")[:200] + ("..." if len(text) > 200 else "")
            })

            page_info = f", Page {page}" if page is not None else ""
            context_blocks.append(
                f"[Source {i}: {filename}{page_info} | Relevance Score: {score:.4f}]\n{text}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        # Build conversation dialogue summary if history exists
        history_context = ""
        if conversation_history:
            recent_turns = []
            for turn in conversation_history[-4:]:
                r = "User" if turn.get("role") == "user" else "Ultron"
                recent_turns.append(f"{r}: {turn.get('content', '').strip()}")
            if recent_turns:
                history_context = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(recent_turns) + "\n\n"

        # Step 4: Construct Grounding Prompt
        grounded_prompt = (
            "You are Ultron, a precise local AI assistant.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question using ONLY the provided retrieved context below.\n"
            "2. If the user refers to people or topics from prior dialogue (e.g., 'he', 'that book'), "
            "use the dialogue context to resolve the reference, but ground your factual answer in the retrieved context.\n"
            "3. Do NOT fabricate or invent facts not present in the retrieved passages.\n"
            "4. If the retrieved context does not contain enough information to answer, state clearly "
            "that the uploaded documents do not contain the answer.\n"
            "5. Cite the source document name and page number when available.\n\n"
            f"{history_context}"
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{clean_msg}\n\n"
            "GROUNDED ANSWER:"
        )

        # Step 5: Query UltronBackend (Ollama / llama3:latest)
        ollama_available = True
        generation_available = True
        try:
            answer = self.backend.respond(grounded_prompt)
        except ConnectionError:
            ollama_available = False
            generation_available = False
            answer = (
                "Semantic retrieval succeeded and identified relevant source passages, "
                "but Ollama is currently offline, so llama3:latest cannot generate the final answer. "
                "See the retrieved sources below for the relevant document excerpts."
            )
        except TimeoutError:
            ollama_available = False
            generation_available = False
            answer = "Semantic retrieval succeeded, but request to Ollama timed out."
        except Exception as ex:
            ollama_available = False
            generation_available = False
            answer = f"Semantic retrieval succeeded, but LLM generation encountered an error: {str(ex)}"

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "answer": answer,
            "mode": "rag",
            "search_query": search_query,
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "latency_ms": round(total_ms, 2),
            "retrieved_count": len(retrieved),
            "empty_store": False,
            "ollama_available": ollama_available,
            "generation_available": generation_available
        }

    def chat_stream(
        self,
        message: str,
        mode: str = "rag",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        min_score: Optional[float] = 0.20
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens incrementally for both General Chat and Document RAG.

        Yields standardized event dictionaries:
        1. {"type": "meta", ...} - Metadata containing mode, sources, search_query, retrieval_ms.
        2. {"type": "token", "token": str} - Incremental decoded tokens from Llama 3.
        3. {"type": "done", "total_ms": float, "full_answer": str} - Final completion event.
        4. {"type": "error", "error": str} - Failure event if Ollama is unreachable.
        """
        clean_msg = message.strip() if message else ""
        if not clean_msg:
            yield {"type": "error", "error": "Message cannot be empty."}
            return

        mode_clean = mode.lower().strip()
        if mode_clean not in {"general", "rag"}:
            mode_clean = "rag"

        t_start = time.perf_counter()

        # ----------------------------------------------------
        # MODE 1: GENERAL CHAT (STREAMING)
        # ----------------------------------------------------
        if mode_clean == "general":
            yield {
                "type": "meta",
                "mode": "general",
                "sources": [],
                "retrieval_ms": 0.0,
                "retrieved_count": 0,
                "empty_store": False
            }

            accumulated_answer = []
            try:
                for chunk in self.backend.respond_stream(clean_msg, conversation=conversation_history):
                    token = chunk.get("token", "")
                    if token:
                        accumulated_answer.append(token)
                        yield {"type": "token", "token": token}

                total_ms = (time.perf_counter() - t_start) * 1000
                yield {
                    "type": "done",
                    "total_ms": round(total_ms, 2),
                    "full_answer": "".join(accumulated_answer).strip()
                }

            except ConnectionError:
                err_msg = (
                    "Ollama is currently offline. Please start Ollama using 'ollama serve' in your terminal."
                )
                yield {"type": "token", "token": err_msg}
                yield {"type": "done", "total_ms": round((time.perf_counter() - t_start) * 1000, 2), "full_answer": err_msg}
            except Exception as ex:
                err_msg = f"Generation error: {str(ex)}"
                yield {"type": "error", "error": err_msg}
                yield {"type": "token", "token": err_msg}
                yield {"type": "done", "total_ms": round((time.perf_counter() - t_start) * 1000, 2), "full_answer": err_msg}
            return

        # ----------------------------------------------------
        # MODE 2: DOCUMENT RAG (STREAMING)
        # ----------------------------------------------------
        if len(self.vector_store) == 0:
            empty_msg = (
                "Your local knowledge base is currently empty. "
                "Please upload a document (.txt, .md, .pdf, or .docx) using the Document Panel "
                "to use Document RAG mode, or switch to General Chat mode to talk with Ultron directly."
            )
            yield {
                "type": "meta",
                "mode": "rag",
                "sources": [],
                "retrieval_ms": 0.0,
                "retrieved_count": 0,
                "empty_store": True
            }
            yield {"type": "token", "token": empty_msg}
            yield {"type": "done", "total_ms": 0.0, "full_answer": empty_msg}
            return

        # Step 1: Follow-Up Reformulation
        search_query = clean_msg
        if conversation_history:
            search_query = self.reformulator.reformulate(clean_msg, conversation_history)

        # Step 2: Semantic Retrieval
        t_ret_start = time.perf_counter()
        retrieved = self.retriever.retrieve(
            query=search_query,
            top_k=top_k,
            min_score=min_score,
            hybrid=True
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        if not retrieved:
            fallback = self.retriever.retrieve(query=search_query, top_k=1, min_score=None, hybrid=True)
            best_score = fallback[0]["score"] if fallback else 0.0
            no_info_msg = (
                "I searched the indexed documents, but could not find information relevant to your question "
                f"(highest match similarity was {best_score:.4f}, below threshold). "
                "The uploaded documents do not appear to contain the answer. "
                "You can switch to General Chat mode if you want me to answer using general knowledge."
            )
            yield {
                "type": "meta",
                "mode": "rag",
                "sources": [],
                "retrieval_ms": round(retrieval_ms, 2),
                "retrieved_count": 0,
                "empty_store": False
            }
            yield {"type": "token", "token": no_info_msg}
            yield {"type": "done", "total_ms": round((time.perf_counter() - t_start) * 1000, 2), "full_answer": no_info_msg}
            return

        # Format sources and context
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
                "preview": text.replace("\n", " ")[:200] + ("..." if len(text) > 200 else "")
            })

            page_info = f", Page {page}" if page is not None else ""
            context_blocks.append(
                f"[Source {i}: {filename}{page_info} | Relevance Score: {score:.4f}]\n{text}"
            )

        # Emit metadata event first so UI displays source cards immediately
        yield {
            "type": "meta",
            "mode": "rag",
            "search_query": search_query,
            "sources": sources,
            "retrieval_ms": round(retrieval_ms, 2),
            "retrieved_count": len(retrieved),
            "empty_store": False
        }

        context_str = "\n\n---\n\n".join(context_blocks)
        history_context = ""
        if conversation_history:
            recent_turns = []
            for turn in conversation_history[-4:]:
                r = "User" if turn.get("role") == "user" else "Ultron"
                recent_turns.append(f"{r}: {turn.get('content', '').strip()}")
            if recent_turns:
                history_context = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(recent_turns) + "\n\n"

        grounded_prompt = (
            "You are Ultron, a precise local AI assistant.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question using ONLY the provided retrieved context below.\n"
            "2. If the user refers to people or topics from prior dialogue (e.g., 'he', 'that book'), "
            "use the dialogue context to resolve the reference, but ground your factual answer in the retrieved context.\n"
            "3. Do NOT fabricate or invent facts not present in the retrieved passages.\n"
            "4. If the retrieved context does not contain enough information to answer, state clearly "
            "that the uploaded documents do not contain the answer.\n"
            "5. Cite the source document name and page number when available.\n\n"
            f"{history_context}"
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{clean_msg}\n\n"
            "GROUNDED ANSWER:"
        )

        accumulated_answer = []
        try:
            for chunk in self.backend.respond_stream(grounded_prompt):
                token = chunk.get("token", "")
                if token:
                    accumulated_answer.append(token)
                    yield {"type": "token", "token": token}

            total_ms = (time.perf_counter() - t_start) * 1000
            yield {
                "type": "done",
                "total_ms": round(total_ms, 2),
                "full_answer": "".join(accumulated_answer).strip()
            }

        except ConnectionError:
            offline_msg = (
                "Semantic retrieval succeeded and identified relevant source passages, "
                "but Ollama is currently offline, so llama3:latest cannot generate the final answer. "
                "See the retrieved sources above for the relevant document excerpts."
            )
            yield {"type": "token", "token": offline_msg}
            yield {"type": "done", "total_ms": round((time.perf_counter() - t_start) * 1000, 2), "full_answer": offline_msg}
        except Exception as ex:
            err_msg = f"Generation error: {str(ex)}"
            yield {"type": "error", "error": err_msg}
            yield {"type": "token", "token": err_msg}
            yield {"type": "done", "total_ms": round((time.perf_counter() - t_start) * 1000, 2), "full_answer": err_msg}

    def unload_model(self) -> bool:
        """Unload the LLM from GPU VRAM."""
        return self.backend.unload_model()

    def ask_question(
        self,
        message: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Legacy RAG Q&A wrapper preserved for full backward compatibility with tests.
        """
        return self.chat(message=message, mode="rag", top_k=top_k, min_score=None)

    def summarize_document(
        self,
        document_id: str,
        style: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Summarize an entire indexed document using hierarchical summarization.

        :param document_id: Unique ID of the document to summarize.
        :param style: 'short', 'detailed', or 'key_points'.
        :return: Summary result dictionary.
        """
        doc = self.document_manager.get_document(document_id)
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        stored_path = Path(doc.get("stored_path", ""))
        if not stored_path.exists():
            raise FileNotFoundError(f"Underlying file for document '{document_id}' was not found on disk.")

        return self.summarizer.summarize_document(
            file_path=stored_path,
            style=style
        )

    def direct_retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Perform direct semantic retrieval without LLM generation."""
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            hybrid=True
        )

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
