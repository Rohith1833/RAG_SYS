import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# --------------------------------------------------
# Import Existing Project Dependencies
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))

from ultron_backend import UltronBackend
from document_loader import DocumentLoader
from chunker import TextChunker
from retriever import KeywordRetriever


class RAGPipeline:
    """
    RAGPipeline coordinates the complete Retrieval-Augmented Generation workflow:
    1. Document Loading
    2. Text Chunking
    3. In-Memory Passage Storage
    4. Keyword-based Retrieval
    5. Context Injection
    6. Grounded Answer Generation via UltronBackend (llama3:latest)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        backend: Optional[UltronBackend] = None
    ):
        """
        Initialize the RAG pipeline.

        :param chunk_size: Character length per chunk.
        :param overlap: Character overlap between consecutive chunks.
        :param backend: Existing UltronBackend instance. If None, creates a new one.
        """
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.backend = backend or UltronBackend(model="llama3:latest")
        self.chunks: List[str] = []
        self.retriever: Optional[KeywordRetriever] = None

    def load_document(self, file_path: str | Path) -> int:
        """
        Load a text file, chunk it, and index it for keyword retrieval.

        :param file_path: Path to the text document.
        :return: Total number of chunks generated and stored.
        """
        raw_text = DocumentLoader.load_text(file_path)
        self.chunks = self.chunker.chunk_text(raw_text)
        self.retriever = KeywordRetriever(self.chunks)
        return len(self.chunks)

    def retrieve_chunks(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks matching the user's query keywords.

        :param query: The user's question.
        :param top_k: Maximum chunks to retrieve.
        :return: List of scored chunk dictionaries: [{"score": int, "text": str}]
        """
        if not self.retriever:
            raise RuntimeError("No document has been loaded into the RAG pipeline yet.")

        return self.retriever.retrieve(query=query, top_k=top_k)

    def build_rag_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Assemble the grounded prompt containing the retrieved context and question.

        :param query: The user's question.
        :param retrieved_chunks: Relevant passages retrieved from the index.
        :return: Formatted prompt string for the LLM.
        """
        if not retrieved_chunks:
            context_body = "(No relevant documents or passages were found for this query.)"
        else:
            context_body = "\n\n---\n\n".join(
                [f"[Passage {i+1} | Score {c['score']}]:\n{c['text']}" for i, c in enumerate(retrieved_chunks)]
            )

        prompt = (
            "Use the following retrieved context to answer the user's question.\n\n"
            "Context:\n"
            f"{context_body}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer using the provided context. If the answer cannot be found in the context, "
            "clearly say that the provided documents do not contain enough information."
        )

        return prompt

    def query(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Execute the end-to-end RAG flow: retrieve context and generate a grounded answer.

        :param question: The user's query.
        :param top_k: Number of chunks to retrieve for context.
        :return: Dictionary containing the retrieved chunks, formatted prompt, and LLM answer.
        """
        # Step 1: Retrieval
        retrieved = self.retrieve_chunks(question, top_k=top_k)

        # Step 2: Context Building
        rag_prompt = self.build_rag_prompt(question, retrieved)

        # Step 3: LLM Generation through UltronBackend
        answer = self.backend.respond(rag_prompt)

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "prompt": rag_prompt,
            "answer": answer
        }
