import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from rag_service import RAGService
from query_reformulator import QueryReformulator
from summarizer import DocumentSummarizer
from database import Database


def test_duplicate_upload_deduplication():
    print("Testing document deduplication via SHA-256...")
    with tempfile.TemporaryDirectory() as tmp_idx, tempfile.TemporaryDirectory() as tmp_up:
        service = RAGService(index_dir=Path(tmp_idx), uploads_dir=Path(tmp_up))
        content = b"Ultron architecture notes: Transformers, BPE tokenization, and vector search."
        
        # 1st upload
        res1 = service.upload_document("notes.txt", content)
        assert res1["status"] == "success"
        doc_id = res1["document"]["document_id"]
        initial_chunks = len(service.vector_store)

        # 2nd upload with identical content
        res2 = service.upload_document("notes_copy.txt", content)
        assert res2["status"] == "already_indexed"
        assert res2["document"]["document_id"] == doc_id
        assert len(service.vector_store) == initial_chunks
        print("  [PASS] Duplicate document upload detected and prevented redundant indexing.")


def test_conversation_aware_reformulation():
    print("Testing conversation-aware follow-up reformulation...")
    reformulator = QueryReformulator()
    history = [
        {"role": "user", "content": "Who wrote The Stranger?"},
        {"role": "assistant", "content": "The Stranger was written by Albert Camus in 1942."}
    ]
    follow_up = "When was he born?"
    assert reformulator.is_ambiguous(follow_up) is True

    rewritten = reformulator.reformulate(follow_up, history)
    # Heuristic or LLM should associate Camus
    assert "Camus" in rewritten or "Albert" in rewritten or "The Stranger" in rewritten
    print(f"  [PASS] Reformulated '{follow_up}' -> '{rewritten}'.")


def test_general_chatbot_mode():
    print("Testing general chatbot mode (no RAG retrieval)...")
    with tempfile.TemporaryDirectory() as tmp_idx, tempfile.TemporaryDirectory() as tmp_up:
        service = RAGService(index_dir=Path(tmp_idx), uploads_dir=Path(tmp_up))
        # Vector store is empty, but general chat should work!
        assert len(service.vector_store) == 0

        res = service.chat("Explain what recursion is in 1 sentence.", mode="general")
        assert res["mode"] == "general"
        assert res["retrieved_count"] == 0
        assert len(res["sources"]) == 0
        assert len(res["answer"]) > 10
        print(f"  [PASS] General chat responded without RAG: '{res['answer'][:60]}...'")


def test_document_summarization_pipeline():
    print("Testing complete document summarization pipeline...")
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write(
            "Project Ultron is an autonomous local AI system. "
            "It incorporates a custom Transformer v0.1 model built from scratch, "
            "a 1.3M parameter v0.2 Byte-Level BPE model trained on TinyStories, "
            "and a local semantic RAG engine using all-MiniLM-L6-v2 embeddings. "
            "The system is designed for privacy, local operation, and grounded reasoning."
        )
        sample_doc = Path(f.name)

    try:
        summarizer = DocumentSummarizer()
        res = summarizer.summarize_document(sample_doc, style="short")
        assert "summary" in res
        assert len(res["summary"]) > 20
        assert res["sections_processed"] >= 1
        print(f"  [PASS] Document summarizer produced summary ({res['total_characters']} chars in {res['elapsed_seconds']}s): '{res['summary'][:70]}...'")
    finally:
        sample_doc.unlink()


def test_ollama_offline_resilience():
    print("Testing RAG resilience when Ollama generation is offline...")
    with tempfile.TemporaryDirectory() as tmp_idx, tempfile.TemporaryDirectory() as tmp_up:
        service = RAGService(index_dir=Path(tmp_idx), uploads_dir=Path(tmp_up))
        content = b"The capital of Australia is Canberra. It was selected as a compromise in 1908."
        service.upload_document("australia.txt", content)

        # Invalidate Ollama client URL to simulate offline service
        service.backend.client.base_url = "http://127.0.0.1:9999"

        # Ask question in RAG mode with Ollama offline
        res = service.chat("What is the capital of Australia?", mode="rag")
        assert res["ollama_available"] is False
        assert res["generation_available"] is False
        assert len(res["sources"]) >= 1
        assert "Canberra" in res["sources"][0]["preview"]
        assert "Ollama is currently offline" in res["answer"]
        print("  [PASS] Handled offline Ollama gracefully: semantic sources preserved and reported to user.")


def main():
    print("=" * 60)
    print("TEST SUITE 3: ADVANCED FEATURES (INGESTION, REFORMULATION, SUMMARIZATION, OFFLINE)")
    print("=" * 60)
    test_duplicate_upload_deduplication()
    test_conversation_aware_reformulation()
    test_general_chatbot_mode()
    test_document_summarization_pipeline()
    test_ollama_offline_resilience()
    print("=" * 60)
    print("TEST SUITE 3 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

