import sys
import tempfile
import shutil
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from rag_service import RAGService


def test_rag_service_lifecycle():
    print("=" * 65)
    print("TESTING RAG SERVICE LIFECYCLE")
    print("=" * 65)

    # Use temporary directories for isolation during testing
    temp_dir = Path(tempfile.mkdtemp(prefix="ultron_rag_test_"))
    index_dir = temp_dir / "index"
    uploads_dir = temp_dir / "uploads"

    try:
        service = RAGService(index_dir=index_dir, uploads_dir=uploads_dir)

        # 1. Health check
        print("\n[1] Testing Health Check...")
        health = service.get_health()
        print(f"    Status: {health['status']}, Ollama: {health['ollama']}, Device: {health['device']}")
        assert health["model"] == "llama3:latest"
        assert health["embedding_model"] == "all-MiniLM-L6-v2"
        assert health["documents"] == 0
        assert health["chunks"] == 0
        print("    [PASS] Health check verified.")

        # 2. Empty Knowledge Base Query
        print("\n[2] Testing Empty Store Query...")
        res_empty = service.ask_question("What is Ultron?")
        assert res_empty["empty_store"] is True
        assert "empty" in res_empty["answer"].lower()
        print("    [PASS] Empty knowledge base handled gracefully.")

        # 3. Upload Text Document
        print("\n[3] Testing Document Upload (TXT)...")
        sample_doc = (
            "Project Ultron Core Architecture Specification.\n"
            "Ultron v0.2 is trained on the TinyStories dataset.\n"
            "Ultron v0.2 uses a Byte-Level BPE tokenizer with 2000 vocabulary tokens.\n"
            "Ultron v0.2 has 1,323,728 parameters.\n"
            "Ultron uses llama3:latest running through Ollama for local RAG answering.\n"
        )
        res_upload = service.upload_document("ultron_spec.txt", sample_doc.encode("utf-8"))
        assert res_upload["status"] == "success"
        doc_id = res_upload["document"]["document_id"]
        assert res_upload["chunks_indexed"] >= 1
        print(f"    [PASS] Document uploaded. ID: {doc_id}, Chunks: {res_upload['chunks_indexed']}")

        # 4. List Documents
        print("\n[4] Testing Document Listing...")
        docs = service.list_documents()
        assert len(docs) == 1
        assert docs[0]["document_id"] == doc_id
        assert docs[0]["filename"] == "ultron_spec.txt"
        print(f"    [PASS] Document listed successfully: {docs[0]['filename']}")

        # 5. Grounded Question Answering
        print("\n[5] Testing Grounded Question Answering...")
        res_answer = service.ask_question("What is the parameter count of Ultron v0.2?")
        print(f"    Answer: {res_answer['answer']}")
        print(f"    Sources count: {len(res_answer['sources'])}")
        assert len(res_answer["sources"]) > 0
        assert "1,323,728" in res_answer["answer"]
        first_src = res_answer["sources"][0]
        assert first_src["filename"] == "ultron_spec.txt"
        assert first_src["score"] > 0.4
        print(f"    [PASS] Grounded answer generated with similarity score: {first_src['score']:.4f}")

        # 6. Unknown Question (Anti-Hallucination)
        print("\n[6] Testing Unknown Question Grounding...")
        res_unknown = service.ask_question("What is the author's favorite coffee brand?")
        print(f"    Answer: {res_unknown['answer']}")
        # The answer should state that the information is not provided
        assert any(phrase in res_unknown["answer"].lower() for phrase in [
            "not mentioned", "not contain", "not provide", "not available", "does not say", "no mention"
        ])
        print("    [PASS] Model refused to fabricate unknown information.")

        # 7. Persistence Check (Reload service from index_dir)
        print("\n[7] Testing Vector Store Persistence Across Restarts...")
        service2 = RAGService(index_dir=index_dir, uploads_dir=uploads_dir)
        assert len(service2.vector_store) == len(service.vector_store)
        assert len(service2.list_documents()) == 1
        print("    [PASS] Index loaded cleanly from disk.")

        # 8. Document Deletion
        print("\n[8] Testing Document Deletion...")
        del_res = service2.delete_document(doc_id)
        assert del_res["status"] == "success"
        assert len(service2.list_documents()) == 0
        assert len(service2.vector_store) == 0
        print("    [PASS] Document and associated vectors deleted.")

        # 9. Clear Index
        print("\n[9] Testing Clear Index...")
        # Upload another document first
        service2.upload_document("test.md", b"# Markdown Test\n\nContent")
        assert len(service2.list_documents()) == 1
        clear_res = service2.clear_index()
        assert clear_res["status"] == "success"
        assert len(service2.list_documents()) == 0
        assert len(service2.vector_store) == 0
        print("    [PASS] Clear index verified.")

        print("\n" + "=" * 65)
        print("ALL RAG SERVICE TESTS PASSED")
        print("=" * 65)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_rag_service_lifecycle()
