import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "ui"))
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))

from app import app, rag_service, db


def run_scenarios():
    client = app.test_client()

    print("=" * 70)
    print("REAL-WORLD SCENARIO VERIFICATION SUITE (TESTS A THROUGH N)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # TEST A: Upload a TXT document and ask a question about it
    # -----------------------------------------------------------------
    print("\n--- TEST A: Upload TXT document and ask a question about it ---")
    txt_content = b"The mission of Ultron is to provide sovereign local intelligence without external cloud dependence."
    res = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(txt_content), 'scenario_mission.txt')
    })
    assert res.status_code == 200
    doc_a_id = res.get_json()["document"]["document_id"]
    print(f"  Uploaded TXT (id={doc_a_id})")

    res_q = client.post("/api/chat", json={
        "message": "What is the mission of Ultron?",
        "mode": "rag"
    })
    assert res_q.status_code == 200
    ans_a = res_q.get_json()
    assert len(ans_a["sources"]) >= 1
    assert "scenario_mission.txt" in ans_a["sources"][0]["filename"]
    print(f"  [PASS TEST A] Answer: {ans_a['answer'][:80]}...")

    # -----------------------------------------------------------------
    # TEST B: Upload a PDF and ask a question about a particular page
    # -----------------------------------------------------------------
    print("\n--- TEST B: Upload PDF and ask question with page reference ---")
    import pypdf
    writer = pypdf.PdfWriter()
    p1 = writer.add_blank_page(width=300, height=300)
    # Use existing sample PDF or create a 2-page PDF
    pdf_path = PROJECT_ROOT / "scratch" / "sample_doc.pdf"
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
    else:
        # Fallback minimal 1-page PDF
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
            b"4 0 obj << /Length 55 >> stream\n"
            b"BT /F1 12 Tf 72 712 Td (Ultron Page 1 contains key specifications) Tj ET\n"
            b"endstream endobj\n5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000349 00000 n \ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n418\n%%EOF"
        )
    res_b = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(pdf_bytes), 'scenario_manual.pdf')
    })
    assert res_b.status_code == 200
    doc_b_id = res_b.get_json()["document"]["document_id"]
    print(f"  Uploaded PDF (id={doc_b_id})")

    res_qb = client.post("/api/chat", json={
        "message": "What does Page 1 contain?",
        "mode": "rag"
    })
    assert res_qb.status_code == 200
    ans_b = res_qb.get_json()
    assert len(ans_b["sources"]) >= 1
    assert ans_b["sources"][0]["page"] is not None
    print(f"  [PASS TEST B] Answer with Page {ans_b['sources'][0]['page']} reference: {ans_b['answer'][:80]}...")

    # -----------------------------------------------------------------
    # TEST C: Upload a Markdown document and summarize it
    # -----------------------------------------------------------------
    print("\n--- TEST C: Upload Markdown document and summarize it ---")
    md_content = (
        b"# Ultron Technical Overview\n\n"
        b"## Component 1: Custom PyTorch BPE Model\n"
        b"Ultron includes a 1.3M parameter model trained on TinyStories.\n\n"
        b"## Component 2: RAG Pipeline\n"
        b"Semantic search uses 384-dimensional dense vectors and cosine similarity."
    )
    res_c = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(md_content), 'scenario_guide.md')
    })
    assert res_c.status_code == 200
    doc_c_id = res_c.get_json()["document"]["document_id"]

    res_sum_c = client.post(f"/api/documents/{doc_c_id}/summarize", json={"style": "short"})
    assert res_sum_c.status_code == 200
    sum_c = res_sum_c.get_json()
    assert len(sum_c["summary"]) > 20
    print(f"  [PASS TEST C] Markdown Summary: {sum_c['summary'][:90]}...")

    # -----------------------------------------------------------------
    # TEST D: Upload a long document and request a detailed summary
    # -----------------------------------------------------------------
    print("\n--- TEST D: Upload long document and request detailed summary ---")
    long_sections = []
    for i in range(1, 5):
        long_sections.append(
            f"Chapter {i}: System Engineering Deep Dive. "
            f"This section discusses layer normalization, residual connections, "
            f"multi-head self-attention mechanisms, and feed-forward neural layers. "
            f"Ultron maintains complete execution sovereignty on the host GPU without network calls. "
            * 8
        )
    long_content = "\n\n".join(long_sections).encode("utf-8")
    res_d = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(long_content), 'scenario_long_spec.txt')
    })
    assert res_d.status_code == 200
    doc_d_id = res_d.get_json()["document"]["document_id"]
    print(f"  Uploaded long document ({len(long_content)} bytes)")

    res_sum_d = client.post(f"/api/documents/{doc_d_id}/summarize", json={"style": "detailed"})
    assert res_sum_d.status_code == 200
    sum_d = res_sum_d.get_json()
    assert len(sum_d["summary"]) > 50
    print(f"  [PASS TEST D] Detailed Summary ({sum_d['total_characters']} chars): {sum_d['summary'][:90]}...")

    # -----------------------------------------------------------------
    # TEST E: Ask a follow-up question referring to the previous answer
    # -----------------------------------------------------------------
    print("\n--- TEST E: Conversation-aware follow-up question ---")
    res_conv = client.post("/api/conversations", json={"title": "Follow-Up Test", "mode": "rag"})
    conv_id = res_conv.get_json()["id"]

    # Turn 1
    t1 = client.post("/api/chat", json={
        "message": "What is the mission of Ultron?",
        "mode": "rag",
        "conversation_id": conv_id
    })
    assert t1.status_code == 200

    # Turn 2: Follow-up with relative pronoun 'it'
    t2 = client.post("/api/chat", json={
        "message": "Does it require external cloud servers?",
        "mode": "rag",
        "conversation_id": conv_id
    })
    assert t2.status_code == 200
    ans_e = t2.get_json()
    assert len(ans_e["sources"]) >= 1
    print(f"  [PASS TEST E] Follow-up answer: {ans_e['answer'][:90]}...")

    # -----------------------------------------------------------------
    # TEST F: Ask a question whose answer is absent from the documents
    # -----------------------------------------------------------------
    print("\n--- TEST F: Question absent from documents ---")
    res_f = client.post("/api/chat", json={
        "message": "What is the exact recipe for baking sourdough bread?",
        "mode": "rag"
    })
    assert res_f.status_code == 200
    ans_f = res_f.get_json()
    ans_f_text = ans_f["answer"].lower()
    refused = any(term in ans_f_text for term in [
        "do not contain", "don't contain", "don't have", "not found",
        "could not find", "does not contain", "doesn't contain", "no information"
    ])
    assert refused, f"Model failed to refuse fabrication: {ans_f['answer']}"
    print(f"  [PASS TEST F] Successfully refused fabrication: {ans_f['answer'][:90]}...")

    # -----------------------------------------------------------------
    # TEST G: Upload multiple documents and ask a question comparing them
    # -----------------------------------------------------------------
    print("\n--- TEST G: Multiple documents comparison ---")
    doc1 = b"Model Alpha has 10 million parameters and runs on CPU."
    doc2 = b"Model Beta has 50 million parameters and requires a GPU."
    client.post("/api/upload", content_type='multipart/form-data', data={'file': (io.BytesIO(doc1), 'alpha.txt')})
    client.post("/api/upload", content_type='multipart/form-data', data={'file': (io.BytesIO(doc2), 'beta.txt')})

    res_g = client.post("/api/chat", json={
        "message": "Compare Model Alpha and Model Beta parameter sizes and hardware requirements.",
        "mode": "rag"
    })
    assert res_g.status_code == 200
    ans_g = res_g.get_json()
    assert len(ans_g["sources"]) >= 2
    print(f"  [PASS TEST G] Comparison answer with {len(ans_g['sources'])} sources: {ans_g['answer'][:90]}...")

    # -----------------------------------------------------------------
    # TEST H: Restart persistence check (verify disk reload)
    # -----------------------------------------------------------------
    print("\n--- TEST H: State persistence across restarts ---")
    from vector_store import VectorStore
    new_store = VectorStore()
    loaded = new_store.load(rag_service.index_dir)
    assert loaded is True
    assert len(new_store) > 0
    print(f"  [PASS TEST H] Successfully verified {len(new_store)} vector chunks restored from disk.")

    # -----------------------------------------------------------------
    # TEST I: Ollama offline verification (ingestion and retrieval still work)
    # -----------------------------------------------------------------
    print("\n--- TEST I: Offline Ollama resilience ---")
    # Temporarily set offline URL
    orig_url = rag_service.backend.client.base_url
    rag_service.backend.client.base_url = "http://127.0.0.1:9999"

    # Ingestion must still work!
    res_i = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(b"Offline document indexing test without Ollama."), 'offline_doc.txt')
    })
    assert res_i.status_code == 200
    print("  Document ingested while Ollama is offline.")

    # Direct retrieval must still work!
    res_ret_i = client.post("/api/retrieve", json={"query": "Offline indexing", "top_k": 1})
    assert res_ret_i.status_code == 200
    assert res_ret_i.get_json()["count"] >= 1
    print("  Direct semantic retrieval worked while Ollama is offline.")

    # Chat must report offline status gracefully without crashing
    res_chat_i = client.post("/api/chat", json={"message": "What is in offline_doc?", "mode": "rag"})
    assert res_chat_i.status_code == 200
    assert res_chat_i.get_json()["ollama_available"] is False
    print("  [PASS TEST I] Ingestion and retrieval verified while Ollama is offline.")

    # -----------------------------------------------------------------
    # TEST J: Ollama online verification
    # -----------------------------------------------------------------
    print("\n--- TEST J: Restore Ollama online status ---")
    rag_service.backend.client.base_url = orig_url
    res_j = client.post("/api/chat", json={"message": "Say hello Ultron", "mode": "general"})
    assert res_j.status_code == 200
    assert res_j.get_json()["ollama_available"] is True
    print(f"  [PASS TEST J] Ollama restored: '{res_j.get_json()['answer'][:50]}...'")

    # -----------------------------------------------------------------
    # TEST K: General question without document mode
    # -----------------------------------------------------------------
    print("\n--- TEST K: General question in General Chat mode ---")
    res_k = client.post("/api/chat", json={
        "message": "Write a 3-line Python function to compute the square of a number.",
        "mode": "general"
    })
    assert res_k.status_code == 200
    ans_k = res_k.get_json()
    assert ans_k["mode"] == "general"
    assert len(ans_k["sources"]) == 0
    print(f"  [PASS TEST K] General answer: '{ans_k['answer'][:70]}...'")

    # -----------------------------------------------------------------
    # TEST L: Delete document and verify chunk/embedding removal
    # -----------------------------------------------------------------
    print("\n--- TEST L: Document deletion and vector purge ---")
    chunks_before = len(rag_service.vector_store)
    res_l = client.delete(f"/api/documents/{doc_a_id}")
    assert res_l.status_code == 200
    chunks_after = len(rag_service.vector_store)
    assert chunks_after < chunks_before
    print(f"  [PASS TEST L] Deleted document {doc_a_id}, chunks reduced from {chunks_before} to {chunks_after}.")

    # -----------------------------------------------------------------
    # TEST M: Invalid uploads and malformed requests
    # -----------------------------------------------------------------
    print("\n--- TEST M: Malformed uploads and requests ---")
    # Empty file
    res_m1 = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(b""), 'empty.txt')
    })
    assert res_m1.status_code == 400

    # Unsupported extension
    res_m2 = client.post("/api/upload", content_type='multipart/form-data', data={
        'file': (io.BytesIO(b"data"), 'virus.exe')
    })
    assert res_m2.status_code == 400

    # Empty chat query
    res_m3 = client.post("/api/chat", json={"message": "   ", "mode": "general"})
    assert res_m3.status_code == 400
    print("  [PASS TEST M] Empty files, unsupported extensions, and blank prompts cleanly rejected.")

    # -----------------------------------------------------------------
    # TEST N: UI screen size responsiveness check
    # -----------------------------------------------------------------
    print("\n--- TEST N: Responsive UI viewport check ---")
    index_html = (PROJECT_ROOT / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in index_html

    css_text = (PROJECT_ROOT / "ui" / "static" / "style.css").read_text(encoding="utf-8")
    assert "@media (max-width: 859px)" in css_text
    assert "@media (min-width: 1200px)" in css_text
    print("  [PASS TEST N] Mobile and desktop responsive layouts verified.")

    print("\n" + "=" * 70)
    print("ALL REAL-WORLD SCENARIOS (A THROUGH N) PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_scenarios()
