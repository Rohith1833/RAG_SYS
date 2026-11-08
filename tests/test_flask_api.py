import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "ui"))
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))

from app import app


def test_flask_api():
    client = app.test_client()

    print("=" * 60)
    print("TEST SUITE 4: COMPLETE FLASK REST API VERIFICATION")
    print("=" * 60)

    # 1. GET /
    print("[1] Testing GET / (Web Canvas UI)...")
    res = client.get("/")
    assert res.status_code == 200
    assert b"ULTRON" in res.data
    print("  [PASS] Index page rendered successfully.")

    # 2. GET /api/health
    print("[2] Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "embedding_model" in data
    assert "device" in data
    assert data["embedding_dimension"] == 384
    print(f"  [PASS] Health API verified: status={data['status']}, device={data['device']}.")

    # 3. GET /api/info (v0.2 PyTorch Model)
    print("[3] Testing GET /api/info (v0.2 PyTorch model telemetry)...")
    res = client.get("/api/info")
    assert res.status_code == 200
    info = res.get_json()
    assert info["model"] == "Ultron"
    assert info["version"] == "v0.2"
    assert info["parameters"] == 1323728
    print(f"  [PASS] Model info verified: {info['parameters']:,} parameters on {info['device']}.")

    # 4. POST /api/generate (v0.2 PyTorch Inference)
    print("[4] Testing POST /api/generate (v0.2 PyTorch story generation)...")
    res = client.post("/api/generate", json={"prompt": "Once upon a time", "max_new_tokens": 15})
    assert res.status_code == 200
    gen_data = res.get_json()
    assert "text" in gen_data
    assert gen_data["tokens_generated"] > 0
    print(f"  [PASS] v0.2 Generation verified: {gen_data['tokens_generated']} tokens in {gen_data['latency_ms']}ms.")

    # 5. POST /api/conversations & GET /api/conversations
    print("[5] Testing Conversation Session Management...")
    res = client.post("/api/conversations", json={"title": "API Test Session", "mode": "general"})
    assert res.status_code == 201
    conv = res.get_json()
    conv_id = conv["id"]

    res_list = client.get("/api/conversations")
    assert res_list.status_code == 200
    conv_list = res_list.get_json()["conversations"]
    assert any(c["id"] == conv_id for c in conv_list)
    print(f"  [PASS] Conversation session created (id={conv_id}) and listed.")

    # 6. POST /api/chat (General Mode)
    print("[6] Testing POST /api/chat in General Chat mode...")
    res_chat = client.post("/api/chat", json={
        "message": "Respond with the single word: CONFIRMED",
        "mode": "general",
        "conversation_id": conv_id
    })
    assert res_chat.status_code == 200
    chat_data = res_chat.get_json()
    assert chat_data["mode"] == "general"
    assert len(chat_data["answer"]) > 0
    print(f"  [PASS] General chat responded: '{chat_data['answer'][:50]}...'")

    # Verify message persistence
    res_conv_get = client.get(f"/api/conversations/{conv_id}")
    assert res_conv_get.status_code == 200
    messages = res_conv_get.get_json()["messages"]
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    print("  [PASS] Messages successfully persisted in SQLite database.")

    # 7. POST /api/upload (Document Ingestion)
    print("[7] Testing POST /api/upload (Document Ingestion)...")
    file_content = b"The Ultron AI Assistant integrates a custom BPE Transformer with dense semantic RAG."
    data = {
        'file': (io.BytesIO(file_content), 'api_test_spec.txt')
    }
    res_up = client.post("/api/upload", content_type='multipart/form-data', data=data)
    assert res_up.status_code == 200
    up_data = res_up.get_json()
    doc_id = up_data["document"]["document_id"]
    assert up_data["status"] == "success"
    assert up_data["chunks_indexed"] >= 1
    print(f"  [PASS] Document ingested via API (id={doc_id}, chunks={up_data['chunks_indexed']}).")

    # 8. POST /api/chat (Document RAG Mode)
    print("[8] Testing POST /api/chat in Grounded Document RAG mode...")
    res_rag = client.post("/api/chat", json={
        "message": "What does the Ultron AI Assistant integrate?",
        "mode": "rag",
        "conversation_id": conv_id
    })
    assert res_rag.status_code == 200
    rag_data = res_rag.get_json()
    assert rag_data["mode"] == "rag"
    assert len(rag_data["sources"]) >= 1
    assert "api_test_spec.txt" in rag_data["sources"][0]["filename"]
    print(f"  [PASS] RAG answer generated with grounded sources: '{rag_data['answer'][:60]}...'")

    # 9. POST /api/retrieve (Direct Semantic Retrieval)
    print("[9] Testing POST /api/retrieve (Direct semantic search)...")
    res_ret = client.post("/api/retrieve", json={"query": "BPE Transformer", "top_k": 2})
    assert res_ret.status_code == 200
    ret_data = res_ret.get_json()
    assert ret_data["count"] >= 1
    print(f"  [PASS] Direct semantic retrieval returned {ret_data['count']} chunks.")

    # 10. POST /api/documents/<id>/summarize
    print("[10] Testing POST /api/documents/<id>/summarize...")
    res_sum = client.post(f"/api/documents/{doc_id}/summarize", json={"style": "short"})
    assert res_sum.status_code == 200
    sum_data = res_sum.get_json()
    assert "summary" in sum_data
    assert len(sum_data["summary"]) > 10
    print(f"  [PASS] Document summarization API returned summary: '{sum_data['summary'][:60]}...'")

    # 11. Cleanup: DELETE /api/documents/<id> & DELETE /api/conversations/<id>
    print("[11] Testing Document and Conversation Deletion Endpoints...")
    res_del_doc = client.delete(f"/api/documents/{doc_id}")
    assert res_del_doc.status_code == 200

    res_del_conv = client.delete(f"/api/conversations/{conv_id}")
    assert res_del_conv.status_code == 200
    print("  [PASS] Document and conversation deletion APIs verified.")

    # 12. Settings: GET & POST /api/settings
    print("[12] Testing Settings API...")
    res_set_get = client.get("/api/settings")
    assert res_set_get.status_code == 200

    res_set_post = client.post("/api/settings", json={"temperature": "0.65", "top_k": "4"})
    assert res_set_post.status_code == 200

    res_set_verify = client.get("/api/settings")
    assert res_set_verify.get_json()["temperature"] == "0.65"
    assert res_set_verify.get_json()["top_k"] == "4"
    print("  [PASS] Settings GET and POST APIs verified.")

    # 13. Token Streaming: POST /api/chat/stream
    print("[13] Testing Real-Time Token Streaming API (/api/chat/stream)...")
    res_stream = client.post("/api/chat/stream", json={"message": "Respond with 'Ultron online'.", "mode": "general"})
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.content_type
    stream_text = res_stream.get_data(as_text=True)
    assert "data: " in stream_text
    assert '"type": "token"' in stream_text
    assert '"type": "done"' in stream_text
    print("  [PASS] Token streaming SSE endpoint verified.")

    # 14. VRAM Management: POST /api/model/unload
    print("[14] Testing VRAM Unload API (/api/model/unload)...")
    res_unload = client.post("/api/model/unload")
    assert res_unload.status_code == 200
    assert res_unload.get_json().get("status") == "success"
    print("  [PASS] Model unload API verified.")

    print("=" * 60)
    print("TEST SUITE 4 PASSED: ALL FLASK API ENDPOINTS FULLY VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    test_flask_api()

