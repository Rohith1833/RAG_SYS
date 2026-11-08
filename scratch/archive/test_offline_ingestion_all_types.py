import urllib.request
import json
import time
from pathlib import Path

def upload_file(filename: str, content: bytes, mime_type: str):
    boundary = "----WebKitFormBoundaryUploadTest456"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + content + footer

    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        elapsed = round((time.time() - t0) * 1000, 2)
        res_data = json.loads(resp.read().decode())
        print(f"[{filename}] Upload status: {res_data.get('status')}, Chunks: {res_data.get('chunks_indexed')}, Store total: {res_data.get('total_chunks_in_store')}, Latency: {elapsed}ms")
        return res_data

def test_chat(query: str):
    payload = json.dumps({"message": query, "top_k": 3}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        elapsed = round((time.time() - t0) * 1000, 2)
        res_data = json.loads(resp.read().decode())
        print(f"\n[Query: '{query}'] (Status: {resp.status}, Server took: {elapsed}ms)")
        print(f"  Ollama Available: {res_data.get('ollama_available')}")
        print(f"  Generation Available: {res_data.get('generation_available')}")
        print(f"  Answer: {res_data.get('answer')}")
        print(f"  Retrieved count: {res_data.get('retrieved_count')} in {res_data.get('retrieval_ms')}ms")
        for s in res_data.get("sources", []):
            print(f"    - Source {s.get('source_index')}: {s.get('filename')} (score: {s.get('score')}) | '{s.get('preview')}'")
        return res_data

if __name__ == "__main__":
    print("=== Testing Uploads with Ollama OFF ===")
    
    # 1. TXT upload
    txt_content = b"Ultron architecture principle: Document ingestion and indexing must operate independently of Ollama. MiniLM creates dense vectors directly in PyTorch.\nUltron knowledge base uses a NumPy matrix for cosine similarity."
    upload_file("offline_test.txt", txt_content, "text/plain")

    # 2. MD upload
    md_content = b"# Local RAG System\n\n- Fact A: All embeddings are 384-dimensional.\n- Fact B: VectorStore persists embeddings.npy and metadata.json.\n- Fact C: Ollama is only called for final response generation."
    upload_file("offline_test.md", md_content, "text/markdown")

    # 3. PDF upload
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 68 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Ultron PDF Knowledge: Ingestion operates without Ollama.) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000362 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"441\n"
        b"%%EOF\n"
    )
    upload_file("offline_test.pdf", pdf_bytes, "application/pdf")

    # 4. Chat query with Ollama OFF
    print("\n=== Testing Chat Query with Ollama OFF ===")
    test_chat("What is the Ultron architecture principle regarding document ingestion?")
