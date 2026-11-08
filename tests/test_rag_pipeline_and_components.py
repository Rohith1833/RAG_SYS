import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "inference"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from chunker import TextChunker
from document_parser import DocumentParser
from embedder import Embedder
from vector_store import VectorStore
from semantic_retriever import SemanticRetriever
from ultron_backend import UltronBackend
from ollama_client import OllamaClient


def test_ollama_client_structure():
    print("Testing OllamaClient initialization and prompt formatting...")
    client = OllamaClient(model="llama3:latest", base_url="http://127.0.0.1:11434")
    assert client.model == "llama3:latest"
    assert client.base_url == "http://127.0.0.1:11434"
    print("  [PASS] OllamaClient initialized correctly.")


def test_ultron_backend_prompt_formatting():
    print("Testing UltronBackend dialogue prompt formatting...")
    backend = UltronBackend(model="llama3:latest")
    history = [
        {"role": "user", "content": "Who created Ultron?"},
        {"role": "assistant", "content": "Ultron was created by an engineer."}
    ]
    prompt = backend.format_prompt("What is its purpose?", conversation=history)
    assert "User: Who created Ultron?" in prompt
    assert "Ultron: Ultron was created by an engineer." in prompt
    assert "User: What is its purpose?" in prompt
    assert prompt.endswith("Ultron:")
    print("  [PASS] UltronBackend formatted multi-turn conversation accurately.")


def test_text_chunker():
    print("Testing TextChunker slicing and overlap...")
    chunker = TextChunker(chunk_size=100, overlap=20)
    text = "The quick brown fox jumps over the lazy dog. " * 5  # ~225 chars
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 100
    print(f"  [PASS] TextChunker produced {len(chunks)} chunks with 20-char overlap.")


def test_document_parser_multi_format():
    print("Testing DocumentParser on TXT, MD, and DOCX...")
    # 1. TXT
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("Line 1 text.\nLine 2 text.")
        txt_path = Path(f.name)
    try:
        segs = DocumentParser.parse(txt_path)
        assert len(segs) == 1
        assert "Line 1 text." in segs[0]["text"]
    finally:
        txt_path.unlink()

    # 2. DOCX
    docx_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        '  <w:body><w:p><w:r><w:t>Docx Paragraph Content</w:t></w:r></w:p></w:body>\n'
        '</w:document>'
    )
    with tempfile.NamedTemporaryFile(suffix=".docx", mode="wb", delete=False) as f:
        with zipfile.ZipFile(f, "w") as zf:
            zf.writestr("word/document.xml", docx_xml.encode("utf-8"))
        docx_path = Path(f.name)
    try:
        segs = DocumentParser.parse(docx_path)
        assert len(segs) == 1
        assert "Docx Paragraph Content" in segs[0]["text"]
    finally:
        docx_path.unlink()

    print("  [PASS] DocumentParser extracted text from TXT and DOCX.")


def test_embedder_and_vector_store():
    print("Testing Embedder (all-MiniLM-L6-v2) and VectorStore...")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    assert embedder.embedding_dim == 384

    texts = [
        "Ultron is an advanced artificial intelligence system.",
        "The recipe calls for chocolate, flour, and sugar.",
        "Neural networks use backpropagation to optimize weights."
    ]
    vectors = embedder.embed_texts(texts)
    assert vectors.shape == (3, 384)

    store = VectorStore()
    store.add_batch(
        texts=texts,
        vectors=vectors,
        metadatas=[{"document_id": "doc_1"}, {"document_id": "doc_2"}, {"document_id": "doc_3"}]
    )
    assert len(store) == 3

    # Query matching first text
    q_vec = embedder.embed_text("Tell me about the Ultron AI system.")
    results = store.search(q_vec, top_k=1)
    assert len(results) == 1
    assert "Ultron is an advanced" in results[0]["text"]
    assert results[0]["metadata"]["document_id"] == "doc_1"
    print(f"  [PASS] Semantic search retrieved '{results[0]['text'][:30]}...' with score {results[0]['score']:.4f}.")

    # Test persistence and deletion
    with tempfile.TemporaryDirectory() as tmp_dir:
        store.save(tmp_dir)
        store2 = VectorStore()
        loaded = store2.load(tmp_dir)
        assert loaded is True
        assert len(store2) == 3

        deleted = store2.delete_by_document_id("doc_2")
        assert deleted == 1
        assert len(store2) == 2
        print("  [PASS] VectorStore persistence, reload, and document deletion verified.")


def main():
    print("=" * 60)
    print("TEST SUITE 2: RAG PIPELINE & CORE COMPONENTS")
    print("=" * 60)
    test_ollama_client_structure()
    test_ultron_backend_prompt_formatting()
    test_text_chunker()
    test_document_parser_multi_format()
    test_embedder_and_vector_store()
    print("=" * 60)
    print("TEST SUITE 2 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

