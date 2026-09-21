import sys
import time
from pathlib import Path

# Add paths for clean resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))

from rag_pipeline_v2 import RAGPipelineV2


def main():
    print("=" * 70)
    print("ULTRON RAG V2 (SEMANTIC EMBEDDINGS) VERIFICATION SUITE")
    print("=" * 70)

    # 1. Initialize Pipeline
    t_init_start = time.perf_counter()
    pipeline = RAGPipelineV2(
        chunk_size=500,
        overlap=100,
        embedding_model="all-MiniLM-L6-v2"
    )
    init_time_sec = time.perf_counter() - t_init_start

    # 2. Performance & Configuration Information (Step 9.9)
    print("\n--- Model & System Telemetry ---")
    print(f"Embedding Model:     {pipeline.embedder.model_name}")
    print(f"Embedding Dimension: {pipeline.embedder.embedding_dim}")
    print(f"Embedder Device:     {pipeline.embedder.device}")
    print(f"Model Load Time:     {init_time_sec:.2f}s")
    print(f"LLM Backend Model:   {pipeline.backend.model}")

    # Check Ollama connection
    try:
        pipeline.backend.check_connection()
        print("[OK] Verified connection to local Ollama server\n")
    except ConnectionError as err:
        print(f"[Connection Error] {err}")
        sys.exit(1)

    # 3. Ingest & Index Document
    doc_path = PROJECT_ROOT / "rag" / "documents" / "ultron_notes.txt"
    t_embed_start = time.perf_counter()
    num_chunks = pipeline.load_document(doc_path)
    embed_time_ms = (time.perf_counter() - t_embed_start) * 1000

    print(f"Document ingested:   {doc_path}")
    print(f"Total Chunks:        {num_chunks}")
    print(f"Chunking + Embedding Time: {embed_time_ms:.2f}ms")
    print("-" * 70)

    # 4. Standard RAG Questions (Step 9.6)
    questions = [
        "What is the parameter count of Ultron v0.2?",
        "Which tokenizer is used by the second version of Ultron?",
        "What model powers the practical Ultron backend?",
        "How long is the context window of Ultron v0.2?",
        "What operating system was used to train Ultron v0.2?"
    ]

    for idx, question in enumerate(questions, 1):
        is_unknown_test = (idx == 5)
        tag = "[GROUNDING TEST: UNKNOWN FACT]" if is_unknown_test else "[SEMANTIC RETRIEVAL TEST]"

        print("\n" + "=" * 70)
        print(f"QUESTION {idx}: \"{question}\"  {tag}")
        print("=" * 70)

        # Retrieve relevant passages
        t0 = time.perf_counter()
        retrieved = pipeline.retrieve_chunks(question, top_k=2)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        print(f"Retrieved {len(retrieved)} chunk(s) in {retrieval_ms:.2f}ms:")
        for c_idx, c in enumerate(retrieved, 1):
            excerpt = c['text'].replace('\n', ' ')
            if len(excerpt) > 110:
                excerpt = excerpt[:110] + "..."
            print(f"  [Passage {c_idx}] Cosine Similarity: {c['score']:.4f}")
            print(f"      Text: \"{excerpt}\"")

        print("\nSending context to UltronBackend (llama3:latest)...")
        result = pipeline.query(question, top_k=2)

        print("\nUltron Answer:")
        print(result["answer"])
        print("-" * 70)

    # 5. Semantic Retrieval Demonstration (Step 9.7)
    # Queries with intentional phrasing divergence (zero or low keyword overlap)
    print("\n" + "=" * 70)
    print("STEP 9.7: SEMANTIC RETRIEVAL DEMONSTRATION (PARAPHRASED QUERIES)")
    print("=" * 70)
    print("Testing queries where wording differs significantly from the document text,\n"
          "demonstrating dense semantic matching where pure keyword matching is weak.\n")

    paraphrased_queries = [
        {
            "query": "What method does the second model use to break text into tokens?",
            "target_fact": "Ultron v0.2 uses a custom Byte-Level BPE tokenizer."
        },
        {
            "query": "Which language model is currently running behind the assistant?",
            "target_fact": "The current practical Ultron backend uses llama3:latest through Ollama."
        }
    ]

    for p_idx, item in enumerate(paraphrased_queries, 1):
        q = item["query"]
        print(f"--- Paraphrased Query {p_idx} ---")
        print(f"Query:        \"{q}\"")
        print(f"Target Fact:  \"{item['target_fact']}\"")

        retrieved = pipeline.retrieve_chunks(q, top_k=2)
        print("Semantic Match Results:")
        for c_idx, c in enumerate(retrieved, 1):
            excerpt = c['text'].replace('\n', ' ')
            if len(excerpt) > 110:
                excerpt = excerpt[:110] + "..."
            print(f"  Rank {c_idx}: Similarity = {c['score']:.4f} | Excerpt: \"{excerpt}\"")

        res = pipeline.query(q, top_k=2)
        print(f"\nUltron Answer:\n{res['answer']}\n")

    print("=" * 70)
    print("RAG V2 TEST SUITE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
