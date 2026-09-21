import sys
from pathlib import Path

# Add project root and subdirectories to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))

from rag_pipeline import RAGPipeline


def main():
    print("=" * 70)
    print("ULTRON RAG V1 PIPELINE VERIFICATION TEST")
    print("=" * 70)

    # 1. Initialize RAG Pipeline
    pipeline = RAGPipeline(chunk_size=500, overlap=100)
    print("[1] Initialized RAGPipeline (chunk_size=500, overlap=100)")

    # 2. Check connection to Ollama backend
    try:
        pipeline.backend.check_connection()
        print(f"[2] Connection verified to local Ollama (Model: {pipeline.backend.model})")
    except ConnectionError as err:
        print(f"[Connection Error] {err}")
        sys.exit(1)

    # 3. Load sample notes document
    doc_path = PROJECT_ROOT / "rag" / "documents" / "ultron_notes.txt"
    num_chunks = pipeline.load_document(doc_path)
    print(f"[3] Document loaded from: {doc_path}")
    print(f"[3] Generated {num_chunks} chunk(s) in memory\n")

    # Display chunk summary
    for i, chunk in enumerate(pipeline.chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
    print("-" * 70)

    # 4. Questions to test
    questions = [
        # Known questions in the document
        "What is the parameter count of Ultron v0.2?",
        "What tokenizer does Ultron v0.2 use?",
        "What model is currently used as Ultron's practical backend?",
        "What is the context length of Ultron v0.2?",
        # Unknown question NOT in document (Step 8.9)
        "What operating system was used to train Ultron v0.2?"
    ]

    for idx, question in enumerate(questions, 1):
        is_unknown_test = (idx == len(questions))
        tag = "[GROUNDING TEST: UNKNOWN FACT]" if is_unknown_test else "[KNOWLEDGE RETRIEVAL TEST]"

        print("\n" + "=" * 70)
        print(f"QUESTION {idx}: {question}  {tag}")
        print("=" * 70)

        # Retrieve relevant chunks first to inspect retrieval stage
        retrieved = pipeline.retrieve_chunks(question, top_k=2)

        print(f"Retrieved {len(retrieved)} chunk(s):")
        if not retrieved:
            print("  (No keyword matches found above threshold)")
        else:
            for c_idx, c in enumerate(retrieved, 1):
                print(f"  [{c_idx}] Score: {c['score']}")
                preview = c['text'].replace('\n', ' ')
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                print(f"      Text excerpt: \"{preview}\"")

        # Execute full query through Ollama
        print("\nQuerying UltronBackend with context...")
        result = pipeline.query(question, top_k=2)

        print("\nUltron Answer:")
        print(result["answer"])
        print("-" * 70)

    print("\n" + "=" * 70)
    print("ALL RAG V1 TESTS EXECUTED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
