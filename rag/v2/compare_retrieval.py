import sys
from pathlib import Path

# Add paths for clean import of both RAG V1 and RAG V2
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))

from rag_pipeline import RAGPipeline as RAGPipelineV1
from rag_pipeline_v2 import RAGPipelineV2


def main():
    print("=" * 70)
    print("EDUCATIONAL COMPARISON: RAG V1 (KEYWORD) VS RAG V2 (SEMANTIC)")
    print("=" * 70)

    doc_path = PROJECT_ROOT / "rag" / "documents" / "ultron_notes.txt"

    # Initialize V1 Pipeline
    v1_pipeline = RAGPipelineV1(chunk_size=500, overlap=100)
    v1_pipeline.load_document(doc_path)

    # Initialize V2 Pipeline
    v2_pipeline = RAGPipelineV2(chunk_size=500, overlap=100)
    v2_pipeline.load_document(doc_path)

    comparison_questions = [
        # 1. Direct match: both keyword and semantic should score well
        "What is the parameter count of Ultron v0.2?",

        # 2. Paraphrased query: Synonymous concepts with different vocabulary
        "What method does the second model use to break text into tokens?",

        # 3. Conceptual query: "language model" instead of "backend", "assistant" instead of "Ultron"
        "Which language model is currently running behind the assistant?",

        # 4. Keyword query with standard vocabulary
        "How long is the context window of Ultron v0.2?"
    ]

    for index, q in enumerate(comparison_questions, 1):
        print("\n" + "=" * 70)
        print(f"[{index}] QUESTION: \"{q}\"")
        print("=" * 70)

        # RAG V1: Keyword Retrieval
        v1_results = v1_pipeline.retrieve_chunks(q, top_k=2)
        print("\n--- [RAG V1: Keyword Matching (Token Frequency)] ---")
        if not v1_results:
            print("  Score: 0 (No matching keywords found)")
        else:
            for rank, item in enumerate(v1_results, 1):
                clean_txt = item['text'].replace('\n', ' ')
                if len(clean_txt) > 110:
                    clean_txt = clean_txt[:110] + "..."
                print(f"  Rank {rank} | Score: {item['score']} match(es)")
                print(f"         Excerpt: \"{clean_txt}\"")

        # RAG V2: Semantic Dense Retrieval
        v2_results = v2_pipeline.retrieve_chunks(q, top_k=2)
        print("\n--- [RAG V2: Dense Semantic Embeddings (Cosine Similarity)] ---")
        if not v2_results:
            print("  Similarity: N/A")
        else:
            for rank, item in enumerate(v2_results, 1):
                clean_txt = item['text'].replace('\n', ' ')
                if len(clean_txt) > 110:
                    clean_txt = clean_txt[:110] + "..."
                print(f"  Rank {rank} | Similarity: {item['score']:.4f}")
                print(f"         Excerpt: \"{clean_txt}\"")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS & OBSERVATIONS:")
    print("1. Direct keyword queries score strongly in both V1 and V2.")
    print("2. Paraphrased queries (e.g. 'break text into tokens' vs 'tokenizer')")
    print("   maintain high semantic similarity in V2 (~0.45-0.70) while V1 relies")
    print("   solely on shared literal tokens.")
    print("3. Dense embeddings map conceptual similarity rather than exact spelling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
