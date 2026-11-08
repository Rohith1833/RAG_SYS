import time
import json
import torch
import os
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "inference"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))

from ollama_client import OllamaClient
from rag_service import RAGService
from embedder import Embedder
from vector_store import VectorStore

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

def measure_system():
    ram_mb = 0.0
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        res = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if res:
            ram_mb = counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return ram_mb, vram_mb

def run_benchmarks():
    print("=" * 60)
    print("ULTRON OPTIMIZED PERFORMANCE BENCHMARK")
    print("=" * 60)

    # 1. Hardware baseline
    ram_mb, vram_mb = measure_system()
    print(f"Initial Process RAM: {ram_mb:.1f} MB | PyTorch VRAM Allocated: {vram_mb:.1f} MB")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    client = OllamaClient(model="llama3:latest", keep_alive="15m", num_ctx=4096)

    # 2. Connection Reuse Latency
    t0 = time.perf_counter()
    ok = client.check_connection()
    conn_ms = (time.perf_counter() - t0) * 1000
    print(f"Ollama Session Ping Latency: {conn_ms:.2f} ms")

    # 3. First-Token Latency (TTFT) and Streaming Generation Benchmark
    test_prompt_general = "Explain what a neural network is in exactly 3 concise sentences."
    print("\n--- Testing General Chat Streaming Generation ---")
    t_start = time.perf_counter()
    ttft_ms = None
    tokens = []
    
    for chunk in client.generate_stream(prompt=test_prompt_general, temperature=0.7):
        if ttft_ms is None and chunk.get("token"):
            ttft_ms = (time.perf_counter() - t_start) * 1000
        tokens.append(chunk.get("token", ""))

    gen_time_general = time.perf_counter() - t_start
    resp_general = "".join(tokens)
    approx_tokens = len(resp_general.split()) * 1.3
    tps_general = approx_tokens / gen_time_general if gen_time_general > 0 else 0

    print(f"Time to First Token (TTFT): {ttft_ms:.2f} ms")
    print(f"General Chat Total Time: {gen_time_general:.2f} s")
    print(f"Response Length: {len(resp_general)} chars ({len(resp_general.split())} words, ~{approx_tokens:.0f} tokens)")
    print(f"Estimated Tokens/sec: {tps_general:.1f} tok/s")
    print(f"Response Snippet: {resp_general[:100]}...")

    # 4. Embedding Generation Benchmark
    print("\n--- Testing Embedding Benchmark ---")
    t0 = time.perf_counter()
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    embedder_load_ms = (time.perf_counter() - t0) * 1000
    print(f"Embedder Model Load Time: {embedder_load_ms:.2f} ms")

    test_sentences = [
        "Ultron is an advanced local AI assistant running on Windows with PyTorch and CUDA.",
        "Sentence embeddings map semantically similar sentences to nearby points in vector space.",
        "Retrieval-Augmented Generation combines dense semantic search with local LLM generation.",
        "Quantized local models allow private offline inference without external cloud APIs."
    ] * 5  # 20 chunks

    t0 = time.perf_counter()
    _ = embedder.embed_text(test_sentences[0])
    single_embed_ms = (time.perf_counter() - t0) * 1000
    print(f"Single Chunk Embedding Time: {single_embed_ms:.2f} ms")

    t0 = time.perf_counter()
    _ = embedder.embed_texts(test_sentences)
    batch_embed_ms = (time.perf_counter() - t0) * 1000
    print(f"Batch Embedding Time (20 chunks): {batch_embed_ms:.2f} ms ({batch_embed_ms/len(test_sentences):.2f} ms/chunk)")

    # 5. Optimized Vector Store Search Benchmark
    print("\n--- Testing Optimized RAG Retrieval Benchmark ---")
    rag_service = RAGService(embedder=embedder)
    t0 = time.perf_counter()
    rag_results = rag_service.retriever.retrieve(
        query="What is the parameter count and architecture of Ultron v0.2?",
        top_k=3,
        hybrid=True
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    print(f"Optimized RAG Retrieval Time across {len(rag_service.vector_store)} chunks: {retrieval_ms:.2f} ms")
    print(f"Retrieved {len(rag_results)} chunks")

    # 6. Full End-to-End RAG Streaming Benchmark
    print("\n--- Testing Full End-to-End RAG Streaming Chat ---")
    t_start = time.perf_counter()
    rag_ttft_ms = None
    rag_tokens = []
    rag_retrieval_time = 0.0

    for ev in rag_service.chat_stream(
        message="What is the parameter count and architecture of Ultron v0.2?",
        mode="rag",
        top_k=3
    ):
        if ev.get("type") == "meta":
            rag_retrieval_time = ev.get("retrieval_ms", 0.0)
        elif ev.get("type") == "token":
            if rag_ttft_ms is None and ev.get("token"):
                rag_ttft_ms = (time.perf_counter() - t_start) * 1000
            rag_tokens.append(ev.get("token", ""))

    rag_total_time = time.perf_counter() - t_start
    rag_full_answer = "".join(rag_tokens)
    approx_rag_tokens = len(rag_full_answer.split()) * 1.3
    tps_rag = approx_rag_tokens / rag_total_time if rag_total_time > 0 else 0

    print(f"RAG Time to First Token (TTFT): {rag_ttft_ms:.2f} ms")
    print(f"Full RAG Chat Total Time: {rag_total_time:.2f} s")
    print(f"Retrieval portion: {rag_retrieval_time:.2f} ms")
    print(f"Estimated RAG Tokens/sec: {tps_rag:.1f} tok/s")
    print(f"RAG Answer Snippet: {rag_full_answer[:120]}...")

    # Final Resource snapshot
    ram_mb, vram_mb = measure_system()
    print("\n--- Resource Utilization ---")
    print(f"Final Process RAM: {ram_mb:.1f} MB | PyTorch VRAM Allocated: {vram_mb:.1f} MB")

    results = {
        "conn_ms": conn_ms,
        "general_ttft_ms": ttft_ms,
        "general_total_s": gen_time_general,
        "general_tps": tps_general,
        "embedder_load_ms": embedder_load_ms,
        "single_embed_ms": single_embed_ms,
        "batch_embed_ms": batch_embed_ms,
        "retrieval_ms": retrieval_ms,
        "rag_ttft_ms": rag_ttft_ms,
        "rag_total_s": rag_total_time,
        "rag_tps": tps_rag,
        "process_ram_mb": ram_mb,
        "pytorch_vram_mb": vram_mb
    }
    with open(PROJECT_ROOT / "scratch" / "optimized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nOptimized results saved to scratch/optimized_results.json")

if __name__ == "__main__":
    run_benchmarks()

