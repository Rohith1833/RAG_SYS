import urllib.request
import json
import time

payload = json.dumps({
    "message": "According to the uploaded Rich Dad Poor Dad text, what is an asset or what does the rich dad teach?",
    "top_k": 3
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:5000/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print("HTTP Status:", resp.status)
    print("Total roundtrip ms:", round((time.time() - t0) * 1000, 2))
    print("Ollama Available:", res.get("ollama_available"))
    print("Generation Available:", res.get("generation_available"))
    print("Retrieval ms:", res.get("retrieval_ms"))
    print("Retrieved Count:", res.get("retrieved_count"))
    print("\nAnswer from Llama 3:\n", res.get("answer"))
    print("\nSources Used:")
    for s in res.get("sources", []):
        print(f"  - Source {s['source_index']}: {s['filename']} (Similarity: {s['score']}) [Page {s.get('page')}, Chunk {s.get('chunk_id')}]")
