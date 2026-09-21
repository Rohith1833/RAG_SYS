# Step 10 Walkthrough: End-to-End Local RAG Assistant Integration

## 1. Overview & What Changed in Step 10
In Step 10, we unified all preceding components into a complete, usable **Local AI Assistant**:
- **Ultron v0.1 & v0.2**: Completely preserved in their frozen baseline states.
- **RAG V1 & V2**: Keyword and dense semantic retrieval foundations preserved under `rag/` and `rag/v2/`.
- **New Ingestion Layer**: Multi-format document parser (`rag/ingestion/document_parser.py`) supporting `.txt`, `.md`, and page-by-page `.pdf` text extraction via lightweight `pypdf`, backed by a document registry (`rag/ingestion/document_manager.py`).
- **Vector Store Persistence**: In-memory NumPy vector store extended with `.npy` binary vector and `.json` metadata persistence (`rag/v2/vector_store.py`).
- **RAG Service**: High-level orchestration service (`backend/rag_service.py`) bridging ingestion, embedding generation, vector search, prompt grounding, and Ollama inference.
- **Enhanced Claymorphism UI**: Extended `ui/templates/index.html`, `ui/static/style.css`, and `ui/static/app.js` with a dual-mode interface featuring a local RAG workspace with drag-and-drop file upload, staged progress states, document registry, grounded chat transcript, and source/similarity score inspection cards.

---

## 2. Text Architecture Diagram

```
                              ┌───────────────────────────┐
                              │         USER / UI         │
                              │ (Claymorphism 3D Canvas)  │
                              └─────────────┬─────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          [1. Upload Document]                            [2. Ask Question]
          (.txt, .md, .pdf)                               ("What is Ultron?")
                     │                                             │
                     ▼                                             ▼
           POST /api/upload                                  POST /api/chat
                     │                                             │
  ┌──────────────────┴──────────────────┐                          │
  ▼                                     ▼                          ▼
[DocumentParser]                [DocumentManager]          [Embedder] (CUDA)
  Page-by-page text               rag/uploads/               all-MiniLM-L6-v2 (384-d)
  extraction                      rag/index/docs.json              │
         │                                                         ▼
         ▼                                                 [SemanticRetriever]
   [TextChunker]                                             Cosine Similarity Search
  500 chars / 100 overlap                                          │
         │                                                         ▼
         ▼                                                 [Top-k Scored Chunks]
     [Embedder]                                            Score, File, Page, Text
  Dense vectors (384-d)                                            │
         │                                                         ▼
         ▼                                                 [Grounding Prompt]
   [VectorStore]                                           Strict Context Injection
  NumPy Matrix + Disk                                              │
  (embeddings.npy & metadata.json)                                 ▼
                                                           [UltronBackend]
                                                                   │
                                                                   ▼
                                                            [OllamaClient]
                                                                   │
                                                                   ▼
                                                            [Local Ollama]
                                                           (llama3:latest)
                                                                   │
                                                                   ▼
                                                            [Grounded Answer]
                                                                   │
                                                                   ▼
                                                          [Render in UI with]
                                                          [Sources & Scores]
```

---

## 3. How the Ingestion Pipeline Works

### A. Document Upload & Storage
1. When a document is dropped or selected in the UI, JavaScript validates that the file has a `.txt`, `.md`, or `.pdf` extension.
2. The browser sends a `multipart/form-data` request to `POST /api/upload`.
3. The server secures the filename via `werkzeug.utils.secure_filename` and assigns a unique `document_id` (e.g. `doc_c01680c817`).
4. The file is saved to `rag/uploads/`.

### B. Parsing (`rag/ingestion/document_parser.py`)
- **TXT & Markdown**: Loaded directly with UTF-8 encoding.
- **PDF**: Parsed page-by-page using lightweight `pypdf.PdfReader`. Each extracted page segment retains its 1-indexed page number (`{"page": 1, "text": "..."}`). Empty, encrypted, or image-only PDFs are rejected cleanly with helpful user messages.

### C. Chunking (`rag/chunker.py`)
- Standard 500-character chunk window with a 100-character sliding overlap.
- Overlap ensures sentences and facts spanning chunk boundaries are not truncated or lost.
- Every individual chunk retains its document lineage: `document_id`, `filename`, `chunk_id`, `page` number, and character count.

### D. Dense Semantic Embeddings (`rag/v2/embedder.py`)
- Raw text chunks are passed in batch to `Embedder.embed_texts()`.
- The `all-MiniLM-L6-v2` SentenceTransformer generates 384-dimensional dense vectors on `cuda` (NVIDIA RTX 3050 Laptop GPU).
- Embeddings are L2-normalized ($||v|| = 1.0$) upon output.

### E. Vector Store & Local Persistence (`rag/v2/vector_store.py`)
- Chunks, vectors, and metadata are stacked into the in-memory NumPy matrix.
- The vector store persists automatically across server restarts by saving:
  - `rag/index/embeddings.npy`: Binary NumPy matrix storing all chunk vectors.
  - `rag/index/metadata.json`: JSON file storing text passages and rich chunk metadata.
  - `rag/index/documents.json`: Document registry tracking document IDs, filenames, and chunk counts.

---

## 4. How Semantic Retrieval & Grounded Generation Work

### A. Question Embedding
- The user's query is converted into a 384-dimensional embedding vector via `Embedder.embed_text(query)`.

### B. Cosine Similarity Vector Search
- Exact cosine similarity is calculated between the query vector $q$ and all document chunk vectors $V$:
  $$\text{Cosine Similarity} = \frac{q \cdot V}{||q|| \cdot ||V||}$$
- Since both vectors are normalized, this simplifies to a fast matrix-vector dot product.
- Chunks are ranked in descending order of similarity, and the top-$k$ (default: $k=3$) passages are selected.

### C. Context Injection & Anti-Hallucination Prompting
The retrieved passages are injected into a strict prompt:
```text
You are Ultron, a helpful and precise local AI assistant.

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided retrieved context below.
2. Do NOT fabricate, extrapolate, or invent facts not present in the context.
3. If the context does NOT contain enough information to answer the question,
   explicitly state that the provided documents do not contain the answer.

RETRIEVED CONTEXT:
[Source 1: project_notes.pdf, Page 2 | Cosine Similarity: 0.7241]
...

USER QUESTION:
What is the parameter count of Ultron v0.2?

GROUNDED ANSWER:
```

### D. LLM Reasoning
- The prompt is dispatched through `UltronBackend` -> `OllamaClient` -> local Ollama HTTP API (`http://127.0.0.1:11434/api/generate`) with model `llama3:latest`.
- Llama 3 synthesizes an answer directly citing the provided passages.

---

## 5. Critical Distinction: RAG vs. Fine-Tuning

| Dimension | Retrieval-Augmented Generation (RAG) | Fine-Tuning |
| :--- | :--- | :--- |
| **Model Weights** | **100% Unchanged**. Weights are frozen. | **Modified**. Gradients update internal weight tensors. |
| **Where Knowledge Lives** | In the external local vector store (`rag/index/`). | Inside the model's neural network parameter matrices. |
| **Updating Knowledge** | Instantaneous. Add or delete document chunks. | Slow & expensive. Requires retraining runs, loss calculation, and new checkpoints. |
| **Hallucination Control** | High. Model is grounded directly in retrieved text passages. | Moderate. Models can still hallucinate memorized facts. |
| **Hardware Overhead** | Minimal. Small embedding model + vector search. | High. Requires extensive VRAM for gradient calculation and optimizer states. |

> [!NOTE]
> The UI explicitly clarifies this distinction: Uploading a document teaches Ultron through local RAG indexing; it never modifies the underlying Llama 3 weights.

---

## 6. How the UI & Flask Server Communicate

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `GET /` | HTML | Serves the dual-mode Ultron Claymorphism interface. |
| `GET /api/health` | JSON | Returns system telemetry: Ollama connection, active model, embedding model, dimension, compute device, document count, and chunk count. |
| `POST /api/upload` | Multipart | Accepts `.txt`, `.md`, or `.pdf`, parses, chunks, embeds, indexes, and returns the registered document record. |
| `GET /api/documents` | JSON | Returns list of all indexed documents with chunk counts and timestamps. |
| `POST /api/chat` | JSON | Accepts `{"message": "..."}`, performs vector search, queries Llama 3, and returns grounded answer with detailed sources list. |
| `DELETE /api/documents/<id>`| JSON | Deletes document, deletes physical file, and purges its vectors from the vector store. |
| `POST /api/index/clear` | JSON | Clears all documents and resets the vector store on disk and in memory. |
| `POST /api/generate` | JSON | [Preserved] Direct inference through the local PyTorch Ultron v0.2 model. |
| `GET /api/info` | JSON | [Preserved] Architecture specification for the PyTorch Ultron v0.2 model. |

---

## 7. How to Run & Verify the Application

### 1. Run Automated Test Suites
```powershell
# Ingestion tests (TXT, MD, PDF parsing, empty files, unsupported types)
.\.venv\Scripts\python.exe rag\ingestion\test_document_parser.py

# RAG Service lifecycle tests (health, upload, query, unknown question, persistence, deletion)
.\.venv\Scripts\python.exe backend\test_rag_service.py

# Regression tests
.\.venv\Scripts\python.exe rag\test_rag.py
.\.venv\Scripts\python.exe rag\v2\test_rag_v2.py
.\.venv\Scripts\python.exe rag\v2\compare_retrieval.py
.\.venv\Scripts\python.exe backend\test_ultron_backend.py
.\.venv\Scripts\python.exe inference\test_ollama_client.py
```

### 2. Start the Server
```powershell
.\.venv\Scripts\python.exe ui\app.py
```

### 3. Open the UI
Open your browser at:
```
http://127.0.0.1:5000
```
- **Local RAG Assistant**: Upload notes or PDFs, ask questions, inspect sources and similarity scores.
- **v0.2 TinyStories Lab**: Switch tabs to test the raw PyTorch v0.2 TinyStories generator.
