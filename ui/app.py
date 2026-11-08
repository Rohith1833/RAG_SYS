import json
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# --------------------------------------------------
# Paths & Module Import Setup
# --------------------------------------------------
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent

sys.path.append(str(PROJECT_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "rag"))
sys.path.append(str(PROJECT_ROOT / "rag" / "v2"))
sys.path.append(str(PROJECT_ROOT / "rag" / "ingestion"))

from ultron_v2 import UltronV2
from rag_service import RAGService
from database import Database

app = Flask(
    __name__,
    template_folder=str(UI_DIR / "templates"),
    static_folder=str(UI_DIR / "static")
)

# Enforce 50 MB max file upload size
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Initialize persistent SQLite database
db = Database()

# Initialize RAG Service once at server startup
rag_service = None
try:
    rag_service = RAGService()
    print("[Ultron RAG] RAG Service initialized successfully.")
except Exception as _rag_err:
    print(f"[Ultron RAG Warning] Could not initialize RAG Service: {_rag_err}")


# --------------------------------------------------
# Ultron v0.2 Model Configuration & Device Setup (Preserved Baseline)
# --------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ultron_v2_checkpoint.pt"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "real" / "tokenizer.json"

# Automatic CUDA / GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
device_label = "GPU" if device.type == "cuda" else "CPU"
device_display = f"{device_label} ({device_name})"

model = None
tokenizer = None
checkpoint_data = None
model_error = None
total_parameters = 1323728
BLOCK_SIZE = 128
EOS_TOKEN_ID = None
BOS_TOKEN_ID = None


def load_model():
    """Load the trained v0.2 checkpoint and BPE tokenizer into UltronV2."""
    global model, tokenizer, checkpoint_data, model_error, total_parameters, BLOCK_SIZE, EOS_TOKEN_ID, BOS_TOKEN_ID
    try:
        print(f"\n[Ultron v0.2] Initializing backend...")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {device_name}")

        if not TOKENIZER_PATH.exists():
            model_error = f"Tokenizer file not found at: {TOKENIZER_PATH}"
            print(f"[Ultron Error] {model_error}")
            return

        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        EOS_TOKEN_ID = tokenizer.token_to_id("<eos>")
        BOS_TOKEN_ID = tokenizer.token_to_id("<bos>")

        if not CHECKPOINT_PATH.exists():
            model_error = f"Checkpoint not found at: {CHECKPOINT_PATH}"
            print(f"[Ultron Error] {model_error}")
            return

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        checkpoint_data = checkpoint

        vocab_size = checkpoint.get("vocab_size", tokenizer.get_vocab_size())
        BLOCK_SIZE = checkpoint.get("block_size", 128)
        embedding_dim = checkpoint.get("embedding_dim", 128)
        num_heads = checkpoint.get("num_heads", 4)
        num_layers = checkpoint.get("num_layers", 4)

        m = UltronV2(
            vocab_size=vocab_size,
            block_size=BLOCK_SIZE,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)

        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()

        total_parameters = checkpoint.get("parameters", sum(p.numel() for p in m.parameters()))
        model = m
        model_error = None

        print(f"Ultron v0.2 loaded successfully.")
        print(f"Parameters: {total_parameters:,}")
        print(f"Vocabulary: {vocab_size:,}")
        print(f"Context: {BLOCK_SIZE}")
        print(f"Model parameters on device: {next(model.parameters()).device}")

    except Exception as e:
        model_error = str(e)
        print(f"[Ultron Error] Failed to load v0.2 model: {e}")


# Initialize v0.2 model on startup
load_model()


def generate_text(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
    """Generate text using the loaded Ultron v0.2 PyTorch model."""
    if model is None or tokenizer is None:
        raise RuntimeError(f"Ultron v0.2 model not initialized: {model_error or 'Unknown error'}")

    clean_prompt = prompt.strip() if prompt else "Once upon a time"
    if not clean_prompt:
        clean_prompt = "Once upon a time"

    encoded = tokenizer.encode(clean_prompt)
    prompt_ids = encoded.ids
    if len(prompt_ids) == 0:
        prompt_ids = [BOS_TOKEN_ID] if BOS_TOKEN_ID is not None else [1]

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    temp = max(float(temperature), 0.01)
    max_tokens = min(max(int(max_new_tokens), 1), 500)

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_tokens):
            context = token_ids[:, -BLOCK_SIZE:]
            logits = model(context)
            logits = logits[:, -1, :]
            logits = logits / temp
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if EOS_TOKEN_ID is not None and next_token.item() == EOS_TOKEN_ID:
                break

            token_ids = torch.cat([token_ids, next_token], dim=1)

    elapsed = time.perf_counter() - start_time
    tokens_generated = token_ids.shape[1] - len(prompt_ids)

    # Decode without raw special token markers (<bos>, <eos>, <unk>)
    full_text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)

    return full_text, clean_prompt, elapsed, tokens_generated


# --------------------------------------------------
# Web Application & API Routes
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def api_info():
    """Ultron v0.2 PyTorch model telemetry."""
    is_ready = model is not None
    model_param_device = str(next(model.parameters()).device) if is_ready else str(device)
    vocab_size = tokenizer.get_vocab_size() if tokenizer else 2000
    architecture_desc = (
        f"{checkpoint_data.get('num_layers', 4)} Layers • {checkpoint_data.get('num_heads', 4)} Heads • {checkpoint_data.get('embedding_dim', 128)} Dim"
        if checkpoint_data else "4 Layers • 4 Heads • 128 Dim"
    )

    return jsonify({
        "status": "online" if is_ready else "error",
        "error": model_error,
        "model": "Ultron",
        "version": "v0.2",
        "vocab_size": vocab_size,
        "context": BLOCK_SIZE,
        "parameters": total_parameters,
        "device": device_display,
        "device_type": device.type,
        "model_device": model_param_device,
        "training_dataset": "TinyStories",
        "architecture": architecture_desc
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Ultron v0.2 PyTorch inference endpoint."""
    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.8)
    max_new_tokens = data.get("max_new_tokens", 100)

    try:
        temperature = float(temperature)
    except (ValueError, TypeError):
        temperature = 0.8

    try:
        max_new_tokens = int(max_new_tokens)
    except (ValueError, TypeError):
        max_new_tokens = 100

    try:
        generated_text, clean_prompt, elapsed_sec, tokens_generated = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        tokens_per_sec = round(tokens_generated / max(elapsed_sec, 0.001), 1)

        return jsonify({
            "text": generated_text,
            "prompt": prompt,
            "sanitized_prompt": clean_prompt,
            "tokens_generated": tokens_generated,
            "latency_ms": round(elapsed_sec * 1000, 1),
            "tokens_per_sec": tokens_per_sec,
            "device": device_display,
            "model_device": str(next(model.parameters()).device) if model else "none"
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Generation error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """System, sub-system, and Ollama health telemetry."""
    if rag_service is None:
        return jsonify({
            "status": "error",
            "error": "RAG Service is not initialized on the server.",
            "ollama": False,
            "generation_ready": False,
            "embedding_ready": False,
            "ingestion_ready": False,
            "retrieval_ready": False,
            "device": str(device)
        }), 500

    try:
        health_info = rag_service.get_health()
        return jsonify(health_info)
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# --------------------------------------------------
# Conversation Management Endpoints
# --------------------------------------------------
@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    """List all saved conversations ordered by recency."""
    try:
        convs = db.list_conversations()
        return jsonify({"conversations": convs})
    except Exception as err:
        return jsonify({"error": f"Failed to list conversations: {str(err)}"}), 500


@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    """Create a new conversation session."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Conversation")
    mode = data.get("mode", "rag")
    try:
        conv = db.create_conversation(title=title, mode=mode)
        return jsonify(conv), 201
    except Exception as err:
        return jsonify({"error": f"Failed to create conversation: {str(err)}"}), 500


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def api_get_conversation(conv_id):
    """Retrieve conversation details and chronological message history."""
    try:
        conv = db.get_conversation(conv_id)
        if not conv:
            return jsonify({"error": f"Conversation '{conv_id}' not found."}), 404
        messages = db.get_messages(conv_id)
        conv["messages"] = messages
        return jsonify(conv), 200
    except Exception as err:
        return jsonify({"error": f"Failed to load conversation: {str(err)}"}), 500


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    """Delete a conversation and its messages."""
    try:
        deleted = db.delete_conversation(conv_id)
        if not deleted:
            return jsonify({"error": f"Conversation '{conv_id}' not found."}), 404
        return jsonify({"status": "success", "message": f"Conversation '{conv_id}' deleted."}), 200
    except Exception as err:
        return jsonify({"error": f"Failed to delete conversation: {str(err)}"}), 500


# --------------------------------------------------
# Chat & Document RAG Endpoints
# --------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Unified dual-mode chat endpoint:
    - mode='general': Unconstrained chat with llama3:latest (no document retrieval).
    - mode='rag': Grounded document question-answering with semantic & hybrid retrieval.
    Persists user & assistant messages to SQLite if conversation_id is provided.
    """
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message") or data.get("prompt") or ""
    mode = data.get("mode", "rag")
    conv_id = data.get("conversation_id")
    top_k = data.get("top_k", 3)
    min_score = data.get("min_score", 0.20)

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 3

    if not message.strip():
        return jsonify({"error": "Question message cannot be empty."}), 400

    # Ensure conversation exists if conv_id provided
    history = []
    if conv_id:
        existing_conv = db.get_conversation(conv_id)
        if not existing_conv:
            # Create on the fly
            title = message.strip()[:40] + ("..." if len(message.strip()) > 40 else "")
            db.create_conversation(title=title, mode=mode)
        else:
            # Auto-title conversation from first user message if it has default title
            if existing_conv.get("title") == "New Conversation":
                title = message.strip()[:40] + ("..." if len(message.strip()) > 40 else "")
                db.update_conversation(conv_id, title=title)

            # Load history turns for conversation-aware context
            raw_msgs = db.get_messages(conv_id)
            history = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

        # Record user message in DB
        db.add_message(conv_id, "user", message.strip())

    try:
        # Call RAGService dual-mode chat
        result = rag_service.chat(
            message=message,
            mode=mode,
            conversation_history=history,
            top_k=top_k,
            min_score=min_score
        )

        # Record assistant reply in DB if in conversation session
        if conv_id:
            db.add_message(
                conversation_id=conv_id,
                role="assistant",
                content=result.get("answer", ""),
                sources=result.get("sources"),
                metadata={
                    "latency_ms": result.get("latency_ms"),
                    "mode": mode,
                    "retrieved_count": result.get("retrieved_count")
                }
            )

        result["conversation_id"] = conv_id
        return jsonify(result), 200

    except ConnectionError as ce:
        return jsonify({
            "error": f"Ollama connection error: {str(ce)}. Please ensure 'ollama serve' is running."
        }), 503
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Answering error: {str(e)}"}), 500


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """
    Server-Sent Events (SSE) streaming chat endpoint.
    Streams tokens in real time from Llama 3 via Ollama.
    Persists final user & assistant messages to SQLite.
    """
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message") or data.get("prompt") or ""
    mode = data.get("mode", "rag")
    conv_id = data.get("conversation_id")
    top_k = data.get("top_k", 3)
    min_score = data.get("min_score", 0.20)

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 3

    if not message.strip():
        return jsonify({"error": "Question message cannot be empty."}), 400

    # Ensure conversation and record user message
    history = []
    if conv_id:
        existing_conv = db.get_conversation(conv_id)
        if not existing_conv:
            title = message.strip()[:40] + ("..." if len(message.strip()) > 40 else "")
            db.create_conversation(title=title, mode=mode)
        else:
            if existing_conv.get("title") == "New Conversation":
                title = message.strip()[:40] + ("..." if len(message.strip()) > 40 else "")
                db.update_conversation(conv_id, title=title)
            raw_msgs = db.get_messages(conv_id)
            history = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

        db.add_message(conv_id, "user", message.strip())

    def event_stream():
        full_answer = ""
        sources = []
        try:
            for ev in rag_service.chat_stream(
                message=message,
                mode=mode,
                conversation_history=history,
                top_k=top_k,
                min_score=min_score
            ):
                if ev.get("type") == "meta":
                    sources = ev.get("sources", [])
                elif ev.get("type") == "token":
                    full_answer += ev.get("token", "")
                elif ev.get("type") == "done":
                    full_answer = ev.get("full_answer", full_answer)
                    if conv_id:
                        db.add_message(
                            conversation_id=conv_id,
                            role="assistant",
                            content=full_answer,
                            sources=sources,
                            metadata={"mode": mode, "latency_ms": ev.get("total_ms")}
                        )
                ev["conversation_id"] = conv_id
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as ex:
            err_payload = {"type": "error", "error": str(ex), "conversation_id": conv_id}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/model/unload", methods=["POST"])
def api_unload_model():
    """Unload Llama 3 model from GPU VRAM on demand."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500
    try:
        success = rag_service.unload_model()
        return jsonify({
            "status": "success" if success else "failed",
            "message": "Model unloaded from GPU memory." if success else "Could not unload model."
        }), 200
    except Exception as err:
        return jsonify({"error": f"Failed to unload model: {str(err)}"}), 500


# --------------------------------------------------
# Document Ingestion & Management Endpoints
# --------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and index a document into the local knowledge base."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part provided in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    filename = file.filename
    ext = Path(filename).suffix.lower()

    if ext not in {".txt", ".md", ".pdf", ".docx"}:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Please upload .txt, .md, .pdf, or .docx files."
        }), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": f"File '{filename}' is empty (0 bytes)."}), 400

        result = rag_service.upload_document(filename=filename, file_bytes=file_bytes)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to ingest document: {str(e)}"}), 500


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """List all indexed documents in the local knowledge base."""
    if rag_service is None:
        return jsonify({"documents": []})

    try:
        docs = rag_service.list_documents()
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/<document_id>", methods=["DELETE"])
def api_delete_document(document_id):
    """Delete a document and purge its vectors from the local store."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    try:
        result = rag_service.delete_document(document_id)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/<document_id>/summarize", methods=["POST"])
def api_summarize_document(document_id):
    """Generate a complete long-document summary."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    data = request.get_json(silent=True) or {}
    style = data.get("style", "detailed")

    try:
        summary_result = rag_service.summarize_document(document_id=document_id, style=style)
        return jsonify(summary_result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except ConnectionError:
        return jsonify({
            "error": "Ollama is currently offline. Complete document summarization requires Ollama (llama3:latest)."
        }), 503
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500


@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    """Direct semantic retrieval test endpoint."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    min_score = data.get("min_score")

    if not query.strip():
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        results = rag_service.direct_retrieve(query=query, top_k=int(top_k), min_score=min_score)
        return jsonify({"query": query, "results": results, "count": len(results)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/index/clear", methods=["POST"])
def api_clear_index():
    """Clear all indexed documents and reset the vector store."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    try:
        result = rag_service.clear_index()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Settings Endpoints
# --------------------------------------------------
@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    """Retrieve saved application settings."""
    try:
        settings = db.get_all_settings()
        # Defaults
        defaults = {
            "model": "llama3:latest",
            "temperature": "0.7",
            "top_k": "3",
            "default_mode": "rag",
            "similarity_threshold": "0.20"
        }
        defaults.update(settings)
        return jsonify(defaults), 200
    except Exception as err:
        return jsonify({"error": str(err)}), 500


@app.route("/api/settings", methods=["POST"])
def api_set_settings():
    """Update application settings."""
    data = request.get_json(silent=True) or {}
    try:
        for k, v in data.items():
            db.set_setting(k, str(v))
        return jsonify({"status": "success", "message": "Settings updated."}), 200
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
