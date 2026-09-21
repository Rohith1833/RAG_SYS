import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# --------------------------------------------------
# Paths & Module Import Setup
# --------------------------------------------------
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent

# Append data and models directories without modifying existing code
sys.path.append(str(PROJECT_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT / "backend"))

from ultron_v2 import UltronV2
from rag_service import RAGService

app = Flask(
    __name__,
    template_folder=str(UI_DIR / "templates"),
    static_folder=str(UI_DIR / "static")
)

# Initialize RAG Service once at server startup
rag_service = None
try:
    rag_service = RAGService()
    print("[Ultron RAG] RAG Service initialized successfully.")
except Exception as _rag_err:
    print(f"[Ultron RAG Warning] Could not initialize RAG Service: {_rag_err}")


# --------------------------------------------------
# Model Configuration & Device Setup
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


# Initialize model on startup
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
# Routes
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def api_info():
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


# --------------------------------------------------
# Step 10: Local RAG Assistant API Endpoints
# --------------------------------------------------
@app.route("/api/health", methods=["GET"])
def api_health():
    """System and local RAG health telemetry."""
    if rag_service is None:
        return jsonify({
            "status": "error",
            "error": "RAG Service is not initialized on the server.",
            "ollama": False,
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


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and index a document into the local RAG knowledge base."""
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part provided in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    filename = file.filename
    ext = Path(filename).suffix.lower()

    if ext not in {".txt", ".md", ".pdf"}:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Please upload .txt, .md, or .pdf files."
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


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Perform semantic retrieval and generate grounded response with Llama 3.
    """
    if rag_service is None:
        return jsonify({"error": "RAG Service is not initialized."}), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message") or data.get("prompt") or ""
    top_k = data.get("top_k", 3)

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 3

    if not message.strip():
        return jsonify({"error": "Question message cannot be empty."}), 400

    try:
        result = rag_service.ask_question(message=message, top_k=top_k)
        return jsonify(result), 200
    except ConnectionError as ce:
        return jsonify({
            "error": f"Ollama connection error: {str(ce)}. Please ensure 'ollama serve' is running."
        }), 503
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"RAG answering error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

