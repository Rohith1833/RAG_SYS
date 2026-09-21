import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "models")
)

from ultron_v2 import UltronV2


# --------------------------------------------------
# Paths
# --------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "ultron_v2_checkpoint.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tokenizer.json"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading BPE tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

VOCAB_SIZE = tokenizer.get_vocab_size()
EOS_TOKEN_ID = tokenizer.token_to_id("<eos>")
BOS_TOKEN_ID = tokenizer.token_to_id("<bos>")
UNK_TOKEN_ID = tokenizer.token_to_id("<unk>")

print(
    "Vocabulary size:",
    VOCAB_SIZE
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

print("\nLoading Ultron v0.2 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

BLOCK_SIZE = checkpoint["block_size"]

print(
    "Checkpoint vocabulary:",
    checkpoint["vocab_size"]
)

print(
    "Checkpoint context:",
    BLOCK_SIZE
)

print(
    "Checkpoint parameters:",
    checkpoint["parameters"]
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = UltronV2(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"]
).to(device)


# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nUltron v0.2 loaded successfully.")


# --------------------------------------------------
# Generation function
# --------------------------------------------------

def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    stop_on_eos=True,
    skip_special_tokens=True
):
    """
    Generate text using the trained Ultron v0.2 model.
    Robustly handles temperature, EOS stopping, context limits, and special tokens.
    """
    if not prompt or not prompt.strip():
        prompt = "Once upon a time"

    # Clamp temperature safely to avoid zero division or infinite logits
    temperature = max(float(temperature), 0.01)

    # Bound max_new_tokens safely
    max_new_tokens = min(max(int(max_new_tokens), 1), 1024)

    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids

    if len(prompt_ids) == 0:
        prompt_ids = [BOS_TOKEN_ID] if BOS_TOKEN_ID is not None else [0]

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only keep the most recent tokens that fit within BLOCK_SIZE context
            context = token_ids[:, -BLOCK_SIZE:]

            logits = model(context)

            # Focus only on prediction for the last position
            logits = logits[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Softmax to obtain sampling distribution
            probabilities = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Early stop if EOS token was predicted
            if stop_on_eos and EOS_TOKEN_ID is not None:
                if next_token.item() == EOS_TOKEN_ID:
                    break

            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1
            )

    generated_ids = token_ids[0].tolist()

    # Decode while removing raw special token markers like <bos>, <eos>, <unk>
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=skip_special_tokens
    )

    return generated_text


# --------------------------------------------------
# Test generation
# --------------------------------------------------

if __name__ == "__main__":
    prompt = "Once upon a time"

    print("\nPrompt:")
    print(prompt)

    print("\nGenerating...\n")

    result = generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.8
    )

    print(result)