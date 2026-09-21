import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "models"))


from tokenizer import CharacterTokenizer
from ultron import UltronModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "ultron_checkpoint.pt"
)

BLOCK_SIZE = 8


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# --------------------------------------------------
# Recreate tokenizer
# --------------------------------------------------

characters = checkpoint["characters"]

vocab_size = checkpoint["vocab_size"]

tokenizer = CharacterTokenizer(
    "".join(characters)
)


# --------------------------------------------------
# Recreate model
# --------------------------------------------------

model = UltronModel(
    vocab_size=vocab_size,
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"]
).to(device)


# Load learned weights.
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Tell PyTorch we're doing inference,
# not training.
model.eval()


# --------------------------------------------------
# Text generation
# --------------------------------------------------

def generate(prompt, max_new_tokens=100, temperature=1.0):

    # Convert prompt into token IDs.
    token_ids = tokenizer.encode(prompt)

    # Convert to a PyTorch tensor.
    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Only keep the most recent tokens
            # that fit inside the model's context.
            context = tokens[:, -BLOCK_SIZE:]

            # Ask Ultron for predictions.
            logits = model(context)

            # We only care about the final position.
            logits = logits[:, -1, :]

            # Temperature controls randomness.
            logits = logits / temperature

            # Convert logits into probabilities.
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Randomly sample the next token
            # according to those probabilities.
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add the new token to our sequence.
            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

    # Convert token IDs back into text.
    return tokenizer.decode(
        tokens[0].tolist()
    )


# --------------------------------------------------
# Run generation
# --------------------------------------------------

prompt = "Ultron"

result = generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8
)

print("\nPrompt:")
print(prompt)

print("\nUltron generated:")
print(result)