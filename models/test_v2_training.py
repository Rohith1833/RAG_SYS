import sys
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "models")
)

from ultron_v2 import UltronV2


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 2000
BLOCK_SIZE = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4

BATCH_SIZE = 4


# --------------------------------------------------
# Paths
# --------------------------------------------------

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tokenizer.json"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "train_tokens.pt"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

print(
    "Tokenizer vocabulary:",
    tokenizer.get_vocab_size()
)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("\nLoading training data...")

data = torch.load(
    TRAIN_DATA_PATH,
    weights_only=True
)

print(
    "Training tokens:",
    len(data)
)


# --------------------------------------------------
# Create a batch
# --------------------------------------------------

def get_batch(data):
    starts = torch.randint(
        0,
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[start:start + BLOCK_SIZE]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + BLOCK_SIZE + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


x, y = get_batch(data)


print("\nBatch created.")

print("Input shape:", x.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# Create model
# --------------------------------------------------

print("\nCreating Ultron v0.2...")

model = UltronV2(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Parameters:",
    parameter_count
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print("\nRunning forward pass...")

logits = model(x)

print(
    "Logits shape:",
    logits.shape
)


# --------------------------------------------------
# Calculate loss
# --------------------------------------------------

print("\nCalculating loss...")

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    logits.reshape(-1, VOCAB_SIZE),
    y.reshape(-1)
)

print(
    "Initial loss:",
    loss.item()
)


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

print("\nRunning backward pass...")

loss.backward()

print("Backward pass completed.")


print("\n================================")
print("V0.2 TRAINING PIPELINE PASSED")
print("================================")