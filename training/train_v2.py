import sys
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------
# Project paths
# --------------------------------------------------

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

BATCH_SIZE = 16

LEARNING_RATE = 3e-4

TRAINING_STEPS = 5000
EVAL_INTERVAL = 250

EVAL_BATCHES = 20


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
# Dataset paths
# --------------------------------------------------

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "train_tokens.pt"
)

VAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "val_tokens.pt"
)


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "ultron_v2_checkpoint.pt"
)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\nLoading training data...")

train_data = torch.load(
    TRAIN_DATA_PATH,
    weights_only=True
)

val_data = torch.load(
    VAL_DATA_PATH,
    weights_only=True
)

print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(val_data)
)


# --------------------------------------------------
# Batch loader
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

    return (
        x.to(device),
        y.to(device)
    )


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
# Loss and optimizer
# --------------------------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def estimate_loss(data):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(data)

            logits = model(x)

            loss = loss_function(
                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                y.reshape(-1)
            )

            total_loss += loss.item()

    model.train()

    return total_loss / EVAL_BATCHES


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStarting v0.2 training...\n")

for step in range(TRAINING_STEPS):

    x, y = get_batch(train_data)

    logits = model(x)

    loss = loss_function(
        logits.reshape(
            -1,
            VOCAB_SIZE
        ),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % EVAL_INTERVAL == 0:

        train_loss = estimate_loss(
            train_data
        )

        val_loss = estimate_loss(
            val_data
        )

        print(
            f"Step {step:5d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )


# --------------------------------------------------
# Save checkpoint
# --------------------------------------------------

print("\nSaving checkpoint...")

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,

        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,

        "parameters": parameter_count,

        "training_steps": TRAINING_STEPS,

        "learning_rate": LEARNING_RATE
    },
    CHECKPOINT_PATH
)


print("\n================================")
print("ULTRON V0.2 TRAINING FINISHED")
print("================================")

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)