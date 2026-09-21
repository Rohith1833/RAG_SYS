import sys
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------
# Make the project's data/ and models/ folders
# available to Python.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "models"))


from tokenizer import CharacterTokenizer
from ultron import UltronModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BLOCK_SIZE = 8
BATCH_SIZE = 32

EMBEDDING_DIM = 16
NUM_HEADS = 4
NUM_LAYERS = 3

LEARNING_RATE = 0.001
TRAINING_STEPS = 1000

EVAL_INTERVAL = 100


# --------------------------------------------------
# Choose CPU or GPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load training text
# --------------------------------------------------

data_path = PROJECT_ROOT / "data" / "training.txt"

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()


print("Characters:", len(text))


# --------------------------------------------------
# Create tokenizer
# --------------------------------------------------

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)

data = torch.tensor(
    tokens,
    dtype=torch.long
)


print("Vocabulary size:", tokenizer.vocab_size)
print("Total tokens:", len(data))


# --------------------------------------------------
# Split data into training and validation sets
# --------------------------------------------------

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]


# --------------------------------------------------
# Create batches
# --------------------------------------------------

def get_batch(data):
    """
    Create a batch of input and target sequences.
    """

    # Random starting positions.
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


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = UltronModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)


print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

loss_function = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(data):
    """
    Calculate average loss without training.
    """

    model.eval()

    with torch.no_grad():

        x, y = get_batch(data)

        logits = model(x)

        loss = loss_function(
            logits.reshape(-1, tokenizer.vocab_size),
            y.reshape(-1)
        )

    model.train()

    return loss.item()


# --------------------------------------------------
# Training loop
# --------------------------------------------------

print("\nStarting training...\n")

for step in range(TRAINING_STEPS):

    # Get training batch.
    x, y = get_batch(train_data)

    # Forward pass.
    logits = model(x)

    # Calculate loss.
    loss = loss_function(
        logits.reshape(-1, tokenizer.vocab_size),
        y.reshape(-1)
    )

    # Clear old gradients.
    optimizer.zero_grad()

    # Backpropagation.
    loss.backward()

    # Update model parameters.
    optimizer.step()

    # Print progress.
    if step % EVAL_INTERVAL == 0:

        train_loss = loss.item()
        val_loss = evaluate(val_data)

        print(
            f"Step {step:4d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )


# --------------------------------------------------
# Save the trained model
# --------------------------------------------------

checkpoint_path = (
    PROJECT_ROOT
    / "checkpoints"
    / "ultron_checkpoint.pt"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": BLOCK_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "characters": tokenizer.characters,
    },
    checkpoint_path
)


print("\nTraining finished.")

print(
    "Checkpoint saved to:",
    checkpoint_path
)