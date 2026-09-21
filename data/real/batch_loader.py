from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "train_tokens.pt"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "val_tokens.pt"
)


BLOCK_SIZE = 128
BATCH_SIZE = 32


print("Loading datasets...")

train_data = torch.load(
    TRAIN_PATH,
    weights_only=True
)

val_data = torch.load(
    VAL_PATH,
    weights_only=True
)


print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


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

    return x, y


print("\nCreating a test batch...")

x, y = get_batch(train_data)


print("Input shape:")
print(x.shape)

print("\nTarget shape:")
print(y.shape)


print("\nFirst training example:")
print("x:", x[0].tolist())
print("y:", y[0].tolist())


print("\nChecking shifted targets...")

for position in range(5):
    print(
        f"x[{position}] = {x[0, position].item()} "
        f"-> y[{position}] = {y[0, position].item()}"
    )


print("\nBatch loader test finished.")