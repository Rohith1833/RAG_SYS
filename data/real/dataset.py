from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tinystories_tokens.pt"
)


print("Loading token dataset...")

data = torch.load(
    DATA_PATH,
    weights_only=True
)

print("Total tokens:", len(data))


split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]


print("\nDataset split:")
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


print("\nSaving splits...")

torch.save(
    train_data,
    PROJECT_ROOT
    / "data"
    / "real"
    / "train_tokens.pt"
)

torch.save(
    val_data,
    PROJECT_ROOT
    / "data"
    / "real"
    / "val_tokens.pt"
)


print("\nFinished.")