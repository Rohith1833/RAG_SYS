import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "models")
)

from ultron_v2 import UltronV2


VOCAB_SIZE = 2000
BLOCK_SIZE = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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


print("Parameters:", parameter_count)


test_tokens = torch.randint(
    0,
    VOCAB_SIZE,
    (2, BLOCK_SIZE),
    device=device
)


print("Input shape:", test_tokens.shape)


with torch.no_grad():
    logits = model(test_tokens)


print("Output shape:", logits.shape)


expected_shape = (
    2,
    BLOCK_SIZE,
    VOCAB_SIZE
)


if tuple(logits.shape) != expected_shape:
    raise RuntimeError(
        f"Unexpected output shape: {logits.shape}"
    )


print("\nV2 model test passed.")