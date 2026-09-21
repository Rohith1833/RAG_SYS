from pathlib import Path

import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tinystories_train.txt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tokenizer.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tinystories_tokens.pt"
)


print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


print("Reading dataset...")

with open(
    TEXT_PATH,
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


print("Characters:", len(text))


print("Tokenizing dataset...")

encoded = tokenizer.encode(text)

token_ids = encoded.ids


print("Total tokens:", len(token_ids))


data = torch.tensor(
    token_ids,
    dtype=torch.long
)


torch.save(
    data,
    OUTPUT_PATH
)


print("\nDataset preparation finished.")
print("Token tensor shape:", data.shape)
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:")
print(OUTPUT_PATH)


print("\nFirst 50 token IDs:")
print(data[:50].tolist())


print("\nFirst 50 decoded tokens:")
print(
    tokenizer.decode(
        data[:50].tolist()
    )
)