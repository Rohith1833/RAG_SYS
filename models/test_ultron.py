import torch

from ultron import UltronModel


# Our tokenizer found 33 unique characters.
vocab_size = 33

# Maximum number of tokens Ultron can process at once.
block_size = 8

# Size of each token representation.
embedding_dim = 16

# Number of attention heads.
num_heads = 4

# Number of Transformer blocks.
num_layers = 3


model = UltronModel(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers
)


# One sequence containing 8 token IDs.
token_ids = torch.tensor([
    [9, 20, 28, 26, 23, 22, 1, 18]
])


logits = model(token_ids)


print("Input shape:")
print(token_ids.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nNumber of parameters:")
print(sum(p.numel() for p in model.parameters()))