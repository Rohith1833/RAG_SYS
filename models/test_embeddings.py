import torch

from embeddings import TokenEmbedding


# Our tokenizer found 33 unique characters.
vocab_size = 33

# Each token will be represented by 16 numbers.
embedding_dim = 16


embedding = TokenEmbedding(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)


# Example token IDs
token_ids = torch.tensor([
    [9, 20, 28, 26, 23, 22, 1, 18]
])


output = embedding(token_ids)


print("Input shape:")
print(token_ids.shape)

print("\nOutput shape:")
print(output.shape)

print("\nEmbedding output:")
print(output)