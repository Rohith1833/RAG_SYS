import torch

from embeddings import TokenEmbedding
from position import PositionalEmbedding


vocab_size = 33
embedding_dim = 16
block_size = 8


token_embedding = TokenEmbedding(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)

position_embedding = PositionalEmbedding(
    block_size=block_size,
    embedding_dim=embedding_dim
)


token_ids = torch.tensor([
    [9, 20, 28, 26, 23, 22, 1, 18]
])


# Convert token IDs into vectors.
token_vectors = token_embedding(token_ids)

# Add positional information.
output = position_embedding(token_vectors)


print("Token IDs shape:")
print(token_ids.shape)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nFinal embedding shape:")
print(output.shape)