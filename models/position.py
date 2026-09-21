import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_embeddings):
        # token_embeddings shape:
        # (batch_size, sequence_length, embedding_dim)

        batch_size, sequence_length, embedding_dim = token_embeddings.shape

        # Create position IDs:
        # [0, 1, 2, ..., sequence_length - 1]
        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        # Look up an embedding vector for each position.
        position_embeddings = self.position_embedding(positions)

        # Add position information to token information.
        return token_embeddings + position_embeddings