import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class UltronModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=3
    ):
        super().__init__()

        self.block_size = block_size

        # Converts token IDs into vectors.
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Gives each position its own learnable vector.
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # Stack multiple Transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        # Final normalization.
        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # Converts each vector into vocabulary scores.
        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        # Make sure the sequence isn't longer than our context.
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds block size {self.block_size}"
            )

        # Token embeddings.
        token_vectors = self.token_embedding(token_ids)

        # Position IDs: 0, 1, 2, ..., sequence_length - 1
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Position embeddings.
        position_vectors = self.position_embedding(
            positions
        )

        # Combine token + position information.
        x = token_vectors + position_vectors

        # Pass through Transformer blocks.
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization.
        x = self.final_layer_norm(x)

        # Convert to vocabulary logits.
        logits = self.output_head(x)

        return logits