import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlockV2(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

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


class UltronV2(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        embedding_dim=128,
        num_heads=4,
        num_layers=4
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockV2(
                embedding_dim=embedding_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds block size {self.block_size}"
            )

        token_vectors = self.token_embedding(
            token_ids
        )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.output_head(x)

        return logits