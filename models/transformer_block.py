import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
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

        # Attention + residual connection
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x