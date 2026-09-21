import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # One projection creates Q, K and V together.
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3
        )

        # Combines all attention heads back together.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        # Create Query, Key and Value.
        qkv = self.qkv(x)

        # Split into Q, K and V.
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape:
        #
        # Before:
        # (batch, sequence, embedding)
        #
        # After:
        # (batch, heads, sequence, head_dim)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Calculate attention scores.
        scores = q @ k.transpose(-2, -1)

        # Scale the scores.
        scores = scores / math.sqrt(self.head_dim)

        # Prevent tokens from looking into the future.
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert scores to probabilities.
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # Use attention weights to combine values.
        output = weights @ v

        # Put all heads back together.
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        # Final linear projection.
        output = self.output_projection(output)

        return output