import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        # Expand the representation.
        self.expand = nn.Linear(
            embedding_dim,
            4 * embedding_dim
        )

        # GELU is a smooth nonlinear activation.
        self.activation = nn.GELU()

        # Compress it back to the original size.
        self.project = nn.Linear(
            4 * embedding_dim,
            embedding_dim
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)

        return x