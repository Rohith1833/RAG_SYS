import torch

from feed_forward import FeedForward


batch_size = 1
sequence_length = 8
embedding_dim = 16


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


feed_forward = FeedForward(
    embedding_dim=embedding_dim
)


output = feed_forward(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)