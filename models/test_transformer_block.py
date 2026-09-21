import torch

from transformer_block import TransformerBlock


batch_size = 1
sequence_length = 8
embedding_dim = 16
num_heads = 4


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)


output = block(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)