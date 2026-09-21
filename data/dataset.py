import torch
from tokenizer import CharacterTokenizer


# Read the training text
with open("data/training.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Create our tokenizer
tokenizer = CharacterTokenizer(text)


# Convert the entire text into token IDs
tokens = tokenizer.encode(text)


# Convert the list into a PyTorch tensor
data = torch.tensor(tokens, dtype=torch.long)


# Number of tokens the model sees at once
block_size = 8


# Pick one starting position
start = 0


# Create input and target sequences
x = data[start:start + block_size]
y = data[start + 1:start + block_size + 1]


print("Input token IDs:")
print(x)

print("\nTarget token IDs:")
print(y)

print("\nInput text:")
print(tokenizer.decode(x.tolist()))

print("\nTarget text:")
print(tokenizer.decode(y.tolist()))