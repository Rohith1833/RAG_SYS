from tokenizer import CharacterTokenizer


# 1. Read our training text
with open("data/training.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Create a tokenizer from the training text
tokenizer = CharacterTokenizer(text)


# 3. Convert the entire training text into token IDs
tokens = tokenizer.encode(text)


# 4. Print information about our dataset
print("Characters in dataset:", len(text))
print("Vocabulary size:", tokenizer.vocab_size)

print("\nFirst 100 characters:")
print(text[:100])

print("\nFirst 100 token IDs:")
print(tokens[:100])