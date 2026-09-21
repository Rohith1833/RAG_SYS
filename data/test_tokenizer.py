from tokenizer import CharacterTokenizer


text = "hello ultron"

tokenizer = CharacterTokenizer(text)

print("Vocabulary:")
print(tokenizer.characters)

print("\nVocabulary size:")
print(tokenizer.vocab_size)

encoded = tokenizer.encode("hello")

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)