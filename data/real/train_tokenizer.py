from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tinystories_train.txt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tokenizer.json"
)

VOCAB_SIZE = 2000


print("Creating Byte-Level BPE tokenizer...")

tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)

tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()


trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)


print("Training tokenizer...")

tokenizer.train(
    [str(DATA_PATH)],
    trainer
)


tokenizer.save(
    str(TOKENIZER_PATH)
)


print("\nTokenizer training finished.")
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Saved to:")
print(TOKENIZER_PATH)


print("\nTesting tokenizer...")

sample = (
    "Once upon a time, Lily went to the park "
    "and played with her friends."
)

encoded = tokenizer.encode(sample)

print("\nOriginal text:")
print(sample)

print("\nToken IDs:")
print(encoded.ids)

print("\nTokens:")
print(encoded.tokens)

print("\nDecoded text:")
print(tokenizer.decode(encoded.ids))