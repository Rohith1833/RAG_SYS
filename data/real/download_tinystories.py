from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tinystories_train.txt"
)

NUM_STORIES = 10_000

print("Loading TinyStories in streaming mode...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Saving stories...")

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    for index, example in enumerate(dataset):

        text = example["text"].strip()

        if text:
            file.write(text)
            file.write("\n\n")

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1:,} stories"
            )

        if index + 1 >= NUM_STORIES:
            break

print("\nFinished.")
print("Dataset saved to:")
print(OUTPUT_PATH)