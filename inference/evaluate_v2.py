import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "models")
)

from ultron_v2 import UltronV2


# --------------------------------------------------
# Paths & Device
# --------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "ultron_v2_checkpoint.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "tokenizer.json"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model_and_tokenizer():
    print(f"Loading Ultron v0.2 on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = UltronV2(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


def generate(
    model,
    tokenizer,
    block_size,
    prompt,
    max_new_tokens=100,
    temperature=0.8
):
    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids
    if len(prompt_ids) == 0:
        prompt_ids = [1]  # <bos>

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    eos_token_id = tokenizer.token_to_id("<eos>")

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[:, -block_size:]
            logits = model(context)
            logits = logits[:, -1, :]
            logits = logits / max(float(temperature), 0.01)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

            token_ids = torch.cat([token_ids, next_tok], dim=1)

    elapsed = time.perf_counter() - t0
    generated_text = tokenizer.decode(
        token_ids[0].tolist(),
        skip_special_tokens=True
    )
    tokens_count = token_ids.shape[1] - len(prompt_ids)
    return generated_text, tokens_count, elapsed


def run_evaluation():
    model, tokenizer, checkpoint = load_model_and_tokenizer()

    print("\n" + "=" * 60)
    print("ULTRON v0.2 EVALUATION RUN")
    print(f"Parameters: {checkpoint['parameters']:,}")
    print(f"Context: {checkpoint['block_size']} tokens")
    print(f"Vocabulary: {checkpoint['vocab_size']}")
    print("=" * 60)

    test_prompts = [
        "Once upon a time",
        "A little boy",
        "One day, a cat",
        "The dog wanted",
        "There was a little girl"
    ]

    for index, prompt in enumerate(test_prompts, 1):
        print(f"\n--- [Prompt {index}/{len(test_prompts)}] ---")
        print(f"Prompt: \"{prompt}\"")

        output, tokens_gen, elapsed = generate(
            model=model,
            tokenizer=tokenizer,
            block_size=checkpoint["block_size"],
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8
        )

        speed = round(tokens_gen / max(elapsed, 0.001), 1)
        print(f"Generated ({tokens_gen} tokens in {round(elapsed*1000, 1)}ms, {speed} tok/s):")
        print(output)
        print("-" * 40)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    run_evaluation()
