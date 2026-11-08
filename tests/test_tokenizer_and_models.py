import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT / "data"))

from tokenizer import CharacterTokenizer
from ultron import UltronModel
from ultron_v2 import UltronV2


def test_tokenizer_v01():
    print("Testing Tokenizer v0.1...")
    corpus = "Hello Ultron! 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz .,!?"
    tok = CharacterTokenizer(corpus)
    text = "Hello Ultron!"
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert decoded == text, f"Mismatch: {decoded} != {text}"
    assert tok.vocab_size == len(set(corpus))
    print(f"  [PASS] Tokenizer v0.1: encoded '{text}' -> {len(tokens)} tokens, roundtrip preserved.")


def test_transformer_v01():
    print("Testing Ultron v0.1 Transformer...")
    vocab_size = 128
    block_size = 32
    embedding_dim = 32
    num_heads = 2
    num_layers = 2

    m = UltronModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    m.eval()

    sample_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = m(sample_input)

    assert logits.shape == (1, 4, vocab_size), f"Shape mismatch: {logits.shape}"
    total_params = sum(p.numel() for p in m.parameters())
    print(f"  [PASS] Ultron v0.1: forward pass successful, parameters = {total_params:,}.")


def test_transformer_v02():
    print("Testing Ultron v0.2 BPE Transformer...")
    vocab_size = 2000
    block_size = 128
    embedding_dim = 128
    num_heads = 4
    num_layers = 4

    m = UltronV2(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    m.eval()

    sample_input = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    with torch.no_grad():
        logits = m(sample_input)

    assert logits.shape == (1, 5, vocab_size), f"Shape mismatch: {logits.shape}"
    total_params = sum(p.numel() for p in m.parameters())
    print(f"  [PASS] Ultron v0.2: forward pass successful, parameters = {total_params:,}.")


def test_checkpoint_v02_exists_and_loads():
    print("Testing Ultron v0.2 Checkpoint Loading...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "ultron_v2_checkpoint.pt"
    assert ckpt_path.exists(), f"Checkpoint missing at {ckpt_path}"
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model_state_dict" in data
    assert "parameters" in data
    print(f"  [PASS] Checkpoint v0.2 verified: {data['parameters']:,} parameters.")


def main():
    print("=" * 60)
    print("TEST SUITE 1: TOKENIZER & TRANSFORMER MODELS (v0.1 & v0.2)")
    print("=" * 60)
    test_tokenizer_v01()
    test_transformer_v01()
    test_transformer_v02()
    test_checkpoint_v02_exists_and_loads()
    print("=" * 60)
    print("TEST SUITE 1 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
