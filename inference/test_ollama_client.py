import sys
from pathlib import Path

# Add project root to sys.path to ensure clean import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "inference"))

from ollama_client import OllamaClient


def main():
    print("=" * 60)
    print("ULTRON OLLAMA BACKEND INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize client with llama3:latest
    model_name = "llama3:latest"
    client = OllamaClient(model=model_name)

    # 2. Check connection to Ollama server first
    try:
        client.check_connection()
        print(f"[OK] Connected to Ollama server at {client.base_url}")
        print(f"[OK] Target Model: {client.model}")
    except ConnectionError as conn_err:
        print(conn_err)
        sys.exit(1)

    # 3. Prompt for the test
    prompt = "Explain what a transformer neural network is in simple words."

    print("\nSending prompt to Ultron's Llama backend...")
    print(f"Prompt: \"{prompt}\"\n")

    # 4. Generate response
    try:
        response_text = client.generate(
            prompt=prompt,
            temperature=0.7
        )

        print("Ultron response:")
        print(response_text)
        print("\n" + "=" * 60)
        print("TEST PASSED: Successfully received response from llama3:latest")
        print("=" * 60)

    except Exception as err:
        print(f"\n[Generation Error] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
