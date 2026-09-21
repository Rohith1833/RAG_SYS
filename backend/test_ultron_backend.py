import sys
from pathlib import Path

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from ultron_backend import UltronBackend


def main():
    print("=" * 60)
    print("ULTRON BACKEND CONVERSATION TEST")
    print("=" * 60)

    # 1. Initialize Ultron backend
    backend = UltronBackend(model="llama3:latest")

    # 2. Check connection
    try:
        backend.check_connection()
        print(f"[OK] Backend initialized with model: {backend.model}")
        print(f"[OK] Connection verified to local Ollama server\n")
    except ConnectionError as err:
        print(f"[Connection Error] {err}")
        sys.exit(1)

    conversation_history = []

    # --------------------------------------------------
    # Turn 1: Purpose query
    # --------------------------------------------------
    first_message = "Hello Ultron. What is your purpose?"
    print(f"User: {first_message}")
    print("Ultron is thinking...\n")

    response_1 = backend.respond(
        user_message=first_message,
        conversation=conversation_history
    )

    print("Ultron:")
    print(response_1)
    print("-" * 60)

    # Record turn in conversation history
    conversation_history.append({"role": "user", "content": first_message})
    conversation_history.append({"role": "assistant", "content": response_1})

    # --------------------------------------------------
    # Turn 2: Contextual follow-up question
    # --------------------------------------------------
    second_message = "What model are you using underneath?"
    print(f"User: {second_message}")
    print("Ultron is thinking...\n")

    response_2 = backend.respond(
        user_message=second_message,
        conversation=conversation_history
    )

    print("Ultron:")
    print(response_2)
    print("=" * 60)
    print("TEST PASSED: Two-turn conversation completed through Ollama (llama3:latest)")
    print("=" * 60)


if __name__ == "__main__":
    main()
