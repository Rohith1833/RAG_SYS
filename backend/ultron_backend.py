import sys
from pathlib import Path
from typing import List, Dict, Optional

# --------------------------------------------------
# Project Paths & Import Setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "inference"))

from ollama_client import OllamaClient


# --------------------------------------------------
# Default System Prompt
# --------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are Ultron, a local AI assistant running on the user's computer. "
    "Give clear, useful, and honest answers. Do not claim to have capabilities "
    "or information you do not have. Explain technical concepts clearly when asked. "
    "You are powered by the local model llama3:latest running through Ollama as the "
    "backend reasoning engine for the Ultron project."
)


class UltronBackend:
    """
    UltronBackend acts as the local assistant brain for Ultron.
    It encapsulates system instructions, conversation history formatting,
    and queries the local Ollama instance running llama3:latest.
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        base_url: str = "http://127.0.0.1:11434"
    ):
        """
        Initialize the Ultron backend.

        :param model: The name of the underlying model (default: 'llama3:latest').
        :param system_prompt: System instructions defining Ultron's persona and rules.
        :param temperature: Sampling temperature for randomness vs determinism.
        :param base_url: The URL of the local Ollama API server.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

        # Reuse our tested OllamaClient
        self.client = OllamaClient(
            model=model,
            base_url=base_url
        )

    def check_connection(self, timeout: float = 1.5) -> bool:
        """Verify that the underlying Ollama service is reachable."""
        return self.client.check_connection(timeout=timeout)

    def format_prompt(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format the user's message and any previous conversation history
        into a structured prompt for the model.

        :param user_message: The latest question or instruction from the user.
        :param conversation: Optional list of previous turns:
                             [{"role": "user"|"assistant", "content": "..."}]
        :return: Formatted prompt string.
        """
        if not conversation:
            return user_message.strip()

        # Build a chronological dialogue history
        dialogue_turns = []
        for turn in conversation:
            role = "User" if turn.get("role") == "user" else "Ultron"
            content = turn.get("content", "").strip()
            if content:
                dialogue_turns.append(f"{role}: {content}")

        # Append the latest user query
        dialogue_turns.append(f"User: {user_message.strip()}")
        dialogue_turns.append("Ultron:")

        return "\n\n".join(dialogue_turns)

    def respond(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate Ultron's response to the user's message, taking any
        prior conversation history into context.

        :param user_message: The user's input text.
        :param conversation: Optional list of past turns for context.
        :return: Ultron's response text.
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        # Convert conversation turns into an input prompt
        prompt = self.format_prompt(user_message, conversation)

        # Query the local Ollama Llama 3 engine
        response_text = self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature
        )

        return response_text.strip()
