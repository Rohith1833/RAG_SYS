import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any

# --------------------------------------------------
# Project Paths & Import Setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "inference"))

from ollama_client import OllamaClient


# --------------------------------------------------
# Default System Prompt (Concise & Focused)
# --------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are Ultron, a sovereign local AI assistant running on the user's computer. "
    "Provide clear, accurate, and concise responses. Do not claim capabilities you do not have. "
    "You are powered by llama3:latest running through local Ollama inference."
)


class UltronBackend:
    """
    UltronBackend acts as the local assistant brain for Ultron.
    It encapsulates system instructions, conversation history formatting,
    and queries the local Ollama instance running llama3:latest, supporting
    both synchronous generation and real-time token streaming.
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "15m",
        num_ctx: int = 4096
    ):
        """
        Initialize the Ultron backend.

        :param model: The name of the underlying model (default: 'llama3:latest').
        :param system_prompt: System instructions defining Ultron's persona and rules.
        :param temperature: Sampling temperature for randomness vs determinism.
        :param base_url: The URL of the local Ollama API server.
        :param keep_alive: Time to retain model in GPU memory (e.g. '15m').
        :param num_ctx: Max context length for Ollama (default: 4096).
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

        # Reuse persistent OllamaClient
        self.client = OllamaClient(
            model=model,
            base_url=base_url,
            keep_alive=keep_alive,
            num_ctx=num_ctx
        )

    def close(self):
        """Clean up underlying network resources."""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def unload_model(self) -> bool:
        """Release the LLM from GPU VRAM."""
        return self.client.unload_model()

    def check_connection(self, timeout: float = 1.5) -> bool:
        """Verify that the underlying Ollama service is reachable."""
        return self.client.check_connection(timeout=timeout)

    def format_prompt(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        max_history_turns: int = 6
    ) -> str:
        """
        Format the user's message and recent conversation history
        into a structured prompt for the model.
        Caps historical turns to prevent unnecessary context overhead.

        :param user_message: The latest question or instruction from the user.
        :param conversation: Optional list of previous turns:
                             [{"role": "user"|"assistant", "content": "..."}]
        :param max_history_turns: Maximum recent turns to include (default: 6).
        :return: Formatted prompt string.
        """
        if not conversation:
            return user_message.strip()

        # Limit to the most recent turns to optimize prompt evaluation latency
        recent_turns = conversation[-max_history_turns:] if len(conversation) > max_history_turns else conversation

        dialogue_turns = []
        for turn in recent_turns:
            role = "User" if turn.get("role") == "user" else "Ultron"
            content = turn.get("content", "").strip()
            if content:
                dialogue_turns.append(f"{role}: {content}")

        dialogue_turns.append(f"User: {user_message.strip()}")
        dialogue_turns.append("Ultron:")

        return "\n\n".join(dialogue_turns)

    def respond(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate Ultron's response synchronously.

        :param user_message: The user's input text.
        :param conversation: Optional list of past turns for context.
        :param system_prompt: Optional override for the system prompt.
        :return: Ultron's response text.
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        prompt = self.format_prompt(user_message, conversation)
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt

        response_text = self.client.generate(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=self.temperature
        )

        return response_text.strip()

    def respond_stream(
        self,
        user_message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream Ultron's response token-by-token.

        :param user_message: The user's input text.
        :param conversation: Optional list of past turns for context.
        :param system_prompt: Optional override for the system prompt.
        :yield: Dicts containing {"token": str, "done": bool, ...}
        """
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        prompt = self.format_prompt(user_message, conversation)
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt

        for chunk in self.client.generate_stream(
            prompt=prompt,
            system_prompt=sys_prompt,
            temperature=self.temperature
        ):
            yield chunk
