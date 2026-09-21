import requests
from typing import Optional


class OllamaClient:
    """
    A lightweight, robust client for communicating with a local Ollama instance.
    Enables Ultron to query local large language models (such as llama3:latest).
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120
    ):
        """
        Initialize the Ollama client.

        :param model: The name of the Ollama model to use (default: 'llama3:latest').
        :param base_url: The base URL where Ollama is listening (default: 'http://127.0.0.1:11434').
        :param timeout: Request timeout in seconds (default: 120s for larger LLM generation).
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self, timeout: float = 1.5) -> bool:
        """
        Verify that the local Ollama server is active and reachable.
        Raises ConnectionError with beginner-friendly instructions if unreachable.
        """
        try:
            response = requests.get(self.base_url, timeout=timeout)
            if response.status_code == 200:
                return True
            else:
                raise ConnectionError(
                    f"Ollama responded with status code {response.status_code} at {self.base_url}."
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"\n[Ollama Connection Error]\n"
                f"Could not connect to Ollama at '{self.base_url}'.\n"
                f"Please ensure Ollama is installed and running.\n"
                f"To start Ollama, open PowerShell or a terminal and run:\n"
                f"    ollama serve\n"
            )
        except requests.exceptions.RequestException as err:
            raise ConnectionError(
                f"Unexpected error connecting to Ollama at '{self.base_url}': {err}"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Send a prompt to the Ollama model and return the generated text.

        :param prompt: The input prompt string for the model.
        :param system_prompt: Optional system instruction to guide model behavior.
        :param temperature: Sampling temperature controlling randomness (e.g., 0.0 to 1.0).
        :return: Generated text response from the model.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Build payload according to Ollama /api/generate specification
        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Non-streaming for clean single-string return
            "options": {
                "temperature": float(temperature)
            }
        }

        # Include optional system instructions if provided
        if system_prompt:
            payload["system"] = system_prompt

        try:
            # Send HTTP POST request to Ollama
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"\n[Ollama Connection Error]\n"
                f"Connection lost or could not connect to Ollama at '{self.base_url}'.\n"
                f"Please verify Ollama is running in the background ('ollama serve')."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"\n[Ollama Timeout Error]\n"
                f"Generation with model '{self.model}' exceeded timeout of {self.timeout} seconds."
            )
        except requests.exceptions.HTTPError as http_err:
            status = response.status_code if response else "unknown"
            raise RuntimeError(
                f"Ollama API error (HTTP {status}): {http_err}. "
                f"Please verify that model '{self.model}' is installed with 'ollama list'."
            )
        except Exception as err:
            raise RuntimeError(f"Unexpected error during Ollama generation: {err}")
