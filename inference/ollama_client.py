import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Any, Generator


class OllamaClient:
    """
    A high-performance, robust client for communicating with a local Ollama instance.
    Enables Ultron to query local large language models (such as llama3:latest)
    with persistent HTTP session reuse, configurable keep-alive, streaming token
    generation, and optimized hardware context options.
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
        keep_alive: str = "15m",
        num_ctx: int = 4096
    ):
        """
        Initialize the Ollama client.

        :param model: The name of the Ollama model to use (default: 'llama3:latest').
        :param base_url: The base URL where Ollama is listening (default: 'http://127.0.0.1:11434').
        :param timeout: Request timeout in seconds (default: 120s for larger LLM generation).
        :param keep_alive: How long the model should remain loaded in VRAM (e.g. '15m', '30m', '-1').
        :param num_ctx: Context window size for Ollama generation (default: 4096 for RTX 3050 4GB).
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

        # Persistent HTTP session for connection pooling and TCP keep-alive
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(total=1, backoff_factor=0.2)
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def close(self):
        """Close the underlying HTTP session resources."""
        if self.session:
            self.session.close()

    def check_connection(self, timeout: float = 1.5) -> bool:
        """
        Verify that the local Ollama server is active and reachable.
        Raises ConnectionError with beginner-friendly instructions if unreachable.
        """
        try:
            response = self.session.get(self.base_url, timeout=timeout)
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

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """Construct the standardized request payload for Ollama /api/generate."""
        merged_options = {
            "temperature": float(temperature),
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
        if options:
            merged_options.update(options)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": keep_alive or self.keep_alive,
            "options": merged_options
        }
        if system_prompt:
            payload["system"] = system_prompt

        return payload

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the Ollama model and return the generated text.
        Preserves original synchronous interface for full backwards-compatibility.

        :param prompt: The input prompt string for the model.
        :param system_prompt: Optional system instruction to guide model behavior.
        :param temperature: Sampling temperature controlling randomness (e.g., 0.0 to 1.0).
        :param options: Optional fine-grained generation parameters.
        :param keep_alive: Optional override for model VRAM residency duration.
        :return: Generated text response from the model.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        endpoint = f"{self.base_url}/api/generate"
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            stream=False,
            options=options,
            keep_alive=keep_alive
        )

        try:
            response = self.session.post(
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

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens incrementally from Ollama /api/generate as they are decoded.

        Yields dictionaries with:
        - {"token": str, "done": False} for each emitted token.
        - {"token": "", "done": True, "eval_count": int, "eval_duration": int, ...} on completion.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        endpoint = f"{self.base_url}/api/generate"
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            stream=True,
            options=options,
            keep_alive=keep_alive
        )

        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    is_done = chunk.get("done", False)
                    token = chunk.get("response", "")
                    yield {
                        "token": token,
                        "done": is_done,
                        "eval_count": chunk.get("eval_count"),
                        "eval_duration": chunk.get("eval_duration"),
                        "prompt_eval_count": chunk.get("prompt_eval_count"),
                        "prompt_eval_duration": chunk.get("prompt_eval_duration")
                    }
                    if is_done:
                        break
                except json.JSONDecodeError:
                    continue

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Connection lost to Ollama at '{self.base_url}'. Please ensure 'ollama serve' is running."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Generation with model '{self.model}' exceeded timeout of {self.timeout} seconds."
            )
        except Exception as err:
            raise RuntimeError(f"Streaming generation error from Ollama: {err}")

    def unload_model(self) -> bool:
        """
        Unload the model immediately from GPU VRAM by setting keep_alive to 0.
        Returns True if successfully communicated to Ollama.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "keep_alive": 0
        }
        try:
            res = self.session.post(endpoint, json=payload, timeout=5)
            return res.status_code == 200
        except Exception:
            return False
