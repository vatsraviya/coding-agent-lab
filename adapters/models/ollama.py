"""A chat client backed by a local Ollama server, using only the standard library.

``OllamaClient`` matches the ``chat(messages) -> str`` interface of
``ScriptedClient``: it forwards the prepared messages to a local Ollama server
and returns only the reply text. It requests JSON-formatted output but does not
validate the reply; ``RepairAgent`` remains the single place that validates
every reply against the three permitted actions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaClient:
    """Send messages to a local Ollama server and return the reply text."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        host: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self._temperature = float(temperature)
        self._host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return one reply from the local model for the given conversation."""

        payload = {
            "model": self._model,
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self._temperature},
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned a non-JSON envelope: {exc}") from exc

        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response shape: {body!r}") from exc
