"""Together.ai API client with logprobs support.

Drop-in replacement for OllamaClient - returns same TokenLogprob and
GenerationResult objects so all proxies and steering work unchanged.

Usage:
    from friktionsllm.together_client import TogetherClient
    client = TogetherClient(model="Qwen/Qwen3.5-9B")
    result = client.generate("What is 2+2?", logprobs=True)
"""

from __future__ import annotations

import os
import time

import httpx

from friktionsllm.config import ModelConfig
from friktionsllm.ollama_client import TokenLogprob, GenerationResult


class TogetherClient:
    """Together.ai API client compatible with OllamaClient interface."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.config = config or ModelConfig()
        self.model = model or self.config.name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        self.base_url = "https://api.together.xyz/v1"
        self._client = httpx.Client(timeout=120)

        if not self.api_key:
            # Try loading from .env
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.environ.get("TOGETHER_API_KEY", "")
            except ImportError:
                pass

    def is_available(self) -> bool:
        """Check if the API is reachable and key is set."""
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        logprobs: bool = True,
    ) -> GenerationResult:
        """Generate a response with optional logprobs.

        Returns the same GenerationResult as OllamaClient.
        """
        model = model or self.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.config.top_logprobs or 5

        last_error = None
        for attempt in range(3):
            start = time.perf_counter()
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                data = resp.json()

                # Parse response
                choice = data["choices"][0]
                text = choice["message"]["content"]

                # Parse logprobs into TokenLogprob objects
                token_logprobs = []
                if logprobs and choice.get("logprobs") and choice["logprobs"].get("content"):
                    for token_data in choice["logprobs"]["content"]:
                        # Build alternatives list
                        alternatives = []
                        if token_data.get("top_logprobs"):
                            for alt in token_data["top_logprobs"]:
                                if alt["token"] != token_data["token"]:
                                    alternatives.append({alt["token"]: alt["logprob"]})

                        token_logprobs.append(TokenLogprob(
                            token=token_data["token"],
                            logprob=token_data["logprob"],
                            top_alternatives=alternatives,
                        ))

                usage = data.get("usage", {})
                return GenerationResult(
                    text=text,
                    token_logprobs=token_logprobs,
                    model=model,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                last_error = e
                time.sleep(2 * (attempt + 1))

        raise last_error

    def close(self):
        """Close the HTTP client."""
        self._client.close()
