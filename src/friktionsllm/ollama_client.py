"""Thin wrapper around Ollama REST API with logprobs support.

All model interaction goes through this module. Uses httpx for direct
REST calls to get reliable access to logprobs (token probabilities),
which is essential for friction measurement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import httpx

from friktionsllm.config import ModelConfig, DEFAULT_MODEL


@dataclass
class TokenLogprob:
    """Log-probability info for a single generated token."""
    token: str
    logprob: float
    top_alternatives: list[dict[str, float]]  # [{token: logprob}, ...]

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def entropy(self) -> float:
        """Entropy computed from top alternatives (lower bound on true entropy)."""
        probs = [math.exp(self.logprob)]
        for alt in self.top_alternatives:
            for _tok, lp in alt.items():
                probs.append(math.exp(lp))
        # Normalize to sum to 1 (since we only have top-k)
        total = sum(probs)
        if total <= 0:
            return 0.0
        probs = [p / total for p in probs]
        return -sum(p * math.log(p) for p in probs if p > 0)

    @property
    def margin(self) -> float:
        """Gap between top-1 and top-2 logprobs. Higher = more confident."""
        if not self.top_alternatives:
            return float("inf")
        second_best = max(
            lp for alt in self.top_alternatives for lp in alt.values()
        )
        return self.logprob - second_best


@dataclass
class GenerationResult:
    """Result of a single generation call."""
    text: str
    token_logprobs: list[TokenLogprob]
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def mean_entropy(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return sum(t.entropy for t in self.token_logprobs) / len(self.token_logprobs)

    @property
    def mean_margin(self) -> float:
        if not self.token_logprobs:
            return float("inf")
        margins = [t.margin for t in self.token_logprobs if t.margin != float("inf")]
        if not margins:
            return float("inf")
        return sum(margins) / len(margins)

    @property
    def max_entropy(self) -> float:
        if not self.token_logprobs:
            return 0.0
        return max(t.entropy for t in self.token_logprobs)

    @property
    def high_entropy_ratio(self) -> float:
        """Fraction of tokens with entropy above 2.0 (empirical threshold)."""
        if not self.token_logprobs:
            return 0.0
        high = sum(1 for t in self.token_logprobs if t.entropy > 2.0)
        return high / len(self.token_logprobs)


class OllamaClient:
    """Client for Ollama REST API with logprobs support."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or DEFAULT_MODEL
        self.base_url = self.config.base_url.rstrip("/")
        self._client = httpx.Client(timeout=300.0)  # 5min timeout for slow CPU inference

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
        """Generate a completion.

        Uses the OpenAI-compatible /v1/chat/completions endpoint when
        logprobs are requested (the native /api/generate doesn't reliably
        return them). Falls back to the native endpoint when logprobs
        are not needed.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            model: Override config model name.
            logprobs: Whether to request logprobs (default True).

        Returns:
            GenerationResult with text, token logprobs, and metadata.
        """
        model = model or self.config.name
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        if logprobs:
            # Use OpenAI-compatible endpoint for reliable logprobs
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tok,
                "temperature": temp,
                "logprobs": True,
                "top_logprobs": self.config.top_logprobs,
            }

            # Retry with backoff - Ollama can OOM on logprobs requests
            last_error = None
            for attempt in range(3):
                start = time.perf_counter()
                try:
                    resp = self._client.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    )
                    latency_ms = (time.perf_counter() - start) * 1000
                    resp.raise_for_status()
                    data = resp.json()

                    token_logprobs = _parse_logprobs(data)
                    text = data["choices"][0]["message"]["content"]
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
                    # Wait and retry - gives Ollama time to free memory
                    time.sleep(2 * (attempt + 1))

            # All retries failed - fall back to no logprobs
            result = self.generate(
                prompt, system=system, temperature=temperature,
                max_tokens=max_tokens, model=model, logprobs=False,
            )
            return result
        else:
            # Use OpenAI-compatible endpoint without logprobs
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

            last_error = None
            for attempt in range(3):
                start = time.perf_counter()
                try:
                    resp = self._client.post(
                        f"{self.base_url}/v1/chat/completions", json=payload
                    )
                    latency_ms = (time.perf_counter() - start) * 1000
                    resp.raise_for_status()
                    data = resp.json()

                    return GenerationResult(
                        text=data["choices"][0]["message"]["content"],
                        token_logprobs=[],
                        model=model,
                        latency_ms=latency_ms,
                        prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                    )
                except httpx.HTTPStatusError as e:
                    last_error = e
                    time.sleep(2 * (attempt + 1))

            raise last_error

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        logprobs: bool = True,
    ) -> GenerationResult:
        """Generate using the OpenAI-compatible chat endpoint.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            model: Override config model name.
            logprobs: Whether to request logprobs.

        Returns:
            GenerationResult with text, token logprobs, and metadata.
        """
        model = model or self.config.name
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.config.top_logprobs

        start = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        token_logprobs = _parse_logprobs(data) if logprobs else []
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return GenerationResult(
            text=text,
            token_logprobs=token_logprobs,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self) -> list[str]:
        """List available local models."""
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _parse_logprobs(data: dict) -> list[TokenLogprob]:
    """Parse logprobs from Ollama API response.

    Ollama's logprob format may vary by version. This function handles
    known formats and returns an empty list if logprobs aren't available.
    """
    token_logprobs = []

    # Try the direct logprobs field (newer Ollama versions)
    if "logprobs" in data:
        for entry in data["logprobs"]:
            top_alts = []
            for alt in entry.get("top_logprobs", [])[1:]:  # skip first (it's the chosen token)
                top_alts.append({alt["token"]: alt["logprob"]})
            token_logprobs.append(TokenLogprob(
                token=entry.get("token", ""),
                logprob=entry.get("logprob", 0.0),
                top_alternatives=top_alts,
            ))

    # Try OpenAI-compatible format via /v1 endpoint
    elif "choices" in data:
        for choice in data.get("choices", []):
            lp_data = choice.get("logprobs", {})
            for i, content_item in enumerate(lp_data.get("content", [])):
                top_alts = []
                for alt in content_item.get("top_logprobs", [])[1:]:
                    top_alts.append({alt["token"]: alt["logprob"]})
                token_logprobs.append(TokenLogprob(
                    token=content_item.get("token", ""),
                    logprob=content_item.get("logprob", 0.0),
                    top_alternatives=top_alts,
                ))

    return token_logprobs
