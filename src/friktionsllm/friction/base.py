"""Base class for friction proxies."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pydantic import BaseModel

from friktionsllm.ollama_client import OllamaClient, GenerationResult


class FrictionResult(BaseModel):
    """Result from a single friction proxy measurement."""
    proxy_name: str
    score: float  # 0.0 (no friction) to 1.0 (maximum friction)
    confidence: float = 1.0  # how reliable is this measurement
    metadata: dict = {}  # proxy-specific details
    latency_ms: float = 0.0


class FrictionProxy(ABC):
    """Abstract base class for friction measurement proxies.

    Each proxy measures a different aspect of model uncertainty/conflict
    and returns a normalized score in [0, 1].
    """

    name: str = "base"

    def __init__(self, client: OllamaClient):
        self.client = client

    @abstractmethod
    def measure(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
        context: dict | None = None,
    ) -> FrictionResult:
        """Measure friction for a given prompt.

        Args:
            prompt: The input prompt.
            generation: Pre-existing generation result (to avoid redundant calls).
            context: Optional context (facts, evidence) for evidence-based proxies.

        Returns:
            FrictionResult with normalized score and metadata.
        """
        ...

    def _timed_measure(self, func, *args, **kwargs) -> tuple:
        """Helper to time a measurement function."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
