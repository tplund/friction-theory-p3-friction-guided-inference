"""Proxy B: Answer Instability.

Measures whether the model gives the same answer when asked multiple times.
Unstable answers indicate high internal conflict - multiple competing "routes"
with similar activation, so the winner varies between runs.

In RACE terms: if the race outcome changes on repeated runs, the competing
routes are very close in timing. That's high friction.
"""

from __future__ import annotations

import numpy as np

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult
from friktionsllm.config import DEFAULT_FRICTION


class InstabilityProxy(FrictionProxy):
    """Friction proxy based on answer variation across multiple samples."""

    name = "instability"

    def __init__(
        self,
        client: OllamaClient,
        *,
        n_samples: int | None = None,
        temperature: float | None = None,
        embedding_model: str | None = None,
    ):
        super().__init__(client)
        self.n_samples = n_samples or DEFAULT_FRICTION.instability_samples
        self.temperature = temperature or DEFAULT_FRICTION.instability_temperature
        self._embedder = None
        self._embedding_model = embedding_model or DEFAULT_FRICTION.embedding_model

    def _get_embedder(self):
        """Lazy-load the sentence transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def measure(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
        context: dict | None = None,
    ) -> FrictionResult:
        """Measure instability by generating multiple responses and comparing.

        Generates n_samples responses at low temperature and computes
        pairwise semantic similarity. High variation = high friction.
        """
        import time
        start = time.perf_counter()

        # Generate multiple samples
        responses: list[str] = []
        if generation is not None:
            responses.append(generation.text)

        while len(responses) < self.n_samples:
            result = self.client.generate(
                prompt,
                temperature=self.temperature,
                logprobs=False,
            )
            responses.append(result.text)

        # Compute pairwise semantic similarity
        embedder = self._get_embedder()
        embeddings = embedder.encode(responses, convert_to_numpy=True)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (pairwise similarities, excluding diagonal)
        n = len(responses)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(float(sim_matrix[i, j]))

        mean_similarity = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0
        min_similarity = float(np.min(pairwise_sims)) if pairwise_sims else 1.0

        # Friction = 1 - similarity (more variation = more friction)
        score = 1.0 - mean_similarity
        score = max(0.0, min(1.0, score))

        latency = (time.perf_counter() - start) * 1000

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=1.0 if n >= 3 else 0.7,
            metadata={
                "n_samples": n,
                "mean_similarity": round(mean_similarity, 4),
                "min_similarity": round(min_similarity, 4),
                "responses": [r[:100] for r in responses],  # truncated
            },
            latency_ms=latency,
        )
