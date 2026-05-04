"""Proxy E: Revision Need.

Compares a "quick" answer (standard generation) with a "deliberate" answer
(chain-of-thought prompted). If the deliberate answer differs significantly
from the quick one, it means the quick answer was produced under high pressure
and a different route won when given more time.

This is the most direct analog to the RACE model: it literally tests whether
"more race time" (deliberation) produces a different winner (answer).
"""

from __future__ import annotations

import time

import numpy as np

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class RevisionProxy(FrictionProxy):
    """Friction proxy based on how much the answer changes with more thinking time."""

    name = "revision"

    def __init__(self, client: OllamaClient, *, embedding_model: str | None = None):
        super().__init__(client)
        self._embedder = None
        self._embedding_model = embedding_model or "all-MiniLM-L6-v2"

    def _get_embedder(self):
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
        """Measure revision need.

        Compares a quick answer with a deliberate (chain-of-thought) answer.
        Large revision = high friction (the quick answer was unreliable).
        """
        start = time.perf_counter()

        # Quick answer (standard generation or reuse existing)
        if generation is not None:
            quick_answer = generation.text
        else:
            result = self.client.generate(prompt, logprobs=False)
            quick_answer = result.text

        # Deliberate answer with chain-of-thought
        deliberate_prompt = (
            f"Think through this step by step before giving your final answer. "
            f"Take your time and be thorough.\n\n"
            f"Question: {prompt}\n\n"
            f"Step-by-step reasoning:"
        )
        deliberate_result = self.client.generate(
            deliberate_prompt, max_tokens=1024, logprobs=False
        )
        deliberate_answer = deliberate_result.text

        # Compute semantic distance between quick and deliberate
        embedder = self._get_embedder()
        embeddings = embedder.encode(
            [quick_answer, deliberate_answer], convert_to_numpy=True
        )
        cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10
        ))

        # Revision distance as friction
        revision_distance = 1.0 - max(0.0, min(1.0, cos_sim))

        # Also measure: how much longer is the deliberate answer?
        length_ratio = len(deliberate_answer) / (len(quick_answer) + 1)

        # Friction score: primarily based on semantic distance
        # Length ratio is a minor additional signal (much longer = needed more thought)
        length_signal = min(length_ratio / 5.0, 1.0) * 0.2  # caps at 20% contribution
        score = 0.8 * revision_distance + length_signal
        score = max(0.0, min(1.0, score))

        latency = (time.perf_counter() - start) * 1000

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=0.9,
            metadata={
                "quick_answer": quick_answer[:200],
                "deliberate_answer": deliberate_answer[:200],
                "cosine_similarity": round(cos_sim, 4),
                "revision_distance": round(revision_distance, 4),
                "quick_length": len(quick_answer),
                "deliberate_length": len(deliberate_answer),
                "length_ratio": round(length_ratio, 2),
            },
            latency_ms=latency,
        )
