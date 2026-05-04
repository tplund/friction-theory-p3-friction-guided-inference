"""Proxy D: Self-Contradiction.

Asks the model to provide both an answer and an alternative answer,
then measures the semantic distance between them.

If two very different answers are both plausible to the model, it signals
high internal conflict - multiple strong competing routes.

In RACE terms: two routes reached near-threshold activation simultaneously.
The race was almost a tie.
"""

from __future__ import annotations

import time

import numpy as np

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class ContradictionProxy(FrictionProxy):
    """Friction proxy based on self-contradiction between primary and alternative answers."""

    name = "contradiction"

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
        """Measure self-contradiction friction.

        Asks for a primary answer and an alternative, then computes
        the semantic distance between them.
        """
        start = time.perf_counter()

        # Get primary answer
        if generation is not None:
            primary_answer = generation.text
        else:
            result = self.client.generate(prompt, logprobs=False)
            primary_answer = result.text

        # Ask for alternative answer
        alt_prompt = (
            f"I asked: {prompt}\n\n"
            f"One answer is: {primary_answer}\n\n"
            f"Now give me a different, alternative answer that could also be "
            f"correct or plausible. Just give the alternative answer, nothing else."
        )
        alt_result = self.client.generate(alt_prompt, logprobs=False)
        alt_answer = alt_result.text

        # Compute semantic distance
        embedder = self._get_embedder()
        embeddings = embedder.encode(
            [primary_answer, alt_answer], convert_to_numpy=True
        )
        cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10
        ))

        # Semantic distance as friction
        # Low similarity between primary and alternative = high contradiction = high friction
        semantic_distance = 1.0 - max(0.0, min(1.0, cos_sim))

        # Also check: does the model think the two answers are compatible?
        compat_prompt = (
            f"Are these two answers compatible (i.e., both could be correct)? "
            f"Answer YES or NO.\n\n"
            f"Answer 1: {primary_answer}\n"
            f"Answer 2: {alt_answer}\n\n"
            f"Verdict:"
        )
        compat_result = self.client.generate(compat_prompt, max_tokens=10, logprobs=False)
        compat_text = compat_result.text.strip().upper()
        incompatible = "NO" in compat_text

        # If answers are incompatible AND semantically distant, friction is high
        # If compatible, reduce friction score
        if incompatible:
            score = semantic_distance
        else:
            score = semantic_distance * 0.5

        score = max(0.0, min(1.0, score))
        latency = (time.perf_counter() - start) * 1000

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=0.8,
            metadata={
                "primary_answer": primary_answer[:200],
                "alt_answer": alt_answer[:200],
                "cosine_similarity": round(cos_sim, 4),
                "semantic_distance": round(semantic_distance, 4),
                "compatibility_verdict": compat_text[:20],
                "incompatible": incompatible,
            },
            latency_ms=latency,
        )
