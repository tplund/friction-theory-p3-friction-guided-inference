"""Proxy C: Evidence Mismatch.

Measures how well the model's answer aligns with provided evidence/context.
Only applicable when external facts are given (RAG-like scenarios).

If the model "drifts away" from given evidence, that's a signal of high
internal friction - the model's learned patterns are competing with the
provided facts and winning.

In RACE terms: the evidence-based route lost the race to a faster
pattern-matching route.
"""

from __future__ import annotations

import time

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class EvidenceProxy(FrictionProxy):
    """Friction proxy based on answer-evidence alignment."""

    name = "evidence"

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
        """Measure evidence mismatch.

        Args:
            context: Must contain "evidence" key with the reference text.
        """
        start = time.perf_counter()

        if not context or "evidence" not in context:
            return FrictionResult(
                proxy_name=self.name,
                score=0.0,
                confidence=0.0,
                metadata={"skipped": "no evidence provided"},
            )

        evidence = context["evidence"]

        # Generate answer if needed
        if generation is None:
            augmented_prompt = (
                f"Based on the following information, answer the question.\n\n"
                f"Information: {evidence}\n\n"
                f"Question: {prompt}"
            )
            generation = self.client.generate(augmented_prompt, logprobs=False)

        answer = generation.text

        # Method 1: Semantic similarity between answer and evidence
        embedder = self._get_embedder()
        emb = embedder.encode([answer, evidence], convert_to_numpy=True)
        import numpy as np
        cos_sim = float(np.dot(emb[0], emb[1]) / (
            np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-10
        ))

        # Method 2: NLI-style check via the model itself
        nli_prompt = (
            f"Does the following answer follow from the given evidence? "
            f"Answer only YES, NO, or PARTIALLY.\n\n"
            f"Evidence: {evidence}\n\n"
            f"Answer: {answer}\n\n"
            f"Verdict:"
        )
        nli_result = self.client.generate(nli_prompt, max_tokens=10, logprobs=False)
        nli_text = nli_result.text.strip().upper()

        if "YES" in nli_text:
            nli_score = 0.0  # aligned -> no friction
        elif "PARTIAL" in nli_text:
            nli_score = 0.5
        else:
            nli_score = 1.0  # not aligned -> high friction

        # Combined score: 50% semantic similarity, 50% NLI
        semantic_friction = 1.0 - max(0.0, min(1.0, cos_sim))
        score = 0.5 * semantic_friction + 0.5 * nli_score
        score = max(0.0, min(1.0, score))

        latency = (time.perf_counter() - start) * 1000

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=0.8,
            metadata={
                "semantic_similarity": round(cos_sim, 4),
                "semantic_friction": round(semantic_friction, 4),
                "nli_verdict": nli_text[:20],
                "nli_score": nli_score,
                "answer_preview": answer[:200],
            },
            latency_ms=latency,
        )
