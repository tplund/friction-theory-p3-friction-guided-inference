"""Proxy A: Distribution Uncertainty.

Measures how spread out the model's token probability distribution is.
This is the cheapest proxy - it piggybacks on the generation call's logprobs.

High entropy = model is uncertain about which token to choose.
Low margin = top two candidates are close -> high internal "conflict".

In RACE terms: high entropy means many competing routes, none clearly winning.
"""

from __future__ import annotations

import numpy as np

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class EntropyProxy(FrictionProxy):
    """Friction proxy based on token-level entropy and margin."""

    name = "entropy"

    def __init__(
        self,
        client: OllamaClient,
        *,
        entropy_high_threshold: float = 3.0,
        calibration_max_entropy: float = 5.0,
        calibration_min_margin: float = 0.0,
        calibration_max_margin: float = 5.0,
    ):
        super().__init__(client)
        self.entropy_high_threshold = entropy_high_threshold
        self.calibration_max_entropy = calibration_max_entropy
        self.calibration_min_margin = calibration_min_margin
        self.calibration_max_margin = calibration_max_margin

    def measure(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
        context: dict | None = None,
    ) -> FrictionResult:
        """Measure entropy-based friction.

        If a generation result with logprobs is already available, reuses it.
        Otherwise, generates a new response.
        """
        # Generate if needed
        if generation is None:
            generation = self.client.generate(prompt, logprobs=True)

        if not generation.token_logprobs:
            return FrictionResult(
                proxy_name=self.name,
                score=0.5,  # unknown -> moderate friction
                confidence=0.0,
                metadata={"error": "no logprobs available"},
                latency_ms=generation.latency_ms,
            )

        # Compute per-token metrics
        entropies = [t.entropy for t in generation.token_logprobs]
        margins = [t.margin for t in generation.token_logprobs
                    if t.margin != float("inf")]

        mean_entropy = float(np.mean(entropies))
        max_entropy = float(np.max(entropies))
        high_entropy_ratio = sum(
            1 for e in entropies if e > self.entropy_high_threshold
        ) / len(entropies)
        mean_margin = float(np.mean(margins)) if margins else float("inf")

        # Compute normalized friction score (0-1)
        # Higher entropy and lower margin both increase friction
        entropy_score = min(mean_entropy / self.calibration_max_entropy, 1.0)
        margin_score = 1.0 - min(
            mean_margin / self.calibration_max_margin, 1.0
        ) if mean_margin != float("inf") else 0.0

        # Weighted combination: entropy contributes 60%, margin 40%
        score = 0.6 * entropy_score + 0.4 * margin_score

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=1.0 if len(entropies) >= 5 else len(entropies) / 5,
            metadata={
                "mean_entropy": round(mean_entropy, 4),
                "max_entropy": round(max_entropy, 4),
                "high_entropy_ratio": round(high_entropy_ratio, 4),
                "mean_margin": round(mean_margin, 4) if mean_margin != float("inf") else None,
                "num_tokens": len(entropies),
                "response_text": generation.text[:200],
            },
            latency_ms=generation.latency_ms,
        )
