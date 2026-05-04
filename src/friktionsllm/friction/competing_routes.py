"""Proxy: Competing Routes (RACE-inspired).

Measures how many tokens have a realistic chance of winning at each
decision point. This is closer to the RACE model than pure entropy:
only routes strong enough to reach the threshold compete, the rest
are irrelevant noise.

Strongest proxy found so far: -0.423 correlation with correctness (p<0.0001).
Free: piggybacks on logprobs from the generation call.
"""

from __future__ import annotations

import math

import numpy as np

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class CompetingRoutesProxy(FrictionProxy):
    """Friction proxy based on number of competing token candidates."""

    name = "competing_routes"

    def __init__(
        self,
        client: OllamaClient,
        *,
        prob_threshold: float = 0.10,
        # Calibration: from competing_routes.jsonl analysis
        # close_call mean=1.35, insufficient_info mean=1.19
        # range is roughly 1.0-2.0, with most values 1.1-1.5
        calibration_min: float = 1.0,  # below this = no friction
        calibration_max: float = 2.0,  # above this = max friction
    ):
        super().__init__(client)
        self.prob_threshold = prob_threshold
        self.calibration_min = calibration_min
        self.calibration_max = calibration_max

    def measure(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
        context: dict | None = None,
    ) -> FrictionResult:
        """Measure competing routes friction.

        If a generation result with logprobs is already available, reuses it.
        Otherwise, generates a new response.
        """
        if generation is None:
            generation = self.client.generate(prompt, logprobs=True)

        if not generation.token_logprobs:
            return FrictionResult(
                proxy_name=self.name,
                score=0.5,
                confidence=0.0,
                metadata={"error": "no logprobs available"},
                latency_ms=generation.latency_ms,
            )

        # Count competing routes at each token position
        competing_counts = []
        margins = []

        for token_data in generation.token_logprobs:
            probs = []

            # Chosen token probability
            if hasattr(token_data, "logprob"):
                probs.append(math.exp(token_data.logprob))

            # Alternative token probabilities
            if hasattr(token_data, "top_alternatives") and token_data.top_alternatives:
                for alt in token_data.top_alternatives:
                    if isinstance(alt, dict):
                        for _token, lp in alt.items():
                            probs.append(math.exp(lp))

            if not probs:
                continue

            # Count how many tokens exceed the probability threshold
            n_competing = sum(1 for p in probs if p > self.prob_threshold)
            competing_counts.append(n_competing)

            # Margin between top-1 and top-2
            probs_sorted = sorted(probs, reverse=True)
            if len(probs_sorted) >= 2:
                margins.append(probs_sorted[0] - probs_sorted[1])

        if not competing_counts:
            return FrictionResult(
                proxy_name=self.name,
                score=0.5,
                confidence=0.0,
                metadata={"error": "no valid token data"},
                latency_ms=generation.latency_ms,
            )

        mean_competing = float(np.mean(competing_counts))
        max_competing = int(max(competing_counts))
        mean_margin = float(np.mean(margins)) if margins else 1.0

        # Normalize to 0-1 score
        # More competing routes = higher friction
        competing_score = (mean_competing - self.calibration_min) / (
            self.calibration_max - self.calibration_min
        )
        competing_score = max(0.0, min(1.0, competing_score))

        # Lower margin = higher friction
        margin_score = 1.0 - mean_margin  # margin is 0-1 (probability space)
        margin_score = max(0.0, min(1.0, margin_score))

        # Combined: 60% competing routes, 40% margin
        score = 0.6 * competing_score + 0.4 * margin_score

        return FrictionResult(
            proxy_name=self.name,
            score=round(score, 4),
            confidence=1.0 if len(competing_counts) >= 5 else len(competing_counts) / 5,
            metadata={
                "mean_competing": round(mean_competing, 4),
                "max_competing": max_competing,
                "mean_margin": round(mean_margin, 4),
                "min_margin": round(min(margins), 4) if margins else None,
                "n_tokens": len(competing_counts),
                "response_text": generation.text[:200],
            },
            latency_ms=generation.latency_ms,
        )
