"""Composite Friction Score.

Combines multiple friction proxies into a single weighted score.
The weights are initial guesses and should be calibrated through
Experiment 1 (proxy discovery).

In RACE terms: the composite score represents the total "pressure"
in the system across all measurable dimensions.
"""

from __future__ import annotations

import time

from friktionsllm.friction.base import FrictionProxy, FrictionResult
from friktionsllm.friction.entropy import EntropyProxy
from friktionsllm.friction.competing_routes import CompetingRoutesProxy
from friktionsllm.friction.instability import InstabilityProxy
from friktionsllm.friction.evidence import EvidenceProxy
from friktionsllm.friction.contradiction import ContradictionProxy
from friktionsllm.friction.revision import RevisionProxy
from friktionsllm.ollama_client import OllamaClient, GenerationResult


# Default weights for the composite score
DEFAULT_WEIGHTS = {
    "entropy": 0.30,
    "instability": 0.25,
    "evidence": 0.15,
    "contradiction": 0.15,
    "revision": 0.15,
}


class CompositeFriction:
    """Combines multiple friction proxies into one score.

    Weights are normalized at runtime so that only the active proxies
    contribute. For example, if evidence proxy is skipped (no context),
    the remaining proxies are renormalized to sum to 1.0.
    """

    def __init__(
        self,
        client: OllamaClient,
        *,
        weights: dict[str, float] | None = None,
        proxy_names: list[str] | None = None,
    ):
        self.client = client
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Initialize requested proxies
        all_proxies = {
            "entropy": EntropyProxy,
            "competing_routes": CompetingRoutesProxy,
            "instability": InstabilityProxy,
            "evidence": EvidenceProxy,
            "contradiction": ContradictionProxy,
            "revision": RevisionProxy,
        }

        names = proxy_names or list(self.weights.keys())
        self.proxies: dict[str, FrictionProxy] = {}
        for name in names:
            if name in all_proxies:
                self.proxies[name] = all_proxies[name](client)

    def measure(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
        context: dict | None = None,
    ) -> FrictionResult:
        """Measure composite friction across all active proxies.

        Returns a single FrictionResult with the weighted composite score
        and individual proxy results in metadata.
        """
        start = time.perf_counter()

        # First, generate once and reuse across proxies
        if generation is None:
            generation = self.client.generate(prompt, logprobs=True)

        # Measure each proxy
        proxy_results: dict[str, FrictionResult] = {}
        active_weights: dict[str, float] = {}

        for name, proxy in self.proxies.items():
            result = proxy.measure(
                prompt, generation=generation, context=context
            )
            # Skip proxies that returned confidence 0 (e.g., evidence with no context)
            if result.confidence > 0:
                proxy_results[name] = result
                active_weights[name] = self.weights.get(name, 0.0)

        # Normalize weights to sum to 1.0
        total_weight = sum(active_weights.values())
        if total_weight > 0:
            normalized = {k: v / total_weight for k, v in active_weights.items()}
        else:
            return FrictionResult(
                proxy_name="composite",
                score=0.5,
                confidence=0.0,
                metadata={"error": "no active proxies"},
            )

        # Weighted composite score
        composite_score = sum(
            normalized[name] * proxy_results[name].score
            for name in proxy_results
        )
        composite_score = max(0.0, min(1.0, composite_score))

        # Average confidence
        avg_confidence = sum(
            r.confidence for r in proxy_results.values()
        ) / len(proxy_results)

        latency = (time.perf_counter() - start) * 1000

        return FrictionResult(
            proxy_name="composite",
            score=round(composite_score, 4),
            confidence=round(avg_confidence, 4),
            metadata={
                "proxy_scores": {
                    name: {
                        "score": r.score,
                        "confidence": r.confidence,
                        "latency_ms": round(r.latency_ms, 1),
                    }
                    for name, r in proxy_results.items()
                },
                "weights_used": {k: round(v, 3) for k, v in normalized.items()},
                "response_text": generation.text[:200],
            },
            latency_ms=latency,
        )

    def measure_fast(
        self,
        prompt: str,
        *,
        generation: GenerationResult | None = None,
    ) -> FrictionResult:
        """Fast friction measurement using competing routes proxy.

        Used during steering rounds where speed matters.
        Competing routes is free (piggybacks on logprobs) and the
        strongest proxy found (-0.423 corr with correctness).
        """
        if "competing_routes" not in self.proxies:
            self.proxies["competing_routes"] = CompetingRoutesProxy(self.client)

        if generation is None:
            generation = self.client.generate(prompt, logprobs=True)

        return self.proxies["competing_routes"].measure(prompt, generation=generation)
