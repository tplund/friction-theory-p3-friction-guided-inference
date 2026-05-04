"""Friction Profile: classify friction TYPE from competing routes distribution.

Instead of using separate proxies (instability, revision) to determine
friction type, we extract it from the SHAPE of the competing routes
distribution within a single generation. This is free - it uses logprobs
we already have.

Three friction profiles:
- SPIKY: few tokens with extreme conflict (high std, high max)
  → "missing info" type → self_critique
- EVEN: consistent mild tøven throughout (low std, moderate mean)
  → "misconception" type → commit (trust round 1)
- SPREAD: many tokens with moderate conflict (high % with 2+)
  → "ambiguity/distraction" type → chain_of_thought

Inspired by Tomas' Behavioral Friction Theory where friction types
(tryghed, kant, mening, besvær) are qualitatively different.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from friktionsllm.ollama_client import GenerationResult


class FrictionType(str, Enum):
    LOW = "low"                     # No significant friction
    SPIKY = "spiky"                 # Few tokens with extreme conflict → missing info
    EVEN = "even"                   # Consistent mild conflict → misconception
    SPREAD = "spread"               # Many tokens with moderate conflict → ambiguity


class FrictionProfile:
    """Extracts friction type from logprobs distribution. Zero extra API calls."""

    def __init__(
        self,
        *,
        friction_threshold: float = 0.20,    # below this = LOW
        spiky_std_threshold: float = 0.45,   # std above this = SPIKY
        spiky_max_threshold: int = 3,        # max competing above this = SPIKY
        spread_pct_threshold: float = 0.15,  # % tokens with 2+ above this = SPREAD
    ):
        self.friction_threshold = friction_threshold
        self.spiky_std_threshold = spiky_std_threshold
        self.spiky_max_threshold = spiky_max_threshold
        self.spread_pct_threshold = spread_pct_threshold

    def classify(self, generation: GenerationResult) -> dict:
        """Classify friction type from a generation's logprobs.

        Returns dict with:
            friction_type: FrictionType
            friction_score: float (overall competing routes score)
            strategy: str (recommended extension strategy)
            profile: dict (detailed stats)
        """
        if not generation.token_logprobs:
            return {
                "friction_type": FrictionType.LOW,
                "friction_score": 0.0,
                "strategy": "commit",
                "profile": {"error": "no logprobs"},
            }

        # Extract per-token competing routes
        competing_per_token = []
        margins_per_token = []

        for td in generation.token_logprobs:
            probs = []
            if hasattr(td, 'logprob'):
                probs.append(math.exp(td.logprob))
            if hasattr(td, 'top_alternatives') and td.top_alternatives:
                for alt in td.top_alternatives:
                    if isinstance(alt, dict):
                        for _tok, lp in alt.items():
                            probs.append(math.exp(lp))

            if not probs:
                continue

            n_competing = sum(1 for p in probs if p > 0.10)
            competing_per_token.append(n_competing)

            probs_sorted = sorted(probs, reverse=True)
            if len(probs_sorted) >= 2:
                margins_per_token.append(probs_sorted[0] - probs_sorted[1])

        if not competing_per_token:
            return {
                "friction_type": FrictionType.LOW,
                "friction_score": 0.0,
                "strategy": "commit",
                "profile": {"error": "no valid tokens"},
            }

        arr = np.array(competing_per_token)
        margins = np.array(margins_per_token) if margins_per_token else np.array([1.0])

        # Compute profile stats
        mean_competing = float(arr.mean())
        max_competing = int(arr.max())
        std_competing = float(arr.std())
        pct_2plus = float((arr >= 2).mean())
        pct_3plus = float((arr >= 3).mean())
        mean_margin = float(margins.mean())
        min_margin = float(margins.min())
        std_margin = float(margins.std())

        # Compute overall friction score (same as CompetingRoutesProxy)
        competing_score = max(0.0, min(1.0, (mean_competing - 1.0) / (2.0 - 1.0)))
        margin_score = max(0.0, min(1.0, 1.0 - mean_margin))
        friction_score = 0.6 * competing_score + 0.4 * margin_score

        profile = {
            "mean_competing": round(mean_competing, 4),
            "max_competing": max_competing,
            "std_competing": round(std_competing, 4),
            "pct_2plus": round(pct_2plus, 4),
            "pct_3plus": round(pct_3plus, 4),
            "mean_margin": round(mean_margin, 4),
            "min_margin": round(min_margin, 4),
            "std_margin": round(std_margin, 4),
            "n_tokens": len(competing_per_token),
        }

        # Classify type
        if friction_score < self.friction_threshold:
            friction_type = FrictionType.LOW
            strategy = "commit"
        elif std_competing >= self.spiky_std_threshold or max_competing >= self.spiky_max_threshold:
            friction_type = FrictionType.SPIKY
            strategy = "self_critique"
        elif pct_2plus >= self.spread_pct_threshold:
            friction_type = FrictionType.SPREAD
            strategy = "chain_of_thought"
        else:
            friction_type = FrictionType.EVEN
            strategy = "commit"

        return {
            "friction_type": friction_type,
            "friction_score": round(friction_score, 4),
            "strategy": strategy,
            "profile": profile,
        }
