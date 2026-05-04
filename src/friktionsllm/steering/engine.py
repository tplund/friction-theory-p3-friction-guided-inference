"""Friction-Steered Inference Engine.

The core loop that uses friction signals to decide whether to:
- COMMIT: accept the current answer (friction is low enough)
- EXTEND: give the model more "thinking time" (friction is high)
- ABSTAIN: decline to answer confidently (friction remains high after max rounds)

This implements the RACE model applied to LLM inference:
- Pressure = internal friction (uncertainty, conflict)
- Race duration = number of computation rounds
- Threshold = commit/abstain thresholds
- Winner = the response that finally passes the threshold

The key insight: under high pressure (high friction), more race time
(more rounds) may produce a better winner (answer). But if the race
can't be resolved, the system should abstain rather than commit to
an unreliable answer.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from friktionsllm.config import SteeringConfig, DEFAULT_STEERING
from friktionsllm.friction.base import FrictionResult
from friktionsllm.friction.composite import CompositeFriction
from friktionsllm.ollama_client import OllamaClient, GenerationResult


class Decision(str, Enum):
    COMMIT = "commit"
    EXTEND = "extend"
    ABSTAIN = "abstain"


class RoundResult(BaseModel):
    """Result of a single steering round."""
    round_number: int
    response: str
    friction: FrictionResult
    decision: Decision
    strategy_used: str | None = None


class SteeringResult(BaseModel):
    """Full result of the friction-steered inference process."""
    prompt: str
    final_answer: str
    decision: Literal["commit", "abstain"]
    rounds_used: int
    rounds: list[RoundResult]
    total_latency_ms: float
    model: str
    friction_trajectory: list[float] = Field(default_factory=list)

    @property
    def friction_decreased(self) -> bool:
        """Did friction decrease across rounds?"""
        if len(self.friction_trajectory) < 2:
            return False
        return self.friction_trajectory[-1] < self.friction_trajectory[0]

    @property
    def answer_changed(self) -> bool:
        """Did the answer change across rounds?"""
        if len(self.rounds) < 2:
            return False
        return self.rounds[0].response != self.rounds[-1].response


class SteeringEngine:
    """The friction-steered inference engine.

    Usage:
        engine = SteeringEngine()
        result = engine.run("What is the capital of France?")
        print(result.final_answer)
        print(result.decision)  # "commit" or "abstain"
    """

    def __init__(self, config: SteeringConfig | None = None):
        self.config = config or DEFAULT_STEERING
        self.client = OllamaClient(self.config.model)
        self.friction = CompositeFriction(
            self.client,
            proxy_names=self.config.friction.fast_proxies,
        )

    def run(
        self,
        prompt: str,
        *,
        context: dict | None = None,
        use_full_proxies: bool = False,
    ) -> SteeringResult:
        """Run friction-steered inference on a prompt.

        Args:
            prompt: The input question/prompt.
            context: Optional context for evidence-based friction.
            use_full_proxies: If True, use all proxies (slower but more accurate).

        Returns:
            SteeringResult with the final answer, decision, and full trajectory.
        """
        import time
        start = time.perf_counter()

        rounds: list[RoundResult] = []
        friction_trajectory: list[float] = []
        current_prompt = prompt

        for round_num in range(1, self.config.max_rounds + 1):
            # Generate response
            # Only use logprobs on round 1 (for friction measurement)
            # Extension rounds use longer prompts that can OOM with logprobs
            use_logprobs = (round_num == 1)
            generation = self.client.generate(
                current_prompt, logprobs=use_logprobs, max_tokens=256 if not use_logprobs else None
            )

            # Measure friction
            if use_logprobs:
                if use_full_proxies and round_num == self.config.max_rounds:
                    full_friction = CompositeFriction(
                        self.client,
                        proxy_names=self.config.friction.full_proxies,
                    )
                    friction_result = full_friction.measure(
                        prompt, generation=generation, context=context
                    )
                else:
                    friction_result = self.friction.measure_fast(
                        prompt, generation=generation
                    )
            else:
                # No logprobs in extension rounds - re-measure original prompt
                # This gives us friction on the original question with fresh context
                regen = self.client.generate(prompt, logprobs=True, max_tokens=64)
                friction_result = self.friction.measure_fast(
                    prompt, generation=regen
                )

            friction_score = friction_result.score
            friction_trajectory.append(friction_score)

            # Decide: commit, extend, or abstain
            # Uses friction profile to determine if abstention is appropriate
            is_last_round = round_num >= self.config.max_rounds

            if friction_score < self.config.commit_threshold:
                decision = Decision.COMMIT
            elif is_last_round:
                if friction_score > self.config.abstain_threshold:
                    # Check friction profile before abstaining
                    # Only abstain if the profile suggests missing information
                    # (high instability / SPIKY with high std)
                    # NOT for close_call/misconceptions where model has an answer
                    from friktionsllm.friction.cr_utils import compute_cr_from_logprobs
                    if use_logprobs and generation.token_logprobs:
                        cr_stats = compute_cr_from_logprobs(generation.token_logprobs)
                        # Low std + high CR = consistent uncertainty (close_call)
                        # → model knows the answer, just tøver. Commit, don't abstain.
                        if cr_stats.std_cr < 0.3 and cr_stats.pct_high_friction < 0.4:
                            decision = Decision.COMMIT  # EVEN profile: commit anyway
                        else:
                            decision = Decision.ABSTAIN  # SPIKY/SPREAD: genuine uncertainty
                    else:
                        decision = Decision.ABSTAIN  # no logprobs, default to abstain
                else:
                    decision = Decision.COMMIT  # commit with moderate friction
            else:
                decision = Decision.EXTEND

            strategy = None
            if decision == Decision.EXTEND:
                strategy = self.config.extend_strategy
                current_prompt = self._apply_extension_strategy(
                    prompt, generation.text, strategy
                )

            rounds.append(RoundResult(
                round_number=round_num,
                response=generation.text,
                friction=friction_result,
                decision=decision,
                strategy_used=strategy,
            ))

            # If committing or abstaining, we're done
            if decision in (Decision.COMMIT, Decision.ABSTAIN):
                break

        # Determine final answer
        final_round = rounds[-1]
        if final_round.decision == Decision.ABSTAIN:
            final_answer = (
                f"I'm not confident enough to give a reliable answer to this question. "
                f"My uncertainty remains high after {len(rounds)} reasoning attempts. "
                f"The most likely answer I can offer (with low confidence) is: "
                f"{final_round.response}"
            )
            decision_str = "abstain"
        else:
            final_answer = final_round.response
            decision_str = "commit"

        total_latency = (time.perf_counter() - start) * 1000

        return SteeringResult(
            prompt=prompt,
            final_answer=final_answer,
            decision=decision_str,
            rounds_used=len(rounds),
            rounds=rounds,
            total_latency_ms=total_latency,
            model=self.config.model.name,
            friction_trajectory=friction_trajectory,
        )

    def run_vanilla(self, prompt: str) -> SteeringResult:
        """Run vanilla (non-steered) inference for baseline comparison.

        Single pass, no friction measurement, no extension.
        """
        import time
        start = time.perf_counter()

        generation = self.client.generate(prompt, logprobs=True)

        # Still measure friction for comparison, but don't act on it
        friction_result = self.friction.measure_fast(prompt, generation=generation)

        total_latency = (time.perf_counter() - start) * 1000

        round_result = RoundResult(
            round_number=1,
            response=generation.text,
            friction=friction_result,
            decision=Decision.COMMIT,
        )

        return SteeringResult(
            prompt=prompt,
            final_answer=generation.text,
            decision="commit",
            rounds_used=1,
            rounds=[round_result],
            total_latency_ms=total_latency,
            model=self.config.model.name,
            friction_trajectory=[friction_result.score],
        )

    def _apply_extension_strategy(
        self,
        original_prompt: str,
        previous_answer: str,
        strategy: str,
    ) -> str:
        """Build an extended prompt based on the chosen strategy.

        All prompts include output hygiene instruction to prevent
        excessively long/verbose responses. Data shows CoT+hygiene
        scores nearly as well as vanilla (4.3 vs 4.6 GPT-4) while
        raw CoT scores 2.1 - the hygiene instruction is critical.
        """
        HYGIENE = " Answer precisely and concisely. If you are unsure about something, say so clearly. Avoid unnecessary repetition."

        if strategy == "chain_of_thought":
            return (
                f"Think through this carefully, step by step, before giving "
                f"your final answer.\n\n"
                f"Question: {original_prompt}\n\n"
                f"Step-by-step reasoning:" + HYGIENE
            )
        elif strategy == "self_critique":
            # Truncate previous answer to avoid OOM on small models
            truncated = previous_answer[:500] if len(previous_answer) > 500 else previous_answer
            return (
                f"I asked: {original_prompt}\n\n"
                f"An initial answer was: {truncated}\n\n"
                f"Please critically review this answer. Check for errors, "
                f"unsupported claims, or missing nuance. Then provide a "
                f"corrected or improved answer." + HYGIENE
            )
        elif strategy == "majority":
            # For majority, we don't change the prompt - the engine
            # would need to generate multiple samples and pick consensus.
            # For now, fall back to chain_of_thought.
            return (
                f"Think through this carefully, step by step, before giving "
                f"your final answer.\n\n"
                f"Question: {original_prompt}\n\n"
                f"Step-by-step reasoning:" + HYGIENE
            )
        else:
            # Default: chain of thought
            return (
                f"Think step by step.\n\n"
                f"Question: {original_prompt}"
            )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
