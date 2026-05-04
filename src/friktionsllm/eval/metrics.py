"""Metrics for evaluating friction-steered inference.

Computes key metrics that test the friction hypothesis:
- Does friction score predict errors?
- Does steering reduce hallucinations?
- Does abstention happen when it should?
- Do extra rounds help where friction is high?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ExperimentMetrics:
    """Computed metrics for one experimental condition."""
    condition: str  # "vanilla" or "friction"
    n_questions: int

    # Core performance
    accuracy: float  # fraction correct
    hallucination_rate: float  # fraction with hallucinations

    # Abstention behavior
    abstention_rate: float  # fraction where model abstained
    appropriate_abstention_rate: float  # of abstentions, how many were correct
    false_abstention_rate: float  # of answerable questions, how many incorrectly abstained

    # Friction-specific
    mean_friction: float
    friction_error_correlation: float  # Spearman correlation
    friction_error_p_value: float
    extra_round_improvement_rate: float  # fraction where extra rounds helped

    # Latency
    mean_latency_ms: float
    median_latency_ms: float

    # Judge scores (means)
    mean_correctness: float
    mean_hallucination_score: float
    mean_calibration: float
    mean_helpfulness: float


def compute_metrics(
    results: list[dict],
    condition: str,
) -> ExperimentMetrics:
    """Compute all metrics from experiment results.

    Args:
        results: List of dicts with keys:
            - correct: bool
            - hallucinated: bool
            - abstained: bool
            - should_abstain: bool (ground truth)
            - friction_score: float
            - rounds_used: int
            - latency_ms: float
            - judge_scores: dict with correctness, hallucination, etc.
            - answer_improved: bool (if rounds > 1, did answer get better?)
        condition: Name of this condition.

    Returns:
        ExperimentMetrics with all computed values.
    """
    n = len(results)
    if n == 0:
        return _empty_metrics(condition)

    # Core performance
    correct = [r["correct"] for r in results]
    hallucinated = [r.get("hallucinated", False) for r in results]
    accuracy = sum(correct) / n
    hallucination_rate = sum(hallucinated) / n

    # Abstention
    abstained = [r.get("abstained", False) for r in results]
    should_abstain = [r.get("should_abstain", False) for r in results]
    abstention_rate = sum(abstained) / n

    # Appropriate abstentions: abstained AND should have
    appropriate = sum(a and s for a, s in zip(abstained, should_abstain))
    total_abstained = sum(abstained)
    appropriate_abstention_rate = appropriate / total_abstained if total_abstained > 0 else 0.0

    # False abstentions: abstained but should NOT have
    answerable = [not s for s in should_abstain]
    false_abstain = sum(a and ans for a, ans in zip(abstained, answerable))
    total_answerable = sum(answerable)
    false_abstention_rate = false_abstain / total_answerable if total_answerable > 0 else 0.0

    # Friction-error correlation
    friction_scores = [r["friction_score"] for r in results]
    errors = [0 if c else 1 for c in correct]
    if len(set(friction_scores)) > 1 and len(set(errors)) > 1:
        corr, p_val = stats.spearmanr(friction_scores, errors)
    else:
        corr, p_val = 0.0, 1.0

    # Extra round improvement
    multi_round = [r for r in results if r.get("rounds_used", 1) > 1]
    if multi_round:
        improved = sum(r.get("answer_improved", False) for r in multi_round)
        extra_round_improvement = improved / len(multi_round)
    else:
        extra_round_improvement = 0.0

    # Latency
    latencies = [r["latency_ms"] for r in results]
    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))

    # Judge scores
    judge_scores = [r.get("judge_scores", {}) for r in results]
    mean_correctness = _safe_mean([s.get("correctness", 0) for s in judge_scores])
    mean_hallucination_score = _safe_mean([s.get("hallucination", 0) for s in judge_scores])
    mean_calibration = _safe_mean([s.get("uncertainty_calibration", 0) for s in judge_scores])
    mean_helpfulness = _safe_mean([s.get("helpfulness", 0) for s in judge_scores])

    return ExperimentMetrics(
        condition=condition,
        n_questions=n,
        accuracy=round(accuracy, 4),
        hallucination_rate=round(hallucination_rate, 4),
        abstention_rate=round(abstention_rate, 4),
        appropriate_abstention_rate=round(appropriate_abstention_rate, 4),
        false_abstention_rate=round(false_abstention_rate, 4),
        mean_friction=round(float(np.mean(friction_scores)), 4),
        friction_error_correlation=round(float(corr), 4),
        friction_error_p_value=round(float(p_val), 6),
        extra_round_improvement_rate=round(extra_round_improvement, 4),
        mean_latency_ms=round(mean_latency, 1),
        median_latency_ms=round(median_latency, 1),
        mean_correctness=round(mean_correctness, 2),
        mean_hallucination_score=round(mean_hallucination_score, 2),
        mean_calibration=round(mean_calibration, 2),
        mean_helpfulness=round(mean_helpfulness, 2),
    )


def compare_conditions(
    vanilla_metrics: ExperimentMetrics,
    friction_metrics: ExperimentMetrics,
) -> dict:
    """Compare vanilla vs friction-steered metrics and compute deltas."""
    deltas = {}
    for field in [
        "accuracy", "hallucination_rate", "abstention_rate",
        "mean_correctness", "mean_hallucination_score",
        "mean_calibration", "mean_helpfulness",
    ]:
        v = getattr(vanilla_metrics, field)
        f = getattr(friction_metrics, field)
        deltas[field] = round(f - v, 4)
        deltas[f"{field}_pct_change"] = round((f - v) / (v + 1e-10) * 100, 1)

    deltas["latency_overhead_ms"] = round(
        friction_metrics.mean_latency_ms - vanilla_metrics.mean_latency_ms, 1
    )
    deltas["latency_overhead_pct"] = round(
        (friction_metrics.mean_latency_ms - vanilla_metrics.mean_latency_ms)
        / (vanilla_metrics.mean_latency_ms + 1e-10) * 100, 1
    )

    return deltas


def _safe_mean(values: list) -> float:
    values = [v for v in values if v is not None and v > 0]
    return float(np.mean(values)) if values else 0.0


def _empty_metrics(condition: str) -> ExperimentMetrics:
    return ExperimentMetrics(
        condition=condition, n_questions=0, accuracy=0, hallucination_rate=0,
        abstention_rate=0, appropriate_abstention_rate=0, false_abstention_rate=0,
        mean_friction=0, friction_error_correlation=0, friction_error_p_value=1,
        extra_round_improvement_rate=0, mean_latency_ms=0, median_latency_ms=0,
        mean_correctness=0, mean_hallucination_score=0, mean_calibration=0,
        mean_helpfulness=0,
    )
