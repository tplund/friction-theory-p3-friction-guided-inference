"""Configuration for FriktionsLLM experiments."""

from pydantic import BaseModel, Field


# --- Model Configuration ---

class ModelConfig(BaseModel):
    """Configuration for a local Ollama model."""
    name: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 128  # keep low with logprobs to avoid OOM on laptop (512 for no-logprobs)
    top_logprobs: int = 5  # number of top token alternatives (20 causes OOM on laptop)


# --- Friction Proxy Configuration ---

class FrictionConfig(BaseModel):
    """Configuration for friction measurement."""
    # Which proxies to use for fast (every-round) measurement
    fast_proxies: list[str] = Field(default=["entropy"])
    # Which proxies to use for full measurement (final decision)
    full_proxies: list[str] = Field(
        default=["entropy", "instability", "revision"]
    )
    # Number of samples for instability proxy
    instability_samples: int = 3
    instability_temperature: float = 0.3
    # Semantic similarity model for instability/contradiction
    embedding_model: str = "all-MiniLM-L6-v2"


# --- Steering Configuration ---

class SteeringConfig(BaseModel):
    """Configuration for the friction-steered inference engine."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    friction: FrictionConfig = Field(default_factory=FrictionConfig)
    max_rounds: int = 3
    commit_threshold: float = 0.025  # calibrated from gemma3:4b entropy distribution (p50)
    abstain_threshold: float = 0.04  # calibrated from gemma3:4b entropy distribution (~p90)
    extend_strategy: str = "chain_of_thought"  # chain_of_thought | self_critique | majority


# --- Evaluation Configuration ---

class EvalConfig(BaseModel):
    """Configuration for evaluation pipeline."""
    judge_model: str = "claude-sonnet-4-20250514"
    judge_scoring_dimensions: list[str] = Field(default=[
        "correctness",
        "hallucination",
        "uncertainty_calibration",
        "abstention_quality",
        "helpfulness",
    ])
    human_control_size: int = 30
    judge_repetitions: int = 2  # run each judgment twice for reliability
    randomize_order: bool = True
    blind_conditions: bool = True


# --- Default Configs ---

DEFAULT_MODEL = ModelConfig()
DEFAULT_FRICTION = FrictionConfig()
DEFAULT_STEERING = SteeringConfig()
DEFAULT_EVAL = EvalConfig()
