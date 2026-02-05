"""Configuration for MERGE-AI experiments."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "merge_ai" / "data"
CONFLICTS_DIR = DATA_DIR / "conflicts"
RESULTS_DIR = DATA_DIR / "results"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    provider: str  # "openai", "anthropic", or "openrouter"
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096

    @property
    def cost_per_1k_input(self) -> float:
        """Estimated cost per 1K input tokens."""
        costs = {
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "claude-sonnet-4-20250514": 0.003,
            "claude-3-5-haiku-20241022": 0.0008,
            # OpenRouter model names
            "anthropic/claude-sonnet-4": 0.003,
            "anthropic/claude-3.5-sonnet": 0.003,
            "anthropic/claude-3.5-haiku": 0.0008,
            "openai/gpt-4o": 0.0025,
            "openai/gpt-4o-mini": 0.00015,
        }
        return costs.get(self.model, 0.003)

    @property
    def cost_per_1k_output(self) -> float:
        """Estimated cost per 1K output tokens."""
        costs = {
            "gpt-4o": 0.01,
            "gpt-4o-mini": 0.0006,
            "claude-sonnet-4-20250514": 0.015,
            "claude-3-5-haiku-20241022": 0.004,
            # OpenRouter model names
            "anthropic/claude-sonnet-4": 0.015,
            "anthropic/claude-3.5-sonnet": 0.015,
            "anthropic/claude-3.5-haiku": 0.004,
            "openai/gpt-4o": 0.01,
            "openai/gpt-4o-mini": 0.0006,
        }
        return costs.get(self.model, 0.015)


# Default model configurations
GPT_4O = ModelConfig(provider="openai", model="gpt-4o", temperature=0.1)
CLAUDE_SONNET = ModelConfig(
    provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.1
)
CLAUDE_HAIKU = ModelConfig(
    provider="anthropic", model="claude-3-5-haiku-20241022", temperature=0.1
)

# OpenRouter model configurations - Best models
OPENROUTER_SONNET = ModelConfig(
    provider="openrouter", model="anthropic/claude-sonnet-4", temperature=0.1
)
OPENROUTER_GPT4O = ModelConfig(
    provider="openrouter", model="openai/gpt-4o", temperature=0.1
)
# Smaller/cheaper models for validation
OPENROUTER_HAIKU = ModelConfig(
    provider="openrouter", model="anthropic/claude-3.5-haiku", temperature=0.1
)


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    # Sample size
    sample_size: int = 30

    # Retry settings
    max_retries: int = 2

    # Budget limits
    max_cost_usd: float = 100.0

    # Models - Using best Claude and GPT via OpenRouter
    primary_model: ModelConfig = field(default_factory=lambda: OPENROUTER_SONNET)
    secondary_model: ModelConfig = field(default_factory=lambda: OPENROUTER_GPT4O)

    # Output settings
    save_intermediate: bool = True
    verbose: bool = True


# Default experiment configuration
DEFAULT_CONFIG = ExperimentConfig()


def get_config() -> ExperimentConfig:
    """Get the current experiment configuration."""
    return DEFAULT_CONFIG


def validate_api_keys() -> tuple[bool, list[str]]:
    """Validate that required API keys are set.

    Returns:
        Tuple of (all_valid, list of missing keys)
    """
    missing = []
    # Check for at least one valid API key
    has_any = False
    if OPENROUTER_API_KEY:
        has_any = True
    if OPENAI_API_KEY:
        has_any = True
    if ANTHROPIC_API_KEY:
        has_any = True

    if not has_any:
        missing.append("OPENROUTER_API_KEY (or OPENAI_API_KEY or ANTHROPIC_API_KEY)")

    return len(missing) == 0, missing
