import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    # LLM models
    decomposer_model:  str = "claude-haiku-4-5-20251001"
    advocate_model:    str = "claude-haiku-4-5-20251001"
    skeptic_model:     str = "claude-haiku-4-5-20251001"
    arbitrator_model:  str = "claude-sonnet-4-6"          # stronger model for final reasoning
    reporter_model:    str = "claude-haiku-4-5-20251001"

    # Debate settings
    max_rounds:                    int   = 2
    confidence_escalation_threshold: float = 0.4
    conflict_threshold:            float = 0.7

    # Search settings
    search_max_results: int = 3
    search_depth:       str = "basic"   # "basic" | "advanced"

    # API keys (read from environment)
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    tavily_api_key:    str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Module-level singleton — import this everywhere rather than
# instantiating Config() in each file.
config = Config()