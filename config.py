import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── LLM Provider Configuration ──────────────────────────────────────────
    # Options: "anthropic", "openai", "groq", "ollama", "mock"
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "anthropic").lower()
    )

    # Agent Model Assignments (Tiered for Cost/Latency Efficiency)
    # Lightweight agents
    claim_analyzer_model: str = field(
        default_factory=lambda: os.environ.get("CLAIM_ANALYZER_MODEL", "claude-haiku-4-5-20251001")
    )
    planner_model: str = field(
        default_factory=lambda: os.environ.get("PLANNER_MODEL", "claude-haiku-4-5-20251001")
    )
    source_credibility_model: str = field(
        default_factory=lambda: os.environ.get("SOURCE_CREDIBILITY_MODEL", "claude-haiku-4-5-20251001")
    )
    advocate_model: str = field(
        default_factory=lambda: os.environ.get("ADVOCATE_MODEL", "claude-haiku-4-5-20251001")
    )
    skeptic_model: str = field(
        default_factory=lambda: os.environ.get("SKEPTIC_MODEL", "claude-haiku-4-5-20251001")
    )
    neutral_model: str = field(
        default_factory=lambda: os.environ.get("NEUTRAL_MODEL", "claude-haiku-4-5-20251001")
    )
    critic_model: str = field(
        default_factory=lambda: os.environ.get("CRITIC_MODEL", "claude-haiku-4-5-20251001")
    )
    
    # Reasoning & Synthesis agents (stronger models)
    judge_model: str = field(
        default_factory=lambda: os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
    )
    final_verifier_model: str = field(
        default_factory=lambda: os.environ.get("FINAL_VERIFIER_MODEL", "claude-sonnet-4-6")
    )

    # ── Autonomous Multi-Agent Loop Thresholds ──────────────────────────────
    max_debate_rounds: int = 3
    max_research_rounds: int = 2
    min_evidence_count: int = 3
    conflict_threshold: float = 0.35          # Disagreement > threshold triggers another debate round
    low_confidence_threshold: float = 0.55    # Confidence < threshold triggers research or debate
    source_credibility_threshold: float = 0.60 # Minimum score for high-credibility classification

    # ── DeGroot Consensus Settings ──────────────────────────────────────────
    convergence_threshold: float = 1e-4
    max_consensus_iterations: int = 50

    # ── Search Settings ─────────────────────────────────────────────────────
    search_max_results: int = 4
    search_depth: str = "basic"  # "basic" | "advanced"
    offline_fallback: bool = True # If Tavily key missing or fails, use intelligent local fallback

    # ── API Keys ────────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY")
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY")
    )

    # ── Server & Database ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", "sqlite:///veritas.db")
    )


# Module-level singleton
config = Config()
