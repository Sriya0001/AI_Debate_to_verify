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

    # FC-MAD Agent Models
    debater_model: str = field(
        default_factory=lambda: os.environ.get("DEBATER_MODEL", "claude-haiku-4-5-20251001")
    )
    summarizer_model: str = field(
        default_factory=lambda: os.environ.get("SUMMARIZER_MODEL", "claude-haiku-4-5-20251001")
    )
    output_agent_model: str = field(
        default_factory=lambda: os.environ.get("OUTPUT_AGENT_MODEL", "claude-sonnet-4-6")
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

    # ── Robustness & Retries ────────────────────────────────────────────────
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_RETRIES", "2"))
    )

    # ── FC-MAD Mode Settings (docs/FCMAD_MATH.md) ───────────────────────────
    num_debaters: int = field(
        default_factory=lambda: int(os.environ.get("FCMAD_NUM_DEBATERS", "3"))
    )
    alpha: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_ALPHA", "0.7"))
    )
    epsilon: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_EPSILON", "0.10"))
    )
    max_rounds_fcmad: int = field(
        default_factory=lambda: int(os.environ.get("FCMAD_MAX_ROUNDS", "4"))
    )
    samples_per_debater: int = field(
        default_factory=lambda: int(os.environ.get("FCMAD_SAMPLES_PER_DEBATER", "3"))
    )
    debater_temperature: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_DEBATER_TEMPERATURE", "0.7"))
    )
    abstain_target_accuracy: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_ABSTAIN_TARGET_ACCURACY", "0.90"))
    )
    calibration_enabled: bool = field(
        default_factory=lambda: os.environ.get("FCMAD_CALIBRATION", "false").lower() in ("true", "1", "yes")
    )
    abstain_threshold: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_ABSTAIN_THRESHOLD", "0.60"))
    )
    temperature_scaling_T: float = field(
        default_factory=lambda: float(os.environ.get("FCMAD_TEMPERATURE_T", "1.0"))
    )
    summarizer_token_cap: int = field(
        default_factory=lambda: int(os.environ.get("FCMAD_SUMMARIZER_TOKEN_CAP", "300"))
    )

    # ── FC-MAD Ablation Toggles (docs/FCMAD_MATH.md Section 10) ─────────────
    fcmad_equal_weights: bool = field(
        default_factory=lambda: os.environ.get("FCMAD_EQUAL_WEIGHTS", "false").lower() in ("true", "1", "yes")
    )
    fcmad_no_degroot: bool = field(
        default_factory=lambda: os.environ.get("FCMAD_NO_DEGROOT", "false").lower() in ("true", "1", "yes")
    )
    fcmad_no_summarizer: bool = field(
        default_factory=lambda: os.environ.get("FCMAD_NO_SUMMARIZER", "false").lower() in ("true", "1", "yes")
    )

    # ── Calibration & Evaluation Harness Settings ───────────────────────────
    calibration_params_path: str = field(
        default_factory=lambda: os.environ.get("CALIBRATION_PARAMS_PATH", "eval/calibration_params.json")
    )
    eval_backbone_model: str = field(
        default_factory=lambda: os.environ.get("EVAL_BACKBONE_MODEL", "claude-haiku-4-5-20251001")
    )
    eval_seeds: list = field(
        default_factory=lambda: [42, 43, 44]
    )
    fever_dataset_path: str = field(
        default_factory=lambda: os.environ.get("FEVER_DATASET_PATH", "data/fever_dev.jsonl")
    )
    xfact_dataset_path: str = field(
        default_factory=lambda: os.environ.get("XFACT_DATASET_PATH", "data/xfact.jsonl")
    )

    # Pricing per 1M tokens for telemetry and cost tracking ($ USD)
    prompt_cost_per_million: float = field(
        default_factory=lambda: float(os.environ.get("PROMPT_COST_PER_MILLION", "1.0"))
    )
    completion_cost_per_million: float = field(
        default_factory=lambda: float(os.environ.get("COMPLETION_COST_PER_MILLION", "3.0"))
    )

    # ── LLM Response Cache Settings ─────────────────────────────────────────
    llm_cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    llm_cache_path: str = field(
        default_factory=lambda: os.environ.get("LLM_CACHE_PATH", ".cache/llm_cache.sqlite")
    )

    @property
    def calibration_on(self) -> bool:
        """Alias for calibration_enabled."""
        return self.calibration_enabled

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
