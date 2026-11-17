from typing import TypedDict, Annotated, List, Dict, Optional, Any
from enum import Enum
import operator

from models.claims import AtomicClaim
from models.evidence import ResearchTask, EvidenceItem, SourceScore
from models.agents import BeliefVector, AgentPosition, SocraticChallenge, JudgeEvaluation
from models.consensus import ConsensusResult
from models.verdict import VerdictType, FinalVerdict
from config import config


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Helper reducer to merge two dictionaries safely."""
    merged = dict(a)
    merged.update(b)
    return merged


def merge_errors(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Reducer to concatenate error messages without losing diagnostics."""
    errors = list(filter(None, [a, b]))
    return "; ".join(errors) if errors else None


class VeritasState(TypedDict):
    """
    Central shared state object flowing through the VERITAS LangGraph.
    Tracks all perceptual, reasoning, debate, consensus, and verification state.
    """
    # ── User Input & Mode ───────────────────────────────────────────────────
    input_text: str
    source_label: Optional[str]
    pipeline_mode: str  # "veritas" | "baseline" | "fcmad"

    # ── Claim Analysis ──────────────────────────────────────────────────────
    atomic_claims: List[AtomicClaim]
    pending_claim_ids: List[str]
    current_claim: Optional[AtomicClaim]

    # ── Autonomous Planning & Evidence Management ───────────────────────────
    research_plan: List[ResearchTask]
    evidence_pool: Annotated[List[EvidenceItem], operator.add]
    source_scores: Dict[str, SourceScore]

    # ── Multi-Agent Reasoning & Positions ───────────────────────────────────
    # Annotated with operator.add so parallel reasoning agents can write without collisions
    agent_positions: Annotated[List[AgentPosition], operator.add]
    critic_challenges: Annotated[List[SocraticChallenge], operator.add]
    critic_research_requests: List[str]

    # ── Trust, Beliefs, & Consensus ─────────────────────────────────────────
    belief_vectors: Annotated[Dict[str, BeliefVector], merge_dicts]
    trust_weights: Dict[str, float]
    trust_matrix: List[List[float]]
    judge_evaluations: Annotated[List[JudgeEvaluation], operator.add]
    consensus_results: List[ConsensusResult]

    # ── Control Flow & Iteration Tracking ───────────────────────────────────
    current_round: int
    max_rounds: int
    research_rounds: int
    max_research_rounds: int
    consensus_converged: bool
    escalate_to_human: bool
    verification_complete: bool

    # ── FC-MAD State (docs/FCMAD_MATH.md) ───────────────────────────────────
    per_round_beliefs: List[List[List[float]]]
    credibilities: List[float]
    disagreement_history: List[float]
    rounds_used: int
    final_distribution: Optional[List[float]]
    raw_confidence: Optional[float]
    calibrated_confidence: Optional[float]
    abstain: bool
    recorded_failures: List[Dict[str, Any]]
    fcmad_debate_summary: Optional[str]
    fcmad_quality_scores: Dict[str, float]
    provided_evidence: Optional[List[EvidenceItem]]
    debater_models: Optional[List[str]]
    ablation: Optional[str]
    fcmad_equal_weights: bool
    fcmad_no_degroot: bool
    fcmad_no_summarizer: bool

    # ── Final Verdict & Audit Trail ─────────────────────────────────────────
    final_verdicts: List[FinalVerdict]
    aggregate_verdict: Optional[VerdictType]
    aggregate_confidence: Optional[float]
    audit_summary: Optional[str]
    audit_trail: Annotated[List[str], operator.add]
    error: Annotated[Optional[str], merge_errors]


def initial_veritas_state(
    input_text: str,
    source_label: Optional[str] = None,
    pipeline_mode: str = "fcmad",
    max_rounds: int = 4,
    max_research_rounds: int = 2,
    provided_evidence: Optional[List[Any]] = None,
    ablation: Optional[str] = None,
    debater_models: Optional[List[str]] = None,
) -> VeritasState:
    """
    Constructs a pristine starting state for a VERITAS verification session.
    Supports "veritas", "baseline", and "fcmad" pipeline modes, with optional FC-MAD ablations.
    """
    evidence_items: List[EvidenceItem] = []
    if provided_evidence:
        for idx, item in enumerate(provided_evidence):
            if isinstance(item, EvidenceItem):
                evidence_items.append(item)
            elif isinstance(item, dict):
                ev = EvidenceItem(
                    evidence_id=item.get("evidence_id", f"E{idx+1}"),
                    claim_id=item.get("claim_id", "C1"),
                    query=item.get("query", "provided_evidence"),
                    url=item.get("url", "http://provided.source"),
                    title=item.get("title", "Provided Evidence"),
                    snippet=item.get("snippet", item.get("text", "")),
                    source_domain=item.get("source_domain", "provided.source"),
                    supports_claim=item.get("supports_claim"),
                    relevance_score=float(item.get("relevance_score", 1.0)),
                )
                evidence_items.append(ev)

    # Resolve ablation flags
    abl = (ablation or "").lower()
    equal_w = getattr(config, "fcmad_equal_weights", False) or abl in ("no_judge", "equal_weights", "no_credibility")
    no_degroot = getattr(config, "fcmad_no_degroot", False) or abl in ("no_degroot", "alpha_1")
    no_summ = getattr(config, "fcmad_no_summarizer", False) or abl in ("no_summarizer", "no_summary")

    return VeritasState(
        input_text=input_text,
        source_label=source_label,
        pipeline_mode=pipeline_mode,
        atomic_claims=[],
        pending_claim_ids=[],
        current_claim=None,
        research_plan=[],
        evidence_pool=evidence_items,
        source_scores={},
        agent_positions=[],
        critic_challenges=[],
        critic_research_requests=[],
        belief_vectors={},
        trust_weights={},
        trust_matrix=[],
        judge_evaluations=[],
        consensus_results=[],
        current_round=0,
        max_rounds=max_rounds,
        research_rounds=0,
        max_research_rounds=max_research_rounds,
        consensus_converged=False,
        escalate_to_human=False,
        verification_complete=False,
        per_round_beliefs=[],
        credibilities=[],
        disagreement_history=[],
        rounds_used=0,
        final_distribution=None,
        raw_confidence=None,
        calibrated_confidence=None,
        abstain=False,
        recorded_failures=[],
        fcmad_debate_summary=None,
        fcmad_quality_scores={},
        provided_evidence=evidence_items if evidence_items else None,
        debater_models=debater_models or list(getattr(config, "debater_models", [])),
        ablation=ablation,
        fcmad_equal_weights=equal_w,
        fcmad_no_degroot=no_degroot,
        fcmad_no_summarizer=no_summ,
        final_verdicts=[],
        aggregate_verdict=None,
        aggregate_confidence=None,
        audit_summary=None,
        audit_trail=[],
        error=None,
    )
