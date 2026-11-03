from typing import TypedDict, Annotated, List, Dict, Optional, Any
from enum import Enum
import operator

from models.claims import AtomicClaim
from models.evidence import ResearchTask, EvidenceItem, SourceScore
from models.agents import BeliefVector, AgentPosition, SocraticChallenge, JudgeEvaluation
from models.consensus import ConsensusResult
from models.verdict import VerdictType, FinalVerdict


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
    pipeline_mode: str  # "veritas" | "baseline"

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
    pipeline_mode: str = "veritas",
    max_rounds: int = 3,
    max_research_rounds: int = 2,
) -> VeritasState:
    """
    Constructs a pristine starting state for a VERITAS verification session.
    """
    return VeritasState(
        input_text=input_text,
        source_label=source_label,
        pipeline_mode=pipeline_mode,
        atomic_claims=[],
        pending_claim_ids=[],
        current_claim=None,
        research_plan=[],
        evidence_pool=[],
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
        final_verdicts=[],
        aggregate_verdict=None,
        aggregate_confidence=None,
        audit_summary=None,
        audit_trail=[],
        error=None,
    )
