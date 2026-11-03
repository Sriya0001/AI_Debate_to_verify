from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime
from models.claims import AtomicClaim
from models.evidence import EvidenceItem, SourceScore
from models.agents import AgentPosition, JudgeEvaluation
from models.consensus import ConsensusResult


class VerdictType(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFORMATION = "NOT_ENOUGH_INFORMATION"


class FinalVerdict(BaseModel):
    """
    Authoritative verdict produced by the Final Verification Agent.
    """
    claim_id: str
    claim_text: str
    verdict: VerdictType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    key_evidence: List[str] = Field(default_factory=list, description="IDs of decisive supporting evidence, e.g. ['E1', 'E5']")
    contradicting_evidence: List[str] = Field(default_factory=list, description="IDs of counter-evidence, e.g. ['E9']")
    why_not_other_verdicts: str
    verification_complete: bool = True
    debate_rounds: int = 1
    consensus_iterations: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AuditSessionResult(BaseModel):
    """
    Complete audit result for an input text session.
    """
    session_id: str
    input_text: str
    mode: str = "veritas"  # "veritas" or "baseline"
    claims: List[AtomicClaim] = Field(default_factory=list)
    verdicts: List[FinalVerdict] = Field(default_factory=list)
    aggregate_verdict: VerdictType
    aggregate_confidence: float
    summary: str
    audit_trail: List[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    total_searches: int = 0
    total_llm_calls: int = 0
    estimated_tokens: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
