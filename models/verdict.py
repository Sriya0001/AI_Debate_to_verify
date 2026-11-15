from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    Extended with FC-MAD distribution, calibration, and failure tracking.
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

    # FC-MAD additions (docs/FCMAD_MATH.md)
    final_distribution: Optional[List[float]] = Field(default=None, description="Consensus belief distribution p* = [p_S, p_R, p_N]")
    raw_confidence: Optional[float] = Field(default=None, description="Max probability in p*")
    calibrated_confidence: Optional[float] = Field(default=None, description="Confidence after calibration/temperature scaling")
    abstain: bool = Field(default=False, description="Whether verdict is abstained (uncertain)")
    recorded_failures: List[Dict[str, Any]] = Field(default_factory=list, description="Failures encountered during verification")
    credibilities: List[float] = Field(default_factory=list, description="Final debater credibilities")
    disagreement_history: List[float] = Field(default_factory=list, description="History of max disagreement E per round")
    rounds_used: Optional[int] = Field(default=None, description="Total debate rounds executed")


class AuditSessionResult(BaseModel):
    """
    Complete audit result for an input text session.
    """
    session_id: str
    input_text: str
    mode: str = "veritas"  # "veritas", "baseline", or "fcmad"
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

    # FC-MAD session additions
    final_distribution: Optional[List[float]] = None
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    abstain: bool = False
    recorded_failures: List[Dict[str, Any]] = Field(default_factory=list)
