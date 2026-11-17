from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models.verdict import VerdictType


class AuditRequest(BaseModel):
    text: str = Field(..., description="The factual claim or text passage to audit")
    source_label: Optional[str] = Field(None, description="e.g. 'GPT-4o response', 'News article'")
    mode: str = Field("fcmad", description="'fcmad' (FC-MAD DeGroot consensus) or 'baseline'")
    max_rounds: Optional[int] = Field(None, ge=1, le=10, description="Max debate rounds (defaults to mode config)")
    max_research_rounds: int = Field(2, ge=1, le=5)
    evidence: Optional[List[Dict[str, Any]]] = Field(None, description="Optional provided evidence (skips search)")


class ClaimVerdictResponse(BaseModel):
    claim_id: str
    claim_text: str
    verdict: VerdictType
    confidence: float
    reasoning: str
    key_evidence: List[str] = []
    contradicting_evidence: List[str] = []
    why_not_other_verdicts: Optional[str] = None
    debate_rounds: int
    consensus_iterations: int
    final_distribution: Optional[List[float]] = None
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    abstain: bool = False


class AuditResponse(BaseModel):
    session_id: str
    mode: str
    aggregate_verdict: VerdictType
    aggregate_confidence: float
    summary: str
    claims: List[ClaimVerdictResponse]
    audit_trail: List[str]
    execution_time_seconds: float
    total_searches: int
    total_llm_calls: int
    final_distribution: Optional[List[float]] = None
    raw_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    abstain: bool = False


class SessionSummary(BaseModel):
    session_id: str
    input_text: str
    aggregate_verdict: str
    aggregate_confidence: float
    created_at: Optional[str]
